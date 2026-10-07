"""Train only a sparse HR abstention gate around a frozen candidate scorer."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.corga.sparse_hr_model import SparseHRCorGA
from core.corga.sparse_hr_refiner import (
    candidate_gate_loss, candidate_selection_loss, sample_training_queries, sparse_sequence_loss,
)
from core.stereo_datasets import SceneFlowDatasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"))
    parser.add_argument("--initial-refiner", default=str(
        ROOT / "checkpoints/corga_sparse_hr_dir3_boundary2_from5k_ft2k/refiner_step2000.pth"))
    parser.add_argument("--candidate-probe", default=str(
        ROOT / "diagnostics/sparse_hr_diagnosis/candidate_probe_train200_test100.pth"))
    parser.add_argument("--sceneflow-root", default="/home/share/yijiayi/sceneflow")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-query", type=int, default=2048)
    parser.add_argument("--coarse-iters", type=int, default=6)
    parser.add_argument("--refine-iters", type=int, default=32)
    parser.add_argument("--hr-iters", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-loss-weight", type=float, default=.5)
    parser.add_argument("--gate-target", choices=["binary", "ratio"], default="binary")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--context-channels", type=int, default=64)
    parser.add_argument("--train-scope", choices=["gate", "structure", "scored"], default="gate")
    parser.add_argument("--context-scoring", action="store_true")
    parser.add_argument("--candidate-loss-weight", type=float, default=0.)
    parser.add_argument("--fresh-structure", action="store_true")
    parser.add_argument("--train-feature", action="store_true")
    parser.add_argument("--feature-lr", type=float, default=1e-5)
    return parser.parse_args()


def save(path, model, optimizer, step, args):
    state = {"refiner": model.refiner.state_dict(),
             "optimizer": optimizer.state_dict(), "step": step,
             "config": {**vars(args), "direction_classes": True,
                        "candidate_guided": True, "gate_init_bias": 0.,
                        "feature_channels": 16,
                        "delta_radius": 2.}}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select an available CUDA GPU explicitly")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = SparseHRCorGA(args.checkpoint, coarse_iters=args.coarse_iters,
                          refine_iters=args.refine_iters, hr_iters=args.hr_iters,
                          hidden_channels=args.hidden_channels,
                          context_channels=args.context_channels,
                          gate_init_bias=0., direction_classes=True,
                          candidate_guided=True, context_scoring=args.context_scoring).cuda()
    initial = torch.load(args.initial_refiner, map_location="cpu", weights_only=False)
    structure_prefixes = ("hidden_transfer.", "coarse_context_encoder.",
                          "fine_context_encoder.", "context_fusion.", "gru.",
                          "candidate_gate.")
    initial_state = initial["refiner"]
    if args.fresh_structure:
        if args.train_scope != "structure":
            raise ValueError("Fresh structure requires structure training")
        reset_prefixes = structure_prefixes + ("direction_head.", "delta_head.", "gate_head.")
        initial_state = {k: v for k, v in initial_state.items()
                         if not k.startswith(reset_prefixes)}
    incompatible = model.refiner.load_state_dict(initial_state, strict=False)
    expected = {key for key in model.refiner.state_dict()
                if key.startswith(("candidate_scorer.", "candidate_gate.",
                                   "candidate_mean", "candidate_std"))}
    if args.fresh_structure:
        expected = {k for k in model.refiner.state_dict() if k.startswith(reset_prefixes)}
        acceptable = (expected,)
    else:
        acceptable = (expected, set())
    new_context_keys = {k for k in model.refiner.state_dict()
                        if k.startswith("context_candidate_scorer.")}
    acceptable = tuple(keys | new_context_keys for keys in acceptable) + acceptable
    if set(incompatible.missing_keys) not in acceptable or incompatible.unexpected_keys:
        raise ValueError(f"Incompatible initial refiner: {incompatible}")
    if any(k.startswith(("candidate_scorer.", "candidate_mean", "candidate_std"))
           for k in incompatible.missing_keys):
        model.refiner.load_candidate_probe(args.candidate_probe)
    model.refiner.requires_grad_(False)
    train_prefixes = (("candidate_gate.",) if args.train_scope == "gate" else structure_prefixes)
    if args.train_scope == "scored":
        train_prefixes += ("candidate_scorer.", "context_candidate_scorer.")
    if args.train_feature:
        train_prefixes += ("feature.",)
    for name, parameter in model.refiner.named_parameters():
        if name.startswith(train_prefixes):
            parameter.requires_grad_(True)
    model.train()
    parameters = [p for p in model.refiner.parameters() if p.requires_grad]
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in parameters),
                      "config": vars(args)}), flush=True)
    groups = [{"params": [p for n, p in model.refiner.named_parameters()
                           if p.requires_grad and not n.startswith("feature.")],
               "lr": args.lr}]
    if args.train_feature:
        groups.append({"params": list(model.refiner.feature.parameters()),
                       "lr": args.feature_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.refiner.load_state_dict(state["refiner"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
    # Model dimensions consume different RNG draws; reset for matched sampling.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = SceneFlowDatasets(
        aug_params={"crop_size": [384, 768], "do_flip": False},
        root=args.sceneflow_root, dstype="frames_finalpass")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    iterator = iter(loader)
    log = output / "train.jsonl"
    skipped = 0
    for step in range(start + 1, args.steps + 1):
        for attempt in range(100):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            left = batch["img1"].cuda(non_blocking=True)
            right = batch["img2"].cuda(non_blocking=True)
            gt = batch["disp"].cuda(non_blocking=True)
            valid = batch["valid"].cuda(non_blocking=True).bool()
            valid &= torch.isfinite(gt) & (gt > 0) & (gt < 768)
            try:
                query = sample_training_queries(gt, valid, n=args.n_query,
                                                band_radius=2, query_mode="mixed")
                break
            except ValueError as exc:
                if str(exc) != "An image has no valid disparity queries":
                    raise
                skipped += 1
        else:
            raise RuntimeError("Could not find a valid training batch in 100 attempts")
        optimizer.zero_grad(set_to_none=True)
        _, (predictions, info) = model.forward_queries(
            left, right, query, return_candidate_info=True)
        disparity_loss = sparse_sequence_loss(predictions, gt, valid, query)
        gate_loss = candidate_gate_loss(info, predictions, gt, valid, query,
                                        target_mode=args.gate_target)
        loss = disparity_loss + args.gate_loss_weight * gate_loss
        selection_loss = disparity_loss.new_zeros(())
        if args.candidate_loss_weight:
            selection_loss = candidate_selection_loss(info, predictions, gt, valid,
                                                     query, model.refiner.offsets)
            loss = loss + args.candidate_loss_weight * selection_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach()),
                  "disparity_loss": float(disparity_loss.detach()),
                  "gate_loss": float(gate_loss.detach()),
                  "candidate_loss": float(selection_loss.detach()),
                  "grad_norm": float(grad_norm.detach()),
                  "skipped_batches": skipped}
        with log.open("a") as file:
            file.write(json.dumps(record) + "\n")
        if step == 1 or step % 50 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(output / f"refiner_step{step}.pth", model, optimizer, step, args)
            save(output / "refiner_latest.pth", model, optimizer, step, args)


if __name__ == "__main__":
    main()
