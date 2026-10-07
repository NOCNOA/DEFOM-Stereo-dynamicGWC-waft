"""Frozen-CorGA stage-1 training for the sparse HR recurrent refiner.

Example: CUDA_VISIBLE_DEVICES=2 python scripts/corga/train_sparse_hr_refiner.py
"""
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
    sample_training_queries, sparse_direction_loss, sparse_sequence_loss,
)
from core.stereo_datasets import SceneFlowDatasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"))
    parser.add_argument("--sceneflow-root", default="/home/share/yijiayi/sceneflow")
    parser.add_argument("--output", default=str(ROOT / "checkpoints/corga_sparse_hr_v1"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-query", type=int, default=4096)
    parser.add_argument("--query-mode", choices=("mixed", "boundary_only"),
                        default="mixed")
    parser.add_argument("--band-radius", type=int, default=4)
    parser.add_argument("--crop-height", type=int, default=384)
    parser.add_argument("--crop-width", type=int, default=768)
    parser.add_argument("--coarse-iters", type=int, default=6)
    parser.add_argument("--refine-iters", type=int, default=32)
    parser.add_argument("--hr-iters", type=int, default=4)
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--delta-radius", type=float, default=2.)
    parser.add_argument("--gate-init-bias", type=float, default=-2.)
    parser.add_argument("--hard-error-weight", type=float, default=0.,
                        help="extra loss weight for initial query error in [1,8] px")
    parser.add_argument("--direction-classes", action="store_true",
                        help="use a supervised decrease/stay/increase update head")
    parser.add_argument("--direction-loss-weight", type=float, default=1.)
    parser.add_argument("--initialize-refiner", default=None,
                        help="load refiner weights without optimizer or step state")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save(path, model, optimizer, step, args):
    data = {"refiner": model.refiner.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "config": vars(args)}
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an explicitly selected CUDA GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = SparseHRCorGA(
        args.checkpoint, coarse_iters=args.coarse_iters,
        refine_iters=args.refine_iters, hr_iters=args.hr_iters,
        feature_channels=args.feature_channels, hidden_channels=args.hidden_channels,
        delta_radius=args.delta_radius,
        gate_init_bias=args.gate_init_bias,
        direction_classes=args.direction_classes).cuda()
    model.train()
    if args.initialize_refiner and args.resume:
        raise ValueError("Use either --initialize-refiner or --resume")
    if args.initialize_refiner:
        initial = torch.load(args.initialize_refiner, map_location="cpu", weights_only=False)
        incompatible = model.refiner.load_state_dict(initial["refiner"], strict=False)
        expected_missing = {"direction_head.weight", "direction_head.bias"} if args.direction_classes else set()
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError(f"Incompatible refiner initialization: {incompatible}")
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=args.lr, weight_decay=1e-5)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.refiner.load_state_dict(state["refiner"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
    dataset = SceneFlowDatasets(
        aug_params={"crop_size": [args.crop_height, args.crop_width], "do_flip": False},
        root=args.sceneflow_root, dstype="frames_finalpass")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if len(loader) == 0:
        raise ValueError("No full training batches available")
    iterator = iter(loader)
    log_path = output / "train.jsonl"
    skipped_batches = 0
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
                                                band_radius=args.band_radius,
                                                query_mode=args.query_mode)
                break
            except ValueError as exc:
                empty_crop = str(exc) in (
                    "An image has no valid GT boundary queries",
                    "An image has no valid disparity queries",
                )
                if args.query_mode != "boundary_only" or not empty_crop:
                    raise
                skipped_batches += 1
        else:
            raise RuntimeError("Could not find a boundary-containing batch in 100 attempts")
        optimizer.zero_grad(set_to_none=True)
        coarse, result = model.forward_queries(
            left, right, query, return_direction_logits=args.direction_classes)
        predictions, direction_logits = result if args.direction_classes else (result, None)
        disparity_loss = sparse_sequence_loss(
            predictions, gt, valid, query, hard_error_weight=args.hard_error_weight)
        direction_loss = sparse_direction_loss(
            direction_logits, predictions, gt, valid, query) if args.direction_classes else None
        loss = disparity_loss + args.direction_loss_weight * direction_loss if args.direction_classes else disparity_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach()),
                  "disparity_loss": float(disparity_loss.detach()),
                  "direction_loss": float(direction_loss.detach()) if direction_loss is not None else None,
                  "grad_norm": float(grad_norm.detach()),
                  "query_count": args.n_query, "batch_size": args.batch_size,
                  "gate_init_bias": args.gate_init_bias,
                  "hard_error_weight": args.hard_error_weight,
                  "query_mode": args.query_mode,
                  "band_radius": args.band_radius,
                  "skipped_batches": skipped_batches}
        with log_path.open("a") as file:
            file.write(json.dumps(record) + "\n")
        if step == 1 or step % 20 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(output / f"refiner_step{step}.pth", model, optimizer, step, args)
            save(output / "refiner_latest.pth", model, optimizer, step, args)
        del coarse, predictions, loss


if __name__ == "__main__":
    main()
