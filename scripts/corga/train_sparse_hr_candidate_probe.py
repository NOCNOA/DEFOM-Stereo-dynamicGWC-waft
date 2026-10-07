"""Train a small nine-candidate score probe on frozen sparse HR features.

Training uses LR-consistent SceneFlow TRAIN boundary points with coarse error
1-4 px. Evaluation uses the fixed 100-image SceneFlow TEST subset and reports
consistent, inconsistent, and out-of-bounds points separately. GT is never an
input to the candidate scorer; it only selects training points and labels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.corga.sparse_hr_model import SparseHRCorGA
from core.corga.sparse_hr_refiner import (
    CandidateScorer, candidate_tokens, dilate, disparity_boundary, sample_map,
)
from core.stereo_datasets import SceneFlowDatasets
from core.utils import frame_utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"))
    parser.add_argument("--refiner", default=str(
        ROOT / "checkpoints/corga_sparse_hr_dir3_boundary2_from5k_ft2k/refiner_step2000.pth"))
    parser.add_argument("--sceneflow-root", default="/home/share/yijiayi/sceneflow")
    parser.add_argument("--train-images", type=int, default=200)
    parser.add_argument("--test-images", type=int, default=100)
    parser.add_argument("--train-queries-per-image", type=int, default=1024)
    parser.add_argument("--test-queries-per-image", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def make_tokens(left_feature, right_feature, xy, disparity, image_size, offsets):
    left = sample_map(left_feature, xy, image_size).unsqueeze(2)
    right_xy = xy.unsqueeze(2).expand(-1, -1, len(offsets), -1).clone()
    right_xy[..., 0] -= disparity + offsets.view(1, 1, -1)
    right = sample_map(right_feature, right_xy, image_size)
    inside = ((right_xy[..., 0] >= 0) & (right_xy[..., 0] <= image_size[1] - 1)
              & (right_xy[..., 1] >= 0) & (right_xy[..., 1] <= image_size[0] - 1))
    return candidate_tokens(left, right, offsets, inside)


@torch.no_grad()
def collect(model, dataset, indices, max_queries, seed, consistent_only):
    tokens, labels, residuals, groups = [], [], [], []
    offsets = model.refiner.offsets
    total_pool = 0
    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        left = sample["img1"].unsqueeze(0).cuda()
        right = sample["img2"].unsqueeze(0).cuda()
        gt = sample["disp"].unsqueeze(0).cuda()
        valid = sample["valid"].unsqueeze(0).cuda().bool()
        valid &= torch.isfinite(gt) & (gt > 0) & (gt < 768)
        right_gt_np = np.asarray(frame_utils.read_gen(dataset.disparity_list[index][1]),
                                 dtype=np.float32)
        if right_gt_np.shape != tuple(gt.shape[-2:]):
            raise ValueError("Left and right GT disparity sizes differ")
        right_gt = torch.from_numpy(right_gt_np.copy()).cuda()[None, None]
        padder, left_pad, right_pad, full_pad, _ = model._coarse_state(left, right)
        coarse = padder.unpad(full_pad)
        boundary = dilate(disparity_boundary(gt, valid, 2.), 2) & valid
        error = (gt - coarse).abs()
        pool = boundary & (error >= 1.) & (error <= 4.)
        positions = torch.nonzero(pool[0, 0], as_tuple=False)
        total_pool += len(positions)
        if not len(positions):
            continue
        yx = positions.cuda()
        xy = yx[:, [1, 0]].float()[None]
        gt_q = gt[0, 0, yx[:, 0], yx[:, 1]][None, :, None]
        base_q = coarse[0, 0, yx[:, 0], yx[:, 1]][None, :, None]
        right_xy = xy.clone()
        right_xy[..., 0] -= gt_q[..., 0]
        right_sample = sample_map(right_gt, right_xy, gt.shape[-2:])[:, :, 0]
        inside = (right_xy[..., 0] >= 0) & (right_xy[..., 0] <= gt.shape[-1] - 1)
        right_valid = torch.isfinite(right_sample) & (right_sample > 0) & (right_sample < 768)
        consistent = (inside & right_valid & ((right_sample - gt_q[..., 0]).abs() <= 1.))[0]
        group = torch.full((len(positions),), 1, dtype=torch.uint8, device=gt.device)
        group[consistent] = 0
        group[~inside[0]] = 2
        available = torch.nonzero(consistent, as_tuple=False).flatten() if consistent_only else torch.arange(len(positions), device=gt.device)
        if not len(available):
            continue
        generator = torch.Generator().manual_seed(seed + int(index))
        chosen = torch.randperm(len(available), generator=generator)[:max_queries].cuda()
        selected = available[chosen]
        xy = xy[:, selected]
        gt_q = gt_q[:, selected]
        base_q = base_q[:, selected]
        group = group[selected]
        pad_left, _, pad_top, _ = padder._pad
        xy_pad = xy + xy.new_tensor((pad_left, pad_top))
        left_feature = model.refiner.feature(left_pad)
        right_feature = model.refiner.feature(right_pad)
        token = make_tokens(left_feature, right_feature, xy_pad, base_q,
                            left_pad.shape[-2:], offsets)[0]
        residual = (gt_q - base_q)[0, :, 0]
        label = (residual.round().clamp(-4, 4) + 4).long()
        tokens.append(token.cpu().half())
        labels.append(label.cpu())
        residuals.append(residual.cpu())
        groups.append(group.cpu())
        if count % 50 == 0 or count == len(indices):
            print(f"collected {count}/{len(indices)} images", flush=True)
    if not tokens:
        raise ValueError("No usable candidate queries were collected")
    return {"tokens": torch.cat(tokens), "labels": torch.cat(labels),
            "residuals": torch.cat(residuals), "groups": torch.cat(groups),
            "pool_pixels": total_pool}


@torch.no_grad()
def evaluate(probe, data, mean, std):
    tokens = data["tokens"].cuda().float()
    scores = []
    for chunk in tokens.split(16384):
        scores.append(probe((chunk - mean) / std))
    learned = torch.cat(scores).argmax(dim=-1).cpu() - 4
    channels = (tokens.shape[-1] - 2) // 3
    raw = -tokens[..., channels:2 * channels].mean(dim=-1)
    raw = raw.argmax(dim=-1).cpu() - 4
    target = data["residuals"]
    group = data["groups"]
    output = {}
    for name, group_id in (("all", None), ("lr_consistent", 0),
                           ("lr_inconsistent", 1), ("right_out_of_bounds", 2)):
        mask = torch.ones_like(group, dtype=torch.bool) if group_id is None else group == group_id
        n = int(mask.sum())
        if not n:
            continue
        t = target[mask]
        majority = max(float((t > 0).float().mean()), float((t < 0).float().mean()))
        result = {"pixels": n, "majority_sign_baseline_percent": 100 * majority,
                  "initial_epe": float(t.abs().mean())}
        for method, pred in (("raw_negative_l1", raw[mask]), ("trained_probe", learned[mask])):
            result[method] = {
                "direction_hit_percent": float(100 * (pred.sign() == t.sign()).float().mean()),
                "closer_to_gt_percent": float(100 * ((t - pred).abs() < t.abs()).float().mean()),
                "epe_after_integer_offset": float((t - pred).abs().mean()),
                "within_one_px_of_gt_percent": float(100 * ((t - pred).abs() <= 1.).float().mean()),
            }
        output[name] = result
    return output


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select an available CUDA GPU explicitly")
    torch.manual_seed(42)
    np.random.seed(42)
    checkpoint = torch.load(args.refiner, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SparseHRCorGA(
        args.checkpoint, coarse_iters=6, refine_iters=32, hr_iters=4,
        feature_channels=config.get("feature_channels", 16),
        hidden_channels=config.get("hidden_channels", 64),
        delta_radius=config.get("delta_radius", 2.),
        gate_init_bias=config.get("gate_init_bias", 0.),
        direction_classes=True).cuda().eval()
    model.refiner.load_state_dict(checkpoint["refiner"], strict=True)
    train_set = SceneFlowDatasets(root=args.sceneflow_root,
                                  dstype="frames_finalpass", things_test=False)
    train_set.is_eval = True  # Keep left/right order fixed for right-GT consistency.
    train_count = sum("/FlyingThings3D/frames_finalpass/TRAIN/" in item[0]
                      for item in train_set.image_list)
    train_indices = np.linspace(0, train_count - 1,
                                min(args.train_images, train_count), dtype=np.int64).tolist()
    test_set = SceneFlowDatasets(root=args.sceneflow_root,
                                 dstype="frames_finalpass", things_test=True)
    test_indices = np.linspace(0, len(test_set) - 1,
                               min(args.test_images, len(test_set)), dtype=np.int64).tolist()
    train = collect(model, train_set, train_indices, args.train_queries_per_image,
                    seed=20261004, consistent_only=True)
    test = collect(model, test_set, test_indices, args.test_queries_per_image,
                   seed=20261004, consistent_only=False)
    del model
    torch.cuda.empty_cache()
    x = train["tokens"].cuda().float()
    labels = train["labels"].cuda()
    mean = x.mean(dim=(0, 1), keepdim=True)
    std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-3)
    x = (x - mean) / std
    scorer = CandidateScorer(config.get("feature_channels", 16)).cuda()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(1, args.steps + 1):
        chosen = torch.randint(len(x), (args.batch_size,), device=x.device)
        logits = scorer(x[chosen])
        loss = F.cross_entropy(logits, labels[chosen])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"probe step {step}/{args.steps}: CE={float(loss):.5f}", flush=True)
    scorer.eval()
    metrics = evaluate(scorer, test, mean, std)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"probe": scorer.state_dict(), "feature_mean": mean.cpu(),
                "feature_std": std.cpu(), "config": vars(args)},
               output.with_suffix(".pth"))
    result = {"train_images": len(train_indices), "test_images": len(test_indices),
              "train_queries": len(train["labels"]), "test_queries": len(test["labels"]),
              "train_pool_pixels": train["pool_pixels"],
              "test_pool_pixels": test["pool_pixels"],
              "train_selection": "GT Edge +/-2, coarse error [1,4] px, LR-consistent",
              "test_selection": "GT Edge +/-2, coarse error [1,4] px",
              "frozen_refiner": args.refiner,
              "steps": args.steps, "metrics": metrics,
              "probe_checkpoint": str(output.with_suffix(".pth"))}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
