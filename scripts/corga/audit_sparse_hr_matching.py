"""Audit sparse HR candidate evidence on fixed SceneFlow TEST images.

Use GT only to form evaluation masks and targets. The model's candidate scores
and direction logits are computed without GT.
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
    ShallowStereoFeature, dilate, disparity_boundary, sample_map,
)
from core.stereo_datasets import SceneFlowDatasets
from core.utils import frame_utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"))
    parser.add_argument("--refiner", default=str(
        ROOT / "checkpoints/corga_sparse_hr_dir3_boundary2_from5k_ft2k/refiner_step2000.pth"))
    parser.add_argument("--old-refiner", default=str(
        ROOT / "checkpoints/corga_sparse_hr_boundary2_gate0_hard4_5k/refiner_step5000.pth"))
    parser.add_argument("--sceneflow-root", default="/home/share/yijiayi/sceneflow")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-queries-per-image", type=int, default=2048)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def candidate_scores(left_feature, right_feature, xy, disparity, image_size,
                     offsets):
    """Score each physical-disparity offset using the existing shallow features."""
    left = sample_map(left_feature, xy, image_size).unsqueeze(2)
    right_xy = xy.unsqueeze(2).expand(-1, -1, len(offsets), -1).clone()
    right_xy[..., 0] -= disparity + offsets.view(1, 1, -1)
    right = sample_map(right_feature, right_xy, image_size)
    return {
        "cosine": F.cosine_similarity(left, right, dim=-1, eps=1e-6),
        "dot_product": (left * right).mean(dim=-1),
        "negative_l1": -(left - right).abs().mean(dim=-1),
    }


def empty_record():
    return {"pixels": 0, "direction_hits": 0, "closer_to_gt": 0,
            "within_one_px_of_gt": 0, "gt_score_above_center": 0,
            "gt_minus_center_score_sum": 0., "best_nonzero": 0}


def add_score_metrics(record, score, target, offsets, mask):
    n = int(mask.sum())
    if not n:
        return
    score = score[mask]
    target = target[mask]
    best = offsets[score.argmax(dim=-1)]
    target_idx = (target.round().clamp(offsets[0], offsets[-1])
                  - offsets[0]).long()
    target_score = score.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    center_score = score[:, len(offsets) // 2]
    record["pixels"] += n
    record["direction_hits"] += int((best.sign() == target.sign()).sum())
    record["best_nonzero"] += int((best != 0).sum())
    record["closer_to_gt"] += int(((target - best).abs() < target.abs()).sum())
    record["within_one_px_of_gt"] += int(((target - best).abs() <= 1.).sum())
    record["gt_score_above_center"] += int((target_score > center_score).sum())
    record["gt_minus_center_score_sum"] += float((target_score - center_score).sum())


def finalize(record):
    n = record["pixels"]
    return {"pixels": n,
            "direction_hit_percent": 100 * record["direction_hits"] / n if n else None,
            "direction_hit_given_nonzero_percent": 100 * record["direction_hits"] / record["best_nonzero"]
            if record["best_nonzero"] else None,
            "nonzero_percent": 100 * record["best_nonzero"] / n if n else None,
            "closer_to_gt_percent": 100 * record["closer_to_gt"] / n if n else None,
            "within_one_px_of_gt_percent": 100 * record["within_one_px_of_gt"] / n if n else None,
            "gt_score_above_center_percent": 100 * record["gt_score_above_center"] / n if n else None,
            "mean_gt_minus_center_score": record["gt_minus_center_score_sum"] / n if n else None}


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select an available CUDA GPU explicitly")
    if args.max_queries_per_image <= 0:
        raise ValueError("max-queries-per-image must be positive")
    new_state = torch.load(args.refiner, map_location="cpu", weights_only=False)
    old_state = torch.load(args.old_refiner, map_location="cpu", weights_only=False)
    config = new_state["config"]
    model = SparseHRCorGA(
        args.checkpoint, coarse_iters=6, refine_iters=32, hr_iters=4,
        feature_channels=config.get("feature_channels", 16),
        hidden_channels=config.get("hidden_channels", 64),
        delta_radius=config.get("delta_radius", 2.),
        gate_init_bias=config.get("gate_init_bias", 0.),
        direction_classes=True).cuda().eval()
    model.refiner.load_state_dict(new_state["refiner"], strict=True)
    old_feature = ShallowStereoFeature(config.get("feature_channels", 16)).cuda().eval()
    old_feature.load_state_dict({key.removeprefix("feature."): value
                                 for key, value in old_state["refiner"].items()
                                 if key.startswith("feature.")}, strict=True)
    dataset = SceneFlowDatasets(root=args.sceneflow_root,
                                dstype="frames_finalpass", things_test=True)
    indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)),
                          dtype=np.int64).tolist()
    groups = ("all", "lr_consistent", "lr_inconsistent", "right_out_of_bounds")
    methods = ("old_cosine", "old_dot_product", "old_negative_l1",
               "new_cosine", "new_dot_product", "new_negative_l1")
    totals = {group: {method: empty_record() for method in methods}
              for group in groups}
    head_totals = {group: {"pixels": 0, "direction_hits": 0, "nonzero": 0,
                           "target_positive": 0, "target_negative": 0}
                   for group in groups}
    pool_count = 0
    sampled_count = 0
    offsets = model.refiner.offsets
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
            raise ValueError("Left and right GT disparity shapes differ")
        right_gt = torch.from_numpy(right_gt_np.copy()).cuda()[None, None]
        padder, left_pad, right_pad, full_pad, state = model._coarse_state(left, right)
        coarse = padder.unpad(full_pad)
        edge2 = dilate(disparity_boundary(gt, valid, 2.), 2) & valid
        error = (gt - coarse).abs()
        pool = edge2 & (error >= 1.) & (error <= 4.)
        positions = torch.nonzero(pool[0, 0], as_tuple=False)
        pool_count += len(positions)
        if not len(positions):
            continue
        generator = torch.Generator().manual_seed(20261004 + index)
        chosen = torch.randperm(len(positions), generator=generator)[:args.max_queries_per_image]
        yx = positions[chosen].cuda()
        xy = yx[:, [1, 0]].float()[None]
        sampled_count += xy.shape[1]
        gt_q = gt[0, 0, yx[:, 0], yx[:, 1]][None, :, None]
        base_q = coarse[0, 0, yx[:, 0], yx[:, 1]][None, :, None]
        target = (gt_q - base_q)[0, :, 0]
        right_xy = xy.clone()
        right_xy[..., 0] -= gt_q[..., 0]
        right_sample = sample_map(right_gt, right_xy, gt.shape[-2:])[:, :, 0]
        inside = (right_xy[..., 0] >= 0) & (right_xy[..., 0] <= gt.shape[-1] - 1)
        right_valid = torch.isfinite(right_sample) & (right_sample > 0) & (right_sample < 768)
        consistent = (inside & right_valid & ((right_sample - gt_q[..., 0]).abs() <= 1.))[0]
        inconsistent = (inside & ~consistent)[0]
        outside = (~inside)[0]
        group_masks = {"all": torch.ones_like(consistent),
                       "lr_consistent": consistent,
                       "lr_inconsistent": inconsistent,
                       "right_out_of_bounds": outside}
        pad_left, _, pad_top, _ = padder._pad
        xy_pad = xy + xy.new_tensor((pad_left, pad_top))
        image_size = left_pad.shape[-2:]
        old_left = old_feature(left_pad)
        old_right = old_feature(right_pad)
        new_left = model.refiner.feature(left_pad)
        new_right = model.refiner.feature(right_pad)
        old_scores = candidate_scores(old_left, old_right, xy_pad, base_q,
                                      image_size, offsets)
        new_scores = candidate_scores(new_left, new_right, xy_pad, base_q,
                                      image_size, offsets)
        _, head_logits = model.refiner(
            left_pad, right_pad, full_pad, state["disp_lr_px"],
            state["hidden_lr"], state["context_lr"], xy_pad,
            iterations=1, feature_left=new_left, feature_right=new_right,
            return_direction_logits=True)
        head_class = head_logits[0][0].argmax(dim=-1)
        head_offset = head_class.float() - 1.
        for group, mask in group_masks.items():
            for name, score in old_scores.items():
                add_score_metrics(totals[group][f"old_{name}"], score[0],
                                  target, offsets, mask)
            for name, score in new_scores.items():
                add_score_metrics(totals[group][f"new_{name}"], score[0],
                                  target, offsets, mask)
            # The direction head predicts a sign, not an offset magnitude.
            head_record = head_totals[group]
            n = int(mask.sum())
            head_record["pixels"] += n
            head_record["target_positive"] += int((target[mask] > 0).sum())
            head_record["target_negative"] += int((target[mask] < 0).sum())
            head_record["direction_hits"] += int(
                (head_offset[mask] == target[mask].sign()).sum())
            head_record["nonzero"] += int((head_offset[mask] != 0).sum())
        if count % 20 == 0 or count == len(indices):
            print(f"{count}/{len(indices)}", flush=True)
    result = {"dataset": "SceneFlow FlyingThings3D finalpass TEST",
              "images": len(indices), "refiner": args.refiner,
              "old_refiner": args.old_refiner,
              "query_pool": "GT Edge +/-2 and initial absolute error in [1,4] px",
              "pool_pixels": pool_count, "sampled_pixels": sampled_count,
              "max_queries_per_image": args.max_queries_per_image,
              "visibility_proxy": "GT left-right disparity consistency <=1 px at warped right coordinate; inconsistent includes occlusion and sampling error",
              "score_metrics": {group: {method: finalize(record)
                                        for method, record in values.items()}
                                for group, values in totals.items()},
              "direction_head_first_step": {
                  group: {"pixels": values["pixels"],
                          "target_positive_percent": 100 * values["target_positive"] / values["pixels"]
                          if values["pixels"] else None,
                          "majority_sign_baseline_percent": 100 * max(
                              values["target_positive"], values["target_negative"]) / values["pixels"]
                          if values["pixels"] else None,
                          "direction_hit_percent": 100 * values["direction_hits"] / values["pixels"]
                          if values["pixels"] else None,
                          "direction_hit_given_nonzero_percent": 100 * values["direction_hits"] / values["nonzero"]
                          if values["nonzero"] else None,
                          "nonzero_percent": 100 * values["nonzero"] / values["pixels"]
                          if values["pixels"] else None}
                  for group, values in head_totals.items()}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
