"""Evaluate sparse HR refinement and GT boundary bands on SceneFlow TEST."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.corga.sparse_hr_model import SparseHRCorGA
from core.corga.sparse_hr_refiner import (
    dilate, disparity_boundary, update_quality_totals, summarize_update_quality,
)
from core.stereo_datasets import SceneFlowDatasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"))
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--sceneflow-root", default="/home/share/yijiayi/sceneflow")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--disjoint-samples", action="store_true",
                        help="Use queries from images disjoint from the default fixed image set")
    parser.add_argument("--mode", choices=("oracle_boundary", "oracle_correctable",
                                           "predicted_boundary", "all"),
                        default="oracle_boundary")
    parser.add_argument("--coarse-iters", type=int, default=6)
    parser.add_argument("--refine-iters", type=int, default=32)
    parser.add_argument("--hr-iters", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--band-radius", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=2.)
    parser.add_argument("--oracle-update-direction", action="store_true",
                        help="posthoc GT sign for each cumulative HR update; keep model magnitude")
    return parser.parse_args()


ERROR_BINS = ((0., 1., "0-1"), (1., 3., "1-3"),
              (3., 8., "3-8"), (8., float("inf"), "8+"))
QUALITY_BINS = ((0., 1., "0-1"), (1., 4., "1-4"),
                (4., 8., "4-8"), (8., float("inf"), "8+"))


def empty_metrics(steps):
    names = ["overall", "edge1", "edge2", "edge4", "nonedge"]
    names.extend(f"selected_init_error_{label}" for _, _, label in ERROR_BINS)
    names.extend(f"edge4_init_error_{label}" for _, _, label in ERROR_BINS)
    return {name: [{"pixels": 0, "error_sum": 0., "bad1": 0, "bad3": 0, "bad5": 0}
                   for _ in range(steps + 1)]
            for name in names}


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires an explicitly selected CUDA GPU")
    state = torch.load(args.refiner, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = SparseHRCorGA(
        args.checkpoint, coarse_iters=args.coarse_iters,
        refine_iters=args.refine_iters, hr_iters=args.hr_iters,
        feature_channels=config.get("feature_channels", 16),
        hidden_channels=config.get("hidden_channels", 64),
        context_channels=config.get("context_channels", 64),
        delta_radius=config.get("delta_radius", 2.),
        gate_init_bias=config.get("gate_init_bias", -2.),
        direction_classes=config.get("direction_classes", False),
        candidate_guided=config.get("candidate_guided", False),
        context_scoring=config.get("context_scoring", False)).cuda().eval()
    model.refiner.load_state_dict(state["refiner"], strict=True)
    dataset = SceneFlowDatasets(root=args.sceneflow_root,
                                dstype="frames_finalpass", things_test=True)
    indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)),
                          dtype=np.int64).tolist()
    if args.disjoint_samples:
        excluded = set(indices)
        selected = []
        for index in indices:
            for shift in range(1, len(dataset) + 1):
                candidate = (index + shift) % len(dataset)
                if candidate not in excluded:
                    selected.append(candidate)
                    excluded.add(candidate)
                    break
            else:
                raise ValueError("Dataset cannot supply a disjoint sample set")
        indices = selected
    totals = empty_metrics(args.hr_iters)
    quality_totals = {}
    selected_pixels = 0
    edge4_pixels = 0
    selected_edge4_pixels = 0
    selected_abs_updates = [0.0] * (args.hr_iters + 1)
    direction_diagnostics = {label: {"pixels": 0, "direction_hits": 0,
                                     "absolute_update_sum": 0.}
                             for _, _, label in ERROR_BINS}
    for count, index in enumerate(indices, start=1):
        sample = dataset[index]
        left = sample["img1"].unsqueeze(0).cuda()
        right = sample["img2"].unsqueeze(0).cuda()
        gt = sample["disp"].unsqueeze(0).cuda()
        valid = sample["valid"].unsqueeze(0).cuda().bool()
        valid &= torch.isfinite(gt) & (gt > 0) & (gt < 768)
        predictions, query_mask = model.infer(
            left, right, mode=args.mode, gt=gt, valid=valid,
            band_radius=args.band_radius, boundary_threshold=args.edge_threshold,
            chunk_size=args.chunk_size)
        if args.oracle_update_direction:
            base = predictions[0]
            target_sign = torch.sign(gt - base)
            predictions = [base] + [
                (base + target_sign * (prediction - base).abs()).clamp(0, 768)
                for prediction in predictions[1:]]
        edge = disparity_boundary(gt, valid, args.edge_threshold)
        edge1 = dilate(edge, 1) & valid
        edge2 = dilate(edge, 2) & valid
        edge4 = dilate(edge, 4) & valid
        masks = {"overall": valid, "edge1": edge1, "edge2": edge2,
                 "edge4": edge4, "nonedge": valid & ~edge4}
        initial_error = (predictions[0] - gt).abs()
        quality_masks = {"selected_overall": query_mask & valid,
                         "selected_edge2": query_mask & edge2}
        for lower, upper, label in QUALITY_BINS:
            error_range = (initial_error >= lower) & (initial_error < upper)
            quality_masks[f"selected_init_error_{label}"] = query_mask & valid & error_range
            quality_masks[f"selected_edge2_init_error_{label}"] = query_mask & edge2 & error_range
        for name, mask in quality_masks.items():
            for iteration, prediction in enumerate(predictions):
                raw = update_quality_totals(predictions[0], prediction, gt, mask)
                if name not in quality_totals:
                    quality_totals[name] = [{k: 0 for k in raw} for _ in predictions]
                for key, value in raw.items():
                    quality_totals[name][iteration][key] += value
        for lower, upper, label in ERROR_BINS:
            error_range = (initial_error >= lower) & (initial_error < upper)
            masks[f"selected_init_error_{label}"] = query_mask & valid & error_range
            masks[f"edge4_init_error_{label}"] = edge4 & error_range
        selected_pixels += int(query_mask.sum())
        edge4_pixels += int(edge4.sum())
        selected_edge4_pixels += int((query_mask & edge4).sum())
        for iteration, prediction in enumerate(predictions):
            selected_abs_updates[iteration] += float(
                (prediction - predictions[0]).abs()[query_mask].sum())
            error = (prediction - gt).abs()
            for name, mask in masks.items():
                value = totals[name][iteration]
                value["pixels"] += int(mask.sum())
                value["error_sum"] += float(error[mask].sum())
                for threshold in (1, 3, 5):
                    value[f"bad{threshold}"] += int((error[mask] > threshold).sum())
        final_update = predictions[-1] - predictions[0]
        desired_update = gt - predictions[0]
        for _, _, label in ERROR_BINS:
            mask = masks[f"edge4_init_error_{label}"] & query_mask
            value = direction_diagnostics[label]
            value["pixels"] += int(mask.sum())
            value["direction_hits"] += int(((final_update * desired_update) > 0)[mask].sum())
            value["absolute_update_sum"] += float(final_update.abs()[mask].sum())
        if count % 10 == 0 or count == len(indices):
            print(f"{count}/{len(indices)}", flush=True)
    results = {}
    for name, sequence in totals.items():
        results[name] = []
        for value in sequence:
            pixels = value["pixels"]
            results[name].append({
                "pixels": pixels,
                "epe": value["error_sum"] / pixels if pixels else None,
                **{f"bad{threshold}_percent": 100 * value[f"bad{threshold}"] / pixels
                   if pixels else None for threshold in (1, 3, 5)},
            })
    summary = {"dataset": "SceneFlow FlyingThings3D finalpass TEST",
               "image_indices": indices, "disjoint_samples": args.disjoint_samples,
               "images": len(indices), "mode": args.mode, "checkpoint": args.checkpoint,
               "oracle_update_direction": args.oracle_update_direction,
               "refiner": args.refiner, "coarse_iters": args.coarse_iters,
               "refine_iters": args.refine_iters, "hr_iters": args.hr_iters,
               "selected_pixels": selected_pixels,
               "mean_abs_update_selected": [value / selected_pixels if selected_pixels else None
                                            for value in selected_abs_updates],
               "edge4_pixels": edge4_pixels,
               "selected_edge4_pixels": selected_edge4_pixels,
               "edge4_query_coverage": selected_edge4_pixels / edge4_pixels
               if edge4_pixels else None,
               "direction_diagnostics": {
                   label: {"pixels": value["pixels"],
                           "direction_hit_percent": 100 * value["direction_hits"] / value["pixels"]
                           if value["pixels"] else None,
                           "mean_abs_update": value["absolute_update_sum"] / value["pixels"]
                           if value["pixels"] else None}
                   for label, value in direction_diagnostics.items()},
               "update_threshold_px": .001,
               "update_quality": {name: [summarize_update_quality(v) for v in sequence]
                                  for name, sequence in quality_totals.items()},
               "metrics": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
