"""Single-pass candidate/gate GT interventions on fixed SceneFlow TEST images."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.corga.sparse_hr_model import SparseHRCorGA
from core.corga.sparse_hr_refiner import (
    dilate, disparity_boundary, scatter_queries,
    update_quality_totals, summarize_update_quality,
)
from core.corga.sparse_hr_diagnostics import optimal_candidate, optimal_gate
from core.stereo_datasets import SceneFlowDatasets

VARIANTS = ("baseline", "model", "model_candidate_gt_gate",
            "gt_candidate_model_gate", "gt_candidate_gt_gate")
BINS = ((0., 1., "0-1"), (1., 4., "1-4"), (4., 8., "4-8"), (8., float("inf"), "8+"))


def add(total, raw):
    for key, value in raw.items():
        total[key] = total.get(key, 0) + value


def metric(prediction, gt, mask):
    error = (prediction - gt).abs()[mask]
    return {"pixels": error.numel(), "error_sum": float(error.sum()),
            **{f"bad{t}": int((error > t).sum()) for t in (1, 3, 5)}}


def finish_metric(raw):
    n = raw["pixels"]
    return {"pixels": n, "epe": raw["error_sum"] / n if n else None,
            **{f"bad{t}_percent": 100 * raw[f"bad{t}"] / n if n else None for t in (1, 3, 5)}}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gpu", type=int, choices=(0, 2, 3), required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--sceneflow-root", default="/home/share/yijiayi/sceneflow")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path = output.with_name(output.stem + "_status.json")

    def status(state, **extra):
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"state": state, "gpu": args.gpu,
                                        "refiner": args.refiner, **extra}, indent=2) + "\n")
        temporary.replace(status_path)

    status("running")
    try:
        saved = torch.load(args.refiner, map_location="cpu", weights_only=False)
        config = saved["config"]
        model = SparseHRCorGA(
            args.checkpoint, coarse_iters=6, refine_iters=32, hr_iters=1,
            feature_channels=config.get("feature_channels", 16),
            hidden_channels=config.get("hidden_channels", 64),
            context_channels=config.get("context_channels", 64),
            delta_radius=config.get("delta_radius", 2.),
            direction_classes=config.get("direction_classes", False),
            candidate_guided=True,
            context_scoring=config.get("context_scoring", False)).cuda().eval()
        model.refiner.load_state_dict(saved["refiner"], strict=True)
        dataset = SceneFlowDatasets(root=args.sceneflow_root,
                                    dstype="frames_finalpass", things_test=True)
        indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=np.int64).tolist()
        pooled, quality, coverage = {}, {}, {}
        captured_gate_input = []

        def capture_input(module, inputs):
            captured_gate_input[:] = [inputs[0]]

        hook = model.refiner.candidate_gate.register_forward_pre_hook(capture_input)
        for number, index in enumerate(indices, 1):
            sample = dataset[index]
            left, right = (sample[k].unsqueeze(0).cuda() for k in ("img1", "img2"))
            gt = sample["disp"].unsqueeze(0).cuda()
            valid = sample["valid"].unsqueeze(0).cuda().bool()
            valid &= torch.isfinite(gt) & (gt > 0) & (gt < 768)
            padder, lp, rp, full_pad, coarse = model._coarse_state(left, right)
            base = padder.unpad(full_pad)
            edge = disparity_boundary(gt, valid, 2.)
            edge2 = dilate(edge, 2) & valid
            edge4 = dilate(edge, 4) & valid
            queries = {"gt_boundary": edge2,
                       "predicted_boundary": dilate(disparity_boundary(base, torch.ones_like(valid), 2.), 2) & valid}
            union = queries["gt_boundary"] | queries["predicted_boundary"]
            xy = torch.nonzero(union[0, 0], as_tuple=False)[:, [1, 0]].float().unsqueeze(0)
            dense = {name: base.clone() for name in VARIANTS if name != "baseline"}
            feature_left, feature_right = model.refiner.feature(lp), model.refiner.feature(rp)
            pad_left, _, pad_top, _ = padder._pad
            for start in range(0, xy.shape[1], args.chunk_size):
                q = xy[:, start:start + args.chunk_size]
                padded_q = q + q.new_tensor((pad_left, pad_top))
                predictions, info = model.refiner(
                    lp, rp, full_pad, coarse["disp_lr_px"], coarse["hidden_lr"],
                    coarse["context_lr"], padded_q, feature_left=feature_left,
                    feature_right=feature_right, return_candidate_info=True)
                proposal, _, scores = info[0]
                gate_input = captured_gate_input[0]
                expected_width = model.refiner.gru.hidden_size + scores.shape[-1] + 3
                if gate_input.shape[-1] != expected_width:
                    raise ValueError("Unexpected gate input layout")
                gt_q = gt[0, 0, q[0, :, 1].long(), q[0, :, 0].long()].view(1, -1, 1)
                residual = gt_q - predictions[0]
                oracle_proposal, oracle_gate = optimal_candidate(
                    residual, model.refiner.offsets, model.refiner.delta_radius)
                modified_input = gate_input.clone()
                modified_input[..., -3:-2] = oracle_proposal / max(model.refiner.delta_radius, 1.)
                # Recompute the gate for the GT-selected proposal. Hidden/context,
                # candidate probabilities and current disparity remain model outputs.
                learned_gate = model.refiner.candidate_gate(modified_input).sigmoid()
                sparse = {"model": predictions[1],
                          "model_candidate_gt_gate": predictions[0] + proposal * optimal_gate(residual, proposal),
                          "gt_candidate_model_gate": predictions[0] + oracle_proposal * learned_gate,
                          "gt_candidate_gt_gate": predictions[0] + oracle_proposal * oracle_gate}
                for name, prediction in sparse.items():
                    dense[name] = scatter_queries(dense[name], q, prediction.clamp(0, 768))
            regions = {"overall": valid, "edge1": dilate(edge, 1) & valid,
                       "edge2": edge2, "edge4": edge4, "nonedge": valid & ~edge4}
            initial_error = (base - gt).abs()
            for mode, selected in queries.items():
                add(coverage.setdefault(mode, {}),
                    {"selected_valid_pixels": int(selected.sum()), "edge2_pixels": int(edge2.sum()),
                     "selected_edge2_pixels": int((selected & edge2).sum())})
                masks = {"selected": selected, "selected_edge2": selected & edge2}
                for low, high, label in BINS:
                    error_mask = (initial_error >= low) & (initial_error < high)
                    masks[f"selected_init_error_{label}"] = selected & error_mask
                    masks[f"selected_edge2_init_error_{label}"] = selected & edge2 & error_mask
                for variant in VARIANTS:
                    prediction = base if variant == "baseline" else torch.where(selected, dense[variant], base)
                    target = pooled.setdefault(mode, {}).setdefault(variant, {})
                    for region, mask in regions.items():
                        add(target.setdefault(region, {}), metric(prediction, gt, mask))
                    if variant != "baseline":
                        target_quality = quality.setdefault(mode, {}).setdefault(variant, {})
                        for label, mask in masks.items():
                            add(target_quality.setdefault(label, {}), update_quality_totals(base, prediction, gt, mask))
            if number % 10 == 0 or number == len(indices):
                print(f"{number}/{len(indices)}", flush=True)
        hook.remove()
        result = {"refiner": args.refiner, "images": len(indices), "image_indices": indices,
                  "hr_iterations": 1, "coarse_iterations": 6, "refine_iterations": 32,
                  "band_radius": 2, "boundary_threshold": 2., "update_threshold_px": .001,
                  "oracle_definition": "Optimize gate for each clamped candidate; select minimum GT error, tie by raw-offset GT proximity. Model gate is recomputed for oracle proposal with original model probabilities/context/hidden.",
                  "limitations": "GT interventions are diagnostic only. The full oracle is an action-range ceiling, not evidence of recoverable stereo correspondence; no visibility mask is applied.",
                  "query_modes": {}}
        for mode in queries:
            counts = coverage[mode]
            result["query_modes"][mode] = {
                "coverage": {**counts, "edge2_query_coverage_percent": 100 * counts["selected_edge2_pixels"] / counts["edge2_pixels"] if counts["edge2_pixels"] else None},
                "metrics": {variant: {region: finish_metric(raw) for region, raw in masks.items()}
                            for variant, masks in pooled[mode].items()},
                "update_quality": {variant: {label: summarize_update_quality(raw) for label, raw in masks.items()}
                                   for variant, masks in quality[mode].items()}}
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)
        status("complete", result=str(output), images=len(indices))
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
