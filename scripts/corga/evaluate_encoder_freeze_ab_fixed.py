"""Compare encoder-freeze checkpoints on identical FlyingThings3D test images."""

import argparse
import csv
import gc
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.corga.defom_cor_ga_stereo_igevpp_encoder_coarse_corr_only import DEFOMStereo
from core.stereo_datasets import SceneFlowDatasets
from core.utils.utils import InputPadder


EXPERIMENT = "defomstereo_corga_igevpp_encoder_coarse_corr_only"
CHECKPOINTS = {
    "source20": ROOT / "checkpoints" / EXPERIMENT / "checkpoint_full_step_000020000.pth",
    "baseline25": ROOT / "checkpoints" / f"{EXPERIMENT}_ab_baseline" / "checkpoint_full_step_000025000.pth",
    "frozen25": ROOT / "checkpoints" / f"{EXPERIMENT}_ab_frozen" / "checkpoint_full_step_000025000.pth",
    "baseline30": ROOT / "checkpoints" / f"{EXPERIMENT}_ab_baseline_25to30_gpu23" / "checkpoint_full_step_000030000.pth",
    "frozen30": ROOT / "checkpoints" / f"{EXPERIMENT}_ab_frozen_25to30_gpu23" / "checkpoint_full_step_000030000.pth",
}


def model_args():
    return SimpleNamespace(
        amp_dtype="bf16", mixed_precision=True, corr_levels=3, corr_radius=4,
        hidden_dims=[128], n_gru_layers=3, context_norm="batch", n_downsample=2,
        train_max_disp=768, max_delta_disp=8.0, num_init_bins=48,
        init_max_offset=48.0, init_corr_scale=10.0,
        init_right_compression="avg", freeze_bininit_base=False,
    )


def load_weights(model, path):
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    del checkpoint
    gc.collect()


def branch_eligible_masks(aux, gt, valid, padder):
    """Use the same per-branch valid-bin-range definition as training."""
    gt_pad = padder.pad(gt[None])[0]
    valid_pad = padder.pad(valid.float()[None])[0] >= 0.5
    candidate_valid = aux["init_valid"].bool()
    _, branches, bins_count, height, width = candidate_valid.shape
    bins = aux["init_bins"].float().view(1, branches, bins_count, 1, 1)
    expanded = bins.expand_as(candidate_valid)
    minimum = expanded.masked_fill(~candidate_valid, float("inf")).amin(dim=2)
    maximum = expanded.masked_fill(~candidate_valid, float("-inf")).amax(dim=2)
    gt_low = F.interpolate(gt_pad.float(), size=(height, width), mode="nearest") / 4.0
    valid_low = F.interpolate(valid_pad.float(), size=(height, width), mode="nearest") >= 0.5
    eligible = candidate_valid.any(dim=2) & valid_low
    eligible &= (gt_low >= minimum) & (gt_low <= maximum)
    eligible = F.interpolate(eligible.float(), size=gt_pad.shape[-2:], mode="nearest") >= 0.5
    return [padder.unpad(eligible[:, i:i + 1])[0] & valid
            for i in range(branches)]


def evaluate_one(model, sample, diagnostics=False):
    left = sample["img1"][None].cuda(non_blocking=True)
    right = sample["img2"][None].cuda(non_blocking=True)
    gt = sample["disp"].cuda()
    valid = sample["valid"].cuda().bool() & torch.isfinite(gt) & (gt.abs() < 768)
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    projection_stats = {}
    hooks = []
    if diagnostics:
        for level, projection in enumerate(model.refine_state_projections):
            def capture(module, inputs, output, level=level):
                source = inputs[0].float()
                result = output.float()
                projection_stats[f"level{level}"] = {
                    "input_rms": float(source.square().mean().sqrt().item()),
                    "output_rms": float(result.square().mean().sqrt().item()),
                    "output_abs_gt4": float((result.abs() > 4).float().mean().item()),
                    "output_abs_gt8": float((result.abs() > 8).float().mean().item()),
                }
            hooks.append(projection.register_forward_hook(capture))
    with torch.inference_mode():
        try:
            predictions, aux = model(left, right, iters=32, scale_iters=6,
                                     test_mode=False, return_aux=True)
        finally:
            for hook in hooks:
                hook.remove()
        curves = {"refine": predictions}
        curves.update({f"coarse{branch}": curve
                       for branch, curve in enumerate(aux["coarse_predictions"])})
        result = {"valid_pixels": int(valid.sum().item())}
        if not result["valid_pixels"]:
            raise ValueError("Image has no valid pixels")
        for name, curve in curves.items():
            epes = []
            bad3 = []
            for prediction in curve:
                pred = padder.unpad(prediction)[0].float()
                if pred.shape != gt.shape:
                    raise ValueError(f"Shape mismatch: {pred.shape} != {gt.shape}")
                error = (pred - gt).abs()[valid]
                if not torch.isfinite(error).all():
                    raise ValueError(f"Nonfinite prediction in {name}")
                epes.append(float(error.mean().item()))
                bad3.append(float((error > 3).float().mean().item()))
            result[name] = epes
            result[f"{name}_bad3"] = bad3
        if diagnostics:
            result["projection"] = projection_stats
            result["hypothesis_weights"] = aux["w_h"].float().mean(dim=(0, 2, 3))[0:3].tolist()
            eligible_masks = branch_eligible_masks(aux, gt, valid, padder)
            for branch, mask in enumerate(eligible_masks):
                result[f"coarse{branch}_eligible_pixels"] = int(mask.sum().item())
                if not mask.any():
                    result[f"coarse{branch}_eligible"] = None
                    continue
                result[f"coarse{branch}_eligible"] = [
                    float((padder.unpad(prediction)[0].float() - gt).abs()[mask].mean().item())
                    for prediction in aux["coarse_predictions"][branch]
                ]
    return result


def summarize(rows):
    output = {}
    for name in ("refine", "coarse0", "coarse1", "coarse2"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        bad3 = np.asarray([row[f"{name}_bad3"] for row in rows], dtype=np.float64)
        weights = np.asarray([row["valid_pixels"] for row in rows], dtype=np.float64)
        output[name] = {
            "mean_image_epe": values.mean(axis=0).tolist(),
            "pixel_weighted_epe": np.average(values, axis=0, weights=weights).tolist(),
            "mean_image_bad3": bad3.mean(axis=0).tolist(),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=128)
    parser.add_argument("--match-encoder", choices=["igevpp", "dinov2"],
                        default="igevpp", help="image encoder used by all selected checkpoints")
    parser.add_argument("--checkpoints", nargs="+",
                        default=list(CHECKPOINTS))
    parser.add_argument("--extra-checkpoint", action="append", default=[],
                        metavar="NAME=PATH",
                        help="add a named checkpoint for the same fixed-image evaluation")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    checkpoint_paths = dict(CHECKPOINTS)
    for specification in args.extra_checkpoint:
        if "=" not in specification:
            parser.error("--extra-checkpoint must have the form NAME=PATH")
        name, path = specification.split("=", 1)
        if not name or not path or name in checkpoint_paths:
            parser.error(f"Invalid or duplicate extra checkpoint name: {name!r}")
        checkpoint_paths[name] = Path(path)
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    for name in args.checkpoints:
        if name not in checkpoint_paths:
            parser.error(f"Unknown checkpoint: {name}")
        if not checkpoint_paths[name].is_file():
            parser.error(f"Missing checkpoint: {checkpoint_paths[name]}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    dataset = SceneFlowDatasets(dstype="frames_finalpass", things_test=True)
    if args.num_images > len(dataset):
        parser.error(f"Requested {args.num_images} images; dataset has {len(dataset)}")
    indices = np.linspace(0, len(dataset) - 1, args.num_images, dtype=int).tolist()
    manifest = [{"index": i, "image": dataset.image_list[i][0]} for i in indices]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    eval_args = model_args()
    eval_args.match_encoder = args.match_encoder
    model = DEFOMStereo(eval_args).cuda().eval()
    model.freeze_bn()
    all_results = {}
    for name in args.checkpoints:
        print(f"Evaluating {name}: {checkpoint_paths[name]}", flush=True)
        load_weights(model, checkpoint_paths[name])
        model.eval()
        model.freeze_bn()
        rows = []
        for sequence, index in enumerate(indices, start=1):
            row = {"index": index, "image": dataset.image_list[index][0]}
            row.update(evaluate_one(model, dataset[index], diagnostics=args.diagnostics))
            rows.append(row)
            if sequence % 16 == 0 or sequence == len(indices):
                print(f"{name}: {sequence}/{len(indices)}; final EPE "
                      f"{np.mean([r['refine'][-1] for r in rows]):.4f}", flush=True)
        summary = summarize(rows)
        all_results[name] = summary
        (args.output / f"{name}_per_image.json").write_text(json.dumps(rows))
        (args.output / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        torch.cuda.empty_cache()
    (args.output / "all_summaries.json").write_text(json.dumps(all_results, indent=2))
    with (args.output / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["checkpoint", "final_epe", "d0_epe", "bin0_epe",
                         "bin1_epe", "bin2_epe", "coarse0_epe",
                         "coarse1_epe", "coarse2_epe"])
        for name, result in all_results.items():
            writer.writerow([name, result["refine"]["mean_image_epe"][-1],
                             result["refine"]["mean_image_epe"][0],
                             *[result[f"coarse{i}"]["mean_image_epe"][0] for i in range(3)],
                             *[result[f"coarse{i}"]["mean_image_epe"][-1] for i in range(3)]])
    print(f"Saved comparison to {args.output / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
