"""Evaluate CorGA's three coarse branches on a fixed SceneFlow TEST sample."""

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.corga.defom_cor_ga import DEFOMStereo
from core.stereo_datasets import SceneFlowDatasets
from core.utils.utils import InputPadder


BUCKETS = (("0-192", 0, 192), ("192-384", 192, 384),
           ("384-512", 384, 512), ("512-768", 512, 768), ("overall", 0, 768))


def summarize(errors, gt):
    """Sufficient statistics for errors shaped [branch, iteration, pixels]."""
    count = errors.shape[-1]
    return {
        "pixels": count,
        "error_sum": errors.sum(-1, dtype=torch.float64).tolist(),
        "bad1": (errors > 1).sum(-1).tolist(),
        "bad3": (errors > 3).sum(-1).tolist(),
        "d1": ((errors > 3) & (errors > .05 * gt)).sum(-1).tolist(),
    }


def aggregate(rows, bucket, branch, iteration):
    pixels = sum(row["buckets"][bucket]["pixels"] for row in rows)
    if not pixels:
        return {"pixels": 0, "epe": None, "bad1_pct": None,
                "bad3_pct": None, "d1_pct": None}
    total = lambda key: sum(row["buckets"][bucket][key][branch][iteration] for row in rows)
    return {"pixels": pixels, "epe": total("error_sum") / pixels,
            "bad1_pct": 100 * total("bad1") / pixels,
            "bad3_pct": 100 * total("bad3") / pixels,
            "d1_pct": 100 * total("d1") / pixels}


def save_results(out, metadata, rows, scales, coarse_iters):
    branches = {}
    for branch, scale in enumerate(scales):
        curve = [aggregate(rows, "overall", branch, iteration)
                 for iteration in range(coarse_iters + 1)]
        valid_images = [row for row in rows if row["buckets"]["overall"]["pixels"]]
        image_epes = [row["buckets"]["overall"]["error_sum"][branch][-1]
                      / row["buckets"]["overall"]["pixels"] for row in valid_images]
        branches[f"scale_{scale:g}"] = {
            "scale": scale, "curve": curve, "final": curve[-1],
            "final_image_mean_epe": float(np.mean(image_epes)),
            "final_by_gt_bucket": {label: aggregate(rows, label, branch, coarse_iters)
                                   for label, _, _ in BUCKETS[:-1]},
            "best_image_count": sum(row["best_branch_by_image_epe"] == branch for row in rows),
            "best_pixel_pct": 100 * sum(row["best_pixel_shares"][branch] for row in rows)
                              / curve[-1]["pixels"],
        }
    summary = {**metadata, "images": len(rows), "branches": branches}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["branch_scale", "coarse_iteration", "gt_bucket", "pixels",
                         "epe_px", "bad1_pct", "bad3_pct", "d1_pct"])
        for branch, scale in enumerate(scales):
            for iteration in range(coarse_iters + 1):
                for label, _, _ in BUCKETS:
                    metric = aggregate(rows, label, branch, iteration)
                    writer.writerow([scale, iteration, label, metric["pixels"],
                                     metric["epe"], metric["bad1_pct"],
                                     metric["bad3_pct"], metric["d1_pct"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for branch in branches.values():
        axes[0].plot(range(coarse_iters + 1), [item["epe"] for item in branch["curve"]],
                     marker="o", label=f"Scale {branch['scale']:g}")
        axes[1].plot(range(coarse_iters + 1), [item["bad3_pct"] for item in branch["curve"]],
                     marker="o", label=f"Scale {branch['scale']:g}")
    for axis, ylabel in zip(axes, ("Pixel-weighted EPE (px)", "Bad3 (%)")):
        axis.set_xlabel("Coarse iteration (0 = initialization)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(coarse_iters + 1))
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle(f"CorGA coarse branches | SceneFlow TEST {len(rows)} image pairs")
    figure.savefig(out / "coarse_curves.png", dpi=170)
    plt.close(figure)
    lines = ["# CorGA coarse branch evaluation", "",
             f"Checkpoint: `{metadata['checkpoint']}`", "",
             f"{len(rows)} fixed SceneFlow TEST image pairs, full resolution, "
             f"{coarse_iters} coarse updates per branch and no refinement.", "",
             "EPE is weighted by valid pixels; Bad3 is the percentage with absolute error >3 px.",
             "Iteration 0 uses bilinear upsampling; iterations 1 onward use the model's learned masks.",
             "", "| Scale | Initial EPE | Final EPE | Bad3 (%) | Best-pixel share (%) |",
             "|---|---:|---:|---:|---:|"]
    for branch in branches.values():
        lines.append(f"| {branch['scale']:g} | {branch['curve'][0]['epe']:.4f} | "
                     f"{branch['final']['epe']:.4f} | {branch['final']['bad3_pct']:.3f} | "
                     f"{branch['best_pixel_pct']:.3f} |")
    lines += ["", "Pixel ties divide their winning share equally among tied branches.",
              "", "## EPE after each coarse update", "",
              "| Iteration | Scale 24 | Scale 48 | Scale 96 |", "|---|---:|---:|---:|"]
    for iteration in range(coarse_iters + 1):
        values = " | ".join(f"{branch['curve'][iteration]['epe']:.4f}"
                            for branch in branches.values())
        lines.append(f"| {iteration} | {values} |")
    lines += ["", "## Final EPE by ground-truth disparity", "",
              "| GT range (px) | Pixels | Scale 24 | Scale 48 | Scale 96 |",
              "|---|---:|---:|---:|---:|"]
    for label, _, _ in BUCKETS[:-1]:
        metrics = [branch["final_by_gt_bucket"][label] for branch in branches.values()]
        values = " | ".join("n/a" if item["epe"] is None else f"{item['epe']:.4f}"
                            for item in metrics)
        lines.append(f"| {label} | {metrics[0]['pixels']:,} | {values} |")
    lines += ["", "Sparse or empty high-disparity buckets do not establish general performance.",
              "", "![Coarse curves](coarse_curves.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/"
                        "defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth")
    parser.add_argument("--data-root", type=Path, default=Path("/home/share/yijiayi/sceneflow"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--coarse-iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234565)
    args = parser.parse_args()
    if args.num_images <= 0 or args.coarse_iters <= 0:
        parser.error("num-images and coarse-iters must be positive")
    # Require explicit physical GPU selection; never allow GPU 1.
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdecimal() or int(gpu) == 1:
        parser.error("Select one GPU other than GPU 1 using CUDA_VISIBLE_DEVICES")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "per_image.jsonl").exists():
        raise FileExistsError("Use a new output directory to preserve existing results")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    scales = state["hypothesis_scales"].flatten().tolist()
    config = SimpleNamespace(
        dinov2_encoder="vitl", mixed_precision=True, amp_dtype="bf16",
        n_downsample=2, corr_levels=3, corr_radius=4, hypothesis_scales=scales,
        hidden_dims=[128], context_norm="batch", n_gru_layers=3,
        max_delta_disp=8., train_max_disp=768.,
    )
    model = DEFOMStereo(config)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()
    model.cuda().eval()
    dataset = SceneFlowDatasets(root=str(args.data_root), dstype="frames_finalpass", things_test=True)
    if args.num_images > len(dataset):
        raise ValueError(f"Requested {args.num_images} images; TEST contains {len(dataset)}")
    indices = sorted(np.random.default_rng(args.seed).choice(
        len(dataset), args.num_images, replace=False).tolist())
    paths = [dataset.image_list[index][0] for index in indices]
    manifest = "\n".join(paths) + "\n"
    (args.output / "manifest.txt").write_text(manifest)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "model_arch": "cor_ga",
        "data_root": str(args.data_root), "dataset": "FlyingThings3D TEST frames_finalpass",
        "sampling": "random without replacement, sorted dataset order", "seed": args.seed,
        "sample_indices": indices, "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "coarse_iterations_per_branch": args.coarse_iters, "refine_iterations": 0,
        "hypothesis_scales": scales, "gpu": gpu, "resolution": "native, padded to a multiple of 32",
        "valid": "dataset valid and finite GT and 0<GT<768",
        "aggregation": "sum absolute errors / sum valid pixels",
        "precision": "model's own BF16 AMP scopes; backbone/stereo heads and FullCorr FP32",
        "update_clamp": "8 quarter-resolution pixels per coarse update; output bounded to [0,768] px",
        "best_pixel_ties": "share equally", "best_image_ties": "first branch",
    }
    (args.output / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with torch.inference_mode(), (args.output / "per_image.jsonl").open("w") as output:
        for number, index in enumerate(indices, 1):
            sample = dataset[index]
            gt = sample["disp"].squeeze(0)
            valid = sample["valid"].squeeze(0).bool() & torch.isfinite(gt) & (gt > 0) & (gt < 768)
            if not valid.any():
                raise ValueError(f"No valid disparity pixels in {sample['imageL_file']}")
            left, right = sample["img1"][None].cuda(), sample["img2"][None].cuda()
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            _, aux = model(left, right, iters=0, scale_iters=args.coarse_iters, return_aux=True)
            if any(len(branch) != args.coarse_iters + 1 for branch in aux["coarse_predictions"]):
                raise ValueError("Unexpected number of coarse predictions")
            predictions = torch.stack([torch.cat([padder.unpad(value).float()
                                                  for value in branch], dim=0).squeeze(1)
                                       for branch in aux["coarse_predictions"]]).cpu()
            errors = (predictions - gt).abs()
            if not torch.isfinite(errors[..., valid]).all():
                raise ValueError(f"Non-finite predictions for {sample['imageL_file']}")
            row = {"index": index, "left": sample["imageL_file"], "buckets": {}}
            for label, low, high in BUCKETS:
                mask = valid & (gt >= low) & (gt < high)
                row["buckets"][label] = summarize(errors[..., mask], gt[mask])
            final_errors = errors[:, -1, valid]
            winners = final_errors == final_errors.min(dim=0).values
            row["best_pixel_shares"] = (winners.double() / winners.sum(0)).sum(1).tolist()
            row["best_branch_by_image_epe"] = int(final_errors.mean(1).argmin())
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
            del aux, predictions, errors, left, right
            if number % 20 == 0 or number == len(indices):
                print(f"Processed {number}/{len(indices)} pairs", flush=True)
    summary = save_results(args.output, metadata, rows, scales, args.coarse_iters)
    for label, branch in summary["branches"].items():
        print(f"{label}: EPE={branch['final']['epe']:.6f}, "
              f"Bad3={branch['final']['bad3_pct']:.4f}%", flush=True)
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
