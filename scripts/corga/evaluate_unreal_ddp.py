"""Distributed, pixel-weighted UnrealStereo4K evaluation for CorGA models.

Launch with torchrun and an explicit CUDA_VISIBLE_DEVICES list excluding GPU 1.
Each rank owns different images; the model is not wrapped in DDP because there
is no training or gradient synchronization during evaluation.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore_ckpt", type=Path, required=True)
    parser.add_argument("--unreal_root", type=Path, default=Path("/home/share/zhaobofan/UnrealStereo4K"))
    parser.add_argument("--unreal_scene", default="all", help="Scene ID or 'all'")
    parser.add_argument("--manifest", type=Path, help="Optional shared left-image manifest")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all selected images")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="New directory for each setting; a matching config-only failed run can be retried")
    parser.add_argument("--model_arch", choices=("cor_ga", "cor_ga_fast", "cor_ga_fast3"),
                        default="cor_ga_fast")
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--scale_iters", type=int, default=6)
    parser.add_argument("--max_disp", type=float, default=768, help="GT validity limit in full-res pixels")
    parser.add_argument("--model_max_disp", type=float, default=768,
                        help="Model output clamp in full-res pixels")
    parser.add_argument("--bad_threshold", type=float, default=3.0)
    parser.add_argument("--max_delta_disp", type=float, default=8.0,
                        help="Refine update limit in quarter-res pixels")
    parser.add_argument("--max_coarse_delta_disp", type=float,
                        help="Coarse update limit; defaults to max_delta_disp")
    parser.add_argument("--disable_delta_clamp", action="store_true")
    parser.add_argument("--disable_coarse_delta_clamp", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=3)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--dinov2_encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl")
    parser.add_argument("--hypothesis_scales", type=float, nargs=3, default=(24, 48, 96))
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=(128,))
    parser.add_argument("--context_norm", choices=("batch", "group", "instance", "none"), default="batch")
    parser.add_argument("--n_gru_layers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--n_downsample", type=int, choices=(2,), default=2)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append missing samples using exactly the existing evaluation settings")
    parser.add_argument("--timeout_minutes", type=int, default=240)
    args = parser.parse_args()
    if args.max_samples < 0 or args.valid_iters < 0 or args.scale_iters < 0:
        parser.error("max_samples, valid_iters, and scale_iters must be non-negative")
    if args.max_disp <= 0 or args.model_max_disp <= 0 or args.bad_threshold <= 0:
        parser.error("disparity limits and bad_threshold must be positive")
    return args


def load_completed(output_dir, rank, allowed_paths, buckets):
    """Validate persisted records and reconstruct totals without using a GPU."""
    totals = torch.zeros(6 + 3 * len(buckets), dtype=torch.float64)
    completed = set()
    for kind in ("results", "skipped"):
        path = output_dir / f"{kind}_rank{rank}.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}; preserve and repair before resuming") from error
                left = record["left"]
                if left not in allowed_paths or left in completed:
                    raise ValueError(f"Foreign or duplicate sample in {path}:{line_number}: {left}")
                completed.add(left)
                if kind == "skipped":
                    totals[2] += 1
                    continue
                totals[1] += 1
                totals[3:6] += torch.tensor([record["valid_pixels"], record["error_sum"], record["bad_sum"]], dtype=torch.float64)
                for index, (_, _, label) in enumerate(buckets):
                    bucket = record["buckets"][label]
                    totals[6 + 3 * index:9 + 3 * index] += torch.tensor(
                        [bucket["pixels"], bucket["error_sum"], bucket["bad_sum"]], dtype=torch.float64)
        # Appending to a valid final record without a newline would corrupt it.
        if path.stat().st_size:
            with path.open("rb") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    raise ValueError(f"Missing final newline in {path}; repair before resuming")
    return completed, totals


def check_visible_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly, excluding physical GPU 1")
    identifiers = [value.strip() for value in visible.split(",")]
    if not all(value.isdecimal() for value in identifiers):
        raise RuntimeError("Use numeric GPU IDs in CUDA_VISIBLE_DEVICES so GPU 1 can be excluded")
    if 1 in {int(value) for value in identifiers}:
        raise RuntimeError("Physical GPU 1 is prohibited for this repository")
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate GPU IDs")


def build_model(args, device):
    from core.corga.defom_cor_ga import DEFOMStereo as CorGA
    from core.corga.defom_cor_ga_fast import DEFOMStereo as CorGAFast
    from core.corga.defom_cor_ga_fast3 import DEFOMStereo as CorGAFast3

    model_class = {"cor_ga": CorGA, "cor_ga_fast": CorGAFast,
                   "cor_ga_fast3": CorGAFast3}[args.model_arch]
    model = model_class(args)
    # PyTorch 2.1 accepts pathlib.Path for torch.load, but its mmap path
    # requires an actual string filename.
    checkpoint_path = str(args.restore_ckpt)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        # A legacy optimizer checkpoint may contain Python objects. Only use
        # this fallback for checkpoints from a trusted source.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    return model.to(device).eval()


def bucket_ranges(max_disp):
    edges = [0.0] + [float(value) for value in (192, 384, 512, 768) if value < max_disp]
    edges.append(float(max_disp))
    return [(lower, upper, f"{lower:g}-{upper:g}")
            for lower, upper in zip(edges[:-1], edges[1:])]


def totals_to_summary(totals, buckets):
    values = totals.tolist()
    selected, images, skipped, pixels, error, bad = values[:6]
    if pixels == 0:
        raise ValueError("No valid pixels were evaluated")
    summary = {"selected_images": int(selected), "evaluated_images": int(images),
               "skipped_images": int(skipped), "valid_pixels": int(pixels),
               "epe": error / pixels, "bad_percent": 100 * bad / pixels,
               "buckets": {}}
    for index, (_, _, label) in enumerate(buckets):
        count, bucket_error, bucket_bad = values[6 + 3 * index:9 + 3 * index]
        summary["buckets"][label] = {
            "pixels": int(count),
            "epe": bucket_error / count if count else None,
            "bad_percent": 100 * bucket_bad / count if count else None,
        }
    return summary


@torch.inference_mode()
def evaluate(args, rank, world_size, device):
    from core.mixed_validation import UnrealStereo4K, select_indices
    from core.utils.utils import InputPadder

    dataset = UnrealStereo4K(args.unreal_root, scene=args.unreal_scene)
    indices = select_indices(dataset, args.manifest,
                             max_samples=args.max_samples or None)
    paths = [dataset.image_list[index][0] for index in indices]
    if len(paths) != len(set(paths)):
        raise ValueError("The selected manifest contains duplicate left images")
    local_indices = indices[rank::world_size]
    buckets = bucket_ranges(args.max_disp)
    allowed_paths = {dataset.image_list[index][0] for index in local_indices}
    completed, totals = (load_completed(args.output_dir, rank, allowed_paths, buckets)
                         if args.resume else (set(), torch.zeros(6 + 3 * len(buckets), dtype=torch.float64)))
    totals[0] = len(local_indices)
    pending = [index for index in local_indices if dataset.image_list[index][0] not in completed]
    print(f"rank {rank}: {len(completed)} saved, {len(pending)} pending", flush=True)
    model = build_model(args, device) if pending else None

    result_path = args.output_dir / f"results_rank{rank}.jsonl"
    skipped_path = args.output_dir / f"skipped_rank{rank}.jsonl"
    with result_path.open("a" if args.resume else "w") as result_file, skipped_path.open("a" if args.resume else "w") as skipped_file:
        for position, index in enumerate(pending, start=1):
            left_path = dataset.image_list[index][0]
            try:
                sample = dataset[index]
            except (OSError, ValueError) as error:
                skipped_file.write(json.dumps({"left": left_path, "error": str(error)}) + "\n")
                skipped_file.flush()
                totals[2] += 1
                continue

            gt = sample["disp"]
            valid = ((sample["valid"] >= 0.5) & torch.isfinite(gt)
                     & (gt > 0) & (gt < args.max_disp))
            count = int(valid.sum())
            if count == 0:
                skipped_file.write(json.dumps({"left": left_path, "error": "no valid disparity pixels"}) + "\n")
                skipped_file.flush()
                totals[2] += 1
                continue

            left = sample["img1"][None].to(device)
            right = sample["img2"][None].to(device)
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            # Match evaluate_stereo.py's default 'reg' correlation path: model
            # internals decide where AMP is safe; FullCorr remains FP32.
            prediction = model(left, right, iters=args.valid_iters,
                               scale_iters=args.scale_iters, test_mode=True)
            prediction = padder.unpad(prediction).float().cpu().squeeze(0)
            if prediction.shape != gt.shape:
                raise ValueError(f"Prediction/GT shape mismatch for {left_path}: "
                                 f"{tuple(prediction.shape)} vs {tuple(gt.shape)}")
            if not torch.isfinite(prediction[valid]).all():
                raise ValueError(f"Non-finite prediction for {left_path}")

            abs_error = (prediction - gt).abs()
            selected_error = abs_error[valid]
            error_sum = selected_error.sum(dtype=torch.float64).item()
            bad_sum = int((selected_error > args.bad_threshold).sum())
            totals[1] += 1
            totals[3] += count
            totals[4] += error_sum
            totals[5] += bad_sum
            record = {"left": left_path, "valid_pixels": count,
                      "error_sum": error_sum, "bad_sum": bad_sum, "buckets": {}}
            for bucket_index, (lower, upper, label) in enumerate(buckets):
                mask = valid & (gt >= lower) & (gt < upper)
                bucket_count = int(mask.sum())
                bucket_error = abs_error[mask]
                bucket_sum = bucket_error.sum(dtype=torch.float64).item()
                bucket_bad = int((bucket_error > args.bad_threshold).sum())
                offset = 6 + 3 * bucket_index
                totals[offset] += bucket_count
                totals[offset + 1] += bucket_sum
                totals[offset + 2] += bucket_bad
                record["buckets"][label] = {"pixels": bucket_count,
                                             "error_sum": bucket_sum, "bad_sum": bucket_bad}
            result_file.write(json.dumps(record) + "\n")
            result_file.flush()
            if rank == 0 and (position % 100 == 0 or position == len(local_indices)):
                print(f"rank 0: {position}/{len(local_indices)} selected images", flush=True)
    return totals, buckets, len(indices), dataset.skipped_disparities


def main():
    args = parse_args()
    check_visible_gpus()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    if not args.restore_ckpt.is_file():
        raise FileNotFoundError(args.restore_ckpt)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise RuntimeError("Launch with torchrun and at least two GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Only small CPU statistics are synchronized; inference is independent.
    dist.init_process_group(backend="gloo", init_method="env://",
                            timeout=timedelta(minutes=args.timeout_minutes))
    try:
        if rank == 0:
            config = vars(args).copy()
            config.pop("resume")
            config.pop("timeout_minutes")
            config.update({"world_size": world_size,
                           "visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"]})
            config = {key: str(value) if isinstance(value, Path) else value
                      for key, value in config.items()}
            # JSON turns argparse's tuple defaults into lists; normalize before
            # comparing against an existing config-only failed run.
            config = json.loads(json.dumps(config))
            config_path = args.output_dir / "config.json"
            if args.output_dir.exists():
                contents = set(args.output_dir.iterdir())
                allowed = {config_path} | {args.output_dir / f"{kind}_rank{rank}.jsonl"
                                          for rank in range(world_size) for kind in ("results", "skipped")}
                allowed.add(args.output_dir / "summary.json")
                if ((not args.resume and contents != {config_path}) or
                        (args.resume and not contents <= allowed) or
                        json.loads(config_path.read_text()) != config):
                    raise FileExistsError(
                        f"Output directory is not an empty matching retry: {args.output_dir}. "
                        "Use a new --output_dir for another run or setting."
                    )
                print(f"Resuming/retrying matching run in {args.output_dir}", flush=True)
            else:
                if args.resume:
                    raise FileNotFoundError("--resume requires an existing output directory")
                args.output_dir.mkdir(parents=True)
                config_path.write_text(json.dumps(config, indent=2) + "\n")
        dist.barrier()
        totals, buckets, selected, unreadable = evaluate(args, rank, world_size, device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            summary = totals_to_summary(totals, buckets)
            if summary["selected_images"] != selected:
                raise RuntimeError("Distributed image sharding was incomplete")
            summary["unreadable_disparity_files"] = unreadable
            summary["bad_threshold"] = args.bad_threshold
            path = args.output_dir / "summary.json"
            path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
            print(f"Saved summary to {path}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
