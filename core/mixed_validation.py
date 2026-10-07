"""Common, reproducible validation over several stereo datasets."""

from pathlib import Path
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from core import stereo_datasets as datasets
from core.utils.utils import InputPadder


NAMES = ("sceneflow", "middlebury_H", "middlebury_F", "unrealstereo4k", "sintel_stereo",
         "crestereo", "tartan_air", "vkitti2")

REFINE_BUCKETS = ((0, 192, "0-192"), (192, 384, "192-384"),
                  (384, 512, "384-512"), (512, 768, "512-768"))


@torch.no_grad()
def validate_refine_curve(model, names, roots, manifests, iters, scale_iters,
                          max_disp, max_samples, result_path):
    """Record pixel-weighted EPE numerators for d0 and every refine update."""
    model.eval()
    device = next(model.parameters()).device
    completed = {}
    output_path = Path(result_path)
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["left"] in completed:
                    raise ValueError(f"Duplicate image in {output_path}: {row['left']}")
                completed[row["left"]] = row
    for name in names:
        dataset = make_dataset(name, roots)
        indices = select_indices(dataset, manifests.get(name), max_samples)
        for index in tqdm(indices, desc=f"{name} refine curve"):
            left_path = dataset.image_list[index][0]
            if left_path in completed:
                continue
            sample = dataset[index]
            left = sample["img1"][None].to(device)
            right = sample["img2"][None].to(device)
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            predictions = model(left, right, iters=iters,
                                scale_iters=scale_iters, test_mode=False)
            if len(predictions) != iters + 1:
                raise ValueError(f"Expected d0 + {iters} predictions, got {len(predictions)}")
            gt = sample["disp"].to(device).squeeze(0)
            valid = (sample["valid"].to(device).squeeze(0) >= 0.5)
            valid &= torch.isfinite(gt) & (gt > 0) & (gt < max_disp)
            if not valid.any():
                raise ValueError(f"No valid GT for {left_path}")
            estimate = torch.stack([padder.unpad(pred).squeeze() for pred in predictions])
            if estimate.shape[1:] != gt.shape:
                raise ValueError(f"Prediction shape mismatch for {left_path}")
            if not torch.isfinite(estimate[:, valid]).all():
                raise ValueError(f"Nonfinite prediction for {left_path}")
            error = (estimate.float() - gt.float().unsqueeze(0)).abs()
            masks = {"0-768": valid}
            masks.update({label: valid & (gt >= lower) & (gt < upper)
                          for lower, upper, label in REFINE_BUCKETS})
            buckets = {label: {"pixels": int(mask.sum().item()),
                               "error_sums": error[:, mask].sum(dim=1).double().cpu().tolist()}
                       for label, mask in masks.items()}
            if sum(buckets[label]["pixels"] for _, _, label in REFINE_BUCKETS) != buckets["0-768"]["pixels"]:
                raise ValueError(f"Bucket counts do not partition GT for {left_path}")
            row = {"left": left_path, "buckets": buckets}
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a") as output:
                output.write(json.dumps(row) + "\n")
                output.flush()
            completed[left_path] = row
            del predictions, estimate, error
        print(f"{name}: {len(indices)} refine-curve images complete", flush=True)


class UnrealStereo4K(datasets.StereoDataset):
    def __init__(self, root, scene="00008"):
        super().__init__(is_eval=True, reader=self.read_disparity)
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"UnrealStereo4K root does not exist: {root}")
        scene_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if scene == "all" else [root / scene]
        self.skipped_disparities = 0
        for scene_dir in scene_dirs:
            if not scene_dir.is_dir():
                raise FileNotFoundError(f"UnrealStereo4K scene does not exist: {scene_dir}")
            for left in sorted((scene_dir / "Image0").glob("*")):
                if left.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                right = scene_dir / "Image1" / left.name
                disp = scene_dir / "Disp0" / (left.stem + ".npy")
                if not right.is_file() or not disp.is_file():
                    raise FileNotFoundError(f"Missing right image or disparity for {left}")
                try:
                    disparity = np.load(disp, mmap_mode="r", allow_pickle=False)
                    if disparity.shape != (2160, 3840) or disparity.dtype != np.float32:
                        raise ValueError(f"Unexpected disparity shape or dtype: {disp}")
                except (OSError, ValueError, EOFError):
                    self.skipped_disparities += 1
                    continue
                self.image_list.append([str(left), str(right)])
                self.disparity_list.append([str(disp)])
        print(f"UnrealStereo4K: {len(self.image_list)} readable samples, "
              f"{self.skipped_disparities} unreadable disparity files")

    @staticmethod
    def read_disparity(filename):
        disp = np.load(filename, allow_pickle=False).astype(np.float32)
        if disp.ndim != 2:
            raise ValueError(f"Expected 2D disparity in {filename}, got {disp.shape}")
        return disp, np.isfinite(disp) & (disp > 0)


def make_dataset(name, roots, unreal_scene="00008"):
    kwargs = {"root": roots[name]} if name in roots else {}
    if name == "sceneflow":
        return datasets.SceneFlowDatasets(dstype="frames_finalpass", things_test=True, **kwargs)
    if name == "middlebury_H":
        return datasets.Middlebury(split="H", is_eval=True, **kwargs)
    if name == "middlebury_F":
        return datasets.Middlebury(split="F", is_eval=True, **kwargs)
    if name == "unrealstereo4k":
        if name not in roots:
            raise ValueError("UnrealStereo4K requires --mixed_root unrealstereo4k=/path/to/data")
        return UnrealStereo4K(roots[name], scene=unreal_scene)
    constructors = {
        "sintel_stereo": datasets.SintelStereo,
        "crestereo": datasets.CreStereo,
        "tartan_air": datasets.TartanAir,
        "vkitti2": datasets.VKITTI2,
    }
    if name == "vkitti2" and name in roots:
        kwargs["depth_root"] = roots.get("vkitti2_depth", roots[name])
    dataset = constructors[name](**kwargs)
    dataset.is_eval = True
    return dataset


def select_indices(dataset, manifest=None, max_samples=100):
    if manifest:
        names = [line.strip() for line in Path(manifest).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if not names:
            raise ValueError(f"Empty validation manifest: {manifest}")
        lookup = {str(Path(pair[0]).resolve()): i for i, pair in enumerate(dataset.image_list)}
        indices = []
        for name in names:
            key = str(Path(name).resolve())
            if key not in lookup:
                raise ValueError(f"Image in {manifest} is not in the dataset: {name}")
            indices.append(lookup[key])
    else:
        indices = list(range(len(dataset)))
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--mixed_max_samples must be positive")
        if len(indices) > max_samples:
            # Cover the full ordered split instead of biasing evaluation toward
            # its first scene or sequence. This selection is deterministic.
            positions = np.linspace(0, len(indices) - 1, max_samples, dtype=np.int64)
            indices = [indices[position] for position in positions]
    if not indices:
        raise ValueError("No validation samples found; check the dataset root and split")
    return indices


@torch.no_grad()
def validate_mixed(model, names, roots, manifests, iters=32, scale_iters=8,
                   mixed_prec=False, max_disp=768, bad_threshold=3.0,
                   max_samples=100, unreal_scene="00008"):
    from core.corga.defom_cor_ga import autocast

    model.eval()
    refine_curve_path = os.environ.get("MIXED_REFINE_CURVE_JSONL")
    if refine_curve_path:
        return validate_refine_curve(model, names, roots, manifests, iters,
                                     scale_iters, max_disp, max_samples,
                                     refine_curve_path)
    device = next(model.parameters()).device
    results = {}
    total_error, total_bad, total_pixels = 0.0, 0, 0
    result_path = os.environ.get("MIXED_RESULT_JSONL")
    save_pred_dir = os.environ.get("MIXED_SAVE_PRED_DIR")
    collect_deltas = os.environ.get("MIXED_DELTA_DIAGNOSTICS") == "1"
    delta_stats = {}
    hooks = []
    if collect_deltas:
        def make_hook(label):
            def measure(_module, _inputs, output):
                values = output.detach().float()
                magnitude = values.abs()
                state = delta_stats.setdefault(label, {"count": 0, "calls": 0,
                    "sum_abs": 0.0, "sum_signed": 0.0, "gt8": 0, "gt16": 0,
                    "gt32": 0, "max_abs": 0.0})
                state["count"] += values.numel()
                state["calls"] += 1
                state["sum_abs"] += magnitude.sum().item()
                state["sum_signed"] += values.sum().item()
                state["gt8"] += int((magnitude > 8).sum())
                state["gt16"] += int((magnitude > 16).sum())
                state["gt32"] += int((magnitude > 32).sum())
                state["max_abs"] = max(state["max_abs"], magnitude.max().item())
            return measure
        for branch, block in enumerate(model.coarse_update_blocks):
            hooks.append(block.disp_head.register_forward_hook(make_hook(f"coarse{branch}")))
        hooks.append(model.refine_update_block.disp_head.register_forward_hook(make_hook("refine")))
    completed = {}
    skipped_path = result_path + ".skipped" if result_path else None
    skipped = set()
    if result_path and Path(result_path).exists():
        for line in Path(result_path).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["left"]] = record
    if skipped_path and Path(skipped_path).exists():
        for line in Path(skipped_path).read_text().splitlines():
            if line.strip():
                skipped.add(json.loads(line)["left"])
    for name in names:
        dataset = make_dataset(name, roots, unreal_scene)
        indices = select_indices(dataset, manifests.get(name), max_samples)
        error_sum, bad_sum, pixel_count, used = 0.0, 0, 0, 0
        bucket_totals = {label: {"pixels": 0, "error_sum": 0.0, "bad_sum": 0}
                         for label in ("0-192", "192-384", "384-512", "512-768")}
        for index in tqdm(indices, desc=name):
            left_path = dataset.image_list[index][0]
            if left_path in skipped:
                continue
            if left_path in completed:
                record = completed[left_path]
                error_sum += record["error_sum"]
                bad_sum += record["bad_sum"]
                pixel_count += record["valid_pixels"]
                used += 1
                for label, value in record["buckets"].items():
                    for metric in ("pixels", "error_sum", "bad_sum"):
                        bucket_totals[label][metric] += value[metric]
                continue
            try:
                sample = dataset[index]
            except (OSError, ValueError) as exc:
                if skipped_path:
                    with open(skipped_path, "a") as output:
                        output.write(json.dumps({"left": left_path, "error": str(exc)}) + "\n")
                        output.flush()
                skipped.add(left_path)
                print(f"Skipping unreadable sample {left_path}: {exc}")
                continue
            left = sample["img1"][None].to(device)
            right = sample["img2"][None].to(device)
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            with autocast(enabled=mixed_prec):
                delta_stats.clear()
                pred = model(left, right, iters=iters, scale_iters=scale_iters, test_mode=True)
            pred = padder.unpad(pred).cpu().squeeze(0)
            gt = sample["disp"]
            if pred.shape != gt.shape:
                raise ValueError(f"Prediction/GT shape mismatch in {sample['imageL_file']}: {pred.shape} vs {gt.shape}")
            if save_pred_dir:
                destination = Path(save_pred_dir)
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / f"{Path(left_path).parent.name}_pred.npy",
                        pred.squeeze(0).numpy())
            if name == "sceneflow":
                # Match validate_things: FlyingThings EPE averages image means,
                # while Bad3 pools all valid pixels across the TEST split.
                valid = (sample["valid"] >= 0.5) & torch.isfinite(gt) & (gt.abs() < max_disp)
            else:
                valid = (sample["valid"] >= 0.5) & torch.isfinite(gt) & (gt > 0) & (gt < max_disp)
            count = int(valid.sum())
            if count == 0:
                if skipped_path:
                    with open(skipped_path, "a") as output:
                        output.write(json.dumps({"left": left_path, "error": "no valid disparity pixels"}) + "\n")
                        output.flush()
                skipped.add(left_path)
                continue
            if not torch.isfinite(pred[valid]).all():
                raise ValueError(f"Nonfinite prediction in {sample['imageL_file']}")
            error = (pred - gt).abs()[valid]
            error_sum += error.sum().item()
            image_bad = int((error > bad_threshold).sum())
            bad_sum += image_bad
            pixel_count += count
            used += 1
            image_buckets = {}
            for lower, upper, label in ((0, 192, "0-192"), (192, 384, "192-384"),
                                        (384, 512, "384-512"), (512, 768, "512-768")):
                mask = valid & (gt >= lower) & (gt < upper)
                bucket_error = (pred - gt).abs()[mask]
                value = {"pixels": int(mask.sum()), "error_sum": bucket_error.sum().item(),
                         "bad_sum": int((bucket_error > bad_threshold).sum())}
                image_buckets[label] = value
                for metric in value:
                    bucket_totals[label][metric] += value[metric]
            regions = None
            if name == "middlebury_F":
                mask_path = Path(left_path).with_name("mask0nocc.png")
                mask_values = torch.from_numpy(np.asarray(Image.open(mask_path)).copy())
                if mask_values.shape != gt.shape[-2:]:
                    raise ValueError(f"Occlusion mask/GT shape mismatch: {mask_path}")
                regions = {}
                for region_name, mask_value in (("occ", 128), ("noc", 255)):
                    region_mask = valid & (mask_values == mask_value)
                    region_error = (pred - gt).abs()[region_mask]
                    region_buckets = {}
                    for lower, upper, label in ((0, 192, "0-192"), (192, 384, "192-384"),
                                                (384, 512, "384-512"), (512, 768, "512-768")):
                        bucket_mask = region_mask & (gt >= lower) & (gt < upper)
                        bucket_error = (pred - gt).abs()[bucket_mask]
                        region_buckets[label] = {
                            "pixels": int(bucket_mask.sum()),
                            "error_sum": bucket_error.sum().item(),
                            "bad_sum": int((bucket_error > bad_threshold).sum()),
                        }
                    regions[region_name] = {
                        "pixels": int(region_mask.sum()),
                        "error_sum": region_error.sum().item(),
                        "bad_sum": int((region_error > bad_threshold).sum()),
                        "buckets": region_buckets,
                    }
                if sum(regions[name]["pixels"] for name in regions) != count:
                    raise ValueError(f"Occlusion masks do not partition valid GT: {mask_path}")
            if result_path:
                record = {"left": left_path, "valid_pixels": count,
                          "error_sum": error.sum().item(), "bad_sum": image_bad,
                          "buckets": image_buckets}
                if regions is not None:
                    record["regions"] = regions
                if collect_deltas:
                    record["raw_delta"] = delta_stats.copy()
                with open(result_path, "a") as output:
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                completed[left_path] = record
        if pixel_count == 0:
            raise ValueError(f"{name}: no valid disparity pixels in selected samples")
        results[name] = {"images": used, "valid_pixels": pixel_count,
                         "epe": error_sum / pixel_count,
                         f"out{bad_threshold:g}": 100.0 * bad_sum / pixel_count}
        results[name]["buckets"] = {
            label: {"pixels": value["pixels"],
                    "epe": value["error_sum"] / value["pixels"] if value["pixels"] else None,
                    f"out{bad_threshold:g}": 100.0 * value["bad_sum"] / value["pixels"] if value["pixels"] else None}
            for label, value in bucket_totals.items()}
        total_error += error_sum
        total_bad += bad_sum
        total_pixels += pixel_count
        print(f"{name}: {results[name]}")
        print(f"{name}: skipped {len(skipped)} unreadable or empty samples")
    key = f"out{bad_threshold:g}"
    results["macro"] = {"epe": float(np.mean([results[n]["epe"] for n in names])),
                        key: float(np.mean([results[n][key] for n in names]))}
    results["pixel_weighted"] = {"epe": total_error / total_pixels,
                                 key: 100.0 * total_bad / total_pixels}
    print(f"Mixed macro: {results['macro']}")
    print(f"Mixed pixel weighted: {results['pixel_weighted']}")
    for hook in hooks:
        hook.remove()
    return results
