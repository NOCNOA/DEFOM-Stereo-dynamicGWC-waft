"""Evaluate the 200k no-mono IGEV checkpoint on full SceneFlow TEST."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "no_mono_igev_clamp_status.json"
CHECKPOINT = ROOT / "checkpoints/defomstereo_vitl_corga_no_mono_igev/defomstereo_vitl_corga_no_mono_igev_200000.pth"
SCENEFLOW = Path("/home/share/yijiayi/sceneflow")
BUCKETS = ("0-192", "192-384", "384-512", "512-768")
GPUS = (0, 2, 3)


def write_status(state, **fields):
    payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def prepare_manifests():
    images = sorted((SCENEFLOW / "FlyingThings3D/frames_finalpass/TEST").glob("*/*/left/*.png"))
    if len(images) != 4370:
        raise ValueError(f"Expected 4370 SceneFlow images, found {len(images)}")
    for shard, gpu in enumerate(GPUS):
        manifest = OUT / f"no_mono_igev_sceneflow_gpu{gpu}_manifest.txt"
        manifest.write_text("\n".join(map(str, images[shard::len(GPUS)])) + "\n")
    return images


def run_shard(gpu, label, flags):
    stem = f"no_mono_igev_200k_sceneflow_{label}_gpu{gpu}"
    manifest = OUT / f"no_mono_igev_sceneflow_gpu{gpu}_manifest.txt"
    result = OUT / f"{stem}.jsonl"
    log = OUT / f"{stem}.log"
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "sceneflow",
        "--mixed_root", f"sceneflow={SCENEFLOW}",
        "--mixed_manifest", f"sceneflow={manifest}",
        "--mixed_max_samples", "4370", "--max_disp", "768",
        "--valid_iters", "32", "--scale_iters", "6",
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", "cor_ga_no_mono_igev",
        "--num_init_bins", "48", "--init_max_offset", "48",
        "--init_corr_scale", "10",
        "--mixed_precision", "--amp_dtype", "bf16",
        "--restore_ckpt", str(CHECKPOINT), *flags,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["MIXED_RESULT_JSONL"] = str(result)
    with log.open("a") as output:
        output.write(f"\nCOMMAND: {' '.join(command)}\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    expected = len(manifest.read_text().splitlines())
    actual = sum(1 for line in result.open() if line.strip())
    if actual != expected:
        raise ValueError(f"{stem}: expected {expected} records, got {actual}")
    return stem


def run_gpu(gpu):
    return [
        run_shard(gpu, "no_clamp", ["--disable_delta_clamp"]),
        run_shard(gpu, "coarse32_refine16", [
            "--max_coarse_delta_disp", "32", "--max_delta_disp", "16",
        ]),
    ]


def load(label):
    rows = {}
    for gpu in GPUS:
        path = OUT / f"no_mono_igev_200k_sceneflow_{label}_gpu{gpu}.jsonl"
        for line in path.open():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate {label} sample: {row['left']}")
            rows[row["left"]] = row
    if len(rows) != 4370:
        raise ValueError(f"{label}: expected 4370 samples, got {len(rows)}")
    return rows


def score(rows, bucket=None):
    values = [row if bucket is None else row["buckets"][bucket] for row in rows.values()]
    count_key = "valid_pixels" if bucket is None else "pixels"
    pixels = sum(value[count_key] for value in values)
    result = {
        "pixels": pixels,
        "pixel_epe": sum(value["error_sum"] for value in values) / pixels,
        "bad3_percent": 100 * sum(value["bad_sum"] for value in values) / pixels,
    }
    if bucket is None:
        result["image_mean_epe"] = sum(
            row["error_sum"] / row["valid_pixels"] for row in rows.values()
        ) / len(rows)
    return result


def summarize():
    runs = {label: load(label) for label in ("no_clamp", "coarse32_refine16")}
    reference = runs["coarse32_refine16"]
    for label, rows in runs.items():
        if rows.keys() != reference.keys():
            raise ValueError(f"Image mismatch: {label}")
        for path, row in rows.items():
            other = reference[path]
            if row["valid_pixels"] != other["valid_pixels"]:
                raise ValueError(f"Valid-pixel mismatch: {label}, {path}")
            for bucket in BUCKETS:
                if row["buckets"][bucket]["pixels"] != other["buckets"][bucket]["pixels"]:
                    raise ValueError(f"Bucket-pixel mismatch: {label}, {path}, {bucket}")
    summary = {
        "checkpoint": str(CHECKPOINT),
        "baseline_cap8_train_log": {
            "images": 4370,
            "image_mean_epe": 0.8808449782482274,
            "bad3_percent": 3.1189774959531444,
        },
        "runs": {label: {
            "images": len(rows), "overall": score(rows),
            "buckets": {bucket: score(rows, bucket) for bucket in BUCKETS},
        } for label, rows in runs.items()},
    }
    summary["runs"]["no_clamp"]["images_better_than_32_16"] = sum(
        runs["no_clamp"][path]["error_sum"] / runs["no_clamp"][path]["valid_pixels"]
        < reference[path]["error_sum"] / reference[path]["valid_pixels"]
        for path in reference
    )
    path = OUT / "no_mono_igev_200k_sceneflow_clamp_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    prepare_manifests()
    write_status("running", checkpoint=str(CHECKPOINT), images=4370,
                 variants=["no_clamp", "coarse32_refine16"], gpus=list(GPUS))
    completed = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {pool.submit(run_gpu, gpu): gpu for gpu in GPUS}
        for future in as_completed(futures):
            completed.extend(future.result())
            write_status("running", completed=sorted(completed), images=4370,
                         variants=["no_clamp", "coarse32_refine16"], gpus=list(GPUS))
    summary = summarize()
    write_status("complete", completed=sorted(completed), summary=str(summary))
    print(summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("error", error=str(error))
        raise
