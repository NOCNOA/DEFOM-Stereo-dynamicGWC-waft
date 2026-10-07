"""Full SceneFlow TEST: pixel-weighted EPE after each CorGA refine update."""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCALE_ITERS = int(os.environ.get("CORGA_SCALE_ITERS", "6"))
if SCALE_ITERS < 0:
    raise ValueError("CORGA_SCALE_ITERS must be non-negative")
OUT = ROOT / "diagnostics/large_delta_eval" / (
    "refine_curve_sceneflow_no_clamp" if SCALE_ITERS == 6
    else f"refine_curve_sceneflow_no_clamp_coarse{SCALE_ITERS}")
DATA = Path("/home/share/yijiayi/sceneflow")
CHECKPOINT = ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"
GPUS = (2, 3)
ITERATIONS = 32
SAMPLE_COUNT = 500
LABELS = ("0-192", "192-384", "384-512", "512-768", "0-768")


def write_status(state, **extra):
    data = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(CHECKPOINT), "gpus": list(GPUS), **extra}
    temporary = OUT / "status.json.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(OUT / "status.json")


def run_shard(gpu):
    manifest = OUT / f"manifest_gpu{gpu}.txt"
    result = OUT / f"curve_gpu{gpu}.jsonl"
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "sceneflow",
        "--mixed_root", f"sceneflow={DATA}",
        "--mixed_manifest", f"sceneflow={manifest}",
        "--mixed_max_samples", str(SAMPLE_COUNT), "--max_disp", "768",
        "--valid_iters", str(ITERATIONS), "--scale_iters", str(SCALE_ITERS),
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", "cor_ga_fast",
        "--hypothesis_scales", "24", "48", "96",
        "--mixed_precision", "--amp_dtype", "bf16",
        "--max_delta_disp", "8", "--disable_delta_clamp",
        "--restore_ckpt", str(CHECKPOINT),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MIXED_REFINE_CURVE_JSONL"] = str(result)
    with (OUT / f"gpu{gpu}.log").open("a") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    expected = set(manifest.read_text().splitlines())
    rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
    if len(rows) != len(expected) or {row["left"] for row in rows} != expected:
        raise ValueError(f"Incomplete or duplicate GPU {gpu} result")
    return rows


def summarize(shards):
    rows = [row for shard in shards for row in shard]
    if len(rows) != SAMPLE_COUNT or len({row["left"] for row in rows}) != SAMPLE_COUNT:
        raise ValueError(f"Expected exactly {SAMPLE_COUNT} distinct TEST images")
    curves = {}
    for label in LABELS:
        pixels = sum(row["buckets"][label]["pixels"] for row in rows)
        error_sums = [sum(row["buckets"][label]["error_sums"][iteration]
                          for row in rows) for iteration in range(ITERATIONS + 1)]
        curves[label] = {"pixels": pixels, "epe": [value / pixels for value in error_sums]}
    if curves["0-768"]["pixels"] != sum(curves[label]["pixels"] for label in LABELS[:-1]):
        raise ValueError("Bucket counts do not partition overall pixels")
    summary = {
        "dataset": str(DATA / "FlyingThings3D/frames_finalpass/TEST"),
        "checkpoint": str(CHECKPOINT),
        "images": len(rows), "iterations": ITERATIONS,
        "coarse_iterations_per_branch": SCALE_ITERS,
        "clamp": "no per-update clamp; final disparity bounded to [0,768]",
        "valid": "valid>=0.5 and finite GT and 0<GT<768",
        "aggregation": "sum absolute error / sum valid pixels per GT bucket",
        "iteration_0": "coarse-fused d0 before refine",
        "curves": curves,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "epe_by_iteration.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["refine_iteration", *LABELS])
        for iteration in range(ITERATIONS + 1):
            writer.writerow([iteration, *(curves[label]["epe"][iteration] for label in LABELS)])
    write_status("complete", images=len(rows), summary=str(OUT / "summary.json"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted((DATA / "FlyingThings3D/frames_finalpass/TEST").glob("*/*/left/*.png"))
    if len(images) != 4370:
        raise ValueError(f"Expected 4370 TEST images, got {len(images)}")
    images = [images[round(index * (len(images) - 1) / (SAMPLE_COUNT - 1))]
              for index in range(SAMPLE_COUNT)]
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    for index, gpu in enumerate(GPUS):
        (OUT / f"manifest_gpu{gpu}.txt").write_text(
            "\n".join(map(str, images[index::len(GPUS)])) + "\n")
    write_status("running", images_expected=len(images))
    try:
        with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            shards = list(pool.map(run_shard, GPUS))
        summarize(shards)
    except Exception as error:
        write_status("error", error=str(error))
        raise


if __name__ == "__main__":
    main()
