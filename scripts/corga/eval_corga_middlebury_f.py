"""Evaluate the two-stage CorGA checkpoint on all 15 Middlebury trainingF scenes."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "corga_middlebury_trainingF_status.json"
SUMMARY = OUT / "corga_middlebury_trainingF_summary.json"
CHECKPOINT = ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"
BUCKETS = ("0-192", "192-384", "384-512", "512-768")
CASES = {
    "baseline_8_8": [],
    "coarse_free_refine8": ["--disable_coarse_delta_clamp"],
}


def status(state, **fields):
    payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint": str(CHECKPOINT), "gpu": 0, **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def read_result(path):
    rows = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate scene in {path}: {row['left']}")
            rows[row["left"]] = row
    return rows


def evaluate(label, flags):
    result = OUT / f"corga_middlebury_trainingF_{label}.jsonl"
    rows = read_result(result)
    if len(rows) == 15:
        return rows
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "middlebury_F",
        "--mixed_root", "middlebury_F=/home/share/yijiayi/Middlebury",
        "--mixed_max_samples", "15", "--max_disp", "768",
        "--mixed_bad_threshold", "2",
        "--valid_iters", "32", "--scale_iters", "6",
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", "cor_ga_fast",
        "--hypothesis_scales", "24", "48", "96",
        "--mixed_precision", "--amp_dtype", "bf16",
        "--max_delta_disp", "8", *flags,
        "--restore_ckpt", str(CHECKPOINT),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["MIXED_RESULT_JSONL"] = str(result)
    status("running", case=label)
    with result.with_suffix(".log").open("a") as output:
        output.write("\nCOMMAND: " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    rows = read_result(result)
    if len(rows) != 15:
        raise ValueError(f"Expected 15 Middlebury scenes, found {len(rows)} for {label}")
    return rows


def aggregate(rows, bucket=None):
    values = [row if bucket is None else row["buckets"][bucket] for row in rows.values()]
    count_key = "valid_pixels" if bucket is None else "pixels"
    pixels = sum(value[count_key] for value in values)
    result = {"pixels": pixels,
              "pixel_epe": sum(value["error_sum"] for value in values) / pixels if pixels else None,
              "out2_percent": 100 * sum(value["bad_sum"] for value in values) / pixels if pixels else None}
    if bucket is None:
        result["scene_mean_epe"] = sum(
            row["error_sum"] / row["valid_pixels"] for row in rows.values()) / len(rows)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    runs = {label: evaluate(label, flags) for label, flags in CASES.items()}
    baseline = runs["baseline_8_8"]
    for label, rows in runs.items():
        if rows.keys() != baseline.keys():
            raise ValueError(f"Scene mismatch: {label}")
        for scene, row in rows.items():
            if row["valid_pixels"] != baseline[scene]["valid_pixels"]:
                raise ValueError(f"Valid pixel mismatch: {label}, {scene}")
            for bucket in BUCKETS:
                if row["buckets"][bucket]["pixels"] != baseline[scene]["buckets"][bucket]["pixels"]:
                    raise ValueError(f"Bucket pixel mismatch: {label}, {scene}, {bucket}")
    summary = {
        "dataset": "/home/share/yijiayi/Middlebury/MiddEval3/trainingF",
        "checkpoint": str(CHECKPOINT), "scenes": 15,
        "valid_disparity_range": "0 < GT < 768 full-resolution pixels",
        "outlier_threshold_full_pixels": 2,
        "runs": {label: {"overall": aggregate(rows),
                         "buckets": {bucket: aggregate(rows, bucket) for bucket in BUCKETS},
                         "per_scene": {Path(path).parent.name: {
                             "pixels": row["valid_pixels"],
                             "epe": row["error_sum"] / row["valid_pixels"],
                             "out2_percent": 100 * row["bad_sum"] / row["valid_pixels"],
                         } for path, row in rows.items()}}
                 for label, rows in runs.items()},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    status("complete", summary=str(SUMMARY))
    print(SUMMARY, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("error", error=str(error))
        raise
