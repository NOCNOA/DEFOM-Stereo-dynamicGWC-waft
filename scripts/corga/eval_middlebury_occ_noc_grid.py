"""Middlebury trainingF coarse-iteration/refine-clamp grid with occ/noc metrics."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "corga_middlebury_occ_noc_grid_status.json"
SUMMARY = OUT / "corga_middlebury_occ_noc_grid_summary.json"
CHECKPOINT = ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"
BUCKETS = ("0-192", "192-384", "384-512", "512-768")
CASES = {
    "default_c6_r8": {"scale_iters": 6, "flags": []},
    "free_coarse_c6_r8": {"scale_iters": 6, "flags": ["--disable_coarse_delta_clamp"]},
    "free_coarse_c12_r8": {"scale_iters": 12, "flags": ["--disable_coarse_delta_clamp"]},
    "free_coarse_c6_free_refine": {"scale_iters": 6, "flags": ["--disable_delta_clamp"]},
    "free_coarse_c12_free_refine": {"scale_iters": 12, "flags": ["--disable_delta_clamp"]},
}


def write_status(state, **fields):
    payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint": str(CHECKPOINT), "gpu": 0, "completed": {
                   label: result_path(label).is_file() and len(read_rows(result_path(label))) == 15
                   for label in CASES}, **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def result_path(label):
    return OUT / f"corga_middlebury_occ_noc_{label}.jsonl"


def read_rows(path):
    rows = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate scene in {path}: {row['left']}")
            rows[row["left"]] = row
    return rows


def evaluate(label, config):
    result = result_path(label)
    rows = read_rows(result)
    if len(rows) == 15 and all("regions" in row for row in rows.values()):
        return rows
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "middlebury_F",
        "--mixed_root", "middlebury_F=/home/share/yijiayi/Middlebury",
        "--mixed_max_samples", "15", "--max_disp", "768",
        "--mixed_bad_threshold", "2",
        "--valid_iters", "32", "--scale_iters", str(config["scale_iters"]),
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", "cor_ga_fast",
        "--hypothesis_scales", "24", "48", "96",
        "--mixed_precision", "--amp_dtype", "bf16",
        "--max_delta_disp", "8", *config["flags"],
        "--restore_ckpt", str(CHECKPOINT),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["MIXED_RESULT_JSONL"] = str(result)
    write_status("running", case=label)
    with result.with_suffix(".log").open("a") as output:
        output.write("\nCOMMAND: " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    rows = read_rows(result)
    if len(rows) != 15 or not all("regions" in row for row in rows.values()):
        raise ValueError(f"Expected 15 scored scenes with occ/noc regions for {label}")
    return rows


def aggregate(rows, region=None, bucket=None):
    values = []
    for row in rows.values():
        value = row if region is None else row["regions"][region]
        if bucket is not None:
            value = value["buckets"][bucket]
        values.append(value)
    count_key = "valid_pixels" if region is None and bucket is None else "pixels"
    pixels = sum(value[count_key] for value in values)
    result = {"pixels": pixels,
              "pixel_epe": sum(value["error_sum"] for value in values) / pixels if pixels else None,
              "out2_percent": 100 * sum(value["bad_sum"] for value in values) / pixels if pixels else None}
    if region is None and bucket is None:
        result["scene_mean_epe"] = sum(
            row["error_sum"] / row["valid_pixels"] for row in rows.values()) / len(rows)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    runs = {}
    for label, config in CASES.items():
        runs[label] = evaluate(label, config)
    reference = runs["default_c6_r8"]
    for label, rows in runs.items():
        if rows.keys() != reference.keys():
            raise ValueError(f"Scene mismatch: {label}")
        for scene, row in rows.items():
            old = reference[scene]
            if row["valid_pixels"] != old["valid_pixels"]:
                raise ValueError(f"Valid-pixel mismatch: {label}, {scene}")
            for region in ("occ", "noc"):
                if row["regions"][region]["pixels"] != old["regions"][region]["pixels"]:
                    raise ValueError(f"Region-pixel mismatch: {label}, {scene}, {region}")
    summary = {
        "dataset": "/home/share/yijiayi/Middlebury/MiddEval3/trainingF",
        "checkpoint": str(CHECKPOINT), "scenes": 15,
        "valid_disparity_range": "0 < GT < 768 full-resolution pixels",
        "occ_mask": "mask0nocc.png value 128",
        "noc_mask": "mask0nocc.png value 255",
        "outlier_threshold_full_pixels": 2,
        "refine_iterations": 32,
        "runs": {label: {
            "coarse_iterations_per_branch": CASES[label]["scale_iters"],
            "flags": CASES[label]["flags"],
            "overall": aggregate(rows),
            "buckets": {bucket: aggregate(rows, bucket=bucket) for bucket in BUCKETS},
            "regions": {region: {
                "overall": aggregate(rows, region=region),
                "buckets": {bucket: aggregate(rows, region=region, bucket=bucket)
                            for bucket in BUCKETS}}
                for region in ("occ", "noc")},
            "per_scene": {Path(path).parent.name: {
                "overall": {"pixels": row["valid_pixels"],
                            "epe": row["error_sum"] / row["valid_pixels"],
                            "out2_percent": 100 * row["bad_sum"] / row["valid_pixels"]},
                "occ": {"pixels": row["regions"]["occ"]["pixels"],
                        "epe": row["regions"]["occ"]["error_sum"] / row["regions"]["occ"]["pixels"],
                        "out2_percent": 100 * row["regions"]["occ"]["bad_sum"] / row["regions"]["occ"]["pixels"]},
                "noc": {"pixels": row["regions"]["noc"]["pixels"],
                        "epe": row["regions"]["noc"]["error_sum"] / row["regions"]["noc"]["pixels"],
                        "out2_percent": 100 * row["regions"]["noc"]["bad_sum"] / row["regions"]["noc"]["pixels"]},
            } for path, row in rows.items()},
        } for label, rows in runs.items()},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    write_status("complete", summary=str(SUMMARY))
    print(SUMMARY, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("error", error=str(error))
        raise
