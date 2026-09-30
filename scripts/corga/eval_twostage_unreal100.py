"""Compare the original two-stage model and matched 10k fine-tunes on 100 UnrealStereo4K pairs."""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.corga.eval_no_mono_igev_clamp import BUCKETS, score


OUT = ROOT / "diagnostics/large_delta_eval"
MANIFEST = OUT / "unreal_allscenes_100_manifest.txt"
STATUS = OUT / "twostage_unreal100_comparison_status.json"
SUMMARY = OUT / "twostage_unreal100_comparison_summary.json"
TRAIN_STATUS = OUT / "twostage_coarse_free_control_10k_status.json"
SOURCE = ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"
CASES = {
    "source_8_8": {"gpu": 0, "checkpoint": SOURCE, "flags": []},
    "source_coarse_free_refine8": {"gpu": 0, "checkpoint": SOURCE,
                                   "flags": ["--disable_coarse_delta_clamp"]},
    "source_no_clamp": {"gpu": 0, "checkpoint": SOURCE,
                        "flags": ["--disable_delta_clamp"]},
    "ft10k_baseline_8_8": {"gpu": 0, "checkpoint": ROOT / "checkpoints/defomstereo_vitl_twostage_768_baseline8_8_ft10k/defomstereo_vitl_twostage_768_baseline8_8_ft10k_ 10000.pth",
                         "flags": []},
    "ft10k_coarse_free_refine8": {"gpu": 0, "checkpoint": ROOT / "checkpoints/defomstereo_vitl_twostage_768_coarse_free_refine8_ft10k/defomstereo_vitl_twostage_768_coarse_free_refine8_ft10k_ 10000.pth",
                                  "flags": ["--disable_coarse_delta_clamp"]},
}
LOCK = threading.Lock()
STAGES = {label: "pending" for label in CASES}
GPU0_MIN_FREE_MIB = 20000


def write_status(state, **fields):
    with LOCK:
        payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "manifest": str(MANIFEST), "stages": dict(STAGES), **fields}
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(STATUS)


def wait_for_manifest():
    while not MANIFEST.is_file():
        time.sleep(15)
    images = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    if len(images) != len(set(images)) or len(images) != 100:
        raise ValueError(f"Expected 100 distinct images in {MANIFEST}")
    return set(images)


def result_path(label):
    return OUT / f"twostage_unreal100_{label}.jsonl"


def wait_for_gpu0(label):
    STAGES[label] = "waiting_for_gpu0"
    write_status("waiting_for_gpu0", minimum_free_mib=GPU0_MIN_FREE_MIB)
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        if int(output.strip().splitlines()[0]) >= GPU0_MIN_FREE_MIB:
            return
        time.sleep(120)


def read_rows(result):
    rows = {}
    if result.is_file():
        for line in result.read_text().splitlines():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate image in {result}: {row['left']}")
            rows[row["left"]] = row
    return rows


def evaluate(label, config, expected):
    checkpoint = config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = result_path(label)
    rows = read_rows(result)
    if set(rows) == expected:
        STAGES[label] = "complete"
        write_status("running")
        return rows
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "unrealstereo4k",
        "--mixed_root", "unrealstereo4k=/home/share/zhaobofan/UnrealStereo4K",
        "--unreal_scene", "all", "--mixed_manifest", f"unrealstereo4k={MANIFEST}",
        "--mixed_max_samples", "100", "--max_disp", "768",
        "--valid_iters", "32", "--scale_iters", "6",
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", "cor_ga_fast",
        "--hypothesis_scales", "24", "48", "96",
        "--mixed_precision", "--amp_dtype", "bf16",
        "--max_delta_disp", "8", *config["flags"],
        "--restore_ckpt", str(checkpoint),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    environment["MIXED_RESULT_JSONL"] = str(result)
    wait_for_gpu0(label)
    STAGES[label] = "evaluating"
    write_status("running")
    with result.with_suffix(".log").open("a") as output:
        output.write("\nCOMMAND: " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    rows = read_rows(result)
    if set(rows) != expected:
        raise ValueError(f"Incomplete or mismatched {label} image set: {len(rows)} images")
    STAGES[label] = "complete"
    write_status("running")
    return rows


def wait_for_training():
    while True:
        if TRAIN_STATUS.is_file():
            state = json.loads(TRAIN_STATUS.read_text()).get("state")
            if state == "complete":
                return
            if state == "error":
                raise RuntimeError(f"10k training/SceneFlow evaluation failed: {TRAIN_STATUS}")
        time.sleep(60)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_status("waiting_for_manifest")
    expected = wait_for_manifest()
    runs = {}
    for label in ("source_8_8", "source_coarse_free_refine8", "source_no_clamp"):
        runs[label] = evaluate(label, CASES[label], expected)
    write_status("waiting_for_training")
    wait_for_training()
    for label in ("ft10k_baseline_8_8", "ft10k_coarse_free_refine8"):
        runs[label] = evaluate(label, CASES[label], expected)
    reference = runs["source_8_8"]
    for label, rows in runs.items():
        for path, row in rows.items():
            other = reference[path]
            if row["valid_pixels"] != other["valid_pixels"]:
                raise ValueError(f"Valid pixel mismatch: {label}, {path}")
            for bucket in BUCKETS:
                if row["buckets"][bucket]["pixels"] != other["buckets"][bucket]["pixels"]:
                    raise ValueError(f"Bucket pixel mismatch: {label}, {path}, {bucket}")
    summary = {
        "manifest": str(MANIFEST), "images": 100,
        "runs": {label: {"gpu": CASES[label]["gpu"],
                         "checkpoint": str(CASES[label]["checkpoint"]),
                         "flags": CASES[label]["flags"], "overall": score(rows),
                         "buckets": {bucket: score(rows, bucket) for bucket in BUCKETS},
                         "images_better_than_source_8_8": sum(
                             row["error_sum"] / row["valid_pixels"]
                             < reference[path]["error_sum"] / reference[path]["valid_pixels"]
                             for path, row in rows.items())}
                 for label, rows in runs.items()},
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
