"""Matched 10k SceneFlow fine-tunes: 8/8 baseline vs free coarse + refine 8."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.corga.eval_no_mono_igev_clamp import BUCKETS, score


OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "twostage_coarse_free_control_10k_status.json"
SUMMARY = OUT / "twostage_coarse_free_control_10k_summary.json"
SOURCE = ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth"
CASES = {
    "baseline_8_8": {"gpu": 2, "name": "defomstereo_vitl_twostage_768_baseline8_8_ft10k", "flags": []},
    "coarse_free_refine8": {"gpu": 0, "name": "defomstereo_vitl_twostage_768_coarse_free_refine8_ft10k",
                            "flags": ["--disable_coarse_delta_clamp"]},
}
LOCK = threading.Lock()
STAGES = {label: "pending" for label in CASES}


def write_status(state, **fields):
    with LOCK:
        payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "source": str(SOURCE), "batch_size": 2, "additional_steps": 10000,
                   "stages": dict(STAGES), "training_gpus": [2, 3],
                   "remaining_evaluation_gpu": 0, **fields}
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(STATUS)


def checkpoint(config):
    name = config["name"]
    return ROOT / "checkpoints" / name / f"{name}_{10000:6d}.pth"


def run_command(command, gpu, log, *, result=None):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if result:
        environment["MIXED_RESULT_JSONL"] = str(result)
    with log.open("a") as output:
        output.write("\nCOMMAND: " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def train(label, config):
    target = checkpoint(config)
    if target.is_file():
        STAGES[label] = "trained"
        write_status("running")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    latest = target.parent / "checkpoint_latest.pth"
    resume = latest if latest.is_file() else SOURCE
    resume_flags = ["--resume_ckpt", str(resume), "--strict_resume"]
    if resume == SOURCE:
        resume_flags.append("--no_resume_optimizer")
    else:
        saved = torch.load(str(resume), map_location="cpu", mmap=True, weights_only=False)
        resume_step = int(saved["step"])
        del saved
        if resume_step < 17727:
            resume_flags.extend(["--resume_batch_skip", str(resume_step)])
    command = [
        sys.executable, str(ROOT / "train_stereo.py"),
        "--model_arch", "cor_ga", "--dinov2_encoder", "vitl",
        "--launcher", "none", "--gpu_ids", "0", "--name", config["name"],
        "--batch_size", "2", "--num_workers", "4",
        "--train_datasets", "sceneflow", "--train_folds", "1",
        "--num_steps", "10000", "--save_ckpt_freq", "5000",
        "--save_latest_ckpt_freq", "1000", "--n_downsample", "2",
        "--train_iters", "18", "--scale_iters", "6",
        "--coarse_loss_weight", "0.2", "--loss_gamma", "0.9",
        "--smooth_l1_beta", "1.0", "--train_max_disp", "768",
        "--eval_max_disp", "768", "--corr_levels", "3", "--corr_radius", "4",
        "--context_norm", "batch", "--n_gru_layers", "3",
        "--hidden_dims", "128", "--image_size", "256", "768",
        "--wdecay", "0.00001", "--seed", "1234565",
        "--val_freq", "10001", "--valid_iters", "32",
        "--amp_dtype", "bf16", "--mixed_precision", "--lr", "0.00001",
        "--hypothesis_scales", "24", "48", "96",
        "--max_delta_disp", "8", *config["flags"], *resume_flags,
    ]
    STAGES[label] = "training"
    write_status("running")
    run_command(command, config["gpu"], target.parent / "train.log")
    if not target.is_file():
        raise FileNotFoundError(target)
    STAGES[label] = "trained"
    write_status("running")


def evaluate(label, config):
    result = OUT / f"twostage_{label}_ft10k_sceneflow.jsonl"
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "sceneflow",
        "--mixed_root", "sceneflow=/home/share/yijiayi/sceneflow",
        "--mixed_max_samples", "4370", "--max_disp", "768",
        "--valid_iters", "32", "--scale_iters", "6",
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", "cor_ga",
        "--mixed_precision", "--amp_dtype", "bf16",
        "--hypothesis_scales", "24", "48", "96",
        "--max_delta_disp", "8", *config["flags"],
        "--restore_ckpt", str(checkpoint(config)),
    ]
    STAGES[label] = "evaluating"
    write_status("running")
    existing = [json.loads(line) for line in result.read_text().splitlines() if line] if result.is_file() else []
    if len(existing) != 4370 or len({row["left"] for row in existing}) != 4370:
        run_command(command, config["gpu"], OUT / f"twostage_{label}_ft10k_sceneflow.log", result=result)
    rows = {}
    for line in result.read_text().splitlines():
        row = json.loads(line)
        if row["left"] in rows:
            raise ValueError(f"Duplicate {label} image: {row['left']}")
        rows[row["left"]] = row
    if len(rows) != 4370:
        raise ValueError(f"Expected 4370 images for {label}, found {len(rows)}")
    STAGES[label] = "complete"
    write_status("running")
    return rows


def run_case(label, config):
    train(label, config)
    return label, evaluate(label, config)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    write_status("running")
    runs = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_case, label, config) for label, config in CASES.items()]
        for future in as_completed(futures):
            label, rows = future.result()
            runs[label] = rows
    baseline = runs["baseline_8_8"]
    experiment = runs["coarse_free_refine8"]
    if baseline.keys() != experiment.keys():
        raise ValueError("Validation image sets differ")
    for path, row in baseline.items():
        other = experiment[path]
        if row["valid_pixels"] != other["valid_pixels"]:
            raise ValueError(f"Valid pixel mismatch: {path}")
        for bucket in BUCKETS:
            if row["buckets"][bucket]["pixels"] != other["buckets"][bucket]["pixels"]:
                raise ValueError(f"Bucket pixel mismatch: {path}, {bucket}")
    summary = {
        "training": {"source": str(SOURCE), "steps": 10000, "batch_size": 2,
                     "learning_rate": 1e-5, "only_difference": "coarse delta clamp"},
        "runs": {label: {"gpu": CASES[label]["gpu"], "checkpoint": str(checkpoint(CASES[label])),
                         "overall": score(rows),
                         "buckets": {bucket: score(rows, bucket) for bucket in BUCKETS}}
                 for label, rows in runs.items()},
        "images_better_with_free_coarse": sum(
            experiment[path]["error_sum"] / experiment[path]["valid_pixels"]
            < row["error_sum"] / row["valid_pixels"] for path, row in baseline.items()),
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
