"""Complete matched 2k-step no-mono IGEV clamp fine-tuning experiments."""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.corga.eval_no_mono_igev_clamp import BUCKETS, score


OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "no_mono_igev_three_way_ft2k_status.json"
SOURCE = ROOT / "checkpoints/defomstereo_vitl_corga_no_mono_igev/defomstereo_vitl_corga_no_mono_igev_200000.pth"
C32_SUMMARY = OUT / "no_mono_igev_c32r16_ft2k_sceneflow_summary.json"
VARIANTS = {
    "no_clamp": {"train_gpu": 2, "flags": ["--disable_delta_clamp"]},
    "coarse16_refine8": {"train_gpu": 3, "flags": ["--max_coarse_delta_disp", "16", "--max_delta_disp", "8"]},
}


def write_status(state, **fields):
    payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def checkpoint_for(label):
    name = f"defomstereo_vitl_corga_no_mono_igev_{label}_ft2k"
    return ROOT / "checkpoints" / name / f"{name}_{2000:6d}.pth"


def train(label):
    config = VARIANTS[label]
    name = f"defomstereo_vitl_corga_no_mono_igev_{label}_ft2k"
    ckpt = checkpoint_for(label)
    if ckpt.is_file():
        return label
    directory = ckpt.parent
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint_latest.pth").exists():
        raise RuntimeError(f"Partial run found in {directory}; inspect before restarting")
    command = [
        sys.executable, str(ROOT / "train_stereo.py"),
        "--model_arch", "cor_ga_no_mono_igev", "--dinov2_encoder", "vitl",
        "--launcher", "none", "--gpu_ids", "0", "--name", name,
        "--batch_size", "2", "--num_workers", "4",
        "--train_datasets", "sceneflow", "--train_folds", "1",
        "--num_steps", "2000", "--save_ckpt_freq", "1000",
        "--save_latest_ckpt_freq", "1000", "--n_downsample", "2",
        "--train_iters", "18", "--scale_iters", "6",
        "--coarse_loss_weight", "0.2", "--bin_loss_mode", "soft_ce",
        "--bin_disp_smooth_l1_weight", "0.2", "--num_init_bins", "48",
        "--init_max_offset", "48", "--init_corr_scale", "10",
        "--init_right_compression", "avg", "--loss_gamma", "0.9",
        "--smooth_l1_beta", "1.0", "--train_max_disp", "768",
        "--eval_max_disp", "768", "--corr_levels", "3", "--corr_radius", "4",
        "--context_norm", "batch", "--n_gru_layers", "3",
        "--hidden_dims", "128", "--image_size", "256", "768",
        "--wdecay", "0.00001", "--seed", "1234565",
        "--val_freq", "10000", "--valid_iters", "32",
        "--amp_dtype", "bf16", "--mixed_precision", "--lr", "0.00001",
        "--resume_ckpt", str(SOURCE), "--strict_resume", "--no_resume_optimizer",
        *config["flags"],
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["train_gpu"])
    with (directory / "train.log").open("w") as output:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    return label


def evaluate_gpu(gpu):
    manifest = OUT / f"no_mono_igev_sceneflow_gpu{gpu}_manifest.txt"
    expected = set(manifest.read_text().splitlines())
    for label, config in VARIANTS.items():
        stem = OUT / f"no_mono_igev_{label}_ft2k_sceneflow_gpu{gpu}"
        jsonl = stem.with_suffix(".jsonl")
        command = [
            sys.executable, str(ROOT / "evaluate_stereo.py"),
            "--datasets", "mixed", "--mixed_names", "sceneflow",
            "--mixed_root", "sceneflow=/home/share/yijiayi/sceneflow",
            "--mixed_manifest", f"sceneflow={manifest}",
            "--mixed_max_samples", "4370", "--max_disp", "768",
            "--valid_iters", "32", "--scale_iters", "6",
            "--corr_levels", "3", "--corr_radius", "4",
            "--dinov2_encoder", "vitl", "--model_arch", "cor_ga_no_mono_igev",
            "--num_init_bins", "48", "--init_max_offset", "48",
            "--init_corr_scale", "10", "--mixed_precision", "--amp_dtype", "bf16",
            "--restore_ckpt", str(checkpoint_for(label)), *config["flags"],
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["MIXED_RESULT_JSONL"] = str(jsonl)
        with stem.with_suffix(".log").open("a") as output:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
        if len(rows) != len(expected) or {row["left"] for row in rows} != expected:
            raise ValueError(f"Incomplete {label} shard on GPU {gpu}")
    return gpu


def load(label):
    rows = {}
    for gpu in (0, 2, 3):
        path = OUT / f"no_mono_igev_{label}_ft2k_sceneflow_gpu{gpu}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate {label} result: {row['left']}")
            rows[row["left"]] = row
    if len(rows) != 4370:
        raise ValueError(f"Expected 4370 {label} results; got {len(rows)}")
    return rows


def summarize():
    c32 = json.loads(C32_SUMMARY.read_text())
    expected_pixels = c32["overall"]["pixels"]
    result = {
        "training": {"source_checkpoint": str(SOURCE), "finetune_steps": 2000,
                     "batch_size": 2, "learning_rate": 1e-5},
        "original_cap8_train_log": c32["reference_original_cap8"],
        "zero_shot_32_16": c32["reference_zero_shot_32_16"],
        "runs": {
            "coarse32_refine16": {
                "checkpoint": c32["checkpoint"], "overall": c32["overall"],
                "buckets": c32["buckets"],
            },
        },
    }
    for label in VARIANTS:
        rows = load(label)
        overall = score(rows)
        if overall["pixels"] != expected_pixels:
            raise ValueError(f"Valid-pixel mismatch for {label}")
        result["runs"][label] = {
            "checkpoint": str(checkpoint_for(label)), "overall": overall,
            "buckets": {bucket: score(rows, bucket) for bucket in BUCKETS},
        }
    path = OUT / "no_mono_igev_three_way_ft2k_sceneflow_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    c32_shell_pid = int(os.environ["C32_SHELL_PID"]) if os.environ.get("C32_SHELL_PID") else None
    if c32_shell_pid is not None:
        os.kill(c32_shell_pid, 0)
    write_status("training", gpus={label: value["train_gpu"] for label, value in VARIANTS.items()})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train, label) for label in VARIANTS]
            for future in as_completed(futures):
                label = future.result()
                write_status("training", completed_variant=label)
    finally:
        # The 32/16 shell is paused while its training child finishes. Resume
        # its three-GPU validation only after the other two training jobs stop.
        if c32_shell_pid is not None:
            os.kill(c32_shell_pid, signal.SIGCONT)
    write_status("waiting_for_32_16_validation")
    deadline = time.monotonic() + 2 * 60 * 60
    while not C32_SUMMARY.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {C32_SUMMARY}")
        time.sleep(30)
    write_status("evaluating", gpus=[0, 2, 3])
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(evaluate_gpu, gpu) for gpu in (0, 2, 3)]
        for future in as_completed(futures):
            gpu = future.result()
            write_status("evaluating", completed_gpu=gpu)
    summary = summarize()
    write_status("complete", summary=str(summary))
    print(summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("error", error=str(error))
        raise
