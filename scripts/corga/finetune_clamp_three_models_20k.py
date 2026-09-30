"""Train three batch-2 clamp variants for 20k steps, then evaluate SceneFlow."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.corga.eval_no_mono_igev_clamp import BUCKETS, score


OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "clamp_three_models_20k_status.json"
SUMMARY = OUT / "clamp_three_models_20k_sceneflow_summary.json"
LOCK = threading.Lock()
CASES = {
    "no_mono_igev_16_8": {
        "gpu": 0,
        "name": "defomstereo_vitl_corga_no_mono_igev_c16r8_plus20k",
        "source": ROOT / "checkpoints/defomstereo_vitl_corga_no_mono_igev_coarse16_refine8_ft2k/defomstereo_vitl_corga_no_mono_igev_coarse16_refine8_ft2k_  2000.pth",
        "arch": "cor_ga_no_mono_igev",
        "flags": ["--max_coarse_delta_disp", "16", "--max_delta_disp", "8"],
        "arch_flags": ["--num_init_bins", "48", "--init_max_offset", "48",
                       "--init_corr_scale", "10", "--init_right_compression", "avg",
                       "--bin_loss_mode", "soft_ce", "--bin_disp_smooth_l1_weight", "0.2"],
    },
    "twostage_no_clamp": {
        "gpu": 2,
        "name": "defomstereo_vitl_twostage_768_no_clamp_ft20k",
        "source": ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth",
        "arch": "cor_ga",
        "flags": ["--disable_delta_clamp"],
        "arch_flags": ["--hypothesis_scales", "24", "48", "96"],
    },
    "twostage_16_8": {
        "gpu": 3,
        "name": "defomstereo_vitl_twostage_768_c16r8_ft20k",
        "source": ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth",
        "arch": "cor_ga",
        "flags": ["--max_coarse_delta_disp", "16", "--max_delta_disp", "8"],
        "arch_flags": ["--hypothesis_scales", "24", "48", "96"],
    },
}


def write_status(state, stages, **fields):
    with LOCK:
        payload = {
            "state": state,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "batch_size": 2,
            "additional_train_steps": 20000,
            "stages": dict(stages),
            **fields,
        }
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(STATUS)


def final_checkpoint(config):
    name = config["name"]
    return ROOT / "checkpoints" / name / f"{name}_{20000:6d}.pth"


def train(label, config, stages):
    checkpoint = final_checkpoint(config)
    if checkpoint.is_file():
        stages[label] = "trained"
        write_status("running", stages)
        return

    directory = checkpoint.parent
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "checkpoint_latest.pth"
    resume = latest if latest.is_file() else config["source"]
    resume_flags = ["--resume_ckpt", str(resume), "--strict_resume"]
    if resume == config["source"]:
        resume_flags.append("--no_resume_optimizer")
    command = [
        sys.executable, str(ROOT / "train_stereo.py"),
        "--model_arch", config["arch"], "--dinov2_encoder", "vitl",
        "--launcher", "none", "--gpu_ids", "0", "--name", config["name"],
        "--batch_size", "2", "--num_workers", "4",
        "--train_datasets", "sceneflow", "--train_folds", "1",
        "--num_steps", "20000", "--save_ckpt_freq", "5000",
        "--save_latest_ckpt_freq", "1000", "--n_downsample", "2",
        "--train_iters", "18", "--scale_iters", "6",
        "--coarse_loss_weight", "0.2", "--loss_gamma", "0.9",
        "--smooth_l1_beta", "1.0", "--train_max_disp", "768",
        "--eval_max_disp", "768", "--corr_levels", "3", "--corr_radius", "4",
        "--context_norm", "batch", "--n_gru_layers", "3",
        "--hidden_dims", "128", "--image_size", "256", "768",
        "--wdecay", "0.00001", "--seed", "1234565",
        "--val_freq", "10000", "--valid_iters", "32",
        "--amp_dtype", "bf16", "--mixed_precision", "--lr", "0.00001",
        *config["arch_flags"], *config["flags"], *resume_flags,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    stages[label] = "training"
    write_status("running", stages)
    with (directory / "train.log").open("a") as output:
        output.write("\nCOMMAND: " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training finished without {checkpoint}")
    stages[label] = "trained"
    write_status("running", stages)


def evaluate(label, config, stages):
    stages[label] = "evaluating"
    write_status("running", stages)
    result = OUT / f"{label}_plus20k_sceneflow.jsonl"
    log = OUT / f"{label}_plus20k_sceneflow.log"
    command = [
        sys.executable, str(ROOT / "evaluate_stereo.py"),
        "--datasets", "mixed", "--mixed_names", "sceneflow",
        "--mixed_root", "sceneflow=/home/share/yijiayi/sceneflow",
        "--mixed_max_samples", "4370", "--max_disp", "768",
        "--valid_iters", "32", "--scale_iters", "6",
        "--corr_levels", "3", "--corr_radius", "4",
        "--dinov2_encoder", "vitl", "--model_arch", config["arch"],
        "--mixed_precision", "--amp_dtype", "bf16",
        "--restore_ckpt", str(final_checkpoint(config)),
        *[arg for arg in config["arch_flags"] if arg not in
          ("--init_right_compression", "avg", "--bin_loss_mode", "soft_ce",
           "--bin_disp_smooth_l1_weight", "0.2")],
        *config["flags"],
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    environment["MIXED_RESULT_JSONL"] = str(result)
    with log.open("a") as output:
        output.write("\nCOMMAND: " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    rows = [json.loads(line) for line in result.read_text().splitlines() if line]
    if len(rows) != 4370 or len({row["left"] for row in rows}) != 4370:
        raise ValueError(f"{label}: expected 4370 unique SceneFlow results, got {len(rows)}")
    stages[label] = "complete"
    write_status("running", stages)


def run_case(label, config, stages):
    if not config["source"].is_file():
        raise FileNotFoundError(config["source"])
    train(label, config, stages)
    evaluate(label, config, stages)
    return label


def summarize():
    runs = {}
    reference = None
    for label, config in CASES.items():
        path = OUT / f"{label}_plus20k_sceneflow.jsonl"
        rows = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate {label} result: {row['left']}")
            rows[row["left"]] = row
        if len(rows) != 4370:
            raise ValueError(f"{label}: expected 4370 images, got {len(rows)}")
        if reference is None:
            reference = rows
        else:
            if rows.keys() != reference.keys():
                raise ValueError(f"Image mismatch for {label}")
            for name, row in rows.items():
                base = reference[name]
                if row["valid_pixels"] != base["valid_pixels"]:
                    raise ValueError(f"Valid-pixel mismatch for {label}, {name}")
                for bucket in BUCKETS:
                    if row["buckets"][bucket]["pixels"] != base["buckets"][bucket]["pixels"]:
                        raise ValueError(f"Bucket mismatch for {label}, {name}, {bucket}")
        runs[label] = {
            "gpu": config["gpu"], "architecture": config["arch"],
            "source_checkpoint": str(config["source"]),
            "checkpoint": str(final_checkpoint(config)),
            "images": len(rows), "overall": score(rows),
            "buckets": {bucket: score(rows, bucket) for bucket in BUCKETS},
        }
    summary = {
        "training": {"batch_size": 2, "additional_steps": 20000,
                     "learning_rate": 1e-5, "amp_dtype": "bf16"},
        "runs": runs,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    return SUMMARY


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {label: "pending" for label in CASES}
    write_status("running", stages)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, label, config, stages): label
                   for label, config in CASES.items()}
        for future in as_completed(futures):
            future.result()
    summary = summarize()
    write_status("complete", stages, summary=str(summary))
    print(summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("error", {}, error=str(error))
        raise
