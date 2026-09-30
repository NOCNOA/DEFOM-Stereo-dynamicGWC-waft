"""Evaluate free coarse updates with a bounded 8px refinement on SceneFlow."""

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
STATUS = OUT / "coarse_unclamped_refine8_sceneflow_status.json"
SUMMARY = OUT / "coarse_unclamped_refine8_sceneflow_summary.json"
SCENEFLOW = Path("/home/share/yijiayi/sceneflow")
GPUS = (0, 2, 3)
LOCK = threading.Lock()
CASES = {
    "twostage_200k": {
        "architecture": "cor_ga_fast",
        "checkpoint": ROOT / "checkpoints/defomstereo_vitl_twostage_768/defomstereo_vitl_twostage_768_200000.pth",
        "arch_flags": ["--hypothesis_scales", "24", "48", "96"],
        "reference": [OUT / f"sceneflow_full_coarse32_refine16_gpu{gpu}.jsonl" for gpu in (2, 3)],
    },
    "no_mono_igev_200k": {
        "architecture": "cor_ga_no_mono_igev",
        "checkpoint": ROOT / "checkpoints/defomstereo_vitl_corga_no_mono_igev/defomstereo_vitl_corga_no_mono_igev_200000.pth",
        "arch_flags": ["--num_init_bins", "48", "--init_max_offset", "48", "--init_corr_scale", "10"],
        "reference": [OUT / f"no_mono_igev_200k_sceneflow_coarse32_refine16_gpu{gpu}.jsonl" for gpu in (0, 2, 3)],
    },
    "no_mono_igev_16_8_ft2k": {
        "architecture": "cor_ga_no_mono_igev",
        "checkpoint": ROOT / "checkpoints/defomstereo_vitl_corga_no_mono_igev_coarse16_refine8_ft2k/defomstereo_vitl_corga_no_mono_igev_coarse16_refine8_ft2k_  2000.pth",
        "arch_flags": ["--num_init_bins", "48", "--init_max_offset", "48", "--init_corr_scale", "10"],
        "reference": [OUT / f"no_mono_igev_coarse16_refine8_ft2k_sceneflow_gpu{gpu}.jsonl" for gpu in (0, 2, 3)],
    },
}


def write_status(state, completed, **fields):
    with LOCK:
        payload = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "gpus": list(GPUS), "completed_shards": sorted(completed), **fields}
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(STATUS)


def manifest_path(gpu):
    return OUT / f"coarse_unclamped_refine8_sceneflow_gpu{gpu}_manifest.txt"


def result_path(case, gpu):
    return OUT / f"{case}_coarse_unclamped_refine8_gpu{gpu}.jsonl"


def prepare_manifests():
    images = sorted((SCENEFLOW / "FlyingThings3D/frames_finalpass/TEST").glob("*/*/left/*.png"))
    if len(images) != 4370:
        raise ValueError(f"Expected 4370 SceneFlow images, found {len(images)}")
    for shard, gpu in enumerate(GPUS):
        manifest_path(gpu).write_text("\n".join(map(str, images[shard::len(GPUS)])) + "\n")


def evaluate_gpu(gpu, completed):
    manifest = manifest_path(gpu)
    expected = set(manifest.read_text().splitlines())
    for label, config in CASES.items():
        result = result_path(label, gpu)
        log = result.with_suffix(".log")
        command = [
            sys.executable, str(ROOT / "evaluate_stereo.py"),
            "--datasets", "mixed", "--mixed_names", "sceneflow",
            "--mixed_root", f"sceneflow={SCENEFLOW}",
            "--mixed_manifest", f"sceneflow={manifest}",
            "--mixed_max_samples", "4370", "--max_disp", "768",
            "--valid_iters", "32", "--scale_iters", "6",
            "--corr_levels", "3", "--corr_radius", "4",
            "--dinov2_encoder", "vitl", "--model_arch", config["architecture"],
            "--mixed_precision", "--amp_dtype", "bf16",
            "--max_delta_disp", "8", "--disable_coarse_delta_clamp",
            "--restore_ckpt", str(config["checkpoint"]), *config["arch_flags"],
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["MIXED_RESULT_JSONL"] = str(result)
        with log.open("a") as output:
            output.write("\nCOMMAND: " + " ".join(command) + "\n")
            output.flush()
            subprocess.run(command, cwd=ROOT, env=environment,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in result.read_text().splitlines() if line]
        if len(rows) != len(expected) or {row["left"] for row in rows} != expected:
            raise ValueError(f"Incomplete {label} shard on GPU {gpu}")
        completed.add(f"{label}_gpu{gpu}")
        write_status("running", completed)
    return gpu


def load(paths):
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["left"] in rows:
                raise ValueError(f"Duplicate SceneFlow image: {row['left']}")
            rows[row["left"]] = row
    if len(rows) != 4370:
        raise ValueError(f"Expected 4370 SceneFlow images, found {len(rows)}")
    return rows


def summarize():
    summary = {"configuration": {"coarse_update_limit": None,
                                  "refine_update_limit_quarter_pixels": 8,
                                  "final_disparity_clamp_full_pixels": [0, 768]},
               "runs": {}}
    for label, config in CASES.items():
        candidate = load([result_path(label, gpu) for gpu in GPUS])
        reference = load(config["reference"])
        if candidate.keys() != reference.keys():
            raise ValueError(f"Image mismatch for {label}")
        for path, row in candidate.items():
            old = reference[path]
            if row["valid_pixels"] != old["valid_pixels"]:
                raise ValueError(f"Valid pixel mismatch for {label}, {path}")
            for bucket in BUCKETS:
                if row["buckets"][bucket]["pixels"] != old["buckets"][bucket]["pixels"]:
                    raise ValueError(f"Bucket pixel mismatch for {label}, {path}, {bucket}")
        summary["runs"][label] = {
            "checkpoint": str(config["checkpoint"]),
            "overall": score(candidate),
            "buckets": {bucket: score(candidate, bucket) for bucket in BUCKETS},
            "reference_overall": score(reference),
            "reference_description": (
                "same checkpoint with 32/16 limits" if label != "no_mono_igev_16_8_ft2k"
                else "same checkpoint with 16/8 limits"),
            "images_better_than_reference": sum(
                row["error_sum"] / row["valid_pixels"]
                < reference[path]["error_sum"] / reference[path]["valid_pixels"]
                for path, row in candidate.items()
            ),
        }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    return SUMMARY


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for config in CASES.values():
        if not config["checkpoint"].is_file():
            raise FileNotFoundError(config["checkpoint"])
    prepare_manifests()
    completed = set()
    write_status("running", completed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(evaluate_gpu, gpu, completed) for gpu in GPUS]
        for future in as_completed(futures):
            future.result()
    summary = summarize()
    write_status("complete", completed, summary=str(summary))
    print(summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("error", set(), error=str(error))
        raise
