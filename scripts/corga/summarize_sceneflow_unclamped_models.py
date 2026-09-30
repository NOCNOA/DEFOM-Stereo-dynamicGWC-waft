"""Compare full FlyingThings3D TEST evaluations with matched valid pixels."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics/large_delta_eval"
BUCKETS = ("0-192", "192-384", "384-512", "512-768")
RUNS = {
    "twostage_200k_coarse32_refine16": [
        OUT / "sceneflow_full_coarse32_refine16_gpu2.jsonl",
        OUT / "sceneflow_full_coarse32_refine16_gpu3.jsonl",
    ],
    "twostage_200k_no_update_limit": [
        OUT / "sceneflow_full_no_delta_clamp_200k_gpu2.jsonl",
        OUT / "sceneflow_full_no_delta_clamp_200k_gpu3.jsonl",
    ],
    "fast3_70k_no_update_limit": [
        OUT / "sceneflow_full_no_delta_clamp_fast3_70k_gpu0.jsonl",
    ],
    "fast3_70k_cap8": [
        OUT / f"sceneflow_full_fast3_70k_cap8_gpu{gpu}.jsonl"
        for gpu in (0, 2, 3)
    ],
}


def load(paths):
    result = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["left"] in result:
                raise ValueError(f"Duplicate image {row['left']}")
            result[row["left"]] = row
    if len(result) != 4370:
        raise ValueError(f"Expected 4370 SceneFlow images, got {len(result)}")
    return result


def score(rows, bucket=None):
    values = [r if bucket is None else r["buckets"][bucket]
              for r in rows.values()]
    count_key = "valid_pixels" if bucket is None else "pixels"
    pixels = sum(r[count_key] for r in values)
    result = {
        "pixels": pixels,
        "pixel_epe": sum(r["error_sum"] for r in values) / pixels,
        "bad3_percent": 100 * sum(r["bad_sum"] for r in values) / pixels,
    }
    if bucket is None:
        result["image_mean_epe"] = sum(
            r["error_sum"] / r["valid_pixels"] for r in values
        ) / len(values)
    return result


def main():
    runs = {name: load(paths) for name, paths in RUNS.items()}
    reference = next(iter(runs.values()))
    for name, rows in runs.items():
        if rows.keys() != reference.keys():
            raise ValueError(f"Image mismatch: {name}")
        for key in reference:
            if rows[key]["valid_pixels"] != reference[key]["valid_pixels"]:
                raise ValueError(f"Valid pixel mismatch: {name}, {key}")
            if any(rows[key]["buckets"][bucket]["pixels"] !=
                   reference[key]["buckets"][bucket]["pixels"] for bucket in BUCKETS):
                raise ValueError(f"Bucket pixel mismatch: {name}, {key}")
    summary = {
        name: {"overall": score(rows),
               "buckets": {bucket: score(rows, bucket) for bucket in BUCKETS}}
        for name, rows in runs.items()
    }
    for candidate, reference_name in (
        ("twostage_200k_no_update_limit", "twostage_200k_coarse32_refine16"),
        ("fast3_70k_no_update_limit", "fast3_70k_cap8"),
    ):
        summary[candidate]["images_better_than_reference"] = sum(
            runs[candidate][key]["error_sum"] / runs[candidate][key]["valid_pixels"]
            < runs[reference_name][key]["error_sum"] / runs[reference_name][key]["valid_pixels"]
            for key in reference
        )
    path = OUT / "sceneflow_unclamped_models_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(path)
    for name, values in summary.items():
        overall = values["overall"]
        print(name, f"EPE {overall['image_mean_epe']:.6f}",
              f"Bad3 {overall['bad3_percent']:.4f}%")


if __name__ == "__main__":
    main()
