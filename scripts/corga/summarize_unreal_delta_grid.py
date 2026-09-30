"""Compare paired UnrealStereo4K disparity-limit evaluations."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "diagnostics/unreal_delta_ablation/scene00008_grid_config.json"
OUTPUT = ROOT / "diagnostics/unreal_delta_ablation/scene00008_grid_summary.json"
BUCKETS = ("0-192", "192-384", "384-512", "512-768")


def load_rows(path, scene):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = {row["left"]: row for row in rows if f"/{scene}/" in row["left"]}
    if len(rows) != 200:
        raise ValueError(f"Expected 200 {scene} records in {path}, got {len(rows)}")
    return rows


def metrics(rows, bucket=None):
    values = [row if bucket is None else row["buckets"][bucket] for row in rows]
    count_key = "valid_pixels" if bucket is None else "pixels"
    pixels = sum(value[count_key] for value in values)
    return {
        "pixels": pixels,
        "epe": sum(value["error_sum"] for value in values) / pixels,
        "bad3_percent": 100 * sum(value["bad_sum"] for value in values) / pixels,
    }


def main():
    config = json.loads(CONFIG.read_text())
    cases = config["comparison_cases"]
    baseline = load_rows(ROOT / cases[0]["result"], "00008")
    summary = {"dataset": config["dataset"], "images": len(baseline), "cases": []}
    for case in cases:
        rows = load_rows(ROOT / case["result"], "00008")
        if rows.keys() != baseline.keys():
            raise ValueError(f"Image mismatch in {case['result']}")
        for key in baseline:
            if rows[key]["valid_pixels"] != baseline[key]["valid_pixels"]:
                raise ValueError(f"Valid-pixel mismatch for {key} in {case['result']}")
            if any(rows[key]["buckets"][b]["pixels"] != baseline[key]["buckets"][b]["pixels"]
                   for b in BUCKETS):
                raise ValueError(f"Bucket-pixel mismatch for {key} in {case['result']}")
        ordered = sorted(rows)
        selected = [rows[key] for key in ordered]
        high = [key for key in ordered if (
            baseline[key]["buckets"]["384-512"]["pixels"]
            + baseline[key]["buckets"]["512-768"]["pixels"]
        ) / baseline[key]["valid_pixels"] >= 0.25]
        summary["cases"].append({
            "coarse_limit": case["coarse"],
            "refine_limit": case["refine"],
            "result": case["result"],
            "overall": metrics(selected),
            "buckets": {bucket: metrics(selected, bucket) for bucket in BUCKETS},
            "images_better_than_8_8": sum(
                rows[key]["error_sum"] < baseline[key]["error_sum"] for key in ordered
            ),
            "high_disparity_images": {
                "count": len(high),
                "improved": sum(rows[key]["error_sum"] < baseline[key]["error_sum"]
                                for key in high),
                "overall": metrics([rows[key] for key in high]),
            },
        })
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(OUTPUT)
    for case in summary["cases"]:
        print(f"{case['coarse_limit']}/{case['refine_limit']}: "
              f"EPE {case['overall']['epe']:.3f}, "
              f"Bad3 {case['overall']['bad3_percent']:.3f}%, "
              f"better images {case['images_better_than_8_8']}/200")


if __name__ == "__main__":
    main()
