"""Summarize the matched Unreal no-update-limit and full SceneFlow runs."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics/large_delta_eval"
BUCKETS = ("0-192", "192-384", "384-512", "512-768")


def read_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({record["left"] for record in records}) != len(records):
        raise ValueError(f"Duplicate image in {path}")
    return {record["left"]: record for record in records}


def scores(records, bucket=None):
    values = [row if bucket is None else row["buckets"][bucket]
              for row in records.values()]
    count_key = "valid_pixels" if bucket is None else "pixels"
    pixels = sum(row[count_key] for row in values)
    return {
        "pixels": pixels,
        "epe_pixel_weighted": sum(row["error_sum"] for row in values) / pixels,
        "bad3_percent": 100 * sum(row["bad_sum"] for row in values) / pixels,
    }


def main():
    baseline = {key: row for key, row in read_records(
        ROOT / "diagnostics/unreal_full_200k/results_gpu0.jsonl").items()
        if "/00008/" in key}
    unclamped = read_records(OUT / "unreal_scene00008_no_delta_clamp.jsonl")
    if len(baseline) != 200 or baseline.keys() != unclamped.keys():
        raise ValueError("Unreal images differ from the 200-image baseline")
    for key in baseline:
        if baseline[key]["valid_pixels"] != unclamped[key]["valid_pixels"]:
            raise ValueError(f"Unreal valid pixels differ for {key}")
        for bucket in BUCKETS:
            if baseline[key]["buckets"][bucket]["pixels"] != unclamped[key]["buckets"][bucket]["pixels"]:
                raise ValueError(f"Unreal bucket pixels differ for {key}: {bucket}")

    scene_parts = [read_records(OUT / f"sceneflow_full_coarse32_refine16_gpu{gpu}.jsonl")
                   for gpu in (2, 3)]
    if scene_parts[0].keys() & scene_parts[1].keys():
        raise ValueError("SceneFlow shards overlap")
    sceneflow = scene_parts[0] | scene_parts[1]
    manifests = [set((OUT / f"sceneflow_test_gpu{gpu}_manifest.txt").read_text().splitlines())
                 for gpu in (2, 3)]
    if len(sceneflow) != 4370 or sceneflow.keys() != manifests[0] | manifests[1]:
        raise ValueError("SceneFlow TEST split is incomplete")

    sf_scores = scores(sceneflow)
    # validate_things reports arithmetic mean of per-image EPE and pooled Bad3.
    sf_scores["epe_image_mean"] = sum(
        row["error_sum"] / row["valid_pixels"] for row in sceneflow.values()
    ) / len(sceneflow)
    result = {
        "unreal_scene00008": {
            "images": len(unclamped),
            "baseline_8_8": scores(baseline),
            "no_update_limit": scores(unclamped),
            "buckets": {
                bucket: {"baseline_8_8": scores(baseline, bucket),
                         "no_update_limit": scores(unclamped, bucket)}
                for bucket in BUCKETS
            },
            "images_improved": sum(unclamped[key]["error_sum"] < baseline[key]["error_sum"]
                                   for key in baseline),
        },
        "sceneflow_flyingthings3d_finalpass_test": {
            "images": len(sceneflow),
            "coarse_limit": 32,
            "refine_limit": 16,
            **sf_scores,
            "buckets": {bucket: scores(sceneflow, bucket) for bucket in BUCKETS},
        },
    }
    path = OUT / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
