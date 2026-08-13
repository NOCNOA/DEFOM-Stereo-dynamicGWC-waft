#!/usr/bin/env python3
"""Count SceneFlow finalpass ground-truth pixels in disparity buckets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/home/share/yijiayi/sceneflow")
SUBSETS = ("FlyingThings3D", "Monkaa", "Driving")


def image_to_disparity(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[parts.index("frames_finalpass")] = "disparity"
    return Path(*parts).with_suffix(".pfm")


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.readline().rstrip()
        if header not in (b"Pf", b"PF"):
            raise ValueError(f"invalid PFM header in {path}: {header!r}")
        channels = 3 if header == b"PF" else 1
        dimensions = re.fullmatch(rb"\s*(\d+)\s+(\d+)\s*", stream.readline())
        if dimensions is None:
            raise ValueError(f"invalid PFM dimensions in {path}")
        width, height = map(int, dimensions.groups())
        scale = float(stream.readline().strip())
        dtype = np.dtype("<f4" if scale < 0 else ">f4")
        values = np.fromfile(stream, dtype=dtype, count=width * height * channels)
    expected = width * height * channels
    if values.size != expected:
        raise ValueError(f"truncated PFM {path}: expected {expected}, got {values.size}")
    return values


def classify_path(path: Path) -> tuple[str, str]:
    parts = path.parts
    subset = next(name for name in SUBSETS if name in parts)
    split = "test" if subset == "FlyingThings3D" and "TEST" in parts else "train"
    return split, subset


def count_file(task: tuple[Path, tuple[float, ...]]) -> tuple[str, str, list[int], int, int]:
    path, thresholds = task
    values = read_pfm(path)
    finite = np.isfinite(values)
    negative = int(np.count_nonzero(finite & (values < 0)))
    nonfinite = int(values.size - np.count_nonzero(finite))
    valid = values[finite & (values >= 0)]
    # searchsorted(..., side="right") implements [lower, upper) buckets.
    bucket_ids = np.searchsorted(thresholds, valid, side="right")
    counts = np.bincount(bucket_ids, minlength=len(thresholds) + 1).tolist()
    split, subset = classify_path(path)
    return split, subset, counts, negative, nonfinite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--view", choices=("left", "right", "both"), default="left")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[192, 384, 512, 768])
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--json", type=Path, help="optionally save the summary as JSON")
    args = parser.parse_args()

    thresholds = tuple(sorted(set(map(float, args.thresholds))))
    if not thresholds or thresholds[0] <= 0:
        parser.error("thresholds must be positive")
    views = ("left", "right") if args.view == "both" else (args.view,)
    images = sorted(
        image
        for subset in SUBSETS
        for view in views
        for image in (args.root / subset / "frames_finalpass").glob(f"**/{view}/*.png")
    )
    if not images:
        parser.error(f"no finalpass images found below {args.root}")
    disparities = [image_to_disparity(image) for image in images]
    missing = [path for path in disparities if not path.is_file()]
    if missing:
        parser.error(f"{len(missing)} images lack disparity files; first: {missing[0]}")

    bucket_labels = [
        f"[{0 if index == 0 else thresholds[index - 1]:g},{upper:g})"
        for index, upper in enumerate(thresholds)
    ] + [f"[{thresholds[-1]:g},inf)"]
    totals = {
        split: {
            "images": 0,
            "subsets": Counter(),
            "buckets": np.zeros(len(bucket_labels), dtype=np.int64),
            "negative_pixels": 0,
            "nonfinite_pixels": 0,
        }
        for split in ("train", "test")
    }

    tasks = ((path, thresholds) for path in disparities)
    with mp.Pool(args.workers) as pool:
        for split, subset, counts, negative, nonfinite in pool.imap_unordered(
            count_file, tasks, chunksize=16
        ):
            group = totals[split]
            group["images"] += 1
            group["subsets"][subset] += 1
            group["buckets"] += counts
            group["negative_pixels"] += negative
            group["nonfinite_pixels"] += nonfinite

    summary = {
        "root": str(args.root),
        "view": args.view,
        "bucket_semantics": "finite, non-negative pixels; lower-inclusive and upper-exclusive",
        "splits": {},
    }
    for split, group in totals.items():
        bucket_counts = {label: int(group["buckets"][i]) for i, label in enumerate(bucket_labels)}
        valid_pixels = int(group["buckets"].sum())
        summary["splits"][split] = {
            "images": group["images"],
            "images_by_subset": dict(sorted(group["subsets"].items())),
            "valid_pixels": valid_pixels,
            "bucket_counts": bucket_counts,
            "bucket_percentages": {
                label: 100.0 * count / valid_pixels for label, count in bucket_counts.items()
            },
            "negative_pixels": group["negative_pixels"],
            "nonfinite_pixels": group["nonfinite_pixels"],
        }

    output = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
