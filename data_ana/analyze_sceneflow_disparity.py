#!/usr/bin/env python3
"""Count SceneFlow finalpass images by their maximum ground-truth disparity."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/home/share/yijiayi/sceneflow")
SUBSETS = ("FlyingThings3D", "Monkaa", "Driving")


def read_pfm_max(path: Path) -> float:
    """Read only what is needed to return the largest finite PFM value."""
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
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"PFM has no finite disparities: {path}")
    return float(finite.max())


def identify_subset(image_path: Path, root: Path) -> str:
    return image_path.relative_to(root).parts[0]


def image_to_disparity(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[parts.index("frames_finalpass")] = "disparity"
    return Path(*parts).with_suffix(".pfm")


def discover_images(root: Path, view: str) -> list[Path]:
    images: list[Path] = []
    for subset in SUBSETS:
        images.extend((root / subset / "frames_finalpass").glob(f"**/{view}/*.png"))
    return sorted(images)


def bucket_label(value: float, thresholds: list[float]) -> str:
    lower = 0.0
    for upper in thresholds:
        if value < upper:
            return f"[{lower:g},{upper:g})"
        lower = upper
    return f"[{thresholds[-1]:g},inf)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--view", choices=("left", "right", "both"), default="left")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[192, 384, 512, 768])
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--csv", type=Path, help="optionally save one row per image")
    parser.add_argument("--json", type=Path, help="optionally save the summary as JSON")
    args = parser.parse_args()

    thresholds = sorted(set(map(float, args.thresholds)))
    if not thresholds or thresholds[0] <= 0:
        parser.error("thresholds must be positive")

    views = ("left", "right") if args.view == "both" else (args.view,)
    images = sorted(path for view in views for path in discover_images(args.root, view))
    if not images:
        parser.error(f"no finalpass images found below {args.root}")

    disparity_paths = [image_to_disparity(path) for path in images]
    missing = [str(path) for path in disparity_paths if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        parser.error(f"{len(missing)} images lack disparity files; first entries:\n{preview}")

    with mp.Pool(args.workers) as pool:
        maxima = pool.map(read_pfm_max, disparity_paths, chunksize=16)

    rows = [
        {
            "subset": identify_subset(image, args.root),
            "image": str(image),
            "disparity": str(disparity),
            "max_disparity": maximum,
            "bucket": bucket_label(maximum, thresholds),
        }
        for image, disparity, maximum in zip(images, disparity_paths, maxima)
    ]

    by_subset = Counter(row["subset"] for row in rows)
    cumulative = {
        str(int(limit) if limit.is_integer() else limit): sum(
            row["max_disparity"] < limit for row in rows
        )
        for limit in thresholds
    }
    bucket_counts = Counter(row["bucket"] for row in rows)
    ordered_buckets = [f"[{0 if i == 0 else thresholds[i - 1]:g},{limit:g})" for i, limit in enumerate(thresholds)]
    ordered_buckets.append(f"[{thresholds[-1]:g},inf)")

    summary = {
        "root": str(args.root),
        "view": args.view,
        "comparison": "strictly less than threshold",
        "total_images": len(rows),
        "images_by_subset": dict(sorted(by_subset.items())),
        "cumulative_counts": cumulative,
        "bucket_counts": {label: bucket_counts[label] for label in ordered_buckets},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
