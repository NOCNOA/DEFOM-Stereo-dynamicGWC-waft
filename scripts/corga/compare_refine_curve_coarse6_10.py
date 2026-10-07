"""Compare matched SceneFlow per-iteration EPE curves for coarse 6 and 10."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path("diagnostics/large_delta_eval")
six_dir = root / "refine_curve_sceneflow_no_clamp"
ten_dir = root / "refine_curve_sceneflow_no_clamp_coarse10"
six = json.loads((six_dir / "summary.json").read_text())
ten = json.loads((ten_dir / "summary.json").read_text())
labels = ("0-192", "192-384", "384-512", "512-768", "0-768")
if six["images"] != ten["images"] or six["checkpoint"] != ten["checkpoint"]:
    raise ValueError("Image count or checkpoint mismatch")
for gpu in (2, 3):
    if (six_dir / f"manifest_gpu{gpu}.txt").read_text() != (ten_dir / f"manifest_gpu{gpu}.txt").read_text():
        raise ValueError(f"Sample manifest differs on GPU {gpu}")
for label in labels:
    if six["curves"][label]["pixels"] != ten["curves"][label]["pixels"]:
        raise ValueError(f"Bucket pixels differ for {label}")

csv_path = ten_dir / "comparison_coarse6_vs_10.csv"
with csv_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["refine_iteration", *[f"{label}_{coarse}"
                                         for label in labels for coarse in ("coarse6", "coarse10", "delta10_minus6")]])
    for iteration in range(33):
        values = []
        for label in labels:
            a = six["curves"][label]["epe"][iteration]
            b = ten["curves"][label]["epe"][iteration]
            values.extend((a, b, b-a))
        writer.writerow([iteration, *values])

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for ax, label in zip(axes.flat, labels):
    ax.plot(range(33), six["curves"][label]["epe"], label="coarse 6")
    ax.plot(range(33), ten["curves"][label]["epe"], label="coarse 10")
    ax.set_title(f"GT {label} px; {six['curves'][label]['pixels']:,} pixels")
    ax.set_xlabel("Refine GRU iteration")
    ax.set_ylabel("Pixel-weighted EPE (px)")
    ax.set_xticks(range(0, 33, 8))
    ax.grid(alpha=.25)
    ax.legend()
axes[1, 2].axis("off")
fig.suptitle("CorGA no clamp | matched SceneFlow TEST 500 images")
output = ten_dir / "comparison_coarse6_vs_10.png"
fig.savefig(output, dpi=170)
print(csv_path)
print(output)
