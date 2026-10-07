"""Plot the saved SceneFlow refine-iteration EPE curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


directory = Path("diagnostics/large_delta_eval/refine_curve_sceneflow_no_clamp")
data = json.loads((directory / "summary.json").read_text())
curves = data["curves"]
iterations = range(data["iterations"] + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
for label in ("0-768", "0-192"):
    axes[0].plot(iterations, curves[label]["epe"], label=label)
axes[1].plot(iterations, curves["192-384"]["epe"], label="192-384")
for label in ("384-512", "512-768"):
    axes[2].plot(iterations, curves[label]["epe"], label=label)
for ax in axes:
    ax.set_xlabel("Refine GRU iteration (0 = coarse-fused initialization)")
    ax.set_ylabel("Pixel-weighted EPE (px)")
    ax.set_xticks(range(0, 33, 4))
    ax.grid(alpha=.25)
    ax.legend(title="GT disparity (px)")
axes[0].set_title("Overall and 0–192 px")
axes[1].set_title("192–384 px")
axes[2].set_title("384–768 px")
fig.suptitle("CorGA two-stage 200k, no clamp | SceneFlow TEST, 500 images")
output = directory / "epe_by_iteration.png"
fig.savefig(output, dpi=180)
print(output)
