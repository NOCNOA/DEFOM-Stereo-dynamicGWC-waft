"""Plot recorded bin-candidate recall; no inference or GPU is required."""

import argparse
import json
import math
from pathlib import Path


METHODS = (
    ("fused", "Fused output (actual)", "#474747"),
    ("top1_oracle", "Top-1 peak / branch (oracle)", "#507AA5"),
    ("top4_oracle", "Top-4 peaks / branch (oracle)", "#3BA5AF"),
    ("all_bin_oracle", "All valid bins (oracle)", "#D39A3A"),
)
THRESHOLDS = (4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Recorded results.json from diagnose_bin_peaks.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output figure, e.g. figure.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = json.loads(args.results.read_text())
    rows = data.get("metrics", [])
    panels = (
        ("random_", "init_bad16_in_right", "all",
         "Random images: poor initialization",
         "Initial error >16 px; GT match inside right image"),
        ("large_disp_enriched_", "covered_in_right", "384-752",
         "Large-disparity enriched images",
         "GT 384-752 px; inside right image and bin range"),
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 220,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)
    width = 0.19
    for ax, (prefix, scope, band, title, subtitle) in zip(axes, panels):
        cohorts = sorted({row["cohort"] for row in rows
                          if row.get("cohort", "").startswith(prefix)
                          and row.get("scope") == scope and row.get("gt_band") == band})
        selected = {row["method"]: row for row in rows
                    if cohorts and row.get("cohort") == cohorts[0]
                    and row.get("scope") == scope and row.get("gt_band") == band}
        usable = [row for method, _, _ in METHODS if (row := selected.get(method))]
        counts = sorted({int(row["queries"]) for row in usable})
        count_label = (f"{counts[0]:,} queries" if len(counts) == 1 else
                       "Query counts differ across methods" if counts else "No matching queries")
        ax.set_title(title, loc="left", fontweight="bold", pad=34)
        ax.text(0, 1.045, subtitle + "\n" + count_label, transform=ax.transAxes,
                fontsize=9, color="#505050", va="bottom")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_ylim(0, 109)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(THRESHOLDS)), [f"≤{t} px" for t in THRESHOLDS])
        ax.set_xlabel("Absolute disparity error threshold")
        if not usable:
            ax.text(0.5, 0.5, "No matching cohort / scope in results", ha="center",
                    va="center", transform=ax.transAxes, color="#666666")
            continue
        missing = []
        for index, (method, label, color) in enumerate(METHODS):
            row = selected.get(method)
            if row is None:
                missing.append(label)
                continue
            values = [row.get(f"recall_{t}px_pct") for t in THRESHOLDS]
            values = [float(v) if v is not None and math.isfinite(float(v))
                      else float("nan") for v in values]
            positions = [x + (index - 1.5) * width for x in range(len(THRESHOLDS))]
            bars = ax.bar(positions, values, width=width * 0.92, color=color,
                          edgecolor="white", linewidth=0.4)
            ax.bar_label(bars, labels=[f"{v:.1f}" if math.isfinite(v) else ""
                                      for v in values], padding=3, fontsize=8)
        if missing:
            ax.text(0.01, 0.98, "Missing: " + "; ".join(missing), fontsize=8,
                    va="top", transform=ax.transAxes, color="#666666")
    axes[0].set_ylabel("Queries within threshold (%)")
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in METHODS],
               loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=2,
               frameon=False, fontsize=9)
    fig.text(0.5, 0.045,
             "Native 1/4 grid; all errors in original-image pixels. Missing candidates count as failures.\n"
             "Oracles use ground truth to select candidates; they are not deployable prediction accuracy.",
             ha="center", va="center", fontsize=8.5, color="#505050")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.27, top=0.81, wspace=0.14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
