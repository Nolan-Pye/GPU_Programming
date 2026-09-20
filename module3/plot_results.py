#!/usr/bin/env python3
"""Render results.csv (written by assignment.exe) as a grouped bar chart
comparing CPU vs. GPU timing, with and without branching, across the
thread-count / block-size configurations that were run."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"
OUT_PATH = HERE / "results_chart.png"

# Fixed categorical order (dataviz palette slots 1-4), light-mode hex.
COLOR_CPU_BASE = "#2a78d6"    # blue
COLOR_GPU_BASE = "#eb6834"    # orange
COLOR_CPU_BRANCH = "#1baf7a"  # aqua
COLOR_GPU_BRANCH = "#eda100"  # yellow

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    if not CSV_PATH.exists():
        sys.exit(f"No {CSV_PATH.name} found — run the assignment a few times first.")

    rows = load_rows(CSV_PATH)
    labels = [f"{r['total_threads']} threads\n{r['block_size']}/block" for r in rows]
    cpu_base = np.array([float(r["cpu_baseline_ms"]) for r in rows])
    gpu_base = np.array([float(r["gpu_baseline_ms"]) for r in rows])
    cpu_branch = np.array([float(r["cpu_branching_ms"]) for r in rows])
    gpu_branch = np.array([float(r["gpu_branching_ms"]) for r in rows])

    x = np.arange(len(rows))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    series = [
        ("CPU, no branch", cpu_base, COLOR_CPU_BASE, -1.5),
        ("GPU, no branch", gpu_base, COLOR_GPU_BASE, -0.5),
        ("CPU, branching", cpu_branch, COLOR_CPU_BRANCH, 0.5),
        ("GPU, branching", gpu_branch, COLOR_GPU_BRANCH, 1.5),
    ]

    for name, values, color, offset in series:
        bars = ax.bar(x + offset * width, values, width, label=name, color=color,
                       edgecolor=SURFACE, linewidth=2)
        for rect, v in zip(bars, values):
            ax.annotate(f"{v:.3g}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7.5, color=INK_SECONDARY)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, log scale)", color=INK_PRIMARY)
    ax.set_title("CPU vs. GPU: baseline vs. branching kernel/method", color=INK_PRIMARY, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(axis="y", colors=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    legend = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
