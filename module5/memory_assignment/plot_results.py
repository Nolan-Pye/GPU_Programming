#!/usr/bin/env python3
"""Turn results.txt (the output of ./run.sh) into two charts:

  speedup_chart.png  - base configuration: each variant's speedup over the
                       global-memory version (single series).
  timings_chart.png  - GPU kernel time for every variant across all five
                       thread-count / block-size configurations.

Usage: uv run --with matplotlib python plot_results.py
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.txt"
SPEEDUP_PNG = HERE / "speedup_chart.png"
TIMINGS_PNG = HERE / "timings_chart.png"

VARIANTS = ["cpu", "host (mapped)", "global", "constant", "shared",
            "register", "combined"]
GPU_VARIANTS = VARIANTS[1:]

# Categorical slots 1-5 of the reference palette, in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

HEADER_RE = re.compile(r"=+ \./memory_assignment (\d+) (\d+) =+")
ROW_RE = re.compile(r"^  (" + "|".join(re.escape(v) for v in VARIANTS)
                    + r")\s+.*?\s(\d+\.\d+)\s+(\d+\.\d+)x")


def parse(path):
    """Returns [(threads, block, {variant: ms})] in file order."""
    runs = []
    for line in path.read_text().splitlines():
        header = HEADER_RE.match(line)
        if header:
            runs.append((int(header[1]), int(header[2]), {}))
            continue
        row = ROW_RE.match(line)
        if row and runs:
            runs[-1][2][row[1]] = float(row[2])
    return runs


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_axisbelow(True)


def speedup_chart(run):
    threads, block, ms = run
    names = list(reversed(VARIANTS))  # top-to-bottom in VARIANTS order
    speedups = [ms["global"] / ms[v] for v in names]

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.barh(names, speedups, height=0.6, color=SERIES[0])
    for y, s in enumerate(speedups):
        ax.annotate(f"{s:.2f}x", (s, y), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK_SECONDARY)
    ax.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_xlabel("Speedup over global memory (higher is faster)",
                  color=INK_SECONDARY)
    ax.tick_params(axis="y", colors=INK_PRIMARY)
    ax.set_title(f"Letter histogram speedup vs. global memory "
                 f"({threads:,} threads, block size {block})",
                 color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(SPEEDUP_PNG, dpi=150, facecolor=SURFACE)


def timings_chart(runs):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    style_axes(ax)
    width = 0.8 / len(runs)
    for i, (threads, block, ms) in enumerate(runs):
        xs = [g + (i - (len(runs) - 1) / 2) * width
              for g in range(len(GPU_VARIANTS))]
        ax.bar(xs, [ms[v] for v in GPU_VARIANTS], width,
               color=SERIES[i], edgecolor=SURFACE, linewidth=1.5,
               label=f"{threads:,} threads / block {block}")
    ax.set_xticks(range(len(GPU_VARIANTS)))
    ax.set_xticklabels(GPU_VARIANTS, color=INK_PRIMARY)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_ylabel("Kernel time (ms, lower is faster)", color=INK_SECONDARY)
    ax.set_title("GPU kernel time by memory type and launch configuration",
                 color=INK_PRIMARY, fontsize=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, ncol=3,
                       loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(TIMINGS_PNG, dpi=150, facecolor=SURFACE)


def main():
    if not RESULTS.exists():
        sys.exit("No results.txt - run: ./run.sh | tee results.txt")
    runs = parse(RESULTS)
    if not runs or any(len(ms) != len(VARIANTS) for _, _, ms in runs):
        sys.exit("results.txt is missing runs or timing rows")
    speedup_chart(runs[0])
    timings_chart(runs)
    print(f"Wrote {SPEEDUP_PNG.name} and {TIMINGS_PNG.name}")


if __name__ == "__main__":
    main()
