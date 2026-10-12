#!/usr/bin/env python3
"""Turn results.txt (the output of ./run.sh) and output/ into three figures:

  speedup_chart.png - base configuration: each variant's speedup over the
                      synchronous pageable baseline (single series).
  blocks_chart.png  - for each block shape (batch of 32): the two blur
                      kernels' time vs. the event pipeline's end-to-end
                      time, showing that overlap hides kernel differences.
  blur_example.png  - one photo before and after the blur.

Usage: uv run --with matplotlib python plot_results.py
"""

import re
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.txt"
SPEEDUP_PNG = HERE / "speedup_chart.png"
BLOCKS_PNG = HERE / "blocks_chart.png"
EXAMPLE_PNG = HERE / "blur_example.png"
EXAMPLE_PHOTO = "football"
BASE_BATCH = 32

VARIANTS = ["sync, pageable", "1 stream, pinned", "2 streams, pinned",
            "4 streams, pinned", "8 streams, pinned",
            "3-stream event pipeline"]
PIPELINE = VARIANTS[-1]

# Categorical slots 1-2 of the reference palette, in fixed order.
SERIES = ["#2a78d6", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

HEADER_RE = re.compile(r"=+ \./streams_assignment (\d+) (\d+) (\d+) =+")
ROW_RE = re.compile(r"^  (" + "|".join(re.escape(v) for v in VARIANTS)
                    + r")\s+.*?\s(\d+\.\d+)\s+(\d+\.\d+)x")
STAGE_RE = re.compile(r"kernel 1 \(horizontal blur\) (\d+\.\d+) \| "
                      r"kernel 2 \(vertical blur\) (\d+\.\d+)")


def parse(path):
    """Returns [{block: (x, y), batch, ms: {variant: ms},
    speedup: {variant: x}, kernel_ms}] in file order."""
    runs = []
    for line in path.read_text().splitlines():
        header = HEADER_RE.match(line)
        if header:
            runs.append({"block": (int(header[1]), int(header[2])),
                         "batch": int(header[3]), "ms": {}, "speedup": {}})
            continue
        if not runs:
            continue
        row = ROW_RE.match(line)
        if row:
            runs[-1]["ms"][row[1]] = float(row[2])
            runs[-1]["speedup"][row[1]] = float(row[3])
        stage = STAGE_RE.search(line)
        if stage:
            runs[-1]["kernel_ms"] = float(stage[1]) + float(stage[2])
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
    names = list(reversed(VARIANTS))  # top-to-bottom in VARIANTS order
    values = [run["speedup"][v] for v in names]

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.barh(names, values, height=0.6, color=SERIES[0])
    for y, s in enumerate(values):
        ax.annotate(f"{s:.2f}x", (s, y), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK_SECONDARY)
    ax.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_xlabel("Speedup over sync, pageable (higher is faster)",
                  color=INK_SECONDARY)
    ax.tick_params(axis="y", colors=INK_PRIMARY)
    bx, by = run["block"]
    fig.suptitle(f"Blurring {run['batch']} photos: speedup by strategy "
                 f"(block {bx}x{by})", color=INK_PRIMARY, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(SPEEDUP_PNG, dpi=150, facecolor=SURFACE)


def blocks_chart(runs):
    shaped = [r for r in runs if r["batch"] == BASE_BATCH]
    labels = [f"{x}x{y}\n({x * y} threads)" for x, y in
              (r["block"] for r in shaped)]
    series = [("Both blur kernels (1 stream)",
               [r["kernel_ms"] for r in shaped]),
              ("Event pipeline, end to end",
               [r["ms"][PIPELINE] for r in shaped])]

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    style_axes(ax)
    width = 0.38
    for i, (name, values) in enumerate(series):
        xs = [g + (i - 0.5) * width for g in range(len(shaped))]
        ax.bar(xs, values, width, color=SERIES[i], edgecolor=SURFACE,
               linewidth=1.5, label=name)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(shaped)))
    ax.set_xticklabels(labels, color=INK_PRIMARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_ylabel("Time (ms, lower is faster)", color=INK_SECONDARY)
    ax.set_xlabel("Block shape", color=INK_SECONDARY)
    fig.suptitle(f"Block shape changes kernel time, but overlap hides it "
                 f"({BASE_BATCH} photos)", color=INK_PRIMARY, fontsize=12,
                 x=0.01, ha="left")
    legend = ax.legend(frameon=False, fontsize=9, ncol=2,
                       loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(BLOCKS_PNG, dpi=150, facecolor=SURFACE)


def blur_example():
    before = HERE / "images" / f"{EXAMPLE_PHOTO}.jpg"
    after = HERE / "output" / f"{EXAMPLE_PHOTO}_blurred.jpg"
    if not after.exists():
        print(f"Skipping {EXAMPLE_PNG.name}: run ./run.sh first")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), facecolor=SURFACE)
    for ax, path, title in ((axes[0], before, "Original"),
                            (axes[1], after, "Gaussian blur (25 taps)")):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(EXAMPLE_PNG, dpi=120, facecolor=SURFACE)


def main():
    if not RESULTS.exists():
        sys.exit("No results.txt - run: ./run.sh | tee results.txt")
    runs = parse(RESULTS)
    if not runs or any(len(r["ms"]) != len(VARIANTS) or "kernel_ms" not in r
                       for r in runs):
        sys.exit("results.txt is missing runs or timing rows")
    speedup_chart(runs[0])
    blocks_chart(runs)
    blur_example()
    print(f"Wrote {SPEEDUP_PNG.name}, {BLOCKS_PNG.name} and "
          f"{EXAMPLE_PNG.name}")


if __name__ == "__main__":
    main()
