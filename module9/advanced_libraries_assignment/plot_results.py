#!/usr/bin/env python3
"""Turn results.txt (the output of ./run.sh all) into three figures:

  outcomes_chart.png - range of final balances (bad case, median, good
                       case) for each stock/bond mix, default plan.
  starters_chart.png - chance of reaching $1M for each mix: early
                       starter vs default plan vs late starter.
  speed_chart.png    - GPU vs CPU time for all mixes vs number of
                       simulated futures.

Usage: uv run --with matplotlib python plot_results.py
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.txt"
OUTCOMES_PNG = HERE / "outcomes_chart.png"
STARTERS_PNG = HERE / "starters_chart.png"
SPEED_PNG = HERE / "speed_chart.png"

# Categorical slots 1-3 (blue, orange, aqua) in fixed order, plus text and
# grid inks. Aqua is below 3:1 contrast on the surface, so every series
# also gets its own marker or a legend entry and direct labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

GOAL = 1_000_000
DEFAULT_RUN = ""
STARTER_RUNS = [("monthly=250 years=40", "Early starter: $250/month, 40 years"),
                ("", "Default: $500/month, 30 years"),
                ("monthly=1000 years=20", "Late starter: $1,000/month, 20 years")]
SPEED_RUNS = [("futures=10000", 10_000), ("futures=100000", 100_000),
              ("", 1_000_000), ("futures=2000000", 2_000_000)]

SECTION_RE = re.compile(r"^===== \./savings_sim ?(.*?) =====$")
MIX_RE = re.compile(r"^\s+(\d+)/(\d+)\s+([\d.]+)%" + r"\s+\$([\d,]+)" * 4)
GPU_RE = re.compile(r"^\s+GPU\s+all mixes \(Thrust\)\s+([\d.]+) ms")
CPU_RE = re.compile(r"^\s+CPU\s+all mixes\s+([\d.]+) ms")


def read_sections(path):
    """Split results.txt into {command-line arguments: [lines]}."""
    sections, current = {}, None
    for line in path.read_text().splitlines():
        match = SECTION_RE.match(line)
        if match:
            current = match.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def mixes(lines):
    """Rows of (label, chance %, bad, median, good) from one run."""
    rows = []
    for m in map(MIX_RE.match, lines):
        if m:
            dollars = [int(v.replace(",", "")) for v in m.groups()[3:6]]
            rows.append((f"{m.group(1)}/{m.group(2)}", float(m.group(3)),
                         *dollars))
    return rows


def first_value(regex, lines):
    return next(float(m.group(1)) for m in map(regex.match, lines) if m)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_outcomes(lines):
    rows = mixes(lines)
    xs = range(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    ax.axhline(GOAL, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(len(rows) - 0.5, GOAL, "Goal $1M", color=INK, fontsize=9,
            ha="right", va="bottom")
    for x, (_, _, bad, median, good) in zip(xs, rows):
        ax.plot([x, x], [bad, good], color=SERIES[0], linewidth=6,
                solid_capstyle="round", alpha=0.35)
        ax.plot(x, median, marker="o", markersize=9, color=SERIES[0],
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(x + 0.12, median, f"${median / 1e6:.2f}M", color=INK,
                fontsize=8, va="center",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_xticks(list(xs), [f"{r[0]}\n{r[1]:.0f}% reach goal"
                             for r in rows])
    ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1e6:.1f}M")
    ax.set_xlabel("Stocks / bonds", color=INK)
    ax.set_ylabel("Balance after 30 years", color=INK)
    ax.set_title("$500 a month for 30 years: bar = bad to good case "
                 "(1 in 10 each side), dot = median",
                 color=INK, loc="left", fontweight="bold", fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTCOMES_PNG, dpi=150)
    plt.close(fig)


def plot_starters(sections):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    width = 0.26
    for i, (args, label) in enumerate(STARTER_RUNS):
        rows = mixes(sections[args])
        xs = [x + (i - 1) * (width + 0.02) for x in range(len(rows))]
        bars = ax.bar(xs, [r[1] for r in rows], width, color=SERIES[i],
                      label=label)
        ax.bar_label(bars, labels=[f"{r[1]:.0f}%" for r in rows],
                     color=INK, fontsize=7, padding=2)
    labels = [r[0] for r in mixes(sections[DEFAULT_RUN])]
    ax.set_xticks(range(len(labels)), labels)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Stocks / bonds", color=INK)
    ax.set_ylabel("Chance of reaching $1M", color=INK)
    ax.set_title("Starting early beats saving more later",
                 color=INK, loc="left", fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(STARTERS_PNG, dpi=150)
    plt.close(fig)


def plot_speed(sections):
    futures = [n for _, n in SPEED_RUNS]
    gpu = [first_value(GPU_RE, sections[a]) for a, _ in SPEED_RUNS]
    cpu = [first_value(CPU_RE, sections[a]) for a, _ in SPEED_RUNS]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for i, (name, times) in enumerate([("GPU (Thrust)", gpu),
                                       ("CPU", cpu)]):
        ax.plot(futures, times, color=SERIES[i], marker=MARKERS[i],
                markersize=8, linewidth=2, label=name)
    for n, g, c in zip(futures, gpu, cpu):
        ax.text(n, c * 1.25, f"{c / g:.0f}x faster", color=INK,
                fontsize=9, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(futures[0] / 2, futures[-1] * 2)
    ax.set_xticks(futures, ["10k", "100k", "1M", "2M"])
    ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Simulated futures", color=INK)
    ax.set_ylabel("Time for all 6 mixes (ms)", color=INK)
    ax.set_title("The GPU's lead grows with the number of futures",
                 color=INK, loc="left", fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(SPEED_PNG, dpi=150)
    plt.close(fig)


def main():
    if not RESULTS.exists():
        sys.exit(f"{RESULTS} not found: run ./run.sh all > results.txt")
    sections = read_sections(RESULTS)
    plot_outcomes(sections[DEFAULT_RUN])
    plot_starters(sections)
    plot_speed(sections)
    for png in (OUTCOMES_PNG, STARTERS_PNG, SPEED_PNG):
        print(f"wrote {png.name}")


if __name__ == "__main__":
    main()
