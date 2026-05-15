#!/usr/bin/env python3
"""
plot_training.py — Plot training rewards from training log file.

Usage:
    python plot_training.py
    python plot_training.py --input last_training_prints.txt --output training_plot.png
"""

import re
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_log(filepath: str):
    """Parse training log and return lists of iterations and rewards."""
    iters, archer0, archer1, avg = [], [], [], []

    pattern = re.compile(
        r"iter\s+(\d+)\s*\|\s*archer_0:\s*([\d.]+)\s*\|\s*archer_1:\s*([\d.]+)\s*\|\s*avg:\s*([\d.]+)"
    )

    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                archer0.append(float(m.group(2)))
                archer1.append(float(m.group(3)))
                avg.append(float(m.group(4)))

    return iters, archer0, archer1, avg


def plot(iters, archer0, archer1, avg, output_path: str):
    """Create and save the training plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iters, avg,     color="#2196F3", linewidth=2.0, label="Average",   zorder=3)
    ax.plot(iters, archer0, color="#4CAF50", linewidth=1.5, label="Archer 0",  zorder=2, alpha=0.85)
    ax.plot(iters, archer1, color="#F44336", linewidth=1.5, label="Archer 1",  zorder=2, alpha=0.85)

    # Mark peak avg
    peak_idx = avg.index(max(avg))
    ax.scatter(iters[peak_idx], avg[peak_idx], color="#2196F3", s=80, zorder=5)
    ax.annotate(
        f"peak avg: {avg[peak_idx]:.1f} (iter {iters[peak_idx]})",
        xy=(iters[peak_idx], avg[peak_idx]),
        xytext=(iters[peak_idx] + 5, avg[peak_idx] - 3),
        fontsize=9,
        color="#2196F3",
        arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.2),
    )

    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Training Reward over Iterations — KAZ Vector Policy", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default=os.path.join(_DIR, "last_training_prints.txt"))
    parser.add_argument("--output", type=str, default=os.path.join(_DIR, "training_plot.png"))
    args = parser.parse_args()

    iters, archer0, archer1, avg = parse_log(args.input)
    print(f"Parsed {len(iters)} iterations from {args.input}")
    print(f"Peak avg: {max(avg):.3f} at iter {iters[avg.index(max(avg))]}")
    print(f"Peak archer_0: {max(archer0):.3f} at iter {iters[archer0.index(max(archer0))]}")
    print(f"Peak archer_1: {max(archer1):.3f} at iter {iters[archer1.index(max(archer1))]}")

    plot(iters, archer0, archer1, avg, args.output)


if __name__ == "__main__":
    main()