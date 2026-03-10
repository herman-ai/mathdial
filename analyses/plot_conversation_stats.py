"""
Plot conversation statistics from conversations_stats.json as grouped bar charts.

Usage:
  python analyses/plot_conversation_stats.py \
      --input_file analyses/data/conversations_stats.json \
      --output_dir analyses/plots/conversation_stats
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    # (json_key, title, y_label)
    ("total_turns",      "Avg Total Turns per Conversation",      "Turns"),
    ("teacher_turns",    "Avg Teacher Turns per Conversation",    "Turns"),
    ("teacher_word_len", "Avg Teacher Utterance Length (words)",  "Words"),
    ("teacher_char_len", "Avg Teacher Utterance Length (chars)",  "Characters"),
]


def plot_metric(stats, key, title, ylabel, output_dir):
    labels = [s["label"] for s in stats]
    values = [s[key]["mean"] or 0 for s in stats]

    n = len(labels)
    x = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="right", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"{key}_mean.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_overview(stats, output_dir):
    """Single figure with 4 subplots for a quick overview."""
    overview_metrics = [
        ("total_turns",      "Avg Total Turns"),
        ("teacher_turns",    "Avg Teacher Turns"),
        ("teacher_word_len", "Avg Teacher Utt (words)"),
        ("teacher_char_len", "Avg Teacher Utt (chars)"),
    ]

    labels = [s["label"] for s in stats]
    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Conversation Statistics Across Models", fontsize=18, fontweight="bold", y=1.01)

    for ax, (key, title) in zip(axes.flat, overview_metrics):
        values = [s[key]["mean"] or 0 for s in stats]
        bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(values),
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold"
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.18)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="data/conversation_stats.json")
    parser.add_argument("--output_dir", type=str,
                        default="plots/conversation_stats")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_file) as f:
        stats = json.load(f)

    print(f"Loaded {len(stats)} models from {args.input_file}")
    print(f"Saving plots to {args.output_dir}/\n")

    # Individual metric plots
    for key, title, ylabel in METRICS:
        plot_metric(stats, key, title, ylabel, args.output_dir)

    # Combined overview
    plot_combined_overview(stats, args.output_dir)

    print("\nDone.")
