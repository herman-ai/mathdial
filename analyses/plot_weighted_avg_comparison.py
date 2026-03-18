#!/usr/bin/env python3
"""
Plot weighted-average score distributions for up to 4 JSONL judge-score files
on a single comparison figure, saved as a PDF.

Usage:
    python plot_weighted_avg_comparison.py \
        --files path1.jsonl path2.jsonl path3.jsonl path4.jsonl \
        --labels "Model A" "Model B" "Model C" "Model D" \
        --output comparison.pdf

    --labels is optional; file stems are used as fallback labels.
"""

import argparse
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "socratic_guidance":     1.5,
    "mathematical_accuracy": 1.0,
    "relevance":             1.0,
    "conciseness":           1.0,
    "overall_quality":       2.0,
}

_total_weight = sum(WEIGHTS.values())
WEIGHTS = {k: v / _total_weight for k, v in WEIGHTS.items()}

# Distinct, colorblind-friendly palette with consistent saturation
PALETTE = ["#2166AC", "#D6604D", "#4DAC26", "#7B2D8B", "#E08A00", "#00838F"]
PALETTE = ["#D6604D", "#2166AC","#7B2D8B", "#4DAC26"]
GREY    = "#888888"
DARK    = "#222222"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    import json
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path} line {lineno}: {exc}")
    return records


def weighted_average(scores: dict) -> float:
    total, w_used = 0.0, 0.0
    for key, weight in WEIGHTS.items():
        value = scores.get(key)
        if isinstance(value, (int, float)):
            total  += weight * value
            w_used += weight
    return total / w_used if w_used else float("nan")


def extract_wa_values(records: list[dict]) -> list[float]:
    values = []
    for rec in records:
        wa = weighted_average(rec.get("judge_scores", {}))
        if not math.isnan(wa):
            values.append(wa)
    return values


def desc_stats(values: list) -> dict:
    a = np.array(values, dtype=float)
    return {
        "n":      len(a),
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "p10":    float(np.percentile(a, 10)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "p90":    float(np.percentile(a, 90)),
    }


def kde(values, x_grid, bw=0.15):
    """Simple Gaussian KDE — no scipy dependency."""
    vals = np.array(values)
    result = np.zeros_like(x_grid, dtype=float)
    for v in vals:
        result += np.exp(-0.5 * ((x_grid - v) / bw) ** 2)
    result /= len(vals) * bw * math.sqrt(2 * math.pi)
    return result


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color=DARK)
    ax.set_xlabel(xlabel, fontsize=14, color=DARK)
    ax.set_ylabel(ylabel, fontsize=14, color=DARK)
    ax.tick_params(colors=DARK, labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_comparison(all_values: list[list[float]], labels: list[str], pdf_path: str):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#eeeeee",
        "axes.grid":        True,
        "grid.linewidth":   0.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    all_flat = [v for vals in all_values for v in vals]
    xg = np.linspace(min(all_flat) - 0.3, max(all_flat) + 0.3, 500)

    # All solid lines — differentiated by color alone for a clean look
    for vals, label, colour in zip(all_values, labels, PALETTE):
        k = kde(vals, xg)
        ax.plot(xg, k, color=colour, linewidth=2.5, label=label)
        ax.fill_between(xg, k, alpha=0.08, color=colour)

    style_ax(ax,
             title="Weighted Average Score Distribution — Multi-File Comparison",
             xlabel="Weighted Average Score",
             ylabel="Density (KDE)")

    ax.legend(fontsize=14, frameon=True, framealpha=0.9,
              edgecolor=GREY, loc="upper left")
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        meta = pdf.infodict()
        meta["Title"]   = "Weighted Average Score Comparison"
        meta["Subject"] = ", ".join(labels)

    plt.close(fig)
    print(f"Saved → {pdf_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare weighted-average distributions from up to 6 JSONL judge-score files."
    )
    parser.add_argument(
        "--files", nargs="+", required=True, metavar="PATH",
        help="1–6 JSONL file paths to compare."
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL",
        help="Display labels for each file (must match --files count). "
             "Defaults to file stems."
    )
    parser.add_argument(
        "--output", default="weighted_avg_comparison.pdf", metavar="PDF",
        help="Output PDF path (default: weighted_avg_comparison.pdf)."
    )
    args = parser.parse_args()

    if len(args.files) > 6:
        parser.error("At most 6 files are supported.")

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        parser.error("--labels count must match --files count.")

    all_values = []
    for path, label in zip(args.files, labels):
        print(f"Loading [{label}]: {path}")
        records = load_jsonl(path)
        values  = extract_wa_values(records)
        print(f"  → {len(records)} records, {len(values)} valid weighted averages")
        if not values:
            parser.error(f"No valid scores found in {path}")
        all_values.append(values)

    plot_comparison(all_values, labels, args.output)


if __name__ == "__main__":
    main()