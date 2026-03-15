#!/usr/bin/env python3
"""
Compute score distributions and statistics for all judge scores in a JSONL file,
then produce a set of matplotlib plots saved to a single PDF.

Plots produced:
  1.  Per-dimension bar charts  — frequency of each integer score (1–5)
  2.  Per-dimension KDE / histogram overlay
  3.  Weighted-average histogram with mean/median lines
  4.  Correlation heatmap across all dimensions
  5.  Box-and-whisker comparison across all dimensions
  6.  Cumulative distribution functions (CDF) per dimension
  7.  Grade-band pie chart (weighted average buckets)
  8.  Stats summary table
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ── Config ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    "socratic_guidance":    1.5,
    "mathematical_accuracy": 1,
    "relevance":            1,
    "conciseness":          1,
    "overall_quality":      2,
}

SCORE_DIMS = list(WEIGHTS.keys())
DIM_LABELS = [d.replace("_", "\n") for d in SCORE_DIMS]   # nicer axis labels

# Normalise weights
_total_weight = sum(WEIGHTS.values())
WEIGHTS = {k: v / _total_weight for k, v in WEIGHTS.items()}

# Colour palette
PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
GREY      = "#888888"
DARK      = "#222222"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] line {lineno}: {exc}")
    return records


def weighted_average(scores: dict) -> float:
    total, w_used = 0.0, 0.0
    for key, weight in WEIGHTS.items():
        value = scores.get(key)
        if isinstance(value, (int, float)):
            total  += weight * value
            w_used += weight
    return total / w_used if w_used else float("nan")


# ── Statistics helper ─────────────────────────────────────────────────────────

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


# ── Plot helpers ──────────────────────────────────────────────────────────────

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=DARK)
    ax.set_xlabel(xlabel, fontsize=9, color=DARK)
    ax.set_ylabel(ylabel, fontsize=9, color=DARK)
    ax.tick_params(colors=DARK, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def kde(values, x_grid, bw=0.3):
    """Simple Gaussian KDE (no scipy dependency)."""
    vals = np.array(values)
    result = np.zeros_like(x_grid, dtype=float)
    for v in vals:
        result += np.exp(-0.5 * ((x_grid - v) / bw) ** 2)
    result /= len(vals) * bw * math.sqrt(2 * math.pi)
    return result


# ── Individual plot generators ────────────────────────────────────────────────

def plot_freq_bars(dim_counts, dim_values, pdf):
    """Page 1 – 5 bar charts, one per dimension."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("Score Frequency per Dimension  (scores 1–5)",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)

    for ax, dim, colour in zip(axes, SCORE_DIMS, PALETTE):
        counter = dim_counts[dim]
        total   = sum(counter.values())
        xs      = list(range(1, 6))
        ys      = [counter.get(v, 0) for v in xs]
        pcts    = [100 * y / total if total else 0 for y in ys]

        bars = ax.bar(xs, ys, color=colour, alpha=0.85, edgecolor="white", linewidth=0.6)
        # Annotate bars with %
        for bar, p in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ys) * 0.02,
                    f"{p:.0f}%", ha="center", va="bottom", fontsize=7, color=DARK)

        st = desc_stats(dim_values[dim])
        style_ax(ax, title=dim.replace("_", " ").title(),
                 xlabel="Score", ylabel="Count")
        ax.set_xticks(xs)
        ax.axvline(st["mean"], color=GREY, linestyle="--", linewidth=1,
                   label=f"mean={st['mean']:.2f}")
        ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_kde_histograms(dim_values, pdf):
    """Page 2 – histogram + KDE overlay per dimension."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("Score Distribution with KDE  (per dimension)",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)

    x_grid = np.linspace(0.5, 5.5, 300)
    for ax, dim, colour in zip(axes, SCORE_DIMS, PALETTE):
        vals = dim_values[dim]
        ax.hist(vals, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                density=True, color=colour, alpha=0.4, edgecolor="white")
        k = kde(vals, x_grid)
        ax.plot(x_grid, k, color=colour, linewidth=2)
        st = desc_stats(vals)
        ax.axvline(st["mean"],   color="black",  linestyle="--", linewidth=1,
                   label=f"μ={st['mean']:.2f}")
        ax.axvline(st["median"], color=GREY, linestyle=":",  linewidth=1,
                   label=f"med={st['median']:.2f}")
        style_ax(ax, title=dim.replace("_", " ").title(),
                 xlabel="Score", ylabel="Density")
        ax.set_xlim(0.5, 5.5)
        ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_avg(wa_values, pdf):
    """Page 3 – weighted average histogram."""
    fig, ax = plt.subplots(figsize=(9, 5))
    st = desc_stats(wa_values)

    n, bins, patches = ax.hist(wa_values, bins=20, color=PALETTE[0],
                               alpha=0.75, edgecolor="white", linewidth=0.6)
    # KDE overlay on secondary y
    ax2 = ax.twinx()
    xg  = np.linspace(min(wa_values) - 0.2, max(wa_values) + 0.2, 400)
    ax2.plot(xg, kde(wa_values, xg, bw=0.15), color=PALETTE[0], linewidth=2.5)
    ax2.set_ylabel("Density", fontsize=9, color=DARK)
    ax2.tick_params(colors=DARK, labelsize=8)
    ax2.spines[["top"]].set_visible(False)

    ax.axvline(st["mean"],   color="red",   linestyle="--", linewidth=1.5,
               label=f"mean={st['mean']:.3f}")
    ax.axvline(st["median"], color="orange", linestyle=":",  linewidth=1.5,
               label=f"median={st['median']:.3f}")
    ax.axvspan(st["p25"], st["p75"], alpha=0.12, color="blue",
               label=f"IQR [{st['p25']:.2f}–{st['p75']:.2f}]")

    style_ax(ax, title="Weighted Average Score Distribution",
             xlabel="Weighted Average", ylabel="Count")
    ax.legend(fontsize=9, frameon=False)

    # Stats box
    stats_text = (f"n={st['n']}   mean={st['mean']:.3f}   median={st['median']:.3f}\n"
                  f"std={st['std']:.3f}   min={st['min']:.2f}   max={st['max']:.2f}\n"
                  f"P10={st['p10']:.3f}   P90={st['p90']:.3f}")
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GREY, alpha=0.8))

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(dim_values, pdf):
    """Page 4 – Pearson correlation heatmap."""
    n_dims = len(SCORE_DIMS)
    matrix = np.ones((n_dims, n_dims))
    for i, da in enumerate(SCORE_DIMS):
        for j, db in enumerate(SCORE_DIMS):
            a = np.array(dim_values[da])
            b = np.array(dim_values[db])
            mn = min(len(a), len(b))
            if mn < 2:
                matrix[i, j] = float("nan")
            else:
                matrix[i, j] = float(np.corrcoef(a[:mn], b[:mn])[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    short = [d.replace("_", "\n") for d in SCORE_DIMS]
    ax.set_xticks(range(n_dims)); ax.set_xticklabels(short, fontsize=8)
    ax.set_yticks(range(n_dims)); ax.set_yticklabels(short, fontsize=8)

    for i in range(n_dims):
        for j in range(n_dims):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(matrix[i,j]) < 0.7 else "white")

    style_ax(ax, title="Cross-Dimension Pearson Correlation Matrix")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(dim_values, wa_values, pdf):
    """Page 5 – side-by-side box plots for all dims + weighted avg."""
    all_labels = [d.replace("_", "\n") for d in SCORE_DIMS] + ["weighted\navg"]
    all_data   = [dim_values[d] for d in SCORE_DIMS] + [wa_values]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(all_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=3,
                                   alpha=0.4, linestyle="none"))

    colours = PALETTE + ["#2ca02c"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)

    ax.set_xticks(range(1, len(all_labels) + 1))
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.set_ylim(0.5, 5.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    style_ax(ax, title="Score Distribution — Box & Whisker (with outliers)",
             xlabel="Dimension", ylabel="Score")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_cdfs(dim_values, pdf):
    """Page 6 – empirical CDF per dimension on one axes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for dim, colour in zip(SCORE_DIMS, PALETTE):
        sv = np.sort(dim_values[dim])
        yv = np.arange(1, len(sv) + 1) / len(sv)
        ax.step(sv, yv, where="post", color=colour, linewidth=1.8,
                label=dim.replace("_", " "))

    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1, label="50th pct")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    style_ax(ax, title="Empirical Cumulative Distribution Functions",
             xlabel="Score", ylabel="Cumulative %")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_grade_bands(wa_values, pdf):
    """Page 7 – pie chart of grade bands."""
    bands = {
        "Poor < 2": sum(1 for x in wa_values if x < 2),
        "Below avg\n2–3": sum(1 for x in wa_values if 2 <= x < 3),
        "Average\n3–4":   sum(1 for x in wa_values if 3 <= x < 4),
        "Good\n4–5":      sum(1 for x in wa_values if 4 <= x < 5),
        "Perfect = 5":    sum(1 for x in wa_values if x == 5),
    }
    labels  = list(bands.keys())
    sizes   = list(bands.values())
    colours = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#17becf"]
    explode = [0.04] * len(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colours, explode=explode,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum(sizes)/100))})",
        startangle=140, pctdistance=0.75,
        textprops=dict(fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Grade-Band Distribution  (Weighted Average)",
                 fontsize=12, fontweight="bold", color=DARK, pad=14)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_stats_table(dim_values, wa_values, pdf):
    """Page 8 – summary statistics table."""
    all_dims  = SCORE_DIMS + ["weighted_avg"]
    all_vals  = [dim_values[d] for d in SCORE_DIMS] + [wa_values]
    col_labels = ["n", "mean", "median", "std", "min", "max", "P25", "P75"]
    row_labels = [d.replace("_", " ") for d in all_dims]

    table_data = []
    for vals in all_vals:
        st = desc_stats(vals)
        table_data.append([
            f"{st['n']}",
            f"{st['mean']:.3f}",
            f"{st['median']:.3f}",
            f"{st['std']:.3f}",
            f"{st['min']:.2f}",
            f"{st['max']:.2f}",
            f"{st['p25']:.3f}",
            f"{st['p75']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)

    # Header row styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4C72B0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Row label styling + alternating rows
    for i in range(len(row_labels)):
        tbl[i + 1, -1].set_facecolor("#dde3f0" if i % 2 == 0 else "white")
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor("#dde3f0" if i % 2 == 0 else "white")

    ax.set_title("Descriptive Statistics Summary",
                 fontsize=13, fontweight="bold", color=DARK, pad=16)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(judge_path: str, pdf_path: str) -> None:
    print(f"Loading: {judge_path}")
    records = load_jsonl(judge_path)
    print(f"  → {len(records)} entries")

    dim_values: dict[str, list] = defaultdict(list)
    dim_counts:  dict[str, Counter] = defaultdict(Counter)
    wa_values: list = []

    for rec in records:
        js = rec.get("judge_scores", {})
        wa = weighted_average(js)
        if not math.isnan(wa):
            wa_values.append(wa)
        for dim in SCORE_DIMS:
            v = js.get(dim)
            if isinstance(v, (int, float)):
                dim_values[dim].append(float(v))
                dim_counts[dim][int(v)] += 1

    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#eeeeee",
        "axes.grid":        True,
        "grid.linewidth":   0.5,
    })

    print(f"Generating plots → {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        plot_freq_bars(dim_counts, dim_values, pdf)
        plot_kde_histograms(dim_values, pdf)
        plot_weighted_avg(wa_values, pdf)
        plot_correlation_heatmap(dim_values, pdf)
        plot_boxplots(dim_values, wa_values, pdf)
        plot_cdfs(dim_values, pdf)
        plot_grade_bands(wa_values, pdf)
        plot_stats_table(dim_values, wa_values, pdf)

        # PDF metadata
        meta = pdf.infodict()
        meta["Title"]   = "Judge Score Distributions"
        meta["Subject"] = judge_path

    print(f"Done — {pdf_path}  (8 pages)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores_file_path, out_path = sys.argv[1], sys.argv[2]
    # scores_file_path = '../output/dpo/qwen_judge_scores_no_real_teacher_ckpt1000.jsonl'
    # out_path         = '../analyses/plots/qwen_dpo_human_preferences_batching_score_distributions.pdf'
    if scores_file_path is None or out_path is None:
        raise ValueError('test')
    main(scores_file_path, out_path)