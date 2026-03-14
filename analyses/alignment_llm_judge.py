import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DIMENSIONS = [
    "socratic_guidance",
    "mathematical_accuracy",
    "relevance",
    "conciseness",
    "overall_quality",
]

# Default weights copied from the preference-pair generation pipeline:
# avg_score = (
#     overall_quality * 2 +
#     socratic_guidance * 1.5 +
#     mathematical_accuracy +
#     relevance +
#     conciseness
# ) / 6.5
DEFAULT_WEIGHTS = {
    "socratic_guidance": 1.5,
    "mathematical_accuracy": 1.0,
    "relevance": 1.0,
    "conciseness": 1.0,
    "overall_quality": 2.0,
}
DEFAULT_WEIGHT_DENOM = 6.5


def parse_weights_arg(weights_str: str | None) -> Tuple[Dict[str, float], float]:
    if not weights_str:
        return DEFAULT_WEIGHTS.copy(), DEFAULT_WEIGHT_DENOM

    weights = json.loads(weights_str)
    missing = [d for d in DIMENSIONS if d not in weights]
    if missing:
        raise ValueError(f"Missing weights for dimensions: {missing}")
    weights = {k: float(v) for k, v in weights.items()}
    denom = float(sum(weights.values()))
    if denom <= 0:
        raise ValueError("Sum of weights must be > 0")
    return weights, denom


def load_scores_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

            if "qid" not in obj:
                raise ValueError(f"Missing 'qid' on line {line_num} of {path}")
            if "judge_scores" not in obj or not isinstance(obj["judge_scores"], dict):
                raise ValueError(f"Missing or invalid 'judge_scores' on line {line_num} of {path}")

            row = {"qid": obj["qid"]}
            for dim in DIMENSIONS:
                if dim not in obj["judge_scores"]:
                    raise ValueError(f"Missing dimension '{dim}' on line {line_num} of {path}")
                row[dim] = obj["judge_scores"][dim]
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    df = pd.DataFrame(rows)

    # Preserve duplicate qids by assigning a rollout index within each qid.
    # This lets us align trajectory-level scores rather than assuming one row per qid.
    df["rollout_idx"] = df.groupby("qid").cumcount()

    return df


def add_weighted_score(df: pd.DataFrame, prefix: str, weights: Dict[str, float], denom: float) -> pd.DataFrame:
    cols = [f"{dim}_{prefix}" for dim in DIMENSIONS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for weighted score: {missing}")

    # Force numeric conversion so bad strings / nulls become NaN
    for dim in DIMENSIONS:
        col = f"{dim}_{prefix}"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    weighted_sum = sum(df[f"{dim}_{prefix}"] * weights[dim] for dim in DIMENSIONS)
    df[f"weighted_score_{prefix}"] = weighted_sum / denom

    # Use nullable integer type so NaNs do not crash conversion
    rounded = df[f"weighted_score_{prefix}"].round().clip(lower=1, upper=5)
    df[f"weighted_score_round_{prefix}"] = rounded.astype("Int64")

    return df


def compute_dimension_metrics(df: pd.DataFrame, dimension: str) -> Dict[str, Any]:
    human_col = f"{dimension}_human"
    llm_col = f"{dimension}_llm"

    sub = df[[human_col, llm_col]].dropna().copy()
    h = sub[human_col]
    l = sub[llm_col]

    agreement = float((h == l).mean())
    abs_diff_mean = float((h - l).abs().mean())
    corr = float(h.corr(l)) if h.nunique() > 1 and l.nunique() > 1 else None

    labels = sorted(set(h.tolist()) | set(l.tolist()))
    kappa = float(cohen_kappa_score(h, l, labels=labels))
    cm = confusion_matrix(h, l, labels=labels)

    return {
        "dimension": dimension,
        "n": int(len(sub)),
        "percent_agreement": agreement,
        "cohen_kappa": kappa,
        "mean_absolute_difference": abs_diff_mean,
        "pearson_correlation": corr,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "human_mean": float(h.mean()),
        "llm_mean": float(l.mean()),
        "human_std": float(h.std(ddof=1)) if len(h) > 1 else 0.0,
        "llm_std": float(l.std(ddof=1)) if len(l) > 1 else 0.0,
    }


def compute_weighted_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    sub = df[
        [
            "weighted_score_human",
            "weighted_score_llm",
            "weighted_score_round_human",
            "weighted_score_round_llm",
        ]
    ].dropna().copy()

    h = sub["weighted_score_human"]
    l = sub["weighted_score_llm"]
    h_round = sub["weighted_score_round_human"].astype(int)
    l_round = sub["weighted_score_round_llm"].astype(int)

    labels = sorted(set(h_round.tolist()) | set(l_round.tolist()))
    cm = confusion_matrix(h_round, l_round, labels=labels)

    return {
        "dimension": "weighted_composite",
        "n": int(len(sub)),
        "percent_agreement_rounded": float((h_round == l_round).mean()),
        "cohen_kappa_rounded": float(cohen_kappa_score(h_round, l_round, labels=labels)),
        "mean_absolute_difference": float((h - l).abs().mean()),
        "pearson_correlation": float(h.corr(l)) if h.nunique() > 1 and l.nunique() > 1 else None,
        "human_mean": float(h.mean()),
        "llm_mean": float(l.mean()),
        "human_std": float(h.std(ddof=1)) if len(h) > 1 else 0.0,
        "llm_std": float(l.std(ddof=1)) if len(l) > 1 else 0.0,
        "labels_rounded": labels,
        "confusion_matrix_rounded": cm.tolist(),
    }


def overall_summary(df: pd.DataFrame) -> Dict[str, Any]:
    human_vals = []
    llm_vals = []

    for dim in DIMENSIONS:
        sub = df[[f"{dim}_human", f"{dim}_llm"]].dropna()
        human_vals.extend(sub[f"{dim}_human"].tolist())
        llm_vals.extend(sub[f"{dim}_llm"].tolist())

    human_series = pd.Series(human_vals)
    llm_series = pd.Series(llm_vals)
    labels = sorted(set(human_vals) | set(llm_vals))

    return {
        "n_total_labels": int(len(human_vals)),
        "macro_percent_agreement": float((human_series == llm_series).mean()),
        "macro_cohen_kappa": float(cohen_kappa_score(human_series, llm_series, labels=labels)),
        "macro_mean_absolute_difference": float((human_series - llm_series).abs().mean()),
        "macro_pearson_correlation": float(human_series.corr(llm_series)) if human_series.nunique() > 1 and llm_series.nunique() > 1 else None,
    }


def print_results(
    metrics: List[Dict[str, Any]],
    weighted_metrics: Dict[str, Any],
    summary: Dict[str, Any],
    weights: Dict[str, float],
    denom: float,
) -> None:
    print("\n=== Weights used for weighted composite ===\n")
    print(json.dumps({"weights": weights, "denominator": denom}, indent=2))

    out_df = pd.DataFrame([
        {
            "dimension": m["dimension"],
            "n": m["n"],
            "agreement": round(m["percent_agreement"], 3),
            "kappa": round(m["cohen_kappa"], 3),
            "mean_abs_diff": round(m["mean_absolute_difference"], 3),
            "pearson_r": None if m["pearson_correlation"] is None else round(m["pearson_correlation"], 3),
            "human_mean": round(m["human_mean"], 3),
            "llm_mean": round(m["llm_mean"], 3),
        }
        for m in metrics
    ])

    print("\n=== Human vs LLM Judge Alignment by Dimension ===\n")
    print(out_df.to_string(index=False))

    print("\n=== Weighted Composite Alignment ===\n")
    print(json.dumps({
        "n": weighted_metrics["n"],
        "percent_agreement_rounded": round(weighted_metrics["percent_agreement_rounded"], 3),
        "cohen_kappa_rounded": round(weighted_metrics["cohen_kappa_rounded"], 3),
        "mean_absolute_difference": round(weighted_metrics["mean_absolute_difference"], 3),
        "pearson_correlation": None if weighted_metrics["pearson_correlation"] is None else round(weighted_metrics["pearson_correlation"], 3),
        "human_mean": round(weighted_metrics["human_mean"], 3),
        "llm_mean": round(weighted_metrics["llm_mean"], 3),
    }, indent=2))

    print("\n=== Macro Summary Across All Raw Dimensions ===\n")
    print(json.dumps({
        "n_total_labels": summary["n_total_labels"],
        "macro_percent_agreement": round(summary["macro_percent_agreement"], 3),
        "macro_cohen_kappa": round(summary["macro_cohen_kappa"], 3),
        "macro_mean_absolute_difference": round(summary["macro_mean_absolute_difference"], 3),
        "macro_pearson_correlation": None if summary["macro_pearson_correlation"] is None else round(summary["macro_pearson_correlation"], 3),
    }, indent=2))

    print("\n=== Confusion Matrices (rows = human, cols = llm) ===\n")
    for m in metrics:
        print(f"{m['dimension']}: labels={m['labels']}")
        cm_df = pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"])
        print(cm_df.to_string())
        print()

    print(f"weighted_composite (rounded): labels={weighted_metrics['labels_rounded']}")
    cm_df = pd.DataFrame(
        weighted_metrics["confusion_matrix_rounded"],
        index=weighted_metrics["labels_rounded"],
        columns=weighted_metrics["labels_rounded"],
    )
    print(cm_df.to_string())
    print()


def save_outputs(
    metrics: List[Dict[str, Any]],
    weighted_metrics: Dict[str, Any],
    summary: Dict[str, Any],
    output_prefix: str,
    weights: Dict[str, float],
    denom: float,
) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {
            "dimension": m["dimension"],
            "n": m["n"],
            "percent_agreement": m["percent_agreement"],
            "cohen_kappa": m["cohen_kappa"],
            "mean_absolute_difference": m["mean_absolute_difference"],
            "pearson_correlation": m["pearson_correlation"],
            "human_mean": m["human_mean"],
            "llm_mean": m["llm_mean"],
            "human_std": m["human_std"],
            "llm_std": m["llm_std"],
        }
        for m in metrics
    ] + [{
        "dimension": weighted_metrics["dimension"],
        "n": weighted_metrics["n"],
        "percent_agreement": weighted_metrics["percent_agreement_rounded"],
        "cohen_kappa": weighted_metrics["cohen_kappa_rounded"],
        "mean_absolute_difference": weighted_metrics["mean_absolute_difference"],
        "pearson_correlation": weighted_metrics["pearson_correlation"],
        "human_mean": weighted_metrics["human_mean"],
        "llm_mean": weighted_metrics["llm_mean"],
        "human_std": weighted_metrics["human_std"],
        "llm_std": weighted_metrics["llm_std"],
    }]).to_csv(f"{output_prefix}_metrics.csv", index=False)

    with open(f"{output_prefix}_full.json", "w", encoding="utf-8") as f:
        json.dump({
            "weights": weights,
            "denominator": denom,
            "summary": summary,
            "weighted_composite": weighted_metrics,
            "dimensions": metrics,
        }, f, indent=2)

def build_long_df(merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows = []
    for dim in DIMENSIONS:
        col = f"{dim}_{suffix}"
        sub = merged[["qid", "rollout_idx", col]].dropna().copy()
        sub = sub.rename(columns={col: "score"})
        sub["dimension"] = dim.replace("_", " ")
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def compute_weighted_series(merged: pd.DataFrame, suffix: str, weights: Dict[str, float], denom: float) -> pd.Series:
    vals = sum(pd.to_numeric(merged[f"{dim}_{suffix}"], errors="coerce") * weights[dim] for dim in DIMENSIONS) / denom
    return vals


def descriptive_stats_table(merged: pd.DataFrame, suffix: str, weights: Dict[str, float], denom: float) -> pd.DataFrame:
    rows = []
    for dim in DIMENSIONS:
        s = pd.to_numeric(merged[f"{dim}_{suffix}"], errors="coerce").dropna()
        rows.append({
            "dimension": dim.replace("_", " "),
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=1),
            "min": s.min(),
            "max": s.max(),
            "P25": s.quantile(0.25),
            "P75": s.quantile(0.75),
        })

    w = compute_weighted_series(merged, suffix, weights, denom).dropna()
    rows.append({
        "dimension": "weighted avg",
        "n": len(w),
        "mean": w.mean(),
        "median": w.median(),
        "std": w.std(ddof=1),
        "min": w.min(),
        "max": w.max(),
        "P25": w.quantile(0.25),
        "P75": w.quantile(0.75),
    })
    return pd.DataFrame(rows)


def plot_score_frequency_page(merged: pd.DataFrame, suffix: str, label: str, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(20, 5), constrained_layout=True)
    bins = np.arange(0.5, 6.5, 1)

    for ax, dim in zip(axes, DIMENSIONS):
        s = pd.to_numeric(merged[f"{dim}_{suffix}"], errors="coerce").dropna()
        counts = s.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        pct = counts / counts.sum() * 100 if counts.sum() > 0 else counts

        ax.bar(counts.index, counts.values)
        ax.set_title(dim.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, max(counts.max() * 1.2, 1))

        for x, y, p in zip(counts.index, counts.values, pct.values):
            ax.text(x, y + max(counts.max() * 0.03, 1), f"{p:.0f}%", ha="center", va="bottom", fontsize=9)

        ax.text(0.03, 0.95, f"mean={s.mean():.2f}", transform=ax.transAxes, va="top")

    fig.suptitle(f"Score Frequency per Dimension (scores 1–5) — {label}", fontsize=16)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_kde_page(merged: pd.DataFrame, suffix: str, label: str, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(20, 5), constrained_layout=True)

    for ax, dim in zip(axes, DIMENSIONS):
        s = pd.to_numeric(merged[f"{dim}_{suffix}"], errors="coerce").dropna()

        ax.hist(s, bins=np.arange(0.5, 6.5, 1), density=True, alpha=0.35)
        try:
            s.plot(kind="kde", ax=ax)
        except Exception:
            pass

        ax.axvline(s.mean(), linestyle="--")
        ax.axvline(s.median(), linestyle=":")
        ax.set_title(dim.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_xlim(1, 5)
        ax.text(0.03, 0.95, f"μ={s.mean():.2f}\nmed={s.median():.2f}", transform=ax.transAxes, va="top")

    fig.suptitle(f"Score Distribution with KDE (per dimension) — {label}", fontsize=16)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_distribution_page(merged: pd.DataFrame, suffix: str, label: str, weights: Dict[str, float], denom: float, pdf: PdfPages) -> None:
    w = compute_weighted_series(merged, suffix, weights, denom).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].hist(w, bins=20)
    axes[0].axvline(w.mean(), linestyle="--", label=f"mean={w.mean():.3f}")
    axes[0].axvline(w.median(), linestyle=":", label=f"median={w.median():.3f}")
    axes[0].set_title("Weighted Average Score Distribution")
    axes[0].set_xlabel("Weighted Average")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    try:
        w.plot(kind="kde", ax=axes[1])
    except Exception:
        pass
    axes[1].axvline(w.mean(), linestyle="--")
    axes[1].axvline(w.median(), linestyle=":")
    axes[1].set_title("Weighted Average KDE")
    axes[1].set_xlabel("Weighted Average")
    axes[1].set_ylabel("Density")

    stats_txt = (
        f"n={len(w)} mean={w.mean():.3f} median={w.median():.3f}\n"
        f"std={w.std(ddof=1):.3f} min={w.min():.2f} max={w.max():.2f}\n"
        f"P10={w.quantile(0.10):.3f} P90={w.quantile(0.90):.3f}\n"
        f"IQR [{w.quantile(0.25):.3f}, {w.quantile(0.75):.3f}]"
    )
    fig.text(0.5, -0.02, stats_txt, ha="center", fontsize=10)
    fig.suptitle(f"Weighted Average Score Distribution — {label}", fontsize=16)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap_page(merged: pd.DataFrame, suffix: str, label: str, pdf: PdfPages) -> None:
    cols = [f"{dim}_{suffix}" for dim in DIMENSIONS]
    corr = merged[cols].apply(pd.to_numeric, errors="coerce").corr()
    corr.index = [i.replace(f"_{suffix}", "").replace("_", " ") for i in corr.index]
    corr.columns = [i.replace(f"_{suffix}", "").replace("_", " ") for i in corr.columns]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Cross-Dimension Pearson Correlation Matrix — {label}")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_page(merged: pd.DataFrame, suffix: str, label: str, weights: Dict[str, float], denom: float, pdf: PdfPages) -> None:
    data = []
    labels = []

    for dim in DIMENSIONS:
        s = pd.to_numeric(merged[f"{dim}_{suffix}"], errors="coerce").dropna()
        data.append(s.values)
        labels.append(dim.replace("_", "\n"))

    w = compute_weighted_series(merged, suffix, weights, denom).dropna()
    data.append(w.values)
    labels.append("weighted\navg")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_title(f"Score Distribution — Box & Whisker (with outliers) — {label}")
    ax.set_ylabel("Score")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf_page(merged: pd.DataFrame, suffix: str, label: str, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for dim in DIMENSIONS:
        s = np.sort(pd.to_numeric(merged[f"{dim}_{suffix}"], errors="coerce").dropna().values)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, where="post", label=dim.replace("_", " "))

    ax.axhline(0.5, linestyle="--")
    ax.text(0.98, 0.51, "50th pct", ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlim(1, 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"Empirical Cumulative Distribution Functions — {label}")
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_grade_band_page(merged: pd.DataFrame, suffix: str, label: str, weights: Dict[str, float], denom: float, pdf: PdfPages) -> None:
    w = compute_weighted_series(merged, suffix, weights, denom).dropna()

    bands = {
        "Poor < 2": ((w < 2).sum()),
        "Below avg\n2–3": (((w >= 2) & (w < 3)).sum()),
        "Average\n3–4": (((w >= 3) & (w < 4)).sum()),
        "Good\n4–5": (((w >= 4) & (w < 5)).sum()),
        "Perfect = 5": ((w == 5).sum()),
    }

    labels_ = list(bands.keys())
    counts = np.array(list(bands.values()))
    pct = counts / counts.sum() * 100 if counts.sum() > 0 else counts

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(labels_, counts)
    ax.set_ylabel("Count")
    ax.set_title(f"Grade-Band Distribution (Weighted Average) — {label}")

    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(i, c + max(counts.max() * 0.03, 1), f"{p:.1f}%\n({int(c)})", ha="center", va="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_descriptive_stats_page(merged: pd.DataFrame, suffix: str, label: str, weights: Dict[str, float], denom: float, pdf: PdfPages) -> None:
    stats_df = descriptive_stats_table(merged, suffix, weights, denom).copy()
    for c in ["mean", "median", "std", "min", "max", "P25", "P75"]:
        stats_df[c] = stats_df[c].map(lambda x: f"{x:.3f}")
    stats_df["n"] = stats_df["n"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.axis("off")
    tbl = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax.set_title(f"Descriptive Statistics Summary — {label}")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_visual_report(
    merged: pd.DataFrame,
    suffix: str,
    label: str,
    weights: Dict[str, float],
    denom: float,
    output_pdf: str,
) -> None:
    with PdfPages(output_pdf) as pdf:
        plot_score_frequency_page(merged, suffix, label, pdf)
        plot_kde_page(merged, suffix, label, pdf)
        plot_weighted_distribution_page(merged, suffix, label, weights, denom, pdf)
        plot_correlation_heatmap_page(merged, suffix, label, pdf)
        plot_boxplot_page(merged, suffix, label, weights, denom, pdf)
        plot_ecdf_page(merged, suffix, label, pdf)
        plot_grade_band_page(merged, suffix, label, weights, denom, pdf)
        plot_descriptive_stats_page(merged, suffix, label, weights, denom, pdf)
def generate_human_vs_llm_comparison(
    merged: pd.DataFrame,
    weights: Dict[str, float],
    denom: float,
    output_pdf: str,
) -> None:
    with PdfPages(output_pdf) as pdf:

        # -------- PAGE 1: Mean score comparison --------
        human_means = [
            pd.to_numeric(merged[f"{dim}_human"], errors="coerce").mean()
            for dim in DIMENSIONS
        ]
        llm_means = [
            pd.to_numeric(merged[f"{dim}_llm"], errors="coerce").mean()
            for dim in DIMENSIONS
        ]

        labels = [d.replace("_", " ") for d in DIMENSIONS]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)

        ax.bar(x - width/2, human_means, width, label="Human")
        ax.bar(x + width/2, llm_means, width, label="LLM Judge")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Mean Score")
        ax.set_ylim(0,5.2)
        ax.set_title("Human vs LLM Mean Scores by Dimension")

        ax.legend()

        pdf.savefig(fig)
        plt.close(fig)


        # -------- PAGE 2: Weighted score distribution --------
        h = compute_weighted_series(merged, "human", weights, denom).dropna()
        l = compute_weighted_series(merged, "llm", weights, denom).dropna()

        fig, axes = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)

        axes[0].hist(h, bins=20, alpha=0.6, label="Human")
        axes[0].hist(l, bins=20, alpha=0.6, label="LLM Judge")
        axes[0].set_title("Weighted Score Distribution")
        axes[0].set_xlabel("Weighted Score")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        try:
            h.plot(kind="kde", ax=axes[1], label="Human")
            l.plot(kind="kde", ax=axes[1], label="LLM Judge")
        except Exception:
            pass

        axes[1].set_title("Weighted Score KDE")
        axes[1].set_xlabel("Weighted Score")
        axes[1].legend()

        pdf.savefig(fig)
        plt.close(fig)


        # -------- PAGE 3: ECDF comparison --------
        fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)

        h_sorted = np.sort(h.values)
        l_sorted = np.sort(l.values)

        h_y = np.arange(1, len(h_sorted)+1) / len(h_sorted)
        l_y = np.arange(1, len(l_sorted)+1) / len(l_sorted)

        ax.step(h_sorted, h_y, where="post", label="Human")
        ax.step(l_sorted, l_y, where="post", label="LLM Judge")

        ax.set_xlim(1,5)
        ax.set_ylim(0,1)

        ax.set_xlabel("Weighted Score")
        ax.set_ylabel("Cumulative %")

        ax.set_title("Weighted Score ECDF Comparison")

        ax.legend()

        pdf.savefig(fig)
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare human and LLM judge scores on MathDial tutoring trajectories."
    )
    parser.add_argument("--human_file", required=True, help="Path to human-scored JSONL")
    parser.add_argument("--llm_file", required=True, help="Path to LLM-scored JSONL")
    parser.add_argument(
        "--weights_json",
        default=None,
        help="Optional JSON string of weights by dimension. Default matches preference-pair generation weights.",
    )
    parser.add_argument("--output_prefix", default=None, help="Optional prefix for CSV/JSON outputs")
    args = parser.parse_args()

    weights, denom = parse_weights_arg(args.weights_json)

    human_df = load_scores_jsonl(args.human_file)
    llm_df = load_scores_jsonl(args.llm_file)

    # Optional sanity check: make sure the number of rollouts per qid matches.
    human_counts = human_df.groupby("qid").size()
    llm_counts = llm_df.groupby("qid").size()

    common_qids = set(human_counts.index).intersection(set(llm_counts.index))
    mismatch_qids = sorted([
        qid for qid in common_qids
        if human_counts[qid] != llm_counts[qid]
    ])

    if mismatch_qids:
        print("Warning: some qids have different numbers of rollouts in human vs llm files.")
        print("First few mismatched qids:", mismatch_qids[:10])

    merged = human_df.merge(
        llm_df,
        on=["qid", "rollout_idx"],
        suffixes=("_human", "_llm"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping (qid, rollout_idx) pairs between human and LLM files")

    merged = add_weighted_score(merged, "human", weights, denom)
    merged = add_weighted_score(merged, "llm", weights, denom)

    # Drop rows with missing / non-numeric scores after coercion
    bad_rows = merged[
        merged["weighted_score_human"].isna()
        | merged["weighted_score_llm"].isna()
        | merged["weighted_score_round_human"].isna()
        | merged["weighted_score_round_llm"].isna()
    ]

    if not bad_rows.empty:
        print("Warning: dropping rows with missing/non-numeric weighted scores.")
        cols_to_show = [
            "qid", "rollout_idx",
            "socratic_guidance_human", "mathematical_accuracy_human", "relevance_human", "conciseness_human", "overall_quality_human",
            "socratic_guidance_llm", "mathematical_accuracy_llm", "relevance_llm", "conciseness_llm", "overall_quality_llm",
        ]
        print(bad_rows[cols_to_show].head(10).to_string(index=False))

        merged = merged.dropna(subset=[
            "weighted_score_human",
            "weighted_score_llm",
            "weighted_score_round_human",
            "weighted_score_round_llm",
        ]).copy()

    n_human = len(human_df)
    n_llm = len(llm_df)
    n_merged = len(merged)
    print(f"Loaded {n_human} human rows, {n_llm} llm rows, matched {n_merged} trajectories.\n")

    metrics = [compute_dimension_metrics(merged, dim) for dim in DIMENSIONS]
    weighted_metrics = compute_weighted_metrics(merged)
    summary = overall_summary(merged)

    print_results(metrics, weighted_metrics, summary, weights, denom)

    visual_prefix = args.output_prefix if args.output_prefix else "alignment_results"

    generate_visual_report(
        merged=merged,
        suffix="human",
        label="Human",
        weights=weights,
        denom=denom,
        output_pdf=f"{visual_prefix}_human_visuals.pdf",
    )

    generate_visual_report(
        merged=merged,
        suffix="llm",
        label="LLM Judge",
        weights=weights,
        denom=denom,
        output_pdf=f"{visual_prefix}_llm_visuals.pdf",
    )
    comparison_pdf = f"{visual_prefix}_human_vs_llm_comparison.pdf"

    generate_human_vs_llm_comparison(
        merged=merged,
        weights=weights,
        denom=denom,
        output_pdf=comparison_pdf,
    )

    print(f"Saved comparison report to {comparison_pdf}")

    print(f"Saved visual reports to {visual_prefix}_human_visuals.pdf and {visual_prefix}_llm_visuals.pdf")

    if args.output_prefix:
        save_outputs(metrics, weighted_metrics, summary, args.output_prefix, weights, denom)
        print(f"Saved outputs to {args.output_prefix}_metrics.csv and {args.output_prefix}_full.json")


if __name__ == "__main__":
    main()