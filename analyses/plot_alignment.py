from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_GLOB = "alignment_*.csv"

# Order you want on the x-axis
POLICY_ORDER = [
    "SFT",
    "DPO (LLM-LLM)",
    "DPO (Round 1)",
    "DPO (Round 2)",
    "Human"
]

# Optional mapping from filename/CSV prefix labels to prettier names
POLICY_LABEL_MAP = {
    "human": "Human",
    "sft": "SFT",
    "dpo_round1": "DPO (Round 1)",
    "dpo_round_1": "DPO (Round 1)",
    "dpo_round2": "DPO (Round 2)",
    "dpo_round_2": "DPO (Round 2)",
    "dpo_multiround_r1": "DPO (Round 1)",
    "dpo_multiround_r2": "DPO (Round 2)",
    "dpo_multiround_r3": "DPO (Round 3)",
    "dpo_multiround_r4": "DPO (Round 4)",
    "dpo_llm_llm": "DPO (LLM-LLM)",
    "llm_llm": "DPO (LLM-LLM)",
}

# Optional mapping for judge names
JUDGE_LABEL_MAP = {
    "llama": "LLaMA",
    "mixtral": "Mixtral",
}


def prettify_policy(x: str) -> str:
    if x is None:
        return "Unknown"
    key = str(x).strip().lower()
    return POLICY_LABEL_MAP.get(key, str(x))


def prettify_judge(x: str) -> str:
    if x is None:
        return "Unknown"
    key = str(x).strip().lower()
    return JUDGE_LABEL_MAP.get(key, str(x))


def parse_from_filename(path: Path):
    """
    Expected filename pattern like:
      alignment_llama_SFT.csv
      alignment_mixtral_dpo_round2.csv
    """
    stem = path.stem  # e.g. alignment_llama_SFT
    m = re.match(r"alignment_([^_]+)_(.+)$", stem)
    if not m:
        return None, None
    judge = m.group(1)
    prefix = m.group(2)
    return judge, prefix


def load_alignment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # infer judge/policy from columns first
    judge = None
    policy = None

    if "llm_judge_model" in df.columns and df["llm_judge_model"].notna().any():
        judge = str(df["llm_judge_model"].dropna().iloc[0])

    if "policy_prefix" in df.columns and df["policy_prefix"].notna().any():
        policy = str(df["policy_prefix"].dropna().iloc[0])

    # fallback to filename
    file_judge, file_policy = parse_from_filename(path)
    if judge is None:
        judge = file_judge
    if policy is None:
        policy = file_policy

    df = df.copy()
    df["Judge"] = prettify_judge(judge)
    df["Policy"] = prettify_policy(policy)
    df["source_file"] = path.name
    return df


def collect_all_alignment_results() -> pd.DataFrame:
    files = sorted(Path(".").glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching {CSV_GLOB} found in current directory.")

    parts = []
    for path in files:
        try:
            parts.append(load_alignment_csv(path))
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not parts:
        raise ValueError("No readable alignment CSV files found.")

    combined = pd.concat(parts, ignore_index=True)

    # Standardize expected score column names
    # Prefer human_mean if present; otherwise look for older naming
    if "human_mean" not in combined.columns:
        raise ValueError("Expected column 'human_mean' not found in alignment CSVs.")
    if "policy_mean" not in combined.columns:
        raise ValueError("Expected column 'policy_mean' not found in alignment CSVs.")
    if "metric" not in combined.columns:
        raise ValueError("Expected column 'metric' not found in alignment CSVs.")

    return combined


def build_plot_df(all_df: pd.DataFrame, metric_name: str, value_col: str) -> pd.DataFrame:
    sub = all_df.loc[all_df["metric"] == metric_name, ["Policy", "Judge", value_col]].copy()

    # If there are duplicate rows for same Policy/Judge/metric, average them
    sub = (
        sub.groupby(["Policy", "Judge"], as_index=False)[value_col]
        .mean()
    )

    pivot = sub.pivot(index="Policy", columns="Judge", values=value_col)

    ordered_index = [p for p in POLICY_ORDER if p in pivot.index] + [
        p for p in pivot.index if p not in POLICY_ORDER
    ]
    pivot = pivot.reindex(ordered_index)

    return pivot


def nice_metric_name(metric: str) -> str:
    if metric == "weighted_composite":
        return "Weighted Composite"
    if metric == "raw_macro_summary":
        return "Raw Macro Summary"
    return metric.replace("_", " ").title()


def safe_filename(metric: str) -> str:
    return metric.lower().replace(" ", "_")


def plot_grouped_bars(
    pivot: pd.DataFrame,
    ylabel: str,
    title: str,
    output_png: str,
    ylim_pad: float = 0.03,
):
    judges = [c for c in pivot.columns if c in ["LLaMA", "Mixtral"]] + [
        c for c in pivot.columns if c not in ["LLaMA", "Mixtral"]
    ]

    if not judges:
        print(f"Skipping {output_png}: no judge columns found.")
        return

    x = np.arange(len(pivot.index))
    n_judges = len(judges)
    width = 0.8 / max(n_judges, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals = []

    for i, judge in enumerate(judges):
        vals = pivot[judge].values.astype(float)
        all_vals.extend([v for v in vals if not np.isnan(v)])

        offset = (i - (n_judges - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"{judge} Judge")

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.003,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        pad = max((ymax - ymin) * 0.25, ylim_pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    all_df = collect_all_alignment_results()

    # Plot the policy means for every metric present
    metrics = list(dict.fromkeys(all_df["metric"].tolist()))

    for metric in metrics:
        pivot = build_plot_df(all_df, metric, "policy_mean")

        if pivot.empty:
            continue

        title = f"{nice_metric_name(metric)} by Policy and Judge"
        ylabel = "Policy Mean Score"
        output_png = f"{safe_filename(metric)}_grouped_by_policy.png"

        plot_grouped_bars(
            pivot=pivot,
            ylabel=ylabel,
            title=title,
            output_png=output_png,
        )

    # Optional: if you also want the human means plotted for reference,
    # uncomment this block.
    #
    for metric in metrics:
        pivot = build_plot_df(all_df, metric, "human_mean")
        if pivot.empty:
            continue
        title = f"Human {nice_metric_name(metric)} by Policy Label and Judge"
        ylabel = "Human Mean Score"
        output_png = f"human_{safe_filename(metric)}_grouped_by_policy.png"
        plot_grouped_bars(pivot=pivot, ylabel=ylabel, title=title, output_png=output_png)


if __name__ == "__main__":
    main()