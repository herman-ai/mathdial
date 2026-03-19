from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_GLOB = "alignment_*.csv"

POLICY_ORDER = [
    "SFT",
    "DPO (LLM-LLM)",
    "DPO (Round 1)",
    "DPO (Round 4)",
    "Human",
]

POLICY_LABEL_MAP = {
    "human": "Human",
    "sft": "SFT",

    "normal_dpo_llmvsllm": "DPO (LLM-LLM)",
    "normal_dpo_llm_llm": "DPO (LLM-LLM)",
    "dpo_llm_llm": "DPO (LLM-LLM)",
    "llm_llm": "DPO (LLM-LLM)",
    "llmvsllm": "DPO (LLM-LLM)",

    "dpo_multiround_r1": "DPO (Round 1)",
    "dpo_round1": "DPO (Round 1)",
    "dpo_round_1": "DPO (Round 1)",

    "dpo_multiround_r4": "DPO (Round 4)",
    "dpo_round4": "DPO (Round 4)",
    "dpo_round_4": "DPO (Round 4)",
}

JUDGE_LABEL_MAP = {
    "llama": "LLaMA",
    "mixtral": "Mixtral",
}

JUDGE_ORDER = ["LLaMA", "Mixtral"]

JUDGE_COLOR_MAP = {
    "LLaMA": "#4C72B0",
    "Mixtral": "#DD8452",
}


def normalize_key(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return s


def prettify_policy(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "Unknown"

    raw = str(x).strip()
    key = normalize_key(raw)

    if key in POLICY_LABEL_MAP:
        return POLICY_LABEL_MAP[key]

    if "human" in key:
        return "Human"
    if key == "sft":
        return "SFT"
    if "llm" in key and "dpo" in key:
        return "DPO (LLM-LLM)"
    if "r1" in key or "round1" in key or "round_1" in key:
        return "DPO (Round 1)"
    if "r4" in key or "round4" in key or "round_4" in key:
        return "DPO (Round 4)"

    return raw


def prettify_judge(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "Unknown"
    raw = str(x).strip()
    key = normalize_key(raw)
    return JUDGE_LABEL_MAP.get(key, raw)


def parse_from_filename(path: Path):
    stem = path.stem
    m = re.match(r"alignment_([^_]+)_(.+)$", stem)
    if not m:
        return None, None
    judge = m.group(1)
    policy = m.group(2)
    return judge, policy


def load_alignment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    judge = None
    policy = None

    if "llm_judge_model" in df.columns and df["llm_judge_model"].notna().any():
        judge = df["llm_judge_model"].dropna().iloc[0]

    for col in ["policy_prefix", "policy", "model_label", "model_name"]:
        if col in df.columns and df[col].notna().any():
            policy = df[col].dropna().iloc[0]
            break

    file_judge, file_policy = parse_from_filename(path)
    if judge is None:
        judge = file_judge
    if policy is None:
        policy = file_policy

    df["Judge"] = prettify_judge(judge)
    df["Policy"] = prettify_policy(policy)
    df["source_file"] = path.name
    return df


def collect_all_alignment_results(root=".") -> pd.DataFrame:
    files = sorted(Path(root).glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching {CSV_GLOB} found in {root!r}.")

    parts = []
    for path in files:
        try:
            parts.append(load_alignment_csv(path))
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not parts:
        raise ValueError("No readable alignment CSV files found.")

    combined = pd.concat(parts, ignore_index=True)

    required_cols = ["metric", "policy_mean", "Policy", "Judge"]
    missing = [c for c in required_cols if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    combined["policy_mean"] = pd.to_numeric(combined["policy_mean"], errors="coerce")
    return combined


def ordered_policy_index(values):
    values = list(values)
    ordered = [p for p in POLICY_ORDER if p in values]
    extras = [p for p in values if p not in POLICY_ORDER]
    return ordered + extras


def ordered_judges(values):
    values = list(values)
    ordered = [j for j in JUDGE_ORDER if j in values]
    extras = [j for j in values if j not in ordered]
    return ordered + extras


def build_weighted_composite_pivot(all_df: pd.DataFrame) -> pd.DataFrame:
    sub = all_df[all_df["metric"] == "weighted_composite"].copy()
    sub = sub.dropna(subset=["policy_mean"])

    sub = sub.groupby(["Policy", "Judge"], as_index=False)["policy_mean"].mean()
    pivot = sub.pivot(index="Policy", columns="Judge", values="policy_mean")

    pivot = pivot.reindex(index=ordered_policy_index(pivot.index))
    pivot = pivot.reindex(columns=ordered_judges(pivot.columns))
    return pivot


def build_judge_disagreement_df(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean absolute judge gap across core metrics for each policy.
    Lower = judges agree more.
    """
    sub = all_df.copy()
    sub = sub[sub["metric"] != "weighted_composite"]
    sub = sub.dropna(subset=["policy_mean"])

    grouped = (
        sub.groupby(["Policy", "Judge", "metric"], as_index=False)["policy_mean"]
        .mean()
    )

    pivot = grouped.pivot_table(
        index=["Policy", "metric"],
        columns="Judge",
        values="policy_mean"
    ).reset_index()

    if not {"LLaMA", "Mixtral"}.issubset(set(pivot.columns)):
        raise ValueError("Need both LLaMA and Mixtral results to compute disagreement.")

    pivot["abs_gap"] = (pivot["LLaMA"] - pivot["Mixtral"]).abs()

    out = (
        pivot.groupby("Policy", as_index=False)["abs_gap"]
        .mean()
        .rename(columns={"abs_gap": "mean_abs_judge_gap"})
    )

    out["Policy"] = pd.Categorical(out["Policy"], categories=ordered_policy_index(out["Policy"]), ordered=True)
    out = out.sort_values("Policy").reset_index(drop=True)
    return out


def plot_weighted_composite(pivot: pd.DataFrame, output_path: Path):
    if pivot.empty:
        print("Skipping weighted composite plot: empty data.")
        return

    judges = ordered_judges(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = []

    for i, judge in enumerate(judges):
        vals = pd.to_numeric(pivot[judge], errors="coerce").to_numpy(dtype=float)
        all_vals.extend([v for v in vals if not np.isnan(v)])

        offset = (i - (len(judges) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=f"{judge} Judge",
            color=JUDGE_COLOR_MAP.get(judge, None),
        )

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Weighted Composite Score")
    ax.set_title("Weighted Composite by Policy and Judge")
    ax.legend()

    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        pad = max((ymax - ymin) * 0.25, 0.05)
        ax.set_ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


def plot_judge_disagreement(df: pd.DataFrame, output_path: Path):
    if df.empty:
        print("Skipping judge disagreement plot: empty data.")
        return

    x = np.arange(len(df))
    vals = df["mean_abs_judge_gap"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, vals, color="#55A868")

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.003,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Policy"], rotation=20, ha="right")
    ax.set_ylabel("Mean Absolute Gap: |LLaMA - Mixtral|")
    ax.set_title("Judge Disagreement by Policy (Lower is Better)")

    ymin = 0.0
    ymax = max(vals) if len(vals) else 0.1
    ax.set_ylim(ymin, ymax + max(ymax * 0.2, 0.03))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


def print_summary(weighted_pivot: pd.DataFrame, disagreement_df: pd.DataFrame):
    print("\n=== Weighted composite by policy and judge ===")
    print(weighted_pivot)

    if not weighted_pivot.empty:
        judge_means = weighted_pivot.mean(axis=1, skipna=True).sort_values(ascending=False)
        print("\n=== Average weighted composite across judges (higher is better) ===")
        print(judge_means)

    print("\n=== Judge disagreement by policy (lower is better) ===")
    print(disagreement_df)

    if not disagreement_df.empty:
        most_stable = disagreement_df.sort_values("mean_abs_judge_gap", ascending=True)
        print("\nMost judge-stable policy:")
        print(most_stable.head(1))

    if not weighted_pivot.empty and not disagreement_df.empty:
        avg_score = weighted_pivot.mean(axis=1, skipna=True).rename("avg_weighted_score")
        stability = disagreement_df.set_index("Policy")["mean_abs_judge_gap"]
        summary = pd.concat([avg_score, stability], axis=1)
        summary = summary.rename(columns={"mean_abs_judge_gap": "judge_gap"})
        summary["score_minus_gap"] = summary["avg_weighted_score"] - summary["judge_gap"]
        summary = summary.sort_values("score_minus_gap", ascending=False)

        print("\n=== Combined view: high score + low judge disagreement ===")
        print(summary)


def main():
    root = "."
    outdir = Path("..") / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = collect_all_alignment_results(root)

    print("Detected policies:", sorted(all_df["Policy"].dropna().unique().tolist()))
    print("Detected judges:", sorted(all_df["Judge"].dropna().unique().tolist()))
    print("Detected metrics:", sorted(all_df["metric"].dropna().unique().tolist()))
    print(f"Saving plots to: {outdir.resolve()}")

    weighted_pivot = build_weighted_composite_pivot(all_df)
    disagreement_df = build_judge_disagreement_df(all_df)

    print_summary(weighted_pivot, disagreement_df)

    plot_weighted_composite(
        weighted_pivot,
        outdir / "weighted_composite_by_policy_and_judge.png",
    )

    plot_judge_disagreement(
        disagreement_df,
        outdir / "judge_disagreement_by_policy.png",
    )


if __name__ == "__main__":
    main()