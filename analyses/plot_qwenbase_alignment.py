from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CSV_GLOB = "qwen_base/alignment_*.csv"

POLICY_ORDER = [
    "SFT",
    "DPO (LLM-LLM)",
    "DPO (Round 1)",
    "DPO (Round 2)",
    "DPO (Round 3)",
    "DPO (Round 4)",
]


def load_all(root="."):
    files = list(Path(root).glob(CSV_GLOB))
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        policy = f.stem.split("_", 2)[-1]
        df["Policy"] = policy
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def clean_policy(p):
    p = p.lower()
    if "sft" in p:
        return "SFT"
    if "llm" in p and "dpo" in p:
        return "DPO (LLM-LLM)"
    if "r1" in p:
        return "DPO (Round 1)"
    if "r2" in p:
        return "DPO (Round 2)"
    if "r3" in p:
        return "DPO (Round 3)"
    if "r4" in p:
        return "DPO (Round 4)"
    return p


def build_alignment(df):
    df = df[df["metric"] == "weighted_composite"].copy()
    df["Policy"] = df["Policy"].apply(clean_policy)

    out = (
        df.groupby("Policy", as_index=False)
        .agg(
            human_win_rate=("human_win_rate", "mean"),
            policy_win_rate=("policy_win_rate", "mean"),
            tie_rate=("tie_rate", "mean"),
        )
    )

    out["Policy"] = pd.Categorical(out["Policy"], POLICY_ORDER, ordered=True)
    out = out.sort_values("Policy")

    # add tie-adjusted metric
    out["tie_adjusted"] = out["human_win_rate"] + 0.5 * out["tie_rate"]

    return out


# ========================
# PLOT 1: Alignment accuracy
# ========================
def plot_alignment_accuracy(df):
    x = np.arange(len(df))

    plt.figure(figsize=(9, 5))
    plt.bar(x, df["human_win_rate"], label="Strict")
    plt.bar(x, df["tie_adjusted"], alpha=0.6, label="Tie-adjusted")

    for i, v in enumerate(df["human_win_rate"]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

    plt.xticks(x, df["Policy"], rotation=20)
    plt.ylabel("Agreement with Human Preference")
    plt.title("LLM Judge Alignment Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ========================
# PLOT 2: Outcome breakdown
# ========================
def plot_outcomes(df):
    x = np.arange(len(df))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, df["human_win_rate"], w, label="Human preferred")
    plt.bar(x, df["tie_rate"], w, label="Tie")
    plt.bar(x + w, df["policy_win_rate"], w, label="Policy preferred")

    plt.xticks(x, df["Policy"], rotation=20)
    plt.ylabel("Rate")
    plt.title("Judge Decision Breakdown")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = load_all()
    df = build_alignment(df)

    print(df)

    plot_alignment_accuracy(df)
    plot_outcomes(df)


if __name__ == "__main__":
    main()