import argparse
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

import pandas as pd


METRICS = [
    "socratic_guidance",
    "mathematical_accuracy",
    "relevance",
    "conciseness",
    "overall_quality",
]


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

    missing = [m for m in METRICS if m not in weights]
    if missing:
        raise ValueError(f"Missing weights for metrics: {missing}")

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

            obj = json.loads(line)

            row = {"qid": obj["qid"]}

            for m in METRICS:
                row[m] = obj["judge_scores"][m]

            rows.append(row)

    df = pd.DataFrame(rows)

    df["rollout_idx"] = df.groupby("qid").cumcount()

    return df


def add_weighted_score(df, prefix, weights, denom):

    for m in METRICS:

        col = f"{m}_{prefix}"

        df[col] = pd.to_numeric(df[col], errors="coerce")

    weighted_sum = sum(df[f"{m}_{prefix}"] * weights[m] for m in METRICS)

    df[f"weighted_score_{prefix}"] = weighted_sum / denom

    return df


def assign_outcome(human, policy, tie_threshold=0.0):

    diff = human - policy

    outcome = pd.Series(index=diff.index, dtype="object")

    outcome[diff > tie_threshold] = "human_win"
    outcome[diff < -tie_threshold] = "policy_win"
    outcome[diff.abs() <= tie_threshold] = "tie"

    return outcome


def compute_metrics(human_scores, policy_scores, metric_name, tie_threshold=0.0):

    df = pd.DataFrame(
        {"human": human_scores, "policy": policy_scores}
    ).dropna()

    h = pd.to_numeric(df["human"])
    p = pd.to_numeric(df["policy"])

    outcome = assign_outcome(h, p, tie_threshold)

    margin = h - p

    return {

        "metric": metric_name,
        "n": len(df),

        "human_win_rate": (outcome == "human_win").mean(),
        "policy_win_rate": (outcome == "policy_win").mean(),
        "tie_rate": (outcome == "tie").mean(),

        "mean_margin_human_minus_policy": margin.mean(),
        "mean_abs_margin": margin.abs().mean(),

        "human_mean": h.mean(),
        "policy_mean": p.mean(),

        "human_std": h.std(ddof=1) if len(h) > 1 else 0.0,
        "policy_std": p.std(ddof=1) if len(p) > 1 else 0.0,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--human_scores", required=True)
    parser.add_argument("--policy_scores", required=True)

    parser.add_argument("--prefix", required=True)

    parser.add_argument("--weights_json", default=None)
    parser.add_argument("--tie_threshold", type=float, default=0.0)

    args = parser.parse_args()

    weights, denom = parse_weights_arg(args.weights_json)

    human_df = load_scores_jsonl(args.human_scores)
    policy_df = load_scores_jsonl(args.policy_scores)

    merged = human_df.merge(
        policy_df,
        on=["qid", "rollout_idx"],
        suffixes=("_human", "_policy"),
        how="inner",
    )

    merged = add_weighted_score(merged, "human", weights, denom)
    merged = add_weighted_score(merged, "policy", weights, denom)

    rows = []

    for m in METRICS:

        rows.append(
            compute_metrics(
                merged[f"{m}_human"],
                merged[f"{m}_policy"],
                m,
                args.tie_threshold,
            )
        )

    rows.append(
        compute_metrics(
            merged["weighted_score_human"],
            merged["weighted_score_policy"],
            "weighted_composite",
            args.tie_threshold,
        )
    )

    df = pd.DataFrame(rows)

    # automatically detect judge model
    judge_model = Path(args.policy_scores).parent.name.replace("policy_judge_", "")

    df["llm_judge_model"] = judge_model
    df["policy_prefix"] = args.prefix

    output_name = f"alignment_{judge_model}_{args.prefix}.csv"

    df.to_csv(output_name, index=False)

    print(f"Saved results -> {output_name}")


if __name__ == "__main__":
    main()