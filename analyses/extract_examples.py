import pandas as pd
import json
import argparse

BASELINE_SCORES = "../output/policy_judge_llama_with_llama_pairing/Qwen2.5-1.5B-Instruct_judge_scores.jsonl"
DPO_SCORES = "../output/policy_judge_llama_with_llama_pairing/dpo_multiround_r4_judge_scores.jsonl"

BASELINE_CONVOS = "../output/policy_judge_llama_with_llama_pairing/Qwen2.5-1.5B-Instruct_conversations.jsonl"
DPO_CONVOS = "../output/policy_judge_llama_with_llama_pairing/dpo_multiround_r4_conversations.jsonl"


def load_jsonl(path):
    rows = []
    with open(path.strip(), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scores(path, prefix):
    rows = []
    for obj in load_jsonl(path):
        row = {"qid": obj["qid"]}
        for k, v in obj["judge_scores"].items():
            row[f"{k}_{prefix}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df["rollout_idx"] = df.groupby("qid").cumcount()
    return df


def infer_model_text_key(example_obj):
    meta_keys = {
        "qid",
        "scenario",
        "question",
        "ground_truth",
        "student_incorrect_solution",
        "student_profile",
        "teacher_described_confusion",
        "self-correctness",
        "self-typical-confusion",
        "self-typical-interactions",
        "conversation",
        "judge_model",
        "judge_scores",
        "model_name",
    }

    candidate_keys = []
    for k, v in example_obj.items():
        if k not in meta_keys and isinstance(v, str):
            candidate_keys.append(k)

    if len(candidate_keys) == 1:
        return candidate_keys[0]

    best_key = None
    best_len = -1
    for k in candidate_keys:
        v = example_obj.get(k, "")
        if isinstance(v, str) and len(v) > best_len:
            best_key = k
            best_len = len(v)

    return best_key


def load_convos(path, prefix):
    objs = load_jsonl(path)
    if not objs:
        raise ValueError(f"No rows found in {path}")

    model_text_key = infer_model_text_key(objs[0])
    print(f"[{prefix}] detected model text key: {model_text_key}")

    rows = []
    for obj in objs:
        rows.append({
            "qid": obj["qid"],
            f"question_{prefix}": obj.get("question", ""),
            f"reference_conversation_{prefix}": obj.get("conversation", ""),
            f"model_conversation_{prefix}": obj.get(model_text_key, ""),
        })

    df = pd.DataFrame(rows)
    df["rollout_idx"] = df.groupby("qid").cumcount()
    return df


def select_examples(df, mode, top_k=5, threshold=2, metric="overall_quality"):
    base_col = f"{metric}_base"
    dpo_col = f"{metric}_dpo"

    df = df.copy()
    df["gap"] = df[base_col] - df[dpo_col]
    df["abs_gap"] = df["gap"].abs()
    df["socratic_diff"] = df["socratic_guidance_dpo"] - df["socratic_guidance_base"]

    if mode == 1:
        # 1. Baseline wins big
        out = df[df["gap"] >= threshold].sort_values("gap", ascending=False).head(top_k)

    elif mode == 2:
        # 2. DPO wins big
        out = df[df["gap"] <= -threshold].sort_values("gap", ascending=True).head(top_k)

    elif mode == 3:
        # 3. Close calls
        out = df[df["abs_gap"] <= threshold].sort_values("abs_gap", ascending=True).head(top_k)

    elif mode == 4:
        # 4. DPO more Socratic but loses overall
        out = df[
            (df["socratic_guidance_dpo"] > df["socratic_guidance_base"]) &
            (df["overall_quality_dpo"] < df["overall_quality_base"])
        ].sort_values(
            ["socratic_diff", "gap"],
            ascending=[False, False]
        ).head(top_k)

    elif mode == 5:
        out = df.sample(n=top_k, random_state=42)

    else:
        raise ValueError("mode must be 1, 2, 3, 4, or 5")

    return out


def print_example(row):
    print("\n" + "=" * 120)
    print(f"QID: {row['qid']} | rollout_idx: {row['rollout_idx']} | gap(base-dpo): {row['gap']}")

    question = row.get("question_base", "") or row.get("question_dpo", "")
    if question:
        print("\n--- QUESTION ---")
        print(question)

    print("\n--- BASELINE SCORES ---")
    print({
        "socratic_guidance": row["socratic_guidance_base"],
        "mathematical_accuracy": row["mathematical_accuracy_base"],
        "relevance": row["relevance_base"],
        "conciseness": row["conciseness_base"],
        "overall_quality": row["overall_quality_base"],
    })

    print("\n--- DPO R4 SCORES ---")
    print({
        "socratic_guidance": row["socratic_guidance_dpo"],
        "mathematical_accuracy": row["mathematical_accuracy_dpo"],
        "relevance": row["relevance_dpo"],
        "conciseness": row["conciseness_dpo"],
        "overall_quality": row["overall_quality_dpo"],
    })

    print("\n--- BASELINE MODEL CONVERSATION ---")
    print(row.get("model_conversation_base", "[missing]"))

    print("\n--- DPO R4 MODEL CONVERSATION ---")
    print(row.get("model_conversation_dpo", "[missing]"))

    print("\n--- REFERENCE / SOURCE CONVERSATION ---")
    print(row.get("reference_conversation_base", "[missing]"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, required=True, help="1=baseline wins big, 2=dpo wins big, 3=close calls, 4=dpo more socratic but loses overall")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--metric", type=str, default="overall_quality")
    args = parser.parse_args()

    baseline_scores = load_scores(BASELINE_SCORES, "base")
    dpo_scores = load_scores(DPO_SCORES, "dpo")

    baseline_convos = load_convos(BASELINE_CONVOS, "base")
    dpo_convos = load_convos(DPO_CONVOS, "dpo")

    df = baseline_scores.merge(
        dpo_scores,
        on=["qid", "rollout_idx"],
        how="inner"
    )

    df = df.merge(
        baseline_convos,
        on=["qid", "rollout_idx"],
        how="left"
    ).merge(
        dpo_convos,
        on=["qid", "rollout_idx"],
        how="left"
    )

    selected = select_examples(
        df,
        mode=args.mode,
        top_k=args.top_k,
        threshold=args.threshold,
        metric=args.metric,
    )

    print("\n=== SELECTED EXAMPLES ===")
    cols = ["qid", "rollout_idx", "gap"]
    if "socratic_diff" in selected.columns:
        cols.append("socratic_diff")
    print(selected[cols])

    for _, row in selected.iterrows():
        print_example(row)

    print("\nDone.")


if __name__ == "__main__":
    main()