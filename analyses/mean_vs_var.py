from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("../output/policy_judge_llama_with_llama_pairing")
OUT_DIR = Path("./plots")
OUT_DIR.mkdir(exist_ok=True)

METRIC = "overall_quality"

def infer_model_name(path: Path) -> str:
    name = path.stem.lower()

    if "mathdial_human_teacher" in name:
        return "Human"
    if "qwen_finetuned_base" in name:
        return "SFT"
    if "qwen2.5-1.5b-instruct" in name:
        return "Baseline"
    if "multiround_r4" in name:
        return "DPO R4"
    if "no_real_teacher" in name or "no_real" in name:
        return "DPO LLM"
    if "human_bootstrap" in name or "bootstrap_rerun_r1" in name:
        return "DPO R1"

    return path.stem

    return path.stem

def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

records = []

jsonl_files = sorted(DATA_DIR.glob("*judge_scores*.jsonl"))
print(f"Found {len(jsonl_files)} judge-score files")

for path in jsonl_files:
    print(f"\nReading: {path.name}")
    rows = load_jsonl(path)
    if not rows:
        print("  -> empty, skipping")
        continue

    # inspect first row
    print("  Top-level columns:", list(rows[0].keys()))
    print("  judge_scores type:", type(rows[0].get("judge_scores")).__name__)

    metric_values = []

    for row in rows:
        js = row.get("judge_scores", None)
        if js is None:
            continue

        # Case 1: judge_scores is already a dict
        if isinstance(js, dict):
            val = js.get(METRIC, None)
            if val is not None:
                try:
                    metric_values.append(float(val))
                except (TypeError, ValueError):
                    pass

        # Case 2: judge_scores is a JSON string
        elif isinstance(js, str):
            try:
                parsed = json.loads(js)
                if isinstance(parsed, dict):
                    val = parsed.get(METRIC, None)
                    if val is not None:
                        metric_values.append(float(val))
            except json.JSONDecodeError:
                pass

    if not metric_values:
        print(f"  -> no {METRIC} values found, skipping")
        continue

    model = infer_model_name(path)

    records.append({
        "model": model,
        "file": path.name,
        "mean": float(np.mean(metric_values)),
        "variance": float(np.var(metric_values)),
        "std": float(np.std(metric_values)),
        "n": int(len(metric_values)),
    })

if not records:
    raise ValueError(f"No usable '{METRIC}' values found inside judge_scores.")

summary_df = pd.DataFrame(records)

print("\nRaw summary:")
print(summary_df[["model", "file", "mean", "variance", "std", "n"]])

grouped = (
    summary_df
    .groupby("model", as_index=False)
    .agg({
        "mean": "mean",
        "variance": "mean",
        "std": "mean",
        "n": "sum"
    })
)

print("\nGrouped summary:")
print(grouped)

plt.figure(figsize=(8, 6))

for _, row in grouped.iterrows():
    plt.scatter(row["mean"], row["variance"], s=90)
    plt.text(
        row["mean"],
        row["variance"],
        row["model"],
        fontsize=10,
        ha="left",
        va="bottom"
    )

plt.xlabel(f"Mean {METRIC}")
plt.ylabel("Variance")
plt.title(f"Mean vs Variance by Model ({METRIC})")
plt.grid(True)

out_path = OUT_DIR / f"mean_vs_variance_{METRIC}.png"
plt.savefig(out_path, bbox_inches="tight", dpi=200)
plt.show()

print(f"\nSaved plot to: {out_path}")