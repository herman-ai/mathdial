import json
import argparse
from pathlib import Path
from collections import defaultdict


# ── Weights for each numeric score dimension ────────────────────────────────
# Adjust these to reflect how important each criterion is to you.
WEIGHTS = {
    "socratic_guidance":   1.5,
    "mathematical_accuracy": 1,
    "relevance":           1,
    "conciseness":         1,
    "overall_quality":     2,
}

scores_file_path = '../output/dpo/qwen_judge_scores_dpo_human_preferences.jsonl'
outputs_file_path = '../output/dpo/qwen_output_dpo_human_preferences.jsonl'

# Make sure weights sum to 1.0 (sanity-check only – script will normalise)
_total_weight = sum(WEIGHTS.values())
WEIGHTS = {k: v / _total_weight for k, v in WEIGHTS.items()}


def load_jsonl(path: str) -> list[dict]:
    """Load every line of a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed JSON on line {lineno}: {exc}")
    return records


def weighted_average(scores: dict) -> float:
    """
    Given a judge_scores dict, compute the weighted average of the
    numeric fields defined in WEIGHTS.  Non-numeric fields (e.g. 'reasoning')
    are silently ignored.
    """
    total, weight_used = 0.0, 0.0
    for key, weight in WEIGHTS.items():
        value = scores.get(key)
        if isinstance(value, (int, float)):
            total += weight * value
            weight_used += weight
    if weight_used == 0:
        return float("nan")
    # Re-normalise in case some keys were missing
    return total / weight_used


def main(judge_path: str, outputs_path: str, out_path: str) -> None:
    print(f"Loading judge file  : {judge_path}")
    judge_records = load_jsonl(judge_path)
    print(f"  → {len(judge_records)} entries loaded")

    print(f"Loading outputs file: {outputs_path}")
    output_records = load_jsonl(outputs_path)
    print(f"  → {len(output_records)} entries loaded")

    # ── Index the outputs file by (qid, scenario) so we can look them up ──
    # Key: (qid, scenario) → record
    # Also keep a plain qid → [records] fallback for files without 'scenario'
    outputs_by_qid_scenario: dict[tuple, dict] = {}
    outputs_by_qid: dict[int, list[dict]] = defaultdict(list)

    for rec in output_records:
        qid = rec.get("qid")
        scenario = rec.get("scenario")
        if qid is not None:
            outputs_by_qid[qid].append(rec)
            if scenario is not None:
                outputs_by_qid_scenario[(qid, scenario)] = rec

    # ── Process each judge entry ────────────────────────────────────────────
    results = []
    unmatched = 0

    for judge in judge_records:
        qid        = judge.get("qid")
        model_name = judge.get("model_name", "")
        scores     = judge.get("judge_scores", {})
        wa         = weighted_average(scores)

        # Find the matching output record
        scenario   = None          # judge file may not carry scenario
        output_rec = None

        # Try exact (qid, scenario) match first, then fall back to qid only
        if (qid, scenario) in outputs_by_qid_scenario:
            output_rec = outputs_by_qid_scenario[(qid, scenario)]
        elif qid in outputs_by_qid:
            candidates = outputs_by_qid[qid]
            # If there is only one candidate, use it
            if len(candidates) == 1:
                output_rec = candidates[0]
            else:
                # Multiple scenarios: pick the first (or adjust logic here)
                output_rec = candidates[0]

        if output_rec is None:
            unmatched += 1

        # Pull the model's output text from the output record
        model_output = None
        if output_rec is not None:
            model_output = output_rec.get(model_name)  # keyed by model name
            if model_output is None:
                # Fallback: look for any field that isn't a known metadata key
                KNOWN_KEYS = {
                    "qid", "scenario", "question", "ground_truth",
                    "student_incorrect_solution", "student_profile",
                    "teacher_described_confusion", "self-correctness",
                    "self-typical-confusion", "self-typical-interactions",
                    "conversation",
                }
                for k, v in output_rec.items():
                    if k not in KNOWN_KEYS and isinstance(v, str):
                        model_output = v
                        break

        result = {
            "qid":             qid,
            "model_name":      model_name,
            "judge_model":     judge.get("judge_model"),
            "raw_scores":      {k: v for k, v in scores.items()
                                if isinstance(v, (int, float))},
            "reasoning":       scores.get("reasoning"),
            "weighted_average": round(wa, 4),
            "model_output":    model_output,
            "question":        judge.get("question"),
        }
        if output_rec is not None:
            result["scenario"]       = output_rec.get("scenario")
            result["ground_truth"]   = output_rec.get("ground_truth")

        results.append(result)

    # ── Sort by weighted average ascending (worst first) ───────────────────
    results.sort(key=lambda r: r["weighted_average"])

    bottom_10 = results[:10]

    # ── Summary stats across all results ───────────────────────────────────
    was = [r["weighted_average"] for r in results
           if r["weighted_average"] == r["weighted_average"]]  # exclude NaN

    # ── Write human-readable report ─────────────────────────────────────────
    DIVIDER      = "=" * 80
    THIN_DIVIDER = "-" * 80

    with open(out_path, "w", encoding="utf-8") as fh:
        # ── Header ──────────────────────────────────────────────────────────
        fh.write(DIVIDER + "\n")
        fh.write("  LOWEST-SCORING RESPONSES  —  Bottom 10 by Weighted Average\n")
        fh.write(DIVIDER + "\n")
        if was:
            fh.write(f"  Overall stats across {len(results)} entries:\n")
            fh.write(f"    Min  : {min(was):.4f}   "
                     f"Max : {max(was):.4f}   "
                     f"Mean: {sum(was)/len(was):.4f}\n")
        fh.write(f"  Score weights: " +
                 ", ".join(f"{k} {v*100:.0f}%" for k, v in WEIGHTS.items()) + "\n")
        fh.write(DIVIDER + "\n\n")

        for rank, rec in enumerate(bottom_10, 1):
            fh.write(f"{'#'+str(rank):<4}  QID: {rec['qid']}   "
                     f"Weighted Average: {rec['weighted_average']:.4f}\n")
            fh.write(THIN_DIVIDER + "\n")

            # ── Question ────────────────────────────────────────────────────
            fh.write("QUESTION:\n")
            fh.write(f"  {rec.get('question') or '(not available)'}\n\n")

            # ── Ground truth ────────────────────────────────────────────────
            gt = rec.get("ground_truth")
            if gt:
                fh.write("GROUND TRUTH:\n")
                fh.write(f"  {gt.strip()}\n\n")

            # ── Score breakdown ─────────────────────────────────────────────
            fh.write("SCORES:\n")
            raw = rec.get("raw_scores", {})
            for dim, weight in WEIGHTS.items():
                val = raw.get(dim, "N/A")
                bar = ("█" * int(val)) + ("░" * (5 - int(val))) if isinstance(val, (int, float)) else "     "
                fh.write(f"  {dim:<25}  {bar}  {val}/5"
                         f"  (weight {weight*100:.0f}%)\n")
            fh.write(f"\n  {'Weighted Average':<25}  {rec['weighted_average']:.4f} / 5.0\n\n")

            # ── Judge reasoning ─────────────────────────────────────────────
            reasoning = rec.get("reasoning")
            if reasoning:
                fh.write("JUDGE REASONING:\n")
                # Wrap long reasoning text at 76 chars
                words, line = reasoning.split(), ""
                for word in words:
                    if len(line) + len(word) + 1 > 76:
                        fh.write(f"  {line}\n")
                        line = word
                    else:
                        line = (line + " " + word).strip()
                if line:
                    fh.write(f"  {line}\n")
                fh.write("\n")

            # ── Model output ────────────────────────────────────────────────
            model_output = rec.get("model_output")
            fh.write(f"MODEL OUTPUT  ({rec.get('model_name', 'unknown')}):\n")
            if model_output:
                # Format <EOM> markers as turn separators
                turns = model_output.replace("<EOM>", "\n  ---\n")
                for line in turns.splitlines():
                    fh.write(f"  {line}\n")
            else:
                fh.write("  (no output found)\n")
            fh.write("\n")

            fh.write(f"  Judge model : {rec.get('judge_model', 'unknown')}\n")
            if rec.get("scenario") is not None:
                fh.write(f"  Scenario    : {rec['scenario']}\n")
            fh.write("\n" + DIVIDER + "\n\n")

    print(f"\nDone.")
    print(f"  Report written to  : {out_path}")
    print(f"  Total processed    : {len(results)}")
    print(f"  Unmatched qids     : {unmatched}")
    if was:
        print(f"\n  Weighted-average stats:")
        print(f"    Min  : {min(was):.4f}")
        print(f"    Max  : {max(was):.4f}")
        print(f"    Mean : {sum(was)/len(was):.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(scores_file_path, outputs_file_path, '../analyses/texts/examples.txt')