"""
Step 2 of 2: Judge candidate responses and build preference pairs.

Reads the JSONL output from generate_candidate_responses.py, runs the LLM judge
on each response, and builds pairwise (chosen, rejected) preference pairs.

Input format (from generate_candidate_responses.py):
{
  "prompt": "...",
  "question": "...",
  "solution": "...",
  "conversation_context": "...",
  "responses": ["r0", "r1", ...]
}

Output format (DPO-ready):
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}

Usage:
  python DPO/judge_and_build_pairs.py \
      --input_file  data/candidate_responses/train_candidates.jsonl \
      --output_file data/preference-data-no-real-teacher/train.jsonl \
      --judge_model mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --batch_size 4
"""

import json
import os
import sys
import argparse
import torch
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactivetutoring"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from llm_judge import judge_teacher_responses_batch, DIMENSIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL file from generate_candidate_responses.py")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL path for preference pairs")
    parser.add_argument("--judge_model", type=str,
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--pair_mode", type=str, default="judge", choices=["judge", "expert"],
                        help="'judge': rank model responses with LLM judge; 'expert': pair expert_response > each model response")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of responses to judge per model.generate() call")
    parser.add_argument("--replay_pairs_file", type=str, default="",
                        help="Optional JSONL of past-round preference pairs for replay mixing")
    parser.add_argument("--intermodel_pairs_file", type=str, default="",
                        help="Optional JSONL of inter-model preference pairs for mixing")
    parser.add_argument("--expert_frac", type=float, default=1.0,
                        help="Fraction of current-round (expert-vs-current) pairs to keep")
    parser.add_argument("--replay_frac", type=float, default=0.0,
                        help="Fraction of replay pairs to add (relative to current-round pair count)")
    parser.add_argument("--intermodel_frac", type=float, default=0.0,
                        help="Fraction of intermodel pairs to add (relative to current-round pair count)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pair mixing subsampling")
    parser.add_argument("--max_expert_demos", type=int, default=0,
                        help="If >0 and pair_mode=expert, use at most this many expert demonstration rows")
    return parser.parse_args()


def avg_score(score):
    try:
        return (
            score["overall_quality"] * 2 +
            score["socratic_guidance"] * 1.5 +
            score["mathematical_accuracy"] +
            score["relevance"] +
            score["conciseness"]
        ) / 6.5
    except (TypeError, KeyError):
        return None


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_pref_pairs(path):
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[Warning] replay/intermodel file not found: {path}", flush=True)
        return []
    rows = read_jsonl(path)
    out = []
    for r in rows:
        if all(k in r for k in ("prompt", "chosen", "rejected")):
            out.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    return out


def sample_rows(rows, n, rng):
    if n <= 0 or not rows:
        return []
    n = min(n, len(rows))
    if n == len(rows):
        return list(rows)
    idxs = rng.sample(range(len(rows)), n)
    return [rows[i] for i in idxs]


if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = read_jsonl(args.input_file)
    total = len(candidates)
    current_pairs = []
    skipped_no_pairs = 0
    skipped_judge_fail = 0

    if args.pair_mode == "expert":
        print(f"\n=== Building expert-vs-model pairs from {total} candidate rows ===", flush=True)
        expert_row_indices = [
            idx for idx, row in enumerate(candidates)
            if (row.get("expert_response") or "").strip() and row.get("responses")
        ]
        if args.max_expert_demos > 0 and len(expert_row_indices) > args.max_expert_demos:
            selected_indices = set(rng.sample(expert_row_indices, args.max_expert_demos))
            print(
                f"[Expert] Using {args.max_expert_demos}/{len(expert_row_indices)} expert demonstrations "
                f"(seed={args.seed})",
                flush=True,
            )
        else:
            selected_indices = set(expert_row_indices)
            print(
                f"[Expert] Using all {len(selected_indices)} available expert demonstrations",
                flush=True,
            )

        missing_expert = 0
        for row_idx, row in enumerate(candidates):
            if row_idx not in selected_indices:
                continue
            prompt = row.get("prompt", "")
            expert = (row.get("expert_response") or "").strip()
            responses = row.get("responses", [])
            if not expert:
                missing_expert += 1
                skipped_no_pairs += 1
                continue
            if not responses:
                skipped_no_pairs += 1
                continue
            for response in responses:
                rejected = (response or "").strip()
                if not rejected or rejected == expert:
                    print(f"[Warning] Skipping empty or identical response for row {row_idx}, response: {rejected}, expert: {expert}", flush=True)
                    continue
                current_pairs.append({"prompt": prompt, "chosen": expert, "rejected": rejected})
            if row_idx % 200 == 0:
                print(f"[Status] processed {row_idx}/{total} rows | pairs so far: {len(current_pairs)}", flush=True)
        if missing_expert > 0:
            print(f"[Warning] Missing expert_response in {missing_expert} rows (did you regenerate candidates with updated script?)", flush=True)
    else:
        print(f"\n=== Judging {total} candidate rows from {args.input_file} ===", flush=True)
        print("Loading judge model...", flush=True)
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        judge_model.eval()
        print("Judge model loaded.", flush=True)

        # --- Flatten all (row_idx, response_idx, judge_prompt, question, solution, prompt) ---
        # We judge every response in one flat stream of full batches, then reassemble.
        print("Flattening responses for batched judging...", flush=True)

        flat_items = []   # (row_idx, resp_idx, response, last_student_msg, question, solution)
        row_prompts = {}  # row_idx -> DPO prompt string
        row_responses = {}  # row_idx -> list of responses

        def extract_last_student_message(conversation_context: str) -> str:
            """Return the last Student: line from a str(history) formatted string."""
            last = ""
            for line in conversation_context.splitlines():
                line = line.strip()
                if line.lower().startswith("student:"):
                    last = line.split(":", 1)[1].strip()
            return last

        for row_idx, row in enumerate(candidates):
            if len(row["responses"]) < 2:
                skipped_no_pairs += 1
                continue
            row_prompts[row_idx] = row["prompt"]
            row_responses[row_idx] = row["responses"]
            last_student_msg = extract_last_student_message(row["conversation_context"])
            for resp_idx, response in enumerate(row["responses"]):
                flat_items.append((row_idx, resp_idx, response, last_student_msg, row["question"], row["solution"]))

        print(f"Total responses to judge: {len(flat_items)} across {len(row_prompts)} rows", flush=True)

        # --- Judge all responses in uniform batches ---
        all_scores_flat = [None] * len(flat_items)

        for batch_start in tqdm(range(0, len(flat_items), args.batch_size), desc="Judging batches", unit="batch"):
            batch = flat_items[batch_start:batch_start + args.batch_size]
            _, _, responses_b, last_student_msgs_b, questions_b, solutions_b = zip(*batch)
            scores = judge_teacher_responses_batch(
                judge_model, judge_tokenizer, device,
                questions=list(questions_b),
                incorrect_solutions=list(solutions_b),
                last_student_messages=list(last_student_msgs_b),
                teacher_responses=list(responses_b),
            )
            for k, score in enumerate(scores):
                all_scores_flat[batch_start + k] = score

            if (batch_start // args.batch_size) % 50 == 0:
                done = batch_start + len(batch)
                print(f"[Status] judged {done}/{len(flat_items)} responses", flush=True)

        # --- Reassemble scores per row ---
        # row_scores[row_idx][resp_idx] = avg_score
        from collections import defaultdict
        row_score_map = defaultdict(dict)
        for k, (row_idx, resp_idx, _, _last, _q, _s) in enumerate(flat_items):
            s = avg_score(all_scores_flat[k])
            if s is None:
                skipped_judge_fail += 1
                print(f"[Warning] Judge returned None for row {row_idx} resp {resp_idx}. Skipping this response.", flush=True)
                continue
            row_score_map[row_idx][resp_idx] = s

        print(f"\n=== Building judged preference pairs ===", flush=True)
        for row_idx in sorted(row_score_map.keys()):
            prompt = row_prompts[row_idx]
            responses = row_responses[row_idx]
            avg_scores = [row_score_map[row_idx].get(j) for j in range(len(responses))]
            print(f"[Status] Building pairs for row {row_idx} with {len(responses)} responses", flush=True)
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    if responses[i] == responses[j]:
                        continue
                    if avg_scores[i] is None or avg_scores[j] is None:
                        continue
                    if avg_scores[i] > avg_scores[j]:
                        chosen, rejected = responses[i], responses[j]
                    elif avg_scores[j] > avg_scores[i]:
                        chosen, rejected = responses[j], responses[i]
                    else:
                        continue  # equal scores — skip to avoid noise
                    current_pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    n_current = len(current_pairs)
    keep_current = int(n_current * args.expert_frac)
    kept_current_pairs = sample_rows(current_pairs, keep_current, rng)
    mixed_pairs = list(kept_current_pairs)

    replay_pairs = read_pref_pairs(args.replay_pairs_file)
    intermodel_pairs = read_pref_pairs(args.intermodel_pairs_file)

    replay_take = int(n_current * args.replay_frac)
    intermodel_take = int(n_current * args.intermodel_frac)
    mixed_pairs.extend(sample_rows(replay_pairs, replay_take, rng))
    mixed_pairs.extend(sample_rows(intermodel_pairs, intermodel_take, rng))

    print(
        f"Mix summary | current={n_current} keep={len(kept_current_pairs)} "
        f"replay_pool={len(replay_pairs)} replay_take={min(replay_take, len(replay_pairs))} "
        f"intermodel_pool={len(intermodel_pairs)} intermodel_take={min(intermodel_take, len(intermodel_pairs))}",
        flush=True,
    )

    write_jsonl(args.output_file, mixed_pairs)
    print(f"\n[Done] Wrote {len(mixed_pairs)} preference pairs to {args.output_file}", flush=True)
    print(f"  Skipped (< 2 responses): {skipped_no_pairs}", flush=True)
    print(f"  Skipped (judge returned None): {skipped_judge_fail}", flush=True)
