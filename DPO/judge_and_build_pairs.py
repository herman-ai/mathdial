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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of responses to judge per model.generate() call")
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


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading judge model: {args.judge_model}", flush=True)
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

    candidates = read_jsonl(args.input_file)
    total = len(candidates)
    print(f"\n=== Judging {total} candidate rows from {args.input_file} ===", flush=True)

    # --- Flatten all (row_idx, response_idx, judge_prompt, question, solution, prompt) ---
    # We judge every response in one flat stream of full batches, then reassemble.
    print("Flattening responses for batched judging...", flush=True)

    flat_items = []   # (row_idx, resp_idx, response, last_student_msg, question, solution)
    row_prompts = {}  # row_idx -> DPO prompt string
    row_responses = {}  # row_idx -> list of responses
    skipped_no_pairs = 0

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
    skipped_judge_fail = 0
    for k, (row_idx, resp_idx, _, _last, _q, _s) in enumerate(flat_items):
        s = avg_score(all_scores_flat[k])
        if s is None:
            skipped_judge_fail += 1
            print(f"[Warning] Judge returned None for row {row_idx} resp {resp_idx}. Skipping this response.", flush=True)
            continue
        row_score_map[row_idx][resp_idx] = s

    print(f"\n=== Building preference pairs and writing to {args.output_file} ===", flush=True)
    # --- Build pairs and write incrementally ---
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")
    total_pairs = 0

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
                out_f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
                out_f.flush()
                total_pairs += 1
                print(f"[Status] row {row_idx} | pairs so far: {total_pairs}", end="\r", flush=True)

        
    out_f.close()
    print(f"\n[Done] Wrote {total_pairs} preference pairs to {args.output_file}", flush=True)
    print(f"  Skipped (< 2 responses): {skipped_no_pairs}", flush=True)
    print(f"  Skipped (judge returned None): {skipped_judge_fail}", flush=True)
