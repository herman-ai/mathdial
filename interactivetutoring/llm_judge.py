"""
LLM-as-a-Judge evaluation for the teacher model.

For each conversation in the input JSONL, loads a local Qwen model and asks it
to score the teacher on five pedagogical dimensions:

  1. Socratic guidance   – hints/questions rather than giving away the answer
  2. Mathematical accuracy – are the teacher's corrections factually correct?
  3. Relevance            – does the teacher address the student's specific mistake?
  4. Conciseness          – focused feedback without unnecessary padding/praise
  5. Overall quality      – holistic tutoring effectiveness

Scores are integers 1–5.  Per-dimension means and a combined mean are printed at
the end.  Full per-problem results are written to --export_file.

Usage:
  python llm_judge.py \
      --input_file  ../output/qwen_output.jsonl \
      --model_name  qwen_baseline \
      --judge_model Qwen/Qwen2.5-72B-Instruct \
      --export_file ../output/qwen_judge_scores.jsonl
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import read_jsonl


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert mathematics education researcher evaluating the quality of an
AI math tutor.  You will be given:
  - The math problem being discussed.
  - The student's incorrect solution (so you know what mistake was made).
  - The full tutoring conversation.

Score the TEACHER'S responses on five dimensions using integers 1–5:

1. Socratic guidance (1=just gives the answer, 5=only guiding questions/hints)
2. Mathematical accuracy (1=contains errors, 5=all corrections are correct)
3. Relevance (1=ignores the actual mistake, 5=directly addresses the student's error)
4. Conciseness (1=lots of padding/praise/off-topic text, 5=tight focused feedback)
5. Overall quality (1=poor tutor, 5=excellent tutor)

Respond ONLY with a JSON object in exactly this format (no extra text):
{
  "socratic_guidance": <int 1-5>,
  "mathematical_accuracy": <int 1-5>,
  "relevance": <int 1-5>,
  "conciseness": <int 1-5>,
  "overall_quality": <int 1-5>,
  "reasoning": "<one short paragraph explaining the scores>"
}
"""

USER_TEMPLATE = """\
## Math problem
{question}

## Student's incorrect solution
{incorrect_solution}

## Tutoring conversation
{conversation}
"""


DIMENSIONS = [
    "socratic_guidance",
    "mathematical_accuracy",
    "relevance",
    "conciseness",
    "overall_quality",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_conversation(raw: str) -> str:
    """Convert <EOM>-delimited string to a readable numbered dialogue."""
    turns = [t.strip() for t in raw.split("<EOM>") if t.strip()]
    return "\n".join(f"[{i+1}] {turn}" for i, turn in enumerate(turns))


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model output."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Fall back to finding the first {...} block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in model output:\n{text[:500]}")


def judge_conversation(
    model,
    tokenizer,
    device,
    question: str,
    incorrect_solution: str,
    conversation_raw: str,
    retries: int = 3,
) -> dict:
    """Run the judge model locally and return parsed scores dict."""
    user_content = USER_TEMPLATE.format(
        question=question,
        incorrect_solution=incorrect_solution,
        conversation=format_conversation(conversation_raw),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    for attempt in range(retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,          # greedy for reproducibility
                    temperature=1.0,          # ignored when do_sample=False
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][len(inputs.input_ids[0]):]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            scores = extract_json(text)

            for dim in DIMENSIONS:
                if dim not in scores:
                    raise ValueError(f"Missing key '{dim}' in judge response")
            return scores

        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Judge failed: {e}")

    return {dim: None for dim in DIMENSIONS} | {"reasoning": "JUDGE_FAILED"}


def aggregate_scores(results: list[dict]) -> dict:
    """Compute per-dimension and combined mean, ignoring None values."""
    totals = {dim: [] for dim in DIMENSIONS}
    for r in results:
        scores = r.get("judge_scores", {})
        for dim in DIMENSIONS:
            v = scores.get(dim)
            if v is not None:
                totals[dim].append(v)

    means = {}
    for dim, vals in totals.items():
        means[dim] = round(sum(vals) / len(vals), 3) if vals else None

    scored = [v for v in means.values() if v is not None]
    means["combined_mean"] = round(sum(scored) / len(scored), 3) if scored else None
    return means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for a teacher model."
    )
    parser.add_argument("--input_file",  type=str,
                        default="../output/qwen_output.jsonl")
    parser.add_argument("--model_name",  type=str,
                        default="qwen_baseline",
                        help="Key in the JSONL record that holds the conversation.")
    parser.add_argument("--judge_model", type=str,
                        default="Qwen/Qwen2.5-72B-Instruct",
                        help="HuggingFace model ID or local path to use as judge.")
    parser.add_argument("--export_file", type=str,
                        default="../output/judge_scores.jsonl",
                        help="Where to write per-problem scores.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Workaround for torchvision conflicts in containers
    sys.path = [p for p in sys.path if 'dist-packages' not in p] + \
               [p for p in sys.path if 'dist-packages' in p]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading judge model: {args.judge_model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",   # spreads across all available GPUs automatically
    )
    model.eval()
    print("Judge model loaded.")

    data = read_jsonl(args.input_file)
    results = []

    for i, problem in enumerate(data):
        qid = problem.get("qid", str(i))
        print(f"Judging problem {i+1} (qid={qid}) ...", end=" ", flush=True)

        conversation_raw = problem.get(args.model_name)
        if not conversation_raw:
            print(f"SKIP (key '{args.model_name}' not found)")
            continue

        scores = judge_conversation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=problem.get("question", ""),
            incorrect_solution=problem.get("student_incorrect_solution", ""),
            conversation_raw=conversation_raw,
        )

        print({dim: scores.get(dim) for dim in DIMENSIONS})

        results.append({
            "qid": qid,
            "question": problem.get("question", ""),
            "model_name": args.model_name,
            "judge_model": args.judge_model,
            "judge_scores": scores,
        })

    # Write per-problem results
    os.makedirs(os.path.dirname(os.path.abspath(args.export_file)), exist_ok=True)
    with open(args.export_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-problem scores written to {args.export_file}")

    # Print aggregate summary
    means = aggregate_scores(results)
    print("\n=== Aggregate scores ===")
    for dim in DIMENSIONS:
        print(f"  {dim:<25}: {means[dim]}")
    print(f"  {'combined_mean':<25}: {means['combined_mean']}")
