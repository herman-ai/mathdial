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
from tqdm import tqdm

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

Respond ONLY with a JSON object in exactly this format (no extra text, no reasoning, no markdown):
{"socratic_guidance": <int 1-5>, "mathematical_accuracy": <int 1-5>, "relevance": <int 1-5>, "conciseness": <int 1-5>, "overall_quality": <int 1-5>}
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
    """Convert <EOM>- or |EOM|-delimited string to a readable numbered dialogue."""
    # MathDial original data uses |EOM|; generated conversations use <EOM>
    delimiter = "|EOM|" if "|EOM|" in raw else "<EOM>"
    turns = [t.strip() for t in raw.split(delimiter) if t.strip()]
    return "\n".join(f"[{i+1}] {turn}" for i, turn in enumerate(turns))


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model output."""
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text.strip())

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Greedy: find first '{' and last '}' to handle reasoning fields with braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model output:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Lightweight single-response judge (for pairwise DPO ranking)
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = """\
You are an expert mathematics education researcher. Score a single teacher response on five dimensions (integers 1–5).

1. Socratic guidance (1=gives away answer, 5=only hints/questions)
2. Mathematical accuracy (1=errors present, 5=fully correct)
3. Relevance (1=ignores student's mistake, 5=directly addresses it)
4. Conciseness (1=padded/off-topic, 5=tight focused feedback)
5. Overall quality (1=poor, 5=excellent)

Respond ONLY with JSON, no extra text:
{"socratic_guidance": <int 1-5>, "mathematical_accuracy": <int 1-5>, "relevance": <int 1-5>, "conciseness": <int 1-5>, "overall_quality": <int 1-5>}
"""

RESPONSE_USER_TEMPLATE = """\
## Math problem
{question}

## Student's incorrect solution
{incorrect_solution}

## Last student message
{last_student_message}

## Teacher response to score
{teacher_response}
"""


def judge_teacher_responses_batch(
    model,
    tokenizer,
    device,
    questions: list,
    incorrect_solutions: list,
    last_student_messages: list,
    teacher_responses: list,
    retries: int = 3,
) -> list:
    """Score a batch of individual teacher responses for pairwise DPO ranking.

    Much cheaper than judge_conversations_batch — only the last student message
    and the teacher response under evaluation are included in the prompt.

    Returns a list of score dicts (None values on parse failure).
    """
    prompts = []
    for question, incorrect_solution, last_student_msg, teacher_response in zip(
        questions, incorrect_solutions, last_student_messages, teacher_responses
    ):
        user_content = RESPONSE_USER_TEMPLATE.format(
            question=question,
            incorrect_solution=incorrect_solution,
            last_student_message=last_student_msg,
            teacher_response=teacher_response,
        )
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = original_padding_side

    prompt_lengths = inputs.input_ids.shape[1]

    for attempt in range(retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            results = []
            for i, seq in enumerate(output_ids):
                new_tokens = seq[prompt_lengths:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                try:
                    scores = extract_json(text)
                    for dim in DIMENSIONS:
                        if dim not in scores:
                            raise ValueError(f"Missing key '{dim}' in judge response")
                    results.append(scores)
                except Exception as e:
                    print(f"  [item {i}] Parse failed: {e}")
                    results.append({dim: None for dim in DIMENSIONS})
            return results

        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Batch generate failed: {e}")

    return [{dim: None for dim in DIMENSIONS} for _ in prompts]


def judge_conversation(
    model,
    tokenizer,
    device,
    question: str,
    incorrect_solution: str,
    conversation_raw: str,
    retries: int = 3,
) -> dict:
    """Run the judge model on a single conversation and return parsed scores dict."""
    results = judge_conversations_batch(
        model, tokenizer, device,
        questions=[question],
        incorrect_solutions=[incorrect_solution],
        conversation_raws=[conversation_raw],
        retries=retries,
    )
    return results[0]


def judge_conversations_batch(
    model,
    tokenizer,
    device,
    questions: list,
    incorrect_solutions: list,
    conversation_raws: list,
    retries: int = 3,
) -> list:
    """Run the judge model on a batch of conversations. Returns a list of score dicts."""
    prompts = []
    for question, incorrect_solution, conversation_raw in zip(questions, incorrect_solutions, conversation_raws):
        user_content = USER_TEMPLATE.format(
            question=question,
            incorrect_solution=incorrect_solution,
            conversation=format_conversation(conversation_raw),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Left-pad so all sequences end at the same position before generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = original_padding_side

    prompt_lengths = inputs.input_ids.shape[1]

    for attempt in range(retries):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            results = []
            for i, seq in enumerate(output_ids):
                new_tokens = seq[prompt_lengths:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                try:
                    scores = extract_json(text)
                    for dim in DIMENSIONS:
                        if dim not in scores:
                            raise ValueError(f"Missing key '{dim}' in judge response")
                    results.append(scores)
                except Exception as e:
                    print(f"  [item {i}] Parse failed: {e}")
                    results.append({dim: None for dim in DIMENSIONS} | {"reasoning": "JUDGE_FAILED"})
            return results

        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Batch generate failed: {e}")

    return [{dim: None for dim in DIMENSIONS} | {"reasoning": "JUDGE_FAILED"} for _ in prompts]


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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of conversations to judge in one model.generate() call.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Workaround for torchvision conflicts in containers
    sys.path = [p for p in sys.path if 'dist-packages' not in p] + \
               [p for p in sys.path if 'dist-packages' in p]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading judge model: {args.judge_model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",   # spreads across all available GPUs automatically
    )
    model.eval()
    print("Judge model loaded.")

    data = read_jsonl(args.input_file)
    results = []

    # Collect valid problems first
    valid_problems = []
    for i, problem in enumerate(data):
        conversation_raw = problem.get(args.model_name)
        if not conversation_raw:
            print(f"SKIP problem {i+1} (key '{args.model_name}' not found)")
            continue
        valid_problems.append((i, problem))

    # Process in batches
    for batch_start in tqdm(range(0, len(valid_problems), args.batch_size), desc="Judging", unit="batch"):
        batch = valid_problems[batch_start:batch_start + args.batch_size]
        print(f"Judging problems {batch_start+1}–{batch_start+len(batch)} / {len(valid_problems)} ...", flush=True)

        batch_scores = judge_conversations_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            questions=[p.get("question", "") for _, p in batch],
            incorrect_solutions=[p.get("student_incorrect_solution", "") for _, p in batch],
            conversation_raws=[p.get(args.model_name) for _, p in batch],
        )

        for (i, problem), scores in zip(batch, batch_scores):
            qid = problem.get("qid", str(i))
            print(f"  qid={qid}: ", {dim: scores.get(dim) for dim in DIMENSIONS})
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
