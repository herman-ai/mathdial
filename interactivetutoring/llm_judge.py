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

# SYSTEM_PROMPT = """\
# You are an expert mathematics education researcher evaluating the quality of an
# AI math tutor.  You will be given:
#   - The math problem being discussed.
#   - The student's incorrect solution (so you know what mistake was made).
#   - The full tutoring conversation.

# Score the TEACHER'S responses on five dimensions using integers 1–5:

# 1. Socratic guidance (1=just gives the answer, 5=only guiding questions/hints)
# 2. Mathematical accuracy (1=contains errors, 5=all corrections are correct)
# 3. Relevance (1=ignores the actual mistake, 5=directly addresses the student's error)
# 4. Conciseness (1=lots of padding/praise/off-topic text, 5=tight focused feedback)
# 5. Overall quality (1=poor tutor, 5=excellent tutor)

# Respond ONLY with a JSON object in exactly this format (no extra text, no reasoning, no markdown):
# {"socratic_guidance": <int 1-5>, "mathematical_accuracy": <int 1-5>, "relevance": <int 1-5>, "conciseness": <int 1-5>, "overall_quality": <int 1-5>}
# """

SYSTEM_PROMPT = """\
You are an expert mathematics education researcher evaluating the quality of an
AI math tutor.  You will be given:
  - The math problem being discussed.
  - The student's incorrect solution (so you know what mistake was made).
  - The full tutoring conversation.

Your task is to evaluate only the TEACHER'S utterances across the full tutoring
conversation. Use the student's responses only as context for understanding the student's misconception and whether the teacher addresses it.

Score the TEACHER'S responses in the conversation on five dimensions using integers 1–5.

Use this scale:
1 = very poor
2 = weak
3 = mixed / partially helpful
4 = good
5 = excellent


A score of 5 should be reserved only for responses that fully satisfy the
definition of the metric with no major weaknesses. If a response is good but missing something important, it should receive a 4
rather than a 5.

1. Socratic guidance
How well the teacher helps the student reason toward the solution rather than
simply giving the answer away.

1 = mostly gives away the answer or key reasoning steps
3 = mixes explanation with some useful guiding questions
5 = primarily uses productive hints or questions that help the student reason

High Socratic guidance requires that the questions move the student toward
correcting their specific misconception. Generic or vague questions should
not receive high scores.

2. Mathematical accuracy
Whether the teacher's explanations, hints, and corrections are mathematically correct.

1 = contains clear mathematical errors or misleading statements
3 = mostly correct but includes some unclear or potentially confusing reasoning
5 = fully mathematically correct throughout

If the teacher introduces an incorrect intermediate claim, this score should be lower.

3. Relevance
Whether the teacher addresses the student's actual mistake.

1 = ignores the student's misconception
3 = partially related but not well targeted to the student's mistake
5 = directly addresses the student's specific error or misunderstanding

Do not give a high relevance score just because the teacher is generally talking
about the same topic. The teacher should address the specific mistake.

4. Conciseness
Whether the teacher provides focused guidance without unnecessary padding,
praise, repetition, or off-topic content.

1 = very repetitive, padded, or off-topic
3 = somewhat wordy or repetitive but still useful
5 = concise and focused

Praise or friendly language alone should not increase the score.

5. Overall quality
Overall pedagogical quality of the teacher's responses across the conversation.

Consider whether the teacher:
- identifies the student's misunderstanding,
- helps the student make progress toward the correct reasoning,
- avoids introducing confusion,
- and provides effective tutoring overall.

A mathematically correct but overly revealing response should not receive the
highest overall score.
A question-based but vague or misleading response should also not receive the
highest overall score.

---

Below are examples showing how tutoring conversations should be scored.
Each example includes an explanation of why the scores were assigned.

Example 1

Math problem
Jason drives past 3 convenience stores on his way to work. The distance between the first store and the second store is 6 miles. The distance between the second store and third store is 2/3 longer than the distance between the first two stores. The distance from Jason's house to the first store and from the third store to work is 4 miles each. How many miles does Jason drive to work?

Student incorrect solution
The distance between the second store and third store is 2/3 x 6 = 4 miles longer than the distance between the first two stores, so it is 6 + 4 = 10 miles.
The total distance Jason travels between stores is 6 + 10 = 16 miles.
The total distance from his house to work is 16 + 4 + 4 = 24 miles.
Therefore, Jason drives 24 miles to work.

Tutoring conversation
Teacher: What is the distance between the first and second stores?
Student: The distance between the first and second stores is 6 miles.
Teacher: What is the distance between the 2nd and 3rd stores?
Student: The distance between the second and third stores is 2/3 x 6 = 4 miles longer than the distance between the first two stores, so it is 6 + 4 = 10 miles.
Teacher: What is the distance between his house and the first store?
Student: The distance between his house and the first store is 4 miles.
Teacher: What is the distance between the last (3rd) store and work?
Student: The distance between the last (3rd) store and work is also 4 miles.
Teacher: What is the total of these 4 distances?
Student: The total of these 4 distances is 6 + 10 + 4 + 4 = 24 miles.
Teacher: Perfect!

Explanation of scores

Socratic guidance = 4
The teacher mostly uses targeted questions that walk the student through the needed quantities step by step rather than directly giving the answer. This is not a 5 because the questions mainly prompt recall and decomposition of known quantities rather than deeper conceptual reasoning.

Mathematical accuracy = 5
All teacher prompts and implied reasoning are mathematically correct.

Relevance = 5
Each teacher question directly targets the quantities and steps needed to solve the student's specific error in the distance calculation.

Conciseness = 5
The teacher responses are short, focused, and free of unnecessary padding or repetition.

Overall quality = 5
The teacher effectively decomposes the problem into manageable steps and helps the student reconstruct the correct solution path.

Scores
{"socratic_guidance": 4, "mathematical_accuracy": 5, "relevance": 5, "conciseness": 5, "overall_quality": 5}

---

Example 2

Math problem
Nancy is filling an aquarium for her fish. She fills it halfway and goes to answer the door. While she's gone, her cat knocks the aquarium over and spills half the water in it. Then Nancy comes back and triples the amount of water in the aquarium. If the aquarium is 4 feet long, 6 feet wide, and 3 feet high, how many cubic feet of water are in the aquarium?

Student incorrect solution
The aquarium has a volume of 4 x 6 x 3 = 72 cubic feet.
When Nancy fills it halfway, she fills it with 72/2 = 36 cubic feet of water.
When the cat spills half of that, there are 36/2 = 18 cubic feet of water left.
When Nancy triples that amount, she adds 18 x 3 = 54 cubic feet of water.
The aquarium now has 18 + 54 = 72 cubic feet of water.
72

Tutoring conversation
Teacher: Steven, If you had 4 of something and tripled that amount, how much would you have?
Student: I would have 12 of something.
Teacher: So if Nancy triples the 18 cubic feet of water, how much would she have?
Student: She would have 54 cubic feet of water.
Teacher: Exactly correct!

Explanation of scores

Socratic guidance = 3
The teacher uses guiding questions, but the interaction is short and fairly direct. It is not a 4 or 5 because the teacher does not deeply probe the student's misconception or build much reasoning beyond the key multiplication step.

Mathematical accuracy = 5
The teacher's prompts and confirmation are mathematically correct.

Relevance = 4
The teacher addresses the student's incorrect final step, but the correction is somewhat narrow and does not fully unpack the broader misunderstanding.

Conciseness = 5
The teacher responses are very brief and focused.

Overall quality = 4
The tutoring is effective and correct, but somewhat shallow because it quickly steers the student to the answer rather than more fully developing the reasoning.

Scores
{"socratic_guidance": 3, "mathematical_accuracy": 5, "relevance": 4, "conciseness": 5, "overall_quality": 4}

---

Example 3

Math problem
John is very unfit and decides to work up to doing a push-up. He trains 5 days a week for them and starts with wall push-ups. He adds 1 rep a day and once he gets to 15 reps he works his way up to high elevation push-ups, then low elevation push-ups, and finally full push-ups, doing one more rep a day at each step. If he starts with 1 wall push-up on Monday, how many weeks will it take him to get to 15 full push-ups?

Student incorrect solution
To get to 15 reps, John will take 15 - 1 = 14 days.
To get to high elevation push-ups, he will take 15 + (15-1) = 29 days.
To get to low elevation push-ups, he will take 29 + (15-1) = 43 days.
To get to full push-ups, he will take 43 + (15-1) = 57 days.
Since he trains 5 days a week, it will take him 57/5 = 11.4 weeks to get to 15 full push-ups.
11.4

Tutoring conversation
Teacher: Stephanie, How many days will it take to do one step, if John adds one pushup per day?
Student: It will take one day to do one step, since John adds one pushup per day.
Teacher: So how many days will it take to do 15 wall push-ups?
Student: It will take 15 days to do 15 wall push-ups, since he adds one pushup per day.
Teacher: How many days to get to 15 high elevation pus-ups ?
Student: It will take 29 days to get to 15 high elevation push-ups, since he adds one pushup per day and it takes 14 days to get to 15 wall push-ups.
Teacher: Re-calculate, you jusy said iy took 15 days to get to 15 wall pushups, which is correct.
Student: It will take 29 days to get to 15 high elevation push-ups, since he adds one pushup per day and it takes 14 days to get to 15 wall push-ups.
Teacher: If it took 15 days to get to 15 wall push-ups, how many days would it take just to get to 15 high elevation pushups?
Student: It would take 14 days to get to 15 high elevation push-ups.
Teacher: You just changed the number of days for wall push ups again, 14 is NOT correct.
Student: It would take 15 days to get to 15 high elevation push-ups.
Teacher: 14 x 1 will never equal 15.
Student: It would take 14 days to get to 15 high elevation push-ups.
Teacher: Adding 1 fourteen times doesn't equal 15.
Student: It would take 15 days to get to 15 high elevation push-ups.
Teacher: Does 15 plus 15 equal 30?
Student: Yes, 15 plus 15 equals 30.
Teacher: You just said 15 + 15 = 29!
Student: I'm sorry, I meant to say that 15 + 15 equals 30.

Explanation of scores

Socratic guidance = 2
The teacher asks some questions, but much of the interaction consists of repeated correction and telling rather than productive scaffolding. This is not a 3 because the teacher does not consistently guide the student toward reconstructing the full progression logic.

Mathematical accuracy = 5
The teacher's arithmetic corrections are mathematically correct.

Relevance = 3
The teacher stays related to the student's error, but often focuses on local arithmetic mistakes instead of clearly rebuilding the full progression logic. This is not a 4 or 5 because the deeper misconception is not addressed cleanly.

Conciseness = 2
The teacher is repetitive and the interaction becomes inefficient. This is not a 1 because the conversation is still on-topic and attempts to correct the student's reasoning.

Overall quality = 2
The tutoring does not smoothly help the student reconstruct the reasoning and becomes stuck in repeated correction. It is weak overall even though the teacher's arithmetic statements are correct.

Scores
{"socratic_guidance": 2, "mathematical_accuracy": 5, "relevance": 3, "conciseness": 2, "overall_quality": 2}

---

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
                    print(f"  [item {i}] Raw judge output:\n{text if text else '<EMPTY_OUTPUT>'}")
                    results.append({dim: None for dim in DIMENSIONS})
            return results

        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Batch generate failed: {e}")
            print("  Raw judge output unavailable: generation failed before decode.")

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
                    print(f"  [item {i}] Raw judge output:\n{text if text else '<EMPTY_OUTPUT>'}")
                    results.append({dim: None for dim in DIMENSIONS} | {"reasoning": "JUDGE_FAILED"})
            return results

        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Batch generate failed: {e}")
            print("  Raw judge output unavailable: generation failed before decode.")

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
