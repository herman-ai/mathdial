"""
Step 1 of 2: Generate N candidate teacher responses per turn.

Loads only the teacher model (small/fast). For each teacher turn in the dataset,
generates --num_generations responses and saves them to a JSONL file for judging.

Output format (one line per teacher turn opportunity):
{
  "prompt": "<chat template string>",
  "question": "...",
  "solution": "...",
  "conversation_context": "<str(history)>",
  "responses": ["response_0", "response_1", ...]
}

Usage:
  python DPO/generate_candidate_responses.py \
      --output_file data/candidate_responses/train_candidates.jsonl \
      --split train \
      --num_generations 5
"""

import json
import os
import re
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactivetutoring"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from history import History
from message import Message
from roles import Roles
from qwen_baseline import QwenTeacher
from qwen_base_teacher import BaseModelTeacher


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "Qwen_SFT_model",
    "finetuned_unweighted_qwen_instruct_teacher_model"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to process")
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Number of candidate responses per teacher turn")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of generations to run in one model.generate() call")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL path. Defaults to data/candidate_responses/<split>_candidates.jsonl")
    parser.add_argument("--max_conversations", type=int, default=0,
                        help="If >0, process only first N conversations from the split (for smoke tests)")
    parser.add_argument("--base_model", action="store_true",
                        help="Use BaseModelTeacher (plain-text completion) instead of QwenTeacher (chat template). "
                             "Required when model_path points to a base-model SFT checkpoint.")
    return parser.parse_args()


def parse_conversation(conversation_str):
    turns = []
    for turn in conversation_str.split("|EOM|"):
        turn = turn.strip()
        if not turn or ": " not in turn:
            continue
        speaker, turn_text = turn.split(": ", 1)
        turns.append((speaker.strip(), turn_text.strip()))
    return turns


def clean_response(turn_text):
    turn_text = turn_text.strip()
    turn_text = re.sub(r"^\s*(Teacher|Assistant):\s*", "", turn_text)
    turn_text = re.split(r"\nTeacher:|\nStudent:|\nAssistant:|\nUser:|\|EOM\|", turn_text)[0].strip()
    return turn_text


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()

    default_out = os.path.join(
        os.path.dirname(__file__), "..", "data", "candidate_responses",
        f"{args.split}_candidates.jsonl"
    )
    output_file = args.output_file or default_out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading teacher model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()
    if args.base_model:
        print("Using BaseModelTeacher (plain-text completion format)", flush=True)
        teacher = BaseModelTeacher(model, tokenizer, device)
    else:
        teacher = QwenTeacher(model, tokenizer, device)
    print(f"Teacher model loaded on {device}", flush=True)

    dataset = load_dataset("eth-nlped/mathdial")[args.split]
    if args.max_conversations and args.max_conversations > 0:
        cap = min(args.max_conversations, len(dataset))
        dataset = dataset.select(range(cap))
    total_convs = len(dataset)
    rows = []

    print(f"\n=== Generating candidates for split='{args.split}' ({total_convs} conversations) ===", flush=True)

    for conv_idx, conversation in enumerate(tqdm(dataset, desc="Conversations", unit="conv")):
        if conv_idx % 100 == 0:
            print(f"[Status] conv {conv_idx}/{total_convs} | candidate rows so far: {len(rows)}", flush=True)

        question = conversation["question"]
        solution = conversation["ground_truth"]
        turns = parse_conversation(conversation["conversation"])
        history = History()

        for speaker, turn_text in turns:
            if speaker.lower() == "teacher":
                if history.messages and history.messages[-1].persona == Roles.STUDENT:
                    prompt = teacher.build_prompt(history, question, solution)
                    conversation_context = str(history)

                    # Generate in sub-batches to avoid OOM
                    all_responses = []
                    remaining = args.num_generations
                    while remaining > 0:
                        bs = min(args.batch_size, remaining)
                        batch_responses = teacher.response_batch(
                            [history] * bs,
                            [question] * bs,
                            [solution] * bs,
                        )
                        all_responses.extend([clean_response(r) for r in batch_responses])
                        remaining -= bs

                    rows.append({
                        "prompt": prompt,
                        "question": question,
                        "solution": solution,
                        "conversation_context": conversation_context,
                        "expert_response": turn_text.strip(),
                        "responses": all_responses,
                    })

                history.add_message(Message(Roles.TEACHER, turn_text.strip()))
            else:
                history.add_message(Message(Roles.STUDENT, turn_text.strip()))

    write_jsonl(output_file, rows)
    print(f"\n[Done] Wrote {len(rows)} candidate rows to {output_file}", flush=True)
