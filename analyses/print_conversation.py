#!/usr/bin/env python3
"""
Print the conversation from a specific line of a JSONL file,
replacing |EOM| tokens with newlines for readability.

Usage:
    python print_conversation.py <jsonl_file> <line_number>

Example:
    python print_conversation.py ../output/qwen_output.jsonl 42
"""

import json
import sys


def print_conversation(jsonl_path: str, line_number: int) -> None:
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for current, line in enumerate(fh, 1):
            if current == line_number:
                break
        else:
            print(f"[ERROR] File only has {current} lines — line {line_number} not found.")
            sys.exit(1)

    line = line.strip()
    if not line:
        print(f"[ERROR] Line {line_number} is empty.")
        sys.exit(1)

    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse JSON on line {line_number}: {exc}")
        sys.exit(1)

    # Find the conversation field — try common key names
    conversation = None
    for key in ("conversation", "dpo_qwen_instruct_model_human_preferences",
                record.get("model_name", "")):
        if key and key in record and isinstance(record[key], str):
            conversation = record[key]
            conv_key = key
            break

    # Fallback: first string value that contains |EOM|
    if conversation is None:
        for key, value in record.items():
            if isinstance(value, str) and "|EOM|" in value:
                conversation = value
                conv_key = key
                break

    DIVIDER = "=" * 70
    THIN    = "-" * 70

    print(DIVIDER)
    print(f"  File : {jsonl_path}")
    print(f"  Line : {line_number}")
    print(f"  QID  : {record.get('qid', 'N/A')}")
    if record.get("scenario") is not None:
        print(f"  Scenario : {record['scenario']}")
    print(DIVIDER)

    if record.get("question"):
        print("\nQUESTION:")
        print(f"  {record['question']}\n")

    if conversation is None:
        print("[WARN] No conversation field found. Available keys:")
        for k, v in record.items():
            preview = str(v)[:80].replace("\n", " ")
            print(f"  {k}: {preview}")
        sys.exit(0)

    print(f"CONVERSATION  (field: '{conv_key}'):")
    print(THIN)

    turns = conversation.split("|EOM|")
    for turn in turns:
        turn = turn.strip()
        if turn:
            print(turn)
            print()

    print(DIVIDER)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python print_conversation.py <jsonl_file> <line_number>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        line_num = int(sys.argv[2])
        if line_num < 1:
            raise ValueError
    except ValueError:
        print(f"[ERROR] Line number must be a positive integer, got: {sys.argv[2]}")
        sys.exit(1)

    print_conversation(path, line_num)