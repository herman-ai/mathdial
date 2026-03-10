"""
Analyze conversation statistics for MathDial dataset and model outputs.

Computes per-conversation and aggregate statistics:
  - Number of turns (total, teacher-only, student-only)
  - Teacher utterance lengths (characters and words)
  - Student utterance lengths (characters and words)

Works on:
  - Original MathDial: use --conversation_key conversation
  - Model outputs:     use --conversation_key <model_name>

The conversation field is expected to be |EOM|-delimited with turns like:
  "Teacher: ..." or "Student: ..."

Usage examples:
  # Original MathDial test set
  python analyses/conversation_stats.py \
      --input_file data/test.jsonl \
      --label "MathDial (original)"

  # DPO no-real-teacher model output
  python analyses/conversation_stats.py \
      --input_file output/dpo/qwen_output_dpo_no_real_teacher.jsonl \
      --conversation_key dpo_qwen_instruct_model_no_real_teacher \
      --label "DPO (no real teacher)"

  # Compare multiple files at once
  python analyses/conversation_stats.py \
      --input_file data/test.jsonl output/dpo/qwen_output_dpo_no_real_teacher.jsonl \
      --conversation_key conversation dpo_qwen_instruct_model_no_real_teacher \
      --label "Original" "DPO (no real teacher)"
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_turns(conversation_str: str):
    """Parse conversation into list of (speaker, text) tuples.

    Handles both MathDial format (|EOM| delimiter) and model output format (<EOM> delimiter).
    """
    import re
    # Normalise both delimiter styles to a single separator
    conversation_str = conversation_str.replace("|EOM|", "<EOM>")
    turns = []
    for raw in conversation_str.split("<EOM>"):
        raw = raw.strip()
        if not raw:
            continue
        if ": " not in raw:
            continue
        speaker, text = raw.split(": ", 1)
        speaker = speaker.strip()
        role = "Teacher" if speaker.lower() == "teacher" else "Student"
        # Strip MathDial teacher tags like "(focus)", "(telling)", "(generic)"
        text = re.sub(r"^\([^)]*\)\s*", "", text.strip())
        turns.append((role, text.strip()))
    return turns


def compute_stats(values):
    """Return dict of descriptive stats for a list of numbers."""
    if not values:
        return {k: None for k in ["count", "mean", "median", "std", "min", "p25", "p75", "max"]}
    arr = np.array(values, dtype=float)
    return {
        "count":  int(len(arr)),
        "mean":   round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std":    round(float(np.std(arr)), 2),
        "min":    round(float(np.min(arr)), 2),
        "p25":    round(float(np.percentile(arr, 25)), 2),
        "p75":    round(float(np.percentile(arr, 75)), 2),
        "max":    round(float(np.max(arr)), 2),
    }


def analyze_file(path, conversation_key, label):
    data = read_jsonl(path)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  File : {path}")
    print(f"  Key  : {conversation_key}")
    print(f"{'='*60}")

    total_turns_all = []
    teacher_turns_all = []
    student_turns_all = []
    teacher_char_lens = []
    teacher_word_lens = []
    student_char_lens = []
    student_word_lens = []
    skipped = 0

    for i, row in enumerate(data):
        conv_raw = row.get(conversation_key)
        if not conv_raw:
            skipped += 1
            continue

        turns = parse_turns(conv_raw)
        if not turns:
            skipped += 1
            continue

        teacher_turns = [(role, text) for role, text in turns if role == "Teacher"]
        student_turns = [(role, text) for role, text in turns if role == "Student"]

        total_turns_all.append(len(turns))
        teacher_turns_all.append(len(teacher_turns))
        student_turns_all.append(len(student_turns))

        for _, text in teacher_turns:
            teacher_char_lens.append(len(text))
            teacher_word_lens.append(len(text.split()))

        for _, text in student_turns:
            student_char_lens.append(len(text))
            student_word_lens.append(len(text.split()))

    n = len(data) - skipped
    print(f"\n  Conversations analyzed : {n}  (skipped: {skipped})")

    def print_stats(title, stats):
        print(f"\n  {title}")
        print(f"    count  : {stats['count']}")
        print(f"    mean   : {stats['mean']}")
        print(f"    median : {stats['median']}")
        print(f"    std    : {stats['std']}")
        print(f"    min    : {stats['min']}")
        print(f"    p25    : {stats['p25']}")
        print(f"    p75    : {stats['p75']}")
        print(f"    max    : {stats['max']}")

    print_stats("Total turns per conversation",       compute_stats(total_turns_all))
    print_stats("Teacher turns per conversation",     compute_stats(teacher_turns_all))
    print_stats("Student turns per conversation",     compute_stats(student_turns_all))
    print_stats("Teacher utterance length (chars)",   compute_stats(teacher_char_lens))
    print_stats("Teacher utterance length (words)",   compute_stats(teacher_word_lens))
    print_stats("Student utterance length (chars)",   compute_stats(student_char_lens))
    print_stats("Student utterance length (words)",   compute_stats(student_word_lens))

    return {
        "label": label,
        "n": n,
        "total_turns": compute_stats(total_turns_all),
        "teacher_turns": compute_stats(teacher_turns_all),
        "student_turns": compute_stats(student_turns_all),
        "teacher_char_len": compute_stats(teacher_char_lens),
        "teacher_word_len": compute_stats(teacher_word_lens),
        "student_char_len": compute_stats(student_char_lens),
        "student_word_len": compute_stats(student_word_lens),
    }


def print_comparison_table(all_stats):
    """Print a side-by-side comparison table across all analyzed files."""
    metrics = [
        ("Total turns / conv (mean)",       lambda s: s["total_turns"]["mean"]),
        ("Total turns / conv (median)",     lambda s: s["total_turns"]["median"]),
        ("Teacher turns / conv (mean)",     lambda s: s["teacher_turns"]["mean"]),
        ("Student turns / conv (mean)",     lambda s: s["student_turns"]["mean"]),
        ("Teacher utt length chars (mean)", lambda s: s["teacher_char_len"]["mean"]),
        ("Teacher utt length words (mean)", lambda s: s["teacher_word_len"]["mean"]),
        ("Student utt length chars (mean)", lambda s: s["student_char_len"]["mean"]),
        ("Student utt length words (mean)", lambda s: s["student_word_len"]["mean"]),
    ]

    labels = [s["label"] for s in all_stats]
    col_w = max(30, max(len(l) for l in labels) + 2)
    metric_w = 35

    print(f"\n\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"  {'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("  " + "-" * (metric_w + col_w * len(labels)))
    for metric_label, fn in metrics:
        row = f"  {metric_label:<{metric_w}}"
        for s in all_stats:
            val = fn(s)
            row += f"{str(val):>{col_w}}"
        print(row)
    print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, nargs="+", required=True,
                        help="One or more JSONL files to analyze")
    parser.add_argument("--conversation_key", type=str, nargs="+", default=None,
                        help="Conversation field name(s). Defaults to 'conversation' for each file.")
    parser.add_argument("--label", type=str, nargs="+", default=None,
                        help="Display labels for each file. Defaults to filename.")
    parser.add_argument("--export_file", type=str, default=None,
                        help="Optional path to write JSON results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_files = len(args.input_file)

    # Fill defaults
    keys = args.conversation_key or (["conversation"] * n_files)
    labels = args.label or [os.path.basename(f) for f in args.input_file]

    if len(keys) == 1 and n_files > 1:
        keys = keys * n_files
    if len(labels) == 1 and n_files > 1:
        labels = labels * n_files

    if len(keys) != n_files or len(labels) != n_files:
        print("ERROR: --input_file, --conversation_key, and --label must have the same number of entries.")
        sys.exit(1)

    all_stats = []
    for path, key, label in zip(args.input_file, keys, labels):
        stats = analyze_file(path, key, label)
        all_stats.append(stats)

    if len(all_stats) > 1:
        print_comparison_table(all_stats)

    if args.export_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_file)), exist_ok=True)
        with open(args.export_file, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"Results written to {args.export_file}")
