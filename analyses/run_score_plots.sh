#!/usr/bin/env bash
# Runs plot_dpo_score_distributions.py for every *judge_scores*.jsonl found in:
#   ../output/dpo/          → ../analyses/plots/dpo/
#   ../output/              → ../analyses/plots/
#
# Output filename: strip "qwen_judge_scores_" prefix, append "_score_distributions.pdf"
#   e.g. qwen_judge_scores_dpo_no_real_teacher.jsonl
#        → qwen_dpo_no_real_teacher_score_distributions.pdf
#
# Usage:
#   chmod +x run_score_plots.sh
#   ./run_score_plots.sh

set -uo pipefail

ROOT_INPUT_DIR="../output"
ROOT_OUTPUT_DIR="../analyses/plots"
SCRIPT="plot_dpo_score_distributions.py"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$ROOT_INPUT_DIR" ]]; then
    echo "[ERROR] Input directory not found: $ROOT_INPUT_DIR"
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Python script not found: $SCRIPT"
    exit 1
fi

# ── Helper: compute output PDF path from input file ───────────────────────────
# Strips "qwen_judge_scores_" prefix and appends "_score_distributions.pdf"
make_output_path() {
    local input_file="$1"
    local out_dir="$2"
    local stem
    stem=$(basename "$input_file" .jsonl)
    local clean="${stem#qwen_judge_scores_}"
    echo "${out_dir}/qwen_${clean}_score_distributions.pdf"
}

# ── Build list of (input, output) pairs ──────────────────────────────────────
shopt -s nullglob

declare -a input_files=()
declare -a output_files=()

# Files directly in ../output/ (subfolders like dpo/ are intentionally excluded)
for f in "$ROOT_INPUT_DIR"/*judge_scores*.jsonl; do
    input_files+=("$f")
    output_files+=("$(make_output_path "$f" "$ROOT_OUTPUT_DIR")")
done

if [[ ${#input_files[@]} -eq 0 ]]; then
    echo "[WARN] No *judge_scores*.jsonl files found under $ROOT_INPUT_DIR"
    exit 0
fi

echo "Found ${#input_files[@]} file(s) to process"
echo "========================================"
echo "Files discovered:"
for i in "${!input_files[@]}"; do
    echo "  [$i] ${input_files[$i]}"
done
echo "========================================"
echo ""

success=0
failure=0

for i in "${!input_files[@]}"; do
    input_file="${input_files[$i]}"
    output_file="${output_files[$i]}"

    mkdir -p "$(dirname "$output_file")"

    echo "Input  : $input_file"
    echo "Output : $output_file"

    if python "$SCRIPT" "$input_file" "$output_file"; then
        echo "  [OK]"
        success=$((success + 1))
    else
        echo "  [FAILED] — skipping"
        failure=$((failure + 1))
    fi

    echo ""
done

echo "========================================"
echo "Done.  Success: $success   Failed: $failure"