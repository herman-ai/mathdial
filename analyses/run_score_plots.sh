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
# Mirrors the subdirectory structure of the input under ROOT_OUTPUT_DIR,
# and strips "judge_scores_" / "_judge_scores" from the filename.
make_output_path() {
    local input_file="$1"
    local stem subdir out_dir clean
    stem=$(basename "$input_file" .jsonl)
    # Preserve subdirectory relative to ROOT_INPUT_DIR
    subdir=$(dirname "$input_file")
    subdir="${subdir#$ROOT_INPUT_DIR}"   # strip leading input root
    subdir="${subdir#/}"                  # strip leading slash if any
    out_dir="$ROOT_OUTPUT_DIR${subdir:+/$subdir}"
    # Remove "judge_scores_" or "_judge_scores" wherever they appear
    clean="${stem/judge_scores_/}"
    clean="${clean/_judge_scores/}"
    echo "${out_dir}/${clean}_score_distributions.pdf"
}

# ── Build list of (input, output) pairs ──────────────────────────────────────
shopt -s nullglob

declare -a input_files=()
declare -a output_files=()

# Recursively find all *judge_scores*.jsonl under ROOT_INPUT_DIR
while IFS= read -r -d '' f; do
    input_files+=("$f")
    output_files+=("$(make_output_path "$f")")
done < <(find "$ROOT_INPUT_DIR" -name "*judge_scores*.jsonl" -print0 | sort -z)

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