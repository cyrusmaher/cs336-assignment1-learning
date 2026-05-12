#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-${INPUT_PATH:-/Users/cyrusmaher/Documents/CS336/TinyStoriesV2-GPT4-train.txt}}"
VOCAB_SIZE="${VOCAB_SIZE:-10000}"
DESIRED_NUM_CHUNKS="${DESIRED_NUM_CHUNKS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/problem_2_5_train_bpe_tinystories}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/problem_2_5_train_bpe_tinystories.py"

mkdir -p "${OUTPUT_DIR}"

echo "problem=train_bpe_tinystories"
echo "input_path=${INPUT_PATH}"
echo "vocab_size=${VOCAB_SIZE}"
echo "special_tokens=<|endoftext|>"
echo "desired_num_chunks=${DESIRED_NUM_CHUNKS}"
echo "output_dir=${OUTPUT_DIR}"

/usr/bin/time -l uv run python "${PYTHON_SCRIPT}" "${INPUT_PATH}" "${VOCAB_SIZE}" "${DESIRED_NUM_CHUNKS}" "${OUTPUT_DIR}"
