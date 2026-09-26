#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${LORA_MODEL_PATH:?Set LORA_MODEL_PATH to the merged LoRA checkpoint directory.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p results/vqa_rad/day8_resolution

for vt in 256 512 1024; do
  max_pixels=$((vt * 28 * 28))
  python src/eval_vqa_rad_resolution.py \
    --model "$LORA_MODEL_PATH" \
    --dataset data/vqa_rad \
    --manifest manifests/vqa_rad_test_full451.csv \
    --output-csv "results/vqa_rad/day8_resolution/vt${vt}_full451_predictions.csv" \
    --summary-json "results/vqa_rad/day8_resolution/vt${vt}_full451_summary.json" \
    --min-pixels 3136 \
    --max-pixels "$max_pixels" \
    --seed 42 \
    --max-new-tokens 32
done
