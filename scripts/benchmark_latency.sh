#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${LORA_MODEL_PATH:?Set LORA_MODEL_PATH to the merged LoRA checkpoint directory.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p results/vqa_rad/day8_resolution
python src/benchmark_day8_latency.py \
  --model "$LORA_MODEL_PATH" \
  --dataset data/vqa_rad \
  --manifest manifests/vqa_rad_test_smoke50.csv \
  --output-csv results/vqa_rad/day8_resolution/day8_latency_repeated.csv \
  --summary-json results/vqa_rad/day8_resolution/day8_latency_repeated_summary.json \
  --seed 42 \
  --max-new-tokens 32 \
  --warmup-samples 3
