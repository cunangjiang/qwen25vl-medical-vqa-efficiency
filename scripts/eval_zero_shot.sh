#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-VL-3B-Instruct directory.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p results/vqa_rad
python src/eval_vqa_rad_zero_shot.py \
  --model "$MODEL_PATH" \
  --dataset data/vqa_rad \
  --manifest manifests/vqa_rad_test_full451.csv \
  --output-csv results/vqa_rad/zero_shot_full451_predictions.csv \
  --summary-json results/vqa_rad/zero_shot_full451_summary.json \
  --seed 42 \
  --max-new-tokens 32
