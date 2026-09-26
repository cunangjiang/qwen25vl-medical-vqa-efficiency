#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-VL-3B-Instruct directory.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

swift sft \
  --model "$MODEL_PATH" \
  --check_model false \
  --dataset "$ROOT_DIR/data/vqa_rad_sft/train.jsonl" \
  --val_dataset "$ROOT_DIR/data/vqa_rad_sft/val.jsonl" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --learning_rate 1e-4 \
  --max_length 4096 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 5 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 2 \
  --load_from_cache_file true \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --output_dir "$ROOT_DIR/outputs/lora_1ep"
