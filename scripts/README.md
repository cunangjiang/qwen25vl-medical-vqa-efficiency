# Reproduction scripts

These scripts keep machine-specific paths outside version control.

Set the local model path at runtime:

```bash
export MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export LORA_MODEL_PATH=/path/to/merged-lora-checkpoint
export CUDA_VISIBLE_DEVICES=0
```

The scripts assume `data/`, `manifests/`, `outputs/`, and raw `results/` are local working directories and are excluded from the public Git repository.
