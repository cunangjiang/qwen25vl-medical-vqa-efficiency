# Analysis scripts

The Day 9 scripts reconstruct the correctness-flip tables, manual semantic taxonomy, and qualitative case package used in the project. They expect the private/generated `results/vqa_rad/` prediction files to exist locally.

The Day 10 scripts are release-validity checks:

- `day10_audit_image_overlap.py`: measures image-identity overlap between actual SFT train/val and test.
- `day10_audit_tokenizer_parity.py`: checks base vs merged-checkpoint prompt token IDs.
- `day10_overlap_stratified_metrics.py`: stratifies zero-shot/LoRA metrics by whether the test image appeared in actual SFT training.

These scripts do not change model weights.
