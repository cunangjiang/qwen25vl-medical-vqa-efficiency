# Metrics

This directory contains compact public summaries derived from the frozen experiment outputs.

- `final_results_summary.csv`: quality plus repeated-latency fields by condition.
- `final_results_summary.json`: machine-readable training, quality, latency, validity, and semantic-review summary.
- `training_summary.json`: formal 1-epoch LoRA training metadata.
- `main_zero_vs_lora.csv`: source table for the zero-shot vs LoRA figure.
- `visual_budget_quality.csv`: source table for the visual-budget quality figure.
- `latency_repeated_summary.csv`: source table for the repeated latency figure.
- `semantic_flip_profile.csv`: affected-case semantic-target counts.
- `strict_vs_semantic_net.csv`: strict-EM vs semantic-aware transition net counts.
- `overlap_stratified_metrics.csv`: Day 10 diagnostic split by image exposure during actual SFT training.

Quality metrics use the full 451-QA test split. Latency metrics use a separate fixed 50-QA repeated benchmark. The two populations must not be mixed when reporting average visual-token counts.
