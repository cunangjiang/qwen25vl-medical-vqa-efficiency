# GitHub Release Checklist

## Recommended repository metadata

**Repository name**

```text
qwen25vl-medical-vqa-efficiency
```

**Repository description**

```text
Qwen2.5-VL-3B on VQA-RAD: 1-epoch LoRA adaptation, visual-token budget/latency profiling, and semantic error analysis.
```

**Recommended topics**

```text
qwen-vl
qwen2-5-vl
vision-language-model
multimodal
medical-vqa
medical-ai
lora
parameter-efficient-finetuning
efficient-inference
visual-token
latency-benchmark
error-analysis
pytorch
huggingface
```

## Files that should be public

```text
README.md
LICENSE
.gitignore
requirements.txt

assets/figures/*

docs/PROJECT_RECORD.md
docs/THIRD_PARTY.md
docs/GITHUB_RELEASE_CHECKLIST.md
docs/RELEASE_AUDIT.md

src/*.py
src/analysis/*.py

scripts/*.sh
scripts/README.md

experiments/metrics/*
```

## Files/directories that should stay private

Do not copy these from the experiment server into the public repository:

```text
models/
outputs/
checkpoint-*/
data/
cache/
logs/
manifests/

results/vqa_rad/*_predictions.csv
results/vqa_rad/day8_resolution/*_predictions.csv
results/vqa_rad/day9_analysis/raw or intermediate casebooks

trainer_state.json
optimizer.pt
scheduler.pt
rng_state*.pth
raw args.json / raw logging.jsonl
screenlog.*
nohup.out
wandb/
.env*
env.sh
```

The public `experiments/metrics/` directory contains compact summaries only; it is separate from the private raw `results/` workspace.

## Recommended first commit message

```text
release: publish Medical VQA LoRA and visual-budget study
```

Alternative shorter form:

```text
feat: release Qwen2.5-VL medical VQA study
```

## Safe local Git workflow

Create the GitHub repository as an **empty repository** (do not ask GitHub to pre-create a README, license, or `.gitignore`), then run these commands inside the extracted release directory.

```bash
git init
git branch -M main

git status --short
```

Run the pre-commit checks first:

```bash
find . -type f -size +20M -print

grep -RInE \
  'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN [A-Z ]*PRIVATE KEY|/media/root|/home/dy|f34190af' \
  . \
  --exclude='*.png' \
  --exclude='*.pdf' \
  --exclude='GITHUB_RELEASE_CHECKLIST.md' \
  --exclude-dir='.git' || true

python -m compileall -q src
for f in scripts/*.sh; do bash -n "$f"; done
```

Stage files **explicitly**; do not use `git add .`:

```bash
git add README.md LICENSE .gitignore requirements.txt

git add assets/figures

git add docs/PROJECT_RECORD.md \
        docs/THIRD_PARTY.md \
        docs/GITHUB_RELEASE_CHECKLIST.md \
        docs/RELEASE_AUDIT.md

git add src

git add scripts

git add experiments/metrics
```

Review exactly what is staged:

```bash
git status --short
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
```

Before committing, verify that the staged list contains no model/data/checkpoint files:

```bash
git diff --cached --name-only | grep -E \
  '(^|/)(models|outputs|data|cache|logs|manifests)/|checkpoint|safetensors|_predictions\.csv$' \
  && echo 'STOP: review unexpected staged files' \
  || echo 'Staged file list looks clean'
```

Commit:

```bash
git commit -m "release: publish Medical VQA LoRA and visual-budget study"
```

Add the remote and push:

```bash
git remote add origin git@github.com:<YOUR_GITHUB_USERNAME>/qwen25vl-medical-vqa-efficiency.git
git push -u origin main
```

If you use HTTPS instead of SSH:

```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/qwen25vl-medical-vqa-efficiency.git
git push -u origin main
```

## After pushing

Check the GitHub page in this order:

1. README images render correctly.
2. No model/data/checkpoint files appear in the repository tree.
3. `LICENSE` is detected as Apache-2.0 for this repository's own code.
4. `docs/THIRD_PARTY.md` is visible and clearly separates Qwen/VQA-RAD licensing.
5. The repository description and topics match the recommendations above.
6. The qualitative figure is readable on desktop and mobile.
7. `experiments/metrics/final_results_summary.csv` matches README headline numbers.

Do not upload a merged LoRA checkpoint as a GitHub Release asset unless you separately re-check the base-model license and redistribution terms.
