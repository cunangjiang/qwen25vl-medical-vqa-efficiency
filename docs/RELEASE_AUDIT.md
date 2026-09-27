# Public Release Audit

This document records the final public-repository checks performed after the Day 10 experiment audit and source cleanup.

## Repository scope

The release repository contains only:

- project documentation;
- compact result summaries;
- final figures;
- selected training/evaluation/analysis source code;
- portable shell entry points.

It intentionally excludes the private experiment workspace.

## Large-file audit

Final release size at audit time: approximately **2.4 MB**.

No file larger than **10 MB** was present.

Excluded from the release:

- ~7 GB base-model directory;
- ~7 GB training/output directory;
- checkpoints and merged weights;
- optimizer state;
- dataset/cache directories;
- raw logs and full prediction tables.

## Secret / machine-path audit

The release tree was scanned for common credential/token signatures, private-key headers, and experiment-server absolute-path markers.

No matching secret or server-specific absolute path remained in the public release files.

The Conda environment name retained in `PROJECT_RECORD.md` is an experiment-software record, not a credential or filesystem path.

## Source syntax checks

The final `src/` tree passed:

```text
python -m compileall -q src
```

The public shell scripts passed:

```text
bash -n scripts/*.sh
```

## Public source-code selection

Included as core reproducibility code:

- single-image inference;
- VQA-RAD download/local-save helper;
- full/smoke manifest preparation;
- image-disjoint SFT train/validation preparation;
- zero-shot evaluation;
- visual-budget evaluation;
- repeated latency benchmark;
- zero-shot vs LoRA comparison;
- Day 9 semantic/error-analysis scripts;
- Day 10 image-overlap/tokenizer validity audits;
- public figure generation.

Excluded as redundant/debug-only:

- early dataset-candidate probe;
- duplicate smoke-only preprocessing helper;
- raw server-bound training shell scripts;
- experiment logs.

## Result-integrity checks

Headline result files were regenerated from frozen experiment outputs rather than manually retyped where source CSV/JSON existed.

The release explicitly preserves these limitations:

- the VQA-RAD dataset-provided QA split is not image-disjoint;
- project normalized EM is not presented as an official VQA-RAD metric;
- Open token F1 is diagnostic only;
- the repeated latency benchmark uses a fixed 50-QA population separate from the 451-QA quality population;
- diagnostic dynamic PyTorch allocation is not labeled total GPU peak memory;
- VT512 is not claimed to improve medical capability;
- the project does not claim QLoRA, 7B experiments, new token pruning, or SOTA.

## License boundary

The root Apache-2.0 license applies to this repository's original code/documentation only.

It does not re-license:

- Qwen2.5-VL-3B-Instruct or its weights;
- VQA-RAD;
- third-party libraries or upstream source code.

See `THIRD_PARTY.md` for the external components used by the project.
