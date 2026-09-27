# Third-Party Components and Data

This repository is a research/portfolio project built on third-party models,
datasets, libraries, and methods. The repository's own license, if added,
does **not** replace or override any third-party license.

The public release intentionally does **not** include Qwen model weights,
LoRA checkpoints, Hugging Face caches, or the full VQA-RAD dataset.

## Qwen2.5-VL-3B-Instruct

- Project/model: **Qwen2.5-VL-3B-Instruct**
- Provider: Qwen / Alibaba Cloud
- Use in this project: base vision-language model for inference and LoRA SFT
- Model repository:
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- License shown by the model repository:
  **Qwen RESEARCH LICENSE AGREEMENT**
- Official license file:
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE

No Qwen model weights or merged fine-tuned weights are redistributed here.
Users must obtain the model separately and comply with the model's own license.

The official license should be consulted directly for current usage,
redistribution, and commercial-use conditions.

## VQA-RAD

- Dataset: **VQA-RAD**
- Original paper:
  Jason J. Lau, Soumya Gayen, Asma Ben Abacha, Dina Demner-Fushman,
  "A dataset of clinically generated visual questions and answers about
  radiology images," Scientific Data, 2018.
- Distribution used in this project:
  https://huggingface.co/datasets/flaviagiammarino/vqa-rad
- License reported by that distribution: **CC0-1.0**
- Use in this project: Medical VQA train/validation/test experiments and
  representative qualitative analysis.

The dataset is not mirrored in this repository. Public figures that contain
representative VQA-RAD images are included only for documenting experimental
behavior and retain visible source/watermark information where present.

Suggested citation:

```bibtex
@article{lau2018dataset,
  title={A dataset of clinically generated visual questions and answers about radiology images},
  author={Lau, Jason J and Gayen, Soumya and Ben Abacha, Asma and Demner-Fushman, Dina},
  journal={Scientific Data},
  volume={5},
  number={1},
  pages={1--10},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## LoRA / PEFT

- Method: **LoRA (Low-Rank Adaptation)**
- PEFT implementation:
  https://github.com/huggingface/peft
- PEFT license: **Apache-2.0**
- Use in this project: parameter-efficient adaptation of the language-model
  side of Qwen2.5-VL.

LoRA is an existing method and is not claimed as a contribution of this project.

## ms-swift

- Project:
  https://github.com/modelscope/ms-swift
- License: **Apache-2.0**
- Use in this project: LoRA SFT and model export/merge workflow.

## Hugging Face Transformers

- Project:
  https://github.com/huggingface/transformers
- License: **Apache-2.0**
- Use in this project: model/processor loading, generation, and Qwen2.5-VL
  inference.

## Hugging Face Datasets

- Project:
  https://github.com/huggingface/datasets
- License: **Apache-2.0**
- Use in this project: local dataset loading and processing.

## qwen-vl-utils

- Project:
  https://github.com/QwenLM/Qwen2-VL/tree/main/qwen-vl-utils
- License: **Apache-2.0**
- Use in this project: multimodal visual-input preprocessing utilities.

## PyTorch

- Project:
  https://github.com/pytorch/pytorch
- Main project license: **BSD-3-Clause**
- Use in this project: model execution, CUDA inference, training, and timing.

## Notes on repository licensing

A future root `LICENSE` applies only to material that the repository owner has
the right to license. It does not re-license:

- Qwen model weights or Qwen materials;
- VQA-RAD;
- third-party Python packages;
- any other upstream code retained with its original copyright/license header.

When upstream source code is copied or modified rather than merely imported,
its original copyright/license notices must be retained as required by that
upstream license.
