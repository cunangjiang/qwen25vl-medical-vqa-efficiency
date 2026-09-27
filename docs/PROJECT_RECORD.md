# PROJECT_RECORD — Qwen2.5-VL Medical VQA

> 本文件是项目 B 的技术档案 / 实验交接记录，面向未来复盘、面试准备和后续论文工作。  
> 它比根目录 `README.md` 更详细，但仍只记录本项目真实完成的实验。  
> 所有指标、设置和结论均来自已经冻结的 Day 6–Day 10 实验与 validity audit。

---

## 0. Project Snapshot

**Project:** Qwen2.5-VL Medical VQA — LoRA Adaptation & Visual Token Budget Analysis

**Main questions**

1. 在固定 prompt / evaluator 下，轻量 LoRA domain adaptation 如何改变 Qwen2.5-VL-3B 在 Medical VQA 上的表现？
2. 固定 LoRA 模型后，降低 `max_pixels / visual-token budget` 能节省多少推理成本？
3. 哪些医学语义问题对视觉预算压缩更敏感？
4. strict exact match 的 gain/loss 中，有多少是真正的 semantic correction/regression，有多少来自 wording / granularity / evaluator sensitivity？

**Scope intentionally excluded**

- 不追求 VQA-RAD SOTA；
- 不训练 7B；
- 不做 QLoRA；
- 不做 3D medical imaging；
- 不做大规模医学预训练；
- 不实现新的 visual-token pruning architecture；
- 不把本项目包装成“提出新模型”。

---

# 1. Environment

## 1.1 Hardware

正式训练与主要推理实验在单张：

```text
NVIDIA GeForce RTX 3090
```

上完成。

项目没有依赖多 GPU 才能完成核心结果。

## 1.2 Conda environments

项目期间使用了两个独立环境：

```text
Inference:
qwen25vl_med

Training:
qwen25vl_med_swift
```

这样做的原因是把 inference/evaluation 依赖与 ms-swift training stack 隔离，避免大范围修改已经跑通的推理环境。

## 1.3 Recorded package versions

项目期间记录的主要版本：

```text
Python              3.10.21

torch               2.6.0+cu124
torchvision         0.21.0+cu124
torchaudio           2.6.0+cu124

transformers        4.57.6
accelerate          1.15.0
datasets            5.0.1
huggingface_hub     0.36.2

qwen-vl-utils       0.0.8 + decord
pandas              2.3.3
numpy               2.2.6

ms-swift            4.5.3
peft                0.20.0
```

训练日志中模型配置也记录了：

```text
transformers_version = 4.57.6
attention implementation = SDPA
```

> 注意：公开仓库最终的 dependency file 应只保留真正复现核心流程需要的依赖，不必把整个 Conda 环境原样 freeze。

---

# 2. Base Model

底座模型：

```text
Qwen2.5-VL-3B-Instruct
```

本项目没有修改其原始 architecture，也没有声称模型本身或 LoRA 方法由本项目提出。

本项目围绕该模型新增的是：

- Medical VQA evaluation pipeline；
- VQA-RAD SFT preparation；
- LoRA adaptation experiment；
- visual-budget controlled experiment；
- latency / visual-token profiling；
- fine-grained error analysis；
- semantic-aware transition review；
- validity audits。

模型权重与 merged LoRA 权重均不进入公开 GitHub。

---

# 3. Dataset

## 3.1 Dataset selected

正式主数据集：

```text
VQA-RAD
Hugging Face repo used during project:
flaviagiammarino/vqa-rad
```

项目早期同时 probe 过 SLAKE，但为了在短周期内保持一套稳定的 preprocessing / evaluator / experiment pipeline，最终只选择 VQA-RAD 作为主数据集。

## 3.2 Dataset shape used in this project

本地获得的版本：

```text
Source train QA: 1793
Source test QA:   451

Source train unique images: 313
Source test unique images:  203
```

本项目使用 dataset-provided test split 的全部 451 QA 作为正式 test。

## 3.3 Closed / Open definition

项目没有使用外部 question taxonomy 作为 closed/open 标签。

定义为：

```text
GT answer after strip + lowercase exactly equals "yes" or "no"
    → closed

otherwise
    → open
```

451-QA test：

```text
Closed: 251
Open:   200
```

因此这里的 closed/open 是**项目内部基于 GT 的规则**，不是 VQA-RAD source taxonomy。

---

# 4. SFT Data Preparation

核心源码：

```text
src/prepare_vqa_rad_sft.py
```

## 4.1 Image identity

为了防止同一 source-train image 同时进入 SFT train 和 validation，使用 RGB 图像内容 hash：

```text
SHA1(
    string(image.size)
    +
    image RGB bytes
)
```

作为 image identity。

## 4.2 Train / Val split

只在 VQA-RAD source train（1793 QA）内部划分：

```text
validation image ratio = 0.10
seed = 42
```

得到：

```text
SFT train:
1612 QA
282 unique images

SFT val:
181 QA
31 unique images

Train ∩ Val images:
0
```

question-type counts：

```text
SFT train:
closed 836
open   776

SFT val:
closed 104
open    77
```

这一层 train/val split 是 image-disjoint。

## 4.3 SFT record format

每个 record 包含：

```text
user:
<image>
Answer the medical visual question with only a short answer.
Do not explain your reasoning.
For yes/no questions, answer only yes or no.
Question: {question}

assistant:
{ground-truth answer}
```

并引用导出的本地 PNG image path。

---

# 5. Prompt

正式 zero-shot / LoRA evaluation 与 SFT 使用统一的短答案约束：

```text
Answer the medical visual question with only a short answer.
Do not explain your reasoning.
For yes/no questions, answer only yes or no.
Question: {question}
```

这样做的目的：

- 减少解释型长输出；
- 降低 answer extraction 的额外复杂度；
- 对 yes/no question 强制较稳定输出格式；
- 让 zero-shot、LoRA、不同 visual budget 尽量保持相同文本条件。

---

# 6. Evaluation Protocol

正式 evaluator：

```text
src/eval_vqa_rad_zero_shot.py
src/eval_vqa_rad_resolution.py
```

## 6.1 Generation

正式推理设置：

```text
dtype: FP16
attention: SDPA
generation: greedy
do_sample: False
max_new_tokens: 32
seed: 42
```

## 6.2 Answer normalization

### Open answer

顺序：

```text
1. Unicode NFKC normalization
2. lowercase
3. punctuation -> spaces
4. remove English articles: a / an / the
5. collapse whitespace
```

### Closed prediction

对 closed question：

```text
extract the first standalone "yes" or "no"
```

若没有找到 standalone yes/no，则退回普通 normalization。

## 6.3 Metrics

主指标：

```text
project normalized exact match (EM)
```

此外：

```text
Open token F1
```

只作为 diagnostic metric。

**重要边界：**

这些不是本项目确认过的 VQA-RAD official metric，因此：

- 不把它写成官方 VQA-RAD score；
- 不直接拿去和使用其他 evaluator / split / dataset version 的论文数字比较；
- 不声称 SOTA。

---

# 7. Day 6 — Zero-shot Baseline

正式 model：

```text
Qwen2.5-VL-3B-Instruct
```

正式 test：

```text
451 QA
251 closed
200 open
```

结果：

| Metric | Zero-shot |
|---|---:|
| Overall normalized EM | 51.66% (233 / 451) |
| Closed EM | 73.71% (185 / 251) |
| Open EM | 24.00% (48 / 200) |
| Open token F1 | 31.86% |
| Avg visual tokens | 871.6 |

正式结果文件：

```text
results/vqa_rad/zero_shot_full451_summary.json
results/vqa_rad/zero_shot_full451_predictions.csv
```

原始 evaluator 内 diagnostic timing：

```text
avg preprocess: ~21.81 ms
avg generate:   ~311.14 ms
avg E2E:        ~344.78 ms
```

但这不是 Day 8 最终 standardized latency benchmark，因此公开项目中的 latency 主结论不使用这组。

---

# 8. Day 7 — LoRA SFT

正式训练入口：

```text
scripts/day7_lora_1ep.sh
```

训练框架：

```text
ms-swift + PEFT
```

## 8.1 Training config

| Item | Value |
|---|---|
| tuner | LoRA |
| target modules | `all-linear` |
| LLM | unfrozen, LoRA-adapted |
| Vision encoder | frozen |
| Aligner | frozen |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| dtype | BF16 |
| attention | SDPA |
| epoch | 1 |
| train QA | 1612 |
| val QA | 181 |
| per-device train batch | 1 |
| per-device eval batch | 1 |
| grad accumulation | 8 |
| gradient checkpointing | True |
| learning rate | 1e-4 |
| optimizer | AdamW (`adamw_torch`) |
| scheduler | cosine |
| warmup ratio | 0.05 |
| max length | 4096 |
| seed / data seed | 42 / 42 |
| eval strategy | epoch |
| save strategy | epoch |
| quantization | None |

所以该实验是：

```text
regular BF16 LoRA
```

不是 QLoRA。

## 8.2 Parameters

训练日志：

```text
Total parameters:
3769.5898M

Trainable:
14.9668M

Trainable ratio:
0.3970%
```

## 8.3 Training outcome

```text
Optimizer steps: 202
Runtime:         30m 43s
Train loss:      0.74308841
Eval loss:       0.41842237
Eval token acc:  0.86019971
Framework memory report: ~12.03 GiB
```

`eval_token_acc` 是 teacher-forced validation token accuracy，**不是 VQA accuracy**，不能作为项目主指标。

原始 checkpoint：

```text
outputs/day7_lora_1ep/v0-20260925-113629/checkpoint-202
```

merged model：

```text
outputs/day7_lora_1ep/v0-20260925-113629/checkpoint-202-merged
```

这些路径只属于原始实验工作区，checkpoint / merged weights 不进入公开 GitHub。

## 8.4 LoRA full-test result

| Metric | Zero-shot | LoRA | Δ |
|---|---:|---:|---:|
| Overall EM | 51.66% | 56.98% | +5.32 pp |
| Closed EM | 73.71% | 80.88% | +7.17 pp |
| Open EM | 24.00% | 27.00% | +3.00 pp |
| Open token F1 | 31.86% | 38.76% | +6.90 pp |

LoRA full451：

```text
257 / 451 correct
```

结果文件：

```text
results/vqa_rad/lora_1ep_full451_summary.json
results/vqa_rad/lora_1ep_full451_predictions.csv
results/vqa_rad/day7_zero_vs_lora_comparison.csv
```

---

# 9. Day 8 — Visual Resolution / Token Budget

核心源码：

```text
src/eval_vqa_rad_resolution.py
```

固定：

```text
LoRA merged model
same prompt
same evaluator
same test split
same generation config
```

只改变 processor 的 visual budget。

## 9.1 Qwen visual-budget relation used

项目中使用：

```text
patch_size = 14
merge_size = 2
```

因此一个 merged LLM visual token 近似对应：

```text
28 × 28 = 784 resized pixels
```

预算定义：

```text
min_pixels = 4 × 28 × 28 = 3136

VT256:
max_pixels = 256 × 28 × 28 = 200,704

VT512:
max_pixels = 512 × 28 × 28 = 401,408

VT1024:
max_pixels = 1024 × 28 × 28 = 802,816
```

这些是 `max_pixels` 上限，不意味着每个 image 都恰好产生 256 / 512 / 1024 visual tokens。

最终分析优先使用**实际测得的 visual token count**。

## 9.2 Full-test quality

| Condition | Avg VT | Overall EM | Closed EM | Open EM | Open F1 |
|---|---:|---:|---:|---:|---:|
| LoRA Native | 871.6 | 56.98% | 80.88% | 27.00% | 38.76% |
| VT1024 | 694.3 | 57.21% | 81.27% | 27.00% | 38.88% |
| VT512 | 422.8 | 57.87% | 81.67% | 28.00% | 39.33% |
| VT256 | 238.4 | 54.32% | 75.70% | 27.50% | 39.26% |

相对 Native：

```text
VT1024:
Avg VT 871.6 → 694.3
~20.3% reduction

VT512:
Avg VT 871.6 → 422.8
~51.5% reduction

VT256:
Avg VT 871.6 → 238.4
~72.6% reduction
```

最安全的项目结论：

```text
VT512:
约减半 visual-token workload，
在这个固定 test split 上没有观察到 overall EM loss。

VT256:
更激进压缩后出现明显 overall EM degradation。
```

不把 VT512 的 57.87% vs 56.98% 写成“低分辨率提升医学能力”。

## 9.3 Correctness flips

Native → VT256：

```text
gain  = 8
loss  = 20
net   = -12
```

Native → VT512：

```text
gain  = 10
loss  = 6
net   = +4
```

Native → VT1024：

```text
gain  = 2
loss  = 1
net   = +1
```

说明 prediction response 对视觉预算不是严格单调的。

---

# 10. Day 8 — Standardized Latency Benchmark

正式 benchmark：

```text
src/benchmark_day8_latency.py
```

最终公开 latency 使用这一套，不使用 evaluator 内的早期单次 timing。

## 10.1 Protocol

```text
Benchmark set:
fixed 50 QA

Conditions:
VT256
VT512
VT1024
LoRA Native

Model:
loaded once

Warm-up:
3 samples per condition

Formal repeats:
4

Measurements:
50 samples × 4 repeats × 4 conditions
= 800 measurements
```

condition order 采用 rotation：

```text
Repeat 1:
VT256 -> VT512 -> VT1024 -> Native

Repeat 2:
VT512 -> VT1024 -> Native -> VT256

Repeat 3:
VT1024 -> Native -> VT256 -> VT512

Repeat 4:
Native -> VT256 -> VT512 -> VT1024
```

目的是减轻固定顺序带来的 system drift / thermal / caching bias。

## 10.2 Timing boundary

明确：

```text
Model loaded once:
Yes

Dataset retrieval included in E2E:
No

PIL conversion included in E2E:
No

Processor / preprocessing included:
Yes

GPU transfer included in E2E:
Yes

Generation included:
Yes

Decoding included:
Yes

CUDA synchronize around generation:
Yes

empty_cache inside timing:
No

reset_peak_memory inside timing:
No
```

因此 E2E 是：

```text
processor preparation
+ GPU transfer
+ generation
+ decode
```

但不包含 dataset retrieval 与 PIL conversion。

## 10.3 Results

| Condition | Avg VT (benchmark50) | Output tokens | Generate mean | Generate median | Generate p95 | E2E mean | E2E median | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VT256 | 241.1 | 2.94 | 152.2 ms | 125.2 ms | 280.1 ms | 172.8 ms | 150.9 ms | 298.2 ms |
| VT512 | 436.8 | 2.90 | 200.5 ms | 190.6 ms | 301.5 ms | 222.2 ms | 217.1 ms | 335.4 ms |
| VT1024 | 759.9 | 2.68 | 291.5 ms | 325.9 ms | 416.8 ms | 316.6 ms | 357.1 ms | 455.6 ms |
| Native | 992.2 | 2.76 | 364.4 ms | 369.9 ms | 611.4 ms | 394.7 ms | 404.8 ms | 665.0 ms |

paired vs Native：

| Condition | Generate reduction mean | Generate reduction median | E2E reduction mean | E2E reduction median | Measurements faster |
|---|---:|---:|---:|---:|---:|
| VT256 | 49.0% | 61.6% | 47.0% | 59.1% | 97% |
| VT512 | 35.5% | 47.0% | 34.2% | 45.5% | 95% |
| VT1024 | 13.6% | 11.2% | 13.3% | 11.6% | 81% |

**注意：**

```text
Full451 Avg VT
!=
Benchmark50 Avg VT
```

quality table 与 latency table 的 population 不同，不应把两个 Avg VT 混用。

正式结果：

```text
results/vqa_rad/day8_resolution/day8_latency_repeated.csv
results/vqa_rad/day8_resolution/day8_latency_repeated_summary.json
```

---

# 11. Memory Profiling

早期 full-test evaluator 记录：

```text
dynamic peak allocated
=
peak PyTorch allocated
-
model baseline allocated
```

大致：

```text
Native:  ~0.182 GiB
VT1024:  ~0.148 GiB
VT512:   ~0.093 GiB
VT256:   ~0.057 GiB
```

但 evaluator 在每个 sample generation 前使用：

```text
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

因此这里测到的是：

```text
diagnostic dynamic PyTorch allocation above baseline
```

不是整卡真实 peak VRAM，也不是 standardized repeated memory benchmark。

处理原则：

- `PROJECT_RECORD.md` 保留；
- README headline 不写成 “Peak GPU Memory”；
- 简历不以这组数字为主要卖点。

---

# 12. Day 9 — Master Error Analysis

核心产物：

```text
results/vqa_rad/day9_analysis/day9_master_comparison.csv
```

对齐 451 QA 的五个 condition：

```text
zero_native
lora_native
lora_vt256
lora_vt512
lora_vt1024
```

## 12.1 Zero → LoRA correctness transition

All：

```text
gain:          42
loss:          18
net:          +24
both correct: 215
both wrong:   176
flip rate:   13.30%
```

Closed：

```text
gain 27
loss  9
net +18
```

Open：

```text
gain 15
loss  9
net  +6
```

这里：

```text
gain
=
before incorrect, after correct

loss
=
before correct, after incorrect
```

所以不能把 “42 gains” 说成 “多对了 42 道”。

净正确数增加：

```text
42 - 18 = 24
```

---

# 13. Fine-grained Semantic Target Taxonomy

Day 9 对 correctness-flip QA 做了项目内部人工语义分类。

这不是 VQA-RAD 官方 taxonomy。

9 个 semantic targets：

```text
finding_presence_absence
morphology_size_attribute
localization_laterality
anatomy_visibility_normality
modality_plane_sequence
anatomy_structure_identification
diagnosis_condition
finding_description
context_demographic_device
```

85 个 unique affected QA 的分布：

| Semantic target | QA |
|---|---:|
| finding_presence_absence | 29 |
| morphology_size_attribute | 19 |
| localization_laterality | 14 |
| anatomy_visibility_normality | 6 |
| modality_plane_sequence | 4 |
| anatomy_structure_identification | 4 |
| diagnosis_condition | 3 |
| finding_description | 3 |
| context_demographic_device | 3 |

## 13.1 LoRA pattern

LoRA gains：

```text
finding_presence_absence: 17
```

是最大的一类。

LoRA losses：

```text
localization_laterality: 8
morphology_size_attribute: 5
```

较集中。

因此只能描述为：

> 在发生 correctness flip 的这批 QA 中，LoRA gains 最集中于 finding presence/absence，而 notable losses 更集中于 localization/laterality。

不能说：

> LoRA 对所有 presence question 都更强，或者对所有 localization question 都更差。

因为这里不是 category-wise accuracy，而只是 flip subset composition。

## 13.2 VT256 pattern

VT256 的 20 个 losses：

```text
morphology / size / attribute:    8
finding presence / absence:       6
anatomy visibility / normality:   4
anatomy structure identification: 1
modality / plane / sequence:      1
```

这支持：

> aggressive compression 的 regression 在 affected subset 中更多集中于 morphology/attribute、finding presence 和 anatomy visibility。

仍然不能证明具体视觉细节“因为 resize 丢失”导致错误。

---

# 14. Strict EM vs Semantic-aware Transition Review

Day 9 最关键的 methodological finding：

```text
strict-EM transition
!=
semantic transition
```

例如：

```text
GT:
right side

Before:
Right side

After:
right sided
```

strict EM 会产生：

```text
correct -> incorrect
```

但医学位置语义基本没有改变。

## 14.1 Event expansion

85 个 affected QA 被展开成：

```text
107 transition events
```

因为同一个 QA 可能同时属于：

```text
LoRA gain
+
VT512 loss
+
VT256 loss
```

comparison 分开处理：

```text
Zero -> LoRA
LoRA Native -> VT1024
LoRA Native -> VT512
LoRA Native -> VT256
```

## 14.2 Surface transition hints

初始 rule-based hints：

```text
yes/no polarity flip:            74
other semantic flip:             18
laterality reversal:              9
near-match -> exact:              3
strict-EM near-miss candidate:    3
```

随后对非纯 yes/no 的 33 events 进行人工 review。

## 14.3 Final semantic-aware counts

| Comparison | Strict gain / loss | Strict net | Definite semantic correction / regression | Semantic net |
|---|---:|---:|---:|---:|
| Zero → LoRA | 42 / 18 | +24 | 38 / 16 | +22 |
| Native → VT1024 | 2 / 1 | +1 | 2 / 1 | +1 |
| Native → VT512 | 10 / 6 | +4 | 7 / 6 | +1 |
| Native → VT256 | 8 / 20 | −12 | 6 / 19 | −13 |

关键解释：

### LoRA

```text
strict net    +24
semantic net  +22
```

大多数 improvement 是真实语义变化，但不是所有 strict gain/loss 都应该按能力变化解释。

### VT512

```text
strict net    +4
semantic net  +1
```

因此不能把 VT512 的表面 +4 QA 当作压缩提高医学能力的证据。

### VT256

```text
strict net    -12
semantic net  -13
```

说明 aggressive compression 的总体退化并不是由 exact-match wording artifact 主导。

---

# 15. Representative Qualitative Cases

最终展示选 4 个主案例。

## Case A — Same-image mixed LoRA effect

```text
image_id:
fd7bcec7d8
```

同一张 brain image：

### Finding presence

```text
Question:
Is there an acute infarction?

GT:
yes

Zero-shot:
no

LoRA:
yes
```

→ semantic correction

### Laterality

```text
Question:
Which side is the lesion on?

GT:
left

Zero-shot:
left

LoRA:
right
```

→ laterality regression

注意医学影像显示 convention：

```text
screen right
=
patient left
```

因此不能按普通照片的屏幕左右直接判断 patient laterality。

这个案例支持：

> 同一次 adaptation 可以改善 finding recognition，同时破坏 spatial/laterality decision。

不支持：

> LoRA 一定会系统性降低所有 laterality performance。

---

## Case B — Resolution-sensitive anatomy identification

```text
image_id:
a253d30997

QA:
443
```

```text
Question:
what structure lies directly posterior to the appendix?

GT:
psoas muscle

Zero-shot:
Colon

LoRA Native:
the psoas muscle

VT1024:
the psoas muscle

VT512:
the cecum

VT256:
the cecum
```

观察事实：

```text
LoRA corrects the anatomy answer.
The correction survives VT1024 but not VT512 / VT256.
```

安全解释：

> 该 anatomy-identification case 对降低 visual budget 呈现明显 prediction sensitivity。

不能仅凭单例证明：

> resize 因果性地抹掉了某个具体 anatomical detail。

---

## Case C — Exact-match wording sensitivity

```text
image_id:
834c75b256
```

```text
GT:
nodular opacities

LoRA Native:
nodules

VT512:
nodular opacities
```

strict EM：

```text
0 -> 1
```

但语义变化比这个 0→1 分数跳变小得多。

用途：

> 展示为什么 strict EM gain 需要 semantic review。

---

## Case D — Non-monotonic visual-budget response

```text
image_id:
e551b2f8ad

QA:
212
```

```text
Question:
is there a left apical pneumothorax?

GT:
yes

LoRA Native:
yes

VT1024:
no

VT512:
yes

VT256:
no
```

表现：

```text
Native   correct
VT1024   wrong
VT512    correct
VT256    wrong
```

说明：

> 单样本 correctness 与 visual budget 不保证严格单调。

因此 quality–budget 结论必须依赖 aggregate test statistics，而不是少量 case。

---

# 16. Day 10 Validity Audit

Day 10 没有新增大实验，只做发布前 validity checks。

---

## 16.1 Audit A — Actual SFT image overlap with test

结果：

```text
SFT train QA rows: 1612
SFT val QA rows:    181

SFT train unique images: 282
SFT val unique images:    31
Test unique images:       203

Train ∩ Val:
0
```

但 dataset-provided source train/test：

```text
SFT-train ∩ Test unique images:
181 / 203
= 89.2%

Val ∩ Test unique images:
21 / 203
= 10.3%

Source-train ∩ Test unique images:
202 / 203
= 99.5%
```

test QA perspective：

```text
Test QA on SFT-train images:
402 / 451
= 89.1%

Test QA on val images:
44 / 451
= 9.8%

Test QA on any source-train image:
446 / 451
= 98.9%
```

### Interpretation

这不是：

```text
train/test QA row duplication
```

而是：

```text
train/test reuse many of the same images
with different QA instances
```

所以：

> 当前 +5.32 pp 可以作为 dataset-provided QA split 上的 LoRA improvement。

不能说：

> completely unseen-image generalization +5.32 pp。

---

## 16.2 Overlap-stratified diagnostic

基于已经生成的 predictions，未重新 inference。

| Subset | N QA | Unique images | Zero EM | LoRA EM | Δ | Gain | Loss | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All test | 451 | 203 | 51.66% | 56.98% | +5.32 pp | 42 | 18 | +24 |
| Image seen in SFT train | 402 | 181 | 51.49% | 57.21% | +5.72 pp | 39 | 16 | +23 |
| Image not seen in SFT train | 49 | 22 | 53.06% | 55.10% | +2.04 pp | 3 | 2 | +1 |

后一个 subset 很小，而且其中大量 image 来自 SFT validation，因此：

```text
diagnostic only
```

不作为 unseen-image benchmark。

---

## 16.3 Audit B — Tokenizer parity

merged LoRA checkpoint 曾触发 Transformers regex warning。

因此在全部 451 个正式 prompt 上比较：

```text
Base processor/chat template
Merged processor/chat template
Merged tokenizer with explicit regex fix
```

结果：

```text
Chat-template mismatch:
0 / 451

Base vs merged token-ID mismatch:
0 / 451

Base vs fixed-merged token-ID mismatch:
0 / 451
```

vocab：

```text
Base:   151665
Merged: 151665
Fixed:  151665
```

全部 prompt token count：

```text
Base:   27529
Merged: 27529
Fixed:  27529
```

因此：

> warning 没有改变本项目 451 个实际 evaluation prompts 的 tokenization。

所以没有因此重跑 Day 7–9。

---

# 17. Important Pitfalls / Troubleshooting

这一节只记录项目中真实遇到并解决的问题。

## 17.1 Do not confuse source QA split with image-disjoint split

最大的数据协议风险。

一开始直接沿用 dataset-provided train/test 是合理的 benchmark 实现，但 Day 10 audit 发现：

```text
202 / 203 test images
also occur in source train
```

后续如果把项目发展成论文，第一优先级应该重新构造严格 image-disjoint evaluation。

---

## 17.2 Do not treat `eval_token_acc` as VQA performance

ms-swift training log 中：

```text
eval_token_acc ≈ 0.8602
```

这是 teacher-forced token accuracy。

它和：

```text
generated answer exact match
```

不是同一件事。

---

## 17.3 Smoke metrics are debugging only

项目保留过：

```text
train_smoke100
val_smoke20
test_smoke50
```

它们用于：

- 验证 pipeline；
- 检查显存；
- 估算 runtime；
- debug latency setup。

不用于主质量结论。

---

## 17.4 Early latency numbers were superseded

最初 evaluator 顺带记录：

```text
preprocess / generate / E2E
```

但 evaluator 同时承担：

- correctness evaluation；
- per-sample memory reset；
- result serialization。

因此后续专门写：

```text
src/benchmark_day8_latency.py
```

作为正式 latency benchmark。

公开结果只使用 repeated benchmark。

---

## 17.5 Dynamic memory is not total peak GPU memory

Day 6–8 evaluator 的：

```text
dynamic_peak_allocated_gib
```

只代表 PyTorch allocation 相对模型 baseline 的增量。

不能改名为：

```text
GPU peak memory
```

---

## 17.6 Raw image pixels are not visual tokens

Day 9 统计过 affected images 的 raw width/height/pixel area。

但：

```text
raw pixels
!=
processed pixels
!=
visual tokens
```

Qwen processor 还会执行：

- resizing；
- aspect-ratio preservation；
- patch alignment；
- merge。

所以 resolution 分析以：

```text
measured visual_token_count
```

为主。

---

## 17.7 `question_hint` is only a heuristic

早期 casebook 有自动 hint，例如：

```text
yes_no
laterality
modality_plane
...
```

它可能错误地把：

```text
"is this T1, T2, or FLAIR?"
```

归入 yes/no-like category。

最终公开 error analysis 不使用它作为正式 taxonomy，而使用人工 semantic-target taxonomy。

---

## 17.8 Strict EM can manufacture apparent gains/losses

典型：

```text
right side
vs
right sided

both
vs
both sides

T2
vs
T2-weighted

nodules
vs
nodular opacities
```

所以 Day 9 增加 semantic-aware review。

---

## 17.9 Qualitative cases are not independent samples

一些 affected images 对应多个近重复 QA。

例如：

```text
fd7bcec7d8:
4 affected QA
```

因此：

```text
42 QA gains
```

不能解释为：

```text
42 independent medical images improved
```

---

## 17.10 Casebook image-size bug

构建 review casebook 时曾出现：

```text
KeyError
```

因为原 prediction table 没有保留所有后续 image-width/height 字段。

最终通过重新读取 dataset image metadata 补齐 dimension，而没有重新 inference。

经验：

> 分析阶段需要的 metadata 应尽量在正式 prediction CSV 首次生成时就保存齐全。

---

## 17.11 Tokenizer regex warning

merged model inference 出现 tokenizer regex warning。

不要仅根据 warning 直接重跑整个 benchmark。

最终通过 Day 10 token-ID parity audit 确认：

```text
0 / 451 prompt mismatches
```

所以现有结果可以保留。

---

## 17.12 SLAKE was probed but not adopted

项目早期确认 SLAKE 可读取，但为了在有限时间内保持：

```text
one model
one main dataset
one evaluator
one controlled experiment chain
```

最终选择 VQA-RAD，不并行维护第二套正式 benchmark。

这属于 scope control，而不是实验失败。

---

# 18. Files and Result Mapping

以下是**原始实验 workspace** 中的重要相对路径。

公开 release 不一定全部包含。

## Dataset / split

```text
data/vqa_rad/
data/vqa_rad_sft/train.jsonl
data/vqa_rad_sft/val.jsonl
data/vqa_rad_sft/split_stats.json

manifests/vqa_rad_test_full451.csv
manifests/vqa_rad_test_smoke50.csv
```

数据集原图 / cache 不上传 GitHub。

## Training

```text
scripts/day7_lora_smoke.sh
scripts/day7_lora_1ep.sh

outputs/day7_lora_1ep/v0-20260925-113629/args.json
outputs/day7_lora_1ep/v0-20260925-113629/logging.jsonl
```

checkpoint / merged weights 不上传。

## Evaluation

```text
src/eval_vqa_rad_zero_shot.py
src/eval_vqa_rad_resolution.py
src/benchmark_day8_latency.py
```

## Day 6 / Day 7 results

```text
results/vqa_rad/zero_shot_full451_summary.json
results/vqa_rad/zero_shot_full451_predictions.csv

results/vqa_rad/lora_1ep_full451_summary.json
results/vqa_rad/lora_1ep_full451_predictions.csv

results/vqa_rad/day7_zero_vs_lora_comparison.csv
```

公开仓库优先保留 summary / compact metrics，不默认上传 full prediction table。

## Day 8

```text
results/vqa_rad/day8_resolution/
```

关键：

```text
vt256_full451_summary.json
vt512_full451_summary.json
vt1024_full451_summary.json

day8_latency_repeated.csv
day8_latency_repeated_summary.json
```

## Day 9

```text
results/vqa_rad/day9_analysis/
```

关键：

```text
day9_master_comparison.csv
day9_condition_summary.csv
day9_flip_summary.csv
day9_native_vt_bins.csv

day9_review_casebook_taxonomy_stage1.csv
day9_transition_events.csv
day9_transition_events_final.csv

qualitative_cases/
```

## Public figure source tables

Day 10 release assets 中最终整理：

```text
experiments/metrics/main_zero_vs_lora.csv
experiments/metrics/visual_budget_quality.csv
experiments/metrics/latency_repeated_summary.csv
experiments/metrics/semantic_flip_profile.csv
experiments/metrics/strict_vs_semantic_net.csv
```

---

# 19. What Should / Should Not Be Public

## Public

建议公开：

```text
README.md
docs/PROJECT_RECORD.md
docs/THIRD_PARTY.md

assets/figures/*

selected src/*.py
clean public scripts/*.sh

experiments/metrics/*.csv
small summary JSON if needed

.gitignore
LICENSE
dependency file
```

## Do not upload

```text
models/
outputs/
checkpoint*/
merged model weights

data/
Hugging Face cache/

raw training logs
screen output
wandb cache

full experiment cache

server-specific env.sh

secrets / .env / tokens
```

完整 prediction CSV 默认不上传，除非后续明确确认必要性、体积和 dataset redistribution 边界。

---

# 20. Current Public-facing Results

## 20.1 Main adaptation result

```text
Qwen2.5-VL-3B Zero-shot:
51.66% overall normalized EM

1-epoch LoRA:
56.98%

Δ:
+5.32 pp
```

限定语：

```text
VQA-RAD dataset-provided QA split
project-normalized evaluator
not image-disjoint
```

## 20.2 Main efficiency result

VT512：

```text
Full-test Avg VT:
871.6 -> 422.8
-51.5%

Overall EM:
56.98% -> 57.87%
(no observed loss on this fixed split)

Repeated benchmark mean E2E:
394.7 ms -> 222.2 ms

Paired mean E2E reduction:
34.2%

Paired median E2E reduction:
45.5%

Measurements faster:
95%
```

安全表述：

> roughly half the visual-token workload with substantially lower latency and no observed aggregate EM loss on the fixed QA split.

不要写：

> VT512 improves medical accuracy.

## 20.3 Aggressive compression boundary

VT256：

```text
Avg VT:
871.6 -> 238.4
-72.6%

Overall EM:
56.98% -> 54.32%
-2.66 pp

Strict transition net:
-12 QA

Semantic-aware net:
-13 QA
```

---

# 21. Interview-oriented Technical Takeaways

这个项目最值得在面试中强调的是实验能力，不是“新方法”。

## 21.1 VLM pipeline

可以讲清：

```text
dataset
→ multimodal prompt
→ Qwen processor
→ visual tokens
→ LLM generation
→ normalization
→ evaluation
```

## 21.2 Real multimodal training

不是 API 调用，而是实际：

```text
LoRA SFT
gradient accumulation
checkpoint
merge
full inference evaluation
```

## 21.3 Controlled experiment design

Day 8 固定：

```text
model
prompt
evaluator
test split
generation
```

只改变：

```text
max_pixels / visual budget
```

## 21.4 Efficiency benchmark

能够解释：

```text
timing boundary
warm-up
model-load exclusion
repeat
order rotation
paired comparison
mean / median / p95
```

## 21.5 Error analysis

不仅报告：

```text
overall EM
```

还分析：

```text
closed / open
gain / loss
semantic target
strict-EM artifacts
qualitative failure
```

## 21.6 Scientific caution

能够主动指出：

```text
dataset split image reuse
non-official metric
small diagnostic subsets
non-monotonic single-sample behavior
```

而不是只挑有利结果。

---

# 22. Follow-up Research Ideas

以下是**后续方向**，不是本项目已完成贡献。

## 22.1 Strict image-disjoint Medical VQA evaluation

最高优先级。

重新按 image identity 构造：

```text
train / val / test
all image-disjoint
```

然后重新回答：

> LoRA 的 domain adaptation 是否能够 generalize 到真正 unseen medical images？

这是当前 benchmark split 最关键的限制。

## 22.2 Adaptive visual-token budget

Day 8 / Day 9 显示：

- 大量 QA 在较低 budget 下保持稳定；
- VT256 对 morphology / finding / visibility 出现更多 regression；
- 单样本 correctness 具有 non-monotonicity。

可研究：

> 根据 question / uncertainty / evidence demand 自适应分配视觉预算，而不是所有 image 固定同一个 `max_pixels`。

## 22.3 Localization / laterality robustness

LoRA 的 losses 在 affected subset 中明显包含 laterality regression。

可研究：

- radiology orientation awareness；
- laterality-specific supervision；
- localization auxiliary objective；
- evidence grounding。

## 22.4 Visual Evidence Sensitivity

一个更基础的问题：

> 当 visual tokens 大幅减少但 aggregate score 基本不变时，模型究竟使用了多少真正的细粒度视觉证据？

可以结合：

- evidence localization；
- perturbation；
- attribution；
- adaptive token selection。

## 22.5 Better semantic evaluation for Medical VQA

Strict EM 对：

```text
synonyms
granularity
medical terminology
```

比较敏感。

后续可在不掩盖错误的前提下研究：

- canonical medical answer normalization；
- ontology-aware matching；
- LLM-as-judge with constrained rubric；
- semantic equivalence audit。

任何替代 metric 都需要和 strict EM 同时报告，避免过度宽松。

---

# 23. Frozen Conclusions

截至 Day 10，项目主要实验冻结。

可以公开支持的结论：

1. 在当前 VQA-RAD dataset-provided QA split 与项目 evaluator 下，1-epoch LoRA 将 overall normalized EM 从 **51.66% 提升到 56.98%**。
2. LoRA 的 gain/loss 并非均匀：affected subset 中 gains 更集中于 finding presence/absence，而 losses 中 localization/laterality 较突出。
3. 固定 LoRA 后，VT512 将 full-test Avg VT 从 **871.6 降到 422.8（−51.5%）**，同时 fixed split 上没有观察到 overall EM loss。
4. repeated benchmark 中 VT512 相对 native 的 paired mean E2E reduction 为 **34.2%**，median reduction 为 **45.5%**。
5. 更激进的 VT256 将 Avg VT 降低 **72.6%**，同时 overall EM 下降 **2.66 pp**，semantic-aware review 也显示明显净 regression。
6. strict EM 会把少量 wording / granularity transition 放大成 gain/loss，因此 semantic-aware error analysis 对 Medical VQA 很重要。
7. 当前 VQA-RAD QA split 高度复用 train/test images，因此这些结果**不能证明 completely unseen-image generalization**。

---

# 24. Project Status

```text
Day 6:
Zero-shot baseline
DONE

Day 7:
LoRA SFT
DONE

Day 8:
Visual-budget + latency benchmark
DONE

Day 9:
Fine-grained + qualitative error analysis
DONE

Day 10:
Validity audit
GitHub visualization
README / PROJECT_RECORD
Third-party / license documentation
Public source-code cleanup
Compact final metrics
Release security audit
DONE
```

项目实验与公开仓库整理均已冻结：

```text
no new large experiment
```

剩余操作仅为本地 Git 初始化、逐项 staging、commit 与 push。
