import argparse
import csv
import json
import os
import random
import re
import string
import time
import unicodedata
from collections import Counter

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    """
    Internal project normalization, not claimed as an official VQA-RAD metric:
    - unicode normalize
    - lowercase
    - punctuation -> spaces
    - remove English articles a/an/the
    - collapse whitespace
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()

    table = str.maketrans({c: " " for c in string.punctuation})
    text = text.translate(table)

    words = text.split()
    words = [w for w in words if w not in {"a", "an", "the"}]

    return " ".join(words)


def normalize_closed_prediction(text: str) -> str:
    """
    For yes/no questions:
    take the first standalone yes or no if one exists.
    """
    text_norm = unicodedata.normalize("NFKC", str(text)).lower()
    match = re.search(r"\b(yes|no)\b", text_norm)

    if match:
        return match.group(1)

    return normalize_answer(text)


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gt_tokens = normalize_answer(gt).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def mean(values):
    return float(sum(values) / len(values)) if values else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/vqa_rad")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)

    # Day 8: resolution / visual-token budget control.
    parser.add_argument(
        "--budget-name",
        default="custom",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=4 * 28 * 28,
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("===== CONFIG =====")
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Manifest:", args.manifest)
    print("Seed:", args.seed)
    print("Budget name:", args.budget_name)
    print("Min pixels:", args.min_pixels)
    print("Max pixels:", args.max_pixels)
    print(
        "Target max visual tokens:",
        args.max_pixels / (28 * 28),
    )
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Visible GPU:", torch.cuda.get_device_name(0))
    print()

    print("===== LOAD LOCAL DATASET =====")
    ds = load_from_disk(args.dataset)
    test_ds = ds["test"]

    print("Test rows:", len(test_ds))
    print()

    manifest_rows = []

    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    print("Manifest rows:", len(manifest_rows))
    print()

    print("===== LOAD PROCESSOR =====")
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )

    print(
        "Processor min_pixels:",
        processor.image_processor.min_pixels,
    )
    print(
        "Processor max_pixels:",
        processor.image_processor.max_pixels,
    )
    print()

    print("===== LOAD MODEL =====")
    load_start = time.perf_counter()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )

    model.eval()

    # Greedy generation. Remove irrelevant sampling parameters inherited
    # from the model's generation_config to avoid warning messages.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    torch.cuda.synchronize()

    model_load_s = time.perf_counter() - load_start

    base_allocated_gib = torch.cuda.memory_allocated() / 1024**3
    base_reserved_gib = torch.cuda.memory_reserved() / 1024**3

    print(f"Model load time: {model_load_s:.2f} s")
    print(f"Base allocated memory: {base_allocated_gib:.3f} GiB")
    print(f"Base reserved memory : {base_reserved_gib:.3f} GiB")
    print()

    os.makedirs(
        os.path.dirname(args.output_csv) or ".",
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(args.summary_json) or ".",
        exist_ok=True,
    )

    fieldnames = [
        "smoke_id",
        "dataset_index",
        "question_type",
        "question",
        "gt",
        "prediction_raw",
        "gt_normalized",
        "prediction_normalized",
        "exact_match",
        "token_f1",
        "image_width",
        "image_height",
        "image_grid_thw",
        "visual_token_count",
        "input_token_count",
        "output_token_count",
        "preprocess_ms",
        "generate_ms",
        "e2e_ms",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "dynamic_peak_allocated_gib",
    ]

    exact_all = []
    exact_closed = []
    exact_open = []
    open_f1s = []

    preprocess_times = []
    generate_times = []
    e2e_times = []
    visual_token_counts = []
    dynamic_memories = []

    with open(
        args.output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        fout.flush()

        total = len(manifest_rows)

        for run_idx, row in enumerate(manifest_rows, start=1):

            dataset_index = int(row["dataset_index"])
            qtype = row["question_type"]

            ex = test_ds[dataset_index]

            question = ex["question"]
            gt = ex["answer"]
            image = ex["image"].convert("RGB")

            prompt = (
                "Answer the medical visual question with only a short answer. "
                "Do not explain your reasoning. "
                "For yes/no questions, answer only yes or no.\n"
                f"Question: {question}"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            # Disk / Dataset retrieval happened above.
            # E2E timer starts here.
            e2e_start = time.perf_counter()
            preprocess_start = time.perf_counter()

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(
                messages
            )

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            preprocess_ms = (
                time.perf_counter() - preprocess_start
            ) * 1000

            image_grid_thw = None
            visual_tokens = None

            if "image_grid_thw" in inputs:
                grid = inputs["image_grid_thw"]
                image_grid_thw = grid.tolist()

                merge_size = getattr(
                    processor.image_processor,
                    "merge_size",
                    2,
                )

                visual_tokens = sum(
                    int(t * h * w // (merge_size ** 2))
                    for t, h, w in image_grid_thw
                )

            input_token_count = int(
                inputs["input_ids"].shape[1]
            )

            inputs = inputs.to("cuda")

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            gen_start = time.perf_counter()

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            torch.cuda.synchronize()

            generate_ms = (
                time.perf_counter() - gen_start
            ) * 1000

            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            output_token_count = int(
                generated_ids_trimmed[0].shape[0]
            )

            prediction = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            e2e_ms = (
                time.perf_counter() - e2e_start
            ) * 1000

            peak_allocated_gib = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            peak_reserved_gib = (
                torch.cuda.max_memory_reserved() / 1024**3
            )

            dynamic_peak_allocated_gib = max(
                0.0,
                peak_allocated_gib - base_allocated_gib,
            )

            gt_norm = normalize_answer(gt)

            if qtype == "closed":
                pred_norm = normalize_closed_prediction(
                    prediction
                )
            else:
                pred_norm = normalize_answer(
                    prediction
                )

            exact = int(pred_norm == gt_norm)

            f1 = token_f1(prediction, gt)

            exact_all.append(exact)

            if qtype == "closed":
                exact_closed.append(exact)
            else:
                exact_open.append(exact)
                open_f1s.append(f1)

            preprocess_times.append(preprocess_ms)
            generate_times.append(generate_ms)
            e2e_times.append(e2e_ms)

            if visual_tokens is not None:
                visual_token_counts.append(
                    visual_tokens
                )

            dynamic_memories.append(
                dynamic_peak_allocated_gib
            )

            result_row = {
                "smoke_id": row["smoke_id"],
                "dataset_index": dataset_index,
                "question_type": qtype,
                "question": question,
                "gt": gt,
                "prediction_raw": prediction,
                "gt_normalized": gt_norm,
                "prediction_normalized": pred_norm,
                "exact_match": exact,
                "token_f1": f"{f1:.6f}",
                "image_width": image.width,
                "image_height": image.height,
                "image_grid_thw": json.dumps(
                    image_grid_thw
                ),
                "visual_token_count": visual_tokens,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "preprocess_ms": f"{preprocess_ms:.3f}",
                "generate_ms": f"{generate_ms:.3f}",
                "e2e_ms": f"{e2e_ms:.3f}",
                "peak_allocated_gib":
                    f"{peak_allocated_gib:.4f}",
                "peak_reserved_gib":
                    f"{peak_reserved_gib:.4f}",
                "dynamic_peak_allocated_gib":
                    f"{dynamic_peak_allocated_gib:.4f}",
            }

            writer.writerow(result_row)
            fout.flush()

            print(
                f"[{run_idx:02d}/{total:02d}] "
                f"{qtype:6s} | "
                f"tokens={visual_tokens} | "
                f"GT={gt!r} | "
                f"PRED={prediction!r} | "
                f"EM={exact} | "
                f"E2E={e2e_ms:.1f} ms"
            )

            # Explicitly release sample tensors.
            del inputs
            del generated_ids
            del generated_ids_trimmed

    summary = {
        "model": args.model,
        "dataset": "VQA-RAD",
        "split": "test",
        "manifest": args.manifest,
        "seed": args.seed,

        "resolution_budget": {
            "budget_name": args.budget_name,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "target_max_visual_tokens":
                args.max_pixels / (28 * 28),
        },

        "num_samples": len(manifest_rows),
        "num_closed": len(exact_closed),
        "num_open": len(exact_open),

        "metric_definition": {
            "main": "project normalized exact match",
            "closed":
                "extract first standalone yes/no",
            "open":
                "lowercase + punctuation removal + "
                "article removal + whitespace normalization",
            "open_token_f1":
                "diagnostic only; not claimed as official VQA-RAD metric",
        },

        "accuracy_overall": mean(exact_all),
        "accuracy_closed": mean(exact_closed),
        "accuracy_open": mean(exact_open),
        "open_token_f1": mean(open_f1s),

        "avg_visual_tokens": mean(
            visual_token_counts
        ),

        "latency_ms": {
            "avg_preprocess": mean(
                preprocess_times
            ),
            "avg_generate": mean(
                generate_times
            ),
            "avg_e2e": mean(
                e2e_times
            ),
        },

        "memory_gib": {
            "model_base_allocated":
                base_allocated_gib,
            "model_base_reserved":
                base_reserved_gib,
            "avg_dynamic_peak_allocated":
                mean(dynamic_memories),
        },

        "model_load_seconds": model_load_s,
    }

    with open(
        args.summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("===== SUMMARY =====")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))

    print()
    print("Predictions:", args.output_csv)
    print("Summary    :", args.summary_json)


if __name__ == "__main__":
    main()
