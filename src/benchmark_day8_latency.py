import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


PROMPT_TEMPLATE = (
    "Answer the medical visual question with only a short answer. "
    "Do not explain your reasoning. "
    "For yes/no questions, answer only yes or no.\n"
    "Question: {question}"
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_processor(model_path, condition):
    kwargs = {
        "local_files_only": True,
    }

    if condition != "lora_native":
        vt = int(condition.replace("vt", ""))
        kwargs["min_pixels"] = 4 * 28 * 28
        kwargs["max_pixels"] = vt * 28 * 28

    return AutoProcessor.from_pretrained(
        model_path,
        **kwargs,
    )


def prepare_inputs(processor, image, question):
    prompt = PROMPT_TEMPLATE.format(
        question=question
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

    grid = inputs.get("image_grid_thw", None)

    if grid is None:
        visual_tokens = 0
    else:
        merge_size = (
            processor.image_processor.merge_size
        )

        visual_tokens = int(
            sum(
                int(t) * int(h) * int(w)
                // (merge_size ** 2)
                for t, h, w in grid.tolist()
            )
        )

    return inputs, visual_tokens


def run_one(
    model,
    processor,
    image,
    question,
    max_new_tokens,
    record_timing=True,
):
    # Dataset retrieval / PIL conversion are intentionally
    # outside the benchmark timing.

    torch.cuda.synchronize()

    e2e_start = time.perf_counter()

    preprocess_start = time.perf_counter()

    inputs, visual_tokens = prepare_inputs(
        processor,
        image,
        question,
    )

    preprocess_end = time.perf_counter()

    # Transfer is included in E2E but not preprocess/generate.
    inputs = inputs.to("cuda")

    torch.cuda.synchronize()

    generate_start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()

    generate_end = time.perf_counter()

    input_len = inputs["input_ids"].shape[1]

    generated_trimmed = generated[:, input_len:]

    output_token_count = int(
        generated_trimmed.shape[1]
    )

    prediction = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    e2e_end = time.perf_counter()

    if not record_timing:
        return None

    return {
        "visual_token_count": visual_tokens,
        "output_token_count": output_token_count,
        "prediction": prediction,
        "preprocess_ms":
            (preprocess_end - preprocess_start) * 1000,
        "generate_ms":
            (generate_end - generate_start) * 1000,
        "e2e_ms":
            (e2e_end - e2e_start) * 1000,
    }


def percentile95(x):
    return float(np.percentile(x, 95))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        required=True,
    )
    parser.add_argument(
        "--summary-json",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    conditions = [
        "vt256",
        "vt512",
        "vt1024",
        "lora_native",
    ]

    # Latin-square-like rotation:
    # every condition appears once in every order position.
    orders = [
        [
            "vt256",
            "vt512",
            "vt1024",
            "lora_native",
        ],
        [
            "vt512",
            "vt1024",
            "lora_native",
            "vt256",
        ],
        [
            "vt1024",
            "lora_native",
            "vt256",
            "vt512",
        ],
        [
            "lora_native",
            "vt256",
            "vt512",
            "vt1024",
        ],
    ]

    print("===== LOAD DATA =====")

    ds = load_from_disk(args.dataset)["test"]
    manifest = pd.read_csv(args.manifest)

    if "dataset_index" not in manifest.columns:
        raise RuntimeError(
            "Manifest must contain dataset_index."
        )

    print("Test rows:", len(ds))
    print("Benchmark rows:", len(manifest))

    print()
    print("===== LOAD PROCESSORS =====")

    processors = {}

    for condition in conditions:
        processors[condition] = make_processor(
            args.model,
            condition,
        )

        ip = processors[condition].image_processor

        print(
            condition,
            "min_pixels=",
            getattr(ip, "min_pixels", None),
            "max_pixels=",
            getattr(ip, "max_pixels", None),
        )

    print()
    print("===== LOAD MODEL ONCE =====")

    load_start = time.perf_counter()

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
        )
    )

    model.eval()

    load_seconds = time.perf_counter() - load_start

    print(
        f"Model load time: {load_seconds:.2f} s"
    )

    # ------------------------------------------------
    # Warm-up each resolution condition.
    # ------------------------------------------------
    print()
    print("===== WARM-UP =====")

    warmup_rows = manifest.iloc[
        :args.warmup_samples
    ]

    for condition in conditions:
        print(
            f"Warm-up: {condition} "
            f"({len(warmup_rows)} samples)"
        )

        processor = processors[condition]

        for _, m in warmup_rows.iterrows():
            idx = int(m["dataset_index"])

            row = ds[idx]
            image = row["image"].convert("RGB")
            question = str(row["question"])

            run_one(
                model,
                processor,
                image,
                question,
                args.max_new_tokens,
                record_timing=False,
            )

    torch.cuda.synchronize()

    print("Warm-up complete.")

    # ------------------------------------------------
    # Formal repeated benchmark.
    # ------------------------------------------------
    print()
    print("===== FORMAL BENCHMARK =====")

    records = []

    for repeat_idx, order in enumerate(
        orders,
        start=1,
    ):
        print()
        print(
            f"Repeat {repeat_idx}/{len(orders)}"
        )
        print("Order:", " -> ".join(order))

        for order_pos, condition in enumerate(
            order,
            start=1,
        ):
            processor = processors[condition]

            print(
                f"  [{order_pos}/4] {condition}"
            )

            for sample_pos, (_, m) in enumerate(
                manifest.iterrows(),
                start=1,
            ):
                idx = int(m["dataset_index"])

                row = ds[idx]

                # Keep these outside timing, matching
                # the Day6/Day7 evaluator convention.
                image = row["image"].convert("RGB")
                question = str(row["question"])

                result = run_one(
                    model,
                    processor,
                    image,
                    question,
                    args.max_new_tokens,
                    record_timing=True,
                )

                result.update(
                    {
                        "repeat": repeat_idx,
                        "order_position": order_pos,
                        "condition": condition,
                        "sample_position": sample_pos,
                        "dataset_index": idx,
                    }
                )

                records.append(result)

    df = pd.DataFrame(records)

    out_csv = Path(args.output_csv)
    out_json = Path(args.summary_json)

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_csv,
        index=False,
    )

    # ------------------------------------------------
    # Aggregate statistics.
    # ------------------------------------------------
    summary = {
        "model": args.model,
        "dataset": "VQA-RAD",
        "num_benchmark_samples":
            int(len(manifest)),
        "num_repeats":
            int(len(orders)),
        "warmup_samples_per_condition":
            int(args.warmup_samples),
        "model_load_seconds":
            load_seconds,
        "timing_protocol": {
            "model_loaded_once": True,
            "warmup_before_measurement": True,
            "empty_cache_inside_timing": False,
            "reset_peak_memory_inside_timing": False,
            "cuda_synchronize_for_generate":
                True,
            "dataset_retrieval_in_e2e":
                False,
            "pil_conversion_in_e2e":
                False,
            "gpu_transfer_in_e2e":
                True,
            "decode_in_e2e":
                True,
            "budget_order":
                orders,
        },
        "conditions": {},
    }

    for condition in conditions:
        x = df[
            df["condition"] == condition
        ]

        summary["conditions"][condition] = {
            "n_measurements":
                int(len(x)),
            "avg_visual_tokens":
                float(
                    x[
                        "visual_token_count"
                    ].mean()
                ),
            "avg_output_tokens":
                float(
                    x[
                        "output_token_count"
                    ].mean()
                ),
            "generate_ms": {
                "mean":
                    float(
                        x["generate_ms"].mean()
                    ),
                "median":
                    float(
                        x["generate_ms"].median()
                    ),
                "p95":
                    percentile95(
                        x["generate_ms"]
                    ),
            },
            "e2e_ms": {
                "mean":
                    float(
                        x["e2e_ms"].mean()
                    ),
                "median":
                    float(
                        x["e2e_ms"].median()
                    ),
                "p95":
                    percentile95(
                        x["e2e_ms"]
                    ),
            },
        }

    # ------------------------------------------------
    # Paired comparisons against LoRA-native.
    # ------------------------------------------------
    paired = {}

    keys = [
        "repeat",
        "dataset_index",
    ]

    native = df[
        df["condition"] == "lora_native"
    ][
        keys
        + [
            "generate_ms",
            "e2e_ms",
        ]
    ].copy()

    native = native.rename(
        columns={
            "generate_ms":
                "native_generate_ms",
            "e2e_ms":
                "native_e2e_ms",
        }
    )

    for condition in [
        "vt256",
        "vt512",
        "vt1024",
    ]:
        cur = df[
            df["condition"] == condition
        ][
            keys
            + [
                "generate_ms",
                "e2e_ms",
            ]
        ]

        m = cur.merge(
            native,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        gen_reduction = (
            1
            - m["generate_ms"]
            / m["native_generate_ms"]
        ) * 100

        e2e_reduction = (
            1
            - m["e2e_ms"]
            / m["native_e2e_ms"]
        ) * 100

        paired[condition] = {
            "n_pairs":
                int(len(m)),
            "generate_reduction_pct": {
                "mean":
                    float(
                        gen_reduction.mean()
                    ),
                "median":
                    float(
                        gen_reduction.median()
                    ),
            },
            "e2e_reduction_pct": {
                "mean":
                    float(
                        e2e_reduction.mean()
                    ),
                "median":
                    float(
                        e2e_reduction.median()
                    ),
            },
            "fraction_e2e_faster":
                float(
                    (
                        m["e2e_ms"]
                        <
                        m["native_e2e_ms"]
                    ).mean()
                ),
        }

    summary["paired_vs_lora_native"] = paired

    with out_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------
    # Human-readable table.
    # ------------------------------------------------
    print()
    print("===== AGGREGATE RESULTS =====")
    print()

    print(
        f"{'condition':<14}"
        f"{'avg_vt':>10}"
        f"{'out_tok':>10}"
        f"{'gen_mean':>11}"
        f"{'gen_med':>10}"
        f"{'gen_p95':>10}"
        f"{'e2e_mean':>11}"
        f"{'e2e_med':>10}"
        f"{'e2e_p95':>10}"
    )

    for condition in conditions:
        x = summary[
            "conditions"
        ][condition]

        print(
            f"{condition:<14}"
            f"{x['avg_visual_tokens']:>10.1f}"
            f"{x['avg_output_tokens']:>10.2f}"
            f"{x['generate_ms']['mean']:>11.1f}"
            f"{x['generate_ms']['median']:>10.1f}"
            f"{x['generate_ms']['p95']:>10.1f}"
            f"{x['e2e_ms']['mean']:>11.1f}"
            f"{x['e2e_ms']['median']:>10.1f}"
            f"{x['e2e_ms']['p95']:>10.1f}"
        )

    print()
    print("===== PAIRED vs LoRA-native =====")
    print()

    print(
        f"{'condition':<12}"
        f"{'gen_red_mean':>14}"
        f"{'gen_red_med':>14}"
        f"{'e2e_red_mean':>14}"
        f"{'e2e_red_med':>14}"
        f"{'faster_frac':>14}"
    )

    for condition in [
        "vt256",
        "vt512",
        "vt1024",
    ]:
        x = paired[condition]

        print(
            f"{condition:<12}"
            f"{x['generate_reduction_pct']['mean']:>13.1f}%"
            f"{x['generate_reduction_pct']['median']:>13.1f}%"
            f"{x['e2e_reduction_pct']['mean']:>13.1f}%"
            f"{x['e2e_reduction_pct']['median']:>13.1f}%"
            f"{x['fraction_e2e_faster']*100:>13.1f}%"
        )

    print()
    print("Saved:")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
