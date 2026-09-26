import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--question",
        default="What abnormality is visible in this medical image? Answer briefly."
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("===== RUNTIME =====")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch =", torch.__version__)
    print("CUDA available =", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Visible GPU =", torch.cuda.get_device_name(0))
    print()

    print("===== LOAD PROCESSOR =====")
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
    )

    print("===== LOAD MODEL =====")
    t0 = time.perf_counter()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()

    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - t0

    print(f"Model load time: {model_load_s:.2f} s")
    print()

    image = Image.open(args.image).convert("RGB")
    print("===== IMAGE =====")
    print("Path:", args.image)
    print("Original size:", image.size)
    print()

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
                    "text": args.question,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print("===== PROCESSED INPUT =====")

    image_grid_thw = None
    visual_tokens = None

    if "image_grid_thw" in inputs:
        image_grid_thw = inputs["image_grid_thw"].tolist()
        print("image_grid_thw:", image_grid_thw)

        merge_size = getattr(processor.image_processor, "merge_size", 2)

        visual_tokens = sum(
            int(t * h * w // (merge_size ** 2))
            for t, h, w in image_grid_thw
        )

        print("merge_size:", merge_size)
        print("visual token count:", visual_tokens)

    print("input text tokens:", inputs["input_ids"].shape[1])
    print()

    inputs = inputs.to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("===== GENERATE =====")
    t1 = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    generation_s = time.perf_counter() - t1

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

    print()
    print("===== RESULT =====")
    print("Question:", args.question)
    print("Answer:", response)

    print()
    print("===== DIAGNOSTICS =====")
    print(f"Generation time: {generation_s:.3f} s")
    print(f"Peak allocated GPU memory: {peak_allocated_gb:.3f} GiB")
    print(f"Peak reserved GPU memory:  {peak_reserved_gb:.3f} GiB")

    result = {
        "model": args.model,
        "image": args.image,
        "image_size": list(image.size),
        "question": args.question,
        "answer": response,
        "image_grid_thw": image_grid_thw,
        "visual_token_count": visual_tokens,
        "input_text_tokens": int(inputs["input_ids"].shape[1]),
        "model_load_seconds": model_load_s,
        "generation_seconds": generation_s,
        "peak_allocated_gpu_gib": peak_allocated_gb,
        "peak_reserved_gpu_gib": peak_reserved_gb,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print()
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
