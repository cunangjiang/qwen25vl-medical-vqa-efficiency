import argparse
import csv

from datasets import load_from_disk
from transformers import AutoProcessor, AutoTokenizer


PROMPT_PREFIX = (
    "Answer the medical visual question with only a short answer. "
    "Do not explain your reasoning. "
    "For yes/no questions, answer only yes or no.\n"
    "Question: "
)


def main():
    parser = argparse.ArgumentParser(description="Compare base and merged-checkpoint prompt tokenization on the evaluation manifest.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--merged-model", required=True)
    parser.add_argument("--dataset", default="data/vqa_rad")
    parser.add_argument("--manifest", default="manifests/vqa_rad_test_full451.csv")
    args = parser.parse_args()

    test = load_from_disk(args.dataset)["test"]
    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    base_processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    merged_processor = AutoProcessor.from_pretrained(args.merged_model, local_files_only=True)
    fixed_tokenizer = AutoTokenizer.from_pretrained(
        args.merged_model,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    chat_mismatch = 0
    merged_mismatch = 0
    fixed_mismatch = 0
    total_base = total_merged = total_fixed = 0

    for row in rows:
        idx = int(row["dataset_index"])
        prompt = PROMPT_PREFIX + row["question"]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": test[idx]["image"]},
                {"type": "text", "text": prompt},
            ],
        }]

        base_text = base_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        merged_text = merged_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_mismatch += int(base_text != merged_text)

        base_ids = base_processor.tokenizer(base_text, add_special_tokens=False)["input_ids"]
        merged_ids = merged_processor.tokenizer(merged_text, add_special_tokens=False)["input_ids"]
        fixed_ids = fixed_tokenizer(merged_text, add_special_tokens=False)["input_ids"]
        merged_mismatch += int(base_ids != merged_ids)
        fixed_mismatch += int(base_ids != fixed_ids)
        total_base += len(base_ids)
        total_merged += len(merged_ids)
        total_fixed += len(fixed_ids)

    print(f"Prompts: {len(rows)}")
    print(f"Chat-template mismatches: {chat_mismatch}")
    print(f"Base vs merged token-ID mismatches: {merged_mismatch}")
    print(f"Base vs fixed-merged token-ID mismatches: {fixed_mismatch}")
    print(f"Total prompt tokens: base={total_base}, merged={total_merged}, fixed={total_fixed}")


if __name__ == "__main__":
    main()
