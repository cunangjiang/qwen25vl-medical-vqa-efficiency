import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_from_disk


def image_hash(img):
    img = img.convert("RGB")
    h = hashlib.sha1()
    h.update(str(img.size).encode("utf-8"))
    h.update(img.tobytes())
    return h.hexdigest()


def jsonl_hashes(path):
    hashes = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            hashes.append(Path(row["images"][0]).stem)
    return hashes


def main():
    parser = argparse.ArgumentParser(description="Audit image identity overlap across SFT train/val and VQA-RAD test.")
    parser.add_argument("--dataset", default="data/vqa_rad")
    parser.add_argument("--train-jsonl", default="data/vqa_rad_sft/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/vqa_rad_sft/val.jsonl")
    parser.add_argument("--output", default="results/day10_validity_audit/image_overlap.json")
    args = parser.parse_args()

    train_list = jsonl_hashes(args.train_jsonl)
    val_list = jsonl_hashes(args.val_jsonl)
    train = set(train_list)
    val = set(val_list)
    test_ds = load_from_disk(args.dataset)["test"]
    test_list = [image_hash(ex["image"]) for ex in test_ds]
    test = set(test_list)
    source_train = train | val

    result = {
        "sft_train_qa": len(train_list),
        "sft_val_qa": len(val_list),
        "sft_train_unique_images": len(train),
        "sft_val_unique_images": len(val),
        "test_unique_images": len(test),
        "train_val_image_overlap": len(train & val),
        "sft_train_test_unique_image_overlap": len(train & test),
        "val_test_unique_image_overlap": len(val & test),
        "source_train_test_unique_image_overlap": len(source_train & test),
        "test_qa_on_sft_train_images": sum(h in train for h in test_list),
        "test_qa_on_val_images": sum(h in val for h in test_list),
        "test_qa_on_any_source_train_image": sum(h in source_train for h in test_list),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
