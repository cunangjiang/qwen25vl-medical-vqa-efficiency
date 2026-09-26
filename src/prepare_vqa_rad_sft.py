import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


PROMPT_PREFIX = (
    "Answer the medical visual question with only a short answer. "
    "Do not explain your reasoning. "
    "For yes/no questions, answer only yes or no.\n"
    "Question: "
)


def image_hash(img):
    img = img.convert("RGB")
    h = hashlib.sha1()
    h.update(str(img.size).encode("utf-8"))
    h.update(img.tobytes())
    return h.hexdigest(), img


def make_record(question, answer, image_path):
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>\n" + PROMPT_PREFIX + question,
            },
            {
                "role": "assistant",
                "content": str(answer).strip(),
            },
        ],
        # ms-swift consumes these generated JSONL files locally.  The absolute
        # path is created at preprocessing time and is never committed.
        "images": [str(Path(image_path).resolve())],
    }


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def closed_open_stats(rows_raw):
    counts = Counter()
    for row in rows_raw:
        ans = str(row["answer"]).strip().lower()
        counts["closed" if ans in {"yes", "no"} else "open"] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Create the image-disjoint SFT train/validation split used in this project."
    )
    parser.add_argument("--dataset", default="data/vqa_rad")
    parser.add_argument("--output-dir", default="data/vqa_rad_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-image-ratio", type=float, default=0.10)
    parser.add_argument("--train-smoke-size", type=int, default=100)
    parser.add_argument("--val-smoke-size", type=int, default=20)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"
    train_smoke = out_dir / "train_smoke100.jsonl"
    val_smoke = out_dir / "val_smoke20.jsonl"
    stats_path = out_dir / "split_stats.json"

    ds = load_from_disk(str(dataset_dir))
    train_ds = ds["train"]
    test_ds = ds["test"]

    raw_train = []
    train_image_hashes = set()

    for idx, ex in enumerate(train_ds):
        img_hash, img = image_hash(ex["image"])
        train_image_hashes.add(img_hash)

        image_path = image_dir / f"{img_hash}.png"
        if not image_path.exists():
            img.save(image_path)

        raw_train.append({
            "dataset_index": idx,
            "image_hash": img_hash,
            "image_path": image_path,
            "question": ex["question"],
            "answer": ex["answer"],
        })

    unique_images = sorted(train_image_hashes)
    rng = random.Random(args.seed)
    rng.shuffle(unique_images)

    n_val_images = max(1, round(len(unique_images) * args.val_image_ratio))
    val_image_hashes = set(unique_images[:n_val_images])

    train_rows_raw = [r for r in raw_train if r["image_hash"] not in val_image_hashes]
    val_rows_raw = [r for r in raw_train if r["image_hash"] in val_image_hashes]

    train_hashes = {r["image_hash"] for r in train_rows_raw}
    val_hashes = {r["image_hash"] for r in val_rows_raw}
    assert not (train_hashes & val_hashes)

    rng.shuffle(train_rows_raw)
    rng.shuffle(val_rows_raw)

    train_rows = [
        make_record(r["question"], r["answer"], r["image_path"])
        for r in train_rows_raw
    ]
    val_rows = [
        make_record(r["question"], r["answer"], r["image_path"])
        for r in val_rows_raw
    ]

    dump_jsonl(train_jsonl, train_rows)
    dump_jsonl(val_jsonl, val_rows)
    dump_jsonl(train_smoke, train_rows[: min(args.train_smoke_size, len(train_rows))])
    dump_jsonl(val_smoke, val_rows[: min(args.val_smoke_size, len(val_rows))])

    test_hashes = {image_hash(ex["image"])[0] for ex in test_ds}
    source_train_test_overlap = train_image_hashes & test_hashes

    stats = {
        "seed": args.seed,
        "validation_image_ratio": args.val_image_ratio,
        "official_train_qa": len(train_ds),
        "official_test_qa": len(test_ds),
        "unique_train_images": len(train_image_hashes),
        "unique_test_images": len(test_hashes),
        "train_unique_images": len(train_hashes),
        "val_unique_images": len(val_hashes),
        "train_val_image_overlap": len(train_hashes & val_hashes),
        "train_qa": len(train_rows),
        "val_qa": len(val_rows),
        "train_question_types": closed_open_stats(train_rows_raw),
        "val_question_types": closed_open_stats(val_rows_raw),
        "official_train_test_image_overlap": len(source_train_test_overlap),
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Train:", train_jsonl)
    print("Val  :", val_jsonl)
    print("Stats:", stats_path)


if __name__ == "__main__":
    main()
