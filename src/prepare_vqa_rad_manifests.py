import argparse
import csv
import random
from pathlib import Path

from datasets import load_from_disk


def question_type(answer):
    value = " ".join(str(answer).strip().lower().split())
    return "closed" if value in {"yes", "no"} else "open"


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["smoke_id", "dataset_index", "question_type", "question", "answer"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Create the full451 and balanced smoke50 manifests used by the evaluator."
    )
    parser.add_argument("--dataset", default="data/vqa_rad")
    parser.add_argument("--output-dir", default="manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-closed", type=int, default=25)
    parser.add_argument("--smoke-open", type=int, default=25)
    args = parser.parse_args()

    test = load_from_disk(args.dataset)["test"]
    all_rows = []
    for idx, ex in enumerate(test):
        all_rows.append({
            "smoke_id": idx,
            "dataset_index": idx,
            "question_type": question_type(ex["answer"]),
            "question": ex["question"],
            "answer": ex["answer"],
        })

    out_dir = Path(args.output_dir)
    write_manifest(out_dir / "vqa_rad_test_full451.csv", all_rows)

    closed = [r for r in all_rows if r["question_type"] == "closed"]
    open_q = [r for r in all_rows if r["question_type"] == "open"]
    rng = random.Random(args.seed)
    selected = (
        rng.sample(closed, min(args.smoke_closed, len(closed)))
        + rng.sample(open_q, min(args.smoke_open, len(open_q)))
    )
    rng.shuffle(selected)
    selected = [{**r, "smoke_id": i} for i, r in enumerate(selected)]
    write_manifest(out_dir / "vqa_rad_test_smoke50.csv", selected)

    print(f"test={len(all_rows)}, closed={len(closed)}, open={len(open_q)}")
    print(f"wrote: {out_dir / 'vqa_rad_test_full451.csv'}")
    print(f"wrote: {out_dir / 'vqa_rad_test_smoke50.csv'}")


if __name__ == "__main__":
    main()
