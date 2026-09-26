import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk


def image_hash(img):
    img = img.convert("RGB")
    h = hashlib.sha1()
    h.update(str(img.size).encode("utf-8"))
    h.update(img.tobytes())
    return h.hexdigest()


def load_train_hashes(path):
    result = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result.add(Path(row["images"][0]).stem)
    return result


def main():
    parser = argparse.ArgumentParser(description="Stratify zero-shot vs LoRA EM by whether the test image occurred in actual SFT training.")
    parser.add_argument("--dataset", default="data/vqa_rad")
    parser.add_argument("--train-jsonl", default="data/vqa_rad_sft/train.jsonl")
    parser.add_argument("--zero-csv", default="results/vqa_rad/zero_shot_full451_predictions.csv")
    parser.add_argument("--lora-csv", default="results/vqa_rad/lora_1ep_full451_predictions.csv")
    parser.add_argument("--output", default="results/day10_validity_audit/overlap_stratified_metrics.csv")
    args = parser.parse_args()

    train_hashes = load_train_hashes(args.train_jsonl)
    test = load_from_disk(args.dataset)["test"]
    membership = pd.DataFrame([
        {
            "dataset_index": idx,
            "image_hash": image_hash(ex["image"]),
        }
        for idx, ex in enumerate(test)
    ])
    membership["image_seen_in_sft_train"] = membership["image_hash"].isin(train_hashes)

    zero = pd.read_csv(args.zero_csv)[["dataset_index", "exact_match"]].rename(columns={"exact_match": "zero_em"})
    lora = pd.read_csv(args.lora_csv)[["dataset_index", "exact_match"]].rename(columns={"exact_match": "lora_em"})
    df = membership.merge(zero, on="dataset_index", validate="one_to_one").merge(lora, on="dataset_index", validate="one_to_one")

    rows = []
    for name, g in [
        ("all_test", df),
        ("image_seen_in_sft_train", df[df["image_seen_in_sft_train"]]),
        ("image_not_seen_in_sft_train", df[~df["image_seen_in_sft_train"]]),
    ]:
        gain = int(((g.zero_em == 0) & (g.lora_em == 1)).sum())
        loss = int(((g.zero_em == 1) & (g.lora_em == 0)).sum())
        rows.append({
            "subset": name,
            "n_qa": len(g),
            "n_unique_images": g.image_hash.nunique(),
            "zero_em": g.zero_em.mean(),
            "lora_em": g.lora_em.mean(),
            "delta_em_pp": 100 * (g.lora_em.mean() - g.zero_em.mean()),
            "zero_correct": int(g.zero_em.sum()),
            "lora_correct": int(g.lora_em.sum()),
            "gain": gain,
            "loss": loss,
            "net_correct": gain - loss,
        })

    out_df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
