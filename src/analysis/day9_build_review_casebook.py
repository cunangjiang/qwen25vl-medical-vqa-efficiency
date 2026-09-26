from pathlib import Path
import hashlib

import pandas as pd
from datasets import load_from_disk


ROOT = Path("results/vqa_rad")
DAY9 = ROOT / "day9_analysis"

MASTER = DAY9 / "day9_master_comparison.csv"
DATASET = Path("data/vqa_rad")

df = pd.read_csv(MASTER)

ds = load_from_disk(str(DATASET))["test"]


# --------------------------------------------------
# Image identity
# --------------------------------------------------
def image_sha1(img):
    img = img.convert("RGB")
    w, h = img.size

    hsh = hashlib.sha1()
    hsh.update(f"{w}x{h}".encode())
    hsh.update(img.tobytes())

    return hsh.hexdigest()


hash_cache = {}

for idx in df["dataset_index"].astype(int):
    if idx not in hash_cache:
        hash_cache[idx] = image_sha1(
            ds[idx]["image"]
        )

df["image_sha1"] = (
    df["dataset_index"]
    .astype(int)
    .map(hash_cache)
)

# Short readable ID
df["image_id"] = (
    df["image_sha1"]
    .str[:10]
)


# --------------------------------------------------
# Which cases should enter manual review?
# --------------------------------------------------

review_flags = {
    "lora_gain":
        df["zero_to_lora"].eq("gain"),

    "lora_loss":
        df["zero_to_lora"].eq("loss"),

    "vt256_gain":
        df["native_to_lora_vt256"].eq("gain"),

    "vt256_loss":
        df["native_to_lora_vt256"].eq("loss"),

    "vt512_gain":
        df["native_to_lora_vt512"].eq("gain"),

    "vt512_loss":
        df["native_to_lora_vt512"].eq("loss"),

    "vt1024_gain":
        df["native_to_lora_vt1024"].eq("gain"),

    "vt1024_loss":
        df["native_to_lora_vt1024"].eq("loss"),
}


def flags_for_row(i):
    names = []

    for name, mask in review_flags.items():
        if bool(mask.iloc[i]):
            names.append(name)

    return ";".join(names)


df["review_reason"] = [
    flags_for_row(i)
    for i in range(len(df))
]

review = df[
    df["review_reason"].ne("")
].copy()


# --------------------------------------------------
# Useful semantic hint only.
#
# IMPORTANT:
# These are NOT final error labels.
# They only make manual review easier.
# --------------------------------------------------

def question_hint(q):
    q = str(q).lower()

    if (
        q.startswith("is ")
        or q.startswith("are ")
        or q.startswith("do ")
        or q.startswith("does ")
        or q.startswith("was ")
        or q.startswith("were ")
        or q.startswith("can ")
        or q.startswith("has ")
        or q.startswith("have ")
    ):
        return "yes_no"

    if "left or right" in q or "which side" in q or "what side" in q:
        return "laterality"

    if (
        q.startswith("where ")
        or "where is" in q
        or "located" in q
        or "location" in q
    ):
        return "location"

    if (
        "how many" in q
        or "number" in q
        or "measure" in q
        or "cm" in q
        or "size" in q
        or "larger" in q
        or "smaller" in q
    ):
        return "size_count_measurement"

    if (
        "what organ" in q
        or "what structure" in q
        or "which structure" in q
    ):
        return "anatomy"

    if (
        "modality" in q
        or "type of image" in q
        or "plane" in q
        or "view" in q
    ):
        return "modality_plane"

    return "other"


review["question_hint"] = (
    review["question"]
    .map(question_hint)
)


# --------------------------------------------------
# Columns for review
# --------------------------------------------------

cols = [
    "dataset_index",
    "image_id",
    "review_reason",
    "question_type",
    "question_hint",

    "question",
    "gt",

    "zero_native__prediction_raw",
    "lora_native__prediction_raw",
    "lora_vt1024__prediction_raw",
    "lora_vt512__prediction_raw",
    "lora_vt256__prediction_raw",

    "zero_native__exact_match",
    "lora_native__exact_match",
    "lora_vt1024__exact_match",
    "lora_vt512__exact_match",
    "lora_vt256__exact_match",

    "zero_native__token_f1",
    "lora_native__token_f1",
    "lora_vt1024__token_f1",
    "lora_vt512__token_f1",
    "lora_vt256__token_f1",

    "lora_native__visual_token_count",
    "lora_vt1024__visual_token_count",
    "lora_vt512__visual_token_count",
    "lora_vt256__visual_token_count",

    # Blank human-review fields
]

review = review[cols].copy()

review["error_category"] = ""
review["error_subcategory"] = ""
review["semantic_severity"] = ""
review["annotation_note"] = ""


# --------------------------------------------------
# Sort: same image together
# --------------------------------------------------

review = review.sort_values(
    [
        "image_id",
        "dataset_index",
    ]
).reset_index(drop=True)


out = DAY9 / "day9_review_casebook.csv"

review.to_csv(
    out,
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("===== REVIEW CASEBOOK =====")
print("QA cases:", len(review))
print(
    "Unique affected images:",
    review["image_id"].nunique(),
)

print()
print("===== CASES BY REVIEW REASON =====")

reason_counts = {}

for x in review["review_reason"]:
    for r in x.split(";"):
        reason_counts[r] = (
            reason_counts.get(r, 0) + 1
        )

for k in sorted(reason_counts):
    print(f"{k:<15} {reason_counts[k]:>4}")


print()
print("===== QUESTION HINT =====")

print(
    review["question_hint"]
    .value_counts()
    .to_string()
)


print()
print("===== MULTIPLE FLIPS ON SAME IMAGE =====")

g = (
    review
    .groupby("image_id")
    .agg(
        n_cases=("dataset_index", "size"),
        dataset_indices=(
            "dataset_index",
            lambda x:
                ",".join(
                    map(str, sorted(x))
                )
        ),
    )
    .sort_values(
        "n_cases",
        ascending=False,
    )
)

print(
    g[g["n_cases"] >= 2]
    .head(20)
    .to_string()
)


print()
print("Saved:")
print(out)
