from pathlib import Path
import pandas as pd


ROOT = Path("results/vqa_rad/day9_analysis")

CASEBOOK = ROOT / "day9_review_casebook_with_images.csv"
MANIFEST = ROOT / "day9_unique_image_manifest.csv"

OUT = ROOT / "day9_review_casebook_taxonomy_stage1.csv"


df = pd.read_csv(CASEBOOK)
manifest = pd.read_csv(MANIFEST)


# --------------------------------------------------
# Manually reviewed semantic-target taxonomy
# --------------------------------------------------

groups = {
    "finding_presence_absence": [
        26, 43, 51, 52, 56, 61, 63, 64,
        79, 90, 107, 108, 113,
        163, 164, 170, 180, 181,
        204, 205, 212, 215, 216,
        256, 297, 308, 315, 436, 449,
    ],

    "morphology_size_attribute": [
        8, 46, 124, 146, 150, 166,
        190, 226, 244, 249, 279,
        330, 364, 372, 373, 413,
        422, 427, 433,
    ],

    "localization_laterality": [
        6, 72, 95, 102, 103, 105,
        142, 189, 191, 282,
        338, 411, 428, 430,
    ],

    "anatomy_visibility_normality": [
        5, 9, 36, 186, 329, 432,
    ],

    "modality_plane_sequence": [
        306, 310, 358, 369,
    ],

    "anatomy_structure_identification": [
        41, 136, 258, 443,
    ],

    "diagnosis_condition": [
        152, 384, 396,
    ],

    "finding_description": [
        399, 400, 403,
    ],

    "context_demographic_device": [
        171, 300, 447,
    ],
}


# --------------------------------------------------
# Convert mapping to dataset_index -> category
# --------------------------------------------------

index_to_category = {}

for category, indices in groups.items():
    for idx in indices:
        if idx in index_to_category:
            raise RuntimeError(
                f"Duplicate taxonomy assignment: {idx}"
            )

        index_to_category[idx] = category


expected = set(
    df["dataset_index"]
    .astype(int)
)

mapped = set(index_to_category)

missing = expected - mapped
extra = mapped - expected

if missing:
    raise RuntimeError(
        f"Missing taxonomy IDs: {sorted(missing)}"
    )

if extra:
    raise RuntimeError(
        f"Unexpected taxonomy IDs: {sorted(extra)}"
    )


df["semantic_target"] = (
    df["dataset_index"]
    .astype(int)
    .map(index_to_category)
)


# --------------------------------------------------
# Add image-level metadata
# --------------------------------------------------

image_meta = manifest[
    [
        "image_id",
        "width",
        "height",
        "raw_pixels",
        "n_review_qa",
        "raw_exceeds_vt256",
        "raw_exceeds_vt512",
        "raw_exceeds_vt1024",
    ]
].copy()

image_meta = image_meta.rename(
    columns={
        "n_review_qa":
            "affected_qa_on_same_image"
    }
)

df = df.merge(
    image_meta,
    on="image_id",
    how="left",
    validate="many_to_one",
)

df["multi_qa_affected_image"] = (
    df["affected_qa_on_same_image"] > 1
)


# Second-layer taxonomy is intentionally left blank
# for the next review stage.
df["transition_mechanism"] = ""
df["semantic_equivalence_note"] = ""
df["image_review_note"] = ""


df.to_csv(
    OUT,
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("===== SEMANTIC TARGET COUNTS =====")

print(
    df["semantic_target"]
    .value_counts()
    .to_string()
)


# Expand multiple review_reason values
expanded = []

for _, row in df.iterrows():
    for reason in str(
        row["review_reason"]
    ).split(";"):

        expanded.append({
            "reason": reason,
            "semantic_target":
                row["semantic_target"],
            "dataset_index":
                row["dataset_index"],
        })


exp = pd.DataFrame(expanded)


print()
print("===== SEMANTIC TARGET x FLIP TYPE =====")

table = pd.crosstab(
    exp["semantic_target"],
    exp["reason"],
)

wanted = [
    "lora_gain",
    "lora_loss",
    "vt256_gain",
    "vt256_loss",
    "vt512_gain",
    "vt512_loss",
    "vt1024_gain",
    "vt1024_loss",
]

table = table.reindex(
    columns=wanted,
    fill_value=0,
)

print(
    table.to_string()
)


print()
print("===== MULTI-QA IMAGE EFFECT =====")

print(
    "QA rows on images with multiple affected questions:",
    int(df["multi_qa_affected_image"].sum()),
    "/",
    len(df),
)

print(
    "Unique affected images:",
    df["image_id"].nunique(),
)


print()
print("Saved:")
print(OUT)
