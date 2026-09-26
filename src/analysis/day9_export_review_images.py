from pathlib import Path

import pandas as pd
from datasets import load_from_disk


PROJ = Path(".")
CASEBOOK = (
    PROJ
    / "results/vqa_rad/day9_analysis/day9_review_casebook.csv"
)

OUT_DIR = (
    PROJ
    / "results/vqa_rad/day9_analysis/review_images"
)

OUT_CSV = (
    PROJ
    / "results/vqa_rad/day9_analysis/"
      "day9_review_casebook_with_images.csv"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Load casebook + local VQA-RAD test
# --------------------------------------------------

df = pd.read_csv(CASEBOOK)

ds = load_from_disk(
    str(PROJ / "data/vqa_rad")
)["test"]


# --------------------------------------------------
# Export each UNIQUE image once
# --------------------------------------------------

image_paths = {}

for image_id, group in df.groupby("image_id"):

    # Any dataset_index from this group points
    # to the same image.
    dataset_index = int(
        group.iloc[0]["dataset_index"]
    )

    image = (
        ds[dataset_index]["image"]
        .convert("RGB")
    )

    path = (
        OUT_DIR
        / f"{image_id}.png"
    )

    image.save(path)

    image_paths[image_id] = str(
        path.resolve()
    )


# --------------------------------------------------
# Add image path to every QA row
# --------------------------------------------------

df.insert(
    2,
    "image_path",
    df["image_id"].map(
        image_paths
    ),
)

df.to_csv(
    OUT_CSV,
    index=False,
)


# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

print("===== IMAGE REVIEW PACKAGE =====")

print(
    "QA cases:",
    len(df),
)

print(
    "Unique images:",
    df["image_id"].nunique(),
)

print(
    "Exported PNGs:",
    len(
        list(
            OUT_DIR.glob("*.png")
        )
    ),
)

print()
print("===== IMAGE SIZE DISTRIBUTION =====")

records = []

for image_id, path in image_paths.items():

    rows = df[
        df["image_id"] == image_id
    ]

    idx = int(
        rows.iloc[0]["dataset_index"]
    )

    image = ds[idx]["image"]

    w, h = image.size

    records.append(
        {
            "image_id": image_id,
            "width": w,
            "height": h,
            "pixels": w * h,
            "n_review_qa": len(rows),
        }
    )

sizes = pd.DataFrame(records)

print(
    sizes[
        [
            "width",
            "height",
            "pixels",
        ]
    ]
    .describe(
        percentiles=[
            .25,
            .5,
            .75,
            .9,
        ]
    )
    .round(1)
    .to_string()
)

print()
print("===== EXAMPLE FILES =====")

for p in sorted(
    OUT_DIR.glob("*.png")
)[:10]:
    print(p)

print()
print("Saved images:")
print(OUT_DIR)

print()
print("Saved casebook:")
print(OUT_CSV)
