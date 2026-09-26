from pathlib import Path
import shutil
import pandas as pd


ROOT = Path("results/vqa_rad/day9_analysis")

CASEBOOK = ROOT / "day9_review_casebook_taxonomy_stage1.csv"
EVENTS = ROOT / "day9_transition_events_final.csv"

OUTDIR = ROOT / "qualitative_cases"
IMGDIR = OUTDIR / "images"

OUTDIR.mkdir(parents=True, exist_ok=True)
IMGDIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Representative cases selected for final report
# --------------------------------------------------

selected = {
    "fd7bcec7d8":
        "LoRA improves infarction recognition but reverses laterality",

    "a253d30997":
        "LoRA corrects anatomy identification; lower resolution reverses it",

    "27994ed821":
        "LoRA morphology gain is lost under stronger compression",

    "834c75b256":
        "vt512 exact-match gain driven partly by wording/granularity",

    "86d378260b":
        "Compressed budgets correct a native-model finding decision",

    "e551b2f8ad":
        "Non-monotonic resolution-sensitive pneumothorax prediction",
}


cases = pd.read_csv(CASEBOOK)
events = pd.read_csv(EVENTS)


# --------------------------------------------------
# Keep selected cases
# --------------------------------------------------

c = cases[
    cases["image_id"].isin(selected)
].copy()

e = events[
    events["image_id"].isin(selected)
].copy()


c["case_rationale"] = (
    c["image_id"].map(selected)
)

e["case_rationale"] = (
    e["image_id"].map(selected)
)


# --------------------------------------------------
# Copy one image for every selected image_id
# --------------------------------------------------

for image_id in selected:

    rows = c[
        c["image_id"] == image_id
    ]

    if len(rows) == 0:
        raise RuntimeError(
            f"Missing image case: {image_id}"
        )

    src = Path(
        rows.iloc[0]["image_path"]
    )

    dst = IMGDIR / f"{image_id}.png"

    shutil.copy2(src, dst)


# --------------------------------------------------
# Save detailed CSVs
# --------------------------------------------------

c.to_csv(
    OUTDIR / "qualitative_case_qa.csv",
    index=False,
)

e.to_csv(
    OUTDIR / "qualitative_case_events.csv",
    index=False,
)


# --------------------------------------------------
# Generate readable Markdown report
# --------------------------------------------------

lines = []

lines.append(
    "# Day 9 Qualitative Case Study"
)

lines.append("")

lines.append(
    "Selected examples are illustrative case studies, "
    "not independent quantitative evidence."
)

lines.append("")


for image_id, rationale in selected.items():

    qrows = c[
        c["image_id"] == image_id
    ].sort_values(
        "dataset_index"
    )

    erows = e[
        e["image_id"] == image_id
    ]

    lines.append(
        f"## {image_id}"
    )

    lines.append("")

    lines.append(
        f"**Rationale:** {rationale}"
    )

    lines.append("")

    lines.append(
        f"Image: `images/{image_id}.png`"
    )

    lines.append("")

    for _, r in qrows.iterrows():

        idx = int(
            r["dataset_index"]
        )

        lines.append(
            f"### QA {idx}"
        )

        lines.append("")

        lines.append(
            f"- Semantic target: "
            f"`{r['semantic_target']}`"
        )

        lines.append(
            f"- Question: {r['question']}"
        )

        lines.append(
            f"- GT: `{r['gt']}`"
        )

        lines.append(
            f"- Zero-shot: "
            f"`{r['zero_native__prediction_raw']}`"
        )

        lines.append(
            f"- LoRA-native: "
            f"`{r['lora_native__prediction_raw']}`"
        )

        lines.append(
            f"- LoRA-vt1024: "
            f"`{r['lora_vt1024__prediction_raw']}`"
        )

        lines.append(
            f"- LoRA-vt512: "
            f"`{r['lora_vt512__prediction_raw']}`"
        )

        lines.append(
            f"- LoRA-vt256: "
            f"`{r['lora_vt256__prediction_raw']}`"
        )

        qa_events = erows[
            erows["dataset_index"] == idx
        ]

        if len(qa_events):

            lines.append(
                "- Observed transitions:"
            )

            for _, x in qa_events.iterrows():

                lines.append(
                    "  - "
                    f"`{x['comparison']}`: "
                    f"{x['direction']}; "
                    f"`{x['final_mechanism']}`; "
                    f"semantic effect="
                    f"`{x['semantic_effect']}`"
                )

        lines.append("")


md = OUTDIR / "day9_qualitative_cases.md"

md.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# --------------------------------------------------
# Print compact terminal summary
# --------------------------------------------------

print("===== QUALITATIVE CASES =====")

for image_id, rationale in selected.items():

    qrows = c[
        c["image_id"] == image_id
    ]

    print()
    print(image_id)
    print("  ", rationale)

    print(
        "  QA:",
        ",".join(
            map(
                str,
                sorted(
                    qrows[
                        "dataset_index"
                    ].astype(int)
                )
            )
        ),
    )

    for _, x in e[
        e["image_id"] == image_id
    ].iterrows():

        print(
            "   ",
            int(x["dataset_index"]),
            "|",
            x["reason"],
            "|",
            x["final_mechanism"],
            "|",
            x["semantic_effect"],
        )


print()
print("Saved:")
print(OUTDIR)
print(md)
