from pathlib import Path
import pandas as pd


ROOT = Path("results/vqa_rad/day9_analysis")

SRC = ROOT / "day9_review_casebook_taxonomy_stage1.csv"
OUT = ROOT / "day9_transition_events.csv"

df = pd.read_csv(SRC)


# reason -> (before condition, after condition)
COMPARE = {
    "lora_gain":
        ("zero_native", "lora_native"),

    "lora_loss":
        ("zero_native", "lora_native"),

    "vt256_gain":
        ("lora_native", "lora_vt256"),

    "vt256_loss":
        ("lora_native", "lora_vt256"),

    "vt512_gain":
        ("lora_native", "lora_vt512"),

    "vt512_loss":
        ("lora_native", "lora_vt512"),

    "vt1024_gain":
        ("lora_native", "lora_vt1024"),

    "vt1024_loss":
        ("lora_native", "lora_vt1024"),
}


records = []


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def contains_side(x):
    x = norm(x)

    left = "left" in x
    right = "right" in x

    if left and not right:
        return "left"

    if right and not left:
        return "right"

    return None


for _, row in df.iterrows():

    reasons = str(
        row["review_reason"]
    ).split(";")

    for reason in reasons:

        if reason not in COMPARE:
            continue

        before, after = COMPARE[reason]

        before_pred = row[
            f"{before}__prediction_raw"
        ]

        after_pred = row[
            f"{after}__prediction_raw"
        ]

        before_em = int(
            row[f"{before}__exact_match"]
        )

        after_em = int(
            row[f"{after}__exact_match"]
        )

        before_f1 = float(
            row[f"{before}__token_f1"]
        )

        after_f1 = float(
            row[f"{after}__token_f1"]
        )

        gt = norm(row["gt"])

        bp = norm(before_pred)
        ap = norm(after_pred)

        # ------------------------------------------
        # Conservative mechanism HINTS.
        #
        # These are NOT final semantic judgments.
        # ------------------------------------------

        hint = "other_semantic_flip"

        # Very clear yes/no polarity transition
        if (
            bp in {"yes", "no"}
            and ap in {"yes", "no"}
            and bp != ap
        ):
            hint = "yes_no_polarity_flip"

        else:
            before_side = contains_side(bp)
            after_side = contains_side(ap)

            if (
                before_side is not None
                and after_side is not None
                and before_side != after_side
            ):
                hint = "laterality_reversal"

            # Exact match says wrong, but lexical overlap
            # remains reasonably high.
            elif (
                after_em == 0
                and after_f1 >= 0.5
            ):
                hint = "strict_em_nearmiss_candidate"

            elif (
                before_em == 0
                and before_f1 >= 0.5
                and after_em == 1
            ):
                hint = "near_match_to_exact"

        before_vt = row.get(
            f"{before}__visual_token_count",
            None,
        )

        after_vt = row.get(
            f"{after}__visual_token_count",
            None,
        )

        if (
            pd.notna(before_vt)
            and pd.notna(after_vt)
            and float(before_vt) > 0
        ):
            vt_reduction = (
                1
                - float(after_vt)
                / float(before_vt)
            ) * 100
        else:
            vt_reduction = 0.0

        records.append({
            "dataset_index":
                int(row["dataset_index"]),

            "image_id":
                row["image_id"],

            "image_path":
                row["image_path"],

            "reason":
                reason,

            "comparison":
                f"{before} -> {after}",

            "direction":
                (
                    "gain"
                    if reason.endswith("_gain")
                    else "loss"
                ),

            "question_type":
                row["question_type"],

            "semantic_target":
                row["semantic_target"],

            "question":
                row["question"],

            "gt":
                row["gt"],

            "before_prediction":
                before_pred,

            "after_prediction":
                after_pred,

            "before_em":
                before_em,

            "after_em":
                after_em,

            "before_token_f1":
                before_f1,

            "after_token_f1":
                after_f1,

            "before_visual_tokens":
                before_vt,

            "after_visual_tokens":
                after_vt,

            "visual_token_reduction_pct":
                vt_reduction,

            # Rule-based hint only
            "mechanism_hint":
                hint,

            # To be manually reviewed next
            "transition_mechanism":
                "",

            "semantic_equivalent":
                "",

            "review_note":
                "",
        })


events = pd.DataFrame(records)

events = events.sort_values(
    [
        "comparison",
        "semantic_target",
        "dataset_index",
    ]
).reset_index(drop=True)


if len(events) != 107:
    raise RuntimeError(
        f"Expected 107 transition events, got {len(events)}"
    )


events.to_csv(
    OUT,
    index=False,
)


print("===== TRANSITION EVENTS =====")
print("Events:", len(events))
print(
    "Unique QA:",
    events["dataset_index"].nunique(),
)
print(
    "Unique images:",
    events["image_id"].nunique(),
)


print()
print("===== EVENTS BY COMPARISON =====")

print(
    pd.crosstab(
        events["comparison"],
        events["direction"],
    ).to_string()
)


print()
print("===== MECHANISM HINTS =====")

print(
    events["mechanism_hint"]
    .value_counts()
    .to_string()
)


print()
print(
    "===== MECHANISM HINT x SEMANTIC TARGET ====="
)

print(
    pd.crosstab(
        events["semantic_target"],
        events["mechanism_hint"],
    ).to_string()
)


print()
print("Saved:")
print(OUT)
