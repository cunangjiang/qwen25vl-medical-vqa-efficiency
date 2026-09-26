from pathlib import Path
import pandas as pd


ROOT = Path("results/vqa_rad/day9_analysis")

SRC = ROOT / "day9_transition_events.csv"

OUT = ROOT / "day9_nonbinary_transition_review.csv"

df = pd.read_csv(SRC)


# --------------------------------------------------
# Keep everything except straightforward yes/no flips
# --------------------------------------------------

review = df[
    df["mechanism_hint"]
    != "yes_no_polarity_flip"
].copy()


if len(review) != 33:
    raise RuntimeError(
        f"Expected 33 non-binary events, got {len(review)}"
    )


# --------------------------------------------------
# Fields for final manual mechanism taxonomy
# --------------------------------------------------

review["final_mechanism"] = ""

review["semantic_equivalent"] = ""

review["confidence"] = ""

review["manual_note"] = ""


# --------------------------------------------------
# Sort useful cases together
# --------------------------------------------------

hint_order = {
    "strict_em_nearmiss_candidate": 0,
    "near_match_to_exact": 1,
    "laterality_reversal": 2,
    "other_semantic_flip": 3,
}

review["_order"] = (
    review["mechanism_hint"]
    .map(hint_order)
)

review = review.sort_values(
    [
        "_order",
        "comparison",
        "dataset_index",
    ]
).drop(
    columns="_order"
)


review.to_csv(
    OUT,
    index=False,
)


# --------------------------------------------------
# Print all 33 in compact format
# --------------------------------------------------

print("===== NON-BINARY REVIEW =====")
print("Events:", len(review))

print()
print("===== COUNTS =====")

print(
    review["mechanism_hint"]
    .value_counts()
    .to_string()
)


for hint, g in review.groupby(
    "mechanism_hint",
    sort=False,
):

    print()
    print("=" * 110)
    print(hint)
    print("=" * 110)

    for _, r in g.iterrows():

        print(
            f"\nidx={int(r['dataset_index'])}"
            f" | {r['reason']}"
            f" | {r['semantic_target']}"
        )

        print(
            "Q :",
            r["question"],
        )

        print(
            "GT:",
            r["gt"],
        )

        print(
            "BEFORE:",
            r["before_prediction"],
            f"(F1={r['before_token_f1']:.3f})",
        )

        print(
            "AFTER :",
            r["after_prediction"],
            f"(F1={r['after_token_f1']:.3f})",
        )


print()
print("Saved:")
print(OUT)
