from pathlib import Path
import pandas as pd


ROOT = Path("results/vqa_rad/day9_analysis")

SRC = ROOT / "day9_transition_events.csv"
OUT = ROOT / "day9_transition_events_final.csv"

df = pd.read_csv(SRC)


# ==================================================
# Final answer-level semantic taxonomy
# ==================================================
#
# NOTE:
# This describes how the ANSWER changes relative
# to the dataset GT.
#
# It does NOT claim a causal visual failure mode.
# ==================================================

manual = {}


def add(
    idx,
    reason,
    mechanism,
    effect,
    evaluator_sensitive,
    note,
):
    key = (idx, reason)

    if key in manual:
        raise RuntimeError(
            f"Duplicate manual key: {key}"
        )

    manual[key] = {
        "final_mechanism": mechanism,
        "semantic_effect": effect,
        "evaluator_sensitive":
            evaluator_sensitive,
        "manual_note": note,
    }


# --------------------------------------------------
# A. Strict-EM near misses
# --------------------------------------------------

add(
    105, "lora_loss",
    "semantic_equivalent_wording",
    "equivalent",
    True,
    "right side -> right sided; same laterality/location meaning",
)

add(
    338, "lora_loss",
    "semantic_equivalent_wording",
    "equivalent",
    True,
    "both -> both sides; semantically equivalent in context",
)

add(
    411, "lora_loss",
    "answer_granularity_regression",
    "regression",
    True,
    "right lung -> right side; retains laterality but loses anatomical specificity",
)


# --------------------------------------------------
# B. Near match -> exact
# --------------------------------------------------

add(
    403, "vt256_gain",
    "answer_granularity_refinement",
    "refinement",
    True,
    "ring enhancing -> ring-enhancing lesions; answer completed to dataset target",
)

add(
    142, "lora_gain",
    "semantic_equivalent_wording",
    "equivalent",
    True,
    "right -> right hemisphere; context already asks hemisphere",
)

add(
    369, "lora_gain",
    "semantic_equivalent_wording",
    "equivalent",
    True,
    "T2 -> T2-weighted; terminology essentially equivalent",
)


# --------------------------------------------------
# C. Laterality reversals
# --------------------------------------------------

laterality_events = [
    (6,   "lora_loss"),
    (95,  "lora_gain"),
    (102, "lora_loss"),
    (103, "lora_loss"),
    (189, "lora_gain"),
    (191, "lora_gain"),
    (282, "lora_gain"),
    (428, "lora_loss"),
    (430, "lora_loss"),
]

for idx, reason in laterality_events:

    effect = (
        "correction"
        if reason.endswith("_gain")
        else "regression"
    )

    add(
        idx,
        reason,
        "laterality_reversal",
        effect,
        False,
        "left/right semantic decision reversed",
    )


# --------------------------------------------------
# D. Other semantic transitions
# --------------------------------------------------

add(
    41, "vt256_gain",
    "semantic_correction",
    "correction",
    False,
    "muscle -> fat",
)

add(
    258, "vt256_gain",
    "dataset_label_alignment_candidate",
    "ambiguous",
    True,
    "cardiopulmonary -> chest; GT/question ontology is itself imprecise",
)

add(
    310, "vt256_loss",
    "compatible_overspecification",
    "compatible",
    True,
    "MRI -> MR-FLAIR; after-answer is more specific but remains MRI-compatible",
)

add(
    443, "vt256_loss",
    "semantic_regression",
    "regression",
    False,
    "psoas muscle -> cecum",
)

add(
    258, "vt512_gain",
    "dataset_label_alignment_candidate",
    "ambiguous",
    True,
    "cardiopulmonary -> chest; GT/question ontology is itself imprecise",
)

add(
    399, "vt512_gain",
    "near_equivalent_to_exact",
    "near_equivalent",
    True,
    "nodules -> nodular opacities; closely related terminology",
)

add(
    400, "vt512_gain",
    "near_equivalent_to_exact",
    "near_equivalent",
    True,
    "nodules -> nodular opacities; closely related terminology",
)

add(
    443, "vt512_loss",
    "semantic_regression",
    "regression",
    False,
    "psoas muscle -> cecum",
)

add(
    8, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "larger -> smaller",
)

add(
    41, "lora_loss",
    "semantic_regression",
    "regression",
    False,
    "fat -> muscle",
)

add(
    72, "lora_gain",
    "answer_granularity_refinement",
    "refinement",
    True,
    "right lung -> right upper lobe; correct localization becomes more specific",
)

add(
    136, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "yes -> basilar artery; answer type/content corrected",
)

add(
    152, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "yes -> hydrocephalus",
)

add(
    171, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "male -> female",
)

add(
    306, "lora_gain",
    "semantic_equivalent_wording",
    "equivalent",
    True,
    "transverse -> axial; synonymous imaging-plane terminology",
)

add(
    358, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "T2-weighted -> FLAIR",
)

add(
    384, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "No -> appendicitis",
)

add(
    443, "lora_gain",
    "semantic_correction",
    "correction",
    False,
    "Colon -> psoas muscle",
)


if len(manual) != 33:
    raise RuntimeError(
        f"Expected 33 manual events, got {len(manual)}"
    )


# ==================================================
# Fill final taxonomy
# ==================================================

final_mechanism = []
semantic_effect = []
evaluator_sensitive = []
notes = []


for _, row in df.iterrows():

    key = (
        int(row["dataset_index"]),
        row["reason"],
    )

    hint = row["mechanism_hint"]

    # Straight yes/no event
    if hint == "yes_no_polarity_flip":

        final_mechanism.append(
            "binary_polarity_flip"
        )

        semantic_effect.append(
            "correction"
            if row["direction"] == "gain"
            else "regression"
        )

        evaluator_sensitive.append(False)

        notes.append(
            "yes/no answer polarity changed"
        )

    else:

        if key not in manual:
            raise RuntimeError(
                f"Missing manual review: {key}"
            )

        m = manual[key]

        final_mechanism.append(
            m["final_mechanism"]
        )

        semantic_effect.append(
            m["semantic_effect"]
        )

        evaluator_sensitive.append(
            m["evaluator_sensitive"]
        )

        notes.append(
            m["manual_note"]
        )


df["final_mechanism"] = final_mechanism
df["semantic_effect"] = semantic_effect
df["evaluator_sensitive"] = evaluator_sensitive
df["manual_note"] = notes


# ==================================================
# Sanity
# ==================================================

if len(df) != 107:
    raise RuntimeError(
        f"Expected 107 events, got {len(df)}"
    )

if df["final_mechanism"].isna().any():
    raise RuntimeError(
        "Missing final mechanism labels"
    )


df.to_csv(
    OUT,
    index=False,
)


# ==================================================
# Summaries
# ==================================================

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)


print("===== FINAL MECHANISM COUNTS =====")

print(
    df["final_mechanism"]
    .value_counts()
    .to_string()
)


print()
print("===== SEMANTIC EFFECT COUNTS =====")

print(
    df["semantic_effect"]
    .value_counts()
    .to_string()
)


print()
print("===== SEMANTIC EFFECT x COMPARISON =====")

tab = pd.crosstab(
    df["comparison"],
    df["semantic_effect"],
)

wanted = [
    "correction",
    "regression",
    "refinement",
    "equivalent",
    "near_equivalent",
    "compatible",
    "ambiguous",
]

tab = tab.reindex(
    columns=wanted,
    fill_value=0,
)

print(
    tab.to_string()
)


print()
print("===== EVALUATOR-SENSITIVE EVENTS =====")

for comparison, g in df.groupby(
    "comparison"
):

    n = int(
        g["evaluator_sensitive"]
        .astype(bool)
        .sum()
    )

    print(
        f"{comparison:<32}"
        f"{n:>3}/{len(g):<3}"
        f" ({100*n/len(g):5.1f}%)"
    )


print()
print("===== STRICT GAIN/LOSS vs DEFINITE SEMANTIC CHANGE =====")

for comparison, g in df.groupby(
    "comparison"
):

    strict_gain = int(
        (g["direction"] == "gain").sum()
    )

    strict_loss = int(
        (g["direction"] == "loss").sum()
    )

    semantic_corr = int(
        (g["semantic_effect"] == "correction").sum()
    )

    semantic_reg = int(
        (g["semantic_effect"] == "regression").sum()
    )

    print()
    print(comparison)
    print(
        f"  strict:    gain={strict_gain}, "
        f"loss={strict_loss}, "
        f"net={strict_gain-strict_loss:+d}"
    )
    print(
        f"  definite semantic: "
        f"correction={semantic_corr}, "
        f"regression={semantic_reg}, "
        f"net={semantic_corr-semantic_reg:+d}"
    )


print()
print("Saved:")
print(OUT)
