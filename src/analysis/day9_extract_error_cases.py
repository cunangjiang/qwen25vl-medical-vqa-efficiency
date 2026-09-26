from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("results/vqa_rad/day9_analysis")
MASTER = ROOT / "day9_master_comparison.csv"

df = pd.read_csv(MASTER)


# --------------------------------------------------
# Reconstruct native visual-token bins
# --------------------------------------------------

native_vt = df["lora_native__visual_token_count"]

df["native_vt_bin"] = pd.cut(
    native_vt,
    bins=[-np.inf, 256, 512, 1024, np.inf],
    labels=[
        "<=256",
        "257-512",
        "513-1024",
        ">1024",
    ],
)


# --------------------------------------------------
# Useful derived quantities
# --------------------------------------------------

df["lora_open_f1_delta"] = (
    df["lora_native__token_f1"]
    - df["zero_native__token_f1"]
)

for budget in [
    "lora_vt256",
    "lora_vt512",
    "lora_vt1024",
]:
    df[f"{budget}__vt_reduction_pct"] = (
        1
        - df[f"{budget}__visual_token_count"]
        / df["lora_native__visual_token_count"]
    ) * 100


# --------------------------------------------------
# Common columns for review
# --------------------------------------------------

BASE = [
    "dataset_index",
    "question_type",
    "native_vt_bin",
    "question",
    "gt",
]


def save_case(name, mask, extra_cols, sort_by=None, ascending=True):
    cols = BASE + extra_cols

    x = df.loc[mask, cols].copy()

    if sort_by is not None:
        x = x.sort_values(
            sort_by,
            ascending=ascending,
        )

    path = ROOT / f"{name}.csv"
    x.to_csv(path, index=False)

    print(f"{name:<35} {len(x):>4} cases")

    return x


print("===== CASE COUNTS =====")


# ==================================================
# A. LoRA effect
# ==================================================

lora_gain = save_case(
    "lora_gain",
    df["zero_to_lora"].eq("gain"),
    [
        "zero_native__prediction_raw",
        "lora_native__prediction_raw",
        "zero_native__token_f1",
        "lora_native__token_f1",
        "lora_open_f1_delta",
        "lora_native__visual_token_count",
    ],
)

lora_loss = save_case(
    "lora_loss",
    df["zero_to_lora"].eq("loss"),
    [
        "zero_native__prediction_raw",
        "lora_native__prediction_raw",
        "zero_native__token_f1",
        "lora_native__token_f1",
        "lora_open_f1_delta",
        "lora_native__visual_token_count",
    ],
)


# ==================================================
# B. Resolution gains / losses
# ==================================================

resolution_cases = {}

for budget in [
    "lora_vt256",
    "lora_vt512",
    "lora_vt1024",
]:
    short = budget.replace("lora_", "")

    for transition in [
        "gain",
        "loss",
    ]:
        key = f"{short}_{transition}"

        resolution_cases[key] = save_case(
            key,
            df[f"native_to_{budget}"].eq(
                transition
            ),
            [
                "lora_native__prediction_raw",
                f"{budget}__prediction_raw",
                "lora_native__exact_match",
                f"{budget}__exact_match",
                "lora_native__token_f1",
                f"{budget}__token_f1",
                "lora_native__visual_token_count",
                f"{budget}__visual_token_count",
                f"{budget}__vt_reduction_pct",
            ],
        )


# ==================================================
# C. Prediction changed but correctness did NOT
#    Useful for understanding hidden instability.
# ==================================================

for budget in [
    "lora_vt256",
    "lora_vt512",
    "lora_vt1024",
]:
    short = budget.replace("lora_", "")

    transition_col = f"native_to_{budget}"

    mask = (
        df[f"prediction_changed__{budget}"]
        &
        df[transition_col].isin(
            ["both_correct", "both_wrong"]
        )
    )

    save_case(
        f"{short}_text_change_same_correctness",
        mask,
        [
            transition_col,
            "lora_native__prediction_raw",
            f"{budget}__prediction_raw",
            "lora_native__token_f1",
            f"{budget}__token_f1",
            "lora_native__visual_token_count",
            f"{budget}__visual_token_count",
            f"{budget}__vt_reduction_pct",
        ],
    )


# ==================================================
# D. Open-question near misses after LoRA
#
# Wrong by EM but token overlap is reasonably high.
# These are candidates for terminology / granularity
# analysis rather than true visual failure.
# ==================================================

open_nearmiss = save_case(
    "lora_open_nearmiss",
    (
        df["question_type"].eq("open")
        &
        df["lora_native__exact_match"].eq(0)
        &
        df["lora_native__token_f1"].ge(0.5)
    ),
    [
        "lora_native__prediction_raw",
        "lora_native__prediction_normalized",
        "lora_native__token_f1",
        "zero_native__prediction_raw",
        "zero_native__token_f1",
        "lora_open_f1_delta",
        "lora_native__visual_token_count",
    ],
    sort_by="lora_native__token_f1",
    ascending=False,
)


# ==================================================
# E. Open questions where LoRA remains wrong but F1
#    improved substantially over zero-shot.
# ==================================================

open_partial_gain = save_case(
    "lora_open_partial_gain",
    (
        df["question_type"].eq("open")
        &
        df["zero_native__exact_match"].eq(0)
        &
        df["lora_native__exact_match"].eq(0)
        &
        df["lora_open_f1_delta"].ge(0.25)
    ),
    [
        "zero_native__prediction_raw",
        "lora_native__prediction_raw",
        "zero_native__token_f1",
        "lora_native__token_f1",
        "lora_open_f1_delta",
        "lora_native__visual_token_count",
    ],
    sort_by="lora_open_f1_delta",
    ascending=False,
)


# --------------------------------------------------
# Compact previews
# --------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 55)


def preview(title, x, before, after, n=8):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    cols = [
        "dataset_index",
        "question_type",
        "native_vt_bin",
        "question",
        "gt",
        before,
        after,
    ]

    print(
        x[cols]
        .head(n)
        .to_string(index=False)
    )


preview(
    "LoRA GAINS: zero-shot wrong -> LoRA correct",
    lora_gain,
    "zero_native__prediction_raw",
    "lora_native__prediction_raw",
)

preview(
    "LoRA LOSSES: zero-shot correct -> LoRA wrong",
    lora_loss,
    "zero_native__prediction_raw",
    "lora_native__prediction_raw",
)

preview(
    "VT256 LOSSES: LoRA-native correct -> vt256 wrong",
    resolution_cases["vt256_loss"],
    "lora_native__prediction_raw",
    "lora_vt256__prediction_raw",
)

preview(
    "VT512 GAINS: LoRA-native wrong -> vt512 correct",
    resolution_cases["vt512_gain"],
    "lora_native__prediction_raw",
    "lora_vt512__prediction_raw",
)

preview(
    "VT512 LOSSES: LoRA-native correct -> vt512 wrong",
    resolution_cases["vt512_loss"],
    "lora_native__prediction_raw",
    "lora_vt512__prediction_raw",
)

print()
print("Saved under:")
print(ROOT)
