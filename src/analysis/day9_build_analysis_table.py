from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("results/vqa_rad")
OUT = ROOT / "day9_analysis"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "zero_native":
        ROOT / "zero_shot_full451_predictions.csv",

    "lora_native":
        ROOT / "lora_1ep_full451_predictions.csv",

    "lora_vt256":
        ROOT / "day8_resolution/vt256_full451_predictions.csv",

    "lora_vt512":
        ROOT / "day8_resolution/vt512_full451_predictions.csv",

    "lora_vt1024":
        ROOT / "day8_resolution/vt1024_full451_predictions.csv",
}


# --------------------------------------------------
# Load
# --------------------------------------------------

dfs = {}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()

    if len(df) != 451:
        raise RuntimeError(
            f"{name}: expected 451 rows, got {len(df)}"
        )

    if df["dataset_index"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicated dataset_index"
        )

    dfs[name] = (
        df
        .sort_values("dataset_index")
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Verify all five evaluations are aligned
# --------------------------------------------------

reference = dfs["zero_native"]

for name, df in dfs.items():

    if not np.array_equal(
        reference["dataset_index"].values,
        df["dataset_index"].values,
    ):
        raise RuntimeError(
            f"dataset_index mismatch: {name}"
        )

    for col in [
        "question_type",
        "question",
        "gt",
    ]:
        if not (
            reference[col].astype(str).values
            ==
            df[col].astype(str).values
        ).all():
            raise RuntimeError(
                f"{col} mismatch: {name}"
            )


print("===== ALIGNMENT CHECK =====")
print("All five prediction files align on 451 examples.")


# --------------------------------------------------
# Build master table
# --------------------------------------------------

base_cols = [
    "dataset_index",
    "question_type",
    "question",
    "gt",
    "gt_normalized",
    "image_width",
    "image_height",
]

master = reference[base_cols].copy()

master["image_pixels"] = (
    master["image_width"]
    * master["image_height"]
)


copy_cols = [
    "prediction_raw",
    "prediction_normalized",
    "exact_match",
    "token_f1",
    "visual_token_count",
    "input_token_count",
    "output_token_count",
    "preprocess_ms",
    "generate_ms",
    "e2e_ms",
    "dynamic_peak_allocated_gib",
]


for name, df in dfs.items():
    for col in copy_cols:
        master[f"{name}__{col}"] = df[col].values


# --------------------------------------------------
# Convenience columns: correctness transitions
# --------------------------------------------------

def add_transition(
    out,
    before_name,
    after_name,
    label,
):
    before = out[
        f"{before_name}__exact_match"
    ].astype(int)

    after = out[
        f"{after_name}__exact_match"
    ].astype(int)

    conditions = [
        (before == 0) & (after == 1),
        (before == 1) & (after == 0),
        (before == 1) & (after == 1),
        (before == 0) & (after == 0),
    ]

    values = [
        "gain",
        "loss",
        "both_correct",
        "both_wrong",
    ]

    out[label] = np.select(
        conditions,
        values,
        default="unknown",
    )


add_transition(
    master,
    "zero_native",
    "lora_native",
    "zero_to_lora",
)

for budget in [
    "lora_vt256",
    "lora_vt512",
    "lora_vt1024",
]:
    add_transition(
        master,
        "lora_native",
        budget,
        f"native_to_{budget}",
    )


# Did the textual answer itself change?
for budget in [
    "lora_vt256",
    "lora_vt512",
    "lora_vt1024",
]:
    master[
        f"prediction_changed__{budget}"
    ] = (
        master[
            "lora_native__prediction_normalized"
        ].astype(str)
        !=
        master[
            f"{budget}__prediction_normalized"
        ].astype(str)
    )


master_path = OUT / "day9_master_comparison.csv"

master.to_csv(
    master_path,
    index=False,
)


# --------------------------------------------------
# 1. Recompute condition-level metrics
# --------------------------------------------------

condition_rows = []

for name in FILES:
    em = master[
        f"{name}__exact_match"
    ].astype(float)

    f1 = master[
        f"{name}__token_f1"
    ].astype(float)

    vt = master[
        f"{name}__visual_token_count"
    ].astype(float)

    closed = (
        master["question_type"] == "closed"
    )

    open_ = (
        master["question_type"] == "open"
    )

    condition_rows.append({
        "condition": name,

        "num_correct":
            int(em.sum()),

        "overall_em":
            float(em.mean()),

        "closed_correct":
            int(em[closed].sum()),

        "closed_em":
            float(em[closed].mean()),

        "open_correct":
            int(em[open_].sum()),

        "open_em":
            float(em[open_].mean()),

        "open_token_f1":
            float(f1[open_].mean()),

        "avg_visual_tokens":
            float(vt.mean()),
    })


condition_df = pd.DataFrame(condition_rows)

condition_df.to_csv(
    OUT / "day9_condition_summary.csv",
    index=False,
)


# --------------------------------------------------
# 2. Transition / flip summary
# --------------------------------------------------

flip_rows = []


def summarize_transition(
    col,
    comparison,
):
    for qtype in [
        "all",
        "closed",
        "open",
    ]:

        if qtype == "all":
            x = master
        else:
            x = master[
                master["question_type"] == qtype
            ]

        counts = (
            x[col]
            .value_counts()
            .to_dict()
        )

        gain = int(counts.get("gain", 0))
        loss = int(counts.get("loss", 0))

        flip_rows.append({
            "comparison": comparison,
            "question_type": qtype,

            "gain":
                gain,

            "loss":
                loss,

            "net_correct_change":
                gain - loss,

            "both_correct":
                int(
                    counts.get(
                        "both_correct",
                        0,
                    )
                ),

            "both_wrong":
                int(
                    counts.get(
                        "both_wrong",
                        0,
                    )
                ),

            "correctness_flip_rate":
                float(
                    (gain + loss)
                    / len(x)
                ),
        })


summarize_transition(
    "zero_to_lora",
    "zero_native -> lora_native",
)

for budget in [
    "lora_vt256",
    "lora_vt512",
    "lora_vt1024",
]:
    summarize_transition(
        f"native_to_{budget}",
        f"lora_native -> {budget}",
    )


flip_df = pd.DataFrame(flip_rows)

flip_df.to_csv(
    OUT / "day9_flip_summary.csv",
    index=False,
)


# --------------------------------------------------
# 3. Bin samples by LoRA-native visual-token workload
#
# This answers:
# Are high-resolution / high-token samples more
# sensitive to lowering the visual budget?
# --------------------------------------------------

native_vt = master[
    "lora_native__visual_token_count"
]

master["native_vt_bin"] = pd.cut(
    native_vt,
    bins=[
        -np.inf,
        256,
        512,
        1024,
        np.inf,
    ],
    labels=[
        "<=256",
        "257-512",
        "513-1024",
        ">1024",
    ],
)


bin_rows = []

for bin_name, x in master.groupby(
    "native_vt_bin",
    observed=True,
):

    row = {
        "native_vt_bin": str(bin_name),
        "n": len(x),
        "avg_native_vt":
            float(
                x[
                    "lora_native__visual_token_count"
                ].mean()
            ),
    }

    for condition in [
        "lora_native",
        "lora_vt1024",
        "lora_vt512",
        "lora_vt256",
    ]:
        row[
            f"{condition}_em"
        ] = float(
            x[
                f"{condition}__exact_match"
            ].mean()
        )

    for condition in [
        "lora_vt1024",
        "lora_vt512",
        "lora_vt256",
    ]:
        row[
            f"{condition}_correctness_flip_rate"
        ] = float(
            (
                x[
                    f"native_to_{condition}"
                ].isin(
                    ["gain", "loss"]
                )
            ).mean()
        )

        row[
            f"{condition}_prediction_change_rate"
        ] = float(
            x[
                f"prediction_changed__{condition}"
            ].mean()
        )

    bin_rows.append(row)


bin_df = pd.DataFrame(bin_rows)

bin_df.to_csv(
    OUT / "day9_native_vt_bins.csv",
    index=False,
)


# --------------------------------------------------
# 4. Save compact JSON
# --------------------------------------------------

summary = {
    "num_examples": len(master),

    "condition_summary":
        condition_df.to_dict(
            orient="records"
        ),

    "flip_summary":
        flip_df.to_dict(
            orient="records"
        ),

    "native_visual_token_bins":
        bin_df.to_dict(
            orient="records"
        ),
}

with (
    OUT
    / "day9_analysis_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# --------------------------------------------------
# Print useful tables
# --------------------------------------------------

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)


print()
print("===== CONDITION SUMMARY =====")

show = condition_df.copy()

for c in [
    "overall_em",
    "closed_em",
    "open_em",
    "open_token_f1",
]:
    show[c] *= 100

print(
    show.to_string(
        index=False,
        formatters={
            "overall_em":
                "{:.2f}%".format,
            "closed_em":
                "{:.2f}%".format,
            "open_em":
                "{:.2f}%".format,
            "open_token_f1":
                "{:.2f}%".format,
            "avg_visual_tokens":
                "{:.1f}".format,
        },
    )
)


print()
print("===== CORRECTNESS FLIPS =====")

show = flip_df.copy()

show[
    "correctness_flip_rate"
] *= 100

print(
    show.to_string(
        index=False,
        formatters={
            "correctness_flip_rate":
                "{:.2f}%".format,
        },
    )
)


print()
print(
    "===== PERFORMANCE BY "
    "LoRA-NATIVE VISUAL-TOKEN BIN ====="
)

show = bin_df.copy()

pct_cols = [
    c for c in show.columns
    if c.endswith("_em")
    or c.endswith("_rate")
]

for c in pct_cols:
    show[c] *= 100

formatters = {}

for c in pct_cols:
    formatters[c] = "{:.2f}%".format

formatters[
    "avg_native_vt"
] = "{:.1f}".format

print(
    show.to_string(
        index=False,
        formatters=formatters,
    )
)


print()
print("===== SAVED =====")

for p in [
    master_path,
    OUT / "day9_condition_summary.csv",
    OUT / "day9_flip_summary.csv",
    OUT / "day9_native_vt_bins.csv",
    OUT / "day9_analysis_summary.json",
]:
    print(p)
