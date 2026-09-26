from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/vqa_rad"
FIG_DIR = ROOT / "assets/figures"
METRIC_DIR = ROOT / "experiments/metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def percent_axis(ax):
    ax.set_ylim(0, 1)
    ticks = np.linspace(0, 1, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100*x:.0f}%" for x in ticks])


# ------------------------------------------------------------------
# Load only verified experiment outputs.
# ------------------------------------------------------------------
day7 = pd.read_csv(RESULT_ROOT / "day7_zero_vs_lora_comparison.csv")
cond = pd.read_csv(RESULT_ROOT / "day9_analysis/day9_condition_summary.csv")
events = pd.read_csv(RESULT_ROOT / "day9_analysis/day9_transition_events_final.csv")
qa = pd.read_csv(
    RESULT_ROOT / "day9_analysis/qualitative_cases/qualitative_case_qa.csv"
)

with open(
    RESULT_ROOT / "day8_resolution/day8_latency_repeated_summary.json",
    "r",
    encoding="utf-8",
) as f:
    latency = json.load(f)


# ------------------------------------------------------------------
# Compact source tables used by public figures.
# ------------------------------------------------------------------
metric_names = {
    "overall_em": "Overall EM",
    "closed_accuracy": "Closed EM",
    "open_em": "Open EM",
    "open_token_f1": "Open token F1",
}
main_quality = day7.copy()
main_quality["display_metric"] = main_quality["metric"].map(metric_names)
main_quality.to_csv(METRIC_DIR / "main_zero_vs_lora.csv", index=False)

budget_order = ["lora_vt256", "lora_vt512", "lora_vt1024", "lora_native"]
budget_labels = {
    "lora_vt256": "VT256",
    "lora_vt512": "VT512",
    "lora_vt1024": "VT1024",
    "lora_native": "Native",
}
budget_quality = cond[cond["condition"].isin(budget_order)].copy()
budget_quality["condition"] = pd.Categorical(
    budget_quality["condition"], categories=budget_order, ordered=True
)
budget_quality = budget_quality.sort_values("condition")
budget_quality["display_name"] = budget_quality["condition"].map(budget_labels)
budget_quality.to_csv(METRIC_DIR / "visual_budget_quality.csv", index=False)

lat_rows = []
for key in ["vt256", "vt512", "vt1024", "lora_native"]:
    c = latency["conditions"][key]
    paired = latency.get("paired_vs_lora_native", {}).get(key, {})
    lat_rows.append({
        "condition": key,
        "display_name": {
            "vt256": "VT256",
            "vt512": "VT512",
            "vt1024": "VT1024",
            "lora_native": "Native",
        }[key],
        "n_measurements": c["n_measurements"],
        "avg_visual_tokens_benchmark50": c["avg_visual_tokens"],
        "e2e_mean_ms": c["e2e_ms"]["mean"],
        "e2e_median_ms": c["e2e_ms"]["median"],
        "e2e_p95_ms": c["e2e_ms"]["p95"],
        "paired_mean_e2e_reduction_pct":
            paired.get("e2e_reduction_pct", {}).get("mean", np.nan),
        "paired_median_e2e_reduction_pct":
            paired.get("e2e_reduction_pct", {}).get("median", np.nan),
        "fraction_e2e_faster":
            paired.get("fraction_e2e_faster", np.nan),
    })
lat_df = pd.DataFrame(lat_rows)
lat_df.to_csv(METRIC_DIR / "latency_repeated_summary.csv", index=False)

profiles = []
for reason in ["lora_gain", "lora_loss", "vt256_loss"]:
    g = events[events["reason"] == reason].groupby("semantic_target").size()
    for k, v in g.items():
        profiles.append({
            "event_group": reason,
            "semantic_target": k,
            "count": int(v),
        })
profile_df = pd.DataFrame(profiles)
profile_df.to_csv(METRIC_DIR / "semantic_flip_profile.csv", index=False)

comparison_order = [
    "zero_native -> lora_native",
    "lora_native -> lora_vt1024",
    "lora_native -> lora_vt512",
    "lora_native -> lora_vt256",
]
comparison_labels = {
    "zero_native -> lora_native": "Zero → LoRA",
    "lora_native -> lora_vt1024": "Native → VT1024",
    "lora_native -> lora_vt512": "Native → VT512",
    "lora_native -> lora_vt256": "Native → VT256",
}
net_rows = []
for comp in comparison_order:
    g = events[events["comparison"] == comp]
    strict_gain = int((g["direction"] == "gain").sum())
    strict_loss = int((g["direction"] == "loss").sum())
    sem_corr = int((g["semantic_effect"] == "correction").sum())
    sem_reg = int((g["semantic_effect"] == "regression").sum())
    net_rows.append({
        "comparison": comp,
        "display_name": comparison_labels[comp],
        "strict_gain": strict_gain,
        "strict_loss": strict_loss,
        "strict_net": strict_gain - strict_loss,
        "semantic_correction": sem_corr,
        "semantic_regression": sem_reg,
        "semantic_net": sem_corr - sem_reg,
    })
net_df = pd.DataFrame(net_rows)
net_df.to_csv(METRIC_DIR / "strict_vs_semantic_net.csv", index=False)


# ------------------------------------------------------------------
# 01 Pipeline
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 3.7))
ax.axis("off")
boxes = [
    ("Medical image\n+ question", 0.02),
    ("Qwen2.5-VL-3B\nzero-shot", 0.22),
    ("1-epoch LoRA SFT\nLLM-side all-linear", 0.42),
    ("Visual budget scan\nmax_pixels / visual tokens", 0.62),
    ("Quality + efficiency\n+ error analysis", 0.82),
]
box_w, box_h, y = 0.16, 0.42, 0.30
for text_str, x in boxes:
    ax.add_patch(FancyBboxPatch(
        (x, y), box_w, box_h, boxstyle="round,pad=0.02", linewidth=1.5
    ))
    ax.text(
        x + box_w/2, y + box_h/2, text_str,
        ha="center", va="center", fontsize=11
    )
for i in range(len(boxes) - 1):
    x1 = boxes[i][1] + box_w
    x2 = boxes[i+1][1]
    ax.add_patch(FancyArrowPatch(
        (x1 + 0.008, y + box_h/2),
        (x2 - 0.008, y + box_h/2),
        arrowstyle="->", mutation_scale=14, linewidth=1.4
    ))
ax.text(
    0.5, 0.90,
    "Medical VQA adaptation and visual-budget sensitivity study",
    ha="center", va="center", fontsize=15, fontweight="bold"
)
ax.text(
    0.5, 0.08,
    "Base model and LoRA are existing methods; the project contribution is "
    "the controlled evaluation, profiling, and analysis pipeline.",
    ha="center", va="center", fontsize=9
)
save_fig(fig, "01_pipeline")


# ------------------------------------------------------------------
# 02 Zero-shot vs LoRA
# ------------------------------------------------------------------
labels = main_quality["display_metric"].tolist()
x = np.arange(len(labels))
w = 0.36
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.bar(x - w/2, main_quality["zero_shot"], width=w, label="Zero-shot")
ax.bar(x + w/2, main_quality["lora_1ep"], width=w, label="1-epoch LoRA")
for xpos, val in zip(x - w/2, main_quality["zero_shot"]):
    ax.text(xpos, val + 0.025, f"{100*val:.1f}%", ha="center", fontsize=9)
for xpos, val in zip(x + w/2, main_quality["lora_1ep"]):
    ax.text(xpos, val + 0.025, f"{100*val:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
percent_axis(ax)
ax.set_title("Zero-shot vs. 1-epoch LoRA on VQA-RAD")
ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.text(
    0.5, -0.18,
    "Project-normalized metrics on the dataset-provided QA split; "
    "the split is not image-disjoint.",
    ha="center", va="top", transform=ax.transAxes, fontsize=9
)
save_fig(fig, "02_zero_vs_lora")


# ------------------------------------------------------------------
# 03 Quality vs visual-token workload
# ------------------------------------------------------------------
q = budget_quality.sort_values("avg_visual_tokens")
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.plot(q["avg_visual_tokens"], q["overall_em"], marker="o", linewidth=2)
for _, r in q.iterrows():
    ax.annotate(
        f"{r['display_name']}\n{100*r['overall_em']:.2f}%",
        (r["avg_visual_tokens"], r["overall_em"]),
        xytext=(7, 7), textcoords="offset points", fontsize=9
    )
ax.set_xlabel("Average visual tokens per QA (full 451-QA test)")
ax.set_ylabel("Overall normalized EM")
percent_axis(ax)
ax.set_xlim(0, float(q["avg_visual_tokens"].max()) * 1.10)
ax.set_title("Quality vs. visual-token workload")
ax.grid(alpha=0.25)
ax.text(
    0.5, -0.18,
    "VT512 reduces average visual tokens from 871.6 to 422.8 (−51.5%) "
    "with no observed overall-EM loss on this fixed split.",
    ha="center", va="top", transform=ax.transAxes, fontsize=9
)
save_fig(fig, "03_quality_vs_visual_tokens")


# ------------------------------------------------------------------
# 04 Quality-latency tradeoff
# ------------------------------------------------------------------
quality_map = {
    "vt256": float(cond.loc[cond["condition"]=="lora_vt256","overall_em"].iloc[0]),
    "vt512": float(cond.loc[cond["condition"]=="lora_vt512","overall_em"].iloc[0]),
    "vt1024": float(cond.loc[cond["condition"]=="lora_vt1024","overall_em"].iloc[0]),
    "lora_native": float(cond.loc[cond["condition"]=="lora_native","overall_em"].iloc[0]),
}
trade = lat_df.copy()
trade["overall_em_full451"] = trade["condition"].map(quality_map)
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.scatter(trade["e2e_mean_ms"], trade["overall_em_full451"], s=70)
for _, r in trade.iterrows():
    ax.annotate(
        f"{r['display_name']}\n{r['e2e_mean_ms']:.0f} ms",
        (r["e2e_mean_ms"], r["overall_em_full451"]),
        xytext=(7, 7), textcoords="offset points", fontsize=9
    )
ax.set_xlabel("Mean E2E latency (ms)")
ax.set_ylabel("Overall normalized EM")
percent_axis(ax)
ax.set_xlim(0, float(trade["e2e_mean_ms"].max()) * 1.08)
ax.set_title("Quality–latency trade-off")
ax.grid(alpha=0.25)
ax.text(
    0.5, -0.19,
    "Quality: full 451-QA test. Latency: fixed 50-QA set, "
    "4 repeats, model loaded once.",
    ha="center", va="top", transform=ax.transAxes, fontsize=9
)
save_fig(fig, "04_quality_latency_tradeoff")


# ------------------------------------------------------------------
# 05 Fine-grained error profile
# ------------------------------------------------------------------
pivot = profile_df.pivot_table(
    index="semantic_target", columns="event_group", values="count",
    aggfunc="sum", fill_value=0
)
for c in ["lora_gain", "lora_loss", "vt256_loss"]:
    if c not in pivot.columns:
        pivot[c] = 0
pivot["total"] = pivot[["lora_gain","lora_loss","vt256_loss"]].sum(axis=1)
pivot = pivot.sort_values("total", ascending=True).drop(columns="total")

pretty = {
    "finding_presence_absence": "Finding presence / absence",
    "morphology_size_attribute": "Morphology / size / attribute",
    "localization_laterality": "Localization / laterality",
    "anatomy_visibility_normality": "Anatomy visibility / normality",
    "anatomy_structure_identification": "Anatomy structure identification",
    "modality_plane_sequence": "Modality / plane / sequence",
    "diagnosis_condition": "Diagnosis / condition",
    "finding_description": "Finding description",
    "context_demographic_device": "Context / demographic / device",
}
y = np.arange(len(pivot))
h = 0.24
fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.barh(y - h, pivot["lora_gain"], height=h, label="LoRA gain")
ax.barh(y, pivot["lora_loss"], height=h, label="LoRA loss")
ax.barh(y + h, pivot["vt256_loss"], height=h, label="VT256 loss")
ax.set_yticks(y)
ax.set_yticklabels([pretty.get(i, i) for i in pivot.index])
ax.set_xlabel("Number of correctness-flip events")
ax.set_title("Fine-grained error profile on affected QA cases")
ax.legend()
ax.grid(axis="x", alpha=0.25)
ax.text(
    0.5, -0.12,
    "Counts are from the correctness-flip subset, not per-category accuracy rates.",
    ha="center", va="top", transform=ax.transAxes, fontsize=9
)
save_fig(fig, "05_fine_grained_error_profile")


# ------------------------------------------------------------------
# 06 Strict EM net vs semantic-aware net
# ------------------------------------------------------------------
x = np.arange(len(net_df))
w = 0.36
fig, ax = plt.subplots(figsize=(9.5, 5.6))
b1 = ax.bar(x - w/2, net_df["strict_net"], width=w, label="Strict-EM net")
b2 = ax.bar(x + w/2, net_df["semantic_net"], width=w, label="Semantic-aware net")
for bars in [b1, b2]:
    for b in bars:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            v + (0.7 if v >= 0 else -0.7),
            f"{int(v):+d}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
ax.axhline(0, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(net_df["display_name"], rotation=10)
ax.set_ylabel("Net number of QA transitions")
ax.set_title("Strict exact-match changes vs. semantic-aware review")
ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.text(
    0.5, -0.18,
    "Semantic-aware net counts only definite corrections minus definite regressions; "
    "it is a project-internal manual taxonomy.",
    ha="center", va="top", transform=ax.transAxes, fontsize=9
)
save_fig(fig, "06_strict_vs_semantic_net")


# ------------------------------------------------------------------
# 07 Representative qualitative cases
# ------------------------------------------------------------------
case_specs = [
    {
        "image_id": "fd7bcec7d8",
        "title": "A. LoRA mixed behavior on the same image",
        "indices": [107, 102],
        "note":
            "Observed: infarction presence is corrected, while lesion laterality "
            "flips from left to right.",
    },
    {
        "image_id": "a253d30997",
        "title": "B. Resolution-sensitive anatomy identification",
        "indices": [443],
        "note":
            "Observed: LoRA-native / VT1024 answer psoas muscle; "
            "VT512 / VT256 answer cecum.",
    },
    {
        "image_id": "834c75b256",
        "title": "C. Exact-match wording sensitivity",
        "indices": [399],
        "note":
            "Observed: native 'nodules' becomes exact GT wording "
            "'nodular opacities' at VT512.",
    },
    {
        "image_id": "e551b2f8ad",
        "title": "D. Non-monotonic visual-budget response",
        "indices": [212],
        "note":
            "Observed: native ✓, VT1024 ✗, VT512 ✓, VT256 ✗ for the same QA.",
    },
]

font_regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

def get_font(path, size):
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()

font_title = get_font(font_bold, 34)
font_section = get_font(font_bold, 25)
font_body = get_font(font_regular, 21)
font_small = get_font(font_regular, 18)

W, margin, row_h = 1800, 55, 540
H = margin * 2 + row_h * len(case_specs) + 70
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)

draw.text((margin, 22), "Representative qualitative cases", font=font_title, fill="black")
draw.text(
    (margin, 65),
    "Facts shown below are direct prediction transitions; causal explanations "
    "are intentionally not asserted.",
    font=font_small, fill="black"
)

img_x, img_w = margin, 500
text_x = img_x + img_w + 55
y0 = 110

for i, spec in enumerate(case_specs):
    y = y0 + i * row_h
    draw.line((margin, y - 10, W - margin, y - 10), fill="black", width=1)
    draw.text((text_x, y + 5), spec["title"], font=font_section, fill="black")

    img_path = (
        RESULT_ROOT
        / f"day9_analysis/qualitative_cases/images/{spec['image_id']}.png"
    )
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((img_w, row_h - 75))
    ix = img_x + (img_w - im.width)//2
    iy = y + 45 + (row_h - 75 - im.height)//2
    canvas.paste(im, (ix, iy))

    lines = []
    for idx in spec["indices"]:
        r = qa[
            (qa["image_id"] == spec["image_id"])
            & (qa["dataset_index"] == idx)
        ].iloc[0]
        lines.extend([
            f"QA {idx}: {r['question']}",
            f"GT: {r['gt']}",
            f"Zero-shot: {r['zero_native__prediction_raw']}",
            f"LoRA-native: {r['lora_native__prediction_raw']}",
            "VT1024 / VT512 / VT256: "
            f"{r['lora_vt1024__prediction_raw']} / "
            f"{r['lora_vt512__prediction_raw']} / "
            f"{r['lora_vt256__prediction_raw']}",
            "",
        ])
    lines.append(spec["note"])

    wrapped = []
    for line in lines:
        wrapped.extend([""] if not line else textwrap.wrap(str(line), width=92))

    ty = y + 55
    for line in wrapped:
        draw.text((text_x, ty), line, font=font_body, fill="black")
        ty += 31

canvas.save(FIG_DIR / "07_qualitative_cases.png", dpi=(300, 300))
canvas.save(FIG_DIR / "07_qualitative_cases.pdf", "PDF", resolution=300.0)
