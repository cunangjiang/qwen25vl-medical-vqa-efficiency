import json
import csv
from pathlib import Path

ZERO = Path("results/vqa_rad/zero_shot_full451_summary.json")
LORA = Path("results/vqa_rad/lora_1ep_full451_summary.json")

OUT_JSON = Path("results/vqa_rad/day7_zero_vs_lora_comparison.json")
OUT_CSV = Path("results/vqa_rad/day7_zero_vs_lora_comparison.csv")

with ZERO.open() as f:
    z = json.load(f)

with LORA.open() as f:
    l = json.load(f)

rows = [
    {
        "metric": "overall_em",
        "zero_shot": z["accuracy_overall"],
        "lora_1ep": l["accuracy_overall"],
    },
    {
        "metric": "closed_accuracy",
        "zero_shot": z["accuracy_closed"],
        "lora_1ep": l["accuracy_closed"],
    },
    {
        "metric": "open_em",
        "zero_shot": z["accuracy_open"],
        "lora_1ep": l["accuracy_open"],
    },
    {
        "metric": "open_token_f1",
        "zero_shot": z["open_token_f1"],
        "lora_1ep": l["open_token_f1"],
    },
]

for r in rows:
    r["delta"] = r["lora_1ep"] - r["zero_shot"]
    r["delta_pp"] = r["delta"] * 100

comparison = {
    "zero_shot_model": z["model"],
    "lora_model": l["model"],
    "num_samples": l["num_samples"],
    "num_closed": l["num_closed"],
    "num_open": l["num_open"],
    "metrics": rows,
    "avg_visual_tokens": {
        "zero_shot": z["avg_visual_tokens"],
        "lora_1ep": l["avg_visual_tokens"],
    },
    "latency_ms": {
        "zero_shot": z["latency_ms"],
        "lora_1ep": l["latency_ms"],
    },
    "memory_gib": {
        "zero_shot": z["memory_gib"],
        "lora_1ep": l["memory_gib"],
    },
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

with OUT_JSON.open("w") as f:
    json.dump(comparison, f, indent=2)

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "metric",
            "zero_shot",
            "lora_1ep",
            "delta",
            "delta_pp",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("===== DAY 7: ZERO-SHOT vs LoRA =====")

for r in rows:
    print(
        f"{r['metric']:18s} "
        f"{r['zero_shot']*100:6.2f}% -> "
        f"{r['lora_1ep']*100:6.2f}% "
        f"({r['delta_pp']:+6.2f} pp)"
    )

print()
print(
    "Visual tokens:",
    f"{z['avg_visual_tokens']:.3f}",
    "->",
    f"{l['avg_visual_tokens']:.3f}",
)

print()
print("Saved:")
print(OUT_JSON)
print(OUT_CSV)
