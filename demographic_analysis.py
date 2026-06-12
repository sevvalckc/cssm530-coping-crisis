import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import json

# ── 1. Load corpus predictions ──
df = pd.read_csv("data/cssm530_corpus_predictions.csv")
df["date"] = pd.to_datetime(df["date"])

# ── 2. Extract demographic variables from Politus zip ──
zip_path = "path/to/deprem_tweets_260401.zip"
records = []
with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("deprem_tweets_260401.jsonl") as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "tweet_id": record.get("_id"),
                "gender": record.get("gender"),
                "org": record.get("org"),
                "age_group": record.get("age_group")
            })

demog_df = pd.DataFrame(records)
demog_df["tweet_id"] = demog_df["tweet_id"].astype(str)
df["tweet_id"] = df["tweet_id"].astype(str)

# ── 3. Merge ──
merged = df.merge(demog_df, on="tweet_id", how="left")

# ── 4. Compute coping rates by demographic group ──
PRED_COLS = ["pred_problem", "pred_emotion", "pred_meaning"]
PRED_NAMES = ["Problem-Focused", "Emotion-Focused", "Meaning-Making"]

results = {}
for var in ["gender", "org", "age_group"]:
    results[var] = merged.groupby(var)[PRED_COLS].mean().round(3)
    print(f"\n=== {var.upper()} ===")
    print(results[var])

print("\n=== PHASE x GENDER ===")
print(merged.groupby(["phase", "gender"])[PRED_COLS].mean().round(3))

# ── 5. Figure 5 ──
C = {"problem": "#5B9BD5", "emotion": "#E07B8A", "meaning": "#F5A962"}
colors = [C["problem"], C["emotion"], C["meaning"]]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
fig.suptitle("Coping Style Distribution by Demographic Groups", fontsize=14, fontweight="bold")

width = 0.25

for ax, var, xlabels in zip(
    axes,
    ["gender", "org", "age_group"],
    [None, ["Organization", "Person"], ["<=18", "19-29", "30-39", ">=40"]]
):
    data = results[var]
    if xlabels:
        data = data.reindex(["org", "person"]) if var == "org" else data.reindex(["<=18", "19-29", "30-39", ">=40"])
    x = np.arange(len(data))
    for i, (col, name, color) in enumerate(zip(PRED_COLS, PRED_NAMES, colors)):
        ax.bar(x + i * width, data[col], width, label=name, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(xlabels if xlabels else data.index)
    ax.set_title(var.replace("_", " ").title())
    ax.set_ylim(0, 0.7)
    if ax is axes[0]:
        ax.set_ylabel("Mean Coping Rate")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig5_demographic_coping.png", dpi=300, bbox_inches="tight")
plt.show()
print("Figure 5 saved.")
