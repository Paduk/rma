import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

file_path = "all_performance.tsv"
df = pd.read_csv(file_path, sep="\t")

baseline_df = df[df["method"] == "Baseline"]
ours_df     = df[df["method"] == "Ours"]

turn_cols   = ["turn_2", "turn_3", "turn_4", "turn_5"]
turn_labels = [2, 3, 4, 5]

models      = sorted(baseline_df["model"].unique())
cmap        = get_cmap("tab10")
color_map   = {m: cmap(i % 10) for i, m in enumerate(models)}

plt.figure(figsize=(12, 6))

for model in models:
    row = baseline_df[baseline_df["model"] == model]
    if not row.empty:
        plt.plot(
            turn_labels,
            row[turn_cols].values.flatten(),
            marker="s",
            linestyle="-",
            linewidth=2.8,
            color=color_map[model],
            label=f"{model} – Baseline",
        )

for model in models:
    row = ours_df[ours_df["model"] == model]
    if not row.empty:
        plt.plot(
            turn_labels,
            row[turn_cols].values.flatten(),
            marker="o",
            linestyle="--",
            linewidth=1.0,
            color="lightgray",
            alpha=0.4,
        ) 

plt.title("Baseline Trend across Turns (Ours shaded for context)")
plt.xlabel("Turn")
plt.ylabel("Accuracy (%)")
plt.xticks(turn_labels)
plt.grid(axis="y", linestyle=":")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()