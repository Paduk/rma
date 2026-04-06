import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

file_path = "../datasets/result/all_performance.tsv"
df = pd.read_csv(file_path, sep="\t")

df = df[df["method"].isin(["Ours", "Baseline"])]

turn_cols   = ["turn_2", "turn_3", "turn_4", "turn_5"]
turn_labels = [2, 3, 4, 5]

models      = sorted(df["model"].unique())
cmap        = get_cmap("tab10")
model_color = {m: cmap(i % 10) for i, m in enumerate(models)}

plt.figure(figsize=(12, 6))

for model in models:
    sub = df[df["model"] == model]

    ours     = sub[sub["method"] == "Ours"]
    baseline = sub[sub["method"] == "Baseline"]
    
    if not ours.empty:
        plt.plot(
            turn_labels,
            ours[turn_cols].values.flatten(),
            marker="o",
            linestyle="-",
            linewidth=2.2,
            color=model_color[model],
            label=f"{model} – Ours",
        )
    
    if not baseline.empty:
        plt.plot(
            turn_labels,
            baseline[turn_cols].values.flatten(),
            marker="s",
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.6,
            label=f"{model} – Baseline",
        )

plt.title("Multi-turn Accuracy Comparison (Turn 2→5)")
plt.xlabel("Turn")
plt.ylabel("Accuracy (%)")
plt.xticks(turn_labels)
plt.grid(axis="y", linestyle=":")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
