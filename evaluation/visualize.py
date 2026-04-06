import pandas as pd

# 엑셀 파일 읽기 (업로드된 파일 이름 예시)
df = pd.read_excel("test.xlsx")
print(df.head(10))

# wide → long 변환
df_long = df.melt(
    id_vars=["Model", "Test Type"],
    value_vars=["API Sel P2", "API Sel P3", "Param Fill P2", "Param Fill P3"],
    var_name="Metric_Phase",
    value_name="Score"
)

# Metric, Phase 분리
df_long["Metric"] = df_long["Metric_Phase"].str.extract(r"(API Sel|Param Fill)").replace({
    "API Sel": "API Selection",
    "Param Fill": "Parameter Filling"
})
df_long["Phase"] = df_long["Metric_Phase"].str.extract(r"P(\d+)").astype(int)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

for metric in ["API Selection", "Parameter Filling"]:
    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df_long[df_long["Metric"] == metric],
        x="Phase", y="Score",
        hue="Model", style="Test Type",
        markers=True, dashes=False
    )
    plt.title(f"{metric} Performance: Phase 2 to Phase 3")
    plt.ylim(0, 100)
    plt.xticks([2, 3])
    plt.xlabel("Phase")
    plt.ylabel("Score")
    plt.legend(title="Model / Test Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
