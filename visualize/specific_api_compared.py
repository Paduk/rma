import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
models = ['o4-mini', 'gpt41', 'gemini-2.0-flash', 'gpt-4o-mini', 'llama3-sft', 'phi-4-sft', 'qwen2.5-3b', 'qwen3-4b']

ours_all = [83.9, 81.71, 81, 79.91, 81.54, 83.08, 82.2, 82.62]
ours_multi = [74.07, 74.53, 71.63, 71.87, 69.79, 70.29, 69.59, 71.12]
baseline_all = [76.24, 75.14, 76.21, 72.04, 72.63, 76, 72.04, 75.01]
baseline_multi = [61.89, 62.62, 61.09, 57.56, 58.01, 60.87, 54.97, 59.74]

# 바의 위치 설정
x = np.arange(len(models))
width = 0.2

# 그래프 그리기
fig, ax = plt.subplots(figsize=(14, 7))

bars_ours_all = ax.bar(x - 1.5*width, ours_all, width, label='Ours - All', color='skyblue')
bars_ours_multi = ax.bar(x + 0.5*width, ours_multi, width, label='Ours - Multi-specific', color='dodgerblue')
bars_baseline_all = ax.bar(x - 0.5*width, baseline_all, width, label='Baseline - All', color='lightcoral')
bars_baseline_multi = ax.bar(x + 1.5*width, baseline_multi, width, label='Baseline - Multi-specific', color='salmon')

# annotation 함수 정의
def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# annotation 추가
annotate_bars(ax, bars_ours_all)
annotate_bars(ax, bars_ours_multi)
annotate_bars(ax, bars_baseline_all)
annotate_bars(ax, bars_baseline_multi)

# 성능 하락 화살표 및 수치 표시
for i in range(len(models)):
    # Ours 성능 하락
    decrease_ours = ours_all[i] - ours_multi[i]
    ax.annotate('', xy=(x[i]-1.5*width, ours_multi[i]), xytext=(x[i]-1.5*width, ours_all[i]),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8))
    ax.text(x[i]-1.5*width, ours_multi[i] - 4, f'-{decrease_ours:.1f}%', ha='center', va='top', fontsize=8, color='blue')

    # Baseline 성능 하락
    decrease_baseline = baseline_all[i] - baseline_multi[i]
    ax.annotate('', xy=(x[i]+0.5*width, baseline_multi[i]), xytext=(x[i]+0.5*width, baseline_all[i]),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8))
    ax.text(x[i]+0.5*width, baseline_multi[i] - 4, f'-{decrease_baseline:.1f}%', ha='center', va='top', fontsize=8, color='red')

# 축과 레이블 설정
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Performance (%)', fontsize=12)
ax.set_title('Performance Comparison: All vs Multi-specific', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()