import pandas as pd
import matplotlib.pyplot as plt

file_path = "datasets/result/all_history_coreference.tsv"
coref_data = pd.read_csv(file_path, sep="\t")

# 특정 턴 선택 (예: Turn 5)
selected_turn = 'Turn4'
turn_data = coref_data[coref_data['turn'] == selected_turn]

# 데이터 준비
models = turn_data['model'].unique()
turn_distances = ['coref_turn1', 'coref_turn2', 'coref_turn3', 'coref_turn4']
distance_labels = ['1-turn', '2-turn', '3-turn', '4-turn']

plt.figure(figsize=(12, 6))

# 각 모델별로 Turn 5 데이터만 그래프 그리기
for model in models:
    model_row = turn_data[turn_data['model'] == model]
    accuracies = model_row[turn_distances].values.flatten()
    plt.plot(distance_labels, accuracies, marker='o', label=model)

plt.title(f'Performance by Coreference Distance ({selected_turn})')
plt.xlabel('Coreference Distance')
plt.ylabel('Accuracy (%)')
plt.grid(axis='y', linestyle=':')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
