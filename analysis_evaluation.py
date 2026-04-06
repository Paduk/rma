import pandas as pd
import ast

# 1. Read the TSV file
input_file = 'gem_logs/gemini_it_all_re_rma.tsv'  # TSV file path
input_file = 'gem_logs/gemini_it_all_history_base.tsv'
output_file = input_file.replace('.tsv', '_detailed.tsv')  # Output file path
df = pd.read_csv(input_file, sep='\t', dtype=str)

# 2. Compute overall accuracy per file (based on the 'all' column) and print it
accuracy = (
    df.groupby('file')['all']
      .apply(lambda x: (x == 'pass').mean() * 100)
      .round(2)
)
print("Overall accuracy per file (%):")
print(accuracy.to_string())

# 3. Extract ground-truth plan from the 'gt' column
df['gt_plan'] = df['gt'].apply(lambda x: ast.literal_eval(x)['plan'])

# 4. Aggregate wrong-case stats per file & gt_plan
records = []
file_totals = df['file'].value_counts().to_dict()
for file_name, group_file in df.groupby('file'):
    total = file_totals[file_name]
    for plan_name, group_plan in group_file.groupby('gt_plan'):
        wrong_count = (group_plan['all'] != 'pass').sum()
        wrong_pct = round(wrong_count / total * 100, 2)        
        records.append({
            'file': file_name,
            'gt_plan': plan_name,
            'total_cases': total,
            'wrong_count': wrong_count,
            'wrong_pct': wrong_pct
        })

# 5. Create and sort DataFrame
stats_df = pd.DataFrame(records)
stats_df = stats_df.sort_values(['file', 'wrong_pct'], ascending=[True, False])

# 6. Save detailed stats to TSV
stats_df.to_csv(output_file, sep='\t', index=False)

print(f"Saved detailed stats to {output_file}")
