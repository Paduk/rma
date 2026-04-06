import pandas as pd
import json, ast

# 분석할 파일 목록
files = [
    "testset/it5_o4_1_tc.tsv",
    "testset/it5_o4_2_tc.tsv",
    "testset/it5_o4_3_tc.tsv"
]

# 파일 경로 → plan별 개수를 저장할 딕셔너리
counts_dict = {}

for path in files:
    # TSV에서 answer 컬럼만 읽어서 JSON 파싱 후 plan 추출
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=["answer"])
    plans = df["answer"] \
        .apply(ast.literal_eval) \
        .apply(lambda ans: ans.get("plan"))
    
    # plan별 빈도 계산
    counts = plans.value_counts()
    counts_dict[path] = counts

# 모든 파일에 공통으로 있는 plan의 교집합 계산
common_plans = set.intersection(
    *(set(cnt.index) for cnt in counts_dict.values())
)

# 결과 출력
print("📋 모든 파일에 공통으로 등장하는 plan과 각 파일별 개수:")
for plan in sorted(common_plans):
    line = [f"{plan}"]
    for path in files:
        cnt = counts_dict[path].get(plan, 0)
        line.append(f"{path.split('/')[-1]}: {cnt}")
    print("  - " + " | ".join(line))
