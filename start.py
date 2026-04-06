import pandas as pd

# JSONL 파일 경로
file_path = "o4_datagen/it4_s3_idxed_rewrite.jsonl"
file_path = "gemini_datagen/it4_s3_idxed_rewrite.jsonl"
file_path = "gpt41_datagen/it4_s3_idxed_rewrite_.jsonl"

# 1) JSONL 읽기
df = pd.read_json(file_path, lines=True)

# 2) 건수 집계
counts = df["next_turn_plan"].value_counts(dropna=False)   # NaN 포함하려면 dropna=False

# 3) 비율(%) 계산
total = len(df)
percents = (counts / total * 100).round(2)

# 4) 결과 DataFrame 구성
result = pd.DataFrame({
    "count": counts,
    "percent(%)": percents
})

# 5) 빈도순(건수 내림차순)으로 정렬해 보기 좋게 출력
print(result.sort_values("count", ascending=False).to_string())
