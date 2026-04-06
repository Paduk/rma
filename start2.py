import pandas as pd

# ---------- 설정 ----------
in_path = "gpt41_datagen/it4_s3_idxed_rewrite.jsonl"
out_path = "gpt41_datagen/it4_s3_idxed_rewrite_.jsonl"
frac_keep = 0.25                  # 1/4만 유지
random_seed = 42                  # 재현 가능성용
# --------------------------

# 1) JSONL 읽기
df = pd.read_json(in_path, lines=True)

# 2) plan별 샘플링 규칙 적용
def downsample_if_needed(group):
    """group: DataFrame for one plan"""
    if len(group) > 150:
        # shuffle 후 1/4만 선택
        return group.sample(frac=frac_keep, random_state=random_seed)
    return group

filtered_df = (
    df.groupby("next_turn_plan", group_keys=False)
      .apply(downsample_if_needed)
      .reset_index(drop=True)
)

# 3) 결과 저장 (JSONL, UTF-8, 줄 단위)
filtered_df.to_json(out_path, orient="records", lines=True, force_ascii=False)

print(f"Saved {len(filtered_df)} rows → {out_path}")
