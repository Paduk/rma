import pandas as pd
import json
import random

# ---------- 파일 경로 ----------
DATA_CSV   = "all_clean_data.csv"
DESC_JSON  = "apis/plugin_des.json"
OUTPUT_CSV = "metatool_testset.tsv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ---------- 데이터 로드 ----------
df = pd.read_csv(DATA_CSV)          # Query, Tool
with open(DESC_JSON, "r", encoding="utf-8") as f:
    desc_map = json.load(f)         # {tool: description}

all_tools = df["Tool"].unique().tolist()

# ---------- 각 Tool 당 5개만 샘플 ----------
sampled = (
    df.groupby("Tool", group_keys=False)
      .head(5)
      .copy()
)

# ---------- 후보 Tool 컬럼 ----------
def build_candidates(correct_tool: str) -> list[str]:
    others = [t for t in all_tools if t != correct_tool]
    candidates = random.sample(others, 4) + [correct_tool]
    random.shuffle(candidates)
    return candidates

sampled["candidates"] = sampled["Tool"].apply(build_candidates)

# ---------- Plan 컬럼으로 변환 ----------
sampled.rename(columns={"Tool": "answer"}, inplace=True)
sampled["answer"] = sampled["answer"].apply(
    lambda t: {"plan": t, "arguments": {}}   # TSV에 저장 가능한 문자열
)

# ---------- Description 컬럼 제거 ----------
# (혹시 앞에서 만들었다면 삭제; 안 만들었으면 이 줄은 무관)
sampled = sampled.drop(columns=[c for c in sampled.columns if c.lower() == "description"], errors="ignore")

# ---------- 결과 저장 ----------
sampled.to_csv(OUTPUT_CSV, sep="\t", index=False)
print(f"Saved {len(sampled)} rows → {OUTPUT_CSV}")
