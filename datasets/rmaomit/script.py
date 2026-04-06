#!/usr/bin/env python
# merged_row_and_total_avgs.py
"""
1) base.tsv + complex.tsv를 읽어
   (bucket=rma|history, it2~it5)별 전체 평균 → overall_averages.tsv
2) 같은 기준으로 각 Row마다 평균 + 요약 컬럼(rma_mean, history_mean, applied_rma)
   → row_averages.tsv
"""

import re, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# --------------------- 설정 ---------------------
FILES       = ["base.tsv", "complex.tsv"]           # TSV 경로
IT_BUCKETS  = {"it2", "it3", "it4", "it5"}
SKIP_FILE   = "it2_NR_tc.tsv"
BUCKETS     = ("rma", "history")

# applied_rma: history_mean을 써야 하는 예외 행들
USE_HISTORY_ROWS = {
    "ACTION_CREATE_DOCUMENT",
    "ACTION_EDIT_DOCUMENT",
    "ACTION_GET_CONTENT",
    "ACTION_GET_RINGTONE",
    "ACTION_SET_ALARM",
    "ACTION_SHOW_ALARMS",
    "ACTION_SHOW_TIMERS",
    "ACTION_VIDEO_CAPTURE",
    "ACTION_VIEW_CALL_LOG",
    "ACTION_VIEW_EVENT",
    "ACTION_VIEW_RECENT_APPS",
    "ACTION_VIEW_SENT_EMAILS",
    "ACTION_VIEW_SENT_MESSAGES",
    "ACTION_VIEW_WEB_HISTORY",
}
# ------------------------------------------------

# (bucket, it) → [values] : 전체 평균용
total_vals = {(b, it): [] for b in BUCKETS for it in IT_BUCKETS}

# row_label → (bucket, it) → [values] : row 평균용
row_vals: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
    lambda: {(b, it): [] for b in BUCKETS for it in IT_BUCKETS}
)

# ------------------------------------------------
def usable_columns(df: pd.DataFrame):
    """(col_idx, bucket, it_tag) 목록 리턴"""
    cols = []
    for j in range(1, df.shape[1]):               # col-0 = 라벨
        bucket = str(df.iat[0, j]).strip().lower()
        if bucket not in BUCKETS:
            continue
        for fname in str(df.iat[1, j]).split("|"):
            fname = fname.strip()
            if fname == SKIP_FILE:
                continue
            m = re.match(r"(it\d+)", fname)
            if m and m.group(1) in IT_BUCKETS:
                cols.append((j, bucket, m.group(1)))
                break
    return cols

def accumulate(path: Path):
    df   = pd.read_csv(path, sep="\t", header=None, dtype=str)
    cols = usable_columns(df)

    for i in range(2, df.shape[0]):               # 데이터 행
        row_label = str(df.iat[i, 0]).strip()
        for j, bucket, it_tag in cols:
            cell = df.iat[i, j]
            if pd.isna(cell):
                continue
            try:
                val = float(cell)
            except ValueError:
                continue
            total_vals[(bucket, it_tag)].append(val)
            row_vals[row_label][(bucket, it_tag)].append(val)

# ------------ 메인 ------------
for fname in FILES:
    p = Path(fname)
    if not p.exists():
        print(f"[!] {p} 없음 – 건너뜀", file=sys.stderr)
        continue
    accumulate(p)

# 1) ─────────── 전체 평균 ───────────
overall_rows = []
for it in sorted(IT_BUCKETS, key=lambda x: int(x[2:])):  # it2, it3, ...
    row = {"it": it}
    for bucket in BUCKETS:
        vals = total_vals[(bucket, it)]
        row[bucket] = sum(vals) / len(vals) if vals else None
    overall_rows.append(row)

df_overall = pd.DataFrame(overall_rows, columns=["it", *BUCKETS])
df_overall.to_csv("overall_averages.tsv", sep="\t", index=False, na_rep="")

print("\n=== 전체 평균 (base + complex) ===")
print(df_overall.to_string(index=False))

# 2) ─────────── Row 평균 + applied_rma ───────────
row_records = []
macro_placeholder = None          # MACRO 행 임시 보관

for row_label, bucket_map in row_vals.items():
    rec = {"row_label": row_label}

    # (a) it2~it5 값
    for it in IT_BUCKETS:
        for bucket in BUCKETS:
            vals = bucket_map[(bucket, it)]
            rec[f"{bucket}_{it}"] = sum(vals) / len(vals) if vals else None

    # (b) rma_mean / history_mean
    rma_vals  = [v for k, v in rec.items() if k.startswith("rma_")     and v is not None]
    hist_vals = [v for k, v in rec.items() if k.startswith("history_") and v is not None]

    rec["rma_mean"]     = sum(rma_vals)  / len(rma_vals)  if rma_vals  else None
    rec["history_mean"] = sum(hist_vals) / len(hist_vals) if hist_vals else None

    # (c) applied_rma (예외 규칙)
    if row_label in USE_HISTORY_ROWS:
        rec["applied_rma"] = rec["history_mean"]
    else:
        rec["applied_rma"] = rec["rma_mean"]

    # ▸ MACRO 행은 일단 리스트에 넣지 않고 따로 저장
    if row_label.upper() == "MACRO":
        macro_placeholder = rec
    else:
        row_records.append(rec)

# (d) MACRO 행의 applied_rma = 모든 일반 행 applied_rma 평균
if macro_placeholder is not None:
    applied_vals = [r["applied_rma"] for r in row_records if r["applied_rma"] is not None]
    macro_placeholder["applied_rma"] = (
        sum(applied_vals) / len(applied_vals) if applied_vals else None
    )
    row_records.append(macro_placeholder)   # 리스트 맨 끝에 추가

# 컬럼 순서
cols = (
    ["row_label"]
    + [f"{b}_{it}" for it in sorted(IT_BUCKETS, key=lambda x: int(x[2:]))
                      for b in BUCKETS]
    + ["rma_mean", "history_mean", "applied_rma"]
)

df_rows = pd.DataFrame(row_records, columns=cols)
df_rows.to_csv("row_averages.tsv", sep="\t", index=False, na_rep="")

print("\n=== Row 별 평균 (상위 5행) ===")
print(df_rows.head().to_string(index=False))

print("\n[완료] overall_averages.tsv, row_averages.tsv 생성")
