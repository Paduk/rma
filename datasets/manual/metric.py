import json
import ast
import pandas as pd
from pathlib import Path

def safe_get_plan(text):
    """Return plan value from a JSON‑like string; None if unavailable/parse error."""
    if pd.isna(text):
        return None
    # Try strict JSON first
    try:
        obj = json.loads(text)
        return obj.get("plan")
    except Exception:
        pass
    # Fallback: try Python‑style literal (single quotes, etc.)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj.get("plan")
    except Exception:
        pass
    return None

# ---------- 1. read & merge ---------------------------------------------------

tsv_paths = [
    #Path("ad/rewrite-qwen3-base.tsv"), # 56.77
    #Path("ad/history-qwen3-base.tsv"), # 49.88
    #Path("ad/rewrite-qwen3.tsv"), # 77.73
    #Path("ad/history-qwen3.tsv"), # 64.37
    Path("ad/rma-qwen3.tsv"), # 71.39
    #Path("ad/rewrite-cloud-gpt41.tsv"), # 78.51
    #Path("ad/history-cloud-gpt4o-mini.tsv"), # 65.97
    #Path("ad/rewrite-cloud-gpt4o-mini.tsv"), # 79.86
    #Path("ad/history-cloud-gemini-2.0-flash.tsv"), # 69.68
    #Path("ad/rewrite-cloud-gemini-2.0-flash.tsv"), # 76.16
    #Path("ad/history-cloud-o4-mini.tsv"), # 69.68
    #Path("ad/rewrite-cloud-o4-mini.tsv"), # 76.16
    #Path("history-cloud-gemini2.tsv"), # 75.00
    # Path("rewrite-cloud-gemini2.tsv"), # 79.86    
    #Path("qwen25-manual-rewrite.tsv"), # 69.68
    # Path("../lawbase/phi4-complex-rewrite.tsv"),
    # Path("../lawbase/phi4-base-rewrite.tsv"),
    # Path("../lawbase/phi4-complex-history.tsv"),
    # Path("../lawbase/phi4-base-history.tsv"),
    # Path("../lawbase/llama-complex-history.tsv"),
    # Path("../lawbase/llama-base-history.tsv"), # 51.07
    # Path("../lawbase/llama-complex-rewrite.tsv"),
    # Path("../lawbase/llama-base-rewrite.tsv"), # 55.79
]

df = pd.concat([pd.read_csv(p, sep="\t", dtype=str) for p in tsv_paths],
               ignore_index=True)

# drop unwanted file
df = df[df["file"] != "it2_NR_tc.tsv"]

# ---------- 2. normalise pass/fail -------------------------------------------
df["all"] = df["all"].str.lower().where(df["all"].str.lower() == "pass", "fail")

# ---------- 3. extract gt_plan ------------------------------------------------
df["gt_plan"] = df["gt"].apply(safe_get_plan)

# ---------- 4. per‑file micro accuracy ---------------------------------------
file_micro = (
    df.groupby("file")["all"]
      .apply(lambda s: (s == "pass").mean())
      .reset_index(name="micro_accuracy")
)

# ---------- 5. per‑file macro accuracy (plan averaged) ------------------------
plan_acc = (
    df[~df["gt_plan"].isna()]                # ignore rows w/o plan
      .groupby(["file", "gt_plan"])["all"]
      .apply(lambda s: (s == "pass").mean())
      .reset_index(name="plan_accuracy")
)

file_macro = (
    plan_acc.groupby("file")["plan_accuracy"]
            .mean()
            .reset_index(name="macro_accuracy")
)

# ---------- 6. merge & compute prefix ----------------------------------------
file_acc = file_micro.merge(file_macro, on="file", how="left")
#file_acc["prefix"] = file_acc["file"].str.extract(r'^(it\d+)')
file_acc["prefix"] = file_acc["file"].str.extract(r'^(turn\d+)')

# ---------- 7. prefix averages ------------------------------------------------
prefix_acc = (
    file_acc.groupby("prefix")[["micro_accuracy", "macro_accuracy"]]
            .mean()
            .reset_index()
)

overall_micro = prefix_acc["micro_accuracy"].mean()
overall_macro = prefix_acc["macro_accuracy"].mean()

print("=== Per‑file accuracies ===")
print(file_acc.to_string(index=False))

print("\n=== Prefix‑level averages ===")
print(prefix_acc.to_string(index=False))

print(f"\nOverall micro accuracy across prefixes:  {overall_micro:.6f}")
print(f"Overall macro accuracy across prefixes:  {overall_macro:.6f}")
