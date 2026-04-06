import json
import random
from collections import defaultdict

# --- user-configurable variables --------------------------------------------
#jsonl_path   = "gpt41_datagen/it1_s1_o.jsonl"        # path to your input file
jsonl_path   = "gemini_datagen/it1_s2_idxed.jsonl"        # path to your input file
api_path = "apis/droidcall_apis.jsonl"
# target_apis  = [                      # keep only these plans
#     "ACTION_CREATE_DOCUMENT",
#     "ACTION_SEND_EMAIL",
#     "ACTION_VIEW_CALL_LOG",
#     # … add more …
# ]
sample_size  = 10                     # how many items per plan
random_seed  = 42                     # make sampling reproducible
# -----------------------------------------------------------------------------

random.seed(random_seed)

# read the jsonl file ---------------------------------------------------------
target_apis = set()
with open(api_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:                      # skip empty lines
            obj = json.loads(line)
            # robust access to answer.plan
            plan = obj.get("name")
            target_apis.add(plan)

records = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:                      # skip empty lines
            obj = json.loads(line)
            # robust access to answer.plan
            plan = (obj.get("answer") or {}).get("plan")
            if plan in target_apis:
                records.append(obj)
# bucket by plan and take N samples each -------------------------------------
by_plan = defaultdict(list)
for obj in records:
    plan = obj["answer"]["plan"]      # safe: guaranteed above
    by_plan[plan].append(obj)

sampled = []
for plan, items in by_plan.items():
    # less than sample_size? just take all
    k = min(sample_size, len(items))
    sampled.extend(random.sample(items, k))

# 'sampled' now contains the filtered+sampled subset
# ---------------------------------------------------------------------------
# If you want to write it back out:
out_path = "datasets/droidcall_gemini.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for obj in sampled:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Saved {len(sampled)} records to {out_path}")
