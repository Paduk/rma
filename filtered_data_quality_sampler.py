import random
from utils.frequently_used_tools import read_jsonl, save_jsonl, get_arg_parse
from collections import defaultdict

args = get_arg_parse()
datas = read_jsonl(args.t)  # "datagen/it2_s1_int.jsonl"
output_file_name = f"{'.'.join(args.t.split('.')[:-1])}_sampler.jsonl"  # "datagen/it2_s1_int.jsonl"
output_datas = []

# 1. plan별로 그룹핑
plan_groups = defaultdict(list)
for data in datas:
    plan = data['answer']['plan']
    plan_groups[plan].append(data)

# 2. 각 plan별로 최대 3개씩 랜덤 샘플링해서 output_datas에 추가
for plan, items in plan_groups.items():
    sample_size = min(3, len(items))
    sampled = random.sample(items, sample_size)
    output_datas.extend(sampled)

bad_count = 0

for idx, data in enumerate(output_datas, start=1):    
    print(f"\n[{idx}/{len(output_datas)}]")
    
    for key in data:
        if key == 'conversation_history':
            for history in data[key]:
                print(f"{key}: {history}")
        else:
            print(f"{key}: {data[key]}")
    
    while True:
        inp = input("if data has issue, put 1 else 0 (0/1): ").strip()
        if inp in ("0", "1"):
            break
        print("0 or 1 only")

    label = int(inp)    
    data["is_problematic"] = label
    
    bad_count += label

total = len(output_datas)
ratio = bad_count / total if total > 0 else 0

print(f"\nratio of problematic: {ratio:.2f}")

save_jsonl(output_file_name, output_datas)
print(f"len of plans: {len(plan_groups)}")
print(f"len of sampled data: {len(output_datas)}")
print(f"output_file_name: {output_file_name} done")
