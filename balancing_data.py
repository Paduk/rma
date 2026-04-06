import json
import argparse
import random
import pdb
from utils.frequently_used_tools import get_arg_parse, read_jsonl, print_data_cnt_per_plan
    
def get_datas(datas):    
    cnt = {}
    detail_cnt = {}

    for data in datas:                
        plan = data["answer"]["plan"]        
        u_idx = data["unique_idx"]
        u_keys = u_idx.split('-') # prev_plan, plan, _        
        plan = u_keys[-2]                
        key = '-'.join(u_keys[::2]) # f"{prev_plan}-{plan}"        
        cnt[plan] = cnt.get(plan, 0) + 1
        detail_cnt[key] = detail_cnt.get(key, 0) + 1
        #detail_cnt[u_idx] = detail_cnt.get(u_idx, 0) + 1
        
    return cnt, detail_cnt

#python3 balancing_data.py --d1 datagen/it2_s1.jsonl --d2 datagen/it1_supplyments.jsonl --o datagen/it2_s1_int.jsonl

args = get_arg_parse()

d1 = read_jsonl(args.t1) # "datagen/it2_s1.jsonl"
d2 = read_jsonl(args.t2) # "datagen/it1_supplyments.jsonl"
for d in d2:
    d1.append(d)

print(f"start data cnt: {len(d1)}")
unique_index = set()
intergrated = []
for d in d1: # u_idx 중복 제거
    u_idx = d["unique_idx"]
    if u_idx in unique_index:
        if d["query"] == d["rewrited_query"]:
            continue
    else:
        unique_index.add(u_idx)
    
    intergrated.append(d)

intergrated_datas = [] # u_idx 중복 제거 후 한번 더 확인
unique_index = set()
print(f"1st filtered cnt: {len(intergrated)}")
for d in intergrated:
    u_idx = d["unique_idx"]
    if u_idx in unique_index:        
        continue
    else:
        unique_index.add(u_idx)
    
    intergrated_datas.append(d)

print(f"2nd filtered cnt: {len(intergrated_datas)}")

target_key = {}
datacnt_per_plan = {}
c2, d_c2 = get_datas(intergrated_datas)
for k in c2:
    cnt = c2[k]    
    if cnt > 100:                
        print(k, c2[k])
        for dk in d_c2:
            #pp, p = dk.split('-')
            sub_plan_keys = dk.split('-')
            p = sub_plan_keys[-1]
            pp = '-'.join(sub_plan_keys)
            if k == p:                                
                if cnt > 800:
                    ratio = 0.04
                elif cnt > 500:
                    ratio = 0.05
                elif cnt > 250:
                    ratio = 0.1
                elif cnt > 150:
                    ratio = 0.2                
                elif cnt > 100:
                    ratio = 0.1
                #print(dk, d_c2[dk])

                f_cnt = int(d_c2[dk] * ratio)
                if f_cnt < 1:
                    f_cnt = 1                    

                if k in datacnt_per_plan and datacnt_per_plan[k] > 50:
                    target_key[dk] = 0
                    continue
                else:
                    target_key[dk] = f_cnt
                
                datacnt_per_plan[k] = datacnt_per_plan.get(k, 0) + f_cnt
        #print()    

for k in datacnt_per_plan:
     print(k, datacnt_per_plan[k])       

total_cnt = 0
redundancy_cnt = 0
output_file = open(args.o, "w")
redundancy_out_file = open(args.o.replace(".jsonl", "_redundancy.jsonl"), "w")

redun_datas = []
gen_datas = []
for d in intergrated_datas:
    planh = d["unique_idx"].split('-')    
    key = '-'.join(planh[::2])    
    if key in target_key:        
        if target_key[key] > 0:
            target_key[key] -= 1
        else:
            redundancy_cnt += 1
            redundancy_out_file.write(json.dumps(d, ensure_ascii=False)+"\n")
            redundancy_out_file.flush()
            redun_datas.append(d)
            continue    
        
    total_cnt += 1
    output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
    output_file.flush()
    gen_datas.append(d)

print(f"redundancy_out_file data: {redundancy_cnt}")
print(f"data minimized {len(d1)} -> {total_cnt}")

print_data_cnt_per_plan(redun_datas, keys="redundancy", model_name=args.model)
print_data_cnt_per_plan(gen_datas, keys="balanced", model_name=args.model)
output_file.close()
redundancy_out_file.close()