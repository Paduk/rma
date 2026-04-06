import json
import argparse
import random

def get_apis(f_path="apis/api_v3.0.1.jsonl"):
    api_file = f_path
    apis = set()
    with open(api_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                apis.add(json.loads(line)["plan"])
    return apis

def get_generated_info(f_path="datagen/it2_s1.jsonl"):
    gen_file = f_path
    gen_apis = set()
    cnt_each_api = dict()
    with open(gen_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    plan = data["answer"]["plan"]
                    gen_apis.add(plan)
                    cnt_each_api[plan] = cnt_each_api.get(plan, 0) + 1
    return gen_apis, cnt_each_api

def get_data(f_path="datagen/it1_s1_spare.jsonl"):
    file = f_path
    datas = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                datas.append(json.loads(line))
    return datas

def convert_to_dict(datas):
    d = {}
    for data in datas:
        plan = data["answer"]["plan"]
        if plan not in d:
            d[plan] = []
        d[plan].append(data)
    
    return d

def add_data(s1_data, s2_data, unique_dict):
    new_d = {"conversation_history": [], "query": "", "rewrited_query": "", "answer": {}, "unique_idx": ""} 
    new_d["conversation_history"] = s2_data["conversation_history"]

    conv_len = len(new_d["conversation_history"])    
    new_d["conversation_history"].append(f'turn {conv_len + 1}: {s2_data["query"]} -> {s2_data["device_response"]}')
    new_d["query"] = new_d["rewrited_query"] = s1_data["query"]
    new_d["answer"] = s1_data["answer"]

    cnt = 1    
    while True:
        u_idx = f'{s2_data["unique_idx"]}-{new_d["answer"]["plan"]}-{cnt}_NR'
        if u_idx in unique_dict:
            cnt += 1
        else:
            break
            
    new_d["unique_idx"] = u_idx
    
    return new_d

def __main__(): 
    # python3 filling_datas.py --s1 datagen/it1_s1_spare.jsonl --s datagen/it1_s2_idxed.jsonl --o datagen/it1_supplyments.jsonl       
    parser = argparse.ArgumentParser(description="data integration")
    #parser.add_argument('--api', type=str, default="apis/api_v3.0.1.jsonl", help='사용자 이름')
    parser.add_argument('--s1', type=str, required=True, help='out_file')
    parser.add_argument('--s2', type=str, required=True, help='out_file')
    parser.add_argument('--o', type=str, required=True, help='out_file')
    args = parser.parse_args()

    s1_path = args.s1 # "datagen/it1_s1_spare.jsonl"
    s2_path = args.s2 # "datagen/it1_s2_idxed.jsonl"
    output_file = open(args.o, "w")
    apis = get_apis()
    gen_apis, cnt_each_api = get_generated_info()

    omitted_plan = []
    for plan in cnt_each_api:
        print(f"{plan}: {cnt_each_api[plan]}")
        if cnt_each_api[plan] < 50:
            omitted_plan.append([plan, 50 - cnt_each_api[plan]])
    
    for api in apis:
        if api not in gen_apis:
            omitted_plan.append([api, 50])

    s1_spare_datas = get_data(s1_path)
    s2_gen_datas = get_data(s2_path)

    s1_d = convert_to_dict(s1_spare_datas)
    s2_d = convert_to_dict(s2_gen_datas)
    u_dict = {}
    
    for o_plan_info in omitted_plan:
        plan = o_plan_info[0]
        req_count = o_plan_info[1]

        for i in range(req_count):
            s1_data = s1_d[plan].pop()
            r_plan = random.choice(list(s2_d.keys()))
            if len(s2_d[r_plan]) == 0:
                i -= 1
                continue
            s2_data = s2_d[r_plan].pop()
            if 'device_response' not in s2_data:
                i -= 1
                print("no_device_response")
                continue
            new_data = add_data(s1_data, s2_data, u_dict)
            output_file.write(json.dumps(new_data, ensure_ascii=False)+"\n")
            output_file.flush()

__main__()        