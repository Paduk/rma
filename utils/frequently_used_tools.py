import json
import argparse
import random
import pdb
import argparse
import os
from openai import OpenAI
from utils import OpenAiGenerateResponse, GoogleGenerateResponse

def get_model_name(arg_model):
    if arg_model == 'o3':
        model_name = 'o3-mini'
    elif arg_model == 'gpt-5.1':
        model_name = 'gpt-5.1'
    elif arg_model == 'gpt-5-mini':
        model_name = 'gpt-5-mini'
    elif arg_model == 'gpt-5-nano':
        model_name = 'gpt-5-nano'
    elif arg_model == 'o4-mini':
        model_name = 'o4-mini'
    elif arg_model == 'gpt-4.1-2025-04-14':
        model_name = 'gpt-4.1-2025-04-14'
    elif arg_model == 'gpt-4o-mini-2024-07-18':
        model_name = 'gpt-4o-mini-2024-07-18'
    elif arg_model == 'gemini-2.5-flash':
        model_name = 'gemini-2.5-flash-preview-04-17'
    elif arg_model == 'gemini-2.0-flash-lite':
        model_name = 'gemini-2.0-flash-lite'
    elif arg_model == 'gemini-2.0-flash':
        model_name = 'gemini-2.0-flash'
    else:
        exit(0)

    if 'gemini' in model_name:
        generate_response = GoogleGenerateResponse(model_name=model_name)    
    elif model_name in ['o3-mini', 'o4-mini', 'gpt-4.1-2025-04-14', 'gpt-4o-mini-2024-07-18', 'gpt-5.1', 'gpt-5-mini', 'gpt-5-nano']:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None), base_url="https://api.openai.com/v1",)
        generate_response = OpenAiGenerateResponse(client=client, model=model_name, system_prompt="")

    return model_name, generate_response

def print_filter_status(message, keys="default", model_name="default"):
    with open(f"logs/{model_name}_filtering_status.txt", "a") as f:
        f.write(f"[{keys}] : {message}\n")

def print_data_cnt_per_plan(datas, keys="default", model_name="default"):
    plan_cnt = {}
    
    unique_index = set()
    for data in datas:   
        if "answer" in data:           
            plan = data["answer"]["plan"]                        
        else:
            plan = data["next_turn_plan"]
        plan_cnt[plan] = plan_cnt.get(plan, 0) + 1                
    
    print(f"title: {keys}")
    for plan in plan_cnt:
        print(f"{plan}: {plan_cnt[plan]}")

    print(f"plan_count: {len(plan_cnt)}")
    print(f"total_lens: {len(datas)}")
    with open(f"logs/{model_name}_gen_logs.txt", "a") as f:
        f.write(f"[{keys}] plan_count: {len(plan_cnt)}, total_len: {len(datas)}\n")

def get_arg_parse():
    parser = argparse.ArgumentParser(description="data integration")    
    parser.add_argument('--t', type=str, required=False, help='target_file')    
    parser.add_argument('--o', type=str, required=False, help='out_file')    
    parser.add_argument('--t1', type=str, required=False, help='target_file')
    parser.add_argument('--t2', type=str, required=False, help='target_file')
    parser.add_argument('--step', type=str, required=False, help='out_file')    
    parser.add_argument('--it', type=str, required=False, help='iteration_file')    
    parser.add_argument('--model', type=str, required=False, help='iteration_file')    
    parser.add_argument('--t_list', type=str, nargs='+', required=False, help='List of iteration files')
    parser.add_argument('--d', action='store_true', help='Enable debug mode')
    parser.add_argument('--s', type=str, required=False, help='start_file')
    parser.add_argument('--api', type=str, default="apis/api_v3.0.1.jsonl", help='사용자 이름')
    parser.add_argument('--test_key', type=str, default="", help='')
    parser.add_argument('--host', type=str, default="http://localhost:11436", help='Ollama host URL')
    args = parser.parse_args()

    return args

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_len_apis(datas):
    len_plans = set()
    for data in datas:
        u_idx = data["unique_idx"]
        plan = u_idx.split('-')[-2]
        len_plans.add(plan)

    print(len(len_plans))

def read_apis(api_file):
    api_dict = {}
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                api_data = json.loads(line)
                api_data.pop("examples", None)
                api_data.pop("returns", None)
                api_data.pop("next_turn_plans", None)
                api_dict[api_data["plan"]] = api_data
    
    return api_dict

def read_simple_apis(api_file):
    with open(api_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)
    return api_data        

def save_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
