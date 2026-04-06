import json
from utils import JsonExtractor, SimilarityFilter, DataFilter
from utils import OpenAiGenerateResponse, GoogleGenerateResponse
from prompts import REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT
from openai import OpenAI
from utils.frequently_used_tools import get_model_name
import os
import argparse
import csv
import ast
parser = argparse.ArgumentParser(description="data integration")
parser.add_argument('--api', type=str, default="apis/api_v3.0.1.jsonl", help='사용자 이름')
#parser.add_argument('--s', type=str, required=True, help='start_file')
#parser.add_argument('--o', type=str, required=True, help='out_file')
parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='out_file')
args = parser.parse_args()

step3_file = "datasets/gem_it5_nonNR_tc_gen_simpler.tsv" # args.s # "datagen/it1_s3_o3_idx.jsonl"
api_file = args.api # "apis/api_v3.0.1.jsonl"
examples_file = 'data_generated_model/datagen/it1_s1.jsonl'
OUTPUT_FILE = "datasets/gem_it5_nonNR_tc_new.tsv" # args.o # "datagen/it1_s4_o3.jsonl"

def chunks(lst, n):
    """리스트 lst를 n개씩 나눠 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    

def read_tsv_as_json(path):
    """
    TSV 파일을 읽어, 각 행을 JSON-like dict로 파싱하여 리스트로 반환합니다.
    - path: 읽을 TSV 파일 경로
    - 반환값: [
          {
            "conversation_history": [...],
            "query": "...",
            "rewrited_query": "...",
            "answer": {...},
            "unique_idx": "...",
            "candidates": [...]
          },
          ...
        ]
    """
    datas = []
    # utf-8-sig: BOM 제거용
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            item = {
                # Python repr 문자열을 파싱
                "conversation_history": ast.literal_eval(row["conversation_history"]),
                "query":               row["query"],
                "rewrited_query":      row["gen_rewrite_query"],
                "answer":              ast.literal_eval(row["answer"]),
                "unique_idx":          row["unique_idx"],
                "candidates":          ast.literal_eval(row["candidates"])
            }
            datas.append(item)
    return datas

prompt_template = REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT
multiturn_responses = read_tsv_as_json(step3_file)
# multiturn_responses = []
# with open(step3_file, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if line:            
#             multiturn_responses.append(json.loads(line))

examples_list = []
with open(examples_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            examples_list.append(json.loads(line))

# 2. api.jsonl 파일의 각 항목을 key가 name인 딕셔너리에 저장합니다.
api_dict = {}
with open(api_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entry = json.loads(line)
            api_dict[entry["plan"]] = entry

# 3. device_response.jsonl의 각 항목을 순회합니다.
def filter_data_list_by_name(data_list, target_name):
    """
    JSON 객체들이 담긴 리스트에서 각 객체의 answers[0].name이 target_name과 일치하는 항목을 5개 찾아서,
    query와 answer (첫 번째 answer 객체)만 포함하는 리스트를 반환합니다.
    
    :param data_list: JSON 객체들이 담긴 리스트
    :param target_name: 비교할 answer의 name 값
    :return: query와 answer 키만 포함된 dict들의 리스트
    """
    results = []
    for record in data_list:
        # answers 리스트가 존재하며 최소 1개 이상의 요소가 있는지 확인
        answer = record.get("answer", [])
        if answer and answer.get("plan") == target_name:
            results.append({
                "query": record.get("query", ""),
                "answer": answer  # 첫번째 answer 전체를 포함
            })
            if len(results) == 5:
                break
    return results


prompts = []
for api_key in api_dict:
    api_item = api_dict[api_key]
    func_name = api_item.get("plan")        
    api_item.pop("next_turn_plans", None)
    
    filtered_datas = []    
    for response in multiturn_responses:    
        #next_turn_func_name = response.get("next_turn_plan")            
        next_turn_func_name = response.get("answer", {}).get("plan")
        if next_turn_func_name == func_name:            
            if "rewrited_query" not in response:
                import pdb
                pdb.set_trace()
            filtered_datas.append({
                "query": response["rewrited_query"]                    
            })
            
    if not filtered_datas:
        continue

    for group in chunks(filtered_datas, 10):         
        copied_api_item = api_item.copy()
        copied_api_item.pop("returns")
        copied_api_item.pop("examples")
        api_data = json.dumps(copied_api_item, indent=2, ensure_ascii=False)
        examples = filter_data_list_by_name(examples_list, func_name)
        examples = [json.dumps(ex, indent=2, ensure_ascii=False) for ex in examples]
        examples_ = "\n".join(examples)
        dataset_str = json.dumps(group, indent=2, ensure_ascii=False)
        prompt_text = prompt_template.format(            
            tool = api_data,
            data = dataset_str,
            examples = examples_
        )        
        prompts.append(prompt_text)

        #break # 우선 하나만        

debug = True
if debug:
    print(len(prompts))
    print(prompts[0])

output_file = open(OUTPUT_FILE, "w")
filters = [JsonExtractor()]

model_name, generate_response = get_model_name(args.model)
    
response_datasets = []
for i, prompt in enumerate(prompts):        
    try:        
        response = generate_response("", [prompt])                    
        for filter_idx, filter in enumerate(filters):
            responses = filter.filter(response)
            
        for idx, res in enumerate(responses):
            print(res)
            response_datasets.append(res)
            output_file.write(json.dumps(res, ensure_ascii=False)+"\n")
            output_file.flush()    
    
        print(f"--- Prompt {i+1} / {len(prompts)}, len(datasets): {len(response_datasets)} ---")    
    except Exception as e:
        print(i, 'error:', e)
    #print("\n")
output_file.close()