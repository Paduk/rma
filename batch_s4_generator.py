import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import JsonExtractor, SimilarityFilter, DataFilter
from utils import OpenAiGenerateResponse, GoogleGenerateResponse
from utils.frequently_used_tools import get_model_name, get_arg_parse
from prompts import REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT

def chunks(lst, n):
    """리스트 lst를 n개씩 나눠 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def filter_data_list_by_name(data_list, target_name):
    """
    JSON 객체 리스트에서 answer.plan이 target_name인 항목을 5개 찾아,
    query와 answer만 담은 리스트로 반환
    """
    results = []
    for record in data_list:
        answer = record.get("answer", {})
        if answer.get("plan") == target_name:
            results.append({
                "query": record.get("query", ""),
                "answer": answer
            })
            if len(results) == 5:
                break
    return results

def main():
    args = get_arg_parse()
    step3_file    = args.s   # "datagen/it1_s3_o3_idx.jsonl"
    api_file      = args.api # "apis/api_v3.0.1.jsonl"
    examples_file = 'data_generated_model/datagen/it1_s1.jsonl'
    output_path   = args.o   # "datagen/it1_s4_o3.jsonl"

    # 1) 데이터 로드
    multiturn_responses = read_jsonl(step3_file)
    examples_list       = read_jsonl(examples_file)

    # 2) API 정의 불러오기
    api_dict = {}
    with open(api_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            api_dict[entry["plan"]] = entry

    # 3) prompts 생성
    prompt_template = REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT
    prompts = []
    for api_key, api_item in api_dict.items():
        func_name = api_key
        # next_turn_plans 제거
        api_item.pop("next_turn_plans", None)

        # 해당 plan에 매칭되는 응답 수집
        filtered_datas = []
        for resp in multiturn_responses:
            if resp.get("next_turn_plan") == func_name and 'rewrited_query' in resp:
                filtered_datas.append({
                    "query": resp["rewrited_query"]
                })

        if not filtered_datas:
            continue

        # 10개씩 묶어서 prompt 생성
        for group in chunks(filtered_datas, 10):
            copied = api_item.copy()
            copied.pop("returns", None)
            copied.pop("examples", None)
            tool_str = json.dumps(copied, indent=2, ensure_ascii=False)

            exs = filter_data_list_by_name(examples_list, func_name)
            exs_str = "\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in exs)

            data_str = json.dumps(group, indent=2, ensure_ascii=False)

            prompt = prompt_template.format(
                tool     = tool_str,
                data     = data_str,
                examples = exs_str
            )
            prompts.append(prompt)
            #break  # 한 셋만

    # 4) 병렬 처리 준비
    filters = [JsonExtractor()]
    model_name, generate_response = get_model_name(args.model)
    response_datasets = []
    
    def process_prompt(p: str):
        raw = generate_response("", [p])
        out = []
        for flt in filters:
            out.extend(flt.filter(raw))
        return out

    # 5) ThreadPoolExecutor로 동시 API 호출 & 파일 기록
    with open(output_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_prompt, p): idx for idx, p in enumerate(prompts)}
            for future in tqdm(as_completed(futures), total=len(prompts)):
                idx = futures[future]
                try:
                    results = future.result()
                    for res in results:
                        print(res)
                        response_datasets.append(res)
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        out_f.flush()
                except Exception as e:
                    print(f"[{idx}] error:", e)
                print(f"--- Prompt {idx+1} / {len(prompts)}, len(datasets): {len(response_datasets)} ---")

if __name__ == "__main__":
    main()
