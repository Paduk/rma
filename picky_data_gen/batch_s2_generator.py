from openai import OpenAI
import pdb
from typing import List, Dict
import os
import json
import argparse
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import OpenAiGenerateResponse, GoogleGenerateResponse
from utils import JsonExtractor, SimilarityFilter, DataFilter
from utils.frequently_used_tools import (
    get_arg_parse,
    get_effective_target_count,
    get_model_name,
    get_predecessor_plan_counts,
    get_target_plan_set,
    is_targeted_run,
)
from prompts import RESPONSE_GENERATION_PROMPT

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def chunks(lst: List, n: int):
    """리스트 lst를 n개씩 나눠 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_prompt(prompt: str,
                   generate_response: Callable[[str, List[str]], str],
                   filters: List[JsonExtractor]) -> List[Dict]:
    resp = generate_response("", [prompt])
    out: List[Dict] = []
    for flt in filters:
        out.extend(flt.filter(resp))
    return out

def main():
    # 1. 파싱 및 초기화
    args = get_arg_parse()
    api_file       = args.api
    generated_path = args.s
    output_path    = args.o
    model_name, generate_response = get_model_name(args.model)
    filters = [JsonExtractor()]
    target_plans = get_target_plan_set(args)
    targeted_run = is_targeted_run(args)
    predecessor_plan_counts = {}
    source_target_plans = set()

    if targeted_run:
        predecessor_plan_counts = get_predecessor_plan_counts(api_file, args)
        source_target_plans = set(predecessor_plan_counts.keys())

    # 2. 데이터 로드
    simple_api_data = read_jsonl(api_file)
    generated       = read_jsonl(generated_path)

    # 3. prompts 생성
    prompt_template = RESPONSE_GENERATION_PROMPT
    prompts: List[str] = []

    for simple_item in simple_api_data:
        simple_name = simple_item.get("plan")
        if targeted_run and simple_name not in source_target_plans:
            continue
        simple_item.pop("next_turn_plans", None)

        filtered_datas = []
        for gen in generated:
            answer = gen.get('answer', {})
            if 'conversation_history' in gen:
                conv_history = [f"{turn}" for turn in gen['conversation_history']]
                query_text = gen.get('rewrited_query', '')
            else:
                conv_history = []
                query_text = gen.get('query', '')

            if answer.get("plan") == simple_name:
                filtered_datas.append({
                    'conversation_history': conv_history,
                    'query': query_text,
                    'answer': answer
                })

        if targeted_run:
            target_count = predecessor_plan_counts.get(simple_name)
            if target_count is not None:
                filtered_datas = filtered_datas[:target_count]

        if not filtered_datas:
            continue

        for group in chunks(filtered_datas, 10):
            dataset_str = json.dumps(group, indent=2, ensure_ascii=False)
            prompt_text = prompt_template.format(
                tool=simple_item,
                dataset=dataset_str
            )
            prompts.append(prompt_text)
            #break  # 우선 한 셋만

    if targeted_run:
        print(f"final_target_plans: {sorted(target_plans)}")
        print(f"source_target_plans: {sorted(source_target_plans)}")
        print(f"source_plan_caps: {predecessor_plan_counts}")
    print(f"prompt_count: {len(prompts)}")

    if args.dry_run:
        print("dry-run enabled; skipping generation")
        return

    # 4. 병렬 처리 및 결과 기록
    response_datasets: List[Dict] = []
    with open(output_path, "w", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(process_prompt, p, generate_response, filters): idx
                for idx, p in enumerate(prompts)
            }
            for future in tqdm(as_completed(futures), total=len(prompts)):
                idx = futures[future]
                try:
                    results = future.result()
                    for j, res in enumerate(results):
                        if res.get('conversation_history'):
                            res['conversation_history'][0]
                        res.get('query')
                        res.get('device_response')
                        #print(res.get('query'), '->', res.get('device_response'), j)
                        #print('-' * 50)
                        output_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                    if results:
                        #print(results[-1]['answer']["plan"])
                        results[-1]['answer']["plan"]
                    response_datasets.extend(results)
                except Exception as e:
                    print(f"Error processing prompt {idx}: {e}")
                print(f"--- Prompt {idx+1} / {len(prompts)}, len(datasets): {len(response_datasets)} ---")

if __name__ == "__main__":
    main()
