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
from utils.frequently_used_tools import get_model_name, get_arg_parse
from prompts import USER_SEED_PROMPT
import pandas as pd

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
    output_path    = args.o # "file1.jsonl"
    tsv_path = args.t1 #"raw/it5_complex_1_tc_filtered.tsv"
    model_name, generate_response = get_model_name(args.model)
    filters = [JsonExtractor()]

    # python3 user_generator.py --model o4-mini --o raw/it5.jsonl --t1 raw/it5_complex_1_tc_filtered.tsv 
    # 3. prompts 생성
    prompt_template = USER_SEED_PROMPT
    prompts: List[str] = []

    COLUMNS = ["conversation_history", "query", "rewrited_query"]
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    for record in df[COLUMNS].to_dict(orient="records"):                
        dataset_str = json.dumps(record, indent=2, ensure_ascii=False)
        prompt_text = prompt_template.format(
            tool=dataset_str
        )
        prompts.append(prompt_text)
        #break  # 우선 한 셋만
    
    response_datasets: List[Dict] = []
    with open(output_path, "w", encoding="utf-8") as output_file:
        # tqdm 으로 진행 상황 표시 (총 프롬프트 개수)
        for idx, prompt in enumerate(tqdm(prompts, desc="Processing", unit="prompt")):
            try:
                # 단일 스레드 순차 실행
                results = process_prompt(prompt, generate_response, filters)                                
                # 결과 출력 및 파일 저장
                for res in results:
                    print(res)
                    output_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                # 누적
                response_datasets.extend(results)

            except Exception as e:
                print(f"Error processing prompt {idx}: {e}")
    #import pdb; pdb.set_trace()    
    # 4. 병렬 처리 및 결과 기록
    # response_datasets: List[Dict] = []
    # with open(output_path, "w", encoding="utf-8") as output_file:
    #     #with ThreadPoolExecutor(max_workers=5) as executor:
    #     with ThreadPoolExecutor(max_workers=1) as executor:
    #         futures = {
    #             executor.submit(process_prompt, p, generate_response, filters): idx
    #             for idx, p in enumerate(prompts)
    #         }
    #         for future in tqdm(as_completed(futures), total=len(prompts)):
    #             idx = futures[future]
    #             try:
    #                 results = future.result()
    #                 for j, res in enumerate(results):
    #                     print(res)                                                
    #                     output_file.write(json.dumps(res, ensure_ascii=False) + "\n")
    #                 if results:
    #                     #print(results[-1]['answer']["plan"])
    #                     results[-1]['answer']["plan"]
    #                 response_datasets.extend(results)
    #             except Exception as e:
    #                 print(f"Error processing prompt {idx}: {e}")
    #             #print(f"--- Prompt {idx+1} / {len(prompts)}, len(datasets): {len(response_datasets)} ---")

if __name__ == "__main__":
    main()