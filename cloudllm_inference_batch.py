import json
import re
import os
import requests
import ast
from datasets import load_dataset
from utils.frequently_used_tools import get_arg_parse, get_model_name
import pdb
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from functools import partial
from prompts import ZERO_SHOT_INFERENCE_GENERATION_PROMPT, ZERO_SHOT_HISTORY_INFERENCE_GENERATION_PROMPT

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_example(ex, generate_response, extract_fn, gt_parser, file_path):
    """
    한 예제를 처리하고 결과 dict를 반환하는 함수
    - generate_response: LLM 호출 함수
    - extract_fn: extract_json_from_markdown 또는 parse_response_json
    - gt_parser: 예측값과 비교할 gt(dict)로 변환하는 함수
    """
    try:
        # 1. 응답 생성
        response = generate_response("", [ex["strprompt"]])[0]
        raw = response.get("text", "")
        
        if not raw:
            raise ValueError("Empty response text (possibly due to finish_reason = 4)")

        # 2. 응답에서 JSON 추출
        try:
            result = extract_fn(raw)
        except Exception:
            result = parse_response_json(raw)

        # 3. GT 파싱 및 비교
        gt = gt_parser(ex["stranswer"])
        if isinstance(gt, str):
            gt = ast.literal_eval(gt)

        plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
        arg_res  = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
        all_res  = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"

    except Exception as e:
        print(f"[Error] {ex.get('file')}: {e}")
        result, plan_res, arg_res, all_res = {"error": str(e)}, "fail", "fail", "fail"
        gt = {}
    
    return {
        "conversation_history": ex.get("conversation_history"),
        "query":                ex.get("query"),
        "rewrited_query":       ex.get("rewrited_query"),
        "candidates":           ex.get("candidates"),
        "generation":           result,
        "gt":                   gt,
        "plan":                 plan_res,
        "arguments":            arg_res,
        "all":                  all_res,
        "file":                 os.path.basename(file_path)
    }

# api 파일을 한 줄씩 읽어 각 plan에 해당하는 데이터를 사전으로 저장
def read_apis(api_file, simple=False):
    """
    - simple=False: JSONL 라인 단위 → dict(plan → api_data)
    - simple=True : 전체 JSON → dict
    """
    with open(api_file, encoding="utf-8") as f:
        if simple:
            return json.load(f)
        else:
            out = {}
            for line in f:
                data = json.loads(line)
                for k in ("examples","returns","next_turn_plans"):
                    data.pop(k, None)
                out[data["plan"]] = data
            return out

def parse_response_json(text):
    text = re.sub(r".*?```(?:json)?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Valid JSON not found")
    return json.loads(match.group())


def print_eval(df, title=None, test_type=None, detail=False):
    if title: print(f"\n## Performance for {title}\n")
    for col in ("plan","arguments","all"):
        acc = round((df[col]=="pass").mean(), 2)
        print(f"{col.title():<10} Accuracy : {acc}")
    print("-"*40)

    if title is None:
        title = ""
    with open("logs/cloud_inference_log.txt", 'a', encoding='utf-8') as f:
        if title:
            f.write(f"\n## Performance for {title}, {test_type}\n")
        for col in ("plan", "arguments", "all"):
            acc = round((df[col] == "pass").mean(), 2)
            f.write(f"{col.title():<10} Accuracy : {acc}\n")
        f.write("-" * 40 + "\n")

    if detail:
        df["gt_plan"] = df["gt"].apply(lambda x: x.get("plan"))
        detail_df = (
            df.groupby("gt_plan")[["plan","arguments","all"]]
              .apply(lambda x: (x=="pass").mean().round(2))
              .reset_index()
        )
        print(detail_df.to_string(index=False))

def extract_json_from_markdown(text):
    """
    마크다운 블록(```json ... ```) 또는 그냥 {...} 블록에서 JSON만 추출하고 dict로 반환
    """
    try:
        # 1. ```json ... ``` 또는 ``` ... ``` 블록 제거
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # 2. fallback: 가장 바깥의 중괄호 블록을 추출
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("JSON 블록을 찾을 수 없습니다.")

        return json.loads(json_str)

    except Exception as e:
        print("JSON 추출/파싱 실패:", e)
        return None

def main(out_file):
    # API 데이터 파일 경로 (환경에 맞게 수정)
    args = get_arg_parse()
    print(args)
    model_name, generate_response = get_model_name(args.model)    

    apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)    
    #sft_apis = read_apis("apis/simple_api.json", simple=True)
    
    test_type = args.t # 'rewrite'  # 'rewrite' or 'history'
    if test_type == 'rewrite':
        prompt_template = ZERO_SHOT_INFERENCE_GENERATION_PROMPT
    elif test_type == 'history':
        prompt_template = ZERO_SHOT_HISTORY_INFERENCE_GENERATION_PROMPT    

    tc_type = 'base' # 'complex'
    # python3 cloudllm_inference_batch.py --model gpt-5-nano --t rewrite --o datasets/rewrite-gpt5-nano-complex.tsv && python3 cloudllm_inference_batch.py --model gpt-5-nano --t rewrite --o datasets/rewrite-gpt5-nano-base.tsv
    # python3 cloudllm_inference_batch.py --model gpt-5-nano --t history --o datasets/history-gpt5-nano-complex.tsv && python3 cloudllm_inference_batch.py --model gpt-5-nano --t history --o datasets/history-gpt5-nano-base.tsv
    # python3 cloudllm_inference_batch.py --model gpt-4.1-2025-04-14 --t rewrite --o datasets/manual/rewrite-cloud-gpt41.tsv
    # python3 cloudllm_inference_batch.py --model gpt-4.1-2025-04-14 --t history --o datasets/manual/history-cloud-gpt41.tsv
    # python3 cloudllm_inference_batch.py --model o4-mini --t rewrite --o datasets/manual/rewrite-cloud-o4-mini.tsv
    # python3 cloudllm_inference_batch.py --model o4-mini --t history --o datasets/manual/history-cloud-o4-mini.tsv
    # python3 cloudllm_inference_batch.py --model gemini-2.0-flash --t rewrite --o datasets/manual/rewrite-cloud-gemini2.tsv && python3 cloudllm_inference_batch.py --model gemini-2.0-flash --t history --o datasets/manual/history-cloud-gemini2.tsv
    # python3 cloudllm_inference_batch.py --model gpt-4o-mini-2024-07-18 --t rewrite --o datasets/manual/rewrite-cloud-gpt4o-mini.tsv && python3 cloudllm_inference_batch.py --model gpt-4o-mini-2024-07-18 --t history --o datasets/manual/history-cloud-gpt4o-mini.tsv
    data_files = {                
        'base': [            
            'datasets/tc/it2_NR_tc.tsv',
            'datasets/tc/it2_nonNR_tc.tsv',
            'datasets/tc/it3_nonNR_tc.tsv',
            'datasets/tc/it4_nonNR_tc.tsv',
            'datasets/tc/it5_nonNR_tc.tsv',            
        ],  
        'manual': [                     
            #'datasets/tc/manual/test.tsv',            
            'datasets/tc/manual/turn2.tsv',
            'datasets/tc/manual/turn3.tsv',            
            'datasets/tc/manual/turn4.tsv',
            'datasets/tc/manual/turn5.tsv',
        ],         
        'complex': [                        
            'datasets/tc/it3_complex_1_tc.tsv',
            'datasets/tc/it4_complex_1_tc.tsv',
            'datasets/tc/it4_complex_2_tc.tsv',
            'datasets/tc/it5_complex_1_tc.tsv',
            'datasets/tc/it5_complex_2_tc.tsv',
            'datasets/tc/it5_complex_3_tc.tsv',                    
        ],                                 
    }        
        
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, apis, prompt_template, test_type):
        api_str = ""
        re_fmt  = {"plan": "str type tool", "arguments": {"key1": "value1"}}
        for plan in ast.literal_eval(example["candidates"]):
            api_data = apis[plan].copy()
            #api_str += f"{plan}: {api_data}\n" # short test
            api_str += json.dumps(api_data, ensure_ascii=False, indent=2) + "\n"
        
        if test_type == "history":
            prompt = prompt_template.format(
                tools=api_str,
                re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),
                conversation_history=example["conversation_history"],
                data=example["query"]
            )
        elif test_type == "rewrite":
            prompt = prompt_template.format(
                tools=api_str,
                re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),                
                #data=example["rewrite_query"]
                data=example["rewrited_query"]
            )
        return {
            "strprompt":    prompt,
            "stranswer":    json.dumps(example["answer"], ensure_ascii=False, indent=2),
            "candidates":   example["candidates"],
            "rewrited_query": example["rewrited_query"],
            "query":        example["query"],
            "conversation_history": example["conversation_history"],
        }

    all_results = []
    for file_path in data_files[tc_type]:
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
        proc = ds.map(
            partial(preprocess_example_it, apis=apis, prompt_template=prompt_template, test_type=test_type)
        )
        import pdb
        file_results = []
        max_workers = min(8, os.cpu_count() * 5)  # I/O 바운드이므로 cpu_count * 상수
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Future 객체와 예제 매핑
            futures = {
                executor.submit(
                    process_example,
                    ex,
                    generate_response,
                    extract_json_from_markdown,                    
                    lambda s: ast.literal_eval(s) if isinstance(s, str) else s,
                    file_path
                ): ex
                for ex in proc
            }
            # 완료된 순서대로 수집 (진행바)
            for future in tqdm(as_completed(futures), total=len(futures), desc=os.path.basename(file_path)):
                file_results.append(future.result())

        df_file = pd.DataFrame(file_results)        
        print_eval(df_file, title=os.path.basename(file_path), test_type=test_type)
        all_results.extend(file_results)

    result = pd.DataFrame(all_results)    
    # print_eval(result, title=os.path.basename(file_path), test_type=test_type)
    result.to_csv(out_file, sep='\t', index=False, encoding='utf-8-sig')        
    
if __name__ == "__main__":
    args = get_arg_parse()
    main(args.o)