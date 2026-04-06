import json
import re
import os
import sys
import argparse
import requests
import ast
import csv
import pdb
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from functools import partial
from train.llama_prompts import ZERO_REWRITE_INFERENCE_LLAMA, ZERO_HISTORY_INFERENCE_LLAMA, SFT_REWRITE_INFERENCE_LLAMA, SFT_HISTORY_INFERENCE_LLAMA
from train.gemma_prompts import SFT_REWRITE_INFERENCE_GEMMA, SFT_HISTORY_INFERENCE_GEMMA
from train.llama_prompts import SFT_REWRITE_INFERENCE_PHI4, SFT_HISTORY_INFERENCE_PHI4, ZERO_REWRITE_INFERENCE_PHI4, ZERO_HISTORY_INFERENCE_PHI4
from train.llama_prompts import SFT_REWRITE_INFERENCE_QWEN25, SFT_HISTORY_INFERENCE_QWEN25, ZERO_REWRITE_INFERENCE_QWEN25, ZERO_HISTORY_INFERENCE_QWEN25
from train.llama_prompts import SFT_REWRITE_INFERENCE_QWEN3, SFT_HISTORY_INFERENCE_QWEN3, ZERO_REWRITE_INFERENCE_QWEN3, ZERO_HISTORY_INFERENCE_QWEN3
# /mnt/data/.cache/hj153lee/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/5a149550068a1eb93398160d8953f5f56c3603e9/
# qwen3 전용으로 ollama 요청하게끔 코드 변경 됨 26.3.7
OLLAMA_HOST = "http://127.0.0.1:11435"
OLLAMA_MODEL_NAME = "qwen3-1.7b-rma-q4km"
OLLAMA_MODEL_NAME = "qwen3-4b-rma-q4km"


def get_arg_parse():
    parser = argparse.ArgumentParser(description="ollama inference for qwen3 rma")
    parser.add_argument('--t', type=str, required=False, help='test type')
    parser.add_argument('--o', type=str, required=False, help='output file')
    parser.add_argument('--test_key', type=str, default="", help='dataset split key')
    parser.add_argument('--d', action='store_true', help='debug mode')
    return parser.parse_args()


def load_tsv_records(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

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
        # print()
        # print(text)
        return None

# Ollama API 호출 함수
#def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11434'):
def generate_text(prompt, model=OLLAMA_MODEL_NAME, host=OLLAMA_HOST):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 512               
            },
            "stream": False
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        raise Exception(f"API request failed: {response.text}")

def print_eval(df, title=None, test_type=None, detail=False):
    metrics = ("plan", "arguments", "all")
    # -- 헤더 출력 --
    if title:
        print(f"\n## Performance for {title}\n")

    # -- 개별 지표 Accuracy --
    accs = []
    for col in metrics:
        acc = df[col].eq("pass").mean()  # 0~1 사이의 값
        accs.append(acc)
        # 소수점 둘째 자리까지 고정 출력
        print(f"{col.title():<10} Accuracy : {acc:.2f}")

    # -- Macro Average (세 지표의 단순 평균) --
    macro_acc = sum(accs) / len(accs)
    print(f"{'Macro':<10} Accuracy : {macro_acc:.2f}")
    print("-" * 40)

    # -- 로그파일에도 동일 내용 쓰기 --
    with open("logs/ollama_inference_log.txt", 'a', encoding='utf-8') as f:
        if title:
            f.write(f"\n## Performance for {title}, {test_type}\n")
        for col, acc in zip(metrics, accs):
            f.write(f"{col.title():<10} Accuracy : {acc:.2f}\n")
        f.write(f"{'Macro':<10} Accuracy : {macro_acc:.2f}\n")
        f.write("-" * 40 + "\n")

    # -- Plan별(detail) 성능과 Plan별 Macro --
    if detail:
        # gt 컬럼에서 plan 이름만 추출
        df["gt_plan"] = df["gt"].apply(lambda x: x.get("plan"))
        detail_df = (
            df.groupby("gt_plan")[list(metrics)]
              .apply(lambda sub: sub.eq("pass").mean().round(2))
              .reset_index()
        )
        print(detail_df.to_string(index=False))

        # Plan별 Macro (각 plan에 대해 plan/arguments/all의 평균)
        macro_by_plan = detail_df[metrics].mean(axis=1)
        detail_df["macro_by_plan"] = macro_by_plan.round(2)
        print("\n# Plan별 Macro Accuracy")
        print(detail_df.to_string(index=False))

def main(out_file):
    # API 데이터 파일 경로 (환경에 맞게 수정)    
    apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)    
    sft_apis = read_apis("apis/simple_api.json", simple=True)
    
    #model_names = ['llama3-history-1st:latest', 'llama3-rewrite-1st:latest',
    model_names = ['phi4-history:latest', 'phi4-rewrite:latest', 'llama-history:latest', 'llama-rewrite:latest', 'qwen3-history:latest', 'qwen3-rewrite:latest', 'qwen25-history:latest', 'qwen25-rewrite:latest', 'qwen25-base:latest', 'phi4-base:latest', 'llama-base:latest', 'qwen3-base:latest']

    model_prompts = {        
        'phi4-history:latest': SFT_HISTORY_INFERENCE_PHI4,
        'phi4-rewrite:latest': SFT_REWRITE_INFERENCE_PHI4,   
        'llama-history:latest': SFT_HISTORY_INFERENCE_LLAMA,
        'llama-rewrite:latest': SFT_REWRITE_INFERENCE_LLAMA,        
        'qwen3-history:latest': SFT_HISTORY_INFERENCE_QWEN3,
        'qwen3-rewrite:latest':SFT_REWRITE_INFERENCE_QWEN3,
        'qwen25-history:latest': SFT_HISTORY_INFERENCE_QWEN25,
        'qwen25-rewrite:latest':SFT_REWRITE_INFERENCE_QWEN25,
        'qwen25-history-base:latest':ZERO_HISTORY_INFERENCE_QWEN25,
        'qwen25-rewrite-base:latest':ZERO_REWRITE_INFERENCE_QWEN25,
        'phi4-history-base:latest':ZERO_HISTORY_INFERENCE_PHI4,
        'phi4-rewrite-base:latest':ZERO_REWRITE_INFERENCE_PHI4,
        'llama-history-base:latest':ZERO_HISTORY_INFERENCE_LLAMA,
        'llama-rewrite-base:latest':ZERO_REWRITE_INFERENCE_LLAMA,
        'qwen3-history-base:latest':ZERO_HISTORY_INFERENCE_QWEN3,
        'qwen3-rewrite-base:latest':ZERO_REWRITE_INFERENCE_QWEN3
    }    

    test_type = args.t
    model_name = OLLAMA_MODEL_NAME
    if test_type == 'history-qwen3':
        prompt_template = model_prompts['qwen3-history:latest']
    elif test_type == 'rewrite-qwen3':
        prompt_template = model_prompts['qwen3-rewrite:latest']
    else:
        print("Invalid test type. Please use 'history-qwen3' or 'rewrite-qwen3'.")
        exit(0)

    
    test_key = args.test_key #'base' # 'rma_complex        
    data_files = {                        
        'base': [         
            #'datasets/tc/it2_NR_tc.tsv',
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
        'advanced_manual': [                                            
            'datasets/tc/manual/ad/turn2.tsv',
            'datasets/tc/manual/ad/turn3.tsv',            
            'datasets/tc/manual/ad/turn4.tsv',
            'datasets/tc/manual/ad/turn5.tsv',
        ],
        'rewrited_advanced_manual': [                                            
            'datasets/tc/manual/ad/qwen3_rewrited/turn2.tsv',
            'datasets/tc/manual/ad/qwen3_rewrited/turn3.tsv',            
            'datasets/tc/manual/ad/qwen3_rewrited/turn4.tsv',
            'datasets/tc/manual/ad/qwen3_rewrited/turn5.tsv',
        ],
        'manual_rewrited': [                     
            #'datasets/tc/manual/test.tsv',            
            'datasets/tc/manual/llama_rewrited/turn2.tsv',
            'datasets/tc/manual/llama_rewrited/turn3.tsv',            
            'datasets/tc/manual/llama_rewrited/turn4.tsv',
            'datasets/tc/manual/llama_rewrited/turn5.tsv',
        ],      
        'complex': [                        
            'datasets/tc/it3_complex_1_tc.tsv',
            'datasets/tc/it4_complex_1_tc.tsv',
            'datasets/tc/it4_complex_2_tc.tsv',
            'datasets/tc/it5_complex_1_tc.tsv',
            'datasets/tc/it5_complex_2_tc.tsv',
            'datasets/tc/it5_complex_3_tc.tsv',                               
        ]
    }        
    
    print(model_name)
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, apis, prompt_template, test_type):
        api_str = ""
        #re_fmt  = {"plan": "str type tool", "arguments": {"key1": "value1"}}        
        for plan in ast.literal_eval(example["candidates"]):
            api_data = apis[plan].copy()
            api_str += f"{plan}: {api_data}\n"
        
        if "history" in test_type:
            prompt = prompt_template.format(
                tools=api_str,
                #re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),
                conversation_history=example["conversation_history"],
                data=example["query"]
            )
        elif "rewrite" in test_type:
            prompt = prompt_template.format(
                tools=api_str,
                #re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),                
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
    print(data_files[test_key])       
    for file_path in data_files[test_key]:
        ds = load_tsv_records(file_path)
        proc = [
            preprocess_example_it(
                ex, apis=sft_apis, prompt_template=prompt_template, test_type=test_type
            )
            for ex in ds
        ]
        # 평가
        print(proc[0]["strprompt"])        
        #exit(0)
        file_results = []
        debug_mode = args.d

        for idx, ex in enumerate(tqdm(proc, desc=f"Processing {os.path.basename(file_path)}")):
            prompt = ex["strprompt"]            
            raw = ""
            gt = ast.literal_eval(ex["stranswer"])
            if type(gt) == str:
                gt = ast.literal_eval(gt)

            if debug_mode:
                print(f"\n[DEBUG] file: {os.path.basename(file_path)}")
                print("[DEBUG] input prompt:")
                print(prompt)
            
            try:
                raw = generate_text(prompt, model=model_name)                                                
                if debug_mode:
                    print("[DEBUG] raw generation:")
                    print(raw)
                #result = ast.literal_eval(raw)                
                try:
                    result = ast.literal_eval(raw)                                                          
                except:                                          
                    result = extract_json_from_markdown(raw)                        

                plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                arg_res  = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                all_res  = "pass" if plan_res=="pass" and arg_res=="pass" else "fail"                
            except Exception as e:
                result   = {"error": str(e)}                
                print(f"Error: {e}, {raw}")
                plan_res = "fail"
                arg_res  = "fail"
                all_res  = "fail"

            file_results.append({
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
            })

            if debug_mode and idx == 0:
                print(f"[DEBUG] processed one sample for {os.path.basename(file_path)}; moving to next file.")
                break

        df_file = pd.DataFrame(file_results)
        print_eval(df_file, title=os.path.basename(file_path), test_type=model_name)
        all_results.extend(file_results)

    result = pd.DataFrame(all_results)
    print_eval(result)
    result.to_csv(out_file, sep='\t', index=False, encoding='utf-8-sig')        
    
# python3 ollama_inference.py --o datasets/result/base_rewrite_gt.tsv --t rewrite
if __name__ == "__main__":
    args = get_arg_parse()
    main(args.o)
