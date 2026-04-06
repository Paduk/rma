import json
import re
import os
import requests
import ast
from datasets import load_dataset
from utils.frequently_used_tools import get_arg_parse
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
def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:12434'): # qwen3
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,
                "format": "json",
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

    # python3 ollama_inference.py --t history-phi4 --o datasets/manual/history-sft-phi4.tsv --test_key manual && python3 ollama_inference.py --t rewrite-phi4 --o datasets/manual/rewrite-sft-phi4.tsv --test_key manual
    # python3 ollama_inference.py --t rewrite-phi4 --o datasets/manual/rma-sft-phi4.tsv --test_key manual_rewrited

    # python3 ollama_inference.py --t history-llama --o datasets/manual/history-sft-llama.tsv --test_key manual && python3 ollama_inference.py --t rewrite-llama --o datasets/manual/rewrite-sft-llama.tsv --test_key manual
    # python3 ollama_inference.py --t rewrite-llama --o datasets/manual/rma-sft-llama.tsv --test_key manual_rewrited

    # python3 ollama_inference.py --t history-base-llama --o datasets/lawbase/llama-complex-history.tsv --test_key complex && python3 ollama_inference.py --t rewrite-base-llama --o datasets/lawbase/llama-complex-rewrite.tsv --test_key complex

    # python3 ollama_inference.py --t history-qwen25 --o datasets/manual/ad/history-qwen25.tsv --test_key advanced_manual && python3 ollama_inference.py --t rewrite-qwen25 --o datasets/manual/ad/rewrite-qwen25.tsv --test_key advanced_manual
    
    # python3 ollama_inference.py --t rewrite-qwen25 --o datasets/manual/ad/rma-qwen25.tsv --test_key rewrited_advanced_manual

    # python3 ollama_inference.py --t history-qwen25-base --o datasets/manual/ad/history-qwen25-base.tsv --test_key advanced_manual && python3 ollama_inference.py --t rewrite-qwen25-base --o datasets/manual/ad/rewrite-qwen25-base.tsv --test_key advanced_manual

    test_type = args.t
    if test_type == 'history-phi4':
        model_name = model_names[0]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-phi4':
        model_name = model_names[1]
        prompt_template = model_prompts[model_name]        
    elif test_type == 'history-llama':
        model_name = model_names[2]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-llama':
        model_name = model_names[3]
        prompt_template = model_prompts[model_name]            
    elif test_type == 'history-qwen3':
        model_name = model_names[4]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-qwen3':
        model_name = model_names[5]
        prompt_template = model_prompts[model_name]            
    elif test_type == 'history-qwen25':
        model_name = model_names[6]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-qwen25':
        model_name = model_names[7]
        prompt_template = model_prompts[model_name]
    elif test_type == 'history-qwen25-base':
        model_name = model_names[8]
        prompt_template = model_prompts['qwen25-history-base:latest']
    elif test_type == 'rewrite-qwen25-base':
        model_name = model_names[8]
        prompt_template = model_prompts['qwen25-rewrite-base:latest']
    elif test_type == 'history-phi4-base':
        model_name = model_names[9]
        prompt_template = model_prompts['phi4-history-base:latest']
    elif test_type == 'rewrite-phi4-base':
        model_name = model_names[9]
        prompt_template = model_prompts['phi4-rewrite-base:latest']
    elif test_type == 'history-llama-base':
        model_name = model_names[10]
        prompt_template = model_prompts['llama-history-base:latest']
    elif test_type == 'rewrite-llama-base':
        model_name = model_names[10]
        prompt_template = model_prompts['llama-rewrite-base:latest']    
    elif test_type == 'history-qwen3-base':
        model_name = model_names[11]
        prompt_template = model_prompts['qwen3-history-base:latest']
    elif test_type == 'rewrite-qwen3-base':
        model_name = model_names[11]
        prompt_template = model_prompts['qwen3-rewrite-base:latest']
    else:        
        print("Invalid test type. Please use 'history' or 'rewrite'.")
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
        'complex': [                        
            'datasets/tc/it3_complex_1_tc.tsv',
            'datasets/tc/it4_complex_1_tc.tsv',
            'datasets/tc/it4_complex_2_tc.tsv',
            'datasets/tc/it5_complex_1_tc.tsv',
            'datasets/tc/it5_complex_2_tc.tsv',
            'datasets/tc/it5_complex_3_tc.tsv',                    
        ],  
        # 'manual': [                     
        #     #'datasets/tc/manual/test.tsv',            
        #     'datasets/tc/manual/turn2.tsv',
        #     'datasets/tc/manual/turn3.tsv',            
        #     'datasets/tc/manual/turn4.tsv',
        #     'datasets/tc/manual/turn5.tsv',
        # ],
        # 'advanced_manual': [                                            
        #     'datasets/tc/manual/ad/turn2.tsv',
        #     'datasets/tc/manual/ad/turn3.tsv',            
        #     'datasets/tc/manual/ad/turn4.tsv',
        #     'datasets/tc/manual/ad/turn5.tsv',
        # ],
        # 'rewrited_advanced_manual': [                                            
        #     'datasets/tc/manual/ad/qwen3_rewrited/turn2.tsv',
        #     'datasets/tc/manual/ad/qwen3_rewrited/turn3.tsv',            
        #     'datasets/tc/manual/ad/qwen3_rewrited/turn4.tsv',
        #     'datasets/tc/manual/ad/qwen3_rewrited/turn5.tsv',
        # ],
        'manual_rewrited': [                     
            #'datasets/tc/manual/test.tsv',            
            'datasets/tc/manual/llama_rewrited/turn2.tsv',
            'datasets/tc/manual/llama_rewrited/turn3.tsv',            
            'datasets/tc/manual/llama_rewrited/turn4.tsv',
            'datasets/tc/manual/llama_rewrited/turn5.tsv',
        ],
        # 'rma_base': [            
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it2_NR_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it2_nonNR_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it3_nonNR_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it4_nonNR_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it5_nonNR_tc.tsv',            
        # ],
        # 'rma_complex': [                        
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it3_complex_1_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it4_complex_1_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it4_complex_2_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it5_complex_1_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it5_complex_2_tc.tsv',
        #     'datasets/tc/Qwen3-4b-integrated-half_rewrited/it5_complex_3_tc.tsv',                    
        # ],
        # 'droidcall': [
        #     'datasets/tc/droidcall_tc.tsv'
        # ]                            
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
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
        # 전처리
        #if "rewrite" in model_name or "history" in model_name:            
        proc = ds.map(
            partial(preprocess_example_it, apis=sft_apis, prompt_template=prompt_template, test_type=test_type)
        )
        # else:            
        #     proc = ds.map(
        #         partial(preprocess_example_it, apis=apis, prompt_template=prompt_template, test_type=test_type)
        #     )
        # 평가
        print(proc[0]["strprompt"])        
        #exit(0)
        file_results = []

        for ex in tqdm(proc, desc=f"Processing {os.path.basename(file_path)}"):
            prompt = ex["strprompt"]            
            
            try:
                raw = generate_text(prompt, model=model_name)                                                
                #result = ast.literal_eval(raw)                
                try:
                    result = ast.literal_eval(raw)                                                          
                except:                                          
                    result = extract_json_from_markdown(raw)                        
                
                gt = ast.literal_eval(ex["stranswer"])
                if type(gt) == str:
                    gt = ast.literal_eval(gt)

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
