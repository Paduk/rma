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
from train.llama_prompts import ZERO_REWRITE_INFERENCE_LLAMA, ZERO_HISTORY_INFERENCE_LLAMA, SFT_REWRITE_INFERENCE_LLAMA, SFT_HISTORY_INFERENCE_LLAMA
from train.gemma_prompts import SFT_REWRITE_INFERENCE_GEMMA, SFT_HISTORY_INFERENCE_GEMMA
from train.llama_prompts import SFT_REWRITE_INFERENCE_PHI4, SFT_HISTORY_INFERENCE_PHI4
from train.llama_prompts import SFT_REWRITE_INFERENCE_QWEN25, SFT_HISTORY_INFERENCE_QWEN25
from train.llama_prompts import SFT_REWRITE_INFERENCE_QWEN3, SFT_HISTORY_INFERENCE_QWEN3

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
        return None

# Ollama API 호출 함수
def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11434'):
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
    sft_apis = read_apis("apis/plugin_des.json", simple=True)    
    
    model_names = ['llama3-history-1st:latest', 'llama3-rewrite-1st:latest',
                 'gemma3-history-1st:latest', 'gemma3-rewrite-1st:latest',
                 'Phi-4-mini-instruct-history-1st:latest', 'Phi-4-mini-instruct-rewrite-1st:latest',
                 'Qwen2.5-3B-Instruct-history-1st:latest', 'Qwen2.5-3B-Instruct-rewrite-1st:latest',
                 'Qwen3-4B-history-1st:latest', 'Qwen3-4B-rewrite-1st:latest',
                 'Phi-4-mini-instruct-rewrite-rma-rewrite-integrated-1st:latest']    

    model_prompts = {
        'llama3-history-1st:latest': SFT_HISTORY_INFERENCE_LLAMA,
        'llama3-rewrite-1st:latest': SFT_REWRITE_INFERENCE_LLAMA,        
        'gemma3-history-1st:latest': SFT_HISTORY_INFERENCE_GEMMA,
        'gemma3-rewrite-1st:latest': SFT_REWRITE_INFERENCE_GEMMA,
        'Phi-4-mini-instruct-history-1st:latest': SFT_HISTORY_INFERENCE_PHI4,
        'Phi-4-mini-instruct-rewrite-1st:latest': SFT_REWRITE_INFERENCE_PHI4,        
        'Qwen2.5-3B-Instruct-history-1st:latest': SFT_HISTORY_INFERENCE_QWEN25,
        'Qwen2.5-3B-Instruct-rewrite-1st:latest': SFT_REWRITE_INFERENCE_QWEN25,
        'Qwen3-4B-history-1st:latest': SFT_HISTORY_INFERENCE_QWEN3,
        'Qwen3-4B-rewrite-1st:latest': SFT_REWRITE_INFERENCE_QWEN3,
        'Phi-4-mini-instruct-rewrite-rma-rewrite-integrated-1st:latest': SFT_REWRITE_INFERENCE_PHI4,
    }
    
    # python3 metatool_inference.py --t rewrite-phi4
    
    test_type = args.t
    if test_type == 'history-llama':
        model_name = model_names[0]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-llama':
        model_name = model_names[1]
        prompt_template = model_prompts[model_name]    
    elif test_type == 'history-gemma':
        model_name = model_names[2]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-gemma':
        model_name = model_names[3]
        prompt_template = model_prompts[model_name]
    elif test_type == 'history-phi4':
        model_name = model_names[4]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-phi4':
        model_name = model_names[5]
        prompt_template = model_prompts[model_name]    
    elif test_type == 'history-qwen2.5':
        model_name = model_names[6]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-qwen2.5':
        model_name = model_names[7]
        prompt_template = model_prompts[model_name]
    elif test_type == 'history-qwen3':
        model_name = model_names[8]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-qwen3':
        model_name = model_names[9]
        prompt_template = model_prompts[model_name]
    elif test_type == 'rewrite-phi4-integrated':
        model_name = model_names[10]
        prompt_template = model_prompts[model_name]
    else:
        print("Invalid test type. Please use 'history' or 'rewrite'.")
        exit(0)

    test_key = 'metatool'
    data_files = {                        
        'metatool': [         
            'metatool_testset.tsv',            
        ],                                      
    }        
    
    test_cloud = True
    if test_cloud:        
        model_name, generate_response = get_model_name(args.model)    

    print(model_name)
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, apis, prompt_template, test_type):
        api_str = ""
        #re_fmt  = {"plan": "str type tool", "arguments": {"key1": "value1"}}        
        for plan in ast.literal_eval(example["candidates"]):            
            api_data = apis[plan]
            api_str += f"{plan}: {api_data}\n"     
    
        prompt = prompt_template.format(
            tools=api_str,
            #re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),                
            data=example["Query"]
        )

        return {
            "strprompt":    prompt,
            "stranswer":    json.dumps(example["answer"], ensure_ascii=False, indent=2),
            "candidates":   example["candidates"],            
            "query":        example["Query"],            
        }
    
    all_results = [] 
    print(data_files[test_key])       
    for file_path in data_files[test_key]:
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']        
        proc = ds.map(
            partial(preprocess_example_it, apis=sft_apis, prompt_template=prompt_template, test_type=test_type)
        )
   
        # 평가
        print(proc[0]["strprompt"])                
        file_results = []

        for ex in tqdm(proc, desc=f"Processing {os.path.basename(file_path)}"):
            prompt = ex["strprompt"]            
            
            try:
                if test_cloud:
                    raw = generate_response("", [prompt])                
                    raw = raw[0]["text"]                
                
                    try:
                        result = extract_json_from_markdown(raw)                        
                    except:                        
                        result = parse_response_json(raw)
                else:
                    raw = generate_text(prompt, model=model_name)                                
                    if 'Phi-4' in model_name or 'Qwen' in model_name:
                        result = ast.literal_eval(raw)
                    else:
                        try:
                            result = extract_json_from_markdown(raw)                        
                        except:                      
                            result = ast.literal_eval(raw)                                      

                gt = ast.literal_eval(ex["stranswer"])
                if type(gt) == str:
                    gt = ast.literal_eval(gt)

                plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                arg_res  = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                all_res  = "pass" if plan_res=="pass" and arg_res=="pass" else "fail"                
                print(result, all_res)
            except Exception as e:
                result   = {"error": str(e)}                
                print(f"Error: {e}, {raw}")
                plan_res = "fail"
                arg_res  = "fail"
                all_res  = "fail"

            file_results.append({                
                "query":                ex.get("Query"),                
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
