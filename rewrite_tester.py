import json
import re
import os
import requests
import ast
from datasets import load_dataset
from utils.frequently_used_tools import get_arg_parse, read_jsonl, get_model_name
import pdb
import random
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from functools import partial
from train.llama_prompts import ZERO_REWRITE_INFERENCE_LLAMA, ZERO_HISTORY_INFERENCE_LLAMA, SFT_REWRITE_INFERENCE_LLAMA, SFT_HISTORY_INFERENCE_LLAMA
from train.gemma_prompts import ZERO_REWRITE_INFERENCE_GEMMA, SFT_REWRITE_INFERENCE_GEMMA
from prompts import FEW_SHOTS_REWRITE_PROMPT, FEW_SHOTS_SMALL_REWRITE_PROMPT

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
                "num_predict": 200,               
                "stop": ["}"]
            },
            "stream": False
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        raise Exception(f"API request failed: {response.text}")

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

def get_examples(datas):    
    random.shuffle(datas)
    #datas = datas[3:]
    print(len(datas))

    new_datas = []
    for data in datas:
        data = {
            "conversation_history": data["conversation_history"],
            "query": data["query"],
            "rewrited_query": data["rewrited_query"]
        }

        new_datas.append(data)
    
    return new_datas

def main(out_file):
    # API 데이터 파일 경로 (환경에 맞게 수정)        
    test_with_cloud = True    
    rewrited_examples = read_jsonl("examples.jsonl") # "datagen/it1_s3_gemini.jsonl"    
    rewrited_examples = get_examples(rewrited_examples)
    
    model_names = ['llama3-3b-it:latest']  # 사용할 모델 이름    
    model_name = model_names[0]    
    
    if test_with_cloud:
        model_name = "gemini-2.0-flash"
        model_name, generate_response = get_model_name(model_name)

    test_type = 'few-small'
    test_type = 'few-large'
    if 'few' in test_type:
        if test_type == 'few-small':
            prompt_template = FEW_SHOTS_SMALL_REWRITE_PROMPT
        elif test_type == 'few-large':
            prompt_template = FEW_SHOTS_REWRITE_PROMPT                

    data_files = {
        'tc': [
            'logs/rewrite.tsv',                        
        ]
    }        
    
    
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, prompt_template, test_type, rewrited_examples=None):
        api_str = ""        
        
        if test_type == "zero":            
            data = {"conversation_history": example["conversation_history"], "query": example["query"]}
            prompt = prompt_template.format(                
                data=json.dumps(data, ensure_ascii=False, indent=2),
            )
        elif "few" in test_type:
            f_ex = ""
            for data in rewrited_examples:                                
                f_ex += json.dumps(data, ensure_ascii=False, indent=2) + "\n"

            data = {"conversation_history": example["conversation_history"], "query": example["query"]}
            prompt = prompt_template.format(
                data=json.dumps(data, ensure_ascii=False, indent=2),
                examples=f_ex,
            )
        return {
            "strprompt":    prompt,
            "stranswer":    example["rewrited_query"],                        
            "query":        example["query"],
            "conversation_history": example["conversation_history"],
        }

    
    all_results = []
    for file_path in data_files['tc']:
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
        # 전처리
                 
        proc = ds.map(
            partial(preprocess_example_it, prompt_template=prompt_template, test_type=test_type, rewrited_examples=rewrited_examples)
        )
        # 평가
        file_results = []

        for ex in tqdm(proc, desc=f"Processing {os.path.basename(file_path)}"):
            prompt = ex["strprompt"]              
            try:
                if test_with_cloud:
                    raw = generate_response("", [prompt])                
                    raw = raw[0]["text"]
                    raw = extract_json_from_markdown(raw)
                else:                
                    raw = generate_text(prompt, model=model_name)                                
                    raw = raw + '}'
                print('--'*20)
                print(ex.get("rewrited_query"))
                print(raw)
            except Exception as e:                
                print(f"Error: {e}")                

            file_results.append({
                "conversation_history": ex.get("conversation_history"),
                "query":                ex.get("query"),
                "rewrited_query":       ex.get("rewrited_query"),                
                "generation":           raw,
                "gt":                   ex["stranswer"],                
            })

        df_file = pd.DataFrame(file_results)        
        all_results.extend(file_results)

    result = pd.DataFrame(all_results)    
    result.to_csv(out_file, sep='\t', index=False, encoding='utf-8-sig')        
    
# python3 rewrite_tester.py --o logs/llama-it-short-ours.tsv
if __name__ == "__main__":
    args = get_arg_parse()
    main(args.o)
