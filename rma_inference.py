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
from train.llama_prompts import SFT_RMA_INFERENCE_LLAMA, ZERO_REWRITE_INFERENCE_LLAMA, ZERO_HISTORY_INFERENCE_LLAMA, SFT_REWRITE_INFERENCE_LLAMA, SFT_HISTORY_INFERENCE_LLAMA
from train.gemma_prompts import ZERO_REWRITE_INFERENCE_GEMMA, SFT_REWRITE_INFERENCE_GEMMA
from train.llama_prompts import SFT_RMA_INFERENCE_PHI4, SFT_RMA_INFERENCE_QWEN3
from prompts import FEW_SHOTS_REWRITE_PROMPT, FEW_SHOTS_SMALL_REWRITE_PROMPT, FEW_SHOTS_SIMPLER_REWRITE_PROMPT

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
#def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11434'):
def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11436'):
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

def fix_unescaped_quotes(raw_str):
    # 큰따옴표 안에 있는 작은따옴표만 이스케이프
    def repl(match):
        content = match.group(0)
        fixed = content.replace("'", "\\'")
        return fixed
    # 큰따옴표 안에 있는 문자열 찾아서 수정
    fixed_raw = re.sub(r'"[^"]*"', repl, raw_str)
    return fixed_raw

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

def save_with_generated_queries(file_path, results, model_name, model_prefix):
    """
    - file_path: 원본 입력 데이터 파일 경로 (TSV)
    - results: main() 안에서 수집한 file_results 리스트. 각 요소는 적어도 'generation' 키를 가짐
    - sep: 구분자 (기본 '\t')
    - encoding: 저장 시 인코딩 (기본 'utf-8-sig')
    - output_file: 덮어쓸 파일 경로. None일 경우 file_path를 덮어씀
    """
    # 1. 원본 로드
    df = pd.read_csv(file_path, sep="\t", dtype=str)

    # 2. 결과에서 generation(생성된 raw 텍스트)만 추출
    gen_list = [item.get('rewrited_query', '') for item in results]

    # 3. 원본보다 결과가 적으면 부족한 만큼 빈 문자열로 채우기
    if len(gen_list) < len(df):
        gen_list.extend([''] * (len(df) - len(gen_list)))

    # 4. 새 컬럼 추가
    df['rewrited_query'] = gen_list

    # 5. 저장    
    filename = os.path.basename(file_path)  # 결과: 'it5_o4_1_tc.tsv'
    filename_no_ext = filename.split('.tsv')[0]  # 결과: 'it5_o4_1_tc'
    #out_path = f"datasets/tc/{model_name.split('-')[0]}-half_rewrited/{filename_no_ext}.tsv"    
    #out_path = f"datasets/tc/manual/ad/{model_prefix}_rewrited/{filename_no_ext}.tsv"    
    out_path = "test.tsv"
    df.to_csv(out_path, sep='\t', index=False, encoding='utf-8-sig')
    print(f"Saved updated dataset with `gen_rewrite_query` to: {out_path}")


def main():
    # API 데이터 파일 경로 (환경에 맞게 수정)        
    rewrited_examples = read_jsonl("examples.jsonl") # "datagen/it1_s3_gemini.jsonl"    
    rewrited_examples = get_examples(rewrited_examples)
    
    model_names = ['phi4-rma:latest', 'llama3-rma:latest', 'qwen3-rma:latest', 'qwen2.5-rma:latest']  # 사용할 모델 이름    
    model_name = model_names[2]   
    model_prefix = model_name.split('-')[0] 
    print(model_name)

    test_type = 'sft'
    if "llama" in model_name:
        prompt_template = SFT_RMA_INFERENCE_LLAMA
    elif "phi" in model_name or "Phi" in model_name:
        prompt_template = SFT_RMA_INFERENCE_PHI4
    elif "Qwen3" in model_name or "Qwen2.5" in model_name or 'qwen3' in model_name or 'qwen25' in model_name:
        prompt_template = SFT_RMA_INFERENCE_QWEN3    
    
    data_files = {
        "base": [
            "datasets/tc/new_scale/it2_nonNR_tc.tsv",
            "datasets/tc/new_scale/it3_nonNR_tc.tsv",
            "datasets/tc/new_scale/it4_nonNR_tc.tsv",
            "datasets/tc/new_scale/it5_nonNR_tc.tsv",
        ],
        "complex": [
            "datasets/tc/new_scale/it3_complex_1_tc.tsv",
            "datasets/tc/new_scale/it4_complex_1_tc.tsv",
            "datasets/tc/new_scale/it4_complex_2_tc.tsv",
            "datasets/tc/new_scale/it5_complex_1_tc.tsv",
            "datasets/tc/new_scale/it5_complex_2_tc.tsv",
            "datasets/tc/new_scale/it5_complex_3_tc.tsv",
        ],
        'difficult': [                        
            'datasets/tc/it5_various_nonNR_tc.tsv',
            'datasets/tc/it5_various_dNR_tc.tsv',            
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
    }        
    tc_type = 'base'  # 'base', 'complex', 'difficult' 중 하나로 설정
    
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, prompt_template, test_type, rewrited_examples=None):                
        if test_type == "sft":                        
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
    print(data_files[tc_type])
    for file_path in data_files[tc_type]:
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
        # 전처리
                 
        proc = ds.map(
            partial(preprocess_example_it, prompt_template=prompt_template, test_type=test_type, rewrited_examples=rewrited_examples)
        )
        # 평가
        print(proc[0]["strprompt"])        
        
        file_results = []

        for ex in tqdm(proc, desc=f"Processing {os.path.basename(file_path)}"):
            prompt = ex["strprompt"]                          
            try:                               
                raw = generate_text(prompt, model=model_name)                                 
                raw = raw + '}'                
                print('--'*20)
                print({"rewrited_query": ex.get("rewrited_query")})
                print(raw)
            except Exception as e:                
                print(f"Error: {e}")                
            
            try:     
                file_results.append({
                "conversation_history": ex.get("conversation_history"),
                "query":                ex.get("query"),
                "rewrited_query":       ast.literal_eval(raw)["rewrited_query"],                                
                "gt":                   ex["stranswer"],                
                })
            except Exception as e:                          
                print(raw.split("rewrited_query")[1].split(': ')[1].split('}')[0][1:])                
                file_results.append({
                    "conversation_history": ex.get("conversation_history"),
                    "query":                ex.get("query"),
                    "rewrited_query":       raw.split("rewrited_query")[1].split(': ')[1].split('}')[0][1:],                                
                    "gt":                   ex["stranswer"],                
                })            
            

        df_file = pd.DataFrame(file_results)        
        all_results.extend(file_results)
        save_with_generated_queries(file_path, file_results, model_name, model_prefix)
    
    
# python3 rma_inference.py --o 
if __name__ == "__main__":
    #args = get_arg_parse()
    main()
