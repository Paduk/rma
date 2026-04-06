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
from prompts import COMPLEX_REWRITE_INFERENCE_PROMPT, FEW_SHOTS_REWRITE_PROMPT, FEW_SHOTS_SMALL_REWRITE_PROMPT, FEW_SHOTS_SIMPLER_REWRITE_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_example(ex, generate_response):
    """
    한 예제를 인퍼런스하고 결과 딕셔너리를 반환
    """
    try:
        # 1) LLM 호출
        raw = generate_response("", [ex["strprompt"]])[0]["text"]
        # 2) JSON 파싱
        parsed = extract_json_from_markdown(raw)
        # 3) 실제 rewrited_query 추출
        rewrited = parsed.get("rewrited_query", "")
    except Exception as e:
        print(f"[Error] {e}")
        rewrited = ""
    return {
        "conversation_history": ex.get("conversation_history"),
        "query":                ex.get("query"),
        "rewrited_query":       rewrited,
        "gt":                   ex.get("stranswer"),
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

def save_with_generated_queries(file_path, results, model_name):
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
    out_path = f"datasets/tc/{model_name}_rewrited/{filename_no_ext}.tsv"    
    df.to_csv(out_path, sep='\t', index=False, encoding='utf-8-sig')
    print(f"Saved updated dataset with `gen_rewrite_query` to: {out_path}")


def main():
    # API 데이터 파일 경로 (환경에 맞게 수정)            
    rewrited_examples = read_jsonl("examples.jsonl") # "datagen/it1_s3_gemini.jsonl"    
    rewrited_examples = get_examples(rewrited_examples)        

    model_name = "gemini-2.0-flash"
    model_name = 'o4-mini'
    model_name, generate_response = get_model_name(model_name)
           
    prompt_template = COMPLEX_REWRITE_INFERENCE_PROMPT # FEW_SHOTS_REWRITE_PROMPT                
    

    data_files = {
        'base': [            
            'datasets/tc/it2_NR_tc.tsv',
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
        'difficult': [                        
            'datasets/tc/it5_various_nonNR_tc.tsv',
            'datasets/tc/it5_various_dNR_tc.tsv',            
        ],
    }        
    tc_type = 'base'
    
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, prompt_template, rewrited_examples=None):                            
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
        ds   = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
        proc = ds.map(
            partial(preprocess_example_it,
                    prompt_template=prompt_template,                    
                    rewrited_examples=rewrited_examples)
        )

        print(f"\n>>> {os.path.basename(file_path)} 병렬 처리 시작")

        file_results = []
        max_workers = min(16, os.cpu_count() * 5)  # I/O 바운드 기준으로 조절
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Future 생성
            futures = [
                executor.submit(process_example, ex, generate_response)
                for ex in proc
            ]
            # 완료 순서대로 결과 수집
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc=f"Processing {os.path.basename(file_path)}"):
                file_results.append(fut.result())

        # 병렬 처리 후 저장
        save_with_generated_queries(file_path, file_results, model_name)
    
# python3 rma_inference.py --o 
if __name__ == "__main__":
    args = get_arg_parse()
    main()
