import pdb
import json
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import JsonExtractor, SimilarityFilter, DataFilter
from utils import OpenAiGenerateResponse, GoogleGenerateResponse
from utils.frequently_used_tools import get_model_name, get_arg_parse, read_jsonl
from prompts import FEW_SHOTS_REWRITE_PROMPT
from datasets import load_dataset
from datasets import Dataset
from functools import partial
from openai import OpenAI

def extract_json_from_markdown(text):
    """
    마크다운 블록(```json ... ```) 또는 그냥 {...} 블록에서 JSON만 추출하고 dict로 반환
    """
    try:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("JSON 블록을 찾을 수 없습니다.")

        return json.loads(json_str)

    except Exception as e:
        print("JSON 추출/파싱 실패:", e)
        return None

def chunks(lst, n):
    """리스트 lst를 n개씩 나눠 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def preprocess_example_it(example, prompt_template, rewrited_examples):
    try:
        f_ex = ""
        for data in rewrited_examples:
            f_ex += json.dumps(data, ensure_ascii=False, indent=2) + "\n"

        cv = example["conversation_history"].copy()
        cv.append(f"turn {len(cv)+1}: {example['query']} -> {example['device_response']}")

        data = {"conversation_history": cv, "query": example["next_turn_query"]}
        prompt = prompt_template.format(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            examples=f_ex,
        )

        return {
            "strprompt": prompt,
            "data": json.dumps(dict(example), ensure_ascii=False)
        }

    except Exception as e:
        # 예외가 발생하면 None을 반환        
        print(e)        
        return {}


def main():
    args = get_arg_parse()
    target_file = args.s  # "gemini_datagen/it1_s3_idxed_dedup.jsonl"
    api_file    = args.api
    OUTPUT_FILE = args.o  # "datagen/it1_s3_gemini.jsonl"

    # 1. Load rewrited examples for few‑shot context
    
    rewrited_examples = read_jsonl("examples.jsonl")
    try:
        ds = load_dataset('json', data_files={'train': [target_file]})['train']            
    except:
        print("load_dataset failed, loading from file")
        data = []
        with open(target_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                val = obj.get("device_response")
                if isinstance(val, list):
                    obj["device_response"] = " ".join(map(str, val))
                data.append(obj)

        ds = Dataset.from_list(data)
    # 2. Load target data and apply preprocessing to build prompts
    
    proc = ds.map(
        partial(preprocess_example_it,
                prompt_template=FEW_SHOTS_REWRITE_PROMPT,
                rewrited_examples=rewrited_examples),
        remove_columns=ds.column_names
    )
    print(proc[0]['strprompt'])
    #import pdb; pdb.set_trace()
    proc = proc.filter(lambda x: "strprompt" in x)
    # Convert mapped dataset to a list for parallel processing
    records = list(proc)    
    # 3. Prepare model & filters
    filters = [JsonExtractor()]
    model_name, generate_response = get_model_name(args.model)

    # 4. Define worker function
    def process_record(ex):
        prompt = ex["strprompt"]
        try:
            raw = generate_response("", [prompt])
            # raw might be e.g. [{"text": "..."}]
            text = raw[0]["text"]
            parsed = extract_json_from_markdown(text)
            if not parsed or "rewrited_query" not in parsed:
                raise ValueError("rewrited_query가 파싱되지 않았습니다.")
            new_query = parsed["rewrited_query"]

            data_dict = json.loads(ex["data"])    
            new_data = data_dict.copy()
            #new_data = ex["data"].copy()
            new_data["rewrited_query"] = new_query
            return new_data

        except Exception as e:
            print(f"Error processing record: {e}")
            return None

    # 5. Parallel evaluation with ThreadPoolExecutor
    response_datasets = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_record, ex): idx for idx, ex in enumerate(records)}
            for future in tqdm(as_completed(futures),
                               total=len(records),
                               desc=f"Processing {os.path.basename(target_file)}"):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    response_datasets.append(result)
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()

    print(f"Completed: 총 {len(response_datasets)}개 항목이 생성되었습니다.")

if __name__ == "__main__":
    main()