import json
import re
import os
import requests
import ast
from datetime import datetime, timezone
from pathlib import Path
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
def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11435'): # qwen3
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
    metric_rows = []
    for col in metrics:
        if col == "all":
            acc = compute_plan_macro(df, metric=col)
            label = f"{col.title():<10} Macro Accuracy"
        else:
            acc = df[col].eq("pass").mean()  # 0~1 사이의 값
            label = f"{col.title():<10} Accuracy"
        metric_rows.append((label, acc))
        print(f"{label} : {acc * 100:.2f}%")

    print("-" * 40)

    # -- 로그파일에도 동일 내용 쓰기 --
    with open("logs/ollama_inference_log.txt", 'a', encoding='utf-8') as f:
        if title:
            f.write(f"\n## Performance for {title}, {test_type}\n")
        for label, acc in metric_rows:
            f.write(f"{label} : {acc * 100:.2f}%\n")
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

def init_error_log(error_log_path: Path):
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_log_path, "w", encoding="utf-8"):
        pass


def classify_raw_error(raw: str) -> str:
    stripped = (raw or "").strip()
    if not stripped:
        return "empty_raw"
    if "\n{" in stripped or "}\n{" in stripped:
        return "two_json_objects_newline"
    if re.search(r"\}\s*,\s*\{", stripped):
        return "two_objects_comma_split"

    missing_braces = stripped.count("{") - stripped.count("}")
    if missing_braces > 0:
        return f"truncated_missing_{missing_braces}_brace"
    if missing_braces < 0:
        return f"extra_{abs(missing_braces)}_closing_brace"

    if re.search(r'"[A-Za-z0-9_]+"\s*:\s*-?\d+"', stripped):
        return "number_then_stray_quote"
    if re.search(r'"[A-Za-z0-9_]+"\s*:\s*(true|false|null)"', stripped):
        return "bool_or_null_then_stray_quote"
    return "other_malformed_json"


def append_error_log(
    error_log_path: Path,
    *,
    test_key: str,
    file_name: str,
    row_idx: int,
    model_name: str,
    host: str,
    error: Exception,
    raw: str,
):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_key": test_key,
        "file": file_name,
        "row_idx": row_idx,
        "model_name": model_name,
        "host": host,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_error_type": classify_raw_error(raw),
        "raw": raw,
    }
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")

def extract_turn_from_filename(file_name):
    match = re.search(r"it(\d+)", file_name)
    if not match:
        return None
    return int(match.group(1))

def compute_plan_macro(df, metric="all"):
    if df.empty:
        return float("nan")

    gt_plan_series = df["gt"].apply(
        lambda x: x.get("plan") if isinstance(x, dict) else None
    )
    metric_by_plan = (
        pd.DataFrame({"gt_plan": gt_plan_series, metric: df[metric]})
        .dropna(subset=["gt_plan"])
        .groupby("gt_plan")[metric]
        .apply(lambda sub: sub.eq("pass").mean())
    )
    return metric_by_plan.mean()

def compute_file_accuracy(df, metric="all"):
    if df.empty:
        return float("nan")
    return df[metric].eq("pass").mean()

def get_turn_file_sort_key(file_name):
    turn = extract_turn_from_filename(file_name)
    complex_match = re.search(r"complex_(\d+)", file_name)
    if complex_match:
        # complex 파일은 요청한 출력 순서에 맞게 suffix 역순 정렬
        return (turn if turn is not None else 999, 0, -int(complex_match.group(1)))
    # nonNR/base 파일은 complex 뒤에 오도록 배치
    return (turn if turn is not None else 999, 1, file_name)

def print_turn_macro_summary(df, title=None, metric="all"):
    if df.empty or "turn" not in df.columns:
        return

    file_names = sorted(df["file"].dropna().unique().tolist(), key=get_turn_file_sort_key)
    rows = []
    for turn, sub in sorted(df.dropna(subset=["turn"]).groupby("turn"), key=lambda x: x[0]):
        turn_macro = compute_plan_macro(sub, metric=metric)
        row = {
            "turn": int(turn),
            "samples": len(sub),
            "plan_macro_all": round(turn_macro, 4)
        }
        turn_files = set(sub["file"].dropna().tolist())
        for file_name in file_names:
            if file_name in turn_files:
                row[file_name] = round(compute_file_accuracy(sub[sub["file"] == file_name], metric=metric), 4)
            else:
                row[file_name] = None
        rows.append(row)

    if not rows:
        return

    turn_df = pd.DataFrame(rows)
    ordered_cols = ["turn", "samples", "plan_macro_all"] + file_names
    turn_df = turn_df[ordered_cols]
    if title:
        print(f"\n# Turn-wise Plan Macro for {title}\n")
    else:
        print("\n# Turn-wise Plan Macro\n")
    print(turn_df.fillna("N/A").to_string(index=False))

def parse_test_keys(test_key_arg, data_files):
    if not test_key_arg:
        raise ValueError("`--test_key` is required. Example: --test_key base or --test_key base,complex")

    test_keys = [key.strip() for key in test_key_arg.split(",") if key.strip()]
    if not test_keys:
        raise ValueError("No valid test_key values were provided.")

    deduped_test_keys = []
    for key in test_keys:
        if key not in data_files:
            raise ValueError(f"Invalid test_key: {key}. Available keys: {', '.join(sorted(data_files.keys()))}")
        if key not in deduped_test_keys:
            deduped_test_keys.append(key)

    return deduped_test_keys

"""
python3 /home/hj153lee/RMA/ollama_inference_multi.py \
--t new-base-qwen3 \
--test_key base,complex \
--o /home/hj153lee/RMA/datasets/result/260406-ablation/qwen3-new-base.tsv \
--host http://localhost:21435
"""
def main(out_file):      
    #apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)    
    sft_apis = read_apis("apis/simple_api.json", simple=True)    
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = out_path.with_name(f"{out_path.stem}.errors.jsonl")
    init_error_log(error_log_path)
    test_type_config = {
        "base-phi4": {
            "model_name": "phi4-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "rewrite-phi4": {
            "model_name": "phi4-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_PHI4,
            "prompt_mode": "rewrite",
        },
        "base-llama3": {
            "model_name": "llama3-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        "rewrite-llama3": {
            "model_name": "llama3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
            "prompt_mode": "rewrite",
        },
        "base-qwen3": {
            "model_name": "qwen3-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3": {
            "model_name": "qwen3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "base-qwen3-1.7b": {
            "model_name": "qwen3-base-1.7b:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3-1.7b": {
            "model_name": "qwen3-rewrite-1.7b:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "base-qwen3-0.6b": {
            "model_name": "qwen3-base-0.6b:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3-0.6b": {
            "model_name": "qwen3-rewrite-0.6b:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "base-qwen2.5": {
            "model_name": "qwen2.5-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN25,
            "prompt_mode": "base",
        },
        "rewrite-qwen2.5": {
            "model_name": "qwen2.5-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN25,
            "prompt_mode": "rewrite",
        },
        "base-qwen2.5-base": {
            "model_name": "qwen2.5-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_QWEN25,
            "prompt_mode": "base",
        },
        "rewrite-qwen2.5-base": {
            "model_name": "qwen2.5-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_QWEN25,
            "prompt_mode": "rewrite",
        },
        "base-phi4-base": {
            "model_name": "phi4-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "rewrite-phi4-base": {
            "model_name": "phi4-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_PHI4,
            "prompt_mode": "rewrite",
        },
        "base-llama3-base": {
            "model_name": "llama-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        "rewrite-llama3-base": {
            "model_name": "llama-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_LLAMA,
            "prompt_mode": "rewrite",
        },
        "base-qwen3-base": {
            "model_name": "qwen3-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3-base": {
            "model_name": "qwen3-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "new-base-qwen3": {
            "model_name": "qwen3-qwen-history-all_linear:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "new-base-phi4": {
            "model_name": "phi4-phi-history-1st:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-phi4-e4": {
            "model_name": "phi4-phi-history-1st-e4:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-phi4-e5": {
            "model_name": "phi4-phi-history-1st-e5:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-phi4-e6": {
            "model_name": "phi4-phi-history-1st:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-llama3": {
            "model_name": "llama3-llama-history-all_linear:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        "new-base-qwen2.5": {
            "model_name": "qwen25-qwen-history-all_linear:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_QWEN25,
            "prompt_mode": "base",
        }                 
    }

    test_type = args.t
    config = test_type_config.get(test_type)
    if config is None:
        valid_test_types = ", ".join(sorted(test_type_config.keys()))
        raise ValueError(f"Invalid test type: {test_type}. Available test types: {valid_test_types}")

    model_name = config["model_name"]
    prompt_template = config["prompt_template"]
    prompt_mode = config["prompt_mode"]

    
    '''
    source /mnt/data/miniconda3/bin/activate && conda activate mobile-agent-vllm
    python3 ollama_inference_multi.py \
        --t base-qwen3-1.7b \
        --test_key base,complex \
        --o datasets/result/260330/qwen3-1.7b-base.tsv

    python3 ollama_inference_multi.py \
        --t base-llama3 \
        --test_key base,complex \
        --o datasets/result/260331/scale-llama3-4b-base.tsv

    python3 ollama_inference_multi.py \
        --t base-qwen3 \
        --test_key swap \
        --o datasets/result/scale-qwen3-base-test.tsv
    '''

    data_files = {      
        'swap': [
            'datasets/tc/it2_nonNR_tc_swapped_backup.tsv',
            'datasets/tc/it5_nonNR_tc_swapped_backup.tsv'
        ],
        'base': [                     
            # 'datasets/tc/it2_nonNR_tc.tsv',
            # 'datasets/tc/it3_nonNR_tc.tsv',
            # 'datasets/tc/it4_nonNR_tc.tsv',
            # 'datasets/tc/it5_nonNR_tc.tsv',      
            'datasets/tc/scale/it2_nonNR_tc.tsv',
            'datasets/tc/scale/it3_nonNR_tc.tsv',
            'datasets/tc/scale/it4_nonNR_tc.tsv',
            'datasets/tc/scale/it5_nonNR_tc.tsv',      
               
        ],   
        'complex': [ 
            'datasets/tc/scale/it3_complex_1_tc.tsv',
            'datasets/tc/scale/it4_complex_1_tc.tsv',
            'datasets/tc/scale/it4_complex_2_tc.tsv',
            'datasets/tc/scale/it5_complex_1_tc.tsv',
            'datasets/tc/scale/it5_complex_2_tc.tsv',
            'datasets/tc/scale/it5_complex_3_tc.tsv',                          
            # 'datasets/tc/it3_complex_1_tc.tsv',
            # 'datasets/tc/it4_complex_1_tc.tsv',
            # 'datasets/tc/it4_complex_2_tc.tsv',
            # 'datasets/tc/it5_complex_1_tc.tsv',
            # 'datasets/tc/it5_complex_2_tc.tsv',
            # 'datasets/tc/it5_complex_3_tc.tsv',                    
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

    test_keys = parse_test_keys(args.test_key, data_files)

    print(model_name)
    print(args.host)
    print(f"error_log_path: {error_log_path}")
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, apis, prompt_template, prompt_mode):
        api_str = ""
        #re_fmt  = {"plan": "str type tool", "arguments": {"key1": "value1"}}        
        for plan in ast.literal_eval(example["candidates"]):
            api_data = apis[plan].copy()
            api_str += f"{plan}: {api_data}\n"
        
        if prompt_mode == "base":
            prompt = prompt_template.format(
                tools=api_str,
                #re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),
                conversation_history=example["conversation_history"],
                data=example["query"]
            )
        elif prompt_mode == "rewrite":
            prompt = prompt_template.format(
                tools=api_str,
                #re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),                
                data=example["rewrited_query"]
            )
        else:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

        return {
            "strprompt":    prompt,
            "stranswer":    json.dumps(example["answer"], ensure_ascii=False, indent=2),
            "candidates":   example["candidates"],
            "rewrited_query": example["rewrited_query"],
            "query":        example["query"],
            "conversation_history": example["conversation_history"],
        }
    
    all_results = []
    split_results = {key: [] for key in test_keys}
    print(f"Selected test_keys: {test_keys}")
    for test_key in test_keys:
        print(f"\n# Running split: {test_key}")
        print(data_files[test_key])
        for file_path in data_files[test_key]:
            ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
            # 전처리
            proc = ds.map(
                partial(preprocess_example_it, apis=sft_apis, prompt_template=prompt_template, prompt_mode=prompt_mode)
            )
            # else:
            #     proc = ds.map(
            #         partial(preprocess_example_it, apis=apis, prompt_template=prompt_template, test_type=test_type)
            #     )
            # 평가
            print(proc[0]["strprompt"])
            #exit(0)
            file_results = []

            for row_idx, ex in enumerate(
                tqdm(proc, desc=f"Processing {test_key}/{os.path.basename(file_path)}")
            ):
                prompt = ex["strprompt"]
                raw = ""
                gt = {}

                try:
                    raw = generate_text(prompt, model=model_name, host=args.host)
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
                    append_error_log(
                        error_log_path,
                        test_key=test_key,
                        file_name=os.path.basename(file_path),
                        row_idx=row_idx,
                        model_name=model_name,
                        host=args.host,
                        error=e,
                        raw=raw,
                    )
                    print(f"Error: {e}, {raw}")
                    plan_res = "fail"
                    arg_res  = "fail"
                    all_res  = "fail"

                row = {
                    "test_key":             test_key,
                    "conversation_history": ex.get("conversation_history"),
                    "query":                ex.get("query"),
                    "rewrited_query":       ex.get("rewrited_query"),
                    "candidates":           ex.get("candidates"),
                    "generation":           result,
                    "gt":                   gt,
                    "plan":                 plan_res,
                    "arguments":            arg_res,
                    "all":                  all_res,
                    "file":                 os.path.basename(file_path),
                    "turn":                 extract_turn_from_filename(os.path.basename(file_path))
                }
                file_results.append(row)
                split_results[test_key].append(row)

            df_file = pd.DataFrame(file_results)
            print_eval(df_file, title=f"{test_key}/{os.path.basename(file_path)}", test_type=model_name)
            all_results.extend(file_results)

    result = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(df_split, title=f"split={test_key}", test_type=model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(result, title=f"combined={combined_title}", test_type=model_name)
    print_turn_macro_summary(result, title=f"combined={combined_title}", metric="all")
    result.to_csv(out_path, sep='\t', index=False, encoding='utf-8-sig')        
    
# python3 ollama_inference.py --o datasets/result/base_rewrite_gt.tsv --t rewrite
if __name__ == "__main__":
    args = get_arg_parse()
    main(args.o)
