import json
import re
import os
import requests
import ast
from datetime import datetime, timezone
from pathlib import Path
from datasets import load_dataset
from utils.frequently_used_tools import get_arg_parse, get_model_name
import pdb
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from functools import partial
from utils.oneshot_qwen_prompt import (
    build_api_str_from_candidates,
    render_messages_as_plain_text,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_example(
    ex,
    *,
    row_idx,
    test_key,
    model_name,
    generate_response,
    extract_fn,
    gt_parser,
    file_path,
):
    """
    한 예제를 처리하고 결과 dict를 반환하는 함수
    - generate_response: LLM 호출 함수
    - extract_fn: extract_json_from_markdown 또는 parse_response_json
    - gt_parser: 예측값과 비교할 gt(dict)로 변환하는 함수
    """
    error_payload = None
    prompt = ex.get("strprompt", "")
    raw = ""
    gt = {}

    try:
        # 1. 응답 생성
        response = generate_response("", [prompt])[0]
        raw = response.get("text", "")
        
        if not raw:
            raise ValueError("Empty response text (possibly due to finish_reason = 4)")

        # 2. 응답에서 JSON 추출
        result = parse_model_response(raw, extract_fn)

        # 3. GT 파싱 및 비교
        gt = gt_parser(ex["stranswer"])
        if isinstance(gt, str):
            gt = ast.literal_eval(gt)

        plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
        arg_res  = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
        all_res  = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"

    except Exception as e:
        result, plan_res, arg_res, all_res = {"error": str(e)}, "fail", "fail", "fail"
        error_payload = {
            "test_key": test_key,
            "file": os.path.basename(file_path),
            "row_idx": row_idx,
            "model_name": model_name,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "raw_error_type": classify_raw_error(raw),
            "raw": raw,
        }

    row = {
        "conversation_history": ex.get("conversation_history"),
        "query":                ex.get("query"),
        "rewrited_query":       ex.get("rewrited_query"),
        "candidates":           ex.get("candidates"),
        "raw":                  raw,
        "generation":           result,
        "gt":                   gt,
        "plan":                 plan_res,
        "arguments":            arg_res,
        "all":                  all_res,
        "file":                 os.path.basename(file_path)
    }
    return row, error_payload, prompt, raw

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


def append_error_log(error_log_path: Path, payload: dict):
    final_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(final_payload, ensure_ascii=False))
        f.write("\n")


def parse_model_response(raw: str, extract_fn):
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = extract_fn(raw)
        if not isinstance(parsed, dict):
            parsed = parse_response_json(raw)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a dict.")
    return parsed


def extract_turn_from_filename(file_name):
    match = re.search(r"it(\d+)", file_name)
    if not match:
        return None
    return int(match.group(1))


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


def build_batch_system_message(api_str: str, test_type: str) -> str:
    if test_type == "history":
        return (
            "Given a conversation history, a user query, and a list of available tools, "
            "select the most appropriate tool and generate its arguments. "
            "Only use parameter values that are explicitly stated or can be reasonably inferred "
            "from the query or conversation history. Return compact JSON only with keys "
            "\"plan\" and \"arguments\". Always include both keys. The value of "
            "\"arguments\" must always be an object. If no tool matches the request, "
            "set \"plan\" to \"None\" and \"arguments\" to {}.\n"
            f"<|tool|>{api_str}<|/tool|>"
        )
    if test_type == "rewrite":
        return (
            "Given a user query and a list of available tools, "
            "select the most appropriate tool and generate its arguments. "
            "Only use parameter values that are explicitly stated or can be reasonably inferred "
            "from the query. Return compact JSON only with keys "
            "\"plan\" and \"arguments\". Always include both keys. The value of "
            "\"arguments\" must always be an object. If no tool matches the request, "
            "set \"plan\" to \"None\" and \"arguments\" to {}.\n"
            f"<|tool|>{api_str}<|/tool|>"
        )
    raise ValueError(f"Unsupported test_type: {test_type}")


def build_batch_user_content(example, test_type: str) -> str:
    if test_type == "history":
        return (
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
    if test_type == "rewrite":
        return f"User Query: {example['rewrited_query']}"
    raise ValueError(f"Unsupported test_type: {test_type}")


def build_batch_prompt(example, apis, test_type: str) -> str:
    candidates = ast.literal_eval(example["candidates"])
    api_str = build_api_str_from_candidates(candidates, apis)
    messages = [
        {"role": "system", "content": build_batch_system_message(api_str, test_type)},
        {"role": "user", "content": build_batch_user_content(example, test_type)},
    ]
    return render_messages_as_plain_text(
        messages=messages,
        add_generation_prompt=True,
    )

def main(out_file):
    # API 데이터 파일 경로 (환경에 맞게 수정)
    args = get_arg_parse()
    print(args)
    model_name, generate_response = get_model_name(args.model)    
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = out_path.with_name(f"{out_path.stem}.errors.jsonl")
    init_error_log(error_log_path)
    
    apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)    
    
    test_type = args.t # 'rewrite'  # 'rewrite' or 'history'
    print(f"error_log_path: {error_log_path}")
    print("prompt_style: plain_text_messages")
    # python3 cloudllm_inference_batch.py --model gpt-5-nano --t rewrite --o datasets/rewrite-gpt5-nano-complex.tsv && python3 cloudllm_inference_batch.py --model gpt-5-nano --t rewrite --o datasets/rewrite-gpt5-nano-base.tsv
    # python3 cloudllm_inference_batch.py --model gpt-5-nano --t history --o datasets/history-gpt5-nano-complex.tsv && python3 cloudllm_inference_batch.py --model gpt-5-nano --t history --o datasets/history-gpt5-nano-base.tsv
    # python3 cloudllm_inference_batch.py --model gpt-4.1-2025-04-14 --t rewrite --o datasets/manual/rewrite-cloud-gpt41.tsv
    # python3 cloudllm_inference_batch.py --model gpt-4.1-2025-04-14 --t history --o datasets/manual/history-cloud-gpt41.tsv
    # python3 cloudllm_inference_batch.py --model o4-mini --t rewrite --o datasets/manual/rewrite-cloud-o4-mini.tsv
    # python3 cloudllm_inference_batch.py --model o4-mini --t history --o datasets/manual/history-cloud-o4-mini.tsv
    # python3 cloudllm_inference_batch.py --model gemini-2.0-flash --t rewrite --o datasets/manual/rewrite-cloud-gemini2.tsv && python3 cloudllm_inference_batch.py --model gemini-2.0-flash --t history --o datasets/manual/history-cloud-gemini2.tsv
    # python3 cloudllm_inference_batch.py --model gpt-4o-mini-2024-07-18 --t rewrite --o datasets/manual/rewrite-cloud-gpt4o-mini.tsv && python3 cloudllm_inference_batch.py --model gpt-4o-mini-2024-07-18 --t history --o datasets/manual/history-cloud-gpt4o-mini.tsv
    data_files = {                
        "base": [
            "datasets/tc/scale/it2_nonNR_tc.tsv",
            "datasets/tc/scale/it3_nonNR_tc.tsv",
            "datasets/tc/scale/it4_nonNR_tc.tsv",
            "datasets/tc/scale/it5_nonNR_tc.tsv",
        ], 
        'manual': [                     
            #'datasets/tc/manual/test.tsv',            
            'datasets/tc/manual/turn2.tsv',
            'datasets/tc/manual/turn3.tsv',            
            'datasets/tc/manual/turn4.tsv',
            'datasets/tc/manual/turn5.tsv',
        ],         
        "complex": [
            "datasets/tc/scale/it3_complex_1_tc.tsv",
            "datasets/tc/scale/it4_complex_1_tc.tsv",
            "datasets/tc/scale/it4_complex_2_tc.tsv",
            "datasets/tc/scale/it5_complex_1_tc.tsv",
            "datasets/tc/scale/it5_complex_2_tc.tsv",
            "datasets/tc/scale/it5_complex_3_tc.tsv",
        ],                               
    }
    test_keys = parse_test_keys(args.test_key, data_files)
        
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, apis, prompt_template, test_type):
        prompt = build_batch_prompt(
            example,
            apis,
            test_type=test_type,
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
    printed_first_prompt = False
    printed_first_raw = False
    print(f"Selected test_keys: {test_keys}")
    for test_key in test_keys:
        print(f"\n# Running split: {test_key}")
        print(data_files[test_key])
        for file_path in data_files[test_key]:
            ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']
            proc = ds.map(
                partial(preprocess_example_it, apis=apis, prompt_template=None, test_type=test_type)
            )
            file_results = []
            max_workers = min(8, os.cpu_count() * 5)  # I/O 바운드이므로 cpu_count * 상수

            if len(proc) > 0:
                first_row, first_error_payload, first_prompt, first_raw = process_example(
                    proc[0],
                    row_idx=0,
                    test_key=test_key,
                    model_name=model_name,
                    generate_response=generate_response,
                    extract_fn=extract_json_from_markdown,
                    gt_parser=lambda s: ast.literal_eval(s) if isinstance(s, str) else s,
                    file_path=file_path,
                )
                if not printed_first_prompt:
                    print("first_inference_prompt:")
                    print(first_prompt)
                    print()
                    printed_first_prompt = True
                if not printed_first_raw:
                    print("first_inference_raw:")
                    print(first_raw)
                    print()
                    printed_first_raw = True
                first_row["test_key"] = test_key
                first_row["turn"] = extract_turn_from_filename(os.path.basename(file_path))
                file_results.append(first_row)
                if first_error_payload is not None:
                    append_error_log(error_log_path, first_error_payload)
                    print(
                        f"[Error] {first_error_payload['file']} row={first_error_payload['row_idx']}: "
                        f"{first_error_payload['error_message']}",
                        flush=True,
                    )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_example,
                        ex,
                        row_idx=row_idx,
                        test_key=test_key,
                        model_name=model_name,
                        generate_response=generate_response,
                        extract_fn=extract_json_from_markdown,
                        gt_parser=lambda s: ast.literal_eval(s) if isinstance(s, str) else s,
                        file_path=file_path,
                    ): ex
                    for row_idx, ex in enumerate(proc)
                    if row_idx != 0
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{test_key}/{os.path.basename(file_path)}",
                ):
                    row, error_payload, _, _ = future.result()
                    row["test_key"] = test_key
                    row["turn"] = extract_turn_from_filename(os.path.basename(file_path))
                    file_results.append(row)
                    if error_payload is not None:
                        append_error_log(error_log_path, error_payload)
                        print(
                            f"[Error] {error_payload['file']} row={error_payload['row_idx']}: "
                            f"{error_payload['error_message']}",
                            flush=True,
                        )

            df_file = pd.DataFrame(file_results)
            print_eval(df_file, title=f"{test_key}/{os.path.basename(file_path)}", test_type=test_type)
            all_results.extend(file_results)

    result = pd.DataFrame(all_results)    
    # print_eval(result, title=os.path.basename(file_path), test_type=test_type)
    result.to_csv(out_path, sep='\t', index=False, encoding='utf-8-sig')        
    
if __name__ == "__main__":
    args = get_arg_parse()
    main(args.o)
