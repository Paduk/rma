import json
import re
import os
import requests
import ast
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pdb
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from train.llama_prompts import ZERO_REWRITE_INFERENCE_LLAMA, ZERO_HISTORY_INFERENCE_LLAMA, SFT_REWRITE_INFERENCE_LLAMA, SFT_HISTORY_INFERENCE_LLAMA
from train.gemma_prompts import SFT_REWRITE_INFERENCE_GEMMA, SFT_HISTORY_INFERENCE_GEMMA
from train.llama_prompts import SFT_REWRITE_INFERENCE_PHI4, SFT_HISTORY_INFERENCE_PHI4, ZERO_REWRITE_INFERENCE_PHI4, ZERO_HISTORY_INFERENCE_PHI4
from train.llama_prompts import SFT_REWRITE_INFERENCE_QWEN25, SFT_HISTORY_INFERENCE_QWEN25, ZERO_REWRITE_INFERENCE_QWEN25, ZERO_HISTORY_INFERENCE_QWEN25
from train.llama_prompts import SFT_REWRITE_INFERENCE_QWEN3, SFT_HISTORY_INFERENCE_QWEN3, ZERO_REWRITE_INFERENCE_QWEN3, ZERO_HISTORY_INFERENCE_QWEN3

PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_DIR = PROJECT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from history_retrieval import (
    DEFAULT_RETRIEVAL_MODEL_NAME,
    build_history_selector,
)

GENERIC_SYSTEM_HISTORY_PROMPT = (
    "You are a helpful assistant capable of selecting appropriate tools based on "
    "user queries and generating corresponding parameters. Use information from "
    "the conversation history when relevant. Only use parameter values that are "
    "explicitly stated or can be reasonably inferred from the query. If no tool "
    "matches the query, set the tool to 'None'.\n <|tool|>{tools}<|/tool|>"
)

GENERIC_SYSTEM_REWRITE_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values that "
    "are explicitly stated or can be reasonably inferred from the query.\n "
    "<|tool|>{tools}<|/tool|>"
)

GLM_STOP_SEQUENCES = [
    "<|observation|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
]

SCHEMA_RESAMPLE_TEMPERATURES = [0.0, 0.2, 0.5]

GENERIC_PROMPT_PROFILES = {
    "glm-edge-1.5b": {
        "prompt_model_name": "zai-org/glm-edge-1.5b-chat",
        "prefix": "all_linear",
    },
    "glm-edge-4b": {
        "prompt_model_name": "zai-org/glm-edge-4b-chat",
        "prefix": "all_linear",
        "stop": GLM_STOP_SEQUENCES,
    },
    "qwen2.5": {
        "prompt_model_name": "Qwen/Qwen2.5-3B-Instruct",
        "prefix": "all_linear",
    },
    "llama3.2-1b": {
        "prompt_model_name": "meta-llama/Llama-3.2-1B",
        "prefix": "all_linear",
    },
    "smollm2-1.7b": {
        "prompt_model_name": "HuggingFaceTB/SmolLM2-1.7B",
        "prefix": "simple_template",
    },
    "smollm2-1.7b-instruct": {
        "prompt_model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "prefix": "all_linear",
    },
    "smollm3-3b": {
        "prompt_model_name": "HuggingFaceTB/SmolLM3-3B",
        "prefix": "all_linear",
    },
    "falcon3-1b": {
        "prompt_model_name": "tiiuae/Falcon3-1B-Instruct",
        "prefix": "all_linear",
    },
    "falcon3-1b-base": {
        "prompt_model_name": "tiiuae/Falcon3-1B-Base",
        "prefix": "simple_template",
    },
    "falcon3-3b": {
        "prompt_model_name": "tiiuae/Falcon3-3B-Instruct",
        "prefix": "all_linear",
    },
    "falcon3-3b-base": {
        "prompt_model_name": "tiiuae/Falcon3-3B-Base",
        "prefix": "simple_template",
    },
    "exaone4-1.2b": {
        "prompt_model_name": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "prefix": "all_linear",
    },
    "olmo2-1b": {
        "prompt_model_name": "allenai/OLMo-2-0425-1B",
        "prefix": "simple_template",
    },
    "olmo2-1b-instruct": {
        "prompt_model_name": "allenai/OLMo-2-0425-1B-Instruct",
        "prefix": "all_linear",
    },
    "granite3.3-2b": {
        "prompt_model_name": "ibm-granite/granite-3.3-2b-instruct",
        "prefix": "all_linear",
    },
    "lfm2.5-1.2b": {
        "prompt_model_name": "LiquidAI/LFM2.5-1.2B-Instruct",
        "prefix": "all_linear",
    },
}


def get_arg_parse():
    parser = argparse.ArgumentParser(description="Ollama inference for RMA SFT models")
    parser.add_argument('--t', type=str, required=False, help='target_file')
    parser.add_argument('--o', type=str, required=False, help='out_file')
    parser.add_argument('--t1', type=str, required=False, help='target_file')
    parser.add_argument('--t2', type=str, required=False, help='target_file')
    parser.add_argument('--step', type=str, required=False, help='out_file')
    parser.add_argument('--it', type=str, required=False, help='iteration_file')
    parser.add_argument('--model', type=str, required=False, help='Ollama model name override')
    parser.add_argument('--t_list', type=str, nargs='+', required=False, help='List of iteration files')
    parser.add_argument('--d', action='store_true', help='Enable debug mode')
    parser.add_argument('--s', type=str, required=False, help='start_file')
    parser.add_argument('--api', type=str, default="apis/api_v3.0.1.jsonl", help='사용자 이름')
    parser.add_argument('--test_key', type=str, default="", help='')
    parser.add_argument('--host', type=str, default="http://localhost:11436", help='Ollama host URL')
    parser.add_argument('--temperature', type=float, default=0.0, help='Ollama sampling temperature')
    parser.add_argument(
        '--schema_resample',
        action='store_true',
        help='Retry generation when the output is not a valid candidate/schema-conformant tool call.',
    )
    parser.add_argument(
        '--max_resample_attempts',
        type=int,
        default=3,
        help='Maximum attempts for --schema_resample. Attempts use temperatures 0.0, 0.2, 0.5.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print selected history turns and rendered prompts, then skip Ollama API calls.',
    )
    parser.add_argument(
        '--dry_run_limit',
        type=int,
        default=3,
        help='Examples to print per file in --dry_run. Use 0 for all rows.',
    )
    parser.add_argument(
        '--multi',
        '--num_parallel',
        dest='num_parallel',
        nargs='?',
        type=int,
        const=4,
        default=1,
        help=(
            'Number of concurrent Ollama requests. Omit or set to 1 for the '
            'current sequential behavior; use --multi 3~5 for parallel calls.'
        ),
    )
    parser.add_argument(
        '--prompt_option',
        type=str,
        default='prompt1',
        choices=['prompt1', 'prompt2'],
        help='Prompt variant for scripts that support prompt ablations.',
    )
    parser.add_argument(
        '--reasoning_effort',
        type=str,
        default=None,
        choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh'],
        help='OpenAI reasoning effort for supported reasoning models.',
    )
    parser.add_argument(
        '--history_selection',
        choices=['full', 'last_k', 'retrieval'],
        default='full',
        help='History input mode for base/history evaluation. Default: full.',
    )
    parser.add_argument(
        '--history_top_k',
        type=int,
        default=2,
        help='Number of turns to keep for --history_selection last_k/retrieval.',
    )
    parser.add_argument(
        '--retrieval_model_name',
        default=DEFAULT_RETRIEVAL_MODEL_NAME,
        help='Embedding model used by --history_selection retrieval.',
    )
    parser.add_argument(
        '--retrieval_device',
        default='cpu',
        help='Device for retrieval encoder: cpu, cuda, cuda:0, or auto. Default: cpu.',
    )
    parser.add_argument(
        '--retrieval_max_length',
        type=int,
        default=512,
        help='Max token length for retrieval encoder inputs.',
    )
    parser.add_argument(
        '--retrieval_recency_lambda',
        type=float,
        default=0.1,
        help='Recency bonus weight added to cosine similarity.',
    )
    parser.add_argument(
        '--retrieval_query_prefix',
        default='',
        help='Optional prefix prepended to the query before embedding.',
    )
    parser.add_argument(
        '--retrieval_document_prefix',
        default='',
        help='Optional prefix prepended to each history turn before embedding.',
    )
    parser.add_argument(
        '--retrieval_order',
        choices=['original', 'score'],
        default='original',
        help='Order for selected turns in the prompt. Default keeps dialogue order.',
    )
    parser.add_argument(
        '--retrieval_renumber_turns',
        action='store_true',
        help='Renumber selected history turns as turn 1, turn 2, ... in the prompt.',
    )
    return parser.parse_args()


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


def parse_model_output(raw, stop_markers=None):
    parse_source = truncate_at_stop_markers(raw, stop_markers)
    try:
        result = ast.literal_eval(parse_source)
    except Exception as parse_exc:
        result = extract_json_from_markdown(parse_source)
        if result is None:
            raise ValueError(f"Failed to parse model output: {parse_exc}") from parse_exc

    if not isinstance(result, dict):
        raise ValueError(f"Parsed model output is not a dict: {type(result).__name__}")
    return result


def parse_candidates_value(candidates):
    if isinstance(candidates, list):
        return candidates
    if isinstance(candidates, str):
        parsed = ast.literal_eval(candidates)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("candidates must be a list.")


def get_schema_argument_keys(api_schema, plan):
    schema_entry = api_schema.get(plan)
    if isinstance(schema_entry, list):
        return set(schema_entry)
    if isinstance(schema_entry, dict):
        for field_name in ("arguments", "parameters"):
            value = schema_entry.get(field_name)
            if isinstance(value, dict):
                return set(value.keys())
            if isinstance(value, list):
                return set(value)
    return set()


def is_schema_valid(result, candidates, api_schema):
    if not isinstance(result, dict):
        return False

    plan = result.get("plan")
    if plan is None or plan == "None":
        return False

    try:
        candidate_plans = parse_candidates_value(candidates)
    except Exception:
        return False

    if plan not in candidate_plans:
        return False
    if plan not in api_schema:
        return False

    arguments = result.get("arguments")
    if not isinstance(arguments, dict):
        return False

    allowed_keys = get_schema_argument_keys(api_schema, plan)
    return set(arguments.keys()).issubset(allowed_keys)


def sanitize_model_slug(model_name):
    lower_name = model_name.lower()
    if "qwen3" in lower_name:
        return "qwen3"
    if "qwen2.5" in lower_name or "qwen25" in lower_name:
        return "qwen25"
    if "phi-4" in lower_name or "phi4" in lower_name:
        return "phi4"
    if "llama" in lower_name:
        return "llama3"
    if "gemma" in lower_name:
        return "gemma"
    sanitized = re.sub(r"[^a-z0-9]+", "-", model_name.split("/")[-1].lower()).strip("-")
    return sanitized or "model"


def infer_generic_ollama_model_name(profile_name, prompt_model_name, train_type, prefix):
    model_slug = sanitize_model_slug(prompt_model_name)
    return f"{model_slug}-{profile_name}-{train_type}-{prefix}:latest"


def build_generic_test_type_configs():
    configs = {}
    for profile_name, profile_config in GENERIC_PROMPT_PROFILES.items():
        prompt_model_name = profile_config["prompt_model_name"]
        prefix = profile_config["prefix"]
        for train_type in ("history", "rewrite"):
            config = {
                "model_name": infer_generic_ollama_model_name(
                    profile_name,
                    prompt_model_name,
                    train_type,
                    prefix,
                ),
                "prompt_renderer": "chat_template",
                "prompt_model_name": prompt_model_name,
                "prompt_mode": train_type,
                "chat_template_fallback": "simple",
            }
            if profile_config.get("stop"):
                config["stop"] = profile_config["stop"]
            if train_type == "history":
                configs[f"history-{profile_name}"] = config
                configs[f"base-{profile_name}"] = config
            else:
                configs[f"rewrite-{profile_name}"] = config
    return configs


def render_chat_template(tokenizer, messages, add_generation_prompt):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_simple_inference_prompt(system_msg, user_content):
    return (
        f"System:\n{system_msg}\n\n"
        f"User:\n{user_content}\n\n"
        "Assistant:\n"
    )


def build_generic_prompt_fields(example, api_str, prompt_mode):
    if prompt_mode in ("base", "history"):
        system_msg = GENERIC_SYSTEM_HISTORY_PROMPT.format(tools=api_str)
        user_content = (
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
    elif prompt_mode == "rewrite":
        system_msg = GENERIC_SYSTEM_REWRITE_PROMPT.format(tools=api_str)
        user_content = f"User Query: {example['rewrited_query']}"
    else:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    return system_msg, user_content


def render_generic_inference_prompt(
    example,
    api_str,
    tokenizer,
    prompt_mode,
    chat_template_fallback,
):
    system_msg, user_content = build_generic_prompt_fields(example, api_str, prompt_mode)
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
        )

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use simple fallback only for "
            "models that were trained with --chat_template_fallback simple."
        )
    return render_simple_inference_prompt(system_msg, user_content)


def load_prompt_tokenizer(config):
    from transformers import AutoTokenizer

    prompt_model_name = config["prompt_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        prompt_model_name,
        trust_remote_code=True,
    )
    return tokenizer


def truncate_at_stop_markers(text, stop_markers=None):
    if not stop_markers or not isinstance(text, str):
        return text

    cut_idx = len(text)
    for marker in stop_markers:
        if not marker:
            continue
        marker_idx = text.find(marker)
        if marker_idx != -1:
            cut_idx = min(cut_idx, marker_idx)

    return text[:cut_idx].strip()


# Ollama API 호출 함수
#def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11434'):
def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11435', stop=None, temperature=0.0): # qwen3
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": 512,
        },
        "stream": False
    }
    if stop:
        payload["options"]["stop"] = stop

    response = requests.post(
        f"{host}/api/generate",
        json=payload,
    )
    
    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        raise Exception(f"API request failed: {response.text}")


def get_generation_temperatures(temperature, schema_resample, max_resample_attempts):
    if schema_resample:
        return SCHEMA_RESAMPLE_TEMPERATURES[:max_resample_attempts]
    return [temperature]


def generate_with_optional_schema_resampling(
    prompt,
    model_name,
    host,
    stop,
    temperature,
    candidates,
    api_schema,
    schema_resample,
    max_resample_attempts,
):
    temperatures = get_generation_temperatures(
        temperature=temperature,
        schema_resample=schema_resample,
        max_resample_attempts=max_resample_attempts,
    )
    raw = ""
    result = {}
    parse_error = ""
    schema_valid = False
    attempts = 0

    for attempts, attempt_temperature in enumerate(temperatures, start=1):
        raw = generate_text(
            prompt,
            model=model_name,
            host=host,
            stop=stop,
            temperature=attempt_temperature,
        )
        try:
            result = parse_model_output(raw, stop_markers=stop)
            parse_error = ""
        except Exception as exc:
            result = {"error": str(exc)}
            parse_error = str(exc)
            schema_valid = False
        else:
            schema_valid = is_schema_valid(result, candidates, api_schema)

        if schema_valid or not schema_resample:
            break

    return result, raw, parse_error, attempts, schema_valid


def parse_ground_truth(stranswer):
    gt = ast.literal_eval(stranswer)
    if type(gt) == str:
        gt = ast.literal_eval(gt)
    return gt


def evaluate_example(
    ex,
    test_key,
    file_name,
    model_name,
    host,
    stop,
    temperature,
    api_schema,
    schema_resample,
    max_resample_attempts,
):
    prompt = ex["strprompt"]
    raw = ""
    gt = {}
    parse_error = ""
    error_message = None
    result = {}
    resample_attempts = 0
    schema_valid = False

    try:
        result, raw, parse_error, resample_attempts, schema_valid = (
            generate_with_optional_schema_resampling(
                prompt=prompt,
                model_name=model_name,
                host=host,
                stop=stop,
                temperature=temperature,
                candidates=ex.get("candidates"),
                api_schema=api_schema,
                schema_resample=schema_resample,
                max_resample_attempts=max_resample_attempts,
            )
        )

        gt = parse_ground_truth(ex["stranswer"])

        plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
        arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
        all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
    except Exception as e:
        result = {"error": str(e)}
        if not parse_error:
            parse_error = str(e)
        error_message = f"Error: {e}, {raw}"
        plan_res = "fail"
        arg_res = "fail"
        all_res = "fail"

    row = {
        "test_key":             test_key,
        "conversation_history": ex.get("conversation_history"),
        "original_conversation_history": ex.get("original_conversation_history"),
        "history_selection":    ex.get("history_selection"),
        "selected_turn_ids":    ex.get("selected_turn_ids"),
        "retrieval_scores":     ex.get("retrieval_scores"),
        "original_turn_count":  ex.get("original_turn_count"),
        "query":                ex.get("query"),
        "rewrited_query":       ex.get("rewrited_query"),
        "candidates":           ex.get("candidates"),
        "raw_generation":       raw,
        "generation":           result,
        "parse_error":          parse_error,
        "gt":                   gt,
        "plan":                 plan_res,
        "arguments":            arg_res,
        "all":                  all_res,
        "resample_attempts":    resample_attempts,
        "schema_valid":         schema_valid,
        "file":                 file_name,
        "turn":                 extract_turn_from_filename(file_name)
    }
    return row, error_message


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


def evaluate_multi_example(
    row_idx,
    ex,
    test_key,
    file_name,
    model_name,
    host,
    stop,
    temperature,
    api_schema,
    schema_resample,
    max_resample_attempts,
):
    row, error_message = evaluate_example(
        ex=ex,
        test_key=test_key,
        file_name=file_name,
        model_name=model_name,
        host=host,
        stop=stop,
        temperature=temperature,
        api_schema=api_schema,
        schema_resample=schema_resample,
        max_resample_attempts=max_resample_attempts,
    )
    return row_idx, row, error_message


def print_dry_run_examples(proc, test_key, file_path, limit):
    file_name = os.path.basename(file_path)
    if limit < 0:
        raise ValueError("--dry_run_limit must be >= 0.")

    preview_count = len(proc) if limit == 0 else min(limit, len(proc))
    print(
        f"\n[dry_run] split={test_key} file={file_name} "
        f"showing={preview_count}/{len(proc)}"
    )
    for row_idx, ex in enumerate(proc[:preview_count]):
        print("\n" + "=" * 100)
        print(f"[dry_run] row={row_idx}")
        print(f"query: {ex.get('query')}")
        print(f"rewrited_query: {ex.get('rewrited_query')}")
        print(f"history_selection: {ex.get('history_selection')}")
        print(f"original_turn_count: {ex.get('original_turn_count')}")
        print(f"selected_turn_ids: {ex.get('selected_turn_ids')}")
        print(f"retrieval_scores: {ex.get('retrieval_scores')}")
        print("\n[original_conversation_history]")
        print(ex.get("original_conversation_history"))
        print("\n[selected_conversation_history]")
        print(ex.get("conversation_history"))
        print("\n[rendered_prompt]")
        print(ex.get("strprompt"))
    print("\n" + "=" * 100)
    print("[dry_run] skipped Ollama API calls for this file.")


"""
python3 /home/hj153lee/RMA/ollama_inference_multi.py \
--t new-base-qwen3 \
--test_key base,complex \
--o /home/hj153lee/RMA/datasets/result/260406-ablation/qwen3-new-base.tsv \
--host http://localhost:21435
"""
def main(out_file):      
    if args.num_parallel < 1:
        raise ValueError("--multi/--num_parallel must be >= 1.")
    if args.schema_resample and not 1 <= args.max_resample_attempts <= len(SCHEMA_RESAMPLE_TEMPERATURES):
        raise ValueError(
            "--max_resample_attempts must be between 1 and "
            f"{len(SCHEMA_RESAMPLE_TEMPERATURES)} when --schema_resample is set."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is required to load eval TSV files."
        ) from exc

    #apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)    
    sft_apis = read_apis("apis/simple_api.json", simple=True)    
    out_path = Path(out_file) if out_file else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    elif not args.dry_run:
        raise ValueError("--o is required unless --dry_run is set.")
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
        # "base-qwen2.5-base": {
        #     "model_name": "qwen2.5-base:latest",
        #     "prompt_template": ZERO_HISTORY_INFERENCE_QWEN25,
        #     "prompt_mode": "base",
        # },
        # "rewrite-qwen2.5-base": {
        #     "model_name": "qwen2.5-base:latest",
        #     "prompt_template": ZERO_REWRITE_INFERENCE_QWEN25,
        #     "prompt_mode": "rewrite",
        # },
        # "base-phi4-base": {
        #     "model_name": "phi4-base:latest",
        #     "prompt_template": ZERO_HISTORY_INFERENCE_PHI4,
        #     "prompt_mode": "base",
        # },
        # "rewrite-phi4-base": {
        #     "model_name": "phi4-base:latest",
        #     "prompt_template": ZERO_REWRITE_INFERENCE_PHI4,
        #     "prompt_mode": "rewrite",
        # },
        # "base-llama3-base": {
        #     "model_name": "llama-base:latest",
        #     "prompt_template": ZERO_HISTORY_INFERENCE_LLAMA,
        #     "prompt_mode": "base",
        # },
        # "rewrite-llama3-base": {
        #     "model_name": "llama-base:latest",
        #     "prompt_template": ZERO_REWRITE_INFERENCE_LLAMA,
        #     "prompt_mode": "rewrite",
        #},
        # "base-qwen3-base": {
        #     "model_name": "qwen3-base:latest",
        #     "prompt_template": ZERO_HISTORY_INFERENCE_QWEN3,
        #     "prompt_mode": "base",
        # },
        # "rewrite-qwen3-base": {
        #     "model_name": "qwen3-base:latest",
        #     "prompt_template": ZERO_REWRITE_INFERENCE_QWEN3,
        #     "prompt_mode": "rewrite",
        # },        
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
            "prompt_template": SFT_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        # "new-base-qwen2.5": {
        #     "model_name": "qwen25-qwen-history-all_linear:latest",
        #     "prompt_template": ZERO_HISTORY_INFERENCE_QWEN25,
        #     "prompt_mode": "base",
        # }                 
    }
    test_type_config.update(build_generic_test_type_configs())

    test_type = args.t
    config = test_type_config.get(test_type)
    if config is None:
        valid_test_types = ", ".join(sorted(test_type_config.keys()))
        raise ValueError(f"Invalid test type: {test_type}. Available test types: {valid_test_types}")

    model_name = args.model or config["model_name"]
    prompt_mode = config["prompt_mode"]
    if args.history_selection != "full" and prompt_mode not in ("base", "history"):
        raise ValueError(
            "--history_selection can only be used with base/history prompt modes. "
            f"Current prompt_mode={prompt_mode}."
        )

    history_selector = None
    if args.history_selection != "full":
        if args.history_top_k < 0:
            raise ValueError("--history_top_k must be >= 0.")
        history_selector = build_history_selector(
            strategy=args.history_selection,
            top_k=args.history_top_k,
            retrieval_model_name=args.retrieval_model_name,
            retrieval_device=args.retrieval_device,
            retrieval_max_length=args.retrieval_max_length,
            retrieval_query_prefix=args.retrieval_query_prefix,
            retrieval_document_prefix=args.retrieval_document_prefix,
            retrieval_recency_lambda=args.retrieval_recency_lambda,
            retrieval_order=args.retrieval_order,
            retrieval_renumber_turns=args.retrieval_renumber_turns,
        )

    prompt_tokenizer = None
    if config.get("prompt_renderer") == "chat_template":
        prompt_tokenizer = load_prompt_tokenizer(config)

    
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
        'humanvalid': [
            'datasets/tc/human-eval/it2_nonNR_tc.tsv',
            'datasets/tc/human-eval/it3_complex_1_tc.tsv',
            'datasets/tc/human-eval/it3_nonNR_tc.tsv',
            'datasets/tc/human-eval/it4_complex_1_tc.tsv',
            'datasets/tc/human-eval/it4_complex_2_tc.tsv',            
            'datasets/tc/human-eval/it4_nonNR_tc.tsv',
            'datasets/tc/human-eval/it5_complex_1_tc.tsv',
            'datasets/tc/human-eval/it5_complex_2_tc.tsv',
            'datasets/tc/human-eval/it5_complex_3_tc.tsv',
            'datasets/tc/human-eval/it5_nonNR_tc.tsv',
        ],
        'test':[
            'datasets/tc/scale/it2_nonNR_tc.tsv',
            'datasets/tc/scale/it3_complex_1_tc.tsv',
            'datasets/tc/scale/it3_nonNR_tc.tsv',
            'datasets/tc/scale/it4_complex_1_tc.tsv',
            'datasets/tc/scale/it4_complex_2_tc.tsv',
            'datasets/tc/scale/it4_nonNR_tc.tsv',                        
            'datasets/tc/scale/it5_complex_1_tc.tsv',
            'datasets/tc/scale/it5_complex_2_tc.tsv',
            'datasets/tc/scale/it5_complex_3_tc.tsv',     
            'datasets/tc/scale/it5_nonNR_tc.tsv',  
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_1_tc.tsv',
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_2_tc.tsv',
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_3_tc.tsv',            
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_4_tc.tsv',
            'datasets/tc/scale_turn6_refswap_backfill/it6_nonNR_tc.tsv',
        ],
        'extended': [
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_1_tc.tsv',
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_2_tc.tsv',
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_3_tc.tsv',            
            'datasets/tc/scale_turn6_refswap_backfill/it6_complex_4_tc.tsv',
            'datasets/tc/scale_turn6_refswap_backfill/it6_nonNR_tc.tsv',
        ],
        'base': [                                      
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
    if args.num_parallel > 1:
        print(f"num_parallel: {args.num_parallel}")
    print(f"schema_resample: {args.schema_resample}")
    if args.schema_resample:
        print(
            "schema_resample_temperatures: "
            f"{SCHEMA_RESAMPLE_TEMPERATURES[:args.max_resample_attempts]}"
        )
    print(f"history_selection: {args.history_selection}")
    if args.history_selection != "full":
        print(
            "history_selection_config: "
            f"top_k={args.history_top_k}, "
            f"recency_lambda={args.retrieval_recency_lambda}, "
            f"order={args.retrieval_order}, "
            f"renumber_turns={args.retrieval_renumber_turns}, "
            f"model={args.retrieval_model_name}"
        )
    # 데이터 예시 전처리 함수
    def preprocess_example_it(example, apis, config, prompt_tokenizer=None, history_selector=None):
        api_str = ""
        #re_fmt  = {"plan": "str type tool", "arguments": {"key1": "value1"}}        
        for plan in ast.literal_eval(example["candidates"]):
            api_data = apis[plan].copy()
            api_str += f"{plan}: {api_data}\n"

        prompt_mode = config["prompt_mode"]
        prompt_example = example
        history_selection_name = "full"
        selected_turn_ids = []
        retrieval_scores = []
        original_turn_count = None
        if history_selector is not None and prompt_mode in ("base", "history"):
            selection_result = history_selector.select(
                conversation_history=example["conversation_history"],
                query=example["query"],
            )
            prompt_example = dict(example)
            prompt_example["conversation_history"] = selection_result.conversation_history
            history_selection_name = history_selector.strategy
            selected_turn_ids = selection_result.selected_turn_ids
            retrieval_scores = selection_result.selected_scores
            original_turn_count = selection_result.original_turn_count

        if config.get("prompt_renderer") == "chat_template":
            prompt = render_generic_inference_prompt(
                example=prompt_example,
                api_str=api_str,
                tokenizer=prompt_tokenizer,
                prompt_mode=prompt_mode,
                chat_template_fallback=config.get("chat_template_fallback", "simple"),
            )
        elif prompt_mode in ("base", "history"):
            prompt_template = config["prompt_template"]
            prompt = prompt_template.format(
                tools=api_str,
                #re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),
                conversation_history=prompt_example["conversation_history"],
                data=prompt_example["query"]
            )
        elif prompt_mode == "rewrite":
            prompt_template = config["prompt_template"]
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
            "conversation_history": prompt_example["conversation_history"],
            "original_conversation_history": example["conversation_history"],
            "history_selection": history_selection_name,
            "selected_turn_ids": json.dumps(selected_turn_ids, ensure_ascii=False),
            "retrieval_scores": json.dumps(retrieval_scores, ensure_ascii=False),
            "original_turn_count": original_turn_count,
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
            proc = [
                preprocess_example_it(
                    example,
                    apis=sft_apis,
                    config=config,
                    prompt_tokenizer=prompt_tokenizer,
                    history_selector=history_selector,
                )
                for example in ds
            ]
            # else:
            #     proc = ds.map(
            #         partial(preprocess_example_it, apis=apis, prompt_template=prompt_template, test_type=test_type)
            #     )
            # 평가
            if args.dry_run:
                print_dry_run_examples(
                    proc=proc,
                    test_key=test_key,
                    file_path=file_path,
                    limit=args.dry_run_limit,
                )
                continue

            print(proc[0]["strprompt"])
            #exit(0)
            file_results = []

            if args.num_parallel <= 1:
                for row_idx, ex in enumerate(
                    tqdm(proc, desc=f"Processing {test_key}/{os.path.basename(file_path)}")
                ):
                    row, error_message = evaluate_example(
                        ex=ex,
                        test_key=test_key,
                        file_name=os.path.basename(file_path),
                        model_name=model_name,
                        host=args.host,
                        stop=config.get("stop"),
                        temperature=args.temperature,
                        api_schema=sft_apis,
                        schema_resample=args.schema_resample,
                        max_resample_attempts=args.max_resample_attempts,
                    )
                    if error_message:
                        print(error_message)
                    file_results.append(row)
                    split_results[test_key].append(row)
            else:
                file_name = os.path.basename(file_path)
                rows_by_idx = {}
                error_messages = {}

                with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
                    futures = [
                        executor.submit(
                            evaluate_multi_example,
                            row_idx,
                            ex,
                            test_key,
                            file_name,
                            model_name,
                            args.host,
                            config.get("stop"),
                            args.temperature,
                            sft_apis,
                            args.schema_resample,
                            args.max_resample_attempts,
                        )
                        for row_idx, ex in enumerate(proc)
                    ]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Processing {test_key}/{file_name}",
                    ):
                        row_idx, row, error_message = future.result()
                        rows_by_idx[row_idx] = row
                        if error_message:
                            error_messages[row_idx] = error_message

                file_results = [rows_by_idx[row_idx] for row_idx in range(len(proc))]
                for row_idx in sorted(error_messages):
                    print(error_messages[row_idx])
                split_results[test_key].extend(file_results)

            df_file = pd.DataFrame(file_results)
            print_eval(df_file, title=f"{test_key}/{os.path.basename(file_path)}", test_type=model_name)
            all_results.extend(file_results)

    if args.dry_run:
        print("\n[dry_run] completed without calling Ollama or writing result TSV.")
        return

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
