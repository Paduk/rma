import json
import argparse
import random
import pdb
import argparse
import os
from openai import OpenAI
from utils import OpenAiGenerateResponse, GoogleGenerateResponse


def normalize_target_plans(raw_plans):
    if not raw_plans:
        return []
    return [plan.strip() for plan in raw_plans if plan and plan.strip()]


def format_single_turn_history(query, device_response, turn_idx=1):
    return [f"turn {turn_idx}: {query} -> {device_response}"]


def build_last_turn_only_s3_record(record):
    full_history = list(record.get("conversation_history", []))
    return {
        "conversation_history": [],
        "full_conversation_history": full_history,
        "query": record.get("query", ""),
        "device_response": record.get("device_response", ""),
    }


def build_last_turn_only_rewrite_history(record):
    query = record.get("query", "")
    device_response = record.get("device_response", "")
    return format_single_turn_history(query, device_response)


def parse_target_source_name(source_name):
    base = os.path.basename(source_name)
    parts = base.split("_")
    info = {"turn": None, "mode": None, "refered_turn": None}

    if parts and parts[0].startswith("it"):
        turn_text = parts[0][2:]
        if turn_text.isdigit():
            info["turn"] = turn_text

    if "complex" in parts:
        info["mode"] = "complex"
        idx = parts.index("complex")
        if idx + 1 < len(parts):
            ref_text = parts[idx + 1]
            if ref_text.isdigit():
                info["refered_turn"] = int(ref_text)
    elif "nonNR" in parts:
        info["mode"] = "nonNR"
    elif "NR" in parts:
        info["mode"] = "NR"

    return info


def get_target_turn(args=None):
    if args is not None and getattr(args, "target_turn", None) is not None:
        return str(args.target_turn)

    env_turn = os.environ.get("TURN")
    if env_turn:
        return str(env_turn)

    return None


def get_target_mode(args=None):
    if args is not None and getattr(args, "target_mode", None) is not None:
        return args.target_mode

    env_mode = os.environ.get("TARGET_MODE")
    if env_mode:
        return env_mode

    return None


def get_target_refered_turn(args=None):
    if args is not None and getattr(args, "refered_turn", None) is not None:
        return int(args.refered_turn)

    env_ref = os.environ.get("REFERED_TURN")
    if env_ref and env_ref.isdigit():
        return int(env_ref)

    return None


def read_target_plan_file(file_path, target_turn=None, target_mode=None, target_refered_turn=None):
    plan_to_count = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source_name = parts[0]
            source_meta = parse_target_source_name(source_name)
            if target_turn is not None and source_meta["turn"] != str(target_turn):
                continue
            if target_mode is not None and source_meta["mode"] != target_mode:
                continue
            if target_refered_turn is not None and source_meta["refered_turn"] != int(target_refered_turn):
                continue

            if len(parts) >= 3 and parts[-1].isdigit():
                plan = parts[-2]
                count = int(parts[-1])
            else:
                plan = parts[-1]
                count = None

            if plan not in plan_to_count:
                plan_to_count[plan] = count

    return plan_to_count


def get_target_plan_counts(args):
    plan_to_count = {}
    target_turn = get_target_turn(args)
    target_mode = get_target_mode(args)
    target_refered_turn = get_target_refered_turn(args)

    for plan in normalize_target_plans(getattr(args, "target_plans", None)):
        plan_to_count[plan] = getattr(args, "target_count", None)

    target_file = getattr(args, "target_file", None)
    if target_file:
        for plan, count in read_target_plan_file(
            target_file,
            target_turn=target_turn,
            target_mode=target_mode,
            target_refered_turn=target_refered_turn,
        ).items():
            if plan not in plan_to_count or plan_to_count[plan] is None:
                plan_to_count[plan] = count

    return plan_to_count


def get_target_plan_set(args):
    return set(get_target_plan_counts(args).keys())


def get_effective_target_count(args, plan=None, default=None):
    if getattr(args, "overgen_count", None) is not None:
        return args.overgen_count

    plan_to_count = get_target_plan_counts(args)
    if plan is not None and plan in plan_to_count and plan_to_count[plan] is not None:
        return plan_to_count[plan]

    if getattr(args, "target_count", None) is not None:
        return args.target_count
    return default


def is_targeted_run(args):
    return bool(getattr(args, "target_file", None) or getattr(args, "target_plans", None))


def get_complex_target_specs(args):
    target_file = getattr(args, "target_file", None)
    target_turn = get_target_turn(args)
    target_refered_turn = get_target_refered_turn(args)

    if not target_file:
        return []

    specs = []
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            source_meta = parse_target_source_name(parts[0])
            if source_meta["mode"] != "complex":
                continue
            if target_turn is not None and source_meta["turn"] != str(target_turn):
                continue
            if target_refered_turn is not None and source_meta["refered_turn"] != int(target_refered_turn):
                continue

            plan = parts[-2]
            count = int(parts[-1]) if parts[-1].isdigit() else None
            specs.append(
                {
                    "turn": source_meta["turn"],
                    "refered_turn": source_meta["refered_turn"],
                    "plan": plan,
                    "count": count,
                }
            )

    return specs


def read_api_entries(api_file):
    entries = []
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_predecessor_plan_counts(api_file, args):
    target_plan_counts = get_target_plan_counts(args)
    predecessor_plan_counts = {}

    for entry in read_api_entries(api_file):
        source_plan = entry["plan"]
        matched_targets = []
        for next_turn in entry.get("next_turn_plans", []):
            next_plan = next_turn.get("plan")
            if next_plan in target_plan_counts:
                matched_targets.append(next_plan)

        if not matched_targets:
            continue

        source_cap = 0
        for next_plan in matched_targets:
            effective_count = get_effective_target_count(args, plan=next_plan, default=1)
            if effective_count is None:
                effective_count = 1
            source_cap += effective_count

        predecessor_plan_counts[source_plan] = source_cap

    return predecessor_plan_counts


def get_complex_predecessor_plan_counts(api_file, args):
    complex_specs = get_complex_target_specs(args)
    if not complex_specs:
        return {}

    targets_by_ref = {}
    for spec in complex_specs:
        ref = spec["refered_turn"]
        plan = spec["plan"]
        targets_by_ref.setdefault(ref, {})[plan] = get_effective_target_count(args, plan=plan, default=1)

    predecessors_by_ref = {}
    for ref, target_plan_counts in targets_by_ref.items():
        predecessor_plan_counts = {}
        for entry in read_api_entries(api_file):
            source_plan = entry["plan"]
            matched_targets = []
            for next_turn in entry.get("next_turn_plans", []):
                next_plan = next_turn.get("plan")
                if next_plan in target_plan_counts:
                    matched_targets.append(next_plan)

            if not matched_targets:
                continue

            source_cap = 0
            for next_plan in matched_targets:
                effective_count = target_plan_counts.get(next_plan, 1)
                if effective_count is None:
                    effective_count = 1
                source_cap += effective_count

            predecessor_plan_counts[source_plan] = source_cap

        predecessors_by_ref[ref] = predecessor_plan_counts

    return predecessors_by_ref

def get_model_name(arg_model):
    if arg_model == 'o3':
        model_name = 'o3-mini'
    elif arg_model == 'gpt-5.1':
        model_name = 'gpt-5.1'
    elif arg_model == 'gpt-5-mini':
        model_name = 'gpt-5-mini'
    elif arg_model == 'gpt-5-nano':
        model_name = 'gpt-5-nano'
    elif arg_model == 'o4-mini':
        model_name = 'o4-mini'
    elif arg_model == 'gpt-4.1-2025-04-14':
        model_name = 'gpt-4.1-2025-04-14'
    elif arg_model == 'gpt-4o-mini-2024-07-18':
        model_name = 'gpt-4o-mini-2024-07-18'
    elif arg_model == 'gemini-2.5-flash':
        model_name = 'gemini-2.5-flash-preview-04-17'
    elif arg_model == 'gemini-2.0-flash-lite':
        model_name = 'gemini-2.0-flash-lite'
    elif arg_model == 'gemini-2.0-flash':
        model_name = 'gemini-2.0-flash'
    else:
        exit(0)

    if 'gemini' in model_name:
        generate_response = GoogleGenerateResponse(model_name=model_name)    
    elif model_name in ['o3-mini', 'o4-mini', 'gpt-4.1-2025-04-14', 'gpt-4o-mini-2024-07-18', 'gpt-5.1', 'gpt-5-mini', 'gpt-5-nano']:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None), base_url="https://api.openai.com/v1",)
        generate_response = OpenAiGenerateResponse(client=client, model=model_name, system_prompt="")

    return model_name, generate_response

def print_filter_status(message, keys="default", model_name="default"):
    with open(f"logs/{model_name}_filtering_status.txt", "a") as f:
        f.write(f"[{keys}] : {message}\n")

def print_data_cnt_per_plan(datas, keys="default", model_name="default"):
    plan_cnt = {}
    
    unique_index = set()
    for data in datas:   
        if "answer" in data:           
            plan = data["answer"]["plan"]                        
        else:
            plan = data["next_turn_plan"]
        plan_cnt[plan] = plan_cnt.get(plan, 0) + 1                
    
    print(f"title: {keys}")
    for plan in plan_cnt:
        print(f"{plan}: {plan_cnt[plan]}")

    print(f"plan_count: {len(plan_cnt)}")
    print(f"total_lens: {len(datas)}")
    with open(f"logs/{model_name}_gen_logs.txt", "a") as f:
        f.write(f"[{keys}] plan_count: {len(plan_cnt)}, total_len: {len(datas)}\n")

def get_arg_parse():
    parser = argparse.ArgumentParser(description="data integration")    
    parser.add_argument('--t', type=str, required=False, help='target_file')    
    parser.add_argument('--o', type=str, required=False, help='out_file')    
    parser.add_argument('--t1', type=str, required=False, help='target_file')
    parser.add_argument('--t2', type=str, required=False, help='target_file')
    parser.add_argument('--step', type=str, required=False, help='out_file')    
    parser.add_argument('--it', type=str, required=False, help='iteration_file')    
    parser.add_argument('--model', type=str, required=False, help='iteration_file')    
    parser.add_argument('--t_list', type=str, nargs='+', required=False, help='List of iteration files')
    parser.add_argument('--d', action='store_true', help='Enable debug mode')
    parser.add_argument('--s', type=str, required=False, help='start_file')
    parser.add_argument('--api', type=str, default="apis/api_v3.0.1.jsonl", help='사용자 이름')
    parser.add_argument('--test_key', type=str, default="", help='')
    parser.add_argument('--target-plans', type=str, nargs='+', required=False, help='Only process these plans')
    parser.add_argument('--target-count', type=int, required=False, help='Per-plan target count for picky generation')
    parser.add_argument('--overgen-count', type=int, required=False, help='Per-plan upper bound used during generation')
    parser.add_argument('--target-file', type=str, required=False, help='Text/TSV file containing plan deficiency rows')
    parser.add_argument('--target-turn', type=int, required=False, help='Filter target-file rows by turn number')
    parser.add_argument('--target-mode', type=str, required=False, help='Filter target-file rows by mode such as nonNR or complex')
    parser.add_argument('--refered-turn', type=int, required=False, help='Filter target-file rows by refered turn for complex data')
    parser.add_argument('--filter-model', type=str, required=False, help='Override model used by filtering scripts')
    parser.add_argument('--dry-run', action='store_true', help='Print target settings without generation changes')
    args = parser.parse_args()

    return args

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_len_apis(datas):
    len_plans = set()
    for data in datas:
        u_idx = data["unique_idx"]
        plan = u_idx.split('-')[-2]
        len_plans.add(plan)

    print(len(len_plans))

def read_apis(api_file):
    api_dict = {}
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                api_data = json.loads(line)
                api_data.pop("examples", None)
                api_data.pop("returns", None)
                api_data.pop("next_turn_plans", None)
                api_dict[api_data["plan"]] = api_data
    
    return api_dict

def read_simple_apis(api_file):
    with open(api_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)
    return api_data        

def save_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
