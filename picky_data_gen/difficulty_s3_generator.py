import pdb
import json
import random
from utils.frequently_used_tools import (
    get_arg_parse,
    get_complex_predecessor_plan_counts,
    get_complex_target_specs,
    get_effective_target_count,
    get_model_name,
    get_target_plan_set,
    is_targeted_run,
    read_jsonl,
)
from prompts import COMPLEX_MULTI_TURN_GENERATION_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import JsonExtractor

def chunks(lst, n):
    """리스트 lst를 n개씩 나눠 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_datas(datas):
    """
    datas: list of dicts, each with keys:
      - conversation_history: list of strings, each like "user_query -> device_response"
      - unique_idx: str of the form "PLAN1-xxx-PLAN2-yyy-…"
    """
    processed = []

    for record in datas:
        conv_hist = record.get('conversation_history', [])
        # 현재 턴: 이미 진행된 발화 수 + 1
        current_turn = len(conv_hist) + 1

        # unique_idx에서 짝수번째 요소가 각 턴의 plan
        tokens = record.get('unique_idx', '').split('-')
        plan_candidates = tokens[0::2]

        # “현재 턴보다 2턴 이상 이전” 플랜
        num_eligible = max(0, current_turn - 2)
        previous_plans = plan_candidates[:num_eligible]  # index 0은 가장 과거

        # 각 이전 플랜별로 query, device_response, next_turn_plan을 담은 새 레코드 생성
        for turn_idx, prev_plan in enumerate(previous_plans):
            # conversation_history의 동일 인덱스 항목을 꺼내서 ' -> '로 분리
            hist = conv_hist[turn_idx]
            if ' -> ' not in hist:
                # 분리 구분자가 없으면 건너뛰기
                continue
            query, device_response = hist.split(' -> ', 2)

            if query.startswith('turn'):
                query = query[8:] # removing turn string
            
            new_record = {} # record.copy()
            new_record['query'] = query.strip()
            new_record['device_response'] = device_response.strip()
            new_record['next_turn_plan'] = prev_plan
            c_uidx = record['unique_idx']             
            new_record['unique_idx'] = '-'.join(c_uidx.split('-')[:-2])
            new_record['refered_turn'] = turn_idx + 1
            # print(new_record['query'])
            # print(new_record['device_response'])
            # print(new_record['next_turn_plan'])            
            # print(new_record['unique_idx'])            
            # print()            
            processed.append(new_record)                

    return processed

def read_all_apis(api_file):
    api_dict = {}
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                api_data = json.loads(line)                
                api_dict[api_data["plan"]] = api_data
    
    return api_dict
    
if __name__ == "__main__":    
    args = get_arg_parse()
    targeted_run = is_targeted_run(args)
    complex_target_specs = get_complex_target_specs(args)
    target_plan_set = get_target_plan_set(args)
    predecessor_by_ref = get_complex_predecessor_plan_counts(args.api, args)
    refered_turn_to_plans = {}
    for spec in complex_target_specs:
        refered_turn_to_plans.setdefault(spec["refered_turn"], set()).add(spec["plan"])

    # 데이터 로드
    datas = read_jsonl(args.t)
    apis = read_all_apis(args.api)
    output_path = args.o
    # 전처리 실행
    target_datas = process_datas(datas)
    if targeted_run:
        filtered_target_datas = []
        for rec in target_datas:
            ref = rec.get("refered_turn")
            plan = rec.get("next_turn_plan")
            predecessor_map = predecessor_by_ref.get(ref, {})
            if ref in predecessor_by_ref and plan in predecessor_map:
                filtered_target_datas.append(rec)
        target_datas = filtered_target_datas

    grouped = {}
    for rec in target_datas:
        key = (rec.get('refered_turn'), rec.get('next_turn_plan'))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(rec)

    deduped = {}
    for key, items in grouped.items():
        seen = set()
        deduped[key] = []
        for item in items:
            uid = item['unique_idx']
            if uid not in seen:
                seen.add(uid)
                deduped[key].append(item)
    
    # for plan, records in deduped.items():
    #     print(f"Plan `{plan}`: {len(deduped[plan])} records")
    
    prompts = []
    prompt_template = COMPLEX_MULTI_TURN_GENERATION_PROMPT
    target_plan_cnt = {}
    available_refs = sorted({key[0] for key in deduped.keys()})
    for api_key, api_item in apis.items():
        plan_name = api_item["plan"]
        if api_item["arguments"] == {}:
            #print(f"API {api_key} has no arguments")
            continue
        ref_iter = predecessor_by_ref.items() if targeted_run else [(ref, {}) for ref in available_refs]
        for ref, predecessor_counts in ref_iter:
            if targeted_run and plan_name not in predecessor_counts:
                continue

            filtered = []
            for idx, resp in enumerate(deduped.get((ref, plan_name), [])):
                try:
                    filtered.append(resp)
                except KeyError:
                    continue

            if not filtered:
                continue

            random.shuffle(filtered)
            if targeted_run:
                source_limit = predecessor_counts.get(plan_name)
                if source_limit is not None:
                    filtered = filtered[:source_limit]

            allowed_next_plans = refered_turn_to_plans.get(ref, target_plan_set) if targeted_run else None

            for group in chunks(filtered, 5):
                for nt in api_item.get("next_turn_plans", []):
                    next_plan = nt["plan"]
                    if next_plan not in apis:
                        raise Exception(f"{next_plan}: not found in apis")
                    if targeted_run and next_plan not in allowed_next_plans:
                        continue

                    plan_data = apis[next_plan].copy()
                    plan_data["reason"] = nt.get("reason")
                    plan_data.pop("next_turn_plans", None)
                    description = json.dumps(plan_data, indent=2, ensure_ascii=False)

                    example = nt["example"].copy()
                    example.pop("next_turn_plan", None)
                    example["next_turn_plan"] = next_plan
                    example = json.dumps(example, indent=2, ensure_ascii=False)

                    target_key = (ref, next_plan)
                    if target_plan_cnt.get(target_key, 0) > 60:
                        continue
                    target_plan_cnt[target_key] = target_plan_cnt.get(target_key, 0) + 1

                    for g in group:
                        g["next_turn_plan"] = next_plan

                    prev_data = json.dumps(group, indent=2, ensure_ascii=False)
                    prompt = prompt_template.format(
                        previous_turn_data=prev_data,
                        description=description,
                        example=example
                    )
                    prompts.append(prompt)

    if targeted_run:
        print(f"complex_target_specs: {complex_target_specs}")
        print(f"predecessor_by_ref: {predecessor_by_ref}")
        print(f"target_plans: {sorted(target_plan_set)}")
    print(f"prompt_count: {len(prompts)}")

    if args.dry_run:
        print("dry-run enabled; skipping generation")
        raise SystemExit(0)
    
    filters = [JsonExtractor()]
    
    model_name, generate_response = get_model_name(args.model)

    # 6. 병렬 처리 함수 정의
    def process_prompt(p):
        resp = generate_response("", [p])
        out = []
        for flt in filters:
            out.extend(flt.filter(resp))
        return out

    #prompts = prompts[:10]
    # 7. ThreadPoolExecutor 로 동시 API 호출 및 결과 저장
    with open(output_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_prompt, p): p for p in prompts}
            for future in tqdm(as_completed(futures), total=len(prompts)):
                try:
                    results = future.result()
                    for item in results:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print("Error processing prompt:", e)
    
