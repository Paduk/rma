import pandas as pd
import random
import json
import argparse
import pdb
from utils.frequently_used_tools import read_jsonl, save_jsonl, get_arg_parse

def analysis(dataset): # data count analysis
    nr_cnt_for_all = 0
    nr_for_this_plan = 0
    nr_cnt_per_plan = {}
    not_nr_cnt_per_plan = {}
    for data in dataset:
        u_idx = data["unique_idx"]
        this_plan = u_idx.split('-')[-2]
        this_idx = u_idx.split('-')[-1]
        
        plan = data["answer"]["plan"]
        if '_NR' in this_idx:
            nr_for_this_plan += 1
            nr_cnt_per_plan[plan] = nr_cnt_per_plan.get(plan, 0) + 1
        else:
            not_nr_cnt_per_plan[plan] = not_nr_cnt_per_plan.get(plan, 0) + 1
        
        if '_NR' in u_idx:
            nr_cnt_for_all += 1

    print(f"nr_cnt_for_all: {nr_cnt_for_all}")
    print(f"nr_for_this_plan: {nr_for_this_plan}")
    print(f"nr_cnt_per_plan: {nr_cnt_per_plan}")
    print(f"not_nr_cnt_per_plan: {not_nr_cnt_per_plan}")

def rewrited_query_dup_printer(filenames):
    all_data = []
    for filename in filenames:
        filename = filename.split(".")[0] + "_dedup.jsonl"
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                data["source_file"] = filename  # 파일명을 추가
                all_data.append(data)

    # 전체 데이터 개수 출력
    total_data_count = len(all_data)
    print("전체 데이터 개수:", total_data_count)

    # 2. 각 rewrited_query의 등장 횟수를 계산
    rewrited_query_counts = {}
    for item in all_data:
        query = item.get("rewrited_query")
        if query is None:
            continue
        rewrited_query_counts[query] = rewrited_query_counts.get(query, 0) + 1

    # 3. 중복(rewrited_query의 count가 2 이상인 경우)인 그룹만 선별
    duplicate_groups = {query: count for query, count in rewrited_query_counts.items() if count > 1}

    # 중복 데이터의 종류(중복 그룹)와 중복 데이터 총 개수 출력
    duplicate_group_count = len(duplicate_groups)
    # duplicate_groups에 포함된 모든 데이터 건수를 합산 (즉, 중복된 그룹에 해당하는 전체 데이터 수)
    duplicate_data_count = sum(duplicate_groups.values())

    print("중복된 rewrited_query 종류(그룹) 수:", duplicate_group_count)
    print("중복된 rewrited_query에 해당하는 전체 데이터 개수:", duplicate_data_count)

    # # 4. 중복된 rewrited_query에 해당하는 데이터 상세정보 출력
    # print("\n중복된 각 데이터 상세 정보:")
    # for item in all_data:
    #     query = item.get("rewrited_query")
    #     if query and rewrited_query_counts.get(query, 0) > 1:
    #         # answer 항목에 plan이 없을 경우를 대비하여 dict.get 메서드로 접근
    #         plan = item.get("answer", {}).get("plan")
    #         print(f"파일명: {item['source_file']}, rewrited_query: '{query}', answer.plan: {plan}")

def rewrited_query_dup_checker(filenames):
    # 파일의 각 줄마다 JSON 데이터를 읽어서 저장하며, 데이터에 파일명을 추가합니다.
    all_data = []
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                data["source_file"] = filename  # 어느 파일에서 왔는지 기록
                all_data.append(data)

    # 2. deduplication 처리: 기본은 rewrited_query를 기준으로 중복 제거합니다.
    # 단, unique_idx가 _NR로 끝나면 무조건 포함합니다.
    unique_records = {}  # rewrited_query를 키로 하는 중복 제거 대상
    nr_records = []      # unique_idx가 _NR로 끝나는 경우 모두 포함

    for data in all_data:
        u_idx = data.get("unique_idx", "")
        # unique_idx가 '_NR'로 끝나면 무조건 포함
        if u_idx.endswith("_NR"):
            nr_records.append(data)
            continue

        # rewrited_query가 없는 경우에도 그냥 dedup 없이 포함하거나, 별도 처리할 수 있음
        key = data.get("rewrited_query")
        if key is None:
            # rewrited_query가 없는 경우, 여기서는 unique_records에 임의의 키(예: id(data))를 사용하여 포함할 수 있음
            unique_records[id(data)] = data
        else:
            if key not in unique_records:
                unique_records[key] = data

    # 최종 데이터: deduplication한 결과와 unique_idx가 _NR인 데이터를 합칩니다.
    final_data = list(unique_records.values()) + nr_records

    # 통계 출력: 전체 데이터 개수, 최종 데이터 개수, 제거된 데이터 개수
    total_data_count = len(all_data)
    final_count = len(final_data)
    removed_count = total_data_count - final_count

    print("전체 데이터 개수:", total_data_count)
    print("최종 데이터 개수 (중복 제거 후):", final_count)
    print("제거된 중복 데이터 개수:", removed_count)

    # 3. 원본 파일명별 그룹화: final_data를 source_file을 기준으로 그룹화합니다.
    grouped_by_file = {}
    for data in final_data:
        source = data["source_file"]
        grouped_by_file.setdefault(source, []).append(data)

    # 4. 각 원본 파일명별로 별도의 JSONL 파일에 저장 (파일명 앞에 dedup_ 접두사 추가)
    for source_file, records in grouped_by_file.items():
        output_filename = f"{source_file.split('.')[0]}_dedup.jsonl"
        with open(output_filename, "w", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{source_file} 파일의 중복 제거 후 결과를 {output_filename}로 저장했습니다.")

def weighted_split_dataset(records, ratio=0.9, turn_idx=None):
    #random.shuffle(records)
    # 데이터가 1개인 경우 train에 할당하지 않고, 전체를 tc에 할당
    if len(records) == 1:
        return [], records
    # 그 외의 경우 일반적으로 9:1로 분할
    train_count = int(len(records) * ratio)    
    train_records = records[:train_count]
    tc_records = records[train_count:]

    if turn_idx == 2:
        weighted_count = 140
    elif turn_idx == 3:
        weighted_count = 30
    elif turn_idx == 4:
        weighted_count = 20
    elif turn_idx == 5:
        weighted_count = 10
    if len(train_records) > weighted_count: # train, tc 모두 plan 별로 최대 50개로 제한
        train_records = train_records[:weighted_count]
        tc_records = tc_records[:50] # train option1의 tc와 동일하게 하기 위함
        
    return train_records, tc_records

# 9:1 비율로 train과 tc 데이터셋 분할 함수
def split_dataset(records, ratio=0.9):
    random.shuffle(records)
    # 데이터가 1개인 경우 train에 할당하지 않고, 전체를 tc에 할당
    if len(records) == 1:
        return [], records
    # 그 외의 경우 일반적으로 9:1로 분할
    train_count = int(len(records) * ratio)    
    train_records = records[:train_count]
    tc_records = records[train_count:]

    if len(train_records) > 50: # train, tc 모두 plan 별로 최대 50개로 제한
        train_records = train_records[:50]
        tc_records = tc_records[:50]
        
    return train_records, tc_records


def data_spliter(datas, out_file):
    apis = read_jsonl("apis/api_v3.0.2.jsonl")
    apis_keys = set([api["plan"] for api in apis])
    nr_groups = {}       # 'NR' 포함된 경우
    non_nr_groups = {}   # 'NR' 미포함인 경우

    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  # unique_idx 형식이 예상과 다르면 건너뛰기
        plan = parts[-2]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        has_NR = 'NR' in parts[-1]
        
        if has_NR:            
            nr_groups.setdefault(plan, []).append(record)
        else:                        
            non_nr_groups.setdefault(plan, []).append(record)

    # NR 그룹 내 모든 plan별 분할 결과를 누적
    nr_train_all = []
    nr_tc_all = []
    for plan, records in nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio=0.5)
        nr_train_all.extend(train_records)
        nr_tc_all.extend(tc_records)
        #print(f"NR 그룹 - plan '{plan}': {len(train_records)} 건 train, {len(tc_records)} 건 tc")

    # NR 미포함 그룹 내 모든 plan별 분할 결과를 누적
    non_nr_train_all = []
    non_nr_tc_all = []
    for plan, records in non_nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio=0.5)
        non_nr_train_all.extend(train_records)
        non_nr_tc_all.extend(tc_records)
        print(f"NR 미포함 그룹 - plan '{plan}': {len(train_records)} 건 train, {len(tc_records)} 건 tc")
    
    # NR 그룹의 최종 파일 저장
    save_jsonl(f"datasets/NR_train_{out_file}.jsonl", nr_train_all)
    save_jsonl(f"datasets/NR_tc_{out_file}.jsonl", nr_tc_all)

    # NR 미포함 그룹의 최종 파일 저장 (파일명은 nonNR_ 접두어 사용)
    save_jsonl(f"datasets/nonNR_train_{out_file}.jsonl", non_nr_train_all)
    save_jsonl(f"datasets/nonNR_tc_{out_file}.jsonl", non_nr_tc_all)
    print(f"파일 각각 개수 non_nr_train: {len(non_nr_train_all)}건, non_nr_tc: {len(non_nr_tc_all)}건")
    print(f"파일 각각 개수 nr_train: {len(nr_train_all)}건, nr_tc: {len(nr_tc_all)}건")    
    print(f"파일 저장 완료: NR_train_{out_file}.jsonl, NR_tc_{out_file}.jsonl, nonNR_train_{out_file}.jsonl, nonNR_tc_{out_file}.jsonl")

def integrated_data_spliter(datas, prefix):    
    # load API plans once
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = {api["plan"] for api in apis}

    # group records by turn_idx
    datas_by_turn = {}
    for record in datas:
        turn_idx = record.get("turn_idx")
        if turn_idx is None:
            continue
        datas_by_turn.setdefault(turn_idx, []).append(record)

    # process each turn group separately
    for turn_idx, new_datas in datas_by_turn.items():
        groups = {}

        # assemble candidate lists per record
        for record in new_datas:
            unique_idx = record.get("unique_idx", "")
            parts = unique_idx.split('-')
            if len(parts) < 2:
                continue
            plan = parts[-2]

            N = 5
            # sample other plans + the correct one
            others = [k for k in apis_keys if k != plan]
            sampled = random.sample(others, N - 1)
            sampled.append(plan)
            random.shuffle(sampled)
            record["candidates"] = sampled

            groups.setdefault(plan, []).append(record)

        # split into train/tc for this turn
                
        weighted_train_all, weighted_tc_all = [], []
        for plan, records in groups.items():
            random.shuffle(records) # 향후 Record shuffle 원위치
            weighted_train_records, weighted_tc_records = weighted_split_dataset(records, ratio=0.5, turn_idx=record["turn_idx"])            
            weighted_train_all.extend(weighted_train_records)
            weighted_tc_all.extend(weighted_tc_records)
            print(f"[Turn {turn_idx}] plan '{plan}': {len(train_records)} train, {len(tc_records)} tc")
            print(f"[Turn {turn_idx}] plan '{plan}': {len(weighted_train_records)} weighted train, {len(weighted_tc_records)} weighted tc")

        weighted_train_path = f"datasets/{turn_idx}_weighted_train.tsv"
        weighted_tc_path = f"datasets/{turn_idx}_weighted_tc.tsv"
        pd.DataFrame(weighted_train_all).to_csv(weighted_train_path, sep="\t", index=False)
        pd.DataFrame(weighted_tc_all).to_csv(weighted_tc_path, sep="\t", index=False)
        print(f"[Turn {turn_idx}] saved train: {len(weighted_train_all)} rows -> {weighted_train_path}")
        print(f"[Turn {turn_idx}] saved tc:    {len(weighted_tc_all)} rows -> {weighted_tc_path}")

def data_spliter_pd(datas, prefix, ratio):
    # apis를 읽어와 각 api의 plan 값들을 set으로 추출
    apis = read_jsonl("apis/api_v3.0.2.jsonl")
    apis_keys = set(api["plan"] for api in apis)
    
    nr_groups = {}       # 'NR' 포함된 경우
    non_nr_groups = {}   # 'NR' 미포함인 경우

    # data 분할 및 candidates 추가
    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  # unique_idx 형식이 예상과 다르면 건너뛰기
        plan = parts[-2]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        has_NR = 'NR' in parts[-1]
        if has_NR:
            nr_groups.setdefault(plan, []).append(record)
        else:
            # new feature: rewrite == query 제외
            if record["rewrited_query"] == record["query"]:                
                continue
            non_nr_groups.setdefault(plan, []).append(record)

    total_plan = set()
    # NR 그룹 내 각 plan별 분할 결과를 누적
    nr_train_all = []
    nr_tc_all = []
    nr_plan_cnt = 0
    for plan, records in nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio)
        nr_train_all.extend(train_records)
        nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        nr_plan_cnt += 1
        print(f"NR 그룹 - plan '{plan}': {len(train_records)} 건 train, {len(tc_records)} 건 tc")

    # NR 미포함 그룹 내 각 plan별 분할 결과를 누적
    non_nr_train_all = []
    non_nr_tc_all = []
    non_nr_plan_cnt = 0
    for plan, records in non_nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio)
        non_nr_train_all.extend(train_records)
        non_nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        print(f"NR 미포함 그룹 - plan '{plan}': {len(train_records)} 건 train, {len(tc_records)} 건 tc")
        non_nr_plan_cnt += 1
    
    # pandas DataFrame을 이용해 TSV 파일로 저장 (key는 column 명으로 저장)
    pd.DataFrame(nr_train_all).to_csv(f"datasets/train/{prefix}_NR_train.tsv", sep="\t", index=False)
    pd.DataFrame(nr_tc_all).to_csv(f"datasets/tc/{prefix}_NR_tc.tsv", sep="\t", index=False)
    pd.DataFrame(non_nr_train_all).to_csv(f"datasets/train/{prefix}_nonNR_train.tsv", sep="\t", index=False)
    pd.DataFrame(non_nr_tc_all).to_csv(f"datasets/tc/{prefix}_nonNR_tc.tsv", sep="\t", index=False)

    print(f"파일 저장 완료: prefix: {prefix}")
    print(f"Plan 개수 nr: {nr_plan_cnt}개, non_nr: {non_nr_plan_cnt}개, 총 {len(total_plan)}개")
    print(f"파일 각각 개수 non_nr_train: {len(non_nr_train_all)}건, non_nr_tc: {len(non_nr_tc_all)}건")
    print(f"파일 각각 개수 nr_train: {len(nr_train_all)}건, nr_tc: {len(nr_tc_all)}건")       

def data_convert_to_tc(datas, prefix):
    # apis를 읽어와 각 api의 plan 값들을 set으로 추출
    apis = read_jsonl("apis/api_v3.0.2.jsonl")
    apis_keys = set(api["plan"] for api in apis)
    
    nr_groups = {}       # 'NR' 포함된 경우
    non_nr_groups = {}   # 'NR' 미포함인 경우

    # data 분할 및 candidates 추가
    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  # unique_idx 형식이 예상과 다르면 건너뛰기
        plan = parts[-2]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        has_NR = 'NR' in parts[-1]
        if has_NR:
            nr_groups.setdefault(plan, []).append(record)
        else:
            # new feature: rewrite == query 제외
            if record["rewrited_query"] == record["query"]:                
                continue
            non_nr_groups.setdefault(plan, []).append(record)

    total_plan = set()    
    nr_tc_all = []
    nr_plan_cnt = 0
    for plan, records in nr_groups.items():
        tc_records = records        
        nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        nr_plan_cnt += 1        

    # NR 미포함 그룹 내 각 plan별 분할 결과를 누적    
    non_nr_tc_all = []
    non_nr_plan_cnt = 0
    for plan, records in non_nr_groups.items():
        tc_records = records
        non_nr_tc_all.extend(tc_records)
        total_plan.add(plan)        
        non_nr_plan_cnt += 1
    
    # pandas DataFrame을 이용해 TSV 파일로 저장 (key는 column 명으로 저장)    
    pd.DataFrame(nr_tc_all).to_csv(f"datasets/tc/{prefix}_NR_tc.tsv", sep="\t", index=False)    
    pd.DataFrame(non_nr_tc_all).to_csv(f"datasets/tc/{prefix}_nonNR_tc.tsv", sep="\t", index=False)

    print(f"파일 저장 완료: prefix: {prefix}")
    print(f"Plan 개수 nr: {nr_plan_cnt}개, non_nr: {non_nr_plan_cnt}개, 총 {len(total_plan)}개")
    print(f"파일 각각 개수 non_nr_tc: {len(non_nr_tc_all)}건")
    print(f"파일 각각 개수 nr_tc: {len(nr_tc_all)}건")       

def droid_convert_to_tc(datas, prefix):
    # apis를 읽어와 각 api의 plan 값들을 set으로 추출
    apis = read_jsonl("apis/droidcall_apis.jsonl")
    apis_keys = set(api["name"] for api in apis)
    
    nr_groups = {}       # 'NR' 포함된 경우
    non_nr_groups = {}   # 'NR' 미포함인 경우

    # data 분할 및 candidates 추가
    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  # unique_idx 형식이 예상과 다르면 건너뛰기
        plan = parts[0]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys
        nr_groups.setdefault(plan, []).append(record)

    total_plan = set()    
    nr_tc_all = []
    nr_plan_cnt = 0
    for plan, records in nr_groups.items():
        tc_records = records        
        nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        nr_plan_cnt += 1        
    
    # pandas DataFrame을 이용해 TSV 파일로 저장 (key는 column 명으로 저장)    
    pd.DataFrame(nr_tc_all).to_csv(f"datasets/tc/droidcall_tc.tsv", sep="\t", index=False)    

    print(f"Plan 개수 nr: {nr_plan_cnt}개, 총 {len(total_plan)}개")
    print(f"파일 각각 개수 nr_tc: {len(nr_tc_all)}건")       

def convert(records, ratio=0.8):
    #random.shuffle(records)
    # 데이터가 1개인 경우 train에 할당하지 않고, 전체를 tc에 할당
    if len(records) == 1:
        return [], records
    # 그 외의 경우 일반적으로 9:1로 분할
    train_count = int(len(records) * ratio)    
    train_records = records[:train_count]
    tc_records = records[train_count:]

    if len(train_records) > 50: # train, tc 모두 plan 별로 최대 50개로 제한
        train_records = train_records[:50]
        tc_records = tc_records[:50]    
        
    return train_records, tc_records


def data_converter_pd(datas, prefix):
    # apis를 읽어와 각 api의 plan 값들을 set으로 추출
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = set(api["plan"] for api in apis)
    
    all_data_cnt = 0
    target_data_cnt = 0
    non_nr_groups = {}
    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  # unique_idx 형식이 예상과 다르면 건너뛰기
        plan = parts[-2]        
        all_data_cnt += 1
        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys
            
        if record["rewrited_query"] == record["query"]:                            
            continue
        
        target_data_cnt += 1        
        non_nr_groups.setdefault(plan, []).append(record)

    total_plan = set()
        # NR 미포함 그룹 내 각 plan별 분할 결과를 누적
    non_nr_train_all = []
    non_nr_tc_all = []
    non_nr_plan_cnt = 0
    for plan, records in non_nr_groups.items():
        train_records, tc_records = convert(records) # max 50 per plan
        non_nr_train_all.extend(train_records)
        non_nr_tc_all.extend(tc_records)
        total_plan.add(plan)        
        non_nr_plan_cnt += 1
    
    pd.DataFrame(non_nr_train_all).to_csv(f"testset/{prefix}_train.tsv", sep="\t", index=False)
    pd.DataFrame(non_nr_tc_all).to_csv(f"testset/{prefix}_tc.tsv", sep="\t", index=False)

    print(f"파일 저장 완료: prefix: {prefix}")
    print(f"count: {all_data_cnt}개, target count: {target_data_cnt}개")
    
    print(f"파일 각각 개수 non_nr_train: {len(non_nr_train_all)}건, non_nr_tc: {len(non_nr_tc_all)}건")    


def data_convert_to_tc_by_refered_turn(datas, prefix):
    # TC 만 생성, Train X

    # 1) API plan 목록 가져오기
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = set(api["plan"] for api in apis)

    all_data_cnt = 0
    target_data_cnt = 0
    ref_groups = {}

    #target_plans = ['ACTION_CREATE_DOCUMENT', 'ACTION_EDIT_ALARM', 'ACTION_EDIT_CONTACT', 'ACTION_EDIT_DOCUMENT', 'ACTION_EDIT_VIDEO', 'ACTION_GET_RINGTONE', 'ACTION_INSERT_CONTACT', 'ACTION_INSERT_EVENT', 'ACTION_NAVIGATE_TO_LOCATION', 'ACTION_OPEN_CONTENT', 'ACTION_SHOW_ALARMS', 'ACTION_VIEW_CONTACT', 'ACTION_VIEW_EVENT', 'ACTION_VIEW_RECENT_APPS', 'ACTION_VIEW_SENT_EMAILS', 'ACTION_VIEW_SENT_MESSAGES', 'ACTION_VIEW_WEB_HISTORY', 'dial', 'play_video', 'search_location', 'send_email', 'send_message']
    # 위의 타겟들은 MultiTurn 특징 보기 좋은 플랜들

    # 2) 전처리 및 refered_turn별 1차 그룹핑
    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue
        plan = parts[-2]
        all_data_cnt += 1

        # candidates 샘플링
        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        # rewrited == query 는 건너뛰기
        if record["rewrited_query"] == record["query"]:
            continue
        target_data_cnt += 1

        # refered_turn 값으로 그룹핑
        ref = record.get("refered_turn", "NONE")
        ref_groups.setdefault(ref, []).append(record)

    # 3) refered_turn별로 plan 그룹 → 변환 → 누적 → 단일 저장
    for ref, recs in ref_groups.items():
        if ref is None or ref == "NONE":
            continue
        
        ref_train_all = []
        ref_tc_all    = []

        # plan별 그룹핑
        plan_groups = {}
        for r in recs:
            plan = r["unique_idx"].split('-')[-2]
            plan_groups.setdefault(plan, []).append(r)

        # 각 plan 그룹마다 convert 결과만 모아서 누적
        for plan, grp in plan_groups.items():
            # if plan not in target_plans:
            #     print(f"target plan: {plan} continue")
            #     continue
            #train_records, tc_records = convert(grp)
            tc_records = grp
            #ref_train_all.extend(train_records)
            ref_tc_all.extend(tc_records)

        # 한 번만 파일로 저장
        #train_path = f"testset/{prefix}_{ref}_train.tsv"
        tc_path    = f"datasets/tc/{prefix}_{ref}_tc.tsv"

        #pd.DataFrame(ref_train_all).to_csv(train_path, sep="\t", index=False)
        pd.DataFrame(ref_tc_all).to_csv(tc_path,    sep="\t", index=False)

        #print(f"[{ref}] 전체 Train: {len(ref_train_all)}건 → {train_path}")
        print(f"[{ref}] 전체 Test:  {len(ref_tc_all)}건 → {tc_path}")

    # 4) 전체 처리 결과 로그
    print(f"\n전체 레코드: {all_data_cnt}개, 대상 레코드: {target_data_cnt}개")
    print(f"생성된 refered_turn 그룹 수: {len(ref_groups)}개")


def __main__():
    random.seed(42)
    args = get_arg_parse()    
        
    # datasets = read_jsonl("datagen/it4_s1_int_dedup.jsonl")
    # analysis(datasets)
            
    if args.d: # data split
        target_file = args.t
        prefix = args.o         
        datas = read_jsonl(target_file)        
        
        droid_convert_to_tc(datas, "") # python3 dataset_spliter.py --d --t datasets/droidcall_gemini.jsonl --o ""
        #data_spliter_pd(datas, prefix, ratio=0.7)
        #data_convert_to_tc(datas, prefix)
        #data_convert_to_tc_by_refered_turn(datas, prefix)
        #integrated_data_spliter(datas, prefix)
    else:        
        filenames = args.t_list # ["datagen/it2_s1_int.jsonl", "datagen/it3_s1_int.jsonl", "datagen/it4_s1_int.jsonl"]#
        rewrited_query_dup_checker(filenames)

        filenames = args.t_list # ["datagen/it2_s1_int_dedup.jsonl", "datagen/it3_s1_int_dedup.jsonl", "datagen/it4_s1_int_dedup.jsonl"]#
        rewrited_query_dup_printer(filenames) # 검증

__main__()   
# TC
# data_convert_to_tc_by_refered_turn()
# 1. python3 dataset_spliter.py --d --t datasets/it5_s1_complex_filtered.jsonl --o it5_complex
# 2. python3 dataset_spliter.py --d --t datasets/it4_s1_complex_filtered.jsonl --o it4_complex
# 3. python3 dataset_spliter.py --d --t datasets/it3_s1_complex_filtered.jsonl --o it3_complex
# > history_modified
# 1. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it5_s1_complex_filtered_modified.jsonl --o it5_complex_history
# 2. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it4_s1_complex_filtered_modified.jsonl --o it4_complex_history
# 3. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it3_s1_complex_filtered_modified.jsonl --o it3_complex_history

# data_convert_to_tc()
# 4. python3 dataset_spliter.py --d --t datasets/it5_s1_various_filtered.jsonl --o it5_various
# 5. python3 dataset_spliter.py --d --t datasets/it4_s1_various_filtered.jsonl --o it4_various # (optional)
# 6. python3 dataset_spliter.py --d --t datasets/it3_s1_various_filtered.jsonl --o it3_various # (optional)
# data_spliter_pd()
# 7. python3 dataset_spliter.py --d --t datasets/it2_s1_filtered.jsonl --o it2
# 7. python3 dataset_spliter.py --d --t datasets/it3_s1_filtered.jsonl --o it3
# 7. python3 dataset_spliter.py --d --t datasets/it4_s1_filtered.jsonl --o it4
# 7. python3 dataset_spliter.py --d --t datasets/it5_s1_filtered.jsonl --o it5
# > history_modified
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it2_s1_filtered_modified.jsonl --o it2_history
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it3_s1_filtered_modified.jsonl --o it3_history
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it4_s1_filtered_modified.jsonl --o it4_history
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it5_s1_filtered_modified.jsonl --o it5_history