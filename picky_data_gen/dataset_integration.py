import json
import pdb
import csv
import ast
from utils.frequently_used_tools import get_arg_parse, read_jsonl, print_data_cnt_per_plan, print_filter_status

file_paths = ['it2_s1.jsonl', 'it2_s2.jsonl', 'it2_s3.jsonl', 'it2_s4.jsonl']
o_file_paths = ['None', 'it2_s2_idx.jsonl', 'it2_s3_idx.jsonl', 'd_it3_s1.jsonl']

# query, rewrited_query 중복이 없는지 확인
# step1.jsonl 파일을 읽어 query를 key로, unique_idx를 value로 하는 매핑(dict) 생성

def differentiate_gen_data(plan1, plan2):
    for key in plan1:
        if plan1[key] != plan2[key]:
            print(f"Mismatch in {key}: {plan1[key]} vs {plan2[key]}")

def s2_integrate(start_file, end_file, output_file):    
    query_to_unique = {}
    start_total_cnt = start_suc_cnt = 0
    end_total_cnt = end_suc_cnt = 0
    start_plans = {}
    end_plans = {}
    with open(start_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                plan = data["answer"]["plan"]                
                start_plans[plan] = start_plans.get(plan, 0) + 1
                start_total_cnt += 1
                if 'rewrited_query' in data: # multi-turn                    
                    conv_history = [conv for conv in data.get('conversation_history')]
                    conv = "-".join(conv_history)                
                    query = data.get('rewrited_query')
                else: # single-turn                
                    conv = ''
                    query = data.get('query')
                    
                key = conv + query
                unique_idx = data.get('unique_idx')
                # if '_NR' in unique_idx:
                #     print(key)
                if key is not None and unique_idx is not None:
                    query_to_unique[key] = data
                    start_suc_cnt += 1
                else:
                    print(f"Missing query or unique_idx in line: {data}")
                    query_to_unique['NA'] = data

    with open(end_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.strip():
                try:
                    data = json.loads(line)
                    end_total_cnt += 1                    
                    plan = data["answer"]["plan"]
                    end_plans[plan] = end_plans.get(plan, 0) + 1                
                    ch = data.get('conversation_history')
                    if len(ch) > 0:                
                        conv_history = [conv for conv in data.get('conversation_history')]
                        conv = "-".join(conv_history)                                    
                    else: # single-turn
                        conv = ''
                    query = data.get('query')
                    
                    key = conv + query
                    if key in query_to_unique:
                        matched_data = query_to_unique[key]
                        data['unique_idx'] = matched_data['unique_idx']
                        if 'conversation_history' in matched_data:
                            data['conversation_history'] = matched_data['conversation_history']
                        else:
                            data['conversation_history'] = []

                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        end_suc_cnt += 1
                    else:
                        print(f'{key} not matched, query: {query}')                    
                        continue                    
                        #key 없다는건 LLM Generation 과정에서 뺴먹었을지도
                except Exception as e:
                    print(f"Error processing: {e}")
                    print(f"{data}")
                    continue
    
    #print(f"Total lines in {start_file}: {start_total_cnt}, Successful lines: {start_suc_cnt}")
    #print(f"Total lines in {end_file}: {end_total_cnt}, Successful lines: {end_suc_cnt}")
    #print(f"Start plans: {start_plans}")
    print(f"End plans: {end_plans}")                
    print(f"Len plans: {len(end_plans)}")
    #differentiate_gen_data(start_plans, end_plans)    

def s3_integrate(start_file, end_file, output_file, model_name):    
    # step1.jsonl 파일을 읽어 query를 key로, unique_idx를 value로 하는 매핑(dict) 생성    
    query_to_unique = {}

    start_total_cnt = start_suc_cnt = 0
    end_total_cnt = end_suc_cnt = 0
    start_plans = {}
    end_plans = {}

    with open(start_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)            
                query = data.get('query')                
                conv_history = [conv for conv in data.get('conversation_history')]
                conv = "-".join(conv_history)                
                key = conv + query
                unique_idx = data.get('unique_idx')
                if key is not None and unique_idx is not None:
                    query_to_unique[key] = data
                    start_suc_cnt += 1
                start_total_cnt += 1

    # step2.jsonl 파일을 읽어 query에 해당하는 unique_idx를 추가한 후 결과 저장
    idx1 = idx2 = 0
    s3_all_cnt = s3_filtered_cnt = 0
    gen_datas = []
    with open(end_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.strip():
                data = json.loads(line)            
                s3_all_cnt += 1
                end_total_cnt += 1
                
                try:
                    query = data.get('query')
                    full_conv_history = data.get('full_conversation_history')
                    if full_conv_history is not None:
                        conv_history = [conv for conv in full_conv_history]
                    else:
                        conv_history = [conv for conv in data.get('conversation_history')]
                    idx1 += 1 
                except:
                    print(data)                    
                    idx2 += 1
                    continue 
                conv = "-".join(conv_history)                                
                next_plan = data.get('next_turn_plan')
                next_turn_query = data.get('next_turn_query')
                if next_turn_query == '': # next_turn_query가 생성되지 않은 예시는 버림                    
                    #print("skipped")                                        
                    debug = False
                    if not debug:
                        continue
                if next_plan is None:
                    #print("next_turn None skipped") # 후처리 코드 수정 필요
                    continue

                key = conv + query                                                    
                # query가 일치하는 경우 unique_idx 추가            
                if key in query_to_unique:                    
                    if full_conv_history is not None:
                        data['conversation_history'] = full_conv_history
                        data.pop('full_conversation_history', None)
                    data["prev_plan"] = query_to_unique[key]["answer"]["plan"]
                    data['unique_idx'] = query_to_unique[key]["unique_idx"] + '-' + next_plan
                else:                
                    #data['unique_idx'] = 'NA'
                    #print(f'{key} not matched')                
                    continue

                end_plans[next_plan] = end_plans.get(next_plan, 0) + 1
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                s3_filtered_cnt += 1                
                gen_datas.append(data)
            
    #print(f"Total lines in {start_file}: {start_total_cnt}, Successful lines: {start_suc_cnt}")    
    #print(f"Start plans: {start_plans}")
    print(f"End plans: {end_plans}")                    
    print(f"Len plans: {len(end_plans)}")
    print_data_cnt_per_plan(gen_datas, "s3_integrate", model_name)
    print_filter_status(f"{s3_all_cnt} -> {s3_filtered_cnt}", "s3_integration", model_name)    
    print(idx1, idx2)
    #pdb.set_trace()    


def format_convert(data, u_key):
    converted_data = {}
    turn_idx = len(data['conversation_history']) + 1
    history = f"turn {turn_idx}: {data['query']} -> {data['device_response']}"
    data['conversation_history'].append(history)
    converted_data["conversation_history"] = data['conversation_history']
    converted_data["query"] = data["next_turn_query"]
    converted_data["rewrited_query"] = data["rewrited_query"]
    converted_data["answer"] = data["answer"]
    converted_data["unique_idx"] = data["unique_idx"] + '-' + u_key
    if "refered_turn" in data:
        converted_data["refered_turn"] = data["refered_turn"]
    return converted_data

def s4_integrate(start_file, end_file, output_file, model_name):    
    # step1.jsonl 파일을 읽어 query를 key로, unique_idx를 value로 하는 매핑(dict) 생성

    query_to_unique = {}

    start_total_cnt = start_suc_cnt = 0
    end_total_cnt = end_suc_cnt = 0
    start_plans = {}
    end_plans = {}    
    gen_datas = []
    with open(start_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)            
                    start_total_cnt += 1                    
                    query = data.get('query')
                    ans = data.get('answer')                
                    plan = ans.get('plan')
                    key = f'{query}-{plan}'                    
                    if query is not None and ans is not None:
                        if key in query_to_unique:                            
                            query_to_unique[key].append(ans)
                            #print(len(query_to_unique[key]))
                        else:
                            query_to_unique[key] = [ans]
                        start_suc_cnt += 1
                    else:
                        raise ValueError("Missing query or answer in line")
                except Exception as e:
                    print(f"Error processing: {e}")
    
    # step2.jsonl 파일을 읽어 query에 해당하는 unique_idx를 추가한 후 결과 저장        
    s4_filtered_cnt = s4_all_cnt = 0
    with open(end_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                if line.strip():
                    s4_all_cnt += 1
                    data = json.loads(line)            
                    end_total_cnt += 1
                    query = data.get('rewrited_query')                            
                    next_func = data.get('next_turn_plan')
                    key = f'{query}-{next_func}'                                  
                    if key in query_to_unique:
                        u_key = str(len(query_to_unique[key]))                                                    
                        answer = query_to_unique[key].pop()
                        data['answer'] = answer
                    else:                                        
                        #data['answer'] = 'NA'                    
                        #print(f'{key} not matched')
                        continue
                    
                    data = format_convert(data, u_key)
                    end_plans[next_func] = end_plans.get(next_func, 0) + 1
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    gen_datas.append(data)
                    end_suc_cnt += 1
                    s4_filtered_cnt += 1
            except Exception as e:
                print(f"Error processing: {e}")
                continue
    #print(f"Total lines in {start_file}: {start_total_cnt}, Successful lines: {start_suc_cnt}")
    #print(f"Total lines in {end_file}: {end_total_cnt}, Successful lines: {end_suc_cnt}")
    #print(f"Start plans: {start_plans}")
    #print(f"End plans: {end_plans}")                    
    print(f"Len plans: {len(end_plans)}")
    print_data_cnt_per_plan(gen_datas, "s4_integrate", model_name)
    print_filter_status(f"{s4_all_cnt} -> {s4_filtered_cnt}", "s4_integration", model_name)
    #pdb.set_trace()

def s3_complex_integrate(start_file, end_file, output_file, model_name):        
    # python3 dataset_integration.py --step complex_s3 --t1 ${DATADIR}/it4_s2_idxed.jsonl --t2 ${DATADIR}/it4_s3_complex.jsonl --o ${DATADIR}/it4_s3_complex_idxed.jsonl --model $MODEL &&
    query_to_unique = {}

    start_total_cnt = start_suc_cnt = 0
    end_total_cnt = end_suc_cnt = 0
    start_plans = {}
    end_plans = {}
    
    with open(start_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)            
                unique_idx = data.get('unique_idx')                
                query_to_unique[unique_idx] = data
                

    idx1 = idx2 = 0
    s3_all_cnt = s3_filtered_cnt = 0
    gen_datas = []
    with open(end_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.strip():
                data = json.loads(line)   
                unique_idx = data.get('unique_idx')         
                s3_all_cnt += 1
                end_total_cnt += 1
                
                try:
                    new_data = query_to_unique[unique_idx].copy()                                                                                          
                    idx1 += 1 
                except:
                    print(unique_idx, 'error')                    
                    idx2 += 1
                    continue 
                
                next_plan = data.get('next_turn_plan')
                next_turn_query = data.get('next_turn_query')                
                if next_turn_query == '': # next_turn_query가 생성되지 않은 예시는 버림                    
                    #print("skipped")                                        
                    debug = False
                    if not debug:
                        continue
                if next_plan is None:
                    #print("next_turn None skipped") # 후처리 코드 수정 필요
                    continue
                
                new_data["refered_turn"] = data.get('refered_turn')
                new_data['next_turn_plan'] = next_plan
                new_data['next_turn_query'] = next_turn_query
                new_data['rewrited_query'] = data.get('rewrited_query')
                new_data['unique_idx'] = unique_idx + '-' + next_plan
                
                end_plans[next_plan] = end_plans.get(next_plan, 0) + 1
                f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                s3_filtered_cnt += 1                
                gen_datas.append(new_data)
            
    #print(f"Total lines in {start_file}: {start_total_cnt}, Successful lines: {start_suc_cnt}")    
    #print(f"Start plans: {start_plans}")
    print(f"End plans: {end_plans}")                    
    print(f"Len plans: {len(end_plans)}")
    print_data_cnt_per_plan(gen_datas, "s3_complex", model_name)
    print_filter_status(f"{s3_all_cnt} -> {s3_filtered_cnt}", "s3_complex", model_name)        
    print(idx1, idx2)
    #pdb.set_trace()  
args = get_arg_parse()

if args.step == 's2':
    s2_integrate(args.t1, args.t2, args.o)
elif args.step == 's3':
    s3_integrate(args.t1, args.t2, args.o, args.model)
elif args.step == 's4':
    s4_integrate(args.t2, args.t1, args.o, args.model)
elif args.step == 'complex_s3':
    s3_complex_integrate(args.t1, args.t2, args.o, args.model)

# s2_integrate(file_paths[0], file_paths[1], o_file_paths[1])
# s3_integrate(o_file_paths[1], file_paths[2], o_file_paths[2])
# s4_integrate(file_paths[3], o_file_paths[2], o_file_paths[3])
