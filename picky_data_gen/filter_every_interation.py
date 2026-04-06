import pdb
import json
import os
import re
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset

from openai import OpenAI
from utils import (
    JsonExtractor, SimilarityFilter, DataFilter,
    OpenAiGenerateResponse, GoogleGenerateResponse
)
from utils.frequently_used_tools import (
    get_model_name, get_arg_parse, read_jsonl, read_apis,
    print_data_cnt_per_plan, print_filter_status
)
from prompts import ITERATION_FILTERING_PROMPT, ITERATION_SIMPLE_FILTERING_PROMPT

def filtering_data(datas):
    new = []
    for d in datas:
        if not d["unique_idx"].endswith("_NR"):
            new.append(d)
    return new

def deleting_filtered_data(it_datas, filtered_datas):
    to_remove = set()
    for d in filtered_datas:
        resp = d.get("response")
        if resp == "N/A":
            continue

        if isinstance(resp, dict):
            valid_structure = True
            for v in resp.values():
                if not isinstance(v, dict):
                    valid_structure = False
                    break

            if not valid_structure:
                to_remove.add(d["unique_idx"])
                continue

            if all(v.get("status") == "Pass" for v in resp.values()):
                continue
            if any(v.get("status") == "Fail" for v in resp.values()):
                to_remove.add(d["unique_idx"])
        else:
            # resp가 dict가 아닌 경우에도 지워줌
            to_remove.add(d["unique_idx"])

    return [d for d in it_datas if d["unique_idx"] not in to_remove]

def get_partial_data(tcs, count=10):
    cnt = defaultdict(int)
    out = []
    for tc in tcs:
        p = tc["answer"]["plan"]
        if cnt[p] < count:
            out.append(tc)
            cnt[p] += 1
    return out

def save_jsonl(datas, out_path):
    base = os.path.splitext(out_path)[0]
    with open(f"{base}_filtered.jsonl", "w", encoding="utf-8") as f:
        for d in datas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def filtering_evaluation_pd(datasets, file_path):
    rows = []
    for d in datasets:
        resp = d.get("response")
        if resp == "N/A":
            continue
        row = {
            'id': d['answer']['plan'],
            'conversation_history': " || ".join(d['conversation_history']),
            'query': d['query'],
            'rewrited_query': d['rewrited_query'],
            'unique_idx': d['unique_idx']
        }
        all_pass = True
        for k, r in resp.items():
            row[k] = r['status']
            row[f"{k} Reason"] = r.get('reason', '')
            if r['status'] != 'Pass':
                all_pass = False
        row['Overall Pass'] = 'Pass' if all_pass else 'Fail'
        rows.append(row)
    df = pd.DataFrame(rows)
    resp_keys = list(datasets[0]['response'].keys())
    cols = [
        'id', 'conversation_history', 'query', 'rewrited_query', 'unique_idx'
    ] + sum([[k, f"{k} Reason"] for k in resp_keys], []) + ['Overall Pass']
    df = df[cols]
    print(df.head())
    df.to_csv(f"{os.path.splitext(file_path)[0]}_analysis.tsv", sep='\t', index=False)

def filtering_evaluation_pd_summary(datasets):
    overall = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)
    for d in datasets:
        plan = d['answer']['plan']
        resp = d['response']
        if resp == "N/A":
            continue
        all_pass = True
        for k, r in resp.items():
            if r['status'] == 'Pass':
                overall[plan][k] += 1
            else:
                all_pass = False
        counts[plan] += 1
        if all_pass:
            overall[plan]['Overall Pass'] += 1

    rows = []
    for plan, metrics in overall.items():
        total = counts[plan]
        row = {'id': plan}
        for k in datasets[0]['response'].keys():
            row[k] = f"{(metrics.get(k,0)/total)*100:.2f}%"
        row['Overall Pass'] = f"{(metrics.get('Overall Pass',0)/total)*100:.2f}%"
        row['Total Count'] = total
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ['id'] + list(datasets[0]['response'].keys()) + ['Overall Pass', 'Total Count']
    df = df[cols]
    print(df)
    df.to_csv("plan_response_overall_summary.tsv", sep='\t', index=False)

def __main__():
    args = get_arg_parse()

    # If delete mode
    if args.d:
        filtered = read_jsonl(args.o)
        it_datas  = read_jsonl(args.t)
        all_cnt   = len(it_datas)
        f_it      = deleting_filtered_data(it_datas, filtered)
        save_jsonl(f_it, args.t)
        print_data_cnt_per_plan(f_it, args.model)
        print_filter_status(f"{all_cnt} -> {len(f_it)}", "iteration data", args.model)
        return

    # Normal mode
    apis    = read_apis("apis/api_v3.0.1.jsonl")
    datas   = read_jsonl(args.t)
    datas   = filtering_data(datas)
    datas   = get_partial_data(datas, count=9999999)

    # Prepare arguments schema
    arg_apis = {}
    for plan, meta in apis.items():
        args_spec = {k: v for k, v in meta.get("arguments", {}).items()}
        for k in args_spec:
            args_spec[k].pop("default", None)
            args_spec[k].pop("type", None)
        arg_apis[plan] = args_spec

    # Model & prompt
    filter_model = args.filter_model if args.filter_model else 'gemini-2.0-flash'
    model_name, generate_response = get_model_name(filter_model)
    filters = [JsonExtractor()]
    prompt_template = ITERATION_SIMPLE_FILTERING_PROMPT

    # Worker function
    def process_data(data):
        u_idx = data["unique_idx"]
        plan = data["answer"]["plan"]
        args_spec = arg_apis.get(plan, {}).copy()

        # prune spec to only provided args
        provided = data["answer"]["arguments"]
        for k in list(args_spec):
            if k not in provided:
                args_spec.pop(k)

        inf = {
            "conversation_history": data["conversation_history"],
            "query": data["query"],
            "rewrited_query": data["rewrited_query"],
            "answer": data["answer"],
            "arguments": args_spec
        }

        try:
            prompt = prompt_template.format(data=inf)
            resp = generate_response("", [prompt])
            out = []
            for flt in filters:
                for r in flt.filter(resp):
                    record = inf.copy()
                    record["unique_idx"] = u_idx
                    record["response"]   = r
                    out.append(record)
            if not out:
                inf["unique_idx"] = u_idx
                inf["response"]   = "N/A"
                out.append(inf)
            return out

        except Exception as e:
            print(f"[{u_idx}] Error: {e}")
            return [{"unique_idx": u_idx, "response": "N/A"}]

    # Parallel execution
    response_list = []
    with open(args.o, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=5) as exe:
            futures = {exe.submit(process_data, d): d["unique_idx"] for d in datas}
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Processing data"):
                for rec in future.result():
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    response_list.append(rec)

    # Post-filtering
    it_datas = read_jsonl(args.t)
    f_it     = deleting_filtered_data(it_datas, response_list)
    save_jsonl(f_it, args.t)
    print_data_cnt_per_plan(f_it, args.model)
    print_filter_status(f"{len(it_datas)} -> {len(f_it)}", "iteration data", args.model)

__main__()
