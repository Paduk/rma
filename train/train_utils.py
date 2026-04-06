import datasets
import ast
import pandas as pd
import os
import json
import pdb

def preprocess_example_history(example):    
    api_str = ""    
    candidates = ast.literal_eval(example['candidates'])
    for plan in candidates:
        # apis 사전에서 해당 plan에 대응하는 데이터를 복사하여 문자열로 변환합니다.
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
        #api_str += json.dumps(api_data, indent=2, ensure_ascii=False) + "\n"
        
    prompt = prompt_template.format(
        tools=api_str,
        conversation_history=example["conversation_history"],
        data=example["query"],
        answer=example["answer"]
    )
    
    # 전체 prompt+answer를 토크나이즈합니다.
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True
    )
    input_ids = tokenized["input_ids"]

    # model이 정답 토큰을 시작하는 위치를 찾아, 
    # 사용자 입력 부분은 label 값으로 -100을 할당하고,
    # 모델 생성 부분은 실제 토큰 id를 할당합니다.
    model_start = prompt.find("assistant<|end_header_id|>")
    if model_start == -1:
        raise ValueError("Prompt does not contain 'assistant<|end_header_id|>'")
    model_token_start = len(
        tokenizer(prompt[:model_start], add_special_tokens=False)["input_ids"]
    )
    labels = [-100] * model_token_start + input_ids[model_token_start:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "strprompt": prompt,
        # answer가 딕셔너리인 경우 보기 좋게 json 문자열로 변환합니다.
        "stranswer": json.dumps(example["answer"], indent=2, ensure_ascii=False)
                      if isinstance(example["answer"], dict) else example["answer"],
    }

def preprocess_example_rewrite(example):    
    api_str = ""    
    candidates = ast.literal_eval(example['candidates'])
    for plan in candidates:
        # apis 사전에서 해당 plan에 대응하는 데이터를 복사하여 문자열로 변환합니다.
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
        #api_str += json.dumps(api_data, indent=2, ensure_ascii=False) + "\n"
        
    prompt = prompt_template.format(
        tools=api_str,
        data=example["rewrited_query"],
        answer=example["answer"]
    )
    
    # 전체 prompt+answer를 토크나이즈합니다.
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True
    )
    input_ids = tokenized["input_ids"]

    # model이 정답 토큰을 시작하는 위치를 찾아, 
    # 사용자 입력 부분은 label 값으로 -100을 할당하고,
    # 모델 생성 부분은 실제 토큰 id를 할당합니다.
    model_start = prompt.find("assistant<|end_header_id|>")
    if model_start == -1:
        raise ValueError("Prompt does not contain 'assistant<|end_header_id|>'")
    model_token_start = len(
        tokenizer(prompt[:model_start], add_special_tokens=False)["input_ids"]
    )
    labels = [-100] * model_token_start + input_ids[model_token_start:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "strprompt": prompt,
        # answer가 딕셔너리인 경우 보기 좋게 json 문자열로 변환합니다.
        "stranswer": json.dumps(example["answer"], indent=2, ensure_ascii=False)
                      if isinstance(example["answer"], dict) else example["answer"],
    }
