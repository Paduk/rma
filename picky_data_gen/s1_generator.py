from string import Template
import json
from transformers import AutoTokenizer
import random
from openai import OpenAI
import os
from tqdm import tqdm
from typing import List, Dict, Iterable
import logging
import pdb
#from utils.extract import extract_and_parse_jsons
from utils.prompt import SEED_GENERATION_PROMPT, DATA_GENERATION_PROMPT
from utils import RandomListSampler, JsonlSampler, LLMDataCollector, JsonExtractor, SimilarityFilter, DataFilter
from utils import SimilarityRecord, OpenAiGenerateResponse, HuggingFaceTokenizer, GoogleGenerateResponse
from utils.frequently_used_tools import get_model_name
import argparse
parser = argparse.ArgumentParser(description="data integration")
parser.add_argument('--api', type=str, default="apis/api_v3.0.1.jsonl", help='사용자 이름')
parser.add_argument('--o', type=str, required=True, help='out_file')
parser.add_argument('--model', type=str, default='o3', help='out_file')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INIT_PROMPT = Template(SEED_GENERATION_PROMPT)
GEN_PROMPT = Template(DATA_GENERATION_PROMPT)
#print(SEED_GENERATION_PROMPT)

def format_example(example):   
    try: 
        try:
            tool = example["tools"][0]                
        except Exception as e:
            tool = {}

        if "answers" in example:
            if "id" in example["answers"][0]:
                example["answers"][0].pop('id', "")
            resp = {
                "query": example["query"],
                "answer": example["answers"][0]
            }
        else:
            if "id" in example["answer"]:
                example["answer"].pop('id', "")                    

            resp = {
                "query": example["query"],
                "answer": example["answer"]
            }
    except Exception as e:                
        print(e)        

    if "id" in example:
        example.pop('id', "")
        
    return f'tool: {json.dumps(tool, indent=2, ensure_ascii=False)}\nresponse: {json.dumps(resp, indent=2)}'
    
def check_format(data):
    if "query" not in data or "answers" not in data:
        return False
    if not isinstance(data["query"], str):
        return False
    if not isinstance(data["answer"], dict):
        return False
    
    if "plan" not in data["answer"] or "arguments" not in data["answer"]:
        return False
    if not isinstance(data["answer"]["arguments"], dict):
        return False
        
    return True

MODEL_CLASS_MAP = {
    "gpt": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY", None),
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", None),
    }
}

sample_file = '/workspace/hj153lee/gen_data/DroidCall/data/function_call/processed_xlam.jsonl'
api_file = 'apis/api_v3.0.1.jsonl'
api_file = args.api
OUTPUT_FILE = args.o # "datagen/it1_s1.jsonl"

tokenizer_path = "../qwen2_tokenizer"
#OUTPUT_FILE = "datagen/it1_s1_o3.jsonl"
#OUTPUT_FILE = "datagen/deb_it1_s1_o3.jsonl"
NUM_GENERATE = 50
SIMILARITY_THRESHOLD = 0.75
SAMPLE_NUM = 8
model_class = 'gpt'

class FormatFilter(DataFilter):
    def validate(self, data: Dict[str, str]) -> bool:
        return check_format(data)

all_examples = []
with open(sample_file, "r") as f:
    for line in f.readlines():
        example = json.loads(line)
        if example["tools_num"] == 1 and example["answers_num"] == 1: # simple call
            all_examples.append(example)
    
path = tokenizer_path
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer = HuggingFaceTokenizer(tokenizer)

func2instructions = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for l in f.readlines():
            d = json.loads(l)
            all_examples.append(d)
            #func_name = d["answers"][0]["plan"]
            func_name = d["answer"]["plan"]
            if func_name not in func2instructions:
                func2instructions[func_name] = []
            func2instructions[func_name].append(d)

records = SimilarityRecord(tokenizer)

model_name, generate_response = get_model_name(args.model)

with open(api_file) as f:
    all_tools_ = [json.loads(line) for line in f.readlines()]
all_tools = []
for tool in all_tools_:
    tool.pop("next_turn_plans", None)
    all_tools.append(tool)

output_file = open(OUTPUT_FILE, "a")
similarity_filter = SimilarityFilter(records, key="query", bound=SIMILARITY_THRESHOLD)
#filters = [JsonExtractor(), FormatFilter(), similarity_filter]
filters = [JsonExtractor(), similarity_filter]

unique_idx = 1
for tool_idx, tool in enumerate(all_tools):
    data = []
    for example in func2instructions.get(tool["plan"], []):
        records.add(example["query"])
        data.append(example)
    
    initial_num = len(data)
    if initial_num >= NUM_GENERATE:
        continue
    
    tool_text = json.dumps(tool, indent=4, ensure_ascii=False)    
    class ExampleSampler(RandomListSampler):
        def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
            examples_text = "\n".join([format_example(sample) for sample in samples])
            return {"examples": examples_text, "tool": tool_text}
    
    collector = LLMDataCollector(INIT_PROMPT, ExampleSampler(all_examples, 2), filters,
                                 generate_response=generate_response, verbose=True)
    
    # this is initial collection
    while len(data) <= 0:        
        for d in collector.collect(NUM_GENERATE, "init collection", num_generated=len(data), once=True):                
            try:
                d.pop("next_turn_plans", None)              
                d["answer"]["plan"] = d["answer"]["name"]          
                param = d["answer"]["arguments"]
                d["answer"].pop("name")
                d["answer"].pop("arguments")            
                plan = d["answer"]["plan"]          
                d["answer"]["arguments"] = param                
                d["unique_idx"] = plan + '-' + str(unique_idx)
                unique_idx += 1
                data.append(d)
                output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
                output_file.flush()
            except Exception as e:
                print(e)
    
    class QuerySampler(RandomListSampler):
        def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
            samples = [
                {k: v for k, v in sample.items() if k not in["tools"]}
                for sample in samples
            ]            
            
            examples_text = "\n".join([json.dumps(sample, indent=2, ensure_ascii=False) for sample in samples])
            return {"examples": examples_text, "tool": tool_text}
    
    collector.switch(GEN_PROMPT, QuerySampler(data, SAMPLE_NUM))    
    for d in collector.collect(NUM_GENERATE, "gen collection", len(data)):
        d.pop("next_turn_plans", None)
        #d["tools"] = [tool]
        d["unique_idx"] = d["answer"]["plan"] + '-' + str(unique_idx)
        unique_idx += 1
        data.append(d)
        output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
        output_file.flush()
    
    records = SimilarityRecord(tokenizer)
    similarity_filter.change_record(records)    
print("done")
output_file.close()