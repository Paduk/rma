from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os, sys

# 이 파일(model_merge.py)의 상위 폴더(RMA/)를 sys.path에 추가
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root not in sys.path:
    sys.path.insert(0, root)

from utils.frequently_used_tools import get_arg_parse

def merge_model(base_model_path, lora_model_path, merged_model_path):    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    model = model.merge_and_unload()    
    model.save_pretrained(merged_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)


args = get_arg_parse()
if args.model == "llama":
    base_model_path = "/mnt/data/.cache/hj153lee/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/"
elif args.model == "phi4":
    base_model_path = "/mnt/data/.cache/hj153lee/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/5a149550068a1eb93398160d8953f5f56c3603e9/"
elif args.model == "qwen25":
    base_model_path = "/mnt/data/.cache/hj153lee/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/"
elif args.model == "qwen3":
    base_model_path = "/mnt/data/.cache/hj153lee/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a/"
elif args.model == 'gemma3':
    base_model_path = "/root/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/"
elif args.model == 'qwen3-1.7b':
    base_model_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/"
elif args.model == 'qwen3-0.6b':
    base_model_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/6130ef31402718485ca4d80a6234f70d9a4cf362/"
else:
    print("Invalid model name. Please use 'llama3' or 'phi4'.")
    exit(0)
lora_model_path = args.t1 # "/workspace/hj153lee/RMA/train/llama3-3b-rma/checkpoint-5285"
merged_model_path = args.t2 # "/workspace/hj153lee/ollama/llama3-3b-rma-complex"
print(base_model_path)
print(lora_model_path)
merge_model(base_model_path, lora_model_path, merged_model_path)

# lora_model_path = "/workspace/hj153lee/RMA/train/llama3-rewrite-weighted/checkpoint-2935"
# merged_model_path = "/workspace/hj153lee/ollama/llama3-rewrite-weighted-e5"
# merge_model(base_model_path, lora_model_path, merged_model_path)