#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
source /mnt/data/miniconda3/bin/activate && conda activate mobile-agent-v3-py311
# 1) base model만 다운로드
python3 /home/hj153lee/RMA/train/download_and_merge_adapter.py \
  --adapter-dir /home/hj153lee/RMA/adapter \
  --base-dir /home/hj153lee/models/Qwen3-1.7B \
  --download-only

  # 2) 다운로드 + merge
CUDA_VISIBLE_DEVICES=0 python3 /home/hj153lee/RMA/train/download_and_merge_adapter.py \
  --adapter-dir /home/hj153lee/RMA/adapter-qwen3-base-4b/ \
  --base-dir /mnt/data/hj153lee/models/qwen3-4b-base \
  --merged-dir /mnt/data/hj153lee/qwen3-4b-base-merged

CUDA_VISIBLE_DEVICES=0 python3 /home/hj153lee/RMA/train/download_and_merge_adapter.py \
  --adapter-dir /mnt/data/hj153lee/rma153model/rma/phi-4 \
  --base-dir /mnt/data/hj153lee/phi4 \
  --merged-dir /mnt/data/hj153lee/phi4-rma-merged

CUDA_VISIBLE_DEVICES=7 python3 /home/hj153lee/RMA/train/download_and_merge_adapter.py \
  --adapter-dir /mnt/data/hj153lee/adapter/llama/base \
  --base-dir /mnt/data/hj153lee/llama3 \
  --merged-dir /mnt/data/hj153lee/llama3-base-merged
# 서버 실행
CUDA_VISIBLE_DEVICES=7 \
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test \
/usr/local/bin/ollama serve

OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test \
/usr/local/bin/ollama ps

OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test \
/usr/local/bin/ollama run qwen3-1.7b-rma-q4km

CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test ollama serve

OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test ollama run qwen3-1.7b-rma-q4km
# 아래는 종료 프로세스
ps -ef | grep 'ollama serve' | grep 11435 
pkill -f 'OLLAMA_HOST=127.0.0.1:11435'

source /mnt/data/miniconda3/bin/activate && conda activate mobile-agent-v3-py311
cd /home/hj153lee/RMA
python ollama_inference_qwen3_rma.py --t rewrite-qwen3 --o datasets/result/qwen3_rma_rewrite.tsv --test_key base --d

# 올라마 실행
LD_LIBRARY_PATH=/home/hj153lee/.local/ollama-v0.17.7/lib/ollama:/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v13:/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v12 \
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test \
/home/hj153lee/.local/ollama-v0.17.7/bin/ollama serve

# 모델 확인
LD_LIBRARY_PATH=/home/hj153lee/.local/ollama-v0.17.7/lib/ollama:/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v13:/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v12 \
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test \
/home/hj153lee/.local/ollama-v0.17.7/bin/ollama list

# 모델 등록
cd /home/hj153lee/qwen3-1.7b-artifacts
printf 'FROM ./qwen3-1.7b-rma-q4_k_m.gguf\nPARAMETER temperature 0\n' > Modelfile
LD_LIBRARY_PATH=/home/hj153lee/.local/ollama-v0.17.7/lib/ollama:/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v13:/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v12 \
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/hj153lee/.ollama-qwen3-test \
/home/hj153lee/.local/ollama-v0.17.7/bin/ollama create qwen3-1.7b-rma-q4km -f Modelfile


"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a base model from adapter_config.json and optionally merge LoRA adapter."
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Directory containing adapter_config.json and adapter_model.safetensors",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override base model ID (otherwise read from adapter_config.json)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Local directory to place downloaded base model snapshot",
    )
    parser.add_argument(
        "--merged-dir",
        default=None,
        help="Output directory for merged model (required unless --download-only)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype for merge loading",
    )
    parser.add_argument(
        "--trust-remote-code",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to trust remote model/tokenizer code. 'auto' tries native Transformers first, then falls back to remote code.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download base model and stop",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download from Hugging Face",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for private/gated repos",
    )
    return parser.parse_args()


def resolve_base_model_id(adapter_dir: Path, override_id: str | None) -> str:
    if override_id:
        return override_id

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_id = config.get("base_model_name_or_path")
    if not model_id:
        raise ValueError(
            f"'base_model_name_or_path' is missing in {config_path}"
        )
    return model_id


def download_base_model(
    model_id: str,
    base_dir: Path | None,
    hf_cache_dir: Path | None,
    force_download: bool,
    hf_token: str | None,
) -> str:
    kwargs = {
        "repo_id": model_id,
        "force_download": force_download,
    }
    if hf_token:
        kwargs["token"] = hf_token
    if base_dir is not None and hf_cache_dir is not None:
        raise ValueError("base_dir and hf_cache_dir cannot both be set")
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(base_dir)
        kwargs["local_dir_use_symlinks"] = False
    elif hf_cache_dir is not None:
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(hf_cache_dir)

    snapshot_path = snapshot_download(**kwargs)
    return snapshot_path


def parse_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def resolve_trust_remote_code(mode: str):
    if mode == "true":
        return True
    if mode == "false":
        return False
    return None


def load_with_fallback(loader, pretrained_path: str, kwargs: dict, remote_code_mode: str, label: str):
    trust_remote_code = resolve_trust_remote_code(remote_code_mode)
    if trust_remote_code is not None:
        return loader(pretrained_path, trust_remote_code=trust_remote_code, **kwargs)

    try:
        print(f"[INFO] loading {label} with trust_remote_code=False")
        return loader(pretrained_path, trust_remote_code=False, **kwargs)
    except Exception as exc:
        print(f"[WARN] native {label} load failed: {exc}")
        print(f"[INFO] retrying {label} load with trust_remote_code=True")
        return loader(pretrained_path, trust_remote_code=True, **kwargs)


def merge_adapter(
    base_path: str,
    adapter_dir: Path,
    merged_dir: Path,
    dtype_name: str,
    remote_code_mode: str,
):
    model_kwargs = {
        "torch_dtype": parse_dtype(dtype_name),
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    base_model = load_with_fallback(
        AutoModelForCausalLM.from_pretrained,
        base_path,
        model_kwargs,
        remote_code_mode,
        "model",
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer = load_with_fallback(
        AutoTokenizer.from_pretrained,
        base_path,
        {},
        remote_code_mode,
        "tokenizer",
    )
    tokenizer.save_pretrained(str(merged_dir))


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {adapter_weights}")

    base_model_id = resolve_base_model_id(adapter_dir, args.base_model)
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None

    print(f"[INFO] base model id: {base_model_id}")
    snapshot_path = download_base_model(
        model_id=base_model_id,
        base_dir=base_dir,
        hf_cache_dir=None,
        force_download=args.force_download,
        hf_token=args.hf_token,
    )
    print(f"[INFO] downloaded base snapshot: {snapshot_path}")

    if args.download_only:
        print("[INFO] --download-only set; merge step skipped.")
        return

    merged_dir = (
        Path(args.merged_dir).expanduser().resolve()
        if args.merged_dir
        else (adapter_dir.parent / f"{adapter_dir.name}-merged").resolve()
    )
    print(f"[INFO] merge output dir: {merged_dir}")
    merge_adapter(
        snapshot_path,
        adapter_dir,
        merged_dir,
        args.dtype,
        args.trust_remote_code,
    )
    print("[INFO] merge completed.")


if __name__ == "__main__":
    main()
