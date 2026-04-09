#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from download_and_merge_adapter import load_with_fallback, merge_adapter, parse_dtype
from train_sentence_rewriter import (
    ensure_ollama_server,
    normalize_ollama_host,
    resolve_base_model_path,
    resolve_epoch_label,
    run_command,
    select_export_adapter_dir,
)

POSTPROCESS_STAGE_CHOICES = ("merge", "gguf", "ollama", "all")
DEFAULT_GEMMA4_LLAMA_CPP_DIR = "/home/hj153lee/llama.cpp-gemma4"


def resolve_postprocess_targets(args):
    merged_dir = Path(args.merged_dir).expanduser().resolve()
    gguf_path = Path(args.gguf_path).expanduser().resolve()
    modelfile_path = Path(args.modelfile).expanduser().resolve()
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()
    return merged_dir, gguf_path, modelfile_path, llama_cpp_dir


def ensure_llama_cpp_supports_gemma4(llama_cpp_dir: Path):
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {llama_cpp_dir}")

    script_text = convert_script.read_text(encoding="utf-8")
    if 'Gemma4ForConditionalGeneration' not in script_text:
        raise RuntimeError(
            "The configured llama.cpp does not support Gemma 4 HF->GGUF conversion. "
            f"Current path: {llama_cpp_dir}. "
            f"Use a newer checkout such as {DEFAULT_GEMMA4_LLAMA_CPP_DIR}."
        )


def write_postprocess_metadata(
    merged_dir: Path,
    *,
    model_name: str,
    stage: str,
    epoch_label: str,
    export_source: str,
    export_checkpoint_epoch: int | None,
    adapter_dir: Path | None,
    base_model_path: str | None,
    gguf_path: Path,
    modelfile_path: Path,
    ollama_model_name: str,
    llama_cpp_dir: Path,
):
    merged_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = merged_dir / "postprocess_meta.json"
    metadata = {
        "model_name": model_name,
        "postprocess_stage": stage,
        "epoch_label": epoch_label,
        "export_source": export_source,
        "export_checkpoint_epoch": export_checkpoint_epoch,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "base_model_path": base_model_path,
        "merged_dir": str(merged_dir),
        "gguf_path": str(gguf_path),
        "modelfile_path": str(modelfile_path),
        "ollama_model_name": ollama_model_name,
        "llama_cpp_dir": str(llama_cpp_dir),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[postprocess] wrote metadata: {metadata_path}")


def validate_merged_hf_model(merged_dir: Path, merge_dtype: str, remote_code_mode: str):
    print(f"[postprocess] validating merged HF model: {merged_dir}")
    tokenizer = load_with_fallback(
        AutoTokenizer.from_pretrained,
        str(merged_dir),
        {},
        remote_code_mode,
        "merged tokenizer",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = parse_dtype(merge_dtype)

    model = load_with_fallback(
        AutoModelForCausalLM.from_pretrained,
        str(merged_dir),
        model_kwargs,
        remote_code_mode,
        "merged model",
    )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[postprocess] merged HF load validation succeeded")


def ensure_merged_dir_exists(merged_dir: Path):
    if not (merged_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Merged HF model not found: {merged_dir}. Run with --postprocess_stage merge first."
        )


def ensure_gguf_exists(gguf_path: Path):
    if not gguf_path.exists():
        raise FileNotFoundError(
            f"GGUF file not found: {gguf_path}. Run with --postprocess_stage gguf first."
        )


def convert_merged_to_gguf(merged_dir: Path, gguf_path: Path, llama_cpp_dir: Path, gguf_outtype: str):
    ensure_llama_cpp_supports_gemma4(llama_cpp_dir)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "convert_hf_to_gguf.py",
            str(merged_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            gguf_outtype,
        ],
        cwd=str(llama_cpp_dir),
    )
    print(f"[postprocess] wrote GGUF: {gguf_path}")


def write_modelfile(modelfile_path: Path, gguf_path: Path):
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(
        f"FROM {gguf_path}\nPARAMETER temperature 0\n",
        encoding="utf-8",
    )
    print(f"[postprocess] wrote Modelfile: {modelfile_path}")


def register_ollama_model(args, modelfile_path: Path, ollama_model_name: str):
    ollama_env = os.environ.copy()
    ollama_env["OLLAMA_HOST"] = normalize_ollama_host(args.ollama_host)
    ollama_env["OLLAMA_MODELS"] = args.ollama_models_dir
    if args.ollama_lib_path:
        ollama_env["LD_LIBRARY_PATH"] = args.ollama_lib_path

    ensure_ollama_server(args.ollama_bin, ollama_env, args.ollama_host)
    run_command(
        [
            args.ollama_bin,
            "create",
            str(ollama_model_name),
            "-f",
            str(modelfile_path),
        ],
        env=ollama_env,
    )


def postprocess_gemma4_trained_model(
    args,
    model_name: str,
    output_dir: Path,
    output_root: Path,
    preexisting_checkpoint_names: set[str],
    num_train_epochs: float,
    final_global_step: int | None,
    cache_dir: Path | None = None,
):
    stage = getattr(args, "postprocess_stage", "all")
    if stage not in POSTPROCESS_STAGE_CHOICES:
        raise ValueError(f"Unsupported postprocess stage: {stage}")

    epoch_label = resolve_epoch_label(args.export_checkpoint_epoch, num_train_epochs)
    merged_dir, gguf_path, modelfile_path, llama_cpp_dir = resolve_postprocess_targets(args)
    ollama_model_name = args.ollama_model_name

    adapter_dir = None
    base_model_path = None

    if stage in ("merge", "all"):
        adapter_dir = select_export_adapter_dir(
            output_dir=output_dir,
            preexisting_checkpoint_names=preexisting_checkpoint_names,
            export_checkpoint_epoch=args.export_checkpoint_epoch,
            export_source=args.export_source,
            final_global_step=final_global_step,
        )
        base_model_path = resolve_base_model_path(
            model_name,
            adapter_dir,
            args,
            output_dir,
            cache_dir=cache_dir,
        )
        print(f"[postprocess] merge output dir: {merged_dir}")
        merge_adapter(
            base_model_path,
            adapter_dir,
            merged_dir,
            args.merge_dtype,
            args.trust_remote_code,
        )
        validate_merged_hf_model(
            merged_dir=merged_dir,
            merge_dtype=args.merge_dtype,
            remote_code_mode=args.trust_remote_code,
        )

    if stage in ("gguf", "all"):
        ensure_merged_dir_exists(merged_dir)
        convert_merged_to_gguf(
            merged_dir=merged_dir,
            gguf_path=gguf_path,
            llama_cpp_dir=llama_cpp_dir,
            gguf_outtype=args.gguf_outtype,
        )
        write_modelfile(modelfile_path, gguf_path)

    if stage in ("ollama", "all"):
        ensure_gguf_exists(gguf_path)
        if not modelfile_path.exists():
            write_modelfile(modelfile_path, gguf_path)
        register_ollama_model(
            args=args,
            modelfile_path=modelfile_path,
            ollama_model_name=ollama_model_name,
        )

    write_postprocess_metadata(
        merged_dir=merged_dir,
        model_name=model_name,
        stage=stage,
        epoch_label=epoch_label,
        export_source=args.export_source,
        export_checkpoint_epoch=args.export_checkpoint_epoch,
        adapter_dir=adapter_dir,
        base_model_path=base_model_path,
        gguf_path=gguf_path,
        modelfile_path=modelfile_path,
        ollama_model_name=ollama_model_name,
        llama_cpp_dir=llama_cpp_dir,
    )

    print(f"[postprocess] stage        : {stage}")
    print(f"[postprocess] merged dir   : {merged_dir}")
    print(f"[postprocess] gguf path    : {gguf_path}")
    print(f"[postprocess] modelfile    : {modelfile_path}")
    print(f"[postprocess] ollama model : {ollama_model_name}")
