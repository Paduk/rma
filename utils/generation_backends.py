from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import requests

try:
    import torch
except ImportError:
    torch = None


def _backend_arg_dest(name: str, prefix: str | None = None) -> str:
    return f"{prefix}_{name}" if prefix else name


def _backend_arg_flag(name: str, prefix: str | None = None) -> str:
    return f"--{prefix}_{name}" if prefix else f"--{name}"


def _backend_arg_label(prefix: str | None = None) -> str:
    return f"{prefix} " if prefix else ""


def _get_backend_arg(args, name: str, prefix: str | None = None):
    return getattr(args, _backend_arg_dest(name, prefix))


def add_text_generation_backend_args(
    parser,
    *,
    default_host: str,
    prefix: str | None = None,
    default_temperature: float = 0.0,
    default_num_predict: int = 512,
):
    stage_label = _backend_arg_label(prefix)
    parser.add_argument(
        _backend_arg_flag("backend", prefix),
        dest=_backend_arg_dest("backend", prefix),
        choices=["ollama", "hf"],
        default="ollama",
        help=(
            f"Text generation backend for the {stage_label.strip() or 'main'} stage. "
            "Use ollama for GGUF/Ollama or hf for Transformers inference."
        ),
    )
    parser.add_argument(
        _backend_arg_flag("host", prefix),
        dest=_backend_arg_dest("host", prefix),
        default=default_host,
        help=f"Ollama host URL used only when --{_backend_arg_dest('backend', prefix)} ollama.",
    )
    parser.add_argument(
        _backend_arg_flag("model", prefix),
        dest=_backend_arg_dest("model", prefix),
        default=None,
        help=f"Ollama model name override used only when --{_backend_arg_dest('backend', prefix)} ollama.",
    )
    parser.add_argument(
        _backend_arg_flag("temperature", prefix),
        dest=_backend_arg_dest("temperature", prefix),
        type=float,
        default=default_temperature,
        help=f"Generation temperature shared by ollama and hf backends for the {stage_label.strip() or 'main'} stage.",
    )
    parser.add_argument(
        _backend_arg_flag("num_predict", prefix),
        dest=_backend_arg_dest("num_predict", prefix),
        type=int,
        default=default_num_predict,
        help=f"Maximum generation tokens shared by ollama and hf backends for the {stage_label.strip() or 'main'} stage.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_model_path", prefix),
        dest=_backend_arg_dest("hf_model_path", prefix),
        default=None,
        help=f"Full HF model path or model ID used when --{_backend_arg_dest('backend', prefix)} hf.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_base_model_path", prefix),
        dest=_backend_arg_dest("hf_base_model_path", prefix),
        default=None,
        help="Base HF model path or model ID used with --hf_adapter_path, or plain base-only model source when no adapter is provided.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_adapter_path", prefix),
        dest=_backend_arg_dest("hf_adapter_path", prefix),
        default=None,
        help="PEFT adapter path used with --hf_base_model_path when --backend hf.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_adapter_mode", prefix),
        dest=_backend_arg_dest("hf_adapter_mode", prefix),
        choices=["attach", "merge"],
        default="attach",
        help="For base+adapter HF inference, either keep the adapter attached or merge it in memory once at startup.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_tokenizer_name", prefix),
        dest=_backend_arg_dest("hf_tokenizer_name", prefix),
        default=None,
        help="Optional tokenizer path/model ID override for HF inference. Defaults to the HF model source.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_device_map", prefix),
        dest=_backend_arg_dest("hf_device_map", prefix),
        default="auto",
        help=f"Transformers device_map used when --{_backend_arg_dest('backend', prefix)} hf.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_dtype", prefix),
        dest=_backend_arg_dest("hf_dtype", prefix),
        choices=["auto", "bf16", "fp16", "fp32"],
        default="bf16",
        help=f"Torch dtype used when --{_backend_arg_dest('backend', prefix)} hf.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_attn_implementation", prefix),
        dest=_backend_arg_dest("hf_attn_implementation", prefix),
        default=None,
        help=f"Optional attn_implementation passed to Transformers when --{_backend_arg_dest('backend', prefix)} hf.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_trust_remote_code", prefix),
        dest=_backend_arg_dest("hf_trust_remote_code", prefix),
        action="store_true",
        help="Pass trust_remote_code=True for HF model/tokenizer loading.",
    )
    parser.add_argument(
        _backend_arg_flag("hf_cache_dir", prefix),
        dest=_backend_arg_dest("hf_cache_dir", prefix),
        default=None,
        help="Optional cache directory for HF model/tokenizer loading.",
    )


class TextGenerationBackend(ABC):
    backend_name: str
    model_label: str

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 512,
        response_format_json: bool = False,
        stop: Sequence[str] | None = None,
    ) -> str:
        raise NotImplementedError

    def close(self):
        return None


class OllamaGenerationBackend(TextGenerationBackend):
    def __init__(self, *, host: str, model_name: str):
        self.backend_name = "ollama"
        self.host = host
        self.model_name = model_name
        self.model_label = model_name

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 512,
        response_format_json: bool = False,
        stop: Sequence[str] | None = None,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if stop:
            payload["options"]["stop"] = list(stop)
        if response_format_json:
            payload["format"] = "json"

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.text}")
        return response.json()["response"]


class _StopSequenceCriteria:
    def __init__(self, stop_token_ids):
        self.stop_token_ids = [ids for ids in stop_token_ids if ids]

    def __call__(self, input_ids, _scores, **_kwargs):
        if not self.stop_token_ids or input_ids.shape[0] != 1:
            return False
        tokens = input_ids[0].tolist()
        for stop_ids in self.stop_token_ids:
            if len(tokens) >= len(stop_ids) and tokens[-len(stop_ids) :] == stop_ids:
                return True
        return False


class HuggingFaceGenerationBackend(TextGenerationBackend):
    def __init__(
        self,
        *,
        model,
        tokenizer,
        model_label: str,
    ):
        self.backend_name = "hf"
        self.model = model
        self.tokenizer = tokenizer
        self.model_label = model_label
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    def _get_input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _build_stopping_criteria(self, stop: Sequence[str] | None):
        if not stop:
            return None
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopSequenceCriteria(_StopSequenceCriteria, StoppingCriteria):
            pass

        stop_token_ids = []
        for stop_text in stop:
            if not stop_text:
                continue
            ids = self.tokenizer(stop_text, add_special_tokens=False)["input_ids"]
            if ids:
                stop_token_ids.append(ids)
        if not stop_token_ids:
            return None
        return StoppingCriteriaList([StopSequenceCriteria(stop_token_ids)])

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 512,
        response_format_json: bool = False,
        stop: Sequence[str] | None = None,
    ) -> str:
        del response_format_json

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = self._get_input_device()
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

        generate_kwargs = {
            **encoded,
            "max_new_tokens": num_predict,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        stopping_criteria = self._build_stopping_criteria(stop)
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = stopping_criteria

        with torch.inference_mode():
            generated = self.model.generate(**generate_kwargs)

        prompt_token_count = encoded["input_ids"].shape[-1]
        generated_ids = generated[0][prompt_token_count:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if stop:
            for stop_text in stop:
                if stop_text and raw_text.endswith(stop_text):
                    raw_text = raw_text[: -len(stop_text)]
        return raw_text.strip()


def _resolve_torch_dtype(dtype_name: str):
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for --backend hf but is not installed in this environment."
        )
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_hf_model(model_source: str, *, torch_dtype, device_map: str, cache_dir: str | None, trust_remote_code: bool, attn_implementation: str | None):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "torch_dtype": torch_dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModelForCausalLM.from_pretrained(model_source, **kwargs)
    except TypeError:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        return AutoModelForCausalLM.from_pretrained(model_source, **kwargs)


def _load_hf_tokenizer(tokenizer_source: str, *, cache_dir: str | None, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def _resolve_base_model_from_adapter(hf_base_model_path: str | None, hf_adapter_path: str | None) -> str | None:
    if hf_base_model_path or not hf_adapter_path:
        return hf_base_model_path

    adapter_dir = Path(hf_adapter_path).expanduser()
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        return None
    data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    return data.get("base_model_name_or_path")


def build_text_generation_backend_from_args(
    args,
    *,
    default_model_name: str | None = None,
    prefix: str | None = None,
) -> TextGenerationBackend:
    backend = _get_backend_arg(args, "backend", prefix)
    host = _get_backend_arg(args, "host", prefix)
    model_override = _get_backend_arg(args, "model", prefix)
    hf_model_path = _get_backend_arg(args, "hf_model_path", prefix)
    hf_adapter_path = _get_backend_arg(args, "hf_adapter_path", prefix)
    hf_base_model_path = _resolve_base_model_from_adapter(
        _get_backend_arg(args, "hf_base_model_path", prefix),
        hf_adapter_path,
    )

    if backend == "ollama":
        model_name = model_override or default_model_name
        if not model_name:
            raise ValueError("Ollama backend requires a model name.")
        return OllamaGenerationBackend(host=host, model_name=model_name)

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for --backend hf but is not installed in this environment."
        )

    if hf_model_path and (hf_base_model_path or hf_adapter_path):
        raise ValueError(
            "Use either --hf_model_path, --hf_base_model_path, or --hf_base_model_path + --hf_adapter_path."
        )
    if not hf_model_path and not hf_base_model_path and not hf_adapter_path:
        raise ValueError(
            "HF backend requires --hf_model_path, --hf_base_model_path, or --hf_base_model_path + --hf_adapter_path."
        )
    if hf_adapter_path and not hf_base_model_path:
        raise ValueError(
            "Could not resolve the base model for the adapter. Pass --hf_base_model_path explicitly."
        )

    cache_dir = _get_backend_arg(args, "hf_cache_dir", prefix)
    trust_remote_code = _get_backend_arg(args, "hf_trust_remote_code", prefix)
    torch_dtype = _resolve_torch_dtype(_get_backend_arg(args, "hf_dtype", prefix))

    plain_model_source = hf_model_path or (hf_base_model_path if not hf_adapter_path else None)
    if plain_model_source:
        model_source = plain_model_source
        tokenizer_source = _get_backend_arg(args, "hf_tokenizer_name", prefix) or model_source
        model = _load_hf_model(
            model_source,
            torch_dtype=torch_dtype,
            device_map=_get_backend_arg(args, "hf_device_map", prefix),
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation=_get_backend_arg(args, "hf_attn_implementation", prefix),
        )
        tokenizer = _load_hf_tokenizer(
            tokenizer_source,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        return HuggingFaceGenerationBackend(
            model=model,
            tokenizer=tokenizer,
            model_label=f"hf:{model_source}",
        )

    from peft import PeftModel

    base_model = _load_hf_model(
        hf_base_model_path,
        torch_dtype=torch_dtype,
        device_map=_get_backend_arg(args, "hf_device_map", prefix),
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        attn_implementation=_get_backend_arg(args, "hf_attn_implementation", prefix),
    )
    model = PeftModel.from_pretrained(base_model, hf_adapter_path)
    adapter_mode = _get_backend_arg(args, "hf_adapter_mode", prefix)
    if adapter_mode == "merge":
        model = model.merge_and_unload()

    tokenizer = _load_hf_tokenizer(
        _get_backend_arg(args, "hf_tokenizer_name", prefix) or hf_base_model_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return HuggingFaceGenerationBackend(
        model=model,
        tokenizer=tokenizer,
        model_label=(
            f"hf:{hf_base_model_path} + {hf_adapter_path} "
            f"[{adapter_mode}]"
        ),
    )
