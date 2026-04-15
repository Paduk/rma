from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RmaModelProfile:
    model_name: str
    max_length: int = 1536
    prefix: str = "all_linear"


RMA_MODEL_PROFILES = {
    "qwen": RmaModelProfile("Qwen/Qwen3-4B"),
    "qwen3-1.7b": RmaModelProfile("Qwen/Qwen3-1.7B"),
    "qwen3-0.6b": RmaModelProfile("Qwen/Qwen3-0.6B"),
    "qwen2.5": RmaModelProfile("Qwen/Qwen2.5-3B-Instruct"),
    "phi": RmaModelProfile("microsoft/Phi-4-mini-instruct"),
    "llama": RmaModelProfile("meta-llama/Llama-3.2-3B-Instruct"),
    "gemma": RmaModelProfile("google/gemma-3-4b-it"),
    "glm-edge-1.5b": RmaModelProfile("zai-org/glm-edge-1.5b-chat"),
    "glm-edge-4b": RmaModelProfile("zai-org/glm-edge-4b-chat"),
    "smollm2-1.7b": RmaModelProfile(
        "HuggingFaceTB/SmolLM2-1.7B",
        prefix="simple_template",
    ),
    "smollm2-1.7b-instruct": RmaModelProfile(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ),
    "smollm3-3b": RmaModelProfile("HuggingFaceTB/SmolLM3-3B"),
    "falcon3-1b": RmaModelProfile("tiiuae/Falcon3-1B-Instruct"),
    "falcon3-1b-base": RmaModelProfile(
        "tiiuae/Falcon3-1B-Base",
        prefix="simple_template",
    ),
    "falcon3-3b": RmaModelProfile("tiiuae/Falcon3-3B-Instruct"),
    "falcon3-3b-base": RmaModelProfile(
        "tiiuae/Falcon3-3B-Base",
        prefix="simple_template",
    ),
    "exaone4-1.2b": RmaModelProfile("LGAI-EXAONE/EXAONE-4.0-1.2B"),
    "olmo2-1b": RmaModelProfile(
        "allenai/OLMo-2-0425-1B",
        prefix="simple_template",
    ),
    "olmo2-1b-instruct": RmaModelProfile("allenai/OLMo-2-0425-1B-Instruct"),
    "granite3.3-2b": RmaModelProfile("ibm-granite/granite-3.3-2b-instruct"),
    "lfm2.5-1.2b": RmaModelProfile("LiquidAI/LFM2.5-1.2B-Instruct"),
}


def resolve_profile_model_name(profile: str | None, model_name: str | None, default_model_name: str) -> str:
    if model_name:
        return model_name
    if profile:
        return RMA_MODEL_PROFILES[profile].model_name
    return default_model_name


def resolve_profile_max_length(profile: str | None, max_length: int | None, default_max_length: int) -> int:
    if max_length is not None:
        return max_length
    if profile:
        return RMA_MODEL_PROFILES[profile].max_length
    return default_max_length
