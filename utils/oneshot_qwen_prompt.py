from __future__ import annotations

QWEN3_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-4B"
QWEN3_1P7B_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-1.7B"
QWEN3_0P6B_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
QWEN25_PROMPT_TOKENIZER_NAME = "Qwen/Qwen2.5-3B-Instruct"
PHI_PROMPT_TOKENIZER_NAME = "microsoft/Phi-4-mini-instruct"
LLAMA_PROMPT_TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def is_qwen_model(model_name: str) -> bool:
    return "qwen" in model_name.lower()


def is_phi_model(model_name: str) -> bool:
    lower_name = model_name.lower()
    return "phi-4" in lower_name or "phi4" in lower_name


def is_llama_model(model_name: str) -> bool:
    return "llama" in model_name.lower()


def resolve_prompt_tokenizer_name(model_name: str, prompt_tokenizer_name: str | None = None) -> str:
    if prompt_tokenizer_name:
        return prompt_tokenizer_name
    lower_name = model_name.lower()
    if "qwen3-0.6b" in lower_name:
        return QWEN3_0P6B_PROMPT_TOKENIZER_NAME
    if "qwen3-1.7b" in lower_name:
        return QWEN3_1P7B_PROMPT_TOKENIZER_NAME
    if "qwen2.5" in lower_name or "qwen25" in lower_name:
        return QWEN25_PROMPT_TOKENIZER_NAME
    if is_qwen_model(model_name):
        return QWEN3_PROMPT_TOKENIZER_NAME
    if is_phi_model(model_name):
        return PHI_PROMPT_TOKENIZER_NAME
    if is_llama_model(model_name):
        return LLAMA_PROMPT_TOKENIZER_NAME
    raise ValueError(
        "Unable to infer a prompt tokenizer from model_name. "
        "Pass --prompt_tokenizer_name explicitly for unsupported model names."
    )


def build_api_str_from_candidates(candidates, apis) -> str:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list.")

    lines = []
    for plan_name in candidates:
        api_data = apis[plan_name].copy()
        lines.append(f"{plan_name}: {api_data}")
    return "\n".join(lines)


def build_system_message(api_str: str) -> str:
    return (
        "Given a conversation history, a user query, and a list of available tools, "
        "first rewrite the query by resolving ambiguous references using the "
        "conversation history. Then select the most appropriate tool and generate "
        "its arguments. Return compact JSON only with keys "
        "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
        "keys. The value of \"arguments\" must always be an object. If no tool "
        "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}.\n"
        f"<|tool|>{api_str}<|/tool|>"
    )


def build_user_content(conversation_history: str, query: str) -> str:
    return f"Conversation History: {conversation_history}\nUser Query: {query}"


def build_oneshot_messages(conversation_history: str, query: str, candidates, apis):
    api_str = build_api_str_from_candidates(candidates, apis)
    return [
        {"role": "system", "content": build_system_message(api_str)},
        {"role": "user", "content": build_user_content(conversation_history, query)},
    ]


def render_chat_template(tokenizer, messages, add_generation_prompt: bool, model_name: str) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if is_qwen_model(model_name):
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
