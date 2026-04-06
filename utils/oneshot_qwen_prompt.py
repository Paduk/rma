from __future__ import annotations


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
    if "Qwen/" in model_name:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
