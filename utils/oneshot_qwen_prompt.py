from __future__ import annotations

import json

QWEN3_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-4B"
QWEN3_1P7B_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-1.7B"
QWEN3_0P6B_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
QWEN25_PROMPT_TOKENIZER_NAME = "Qwen/Qwen2.5-3B-Instruct"
PHI_PROMPT_TOKENIZER_NAME = "microsoft/Phi-4-mini-instruct"
LLAMA_PROMPT_TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"

PROMPT3_FEW_SHOT_EXAMPLES = [
    {
        "conversation_history": [
            "turn 1: Launch the camera interface and snap a photo. -> Okay, picture taken! You can find it at content://media/external/images/media/12345."
        ],
        "query": "Show me the photo",
        "rewrited_query": "Show me content://media/external/images/media/12345",
        "plan": "ACTION_OPEN_CONTENT",
        "arguments": {"uri": "content://media/external/images/media/12345"},
    },
    {
        "conversation_history": [
            "turn 1: Are there any timers going on now? -> Yes - there are 2 timers running: Pasta with 6 minutes to go and Workout countdown at 14 minutes."
        ],
        "query": "Add a 20-minute timer for study",
        "rewrited_query": "Add a 20-minute timer for study",
        "plan": "ACTION_SET_TIMER",
        "arguments": {"duration": "20 minutes", "EXTRA_MESSAGE": "study"},
    },
    {
        "conversation_history": [
            "turn 1: Launch the camera interface and snap a photo. -> Okay, picture taken! You can find it at content://media/external/images/media/12345.",
            "turn 2: I want to take a portrait. Open up the good old still camera. -> Still image camera is now open for a portrait.",
            "turn 3: Get the latest image file I just captured. -> Here's your latest capture: content://temp/latest_capture.jpg",
        ],
        "query": "Show me the photo",
        "rewrited_query": "Show me content://media/external/images/media/12345",
        "plan": "ACTION_OPEN_CONTENT",
        "arguments": {"uri": "content://media/external/images/media/12345"},
    },
    {
        "conversation_history": [
            "turn 1: Select those last three images -> All set! Here are the last three images you picked: content://camera/image/005.jpeg, content://camera/image/006.jpeg, content://camera/image/007.jpeg",
            "turn 2: Let's capture the snapshot. -> All set! Here's your snapshot: content://media/external/images/media/67890",
        ],
        "query": "Send them to project.team@example.com with subject 'Site Photos' and body 'Please check the latest shots.'",
        "rewrited_query": "Send content://camera/image/005.jpeg, content://camera/image/006.jpeg, content://camera/image/007.jpeg to project.team@example.com with subject 'Site Photos' and body 'Please check the latest shots.'",
        "plan": "send_email",
        "arguments": {
            "to": "project.team@example.com",
            "subject": "Site Photos",
            "body": "Please check the latest shots.",
            "attachments": [
                "content://camera/image/005.jpeg",
                "content://camera/image/006.jpeg",
                "content://camera/image/007.jpeg",
            ],
        },
    },
]


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


def build_system_message(api_str: str, prompt_option: str = "prompt1") -> str:
    if prompt_option in {"prompt1", "prompt3"}:
        return (
            "Given a conversation history, a user query, and a list of available tools, "
            "first rewrite the query by resolving only ambiguous pronouns or omitted "
            "references using the conversation history. If there are no ambiguous "
            "pronouns or omitted references, rewrited_query may be identical to the "
            "user query. Then, based on the rewrited_query, select the most appropriate "
            "tool and generate its arguments. Only use parameter values that are "
            "explicitly stated or can be reasonably inferred from the query or "
            "conversation history. Return compact JSON only with keys "
            "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
            "keys. The value of \"arguments\" must always be an object. If no tool "
            "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}.\n"
            f"<|tool|>{api_str}<|/tool|>"
        )
    if prompt_option in {"prompt4", "prompt5"}:
        return (
            "Given a conversation history, a user query, and a list of available tools, "
            "first rewrite the query by resolving only ambiguous pronouns or omitted "
            "references using the conversation history. If there are no ambiguous "
            "pronouns or omitted references, rewrited_query may be identical to the "
            "user query. Then, based on the rewrited_query, select the most appropriate "
            "tool and generate its arguments. Only use parameter values that are "
            "explicitly stated or can be reasonably inferred from the query or "
            "conversation history. Return compact JSON only with keys "
            "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
            "keys. The value of \"arguments\" must always be an object. If no tool "
            "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}.\n"
            "Below are a few examples. Learn the pattern of rewriting and tool selection "
            "from them. Do not copy their contents."
        )
    if prompt_option == "prompt4-rewriting":
        return (
            "Given a conversation history, a user query, and a list of available tools, "
            "first rewrite the query by resolving only ambiguous pronouns or omitted "
            "references using the conversation history. If there are no ambiguous "
            "pronouns or omitted references, rewrited_query may be identical to the "
            "user query. Then, based on the rewrited_query, select the most appropriate "
            "tool and generate its arguments. Only use parameter values that are "
            "explicitly stated or can be reasonably inferred from the query or "
            "conversation history. Return compact JSON only with keys "
            "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
            "keys. The value of \"arguments\" must always be an object. If no tool "
            "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}.\n"
            "Below are a few rewriting-only examples. They show how to rewrite the "
            "user query, not how to choose the final tool call. For the actual task, "
            "you must still return the full JSON with rewrited_query, plan, and "
            "arguments. The final plan must be one of the available tools for the "
            "actual task or \"None\", and arguments must use only fields supported by "
            "the selected tool."
        )
    if prompt_option == "prompt2":
        return (
            "Given a conversation history, a user query, and a list of available tools, "
            "first rewrite the query by resolving only ambiguous pronouns or omitted "
            "references using the conversation history. If there are no ambiguous "
            "pronouns or omitted references, rewrited_query may be identical to the "
            "user query. Use the entire conversation history to resolve references. "
            "The relevant context may come from the immediately previous turn or from "
            "any earlier turn. Choose the prior mention that best matches the user's "
            "current request, rather than always using the most recent turn. Then, "
            "based on the rewrited_query, select the most appropriate tool and generate "
            "its arguments. Only use parameter values that are explicitly stated or can "
            "be reasonably inferred from the query or conversation history. Return "
            "compact JSON only with keys "
            "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
            "keys. The value of \"arguments\" must always be an object. If no tool "
            "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}.\n"
            f"<|tool|>{api_str}<|/tool|>"
        )
    raise ValueError(f"Unsupported prompt_option: {prompt_option}")


def build_user_content(conversation_history: str, query: str) -> str:
    return f"Conversation History: {conversation_history}\nUser Query: {query}"


def build_prompt4_user_content(api_str: str, conversation_history: str, query: str) -> str:
    return (
        "Now solve the actual task. Use only the following available tools.\n"
        f"<|tool|>{api_str}<|/tool|>\n"
        f"{build_user_content(conversation_history, query)}"
    )


def build_prompt4_rewriting_user_content(api_str: str, conversation_history: str, query: str) -> str:
    return (
        "Now solve the actual task. The examples above demonstrate rewriting only. "
        "Use only the following available tools for the final answer.\n"
        f"<|tool|>{api_str}<|/tool|>\n"
        "Return compact JSON with keys \"rewrited_query\", \"plan\", and "
        "\"arguments\". The value of \"plan\" must be one of the available tools "
        "listed above or \"None\". The value of \"arguments\" must be an object and "
        "must contain only parameters supported by the selected tool.\n"
        f"{build_user_content(conversation_history, query)}"
    )


def build_prompt3_few_shot_messages():
    messages = []
    for example in PROMPT3_FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": build_user_content(
                    conversation_history=str(example["conversation_history"]),
                    query=example["query"],
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "rewrited_query": example["rewrited_query"],
                        "plan": example["plan"],
                        "arguments": example["arguments"],
                    },
                    ensure_ascii=True,
                ),
            }
        )
    return messages


def build_prompt4_rewriting_few_shot_messages():
    messages = []
    for example in PROMPT3_FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": build_user_content(
                    conversation_history=str(example["conversation_history"]),
                    query=example["query"],
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "rewrited_query": example["rewrited_query"],
                    },
                    ensure_ascii=True,
                ),
            }
        )
    return messages


def build_history_few_shot_messages():
    messages = []
    for example in PROMPT3_FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": build_user_content(
                    conversation_history=str(example["conversation_history"]),
                    query=example["query"],
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "plan": example["plan"],
                        "arguments": example["arguments"],
                    },
                    ensure_ascii=True,
                ),
            }
        )
    return messages


def build_oneshot_messages(conversation_history: str, query: str, candidates, apis, prompt_option: str = "prompt1"):
    api_str = build_api_str_from_candidates(candidates, apis)
    messages = [
        {"role": "system", "content": build_system_message(api_str, prompt_option=prompt_option)},
    ]
    if prompt_option in {"prompt3", "prompt4", "prompt5"}:
        messages.extend(build_prompt3_few_shot_messages())
    elif prompt_option == "prompt4-rewriting":
        messages.extend(build_prompt4_rewriting_few_shot_messages())
    if prompt_option in {"prompt4", "prompt5"}:
        messages.append(
            {
                "role": "user",
                "content": build_prompt4_user_content(api_str, conversation_history, query),
            }
        )
    elif prompt_option == "prompt4-rewriting":
        messages.append(
            {
                "role": "user",
                "content": build_prompt4_rewriting_user_content(
                    api_str, conversation_history, query
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": build_user_content(conversation_history, query)})
    return messages


def render_chat_template(tokenizer, messages, add_generation_prompt: bool, model_name: str) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_messages_as_plain_text(messages, add_generation_prompt: bool) -> str:
    sections = []
    for message in messages:
        role = str(message.get("role", "")).strip().capitalize() or "User"
        content = str(message.get("content", "")).strip()
        sections.append(f"{role}:\n{content}")
    if add_generation_prompt:
        sections.append("Assistant:\n")
    return "\n\n".join(sections)
