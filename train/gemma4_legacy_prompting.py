from __future__ import annotations

TAGGED_PLANNER_OUTPUT_INSTRUCTION = (
    "Return the answer using exactly these two sections in this order:\n"
    "<plan>\n"
    "...tool name or None...\n"
    "</plan>\n"
    "<arguments>\n"
    "...compact JSON object...\n"
    "</arguments>\n"
    "Always include both sections. The content inside <arguments> must be a valid "
    "compact JSON object. Do not output markdown, code fences, explanations, or "
    "any extra text."
)


LEGACY_SYSTEM_HISTORY_PROMPT = (
    "You are a helpful assistant capable of selecting appropriate tools based on "
    "user queries and generating corresponding parameters. Use information from "
    "the conversation history when relevant. Only use parameter values that are "
    "explicitly stated or can be reasonably inferred from the query. If no tool "
    "matches the query, set the tool to 'None'. "
    f"{TAGGED_PLANNER_OUTPUT_INSTRUCTION}\n<|tool|>{{tools}}<|/tool|>"
)

LEGACY_SYSTEM_REWRITE_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values that "
    "are explicitly stated or can be reasonably inferred from the query. "
    f"{TAGGED_PLANNER_OUTPUT_INSTRUCTION}\n<|tool|>{{tools}}<|/tool|>"
)


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_legacy_gemma4_prompt(
    tokenizer,
    *,
    tools: str,
    prompt_mode: str,
    query: str,
    conversation_history: str | None = None,
) -> str:
    if prompt_mode == "base":
        system_msg = LEGACY_SYSTEM_HISTORY_PROMPT.format(tools=tools)
        user_content = (
            f"Conversation History: {conversation_history}\n"
            f"User Query: {query}"
        )
    elif prompt_mode == "rewrite":
        system_msg = LEGACY_SYSTEM_REWRITE_PROMPT.format(tools=tools)
        user_content = f"User Query: {query}"
    else:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    return apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
    )
