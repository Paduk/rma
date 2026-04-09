from __future__ import annotations

import json

try:
    from .gemma4_legacy_prompting import apply_chat_template
except ImportError:
    from gemma4_legacy_prompting import apply_chat_template


DEFAULT_GEMMA4_PROMPT_TOKENIZER_NAME = "google/gemma-4-E2B-it"

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the query clearly by replacing ambiguous pronouns (like \"it\", "
    "\"that\") with explicit information from the conversation history. Keep "
    "exactly the same sentence structure. Do NOT generate or include any "
    "information, words, or values outside of the provided conversation_history "
    "and query."
)

PLANNING_SYSTEM_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values that "
    "are explicitly stated or can be reasonably inferred from the query.\n"
    "<|tool|>{tools}<|/tool|>"
)


def build_rewrite_messages(*, conversation_history: str, query: str):
    user_content = json.dumps(
        {
            "conversation_history": conversation_history,
            "query": query,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_planning_messages(*, tools: str, rewritten_query: str):
    system_msg = PLANNING_SYSTEM_PROMPT.format(tools=tools)
    user_content = f"User Query: {rewritten_query}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def render_messages(tokenizer, messages, assistant_content: str | None = None) -> str:
    if assistant_content is None:
        return apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
    return apply_chat_template(
        tokenizer,
        messages + [{"role": "assistant", "content": assistant_content}],
        add_generation_prompt=False,
    )


def render_rewrite_prompt(
    tokenizer,
    *,
    conversation_history: str,
    query: str,
    assistant_content: str | None = None,
) -> str:
    messages = build_rewrite_messages(
        conversation_history=conversation_history,
        query=query,
    )
    return render_messages(
        tokenizer,
        messages,
        assistant_content=assistant_content,
    )


def render_planning_prompt(
    tokenizer,
    *,
    tools: str,
    rewritten_query: str,
    assistant_content: str | None = None,
) -> str:
    messages = build_planning_messages(
        tools=tools,
        rewritten_query=rewritten_query,
    )
    return render_messages(
        tokenizer,
        messages,
        assistant_content=assistant_content,
    )
