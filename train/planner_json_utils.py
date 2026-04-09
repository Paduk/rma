from __future__ import annotations

import ast
import json


STRICT_JSON_OUTPUT_INSTRUCTION = (
    'Return only valid JSON with keys "plan" and "arguments". '
    "Use double quotes for all keys and string values. "
    "Do not output markdown, code fences, explanations, or any extra text."
)


def normalize_answer_to_dict(answer):
    if isinstance(answer, dict):
        return answer

    if not isinstance(answer, str):
        raise TypeError(f"Unsupported answer type: {type(answer)!r}")

    stripped = answer.strip()
    if not stripped:
        raise ValueError("Answer is empty.")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    parsed = ast.literal_eval(stripped)
    if isinstance(parsed, dict):
        return parsed

    raise ValueError("Answer could not be normalized into a dict.")


def serialize_answer_to_json(answer, *, indent=None):
    return json.dumps(
        normalize_answer_to_dict(answer),
        ensure_ascii=False,
        indent=indent,
    )
