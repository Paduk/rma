#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI


class GeminiRateLimitError(RuntimeError):
    pass


class ClaudeQuotaError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HammerBench prompt JSONL records against cloud LLMs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input prompt JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output prediction JSONL path. Defaults to <input>.pred.jsonl",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Cloud model alias/name. OpenAI examples: gpt-5-mini, gpt-5.4-mini, "
            "o4-mini. Gemini examples: gemini-2.0-flash, gemini-2.5-flash. "
            "Claude examples: claude-3-5-sonnet-latest, claude-sonnet-4-20250514."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort for supported OpenAI reasoning models. Ignored for Gemini and Claude.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=4,
        help="Number of concurrent cloud requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of input records to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a few prompts and skip cloud requests",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=2,
        help="Examples to print in --dry-run mode",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}.pred.jsonl")


def resolve_model_spec(alias: str) -> Tuple[str, str]:
    openai_aliases = {
        "o3": "o3-mini",
        "o4-mini": "o4-mini",
        "gpt-4.1-2025-04-14": "gpt-4.1-2025-04-14",
        "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
        "gpt-5.1": "gpt-5.1",
        "gpt-5.4": "gpt-5.4",
        "gpt-5.4-mini": "gpt-5.4-mini",
        "gpt-5.4-nano": "gpt-5.4-nano",
        "gpt-5.4-2026-03-05": "gpt-5.4-2026-03-05",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
    }
    if "gemini" in alias.lower():
        return "gemini", alias
    if "claude" in alias.lower():
        return "anthropic", alias
    return "openai", openai_aliases.get(alias, alias)


def import_google_genai():
    try:
        from google import genai
    except Exception:
        genai = importlib.import_module("google.genai")
    return genai


def build_gemini_contents(messages):
    non_system_messages = [
        message for message in messages if message.get("role") != "system"
    ]
    if not non_system_messages:
        return ""
    if len(non_system_messages) == 1:
        return non_system_messages[0].get("content", "")

    lines = []
    for message in non_system_messages:
        role = str(message.get("role") or "user").upper()
        content = message.get("content", "")
        lines.append(f"{role}:\n{content}")
    return "\n\n".join(lines)


def request_gemini_chat(client, model_name, messages, temperature, max_output_tokens):
    types = importlib.import_module("google.genai.types")

    system_instruction = None
    if messages and messages[0].get("role") == "system":
        system_instruction = messages[0].get("content", "")

    response = client.models.generate_content(
        model=model_name,
        contents=build_gemini_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    finish_reason = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)

    payload = {
        "text": getattr(response, "text", "") or "",
        "finish_reason": str(finish_reason) if finish_reason is not None else None,
    }
    if usage is not None:
        payload["prompt_tokens"] = getattr(usage, "prompt_token_count", None)
        payload["completion_tokens"] = getattr(usage, "candidates_token_count", None)
    return payload


def request_openai_chat(client, model_name, messages, reasoning_effort):
    request_kwargs = {
        "model": model_name,
        "messages": messages,
    }
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(**request_kwargs)
    usage = getattr(completion, "usage", None)
    response = {
        "text": completion.choices[0].message.content or "",
        "finish_reason": completion.choices[0].finish_reason,
    }
    if usage is not None:
        response["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        response["completion_tokens"] = getattr(usage, "completion_tokens", None)
    return response


def is_gemini_rate_limit_error(exc):
    message = str(exc).lower()
    markers = [
        "resource_exhausted",
        "quota exceeded",
        "rate-limits",
        "retrydelay",
        "generatecontentinputtokenspermodelperminute-freetier",
        "generaterequestsperminuteperprojectpermodel-freetier",
        "generaterequestsperdayperprojectpermodel-freetier",
    ]
    return any(marker in message for marker in markers)


def is_claude_quota_error(exc):
    message = str(exc).lower()
    markers = [
        "credit balance is too low",
        "insufficient credits",
        "payment required",
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
        "usage limit",
    ]
    try:
        import anthropic
    except ImportError:
        anthropic = None

    if anthropic is not None:
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code in {402, 429}:
                return True
            body = getattr(exc, "body", None)
            if body is not None and any(marker in str(body).lower() for marker in markers):
                return True

    return any(marker in message for marker in markers)


def build_anthropic_messages(messages):
    anthropic_messages = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            continue
        anthropic_messages.append(
            {
                "role": role,
                "content": str(message.get("content", "")),
            }
        )
    return anthropic_messages


def request_anthropic_chat(client, model_name, messages, temperature, max_output_tokens):
    system_instruction = None
    if messages and messages[0].get("role") == "system":
        system_instruction = messages[0].get("content", "")

    response = client.messages.create(
        model=model_name,
        system=system_instruction,
        messages=build_anthropic_messages(messages),
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    text_parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            text_parts.append(block.text)

    usage = getattr(response, "usage", None)
    payload = {
        "text": "".join(text_parts),
        "finish_reason": getattr(response, "stop_reason", None),
    }
    if usage is not None:
        payload["prompt_tokens"] = getattr(usage, "input_tokens", None)
        payload["completion_tokens"] = getattr(usage, "output_tokens", None)
    return payload


def extract_first_json_object(text):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def requires_confirmed_arguments(record):
    return record.get("mode") == "rewrite" and str(
        record.get("prompt_key") or ""
    ).startswith("structured_")


def parse_model_output(record, raw_text):
    mode = record["mode"]
    parsed = extract_first_json_object(raw_text)
    if parsed is None:
        return None, "no_json_object_found"

    if mode == "rewrite":
        rewritten_query = (
            parsed.get("rewritten_query")
            or parsed.get("rewrited_query")
            or parsed.get("query")
        )
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            return None, "missing_rewritten_query"
        confirmed_arguments = parsed.get("confirmed_arguments")
        if confirmed_arguments is None:
            if requires_confirmed_arguments(record):
                return None, "missing_confirmed_arguments"
            return {"rewritten_query": rewritten_query.strip()}, None
        if not isinstance(confirmed_arguments, dict):
            return None, "confirmed_arguments_not_object"
        return {
            "rewritten_query": rewritten_query.strip(),
            "confirmed_arguments": confirmed_arguments,
        }, None

    name = parsed.get("name") or parsed.get("plan")
    arguments = parsed.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None, "missing_name"
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"
    return {"name": name.strip(), "arguments": arguments}, None


def predict_one(idx, record, provider, client, model_name, temperature, max_output_tokens, reasoning_effort):
    try:
        if provider == "openai":
            response = request_openai_chat(
                client=client,
                model_name=model_name,
                messages=record["messages"],
                reasoning_effort=reasoning_effort,
            )
        elif provider == "gemini":
            response = request_gemini_chat(
                client=client,
                model_name=model_name,
                messages=record["messages"],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        elif provider == "anthropic":
            response = request_anthropic_chat(
                client=client,
                model_name=model_name,
                messages=record["messages"],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        raw_text = response.get("text", "")
        parsed_output, parse_error = parse_model_output(record, raw_text)
    except Exception as exc:
        if provider == "gemini" and is_gemini_rate_limit_error(exc):
            raise GeminiRateLimitError(
                f"Gemini rate/quota issue at id={record['id']}: {type(exc).__name__}: {exc}"
            ) from exc
        if provider == "anthropic" and is_claude_quota_error(exc):
            raise ClaudeQuotaError(
                f"Claude billing/quota issue at id={record['id']}: {type(exc).__name__}: {exc}"
            ) from exc
        response = {}
        raw_text = ""
        parsed_output = None
        parse_error = f"request_failed: {type(exc).__name__}: {exc}"
    result = {
        "id": record["id"],
        "data_type": record["data_type"],
        "turn_id": record["turn_id"],
        "mode": record["mode"],
        "model": model_name,
        "parse_ok": parse_error is None,
        "parse_error": parse_error,
        "raw_response": raw_text,
        "parsed_output": parsed_output,
        "gold_call": record.get("gold_call"),
        "finish_reason": response.get("finish_reason"),
        "prompt_tokens": response.get("prompt_tokens"),
        "completion_tokens": response.get("completion_tokens"),
    }
    if record["mode"] == "rewrite":
        result["rewritten_query"] = (
            parsed_output.get("rewritten_query") if parsed_output else None
        )
        result["confirmed_arguments"] = (
            parsed_output.get("confirmed_arguments") if parsed_output else None
        )
    else:
        result["pred_call"] = parsed_output
    return idx, result


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    output_path = args.output or derive_output_path(args.input)

    if args.dry_run:
        for record in records[: args.dry_run_limit]:
            print(f"id\t{record['id']}")
            print(f"mode\t{record['mode']}")
            print("messages")
            print(json.dumps(record["messages"], ensure_ascii=True, indent=2))
            print("---")
        print(f"records_in_input\t{len(records)}")
        print("dry_run\tTrue")
        return

    provider, model_name = resolve_model_spec(args.model)
    if provider == "openai":
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", None),
            base_url="https://api.openai.com/v1",
        )
    elif provider == "gemini":
        try:
            genai = import_google_genai()
        except ImportError as exc:
            raise ImportError(
                "Gemini support requires the official `google-genai` package."
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY (or GOOGLE_API_KEY)."
            )
        client = genai.Client(api_key=api_key)
    elif provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Claude support requires the official `anthropic` package."
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Claude API key not found. Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY)."
            )
        client = Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    results = [None] * len(records)
    try:
        if args.num_parallel <= 1:
            for idx, record in enumerate(records):
                _, result = predict_one(
                    idx,
                    record,
                    provider,
                    client,
                    model_name,
                    temperature=0.0,
                    max_output_tokens=512,
                    reasoning_effort=args.reasoning_effort,
                )
                results[idx] = result
        else:
            executor = ThreadPoolExecutor(max_workers=args.num_parallel)
            try:
                futures = {
                    executor.submit(
                        predict_one,
                        idx,
                        record,
                        provider,
                        client,
                        model_name,
                        0.0,
                        512,
                        args.reasoning_effort,
                    ): idx
                    for idx, record in enumerate(records)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        _, result = future.result()
                    except (GeminiRateLimitError, ClaudeQuotaError):
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    results[idx] = result
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    except (GeminiRateLimitError, ClaudeQuotaError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=True))
            f.write("\n")

    parse_ok_count = sum(1 for result in results if result["parse_ok"])
    print(f"input_path\t{args.input}")
    print(f"output_path\t{output_path}")
    print(f"provider\t{provider}")
    print(f"model\t{model_name}")
    print(f"reasoning_effort\t{args.reasoning_effort}")
    print(f"records_processed\t{len(results)}")
    print(f"parse_ok\t{parse_ok_count}")
    print(f"parse_failed\t{len(results) - parse_ok_count}")
    prompt_tokens = sum(
        result.get("prompt_tokens") or 0
        for result in results
    )
    completion_tokens = sum(
        result.get("completion_tokens") or 0
        for result in results
    )
    print(f"prompt_tokens\t{prompt_tokens}")
    print(f"completion_tokens\t{completion_tokens}")


if __name__ == "__main__":
    main()
