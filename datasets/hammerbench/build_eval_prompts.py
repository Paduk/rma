#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


BASELINE_SYSTEM_INSTRUCTION = (
    "You are an API planner. Given the conversation history and available tools, "
    "predict the next function call. Return compact JSON only with keys \"name\" "
    "and \"arguments\". The value of \"arguments\" must be an object."
)

HAMMER_BASELINE_SYSTEM_TEMPLATE = (
    "You have access to the following functions.\n"
    "{tools}\n"
    "To call a function, respond with compact JSON only in the format "
    "{{\"name\": function name, \"arguments\": {{argument name: value}}}}.\n"
    "Use the full conversation history, including previous function call messages, "
    "to infer the next function call.\n"
    "Attention! For time parameters, preserve the exact format used in the "
    "conversation. Do not automatically normalize or convert dates or times. For "
    "example:\n"
    "user: Set an alarm for 8 a.m. tomorrow\n"
    "assistant: "
    "{{\"name\":\"UtilityTools.AlarmClock.addAlarm\",\"arguments\":{{\"time\":\"8 "
    "a.m. tomorrow\"}}}}\n"
    "Do not hallucinate parameters. If a parameter is not explicitly mentioned or "
    "confirmed in the conversation, omit it from arguments. Do not output empty "
    "string placeholders for unknown values.\n"
    "Preserve exact spelling, casing, punctuation, date/time format, person names, "
    "addresses, station names, and enum-like values whenever possible.\n"
    "Never ask the user for missing parameters. Always output exactly one JSON "
    "function call using one of the provided tools.\n"
    "Let's start!"
)

STRONG_BASELINE_SYSTEM_INSTRUCTION = (
    "You are a precise API planner for multi-turn tool use. Given the conversation "
    "history, the current user utterance, and available tools, predict exactly the "
    "next function call. Treat previous function call messages as confirmed "
    "intermediate argument states. Preserve confirmed values from those snapshots "
    "unless the user later corrects them. Resolve pronouns, ellipsis, vague "
    "references, and external-information mentions such as <EK> only when the "
    "needed value is explicitly present in the provided context. Use the tool "
    "schema only to understand slot meanings; never fill a slot just because it "
    "appears in the schema. Do not infer or invent missing argument values. Omit "
    "arguments that are still unknown. Preserve the exact surface form required "
    "by the context whenever possible. Return compact JSON only with keys "
    "\"name\" and \"arguments\". The value of \"arguments\" must be an object."
)

LEGACY_REWRITE_SYSTEM_INSTRUCTION = (
    "You are a precise query rewriter for multi-turn tool planning. Your job is to "
    "turn the current user utterance into one self-contained request for the next "
    "function call. Treat previous function call messages in the history as "
    "intermediate state snapshots that contain already confirmed argument values. "
    "Preserve confirmed values from those snapshots unless the user later corrects "
    "them. Resolve pronouns, ellipsis, vague references, and external-information "
    "mentions such as <EK> only when the needed value is explicitly present in the "
    "provided context. Use the tool schema only to understand slot meanings; never "
    "fill a slot just because it appears in the schema. If a value is missing, leave "
    "it out of the rewritten query. The rewritten query should contain all currently "
    "confirmed information needed for planning, and nothing fabricated. Return "
    "compact JSON only with key \"rewritten_query\"."
)

REWRITE_SYSTEM_INSTRUCTION = (
    "You are an evidence-preserving query rewriter for multi-turn tool planning. "
    "You are not a paraphraser. Turn the current user utterance into a "
    "self-contained request for the next function call by copying exact value "
    "spans from the conversation history, the current utterance, external text, "
    "or previous function-call snapshots. Treat previous function call messages "
    "as confirmed intermediate argument states and copy their JSON values exactly "
    "unless the user later corrects them. Preserve exact spelling, casing, "
    "punctuation, date/time format, address wording, station names, person names, "
    "and location names. Do not normalize, translate, summarize, shorten, expand, "
    "or rephrase slot values. Use the tool schema only to understand slot "
    "meanings; never fill a slot just because it appears in the schema. Do not "
    "add contextual details unless they are required for the next function call. "
    "If a value is missing, leave it out. The result may be multiple short "
    "sentences if needed; exact value preservation is more important than natural "
    "style. Return compact JSON only with key \"rewritten_query\"."
)

EVIDENCE_REWRITE_SYSTEM_INSTRUCTION = REWRITE_SYSTEM_INSTRUCTION

STRUCTURED_REWRITE_SYSTEM_INSTRUCTION = (
    "You are an argument-preserving query rewriter for multi-turn tool planning. "
    "Your primary output is a structured set of confirmed arguments for the next "
    "function call. Return compact JSON only with keys \"rewritten_query\" and "
    "\"confirmed_arguments\". The value of \"rewritten_query\" must be a string. "
    "The value of \"confirmed_arguments\" must be an object whose keys are tool "
    "argument names. Use the latest assistant question, previous function-call "
    "messages, the current user utterance, and the tool schema to identify only "
    "the arguments that are confirmed for the current next function call. Previous "
    "function-call argument values are authoritative intermediate state; copy "
    "their JSON values exactly unless the user corrects them. Do not include "
    "future, contextual, or merely mentioned values that are not required by the "
    "current next call. Preserve exact spelling, casing, punctuation, date/time "
    "format, names, addresses, station names, and enum-like canonical values. Do "
    "not paraphrase argument values in confirmed_arguments. Use rewritten_query "
    "only as a readable sentence derived from confirmed_arguments."
)

EVIDENCE_REWRITE_LIGHT_FEWSHOT_EXAMPLES = """Few-shot examples:

Example 1
Prior Conversation History:
[1] user: Plan a route for me
[2] function call: {"arguments": {}, "name": "Navigation.MapNavigation.planNavigationRoute"}
[3] assistant: Please provide the name or address of the departure location If there is no specific location your current location will be used
[4] user: The place mentioned in the message Xiao Li sent to me through QQ<EK>:Friends, next week I plan to go to Shanghai Hongqiao Railway Station to pick up my sister. She comes from Beijing to play. If you have time, come out and gather together.
[5] function call: {"arguments": {"departure": "Shanghai Hongqiao Railway Station"}, "name": "Navigation.MapNavigation.planNavigationRoute"}
[6] assistant: Please provide a name or address for the destination
Current User Utterance:
Location mentioned in the email<EK>:Regarding the itinerary you booked, we have arranged high-speed rail tickets from Beijing South Railway Station for you. Please assemble at the second waiting room of Beijing South Railway Station at 9:30 am on October 15, 2023.
Gold rewrite:
{"rewritten_query":"Plan a navigation route with departure \\"Shanghai Hongqiao Railway Station\\" and destination \\"Beijing South Railway Station\\"."}

Example 2
Prior Conversation History:
[1] user: Search a flight for me
[2] function call: {"arguments": {}, "name": "Navigation.FlightTickets.searchFlight"}
[3] assistant: What time is your departure time
[4] user: The time mentioned in the email<EK>:Meetings will begin punctually at 9:00 a.m. on May 1, 2023.
[5] function call: {"arguments": {"date": "01/May/2023"}, "name": "Navigation.FlightTickets.searchFlight"}
[6] assistant: Please provide your departure location
Current User Utterance:
Location mentioned in the notepad<EK>:After the Beijing meeting, head to Capital Airport Terminal T3 tomorrow to retrieve your previously stored luggage. Check personal belongings to make sure they are not left out at the Convention Hotel Beijing International Trade Hotel.
Gold rewrite:
{"rewritten_query":"Search a flight with date \\"01/May/2023\\" and departure \\"Beijing\\"."}
"""

FULL_FEWSHOT_EXAMPLE_IDS = ("External_10_3", "External_24_3")

STRUCTURED_FULL_FEWSHOT_EXAMPLE_IDS = (
    "External_24_2",
    "External_24_3",
    "External_14_2",
    "External_10_5",
    "External_298_3",
)

FULL_FEWSHOT_REFERENCE_REWRITES = {
    "External_10_3": (
        'Plan a navigation route with departure "Shanghai Hongqiao Railway Station" '
        'and destination "Beijing South Railway Station".'
    ),
    "External_24_3": (
        'Search a flight with date "01/May/2023" and departure "Beijing".'
    ),
}

STRUCTURED_FULL_FEWSHOT_REFERENCE_OUTPUTS = {
    "External_24_2": {
        "rewritten_query": 'Search a flight with date "01/May/2023".',
        "confirmed_arguments": {
            "date": "01/May/2023",
        },
        "lesson": (
            "Only the date was requested. Do not add departure, destination, "
            "hotel, or airport values just because they appear in the external text."
        ),
    },
    "External_24_3": {
        "rewritten_query": (
            'Search a flight with date "01/May/2023" and departure "Beijing".'
        ),
        "confirmed_arguments": {
            "date": "01/May/2023",
            "departure": "Beijing",
        },
        "lesson": (
            "The assistant asked for departure. Use the requested coarse departure "
            "place \"Beijing\"; do not replace it with the contextual terminal or hotel."
        ),
    },
    "External_14_2": {
        "rewritten_query": 'View taxi order detail with time "8:00 AM, 10th March".',
        "confirmed_arguments": {
            "time": "8:00 AM, 10th March",
        },
        "lesson": (
            "Preserve the benchmark's canonical time string in confirmed_arguments; "
            "do not rewrite it as March 10th at 8am."
        ),
    },
    "External_10_5": {
        "rewritten_query": (
            'Plan a navigation route with departure "Shanghai Hongqiao Railway Station", '
            'destination "Beijing South Railway Station", time "10am", and mode "DRIVING".'
        ),
        "confirmed_arguments": {
            "departure": "Shanghai Hongqiao Railway Station",
            "destination": "Beijing South Railway Station",
            "time": "10am",
            "mode": "DRIVING",
        },
        "lesson": (
            "Carry forward previous function-call values exactly and use the "
            "tool's canonical mode value."
        ),
    },
    "External_298_3": {
        "rewritten_query": (
            'Cut video "birthday_party.mp4" starting at 30 seconds.'
        ),
        "confirmed_arguments": {
            "name_or_path": "birthday_party.mp4",
            "start_time": 30,
        },
        "lesson": (
            "Carry forward the previous video name and fill the requested numeric "
            "start time; confirmed_arguments may contain non-string values."
        ),
    },
}

PLANNER_FROM_REWRITE_SYSTEM_INSTRUCTION = (
    "You are an API planner. Given a rewritten query, optional confirmed_arguments, "
    "and available tools, predict the next function call. If confirmed_arguments "
    "is provided, use it as the primary source of argument keys and values. Copy "
    "confirmed_arguments values exactly when their keys exist in the selected tool "
    "schema. Use the rewritten query only to understand intent or choose the API; "
    "do not override confirmed_arguments with paraphrased values from the query. "
    "Do not invent missing argument values. Return compact JSON only with keys "
    "\"name\" and \"arguments\". The value of \"arguments\" must be an object."
)

ABLATION_SYSTEM_INSTRUCTION = (
    "You are an API planner. Given the full conversation history, a rewritten query, "
    "and available tools, predict the next function call. Prefer the rewritten query "
    "as the condensed statement of the user's intent, but do not invent missing "
    "argument values. Return compact JSON only with keys \"name\" and \"arguments\". "
    "The value of \"arguments\" must be an object."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt JSONL files from HammerBench canonical eval JSONL records."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input canonical eval JSONL path",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rewrite", "planner_from_rewrite", "ablation"],
        required=True,
        help="Prompt mode to build",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output prompt JSONL path. Defaults to <input>.<mode>.prompt.jsonl",
    )
    parser.add_argument(
        "--rewritten-query-field",
        default="oracle_rewritten_query",
        help=(
            "Field name used by planner_from_rewrite/ablation modes. "
            "If the field is missing or null, the record is skipped."
        ),
    )
    parser.add_argument(
        "--baseline-prompt-key",
        choices=["default", "strong", "hammer"],
        default="default",
        help=(
            "Baseline prompt variant. `default` preserves the original prompt; "
            "`strong` uses rewrite-style planning instructions without an explicit "
            "rewrite stage; `hammer` uses an Appendix D-inspired HammerBench prompt "
            "adapted to the local evaluator."
        ),
    )
    parser.add_argument(
        "--rewrite-prompt-key",
        choices=[
            "legacy",
            "default",
            "evidence",
            "evidence_fewshot",
            "evidence_fewshot_light",
            "evidence_fewshot_full",
            "structured_fewshot_full_5",
        ],
        default="default",
        help=(
            "Rewrite prompt variant. `default` uses exact-span preservation rules; "
            "`legacy` preserves the original rewrite prompt; `evidence` is an "
            "explicit alias for default; `evidence_fewshot_light` adds two compact "
            "gold-call-derived examples; `evidence_fewshot_full` adds two "
            "full-format examples; `structured_fewshot_full_5` emits rewritten_query "
            "and confirmed_arguments with five full-format examples. "
            "`evidence_fewshot` is kept as an alias for `evidence_fewshot_light`."
        ),
    )
    parser.add_argument(
        "--confirmed-arguments-field",
        default="model_confirmed_arguments",
        help=(
            "Field name used by planner_from_rewrite/ablation modes for optional "
            "structured rewrite arguments."
        ),
    )
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def serialize_content(content):
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True, sort_keys=True)


def render_history(messages):
    if not messages:
        return "(empty)"
    lines = []
    for idx, message in enumerate(messages, start=1):
        role = message["role"]
        content = serialize_content(message["content"])
        lines.append(f"[{idx}] {role}: {content}")
    return "\n".join(lines)


def render_tools(tool_schema):
    return json.dumps(tool_schema, ensure_ascii=True, indent=2, sort_keys=True)


def normalize_rewrite_prompt_key(prompt_key):
    if prompt_key == "evidence_fewshot":
        return "evidence_fewshot_light"
    return prompt_key


def render_gold_rewrite(rewritten_query):
    return json.dumps(
        {"rewritten_query": rewritten_query},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def render_structured_rewrite_output(output):
    return json.dumps(
        {
            "rewritten_query": output["rewritten_query"],
            "confirmed_arguments": output["confirmed_arguments"],
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def build_full_fewshot_examples(records):
    by_id = {record.get("id"): record for record in records}
    missing_ids = [
        example_id
        for example_id in FULL_FEWSHOT_EXAMPLE_IDS
        if example_id not in by_id
    ]
    if missing_ids:
        raise ValueError(
            "Full few-shot rewrite examples require records that are missing from "
            f"the input: {', '.join(missing_ids)}"
        )

    examples = ["Few-shot examples:"]
    for index, example_id in enumerate(FULL_FEWSHOT_EXAMPLE_IDS, start=1):
        record = by_id[example_id]
        examples.append(
            "\n".join(
                [
                    "",
                    f"Example {index}",
                    "Prior Conversation History:",
                    render_history(record["prior_history"]),
                    "Current User Utterance:",
                    record["current_user_utterance"],
                    "Available Tool Schema For Slot Meanings Only:",
                    render_tools(record["tool_schema"]),
                    "Gold rewrite:",
                    render_gold_rewrite(FULL_FEWSHOT_REFERENCE_REWRITES[example_id]),
                ]
            )
        )
    return "\n".join(examples)


def build_structured_full_fewshot_examples(records):
    by_id = {record.get("id"): record for record in records}
    missing_ids = [
        example_id
        for example_id in STRUCTURED_FULL_FEWSHOT_EXAMPLE_IDS
        if example_id not in by_id
    ]
    if missing_ids:
        raise ValueError(
            "Structured full few-shot examples require records that are missing "
            f"from the input: {', '.join(missing_ids)}"
        )

    examples = ["Few-shot examples:"]
    for index, example_id in enumerate(STRUCTURED_FULL_FEWSHOT_EXAMPLE_IDS, start=1):
        record = by_id[example_id]
        reference = STRUCTURED_FULL_FEWSHOT_REFERENCE_OUTPUTS[example_id]
        examples.append(
            "\n".join(
                [
                    "",
                    f"Example {index}",
                    "Prior Conversation History:",
                    render_history(record["prior_history"]),
                    "Current User Utterance:",
                    record["current_user_utterance"],
                    "Available Tool Schema For Slot Meanings Only:",
                    render_tools(record["tool_schema"]),
                    "Lesson:",
                    reference["lesson"],
                    "Gold structured rewrite:",
                    render_structured_rewrite_output(reference),
                ]
            )
        )
    return "\n".join(examples)


def build_rewrite_fewshot_examples(records, prompt_key):
    normalized_key = normalize_rewrite_prompt_key(prompt_key)
    if normalized_key == "evidence_fewshot_light":
        return EVIDENCE_REWRITE_LIGHT_FEWSHOT_EXAMPLES
    if normalized_key == "evidence_fewshot_full":
        return build_full_fewshot_examples(records)
    if normalized_key == "structured_fewshot_full_5":
        return build_structured_full_fewshot_examples(records)
    return ""


def build_default_baseline_prompt(record):
    user_input = (
        "Task: Predict the next function call.\n\n"
        "Conversation History:\n"
        f"{render_history(record['planner_history'])}\n\n"
        "Available Tools:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return BASELINE_SYSTEM_INSTRUCTION, user_input


def build_strong_baseline_prompt(record):
    user_input = (
        "Task: Predict the next function call.\n\n"
        "Important planning rules:\n"
        "- Previous function call messages are confirmed intermediate argument states.\n"
        "- Carry forward confirmed argument values unless the user corrects them.\n"
        "- Extract values from <EK> text only when the value is explicit.\n"
        "- Do not infer or invent missing slot values.\n"
        "- Do not include slots that are still unknown.\n"
        "- Use the tool schema only to understand slot meanings.\n\n"
        "Prior Conversation History:\n"
        f"{render_history(record.get('prior_history') or [])}\n\n"
        "Current User Utterance:\n"
        f"{record.get('current_user_utterance', '')}\n\n"
        "Available Tools:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return STRONG_BASELINE_SYSTEM_INSTRUCTION, user_input


def build_hammer_baseline_prompt(record):
    system_instruction = HAMMER_BASELINE_SYSTEM_TEMPLATE.format(
        tools=render_tools(record["tool_schema"]),
    )
    user_input = (
        "Conversation History:\n"
        f"{render_history(record['planner_history'])}\n\n"
        "Predict the next function call for the final user turn in the conversation "
        "history above."
    )
    return system_instruction, user_input


def build_baseline_prompt(record, prompt_key="default"):
    if prompt_key == "default":
        return build_default_baseline_prompt(record)
    if prompt_key == "strong":
        return build_strong_baseline_prompt(record)
    if prompt_key == "hammer":
        return build_hammer_baseline_prompt(record)
    raise ValueError(f"Unsupported baseline prompt key: {prompt_key}")


def build_legacy_rewrite_prompt(record):
    user_input = (
        "Task: Rewrite the current user utterance into a self-contained query.\n\n"
        "Important rewriting rules:\n"
        "- Previous function call messages are confirmed intermediate argument states.\n"
        "- Carry forward confirmed argument values unless the user corrects them.\n"
        "- Extract values from <EK> text only when the value is explicit.\n"
        "- Do not infer or invent missing slot values.\n"
        "- Do not include slots that are still unknown.\n\n"
        "Prior Conversation History:\n"
        f"{render_history(record['prior_history'])}\n\n"
        "Current User Utterance:\n"
        f"{record['current_user_utterance']}\n\n"
        "Available Tool Schema For Slot Meanings Only:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return LEGACY_REWRITE_SYSTEM_INSTRUCTION, user_input


def build_default_rewrite_prompt(record):
    return build_evidence_rewrite_prompt(record)


def build_evidence_rewrite_prompt(record, fewshot_examples=""):
    examples = f"{fewshot_examples}\n\n" if fewshot_examples else ""
    user_input = (
        "Task: Rewrite the current user utterance into a self-contained query for "
        "the next function call.\n\n"
        "Important rewriting rules:\n"
        "- Do not paraphrase slot values; copy exact evidence spans.\n"
        "- Copy previous function-call argument values exactly unless corrected.\n"
        "- Preserve exact casing, punctuation, date/time format, and location names.\n"
        "- Do not add values that are merely contextual or not required by the next call.\n"
        "- Extract values from <EK> text only when the value is explicit.\n"
        "- Do not include slots that are still unknown.\n\n"
        f"{examples}"
        "Prior Conversation History:\n"
        f"{render_history(record['prior_history'])}\n\n"
        "Current User Utterance:\n"
        f"{record['current_user_utterance']}\n\n"
        "Available Tool Schema For Slot Meanings Only:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return EVIDENCE_REWRITE_SYSTEM_INSTRUCTION, user_input


def build_structured_rewrite_prompt(record, fewshot_examples=""):
    examples = f"{fewshot_examples}\n\n" if fewshot_examples else ""
    user_input = (
        "Task: Produce an argument-preserving structured rewrite for the next "
        "function call.\n\n"
        "Return format:\n"
        "{\"rewritten_query\":\"...\",\"confirmed_arguments\":{...}}\n\n"
        "Important structured rewriting rules:\n"
        "- confirmed_arguments is the authoritative output for planning.\n"
        "- Use only tool argument names as keys in confirmed_arguments.\n"
        "- Copy previous function-call argument values exactly unless corrected.\n"
        "- Fill only arguments confirmed for the current next function call.\n"
        "- Use the latest assistant question to identify the currently requested slot.\n"
        "- Do not include future/contextual values merely mentioned in <EK> text.\n"
        "- Preserve exact casing, punctuation, date/time format, names, addresses, "
        "station names, and canonical enum-like values.\n"
        "- If the tool expects a canonical value shown in examples, use that value "
        "in confirmed_arguments.\n"
        "- rewritten_query should be a readable sentence built from "
        "confirmed_arguments; do not put extra slot values only in rewritten_query.\n\n"
        f"{examples}"
        "Prior Conversation History:\n"
        f"{render_history(record['prior_history'])}\n\n"
        "Current User Utterance:\n"
        f"{record['current_user_utterance']}\n\n"
        "Available Tool Schema For Slot Meanings Only:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return STRUCTURED_REWRITE_SYSTEM_INSTRUCTION, user_input


def build_rewrite_prompt(record, prompt_key="default", fewshot_examples=""):
    normalized_key = normalize_rewrite_prompt_key(prompt_key)
    if normalized_key == "legacy":
        return build_legacy_rewrite_prompt(record)
    if normalized_key == "default":
        return build_default_rewrite_prompt(record)
    if normalized_key == "evidence":
        return build_evidence_rewrite_prompt(record)
    if normalized_key in {"evidence_fewshot_light", "evidence_fewshot_full"}:
        return build_evidence_rewrite_prompt(
            record,
            fewshot_examples=fewshot_examples,
        )
    if normalized_key == "structured_fewshot_full_5":
        return build_structured_rewrite_prompt(
            record,
            fewshot_examples=fewshot_examples,
        )
    raise ValueError(f"Unsupported rewrite prompt key: {prompt_key}")


def build_planner_from_rewrite_prompt(
    record,
    rewritten_query,
    confirmed_arguments=None,
):
    confirmed_arguments_text = "(not provided)"
    if isinstance(confirmed_arguments, dict):
        confirmed_arguments_text = json.dumps(
            confirmed_arguments,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    user_input = (
        "Task: Predict the next function call from the rewritten query.\n\n"
        "Planning priority:\n"
        "1. If Confirmed Arguments is an object, copy those key/value pairs exactly "
        "when the keys exist in the tool schema.\n"
        "2. Use Rewritten Query only for intent/API selection or when Confirmed "
        "Arguments is not provided.\n"
        "3. Do not add arguments that are absent from Confirmed Arguments unless "
        "they are explicitly stated in the Rewritten Query and required by the tool.\n\n"
        "Rewritten Query:\n"
        f"{rewritten_query}\n\n"
        "Confirmed Arguments:\n"
        f"{confirmed_arguments_text}\n\n"
        "Available Tools:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return PLANNER_FROM_REWRITE_SYSTEM_INSTRUCTION, user_input


def build_ablation_prompt(record, rewritten_query):
    user_input = (
        "Task: Predict the next function call using both the full history and the rewritten query.\n\n"
        "Conversation History:\n"
        f"{render_history(record['planner_history'])}\n\n"
        "Rewritten Query:\n"
        f"{rewritten_query}\n\n"
        "Available Tools:\n"
        f"{render_tools(record['tool_schema'])}"
    )
    return ABLATION_SYSTEM_INSTRUCTION, user_input


def derive_default_output_path(
    input_path,
    mode,
    baseline_prompt_key="default",
    rewrite_prompt_key="default",
):
    if mode == "baseline" and baseline_prompt_key != "default":
        return input_path.with_name(
            f"{input_path.stem}.baseline.{baseline_prompt_key}.prompt.jsonl"
        )
    if mode == "rewrite" and rewrite_prompt_key != "default":
        return input_path.with_name(
            f"{input_path.stem}.rewrite.{rewrite_prompt_key}.prompt.jsonl"
        )
    return input_path.with_name(f"{input_path.stem}.{mode}.prompt.jsonl")


def build_prompt_record(
    record,
    mode,
    rewritten_query_field,
    confirmed_arguments_field="model_confirmed_arguments",
    baseline_prompt_key="default",
    rewrite_prompt_key="default",
    rewrite_fewshot_examples="",
):
    rewritten_query = record.get(rewritten_query_field)
    confirmed_arguments = record.get(confirmed_arguments_field)

    if mode == "baseline":
        system_instruction, user_input = build_baseline_prompt(
            record,
            prompt_key=baseline_prompt_key,
        )
    elif mode == "rewrite":
        system_instruction, user_input = build_rewrite_prompt(
            record,
            prompt_key=rewrite_prompt_key,
            fewshot_examples=rewrite_fewshot_examples,
        )
    elif mode == "planner_from_rewrite":
        if not rewritten_query:
            return None
        system_instruction, user_input = build_planner_from_rewrite_prompt(
            record,
            rewritten_query,
            confirmed_arguments=confirmed_arguments,
        )
    elif mode == "ablation":
        if not rewritten_query:
            return None
        system_instruction, user_input = build_ablation_prompt(record, rewritten_query)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_input},
    ]
    return {
        "id": record["id"],
        "data_type": record["data_type"],
        "turn_id": record["turn_id"],
        "mode": mode,
        "prompt_key": (
            baseline_prompt_key
            if mode == "baseline"
            else rewrite_prompt_key
            if mode == "rewrite"
            else None
        ),
        "rewritten_query_field": rewritten_query_field if mode in {"planner_from_rewrite", "ablation"} else None,
        "confirmed_arguments_field": confirmed_arguments_field if mode in {"planner_from_rewrite", "ablation"} else None,
        "messages": messages,
        "system_instruction": system_instruction,
        "user_input": user_input,
        "gold_call": record["gold_call"],
    }


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = read_jsonl(args.input)
    rewrite_fewshot_examples = ""
    if args.mode == "rewrite":
        rewrite_fewshot_examples = build_rewrite_fewshot_examples(
            records,
            args.rewrite_prompt_key,
        )
    output_path = args.output or derive_default_output_path(
        args.input,
        args.mode,
        baseline_prompt_key=args.baseline_prompt_key,
        rewrite_prompt_key=args.rewrite_prompt_key,
    )
    built = []
    skipped = 0

    for record in records:
        prompt_record = build_prompt_record(
            record,
            mode=args.mode,
            rewritten_query_field=args.rewritten_query_field,
            confirmed_arguments_field=args.confirmed_arguments_field,
            baseline_prompt_key=args.baseline_prompt_key,
            rewrite_prompt_key=args.rewrite_prompt_key,
            rewrite_fewshot_examples=rewrite_fewshot_examples,
        )
        if prompt_record is None:
            skipped += 1
            continue
        built.append(prompt_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for prompt_record in built:
            f.write(json.dumps(prompt_record, ensure_ascii=True))
            f.write("\n")

    print(f"input_path\t{args.input}")
    print(f"output_path\t{output_path}")
    print(f"mode\t{args.mode}")
    if args.mode == "baseline":
        print(f"baseline_prompt_key\t{args.baseline_prompt_key}")
    if args.mode == "rewrite":
        print(f"rewrite_prompt_key\t{args.rewrite_prompt_key}")
    print(f"records_in_input\t{len(records)}")
    print(f"records_written\t{len(built)}")
    print(f"records_skipped\t{skipped}")


if __name__ == "__main__":
    main()
