#!/usr/bin/env python3
import argparse
import ast
import csv
import re
import shutil
from copy import deepcopy
from pathlib import Path


TARGET_PLANS = [
    "ACTION_EDIT_ALARM",
    "ACTION_SET_RINGTONE",
    "ACTION_NAVIGATE_TO_LOCATION",
    "ACTION_SET_TIMER",
    "ACTION_VIDEO_CAPTURE",
]

WEEKDAY_TRIPLE = ["Monday", "Wednesday", "Friday"]
VIDEO_CAPTURE_SECONDS = [30, 45, 20, 35, 25, 40, 15, 50]
NAVIGATION_QUERIES = [
    "Get directions there.",
    "Navigate there.",
    "Drive me there.",
    "Take me there.",
    "Guide me there.",
    "Start navigation there.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Harden selected plan rows inside it6_complex_1_tc.tsv using templates "
            "derived from hard failure patterns observed in other turn-6 buckets."
        )
    )
    parser.add_argument(
        "--input",
        default="/home/hj153lee/new-rma/datasets/tc/scale_turn6_refswap_backfill/it6_complex_1_tc.tsv",
        help="Target it6_complex_1_tc.tsv file to edit in place.",
    )
    parser.add_argument(
        "--backup",
        default=None,
        help="Optional explicit backup path. Default: <input>.before_harden_failures",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional report path. Default: sibling TSV named it6_complex_1_hardening_report.tsv",
    )
    return parser.parse_args()


def parse_literal(value, fallback):
    try:
        return ast.literal_eval(value)
    except Exception:
        return fallback


def parse_history(value):
    parsed = parse_literal(value, [])
    if not isinstance(parsed, list):
        raise ValueError("conversation_history must parse to a list.")
    return [str(item) for item in parsed]


def parse_answer(value):
    parsed = parse_literal(value, {})
    if not isinstance(parsed, dict):
        return {}
    return parsed


def write_tsv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def format_time_12h(hour: int, minute: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minute:02d} {suffix}"


def shift_time(hour: int, minute: int, delta_minutes: int) -> tuple[int, int]:
    total = (hour * 60 + minute + delta_minutes) % (24 * 60)
    return total // 60, total % 60


def extract_alarm_descriptor(text: str) -> str:
    quoted = re.search(r"'(alarm_id:\s*\d+\s*-\s*[^']+)'", text)
    if quoted:
        return quoted.group(1)
    plain = re.search(r"(alarm_id:\s*\d+\s*-\s*[^.,]+(?:,[^.,]+)?)", text)
    if plain:
        return plain.group(1).strip()
    return ""


def extract_alarm_label(descriptor: str) -> str:
    parts = descriptor.split(",")
    if len(parts) >= 2:
        return parts[1].strip().strip("'")
    return "Updated Alarm"


def extract_alarm_old_time(descriptor: str) -> str:
    match = re.search(r"-\s*([^,]+)", descriptor)
    if match:
        return match.group(1).strip()
    return "that time"


def extract_alarm_descriptor_from_turn1(turn1: str, alarm_id: int | None) -> str:
    if alarm_id is None:
        return ""
    for match in re.finditer(r"(alarm_id:\s*(\d+)\s*-\s*[^,\]]+,\s*[^,\]]+)", turn1):
        descriptor = match.group(1).strip()
        found_id = int(match.group(2))
        if found_id == alarm_id:
            return descriptor
    return ""


def humanize_uri(uri: str) -> str:
    tail = uri.rstrip("/").split("/")[-1]
    tail = re.sub(r"\.[A-Za-z0-9]+$", "", tail)
    tail = tail.replace("_", " ").replace("-", " ").strip()
    return tail or "selected tone"


def build_ringtone_turn1(uri: str, audio_id: int, style: int) -> str:
    if style == 0:
        return (
            f"turn 1: Show me some other tones. -> Here's your pick: "
            f"audio file with ID {audio_id} ({uri})."
        )
    return (
        f"turn 1: Show me some other tones. -> Here's your pick: "
        f"'{humanize_uri(uri)}' ({uri})."
    )


def extract_location(answer_dict: dict) -> str:
    return str(answer_dict.get("arguments", {}).get("location", "")).strip()


def timer_title_case(text: str) -> str:
    value = text.strip()
    if not value:
        return "Focus"
    return value[0].upper() + value[1:]


def parse_timer_candidates(turn1: str):
    assistant = turn1.split("->", 1)[1] if "->" in turn1 else turn1
    text = assistant.replace("–", "-").replace("—", "-")
    text = text.replace("[", "").replace("]", "")
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    candidates = []

    patterns = [
        re.compile(r"(\d+)\s*(minutes?|mins?|min)\s*-\s*([^,.;]+)"),
        re.compile(r"([^,.;:]+?)\s*-\s*(\d+)\s*(minutes?|mins?|min)(?:\s*remaining)?"),
        re.compile(r"(\d+)\s*(minutes?|mins?|min)(?:\s*remaining)?\s*(?:left on|on|for)\s*([^,.;]+)"),
        re.compile(r"([^,.;:]+?)\s*with\s*(\d+)\s*minutes?\s*to\s*go"),
        re.compile(r"([^,.;:]+?)\s*at\s*(\d+)\s*minutes?"),
    ]

    seen = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = [group.strip() for group in match.groups() if group]
            if not groups:
                continue
            if groups[0].isdigit():
                minutes = int(groups[0])
                label = groups[-1]
            else:
                label = groups[0]
                minute_group = next((item for item in groups if item.isdigit()), None)
                if minute_group is None:
                    continue
                minutes = int(minute_group)
            label = re.sub(r"^(the|your)\s+", "", label.strip(), flags=re.IGNORECASE)
            label = re.sub(r"^\d+\)\s*", "", label)
            label = label.strip(" .")
            if not label:
                continue
            key = (minutes, label.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append((minutes, label))
    return candidates


def harden_edit_alarm(row: dict, row_index: int):
    answer_dict = parse_answer(row["answer"])
    args = deepcopy(answer_dict.get("arguments", {}))
    history = parse_history(row["conversation_history"])
    alarm_id = args.get("alarm_id")
    descriptor = extract_alarm_descriptor_from_turn1(history[0], alarm_id)
    if not descriptor:
        descriptor = extract_alarm_descriptor(row["rewrited_query"])
    label = extract_alarm_label(descriptor)
    old_time = extract_alarm_old_time(descriptor)

    current_hour = int(args.get("EXTRA_HOUR", 7))
    current_minute = int(args.get("EXTRA_MINUTES", 0))
    new_hour, new_minute = shift_time(current_hour, current_minute, 35 + (row_index % 3) * 10)
    new_time = format_time_12h(new_hour, new_minute)
    new_label = f"{label} Refresh"

    template_idx = row_index % 3
    if template_idx == 0:
        query = (
            f"Set that one for {new_time} and make it repeat on Monday, "
            f"Wednesday, Friday with vibration disabled."
        )
        rewrited_query = (
            f"Set {descriptor} for {new_time} and make it repeat on Monday, "
            f"Wednesday, Friday with vibration disabled."
        )
        args.update(
            {
                "EXTRA_HOUR": new_hour,
                "EXTRA_MINUTES": new_minute,
                "EXTRA_DAYS": WEEKDAY_TRIPLE,
                "EXTRA_VIBRATE": False,
            }
        )
    elif template_idx == 1:
        query = (
            f"Move that {old_time} alarm to {new_time}, skip the confirmation "
            f"screen, and turn vibration off."
        )
        rewrited_query = (
            f"Move {descriptor} to {new_time}, skip the confirmation screen, "
            f"and turn vibration off."
        )
        args.update(
            {
                "EXTRA_HOUR": new_hour,
                "EXTRA_MINUTES": new_minute,
                "EXTRA_SKIP_UI": True,
                "EXTRA_VIBRATE": False,
            }
        )
    else:
        query = (
            f"Make that {old_time} alarm go off at {new_time}, label it "
            f"'{new_label}', and keep it only on Monday, Wednesday, Friday."
        )
        rewrited_query = (
            f"Make {descriptor} go off at {new_time}, label it '{new_label}', "
            f"and keep it only on Monday, Wednesday, Friday."
        )
        args.update(
            {
                "EXTRA_HOUR": new_hour,
                "EXTRA_MINUTES": new_minute,
                "EXTRA_MESSAGE": new_label,
                "EXTRA_DAYS": WEEKDAY_TRIPLE,
            }
        )

    answer_dict["arguments"] = args
    row["query"] = query
    row["rewrited_query"] = rewrited_query
    row["answer"] = str(answer_dict)
    return "alarm_multi_field_edit"


def harden_set_ringtone(row: dict, row_index: int):
    answer_dict = parse_answer(row["answer"])
    uri = str(answer_dict.get("arguments", {}).get("ringtone_uri", "")).strip()
    history = parse_history(row["conversation_history"])
    audio_id = 401 + row_index
    style = row_index % 2
    history[0] = build_ringtone_turn1(uri=uri, audio_id=audio_id, style=style)

    if style == 0:
        row["query"] = f"Fine, let’s go with the audio file with ID {audio_id}—make it my ringtone."
        row["rewrited_query"] = (
            f"Fine, let’s go with the audio file with ID {audio_id}—make the "
            f"audio file with ID {audio_id} my ringtone."
        )
        template_name = "ringtone_audio_id_choice"
    else:
        row["query"] = "Use that tune as my incoming call ringtone"
        row["rewrited_query"] = f"Use {uri} as my incoming call ringtone"
        template_name = "ringtone_that_tune_reference"

    row["conversation_history"] = str(history)
    row["answer"] = str(answer_dict)
    return template_name


def harden_navigate(row: dict, row_index: int):
    answer_dict = parse_answer(row["answer"])
    location = extract_location(answer_dict)
    query = NAVIGATION_QUERIES[row_index % len(NAVIGATION_QUERIES)]
    row["query"] = query
    rewrite_map = {
        "Get directions there.": f"Get directions to {location}.",
        "Navigate there.": f"Navigate to {location}.",
        "Drive me there.": f"Drive me to {location}.",
        "Take me there.": f"Take me to {location}.",
        "Guide me there.": f"Guide me to {location}.",
        "Start navigation there.": f"Start navigation to {location}.",
    }
    row["rewrited_query"] = rewrite_map.get(query, f"Navigate to {location}.")
    row["answer"] = str(answer_dict)
    return "navigation_there_reference"


def harden_set_timer(row: dict, row_index: int):
    answer_dict = parse_answer(row["answer"])
    args = deepcopy(answer_dict.get("arguments", {}))
    history = parse_history(row["conversation_history"])
    timers = parse_timer_candidates(history[0])

    if not timers:
        row["query"] = "Add another one like the last one for a short reset."
        row["rewrited_query"] = "Add another one like the last timer for a short reset."
        args.update({"duration": "10 minutes", "EXTRA_MESSAGE": "Reset"})
        answer_dict["arguments"] = args
        row["answer"] = str(answer_dict)
        return "timer_fallback_reference"

    template_idx = row_index % 3
    if template_idx == 0 and len(timers) >= 1:
        minutes, label = timers[-1]
        row["query"] = "Add another one like the last one for stretching"
        row["rewrited_query"] = (
            f"Add another {minutes}-minute timer for stretching like the last timer, "
            f"{timer_title_case(label)}."
        )
        args.update({"duration": f"{minutes} minutes", "EXTRA_MESSAGE": "Stretching"})
        template_name = "timer_last_reference"
    elif template_idx == 1 and len(timers) >= 2:
        minutes, label = timers[1]
        row["query"] = "Add another one like the second one for eye rest"
        row["rewrited_query"] = (
            f"Add another {minutes}-minute timer for eye rest like the second timer, "
            f"{timer_title_case(label)}."
        )
        args.update({"duration": f"{minutes} minutes", "EXTRA_MESSAGE": "Eye rest"})
        template_name = "timer_second_reference"
    else:
        minutes, label = timers[-1]
        row["query"] = f"Add a 10-minute cooldown timer after that {timer_title_case(label)} one without confirmation."
        row["rewrited_query"] = (
            f"Add a 10-minute cooldown timer after the {timer_title_case(label)} timer "
            f"without confirmation."
        )
        args.update({"duration": "10 minutes", "EXTRA_MESSAGE": "Cooldown", "EXTRA_SKIP_UI": True})
        template_name = "timer_after_reference"

    answer_dict["arguments"] = args
    row["answer"] = str(answer_dict)
    return template_name


def harden_video_capture(row: dict, row_index: int):
    answer_dict = parse_answer(row["answer"])
    seconds = VIDEO_CAPTURE_SECONDS[row_index % len(VIDEO_CAPTURE_SECONDS)]
    row["query"] = f"Start recording and stop it after {seconds} seconds."
    row["rewrited_query"] = f"Start recording and stop the video recording after {seconds} seconds."
    row["answer"] = str(answer_dict)
    return "video_autostop_constraint"


def harden_row(row: dict, row_index: int):
    answer_dict = parse_answer(row["answer"])
    plan = answer_dict.get("plan")
    if plan == "ACTION_EDIT_ALARM":
        template_name = harden_edit_alarm(row, row_index)
    elif plan == "ACTION_SET_RINGTONE":
        template_name = harden_set_ringtone(row, row_index)
    elif plan == "ACTION_NAVIGATE_TO_LOCATION":
        template_name = harden_navigate(row, row_index)
    elif plan == "ACTION_SET_TIMER":
        template_name = harden_set_timer(row, row_index)
    elif plan == "ACTION_VIDEO_CAPTURE":
        template_name = harden_video_capture(row, row_index)
    else:
        return None
    return plan, template_name


def main():
    args = parse_args()
    input_path = Path(args.input)
    backup_path = Path(args.backup) if args.backup else input_path.with_suffix(input_path.suffix + ".before_harden_failures")
    report_path = (
        Path(args.report)
        if args.report
        else input_path.with_name("it6_complex_1_hardening_report.tsv")
    )

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise ValueError(f"Missing TSV header row: {input_path}")

    if backup_path.exists():
        backup_path.unlink()
    shutil.copy2(input_path, backup_path)

    report_rows = []
    for row_index, row in enumerate(rows):
        answer_dict = parse_answer(row.get("answer"))
        plan = answer_dict.get("plan")
        if plan not in TARGET_PLANS:
            continue

        original = {
            "conversation_history": row["conversation_history"],
            "query": row["query"],
            "rewrited_query": row["rewrited_query"],
            "answer": row["answer"],
            "unique_idx": row.get("unique_idx", ""),
            "plan": plan,
        }
        result = harden_row(row, row_index)
        if result is None:
            continue
        _, template_name = result
        report_rows.append(
            {
                "row_index": row_index,
                "unique_idx": original["unique_idx"],
                "plan": plan,
                "template_name": template_name,
                "query_before": original["query"],
                "query_after": row["query"],
                "rewrited_query_before": original["rewrited_query"],
                "rewrited_query_after": row["rewrited_query"],
                "answer_before": original["answer"],
                "answer_after": row["answer"],
                "turn1_before": parse_history(original["conversation_history"])[0],
                "turn1_after": parse_history(row["conversation_history"])[0],
            }
        )

    write_tsv(input_path, fieldnames, rows)
    write_tsv(
        report_path,
        [
            "row_index",
            "unique_idx",
            "plan",
            "template_name",
            "query_before",
            "query_after",
            "rewrited_query_before",
            "rewrited_query_after",
            "answer_before",
            "answer_after",
            "turn1_before",
            "turn1_after",
        ],
        report_rows,
    )

    print(f"Backed up original file to: {backup_path}")
    print(f"Hardened file in place: {input_path}")
    print(f"Wrote report: {report_path}")
    print(f"Changed rows: {len(report_rows)}")


if __name__ == "__main__":
    main()
