#!/usr/bin/env python3
import argparse
import ast
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_DIR = "datasets/tc/scale"
DEFAULT_OUTPUT_DIR = "datasets/tc/scale_turn6_refswap"
DEFAULT_PLANS = [
    "ACTION_EDIT_DOCUMENT",
    "ACTION_SET_RINGTONE",
    "search_location",
]
DEFAULT_PAIRINGS = [
    ("complex_1", "nonNR"),
    ("complex_2", "complex_4"),
]
DEFAULT_SEED = 20260430

BUCKET_TO_FILENAME = {
    "complex_1": "it6_complex_1_tc.tsv",
    "complex_2": "it6_complex_2_tc.tsv",
    "complex_3": "it6_complex_3_tc.tsv",
    "complex_4": "it6_complex_4_tc.tsv",
    "nonNR": "it6_nonNR_tc.tsv",
}
BUCKET_TO_TARGET_TURN = {
    "complex_1": 1,
    "complex_2": 2,
    "complex_3": 3,
    "complex_4": 4,
    "nonNR": 5,
}
TURN_LABEL_PATTERN = re.compile(r"^\s*turn\s+\d+\s*:\s*", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"^\+?[\d\-\s\(\)]+$")


@dataclass
class RowState:
    bucket: str
    target_turn: int
    row_index: int
    raw: dict
    history: list[str]
    answer_dict: dict
    plan: str
    group_key: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build experimental turn-6 TC files by swapping the referenced history "
            "turn plus query/answer bundles across bucket pairs such as "
            "complex_1 <-> nonNR and complex_2 <-> complex_4."
        )
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Relative paths are resolved from here.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the original turn-6 scale TC TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write swapped turn-6 TC TSV files.",
    )
    parser.add_argument(
        "--plans",
        nargs="+",
        default=DEFAULT_PLANS,
        help="Plans to swap. Default: ACTION_EDIT_DOCUMENT ACTION_SET_RINGTONE search_location",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic pairing within each plan/group.",
    )
    parser.add_argument(
        "--pairings",
        nargs="+",
        default=[f"{left}:{right}" for left, right in DEFAULT_PAIRINGS],
        help="Bucket pairings in left:right form. Default: complex_1:nonNR complex_2:complex_4",
    )
    parser.add_argument(
        "--cross-group-backfill",
        action="store_true",
        help=(
            "After same-group pairing, also pair leftovers across groups within the "
            "same plan. Disabled by default to preserve semantic coherence."
        ),
    )
    parser.add_argument(
        "--aggressive-plans",
        nargs="+",
        default=[],
        help=(
            "Plans that should ignore subtype grouping and swap from a single pooled "
            "bucket-level group. Useful for more aggressive experiments on a specific plan."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the swap summary without writing output TSVs.",
    )
    return parser.parse_args()


def resolve_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def parse_literal(value, fallback):
    if value in (None, ""):
        return fallback
    try:
        return ast.literal_eval(value)
    except Exception:
        return fallback


def parse_history(value) -> list[str]:
    parsed = parse_literal(value, [])
    if not isinstance(parsed, list):
        raise ValueError("conversation_history must parse to a list.")
    return [str(item) for item in parsed]


def parse_answer_dict(value) -> dict:
    parsed = parse_literal(value, {})
    if not isinstance(parsed, dict):
        return {}
    return parsed


def strip_turn_label(text: str) -> str:
    return TURN_LABEL_PATTERN.sub("", str(text)).strip()


def rewrite_turn_label(turn_text: str, target_turn: int) -> str:
    return f"turn {target_turn}: {strip_turn_label(turn_text)}"


def parse_pairing(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Invalid pairing `{text}`. Expected left:right")
    left, right = [item.strip() for item in text.split(":", 1)]
    if left not in BUCKET_TO_FILENAME or right not in BUCKET_TO_FILENAME:
        raise ValueError(f"Unknown bucket pairing `{text}`")
    if left == right:
        raise ValueError(f"Pairing must use distinct buckets: `{text}`")
    return left, right


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"Missing header row: {path}")
    return fieldnames, rows


def get_plan(answer_dict: dict) -> str | None:
    plan = str(answer_dict.get("plan", "")).strip()
    return plan or None


def document_group_key(row: RowState) -> str:
    arguments = row.answer_dict.get("arguments", {})
    uri = str(arguments.get("document_uri", "")).strip().lower()
    suffix = Path(uri.split("?")[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".txt", ".md", ".doc", ".docx", ".odt", ".rtf"}:
        return "text_doc"
    if suffix in {".csv", ".tsv", ".xls", ".xlsx", ".ods"}:
        return "sheet"
    if suffix in {".ppt", ".pptx", ".key"}:
        return "slides"
    if suffix in {".css", ".html", ".xml", ".json", ".js", ".py", ".java", ".swf", ".bin"}:
        return "code_or_binary"
    if suffix:
        return suffix.lstrip(".")
    return "generic"


def ringtone_group_key(row: RowState) -> str:
    arguments = row.answer_dict.get("arguments", {})
    uri = str(arguments.get("ringtone_uri", "")).strip().lower()
    combined_text = " ".join(
        [
            uri,
            str(row.raw.get("query", "")).lower(),
            str(row.raw.get("rewrited_query", "")).lower(),
        ]
    )
    use_alarm = " alarm" in combined_text or combined_text.startswith("alarm")
    if use_alarm:
        return "alarm_sound"
    if any(token in uri for token in ["/audio/media/", "/internal/audio/media/", "/document/audio"]):
        return "media_id"
    if uri.endswith((".mp3", ".wav", ".ogg", ".m4a")):
        return "audio_file"
    return "generic_tone"


def search_location_group_key(row: RowState) -> str:
    arguments = row.answer_dict.get("arguments", {})
    query = str(arguments.get("query", "")).strip()
    lower = query.lower()
    if lower.startswith("content://calendar/event/"):
        return "calendar_event"
    if lower.startswith("content://"):
        return "content_uri"
    if PHONE_PATTERN.fullmatch(query) and any(ch.isdigit() for ch in query):
        return "phone_number"
    address_tokens = [
        "street",
        "st",
        "avenue",
        "ave",
        "lane",
        "ln",
        "road",
        "rd",
        "court",
        "ct",
        "way",
        "drive",
        "dr",
        "boulevard",
        "blvd",
        "place",
        "pl",
        "circle",
        "cir",
    ]
    if any(ch.isdigit() for ch in query) and any(token in lower for token in address_tokens):
        return "street_address"
    return "place_name"


def derive_group_key(plan: str, row: RowState, aggressive_plans: set[str]) -> str:
    if plan in aggressive_plans:
        return "__all__"
    if plan == "ACTION_EDIT_DOCUMENT":
        return document_group_key(row)
    if plan == "ACTION_SET_RINGTONE":
        return ringtone_group_key(row)
    if plan == "search_location":
        return search_location_group_key(row)
    return "generic"


def build_row_states(
    bucket: str,
    rows: list[dict],
    target_plans: set[str],
    aggressive_plans: set[str],
) -> list[RowState]:
    target_turn = BUCKET_TO_TARGET_TURN[bucket]
    out = []
    for row_index, raw_row in enumerate(rows):
        answer_dict = parse_answer_dict(raw_row.get("answer"))
        plan = get_plan(answer_dict)
        if plan not in target_plans:
            continue
        history = parse_history(raw_row.get("conversation_history"))
        if len(history) != 5:
            raise ValueError(
                f"{bucket} row {row_index} has history length {len(history)}, expected 5 for turn-6 TC."
            )
        placeholder = RowState(
            bucket=bucket,
            target_turn=target_turn,
            row_index=row_index,
            raw=dict(raw_row),
            history=history,
            answer_dict=answer_dict,
            plan=plan,
            group_key="",
        )
        placeholder.group_key = derive_group_key(plan, placeholder, aggressive_plans)
        out.append(placeholder)
    return out


def shuffled_copy(rows: list[RowState], seed_text: str, seed: int) -> list[RowState]:
    rng = random.Random(f"{seed}:{seed_text}")
    copied = list(rows)
    rng.shuffle(copied)
    return copied


def build_swapped_row(
    *,
    target_row: RowState,
    source_row: RowState,
    target_bucket: str,
    source_bucket: str,
    pair_serial: int,
) -> dict:
    new_row = dict(target_row.raw)
    target_turn = BUCKET_TO_TARGET_TURN[target_bucket]
    source_turn = BUCKET_TO_TARGET_TURN[source_bucket]
    new_history = list(target_row.history)
    new_history[target_turn - 1] = rewrite_turn_label(
        source_row.history[source_turn - 1],
        target_turn=target_turn,
    )
    new_row["conversation_history"] = repr(new_history)
    new_row["query"] = source_row.raw.get("query", "")
    new_row["rewrited_query"] = source_row.raw.get("rewrited_query", "")
    new_row["answer"] = source_row.raw.get("answer", "")
    if "candidates" in new_row:
        new_row["candidates"] = source_row.raw.get("candidates", new_row.get("candidates", ""))
    new_row["unique_idx"] = (
        f"REFSWAP_{target_bucket}_FROM_{source_bucket}_I{pair_serial:06d}-"
        f"{source_row.plan}-1"
    )
    return new_row


def format_summary_row(
    *,
    left_bucket: str,
    right_bucket: str,
    plan: str,
    group_key: str,
    left_count: int,
    right_count: int,
    paired_count: int,
    pass_name: str,
) -> dict:
    return {
        "left_bucket": left_bucket,
        "right_bucket": right_bucket,
        "plan": plan,
        "group_key": group_key,
        "pass_name": pass_name,
        "left_count": left_count,
        "right_count": right_count,
        "paired_count": paired_count,
        "left_unpaired": left_count - paired_count,
        "right_unpaired": right_count - paired_count,
    }


def build_report_row(
    *,
    pair_serial: int,
    target_bucket: str,
    source_bucket: str,
    plan: str,
    group_key: str,
    target_row: RowState,
    source_row: RowState,
    swapped_row: dict,
) -> dict:
    target_turn = BUCKET_TO_TARGET_TURN[target_bucket]
    source_turn = BUCKET_TO_TARGET_TURN[source_bucket]
    return {
        "pair_serial": pair_serial,
        "target_bucket": target_bucket,
        "source_bucket": source_bucket,
        "plan": plan,
        "group_key": group_key,
        "target_row_index": target_row.row_index,
        "source_row_index": source_row.row_index,
        "target_unique_idx_before": target_row.raw.get("unique_idx", ""),
        "source_unique_idx_before": source_row.raw.get("unique_idx", ""),
        "target_unique_idx_after": swapped_row.get("unique_idx", ""),
        "target_turn": target_turn,
        "source_turn": source_turn,
        "target_turn_before": target_row.history[target_turn - 1],
        "source_turn_before": source_row.history[source_turn - 1],
        "target_turn_after": parse_history(swapped_row["conversation_history"])[target_turn - 1],
        "query_before": target_row.raw.get("query", ""),
        "query_after": swapped_row.get("query", ""),
        "rewrited_query_after": swapped_row.get("rewrited_query", ""),
        "answer_after": swapped_row.get("answer", ""),
    }


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    input_dir = resolve_path(root, args.input_dir)
    output_dir = resolve_path(root, args.output_dir)
    target_plans = set(args.plans)
    aggressive_plans = set(args.aggressive_plans)
    unknown_aggressive_plans = aggressive_plans - target_plans
    if unknown_aggressive_plans:
        raise ValueError(
            "Every aggressive plan must also be included in --plans: "
            f"{sorted(unknown_aggressive_plans)}"
        )
    pairing_specs = [parse_pairing(text) for text in args.pairings]

    file_rows = {}
    fieldnames_by_bucket = {}
    row_states_by_bucket = {}
    for bucket, filename in BUCKET_TO_FILENAME.items():
        path = input_dir / filename
        fieldnames, rows = read_tsv(path)
        file_rows[bucket] = rows
        fieldnames_by_bucket[bucket] = fieldnames
        row_states_by_bucket[bucket] = build_row_states(
            bucket,
            rows,
            target_plans,
            aggressive_plans,
        )

    replacements_by_bucket = {bucket: {} for bucket in BUCKET_TO_FILENAME}
    report_rows = []
    summary_rows = []
    pair_serial = 1

    for left_bucket, right_bucket in pairing_specs:
        left_states = row_states_by_bucket[left_bucket]
        right_states = row_states_by_bucket[right_bucket]
        for plan in sorted(target_plans):
            left_plan_rows = [row for row in left_states if row.plan == plan]
            right_plan_rows = [row for row in right_states if row.plan == plan]
            if not left_plan_rows and not right_plan_rows:
                continue

            left_groups = {}
            right_groups = {}
            for row in left_plan_rows:
                left_groups.setdefault(row.group_key, []).append(row)
            for row in right_plan_rows:
                right_groups.setdefault(row.group_key, []).append(row)

            leftovers_left = []
            leftovers_right = []
            for group_key in sorted(set(left_groups) | set(right_groups)):
                left_group_rows = shuffled_copy(
                    left_groups.get(group_key, []),
                    seed_text=f"{left_bucket}:{right_bucket}:{plan}:{group_key}:left",
                    seed=args.seed,
                )
                right_group_rows = shuffled_copy(
                    right_groups.get(group_key, []),
                    seed_text=f"{left_bucket}:{right_bucket}:{plan}:{group_key}:right",
                    seed=args.seed,
                )
                paired_count = min(len(left_group_rows), len(right_group_rows))
                summary_rows.append(
                    format_summary_row(
                        left_bucket=left_bucket,
                        right_bucket=right_bucket,
                        plan=plan,
                        group_key=group_key,
                        left_count=len(left_group_rows),
                        right_count=len(right_group_rows),
                        paired_count=paired_count,
                        pass_name="same_group",
                    )
                )
                for left_row, right_row in zip(left_group_rows[:paired_count], right_group_rows[:paired_count]):
                    swapped_left = build_swapped_row(
                        target_row=left_row,
                        source_row=right_row,
                        target_bucket=left_bucket,
                        source_bucket=right_bucket,
                        pair_serial=pair_serial,
                    )
                    swapped_right = build_swapped_row(
                        target_row=right_row,
                        source_row=left_row,
                        target_bucket=right_bucket,
                        source_bucket=left_bucket,
                        pair_serial=pair_serial,
                    )
                    replacements_by_bucket[left_bucket][left_row.row_index] = swapped_left
                    replacements_by_bucket[right_bucket][right_row.row_index] = swapped_right
                    report_rows.append(
                        build_report_row(
                            pair_serial=pair_serial,
                            target_bucket=left_bucket,
                            source_bucket=right_bucket,
                            plan=plan,
                            group_key=group_key,
                            target_row=left_row,
                            source_row=right_row,
                            swapped_row=swapped_left,
                        )
                    )
                    report_rows.append(
                        build_report_row(
                            pair_serial=pair_serial,
                            target_bucket=right_bucket,
                            source_bucket=left_bucket,
                            plan=plan,
                            group_key=group_key,
                            target_row=right_row,
                            source_row=left_row,
                            swapped_row=swapped_right,
                        )
                    )
                    pair_serial += 1
                leftovers_left.extend(left_group_rows[paired_count:])
                leftovers_right.extend(right_group_rows[paired_count:])

            if args.cross_group_backfill and leftovers_left and leftovers_right:
                left_backfill = shuffled_copy(
                    leftovers_left,
                    seed_text=f"{left_bucket}:{right_bucket}:{plan}:backfill:left",
                    seed=args.seed,
                )
                right_backfill = shuffled_copy(
                    leftovers_right,
                    seed_text=f"{left_bucket}:{right_bucket}:{plan}:backfill:right",
                    seed=args.seed,
                )
                paired_count = min(len(left_backfill), len(right_backfill))
                summary_rows.append(
                    format_summary_row(
                        left_bucket=left_bucket,
                        right_bucket=right_bucket,
                        plan=plan,
                        group_key="__cross_group__",
                        left_count=len(left_backfill),
                        right_count=len(right_backfill),
                        paired_count=paired_count,
                        pass_name="cross_group_backfill",
                    )
                )
                for left_row, right_row in zip(left_backfill[:paired_count], right_backfill[:paired_count]):
                    swapped_left = build_swapped_row(
                        target_row=left_row,
                        source_row=right_row,
                        target_bucket=left_bucket,
                        source_bucket=right_bucket,
                        pair_serial=pair_serial,
                    )
                    swapped_right = build_swapped_row(
                        target_row=right_row,
                        source_row=left_row,
                        target_bucket=right_bucket,
                        source_bucket=left_bucket,
                        pair_serial=pair_serial,
                    )
                    replacements_by_bucket[left_bucket][left_row.row_index] = swapped_left
                    replacements_by_bucket[right_bucket][right_row.row_index] = swapped_right
                    report_rows.append(
                        build_report_row(
                            pair_serial=pair_serial,
                            target_bucket=left_bucket,
                            source_bucket=right_bucket,
                            plan=plan,
                            group_key="__cross_group__",
                            target_row=left_row,
                            source_row=right_row,
                            swapped_row=swapped_left,
                        )
                    )
                    report_rows.append(
                        build_report_row(
                            pair_serial=pair_serial,
                            target_bucket=right_bucket,
                            source_bucket=left_bucket,
                            plan=plan,
                            group_key="__cross_group__",
                            target_row=right_row,
                            source_row=left_row,
                            swapped_row=swapped_right,
                        )
                    )
                    pair_serial += 1

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Plans: {', '.join(sorted(target_plans))}")
    print(f"Aggressive plans: {', '.join(sorted(aggressive_plans)) or '(none)'}")
    print(f"Pairings: {', '.join(f'{left}:{right}' for left, right in pairing_specs)}")
    print(f"Cross-group backfill: {args.cross_group_backfill}")
    print(f"Report rows: {len(report_rows)}")
    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['left_bucket']}<->{row['right_bucket']} | {row['plan']} | "
            f"{row['group_key']} | {row['pass_name']} | "
            f"left={row['left_count']} right={row['right_count']} paired={row['paired_count']}"
        )

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for bucket, filename in BUCKET_TO_FILENAME.items():
        original_rows = file_rows[bucket]
        output_rows = []
        for row_index, row in enumerate(original_rows):
            output_rows.append(replacements_by_bucket[bucket].get(row_index, row))
        write_tsv(output_dir / filename, fieldnames_by_bucket[bucket], output_rows)

    report_fieldnames = [
        "pair_serial",
        "target_bucket",
        "source_bucket",
        "plan",
        "group_key",
        "target_row_index",
        "source_row_index",
        "target_unique_idx_before",
        "source_unique_idx_before",
        "target_unique_idx_after",
        "target_turn",
        "source_turn",
        "target_turn_before",
        "source_turn_before",
        "target_turn_after",
        "query_before",
        "query_after",
        "rewrited_query_after",
        "answer_after",
    ]
    summary_fieldnames = [
        "left_bucket",
        "right_bucket",
        "plan",
        "group_key",
        "pass_name",
        "left_count",
        "right_count",
        "paired_count",
        "left_unpaired",
        "right_unpaired",
    ]
    write_tsv(output_dir / "turn6_refswap_report.tsv", report_fieldnames, report_rows)
    write_tsv(output_dir / "turn6_refswap_summary.tsv", summary_fieldnames, summary_rows)
    print(f"Wrote swapped turn-6 files to: {output_dir}")
    print(f"Wrote report: {output_dir / 'turn6_refswap_report.tsv'}")
    print(f"Wrote summary: {output_dir / 'turn6_refswap_summary.tsv'}")


if __name__ == "__main__":
    main()
