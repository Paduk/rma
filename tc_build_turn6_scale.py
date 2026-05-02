#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_PAIR_POOL = "datasets/hammerbench/en/single_turn_user_assistant_pairs.dedup.jsonl"
DEFAULT_TC_DIR = "datasets/tc/scale"
DEFAULT_SEED = 20260429

TURN6_SPECS = [
    {
        "output_name": "it6_complex_1_tc.tsv",
        "sources": [
            {
                "base_name": "it5_complex_1_tc.tsv",
                "expected_history_len": 4,
                "final_insert_turn_patterns": [[2], [3], [4], [5]],
            }
        ],
    },
    {
        "output_name": "it6_complex_2_tc.tsv",
        "sources": [
            {
                "base_name": "it5_complex_2_tc.tsv",
                "expected_history_len": 4,
                "final_insert_turn_patterns": [[3], [4], [5]],
            }
        ],
    },
    {
        "output_name": "it6_complex_3_tc.tsv",
        "sources": [
            {
                "base_name": "it5_complex_3_tc.tsv",
                "expected_history_len": 4,
                "final_insert_turn_patterns": [[4], [5]],
            }
        ],
    },
    {
        "output_name": "it6_complex_4_tc.tsv",
        "output_row_count": 374,
        "sources": [
            {
                "base_name": "it4_nonNR_tc.tsv",
                "mix_ratio": 0.5,
                "expected_history_len": 3,
                "final_insert_turn_patterns": [[2, 5], [3, 5]],
            },
            {
                "base_name": "it5_nonNR_tc.tsv",
                "mix_ratio": 0.5,
                "expected_history_len": 4,
                "final_insert_turn_patterns": [[5]],
            },
        ],
    },
    {
        "output_name": "it6_complex_4_tc_v2.tsv",
        "output_row_count": 374,
        "sources": [
            {
                "base_name": "it4_nonNR_tc.tsv",
                "mix_ratio": 0.7,
                "expected_history_len": 3,
                "final_insert_turn_patterns": [[2, 5], [3, 5]],
            },
            {
                "base_name": "it5_nonNR_tc.tsv",
                "mix_ratio": 0.3,
                "expected_history_len": 4,
                "final_insert_turn_patterns": [[5]],
            },
        ],
    },
    {
        "output_name": "it6_nonNR_tc.tsv",
        "output_row_count": 374,
        "sources": [
            {
                "base_name": "it4_nonNR_tc.tsv",
                "mix_ratio": 0.5,
                "expected_history_len": 3,
                "final_insert_turn_patterns": [[2, 3], [2, 4], [3, 4]],
            },
            {
                "base_name": "it5_nonNR_tc.tsv",
                "mix_ratio": 0.5,
                "expected_history_len": 4,
                "final_insert_turn_patterns": [[2], [3], [4]],
            },
        ],
    },
]

TURN_LABEL_PATTERN = re.compile(r"^\s*turn\s+\d+\s*:\s*", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build turn-6 scale TC files by inserting one or more random external "
            "single-turn user-assistant pairs into supported TC samples."
        )
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Relative paths are resolved from here.",
    )
    parser.add_argument(
        "--tc-dir",
        default=DEFAULT_TC_DIR,
        help="Directory containing the scale TC TSV files.",
    )
    parser.add_argument(
        "--pair-pool",
        default=DEFAULT_PAIR_POOL,
        help="JSONL file containing deduplicated single-turn user-assistant pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for pair shuffling and insertion-position sampling.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Reuse pair-pool entries across rows. Default is global sampling without replacement.",
    )
    return parser.parse_args()


def resolve_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def normalize_text(text):
    normalized = WHITESPACE_PATTERN.sub(" ", str(text).replace("\t", " ").replace("\r", " ").replace("\n", " "))
    return normalized.strip()


def load_pair_pool(path: Path):
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            json_start = raw.find("{")
            if json_start == -1:
                continue
            row = json.loads(raw[json_start:])
            user_text = normalize_text(row.get("user", ""))
            assistant_text = normalize_text(row.get("assistant", ""))
            if not user_text or not assistant_text:
                continue
            pairs.append(
                {
                    "source_id": str(row.get("source_id", "")),
                    "turn_index": row.get("turn_index"),
                    "user": user_text,
                    "assistant": assistant_text,
                    "line_idx": line_idx,
                }
            )
    if not pairs:
        raise ValueError(f"No valid pairs found in pair pool: {path}")
    return pairs


def load_tsv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"Missing header row: {path}")
        return reader.fieldnames, rows


def parse_answer_plan(row):
    answer_text = str(row.get("answer", "")).strip()
    if not answer_text:
        return "EMPTY_ANSWER"
    parsed = ast.literal_eval(answer_text)
    if not isinstance(parsed, dict):
        return "NON_DICT_ANSWER"
    plan = str(parsed.get("plan", "")).strip()
    return plan if plan else "EMPTY_PLAN"


def validate_pattern_lengths(source_spec):
    pattern_lengths = {len(pattern) for pattern in source_spec["final_insert_turn_patterns"]}
    if not pattern_lengths:
        raise ValueError(f"Missing final_insert_turn_patterns for source: {source_spec}")
    if len(pattern_lengths) != 1:
        raise ValueError(
            "Every source must use turn-patterns with the same number of inserted turns: "
            f"{source_spec}"
        )
    return pattern_lengths.pop()


def parse_conversation_history(raw_history):
    if raw_history is None:
        return []
    if isinstance(raw_history, list):
        return [str(item) for item in raw_history]

    history_text = str(raw_history).strip()
    if not history_text:
        return []

    try:
        parsed = ast.literal_eval(history_text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse conversation_history: {history_text[:120]}") from exc

    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def strip_turn_label(text: str) -> str:
    return TURN_LABEL_PATTERN.sub("", str(text)).strip()


def renumber_history(history_bodies):
    return [f"turn {idx}: {body}" for idx, body in enumerate(history_bodies, start=1)]


def build_inserted_turn(pair):
    return f"{pair['user']} -> {pair['assistant']}"


def make_pair_unique_fragment(inserted_turn_idx, pair):
    pair_source = pair["source_id"] or f"line{pair['line_idx']}"
    pair_turn = pair.get("turn_index")
    pair_suffix = f"{pair_source}_t{pair_turn}" if pair_turn is not None else pair_source
    return f"ins{inserted_turn_idx}__{pair_suffix}"


def make_multi_insert_unique_idx(base_unique_idx, output_name, inserted_turn_indices, pairs):
    parts = [str(base_unique_idx), Path(output_name).stem]
    for inserted_turn_idx, pair in zip(inserted_turn_indices, pairs):
        parts.append(make_pair_unique_fragment(inserted_turn_idx=inserted_turn_idx, pair=pair))
    return "__".join(part for part in parts if part)


def choose_pair(pair_pool, rng, next_pair_idx, sample_with_replacement):
    if sample_with_replacement:
        return rng.choice(pair_pool), next_pair_idx
    if next_pair_idx >= len(pair_pool):
        raise ValueError(
            "Pair pool exhausted before finishing turn-6 generation. "
            "Either increase the pool size or enable --sample-with-replacement."
        )
    return pair_pool[next_pair_idx], next_pair_idx + 1


def choose_pairs(pair_pool, rng, next_pair_idx, sample_with_replacement, pair_count):
    pairs = []
    for _ in range(pair_count):
        pair, next_pair_idx = choose_pair(
            pair_pool=pair_pool,
            rng=rng,
            next_pair_idx=next_pair_idx,
            sample_with_replacement=sample_with_replacement,
        )
        pairs.append(pair)
    return pairs, next_pair_idx


def select_base_rows(base_rows, requested_count, rng):
    if requested_count > len(base_rows):
        raise ValueError(
            f"Requested {requested_count} rows from a base file with only {len(base_rows)} rows"
        )
    if requested_count == len(base_rows):
        return list(base_rows)

    rows_by_plan = defaultdict(list)
    for row in base_rows:
        rows_by_plan[parse_answer_plan(row)].append(row)

    plan_names = list(rows_by_plan.keys())
    rng.shuffle(plan_names)
    for plan_name in plan_names:
        rng.shuffle(rows_by_plan[plan_name])

    selected_rows = []
    plan_idx = 0
    while len(selected_rows) < requested_count:
        progress_made = False
        for _ in range(len(plan_names)):
            plan_name = plan_names[plan_idx % len(plan_names)]
            plan_idx += 1
            bucket = rows_by_plan[plan_name]
            if not bucket:
                continue
            selected_rows.append(bucket.pop())
            progress_made = True
            if len(selected_rows) >= requested_count:
                break
        if not progress_made:
            break

    if len(selected_rows) != requested_count:
        raise ValueError(
            f"Plan-balanced sampling failed: expected {requested_count} rows, got {len(selected_rows)}"
        )
    return selected_rows


def build_source_plan(spec, base_cache):
    sources = spec["sources"]
    if len(sources) == 1 and "output_row_count" not in spec:
        source = dict(sources[0])
        source["row_count"] = len(base_cache[source["base_name"]]["rows"])
        source["pair_count"] = validate_pattern_lengths(source)
        return [source]

    output_row_count = spec.get("output_row_count")
    if output_row_count is None:
        raise ValueError(f"Mixed-source spec requires output_row_count: {spec['output_name']}")

    total_ratio = sum(float(source.get("mix_ratio", 0.0)) for source in sources)
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"mix_ratio values must sum to 1.0 for {spec['output_name']}")

    planned_sources = []
    remaining_rows = output_row_count
    for idx, source_spec in enumerate(sources):
        planned_source = dict(source_spec)
        if idx == len(sources) - 1:
            row_count = remaining_rows
        else:
            row_count = int(round(output_row_count * float(source_spec["mix_ratio"])))
            remaining_rows -= row_count
        planned_source["row_count"] = row_count
        planned_source["pair_count"] = validate_pattern_lengths(planned_source)
        planned_sources.append(planned_source)

    return planned_sources


def transform_row(row, final_insert_turn_positions, pairs, output_name, expected_history_len):
    history_items = parse_conversation_history(row.get("conversation_history"))
    if len(history_items) != expected_history_len:
        raise ValueError(
            f"Expected {expected_history_len} history items before turn-6 expansion, got {len(history_items)} "
            f"for unique_idx={row.get('unique_idx')}"
        )

    history_bodies = [strip_turn_label(item) for item in history_items]
    inserted_turn_texts = [build_inserted_turn(pair) for pair in pairs]
    if len(final_insert_turn_positions) != len(inserted_turn_texts):
        raise ValueError("Mismatch between insert positions and inserted turn count")

    final_history_len = len(history_bodies) + len(inserted_turn_texts)
    if final_history_len != 5:
        raise ValueError(f"Expected final history length 5, got {final_history_len}")

    insert_map = dict(zip(final_insert_turn_positions, inserted_turn_texts))
    new_history_bodies = []
    history_idx = 0
    for final_turn_idx in range(1, final_history_len + 1):
        if final_turn_idx in insert_map:
            new_history_bodies.append(insert_map[final_turn_idx])
            continue
        if history_idx >= len(history_bodies):
            raise ValueError("Ran out of source history items while rebuilding conversation_history")
        new_history_bodies.append(history_bodies[history_idx])
        history_idx += 1

    if history_idx != len(history_bodies):
        raise ValueError("Source history items remained after rebuilding conversation_history")

    renumbered_history = renumber_history(new_history_bodies)

    new_row = row.copy()
    new_row["conversation_history"] = renumbered_history
    new_row["unique_idx"] = make_multi_insert_unique_idx(
        base_unique_idx=str(row.get("unique_idx", "")),
        output_name=output_name,
        inserted_turn_indices=final_insert_turn_positions,
        pairs=pairs,
    )
    return new_row


def write_tsv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def validate_output_rows(output_name, rows):
    for row_idx, row in enumerate(rows, start=1):
        history_items = parse_conversation_history(row.get("conversation_history"))
        if len(history_items) != 5:
            raise ValueError(
                f"{output_name} row {row_idx} has {len(history_items)} turns instead of 5"
            )


def main():
    args = parse_args()

    root = Path(args.root).resolve()
    tc_dir = resolve_path(root, args.tc_dir)
    pair_pool_path = resolve_path(root, args.pair_pool)

    pair_pool = load_pair_pool(pair_pool_path)
    base_cache = {}
    for spec in TURN6_SPECS:
        for source_spec in spec["sources"]:
            base_name = source_spec["base_name"]
            if base_name in base_cache:
                continue
            fieldnames, rows = load_tsv_rows(tc_dir / base_name)
            base_cache[base_name] = {"fieldnames": fieldnames, "rows": rows}

    total_needed_rows = 0
    for spec in TURN6_SPECS:
        for source_plan in build_source_plan(spec, base_cache):
            total_needed_rows += source_plan["row_count"] * source_plan["pair_count"]

    if not args.sample_with_replacement and len(pair_pool) < total_needed_rows:
        raise ValueError(
            f"Pair pool too small for global sampling without replacement: "
            f"{len(pair_pool)} < {total_needed_rows}"
        )

    rng = random.Random(args.seed)
    shuffled_pair_pool = list(pair_pool)
    rng.shuffle(shuffled_pair_pool)

    next_pair_idx = 0
    for spec in TURN6_SPECS:
        output_path = tc_dir / spec["output_name"]
        source_plans = build_source_plan(spec, base_cache)

        fieldnames = None
        output_rows = []
        for source_plan in source_plans:
            cached = base_cache[source_plan["base_name"]]
            if fieldnames is None:
                fieldnames = cached["fieldnames"]
            elif fieldnames != cached["fieldnames"]:
                raise ValueError(
                    f"Mismatched fieldnames across mixed sources for {spec['output_name']}: "
                    f"{fieldnames} vs {cached['fieldnames']}"
                )

            selected_rows = select_base_rows(
                base_rows=cached["rows"],
                requested_count=source_plan["row_count"],
                rng=rng,
            )
            for row in selected_rows:
                final_insert_turn_positions = rng.choice(source_plan["final_insert_turn_patterns"])
                pairs, next_pair_idx = choose_pairs(
                    pair_pool=shuffled_pair_pool,
                    rng=rng,
                    next_pair_idx=next_pair_idx,
                    sample_with_replacement=args.sample_with_replacement,
                    pair_count=len(final_insert_turn_positions),
                )
                output_rows.append(
                    transform_row(
                        row=row,
                        final_insert_turn_positions=final_insert_turn_positions,
                        pairs=pairs,
                        output_name=spec["output_name"],
                        expected_history_len=source_plan["expected_history_len"],
                    )
                )

        if len(source_plans) > 1:
            rng.shuffle(output_rows)

        validate_output_rows(spec["output_name"], output_rows)
        write_tsv(output_path, fieldnames, output_rows)
        print(
            f"{spec['output_name']}\trows={len(output_rows)}\t"
            f"sources="
            + ",".join(
                f"{source_plan['base_name']}:{source_plan['row_count']}"
                for source_plan in source_plans
            )
        )

if __name__ == "__main__":
    main()
