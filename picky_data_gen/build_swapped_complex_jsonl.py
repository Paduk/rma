import argparse
import json
import os
from collections import defaultdict


FILE_CANDIDATES = (
    "it{turn}_s1_filtered.jsonl",
    "it{turn}_s1_nonnr.jsonl",
    "it{turn}_s1.jsonl",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build complex turn JSONL files by transplanting same-plan source turns "
            "into earlier refered_turn slots."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/home/hj153lee/RMA/o4_datagen",
        help="Directory containing itX_s1*.jsonl inputs",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write itX_s1_complex_swapped.jsonl outputs",
    )
    parser.add_argument(
        "--turns",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Target turns to build. Default: 3 4 5",
    )
    parser.add_argument(
        "--api",
        default="/home/hj153lee/RMA/picky_data_gen/apis/api_v3.0.1.jsonl",
        help="API jsonl for candidate generation when source rows do not include candidates",
    )
    parser.add_argument(
        "--output-suffix",
        default="swapped",
        help="Suffix used in output filenames. Default: swapped",
    )
    parser.add_argument(
        "--max-per-target-ref",
        type=int,
        default=None,
        help="Optional cap for rows emitted per (target_turn, refered_turn)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output files",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_turn_file(input_dir, turn):
    for pattern in FILE_CANDIDATES:
        path = os.path.join(input_dir, pattern.format(turn=turn))
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find input for turn {turn} in {input_dir}")


def read_api_plans(path):
    plans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                plans.append(json.loads(line)["plan"])
    return plans


def get_plan(record):
    answer = record.get("answer", {})
    if isinstance(answer, dict) and answer.get("plan"):
        return answer["plan"]

    unique_idx = record.get("unique_idx", "")
    parts = unique_idx.split("-")
    if len(parts) >= 2:
        return parts[-2]
    return None


def ensure_candidates(record, api_plans):
    candidates = record.get("candidates")
    if isinstance(candidates, list) and candidates:
        return candidates

    plan = get_plan(record)
    if not plan:
        return []

    others = [item for item in api_plans if item != plan][:4]
    return others + [plan]


def build_source_pools(rows):
    pools = defaultdict(list)
    for row in rows:
        plan = get_plan(row)
        if not plan:
            continue
        pools[plan].append(row)
    return pools


def build_swapped_history(source_row, target_row, refered_turn, target_turn):
    source_history = list(source_row.get("conversation_history", []))
    target_history = list(target_row.get("conversation_history", []))
    prefix = source_history[:refered_turn]
    suffix = target_history[refered_turn:]
    new_history = prefix + suffix
    expected_len = target_turn - 1
    if len(new_history) != expected_len:
        raise ValueError(
            f"Unexpected history length for target turn {target_turn}, "
            f"refered_turn {refered_turn}: got {len(new_history)}, expected {expected_len}"
        )
    return new_history


def build_unique_idx(target_turn, refered_turn, source_turn, serial, plan):
    return f"SWAP_T{target_turn}_R{refered_turn}_S{source_turn}_I{serial:06d}-{plan}-1"


def transplant_record(
    source_row,
    target_row,
    target_turn,
    refered_turn,
    source_turn,
    serial,
    api_plans,
):
    plan = get_plan(source_row)
    new_record = {
        "conversation_history": build_swapped_history(
            source_row, target_row, refered_turn, target_turn
        ),
        "query": source_row.get("query", ""),
        "rewrited_query": source_row.get("rewrited_query", ""),
        "answer": source_row.get("answer", {}),
        "unique_idx": build_unique_idx(
            target_turn=target_turn,
            refered_turn=refered_turn,
            source_turn=source_turn,
            serial=serial,
            plan=plan,
        ),
        "refered_turn": refered_turn,
        "candidates": ensure_candidates(source_row, api_plans),
    }
    return new_record


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    api_plans = read_api_plans(args.api)

    turns = sorted(set(args.turns))
    if any(turn < 3 for turn in turns):
        raise ValueError("This script only supports target turns 3, 4, 5 or higher.")

    turn_rows = {}
    turn_paths = {}
    for turn in sorted(set(turns) | {2, 3, 4}):
        try:
            turn_paths[turn] = resolve_turn_file(args.input_dir, turn)
            turn_rows[turn] = read_jsonl(turn_paths[turn])
        except FileNotFoundError:
            if turn in turns or turn in {2, 3, 4}:
                raise

    source_pools_by_turn = {
        turn: build_source_pools(rows) for turn, rows in turn_rows.items()
    }
    source_cursors = defaultdict(int)
    output_rows_by_turn = {}
    summary = []

    serial = 1
    for target_turn in turns:
        target_rows = turn_rows[target_turn]
        target_output_rows = []
        for refered_turn in range(1, target_turn - 1):
            source_turn = refered_turn + 1
            emitted = 0
            skipped_missing_plan = 0
            skipped_exhausted = 0

            for target_row in target_rows:
                if args.max_per_target_ref is not None and emitted >= args.max_per_target_ref:
                    break

                plan = get_plan(target_row)
                if not plan:
                    skipped_missing_plan += 1
                    continue

                source_pool = source_pools_by_turn.get(source_turn, {}).get(plan, [])
                if not source_pool:
                    skipped_missing_plan += 1
                    continue

                cursor_key = (source_turn, plan)
                cursor = source_cursors[cursor_key]
                if cursor >= len(source_pool):
                    skipped_exhausted += 1
                    continue

                source_row = source_pool[cursor]
                source_cursors[cursor_key] += 1

                target_output_rows.append(
                    transplant_record(
                        source_row=source_row,
                        target_row=target_row,
                        target_turn=target_turn,
                        refered_turn=refered_turn,
                        source_turn=source_turn,
                        serial=serial,
                        api_plans=api_plans,
                    )
                )
                emitted += 1
                serial += 1

            summary.append(
                {
                    "target_turn": target_turn,
                    "refered_turn": refered_turn,
                    "source_turn": source_turn,
                    "emitted": emitted,
                    "skipped_missing_plan": skipped_missing_plan,
                    "skipped_exhausted": skipped_exhausted,
                }
            )

        output_rows_by_turn[target_turn] = target_output_rows

    for turn in turns:
        output_path = os.path.join(
            args.output_dir, f"it{turn}_s1_complex_{args.output_suffix}.jsonl"
        )
        print(f"turn={turn}")
        print(f"  input={turn_paths[turn]}")
        print(f"  output={output_path}")
        print(f"  rows={len(output_rows_by_turn[turn])}")
        if not args.dry_run:
            write_jsonl(output_path, output_rows_by_turn[turn])

    print("summary:")
    for item in summary:
        print(
            "  "
            f"turn={item['target_turn']} "
            f"ref={item['refered_turn']} "
            f"source_turn={item['source_turn']} "
            f"emitted={item['emitted']} "
            f"skipped_missing_plan={item['skipped_missing_plan']} "
            f"skipped_exhausted={item['skipped_exhausted']}"
        )


if __name__ == "__main__":
    main()
