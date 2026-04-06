import argparse
import ast
import csv
import os
from collections import defaultdict


DEFAULT_HEADERS = [
    "conversation_history",
    "query",
    "rewrited_query",
    "answer",
    "unique_idx",
    "candidates",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build complex TC TSV files by transplanting same-plan nonNR source rows "
            "into earlier refered_turn slots."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/home/hj153lee/RMA/datasets/tc/scale",
        help="Directory containing it2_nonNR_tc.tsv ~ it5_nonNR_tc.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write complex TSVs. Defaults to input-dir.",
    )
    parser.add_argument(
        "--turns",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Target turns to generate. Default: 3 4 5",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write directly to itX_complex_Y_tc.tsv instead of .candidate files",
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
        help="Print summary without writing files",
    )
    return parser.parse_args()


def read_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        headers = list(reader.fieldnames or DEFAULT_HEADERS)
    return headers, rows


def parse_literal(value, fallback):
    if value in (None, ""):
        return fallback
    try:
        return ast.literal_eval(value)
    except Exception:
        return fallback


def normalize_row(row):
    return {
        "conversation_history": parse_literal(row.get("conversation_history"), []),
        "query": row.get("query", ""),
        "rewrited_query": row.get("rewrited_query", ""),
        "answer": parse_literal(row.get("answer"), {}),
        "unique_idx": row.get("unique_idx", ""),
        "candidates": parse_literal(row.get("candidates"), []),
    }


def load_turn_rows(input_dir, turn):
    path = os.path.join(input_dir, f"it{turn}_nonNR_train.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input TSV: {path}")
    headers, rows = read_tsv(path)
    normalized = [normalize_row(row) for row in rows]
    return path, headers, normalized


def get_plan(row):
    answer = row.get("answer", {})
    if isinstance(answer, dict) and answer.get("plan"):
        return answer["plan"]
    unique_idx = row.get("unique_idx", "")
    parts = unique_idx.split("-")
    if len(parts) >= 2:
        return parts[-2]
    return None


def build_source_pools(rows):
    pools = defaultdict(list)
    for row in rows:
        plan = get_plan(row)
        if plan:
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
            f"Unexpected history length for turn={target_turn}, ref={refered_turn}: "
            f"{len(new_history)} != {expected_len}"
        )
    return new_history


def build_unique_idx(target_turn, refered_turn, source_turn, serial, plan):
    return f"SWAP_T{target_turn}_R{refered_turn}_S{source_turn}_I{serial:06d}-{plan}-1"


def transplant_row(
    source_row,
    target_row,
    target_turn,
    refered_turn,
    source_turn,
    serial,
):
    plan = get_plan(source_row)
    return {
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
        "candidates": source_row.get("candidates", []),
    }


def format_row(row, headers):
    data = {
        "conversation_history": str(row.get("conversation_history", [])),
        "query": row.get("query", ""),
        "rewrited_query": row.get("rewrited_query", ""),
        "answer": str(row.get("answer", {})),
        "unique_idx": row.get("unique_idx", ""),
        "candidates": str(row.get("candidates", [])),
    }
    return {header: data.get(header, "") for header in headers}


def resolve_output_path(output_dir, target_turn, refered_turn, inplace):
    base = os.path.join(output_dir, f"it{target_turn}_complex_{refered_turn}_train.tsv")
    if inplace:
        return base
    return f"{base}"


def main():
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    turns = sorted(set(args.turns))
    if any(turn < 3 for turn in turns):
        raise ValueError("This script supports target turns 3, 4, 5 or higher only.")

    turn_rows = {}
    turn_headers = {}
    turn_paths = {}
    needed_turns = sorted(set(turns) | {2, 3, 4, 5})
    for turn in needed_turns:
        if turn > max(turns):
            continue
        path, headers, rows = load_turn_rows(args.input_dir, turn)
        turn_paths[turn] = path
        turn_headers[turn] = headers
        turn_rows[turn] = rows

    source_pools_by_turn = {
        turn: build_source_pools(rows) for turn, rows in turn_rows.items()
    }
    serial = 1
    summary = []

    for target_turn in turns:
        target_rows = turn_rows[target_turn]
        headers = turn_headers[target_turn]

        for refered_turn in range(1, target_turn - 1):
            source_turn = refered_turn + 1
            source_cursors = defaultdict(int)
            output_rows = []
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

                output_rows.append(
                    transplant_row(
                        source_row=source_row,
                        target_row=target_row,
                        target_turn=target_turn,
                        refered_turn=refered_turn,
                        source_turn=source_turn,
                        serial=serial,
                    )
                )
                emitted += 1
                serial += 1

            output_path = resolve_output_path(
                output_dir=output_dir,
                target_turn=target_turn,
                refered_turn=refered_turn,
                inplace=args.inplace,
            )

            print(f"turn={target_turn} ref={refered_turn}")
            print(f"  source_turn={source_turn}")
            print(f"  input={turn_paths[target_turn]}")
            print(f"  output={output_path}")
            print(f"  rows={len(output_rows)}")
            print(f"  skipped_missing_plan={skipped_missing_plan}")
            print(f"  skipped_exhausted={skipped_exhausted}")

            summary.append(
                {
                    "turn": target_turn,
                    "refered_turn": refered_turn,
                    "source_turn": source_turn,
                    "rows": len(output_rows),
                    "skipped_missing_plan": skipped_missing_plan,
                    "skipped_exhausted": skipped_exhausted,
                    "output_path": output_path,
                }
            )

            if args.dry_run:
                continue

            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
                writer.writeheader()
                for row in output_rows:
                    writer.writerow(format_row(row, headers))

    print("summary:")
    for item in summary:
        print(
            "  "
            f"turn={item['turn']} "
            f"ref={item['refered_turn']} "
            f"source_turn={item['source_turn']} "
            f"rows={item['rows']} "
            f"missing_plan={item['skipped_missing_plan']} "
            f"exhausted={item['skipped_exhausted']}"
        )


if __name__ == "__main__":
    main()
