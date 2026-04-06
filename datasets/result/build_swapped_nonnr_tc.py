import argparse
import ast
import csv
import random
import re
from pathlib import Path

import pandas as pd


DEFAULT_PLANS = ["search_location", "send_email", "send_message", "web_search"]


def make_candidate_path(path):
    return Path(f"{path}.candidate")


def parse_history(value):
    history = ast.literal_eval(value)
    if not isinstance(history, list):
        raise ValueError("conversation_history must parse to a list.")
    return history


def parse_answer_plan(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        return None
    return parsed.get("plan")


def rewrite_turn_label(turn_text, target_turn):
    return re.sub(r"^turn\s+\d+:", f"turn {target_turn}:", turn_text, count=1)


def relabel_history(history, start_turn):
    return [
        rewrite_turn_label(turn_text, start_turn + idx)
        for idx, turn_text in enumerate(history)
    ]


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames
    return fieldnames, rows


def build_row_match_key(query, rewrited_query, answer):
    return (query, rewrited_query, answer)


def load_result_match_keys(path, plans):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["plan_name"] = df["gt"].apply(parse_answer_plan)
    df = df[df["plan_name"].isin(plans)].copy()

    keys_by_file = {}
    for file_name, group in df.groupby("file"):
        keys_by_file[file_name] = {
            build_row_match_key(row["query"], row["rewrited_query"], row["gt"])
            for _, row in group.iterrows()
        }
    return keys_by_file


def filter_plan_rows(rows, plans, allowed_keys=None):
    filtered = []
    for idx, row in enumerate(rows):
        plan = parse_answer_plan(row["answer"])
        if plan not in plans:
            continue

        match_key = build_row_match_key(
            row["query"], row["rewrited_query"], row["answer"]
        )
        if allowed_keys is not None and match_key not in allowed_keys:
            continue

        copied = dict(row)
        copied["_row_index"] = idx
        copied["_plan"] = plan
        copied["_history"] = parse_history(row["conversation_history"])
        filtered.append(copied)
    return filtered


def validate_history_length(rows, turn, label):
    expected = turn - 1
    for row in rows:
        actual = len(row["_history"])
        if actual != expected:
            raise ValueError(
                f"{label} row for plan `{row['_plan']}` has history length {actual}, "
                f"expected {expected} for turn {turn}."
            )


def sample_with_replacement(rng, rows, count):
    if not rows and count:
        raise ValueError("Cannot sample from an empty row set.")
    return [rng.choice(rows) for _ in range(count)]


def build_target_history(source_row, source_turn, target_turn, target_template_row=None):
    source_history = source_row["_history"]
    target_len = target_turn - 1
    source_len = source_turn - 1

    if len(source_history) != source_len:
        raise ValueError(
            f"Source row history length {len(source_history)} does not match "
            f"source turn {source_turn}."
        )

    if target_turn == source_turn:
        return repr(relabel_history(source_history, 1))

    if target_turn < source_turn:
        kept = source_history[-target_len:] if target_len > 0 else []
        return repr(relabel_history(kept, 1))

    if target_template_row is None:
        raise ValueError("A target template row is required when expanding history.")

    template_history = target_template_row["_history"]
    expected_template_len = target_turn - 1
    if len(template_history) != expected_template_len:
        raise ValueError(
            f"Target template history length {len(template_history)} does not match "
            f"target turn {target_turn}."
        )

    prefix_len = target_turn - source_turn
    prefix = template_history[:prefix_len]
    shifted_source = relabel_history(source_history, prefix_len + 1)
    return repr(prefix + shifted_source)


def build_target_row(source_row, source_turn, target_turn, target_template_row=None):
    new_row = dict(source_row)
    new_row["conversation_history"] = build_target_history(
        source_row, source_turn, target_turn, target_template_row=target_template_row
    )
    return new_row


def strip_internal_keys(rows):
    cleaned = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if not k.startswith("_")})
    return cleaned


def sort_rows_by_index(rows):
    return sorted(rows, key=lambda row: row["_row_index"])


def build_swapped_rows(rows_a, rows_b, turn_a, turn_b, plans, seed):
    rng = random.Random(seed)
    out_a = []
    out_b = []

    for plan in plans:
        plan_a = [row for row in rows_a if row["_plan"] == plan]
        plan_b = [row for row in rows_b if row["_plan"] == plan]

        if not plan_a and not plan_b:
            continue
        if not plan_a:
            raise ValueError(f"No rows found in input-a for plan `{plan}`.")
        if not plan_b:
            raise ValueError(f"No rows found in input-b for plan `{plan}`.")

        if turn_b > turn_a:
            sampled_templates_for_b = sample_with_replacement(rng, plan_b, len(plan_a))
            out_b.extend(
                build_target_row(
                    source_row=row_a,
                    source_turn=turn_a,
                    target_turn=turn_b,
                    target_template_row=template_b,
                )
                for row_a, template_b in zip(plan_a, sampled_templates_for_b)
            )
        else:
            out_b.extend(
                build_target_row(
                    source_row=row_a,
                    source_turn=turn_a,
                    target_turn=turn_b,
                )
                for row_a in plan_a
            )

        if turn_a > turn_b:
            sampled_templates_for_a = sample_with_replacement(rng, plan_a, len(plan_b))
            out_a.extend(
                build_target_row(
                    source_row=row_b,
                    source_turn=turn_b,
                    target_turn=turn_a,
                    target_template_row=template_a,
                )
                for row_b, template_a in zip(plan_b, sampled_templates_for_a)
            )
        else:
            out_a.extend(
                build_target_row(
                    source_row=row_b,
                    source_turn=turn_b,
                    target_turn=turn_a,
                )
                for row_b in plan_b
            )

    return sort_rows_by_index(out_a), sort_rows_by_index(out_b)


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_change_report_row(
    *,
    operation,
    source_path,
    output_path,
    source_turn,
    target_turn,
    target_index,
    original_row,
    swapped_row,
):
    return {
        "operation": operation,
        "source_file": str(source_path),
        "output_file": str(output_path),
        "plan": (
            parse_answer_plan(swapped_row["answer"])
            if swapped_row is not None
            else parse_answer_plan(original_row["answer"])
        ),
        "source_turn": source_turn,
        "target_turn": target_turn,
        "target_index": target_index,
        "query": swapped_row["query"] if swapped_row is not None else "",
        "rewrited_query": (
            swapped_row["rewrited_query"] if swapped_row is not None else ""
        ),
        "answer": swapped_row["answer"] if swapped_row is not None else "",
        "original_conversation_history": (
            original_row["conversation_history"] if original_row is not None else ""
        ),
        "swapped_conversation_history": (
            swapped_row["conversation_history"] if swapped_row is not None else ""
        ),
    }


def build_change_report_rows(
    original_target_rows,
    swapped_rows,
    source_path,
    output_path,
    source_turn,
    target_turn,
):
    rows = []
    max_len = max(len(original_target_rows), len(swapped_rows))
    for idx in range(max_len):
        original_row = original_target_rows[idx] if idx < len(original_target_rows) else None
        swapped_row = swapped_rows[idx] if idx < len(swapped_rows) else None
        if original_row is not None and swapped_row is not None:
            operation = "replace"
            target_index = original_row["_row_index"]
        elif original_row is not None:
            operation = "remove"
            target_index = original_row["_row_index"]
        else:
            operation = "insert"
            target_index = ""
        rows.append(
            build_change_report_row(
                operation=operation,
                source_path=source_path,
                output_path=output_path,
                source_turn=source_turn,
                target_turn=target_turn,
                target_index=target_index,
                original_row=original_row,
                swapped_row=strip_internal_keys([swapped_row])[0] if swapped_row is not None else None,
            )
        )
    return rows


def write_result_report(path, rows):
    fieldnames = [
        "operation",
        "source_file",
        "output_file",
        "plan",
        "source_turn",
        "target_turn",
        "target_index",
        "query",
        "rewrited_query",
        "answer",
        "original_conversation_history",
        "swapped_conversation_history",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def replace_rows_in_full_dataset(all_rows, original_target_rows, swapped_rows):
    target_rows = sort_rows_by_index(original_target_rows)
    cleaned_swapped_rows = strip_internal_keys(swapped_rows)

    target_positions = [row["_row_index"] for row in target_rows]
    target_position_set = set(target_positions)
    replaced = []
    replaced_count = 0
    swap_idx = 0
    last_target_position = target_positions[-1] if target_positions else None

    for row_idx, row in enumerate(all_rows):
        if row_idx not in target_position_set:
            replaced.append(row)
            continue

        if swap_idx < len(cleaned_swapped_rows):
            replaced.append(cleaned_swapped_rows[swap_idx])
            replaced_count += 1
            swap_idx += 1

        if row_idx == last_target_position:
            while swap_idx < len(cleaned_swapped_rows):
                replaced.append(cleaned_swapped_rows[swap_idx])
                replaced_count += 1
                swap_idx += 1

    if last_target_position is None:
        replaced.extend(cleaned_swapped_rows)
        replaced_count += len(cleaned_swapped_rows)

    return replaced, replaced_count


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create swapped nonNR TC files for selected plans between two arbitrary "
            "turn datasets."
        )
    )
    parser.add_argument(
        "--input-a",
        "--it2-input",
        dest="input_a",
        default="/home/hj153lee/RMA/datasets/tc/it2_nonNR_tc.tsv",
        help="Path to the first source TC TSV.",
    )
    parser.add_argument(
        "--turn-a",
        type=int,
        default=2,
        help="Turn number represented by input-a. Default: 2.",
    )
    parser.add_argument(
        "--input-b",
        "--it5-input",
        dest="input_b",
        default="/home/hj153lee/RMA/datasets/tc/it5_nonNR_tc.tsv",
        help="Path to the second source TC TSV.",
    )
    parser.add_argument(
        "--turn-b",
        type=int,
        default=5,
        help="Turn number represented by input-b. Default: 5.",
    )
    parser.add_argument(
        "--plans",
        nargs="+",
        default=DEFAULT_PLANS,
        help="Plans to include in the swapped test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for template sampling. Default: 42.",
    )
    parser.add_argument(
        "--output-a",
        "--it2-output",
        dest="output_a",
        help="Output path for the turn-a swapped TC TSV. Default: input-a + .candidate",
    )
    parser.add_argument(
        "--output-b",
        "--it5-output",
        dest="output_b",
        help="Output path for the turn-b swapped TC TSV. Default: input-b + .candidate",
    )
    parser.add_argument(
        "--report-output",
        default="result.tsv",
        help="Path to a TSV report containing only changed rows. Default: result.tsv",
    )
    parser.add_argument(
        "--result-input",
        default=None,
        help=(
            "Optional result TSV used to limit rows to items that actually appeared "
            "in evaluation output. Disabled by default."
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    input_a_path = Path(args.input_a)
    input_b_path = Path(args.input_b)
    output_a_path = Path(args.output_a) if args.output_a else make_candidate_path(input_a_path)
    output_b_path = Path(args.output_b) if args.output_b else make_candidate_path(input_b_path)
    report_output_path = Path(args.report_output)
    plans = set(args.plans)

    fieldnames_a, rows_a = load_rows(input_a_path)
    fieldnames_b, rows_b = load_rows(input_b_path)

    if fieldnames_a != fieldnames_b:
        raise ValueError("input-a and input-b TSV schemas do not match.")

    allowed_keys_by_file = None
    if args.result_input:
        allowed_keys_by_file = load_result_match_keys(Path(args.result_input), plans)

    filtered_a = filter_plan_rows(
        rows_a,
        plans,
        allowed_keys=(
            allowed_keys_by_file.get(input_a_path.name, set())
            if allowed_keys_by_file is not None
            else None
        ),
    )
    filtered_b = filter_plan_rows(
        rows_b,
        plans,
        allowed_keys=(
            allowed_keys_by_file.get(input_b_path.name, set())
            if allowed_keys_by_file is not None
            else None
        ),
    )

    validate_history_length(filtered_a, args.turn_a, input_a_path.name)
    validate_history_length(filtered_b, args.turn_b, input_b_path.name)

    swapped_a, swapped_b = build_swapped_rows(
        filtered_a,
        filtered_b,
        args.turn_a,
        args.turn_b,
        args.plans,
        args.seed,
    )

    full_output_a_rows, replaced_count_a = replace_rows_in_full_dataset(
        rows_a, filtered_a, swapped_a
    )
    full_output_b_rows, replaced_count_b = replace_rows_in_full_dataset(
        rows_b, filtered_b, swapped_b
    )

    write_rows(output_a_path, fieldnames_a, full_output_a_rows)
    write_rows(output_b_path, fieldnames_b, full_output_b_rows)

    report_rows = []
    report_rows.extend(
        build_change_report_rows(
            filtered_b,
            swapped_a,
            input_b_path,
            output_a_path,
            args.turn_b,
            args.turn_a,
        )
    )
    report_rows.extend(
        build_change_report_rows(
            filtered_a,
            swapped_b,
            input_a_path,
            output_b_path,
            args.turn_a,
            args.turn_b,
        )
    )
    write_result_report(report_output_path, report_rows)

    print(f"Swapped turn-{args.turn_a} output written to: {output_a_path}")
    print(f"Swapped turn-{args.turn_b} output written to: {output_b_path}")
    print(f"Change report written to: {report_output_path}")
    print(f"Input A: {input_a_path} (turn {args.turn_a})")
    print(f"Input B: {input_b_path} (turn {args.turn_b})")
    print(f"Plans: {', '.join(args.plans)}")
    print(f"Seed: {args.seed}")
    if args.result_input:
        print(f"Result filter: {args.result_input}")
    print(
        f"Rows replaced in candidates: output-a={replaced_count_a}, "
        f"output-b={replaced_count_b}"
    )


if __name__ == "__main__":
    main()
