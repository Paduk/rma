#!/usr/bin/env python3
import argparse
import ast
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


TARGET_SPECS = {
    3: [2],
    4: [2, 3],
    5: [2, 3, 4],
}

HUMAN_EVAL_COLUMNS = [
    "conversation_history",
    "ori_query",
    "ori_rewrited_query",
    "answer",
    "unique_idx",
    "refered_turn",
    "candidates",
    "plan",
    "new_query",
    "rewrited_query",
    "query",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build human-eval referred-turn TSVs by replacing one history turn "
            "with a balanced sample from scale nonNR data."
        )
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Relative paths are resolved from here.",
    )
    parser.add_argument(
        "--input-dir",
        default="datasets/tc/human-eval",
        help="Directory containing turn3.tsv, turn4.tsv, and turn5.tsv.",
    )
    parser.add_argument(
        "--nonnr-dir",
        default="datasets/tc/scale",
        help="Directory containing it3_nonNR_tc.tsv, it4_nonNR_tc.tsv, and it5_nonNR_tc.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/tc/human-eval/ref-synthetic",
        help="Directory where it*_ref*.tsv files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic balanced sampling.",
    )
    parser.add_argument(
        "--allow-source-reuse",
        action="store_true",
        help=(
            "Permit source row reuse if the available source rows are insufficient. "
            "By default, source rows are not reused."
        ),
    )
    return parser.parse_args()


def resolve_dir(root, directory):
    path = Path(directory)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"Missing header in {path}")
    return fieldnames, rows


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_history(raw_history, path, row_number):
    try:
        history = ast.literal_eval(raw_history)
    except Exception as exc:
        raise ValueError(f"Invalid conversation_history in {path} row {row_number}: {exc}") from exc
    if not isinstance(history, list):
        raise ValueError(f"conversation_history is not a list in {path} row {row_number}")
    return history


def parse_answer_plan(raw_answer):
    if raw_answer is None:
        return "MISSING_ANSWER"
    try:
        parsed = ast.literal_eval(str(raw_answer))
    except Exception:
        return "INVALID_ANSWER"
    if not isinstance(parsed, dict):
        return "INVALID_ANSWER"
    plan = parsed.get("plan")
    if plan is None:
        return "MISSING_PLAN"
    plan = str(plan).strip()
    return plan if plan else "EMPTY_PLAN"


def parse_unique_idx(unique_idx):
    parts = str(unique_idx).split("-")
    if len(parts) % 2 != 0:
        return []
    return [(parts[idx], parts[idx + 1]) for idx in range(0, len(parts), 2)]


def render_unique_idx(components):
    flat = []
    for plan, item_id in components:
        flat.extend([plan, item_id])
    return "-".join(flat)


def source_ref_plan(row, refered_turn):
    components = parse_unique_idx(row.get("unique_idx", ""))
    if len(components) < refered_turn:
        return "UNKNOWN_REF_PLAN"
    return components[refered_turn - 1][0]


def rebuild_unique_idx(base_unique_idx, source_unique_idx, refered_turn):
    base_components = parse_unique_idx(base_unique_idx)
    source_components = parse_unique_idx(source_unique_idx)
    if (
        not base_components
        or not source_components
        or len(base_components) < refered_turn + 1
        or len(source_components) < refered_turn + 1
    ):
        return source_unique_idx

    rebuilt = list(base_components)
    rebuilt[refered_turn - 1] = source_components[refered_turn - 1]
    rebuilt[-1] = source_components[-1]
    return render_unique_idx(rebuilt)


def validate_fieldnames(path, fieldnames, expected):
    if fieldnames != expected:
        raise ValueError(f"Unexpected columns in {path}: {fieldnames}")


def prepare_source_rows(source_path, source_rows, refered_turn):
    prepared = []
    for source_index, row in enumerate(source_rows, start=1):
        history = parse_history(row["conversation_history"], source_path, source_index)
        if len(history) < refered_turn:
            raise ValueError(
                f"{source_path} row {source_index} has only {len(history)} history turns; "
                f"cannot use refered_turn={refered_turn}"
            )
        prepared.append(
            {
                "row": row,
                "source_index": source_index,
                "history": history,
                "ref_plan": source_ref_plan(row, refered_turn),
                "current_plan": parse_answer_plan(row.get("answer")),
            }
        )
    return prepared


def select_balanced_sources(prepared_sources, count, seed, allow_source_reuse):
    buckets = defaultdict(list)
    for item in prepared_sources:
        buckets[item["ref_plan"]].append(item)

    rng = random.Random(seed)
    for plan in sorted(buckets):
        rng.shuffle(buckets[plan])

    selected = []
    offsets = Counter()
    plan_order = sorted(buckets)

    while len(selected) < count:
        made_progress = False
        for plan in plan_order:
            bucket = buckets[plan]
            if offsets[plan] >= len(bucket):
                if not allow_source_reuse:
                    continue
                rng.shuffle(bucket)
                offsets[plan] = 0

            selected.append(bucket[offsets[plan]])
            offsets[plan] += 1
            made_progress = True

            if len(selected) == count:
                break

        if not made_progress:
            raise ValueError("Could not select enough source rows without source reuse.")

    return selected


def build_output_row(base_item, source_item, refered_turn):
    base_row = base_item["row"]
    source_row = source_item["row"]
    new_history = list(base_item["history"])
    new_history[refered_turn - 1] = source_item["history"][refered_turn - 1]

    return {
        "conversation_history": repr(new_history),
        "ori_query": source_row["query"],
        "ori_rewrited_query": source_row["rewrited_query"],
        "answer": source_row["answer"],
        "unique_idx": rebuild_unique_idx(
            base_row.get("unique_idx", ""),
            source_row.get("unique_idx", ""),
            refered_turn,
        ),
        "refered_turn": str(refered_turn),
        "candidates": source_row.get("candidates", ""),
        "plan": parse_answer_plan(source_row.get("answer")),
        "new_query": source_row["query"],
        "rewrited_query": source_row["rewrited_query"],
        "query": source_row["query"],
    }


def prepare_base_rows(base_path, base_rows, target_turn):
    prepared = []
    expected_history_len = target_turn - 1
    for row_index, row in enumerate(base_rows, start=1):
        history = parse_history(row["conversation_history"], base_path, row_index)
        if len(history) != expected_history_len:
            raise ValueError(
                f"{base_path} row {row_index} has {len(history)} history turns; "
                f"expected {expected_history_len}"
            )
        prepared.append({"row": row, "row_index": row_index, "history": history})
    return prepared


def build_detail_rows(file_name, base_items, source_items, refered_turn):
    rows = []
    for base_item, source_item in zip(base_items, source_items):
        rows.append(
            {
                "file_name": file_name,
                "target_row": base_item["row_index"],
                "source_row": source_item["source_index"],
                "refered_turn": refered_turn,
                "source_ref_plan": source_item["ref_plan"],
                "source_current_plan": source_item["current_plan"],
                "source_unique_idx": source_item["row"].get("unique_idx", ""),
            }
        )
    return rows


def build_plan_summary_rows(file_name, selected_sources):
    ref_counts = Counter(item["ref_plan"] for item in selected_sources)
    current_counts = Counter(item["current_plan"] for item in selected_sources)
    rows = []
    for plan in sorted(set(ref_counts) | set(current_counts)):
        rows.append(
            {
                "file_name": file_name,
                "plan": plan,
                "selected_as_refered_turn_plan": ref_counts.get(plan, 0),
                "selected_as_current_plan": current_counts.get(plan, 0),
            }
        )
    return rows


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    input_dir = resolve_dir(root, args.input_dir)
    nonnr_dir = resolve_dir(root, args.nonnr_dir)
    output_dir = resolve_dir(root, args.output_dir)

    manifest_rows = []
    detail_rows = []
    plan_summary_rows = []

    for target_turn, refered_turns in TARGET_SPECS.items():
        base_path = input_dir / f"turn{target_turn}.tsv"
        base_fieldnames, base_rows = read_tsv(base_path)
        validate_fieldnames(base_path, base_fieldnames, HUMAN_EVAL_COLUMNS)
        base_items = prepare_base_rows(base_path, base_rows, target_turn)

        for refered_turn in refered_turns:
            source_turn = refered_turn + 1
            source_path = nonnr_dir / f"it{source_turn}_nonNR_tc.tsv"
            source_fieldnames, source_rows = read_tsv(source_path)
            validate_fieldnames(
                source_path,
                source_fieldnames,
                ["conversation_history", "query", "rewrited_query", "answer", "unique_idx", "candidates"],
            )
            source_items = prepare_source_rows(source_path, source_rows, refered_turn)
            selected_sources = select_balanced_sources(
                source_items,
                count=len(base_items),
                seed=args.seed + target_turn * 100 + refered_turn,
                allow_source_reuse=args.allow_source_reuse or len(source_items) < len(base_items),
            )

            output_file = f"it{target_turn}_ref{refered_turn}.tsv"
            output_path = output_dir / output_file
            output_rows = [
                build_output_row(base_item, source_item, refered_turn)
                for base_item, source_item in zip(base_items, selected_sources)
            ]
            write_tsv(output_path, HUMAN_EVAL_COLUMNS, output_rows)

            manifest_rows.append(
                {
                    "file_name": output_file,
                    "target_turn": target_turn,
                    "refered_turn": refered_turn,
                    "source_file": source_path.name,
                    "row_count": len(output_rows),
                    "output_path": str(output_path.relative_to(root)),
                }
            )
            detail_rows.extend(build_detail_rows(output_file, base_items, selected_sources, refered_turn))
            plan_summary_rows.extend(build_plan_summary_rows(output_file, selected_sources))

    write_tsv(
        output_dir / "ref_synthetic_manifest.tsv",
        ["file_name", "target_turn", "refered_turn", "source_file", "row_count", "output_path"],
        manifest_rows,
    )
    write_tsv(
        output_dir / "ref_synthetic_source_rows.tsv",
        [
            "file_name",
            "target_row",
            "source_row",
            "refered_turn",
            "source_ref_plan",
            "source_current_plan",
            "source_unique_idx",
        ],
        detail_rows,
    )
    write_tsv(
        output_dir / "ref_synthetic_plan_summary.tsv",
        ["file_name", "plan", "selected_as_refered_turn_plan", "selected_as_current_plan"],
        plan_summary_rows,
    )

    for row in manifest_rows:
        print(
            f"{row['file_name']}: {row['row_count']} rows "
            f"(target turn {row['target_turn']}, ref {row['refered_turn']}, source {row['source_file']})"
        )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
