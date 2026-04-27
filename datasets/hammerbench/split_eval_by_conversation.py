#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


HAMMERBENCH_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HAMMERBENCH_DIR
    / "en"
    / "multi-turn.phase1_external_mQsA.turn_gt_1.external.eval.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    HAMMERBENCH_DIR
    / "en"
    / "splits"
    / "multi-turn.phase1_external_mQsA.turn_gt_1.external.eval.seed180"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a HammerBench eval JSONL into train/val/test by conversation_id. "
            "All snapshots from the same conversation stay in the same split."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input eval JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write train/val/test JSONL files and split metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=180,
        help="Random seed used to shuffle conversation_ids before splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio over unique conversation_ids.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio over unique conversation_ids.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio over unique conversation_ids.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def build_splits(conversation_ids, seed, train_ratio, val_ratio, test_ratio):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    shuffled = list(conversation_ids)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = round(total * train_ratio)
    val_count = round(total * val_ratio)
    test_count = total - train_count - val_count

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]

    if len(train_ids) + len(val_ids) + len(test_ids) != total:
        raise AssertionError("Split sizes do not cover all conversation_ids")

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def summarize_split(name, rows, conversation_ids):
    return {
        "name": name,
        "conversation_count": len(conversation_ids),
        "snapshot_count": len(rows),
        "conversation_ids": conversation_ids,
    }


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = read_jsonl(args.input)
    rows_by_conversation = {}
    for row in rows:
        conversation_id = row.get("conversation_id")
        if conversation_id is None:
            raise KeyError(f"Missing conversation_id in row id={row.get('id')}")
        rows_by_conversation.setdefault(conversation_id, []).append(row)

    conversation_ids = sorted(rows_by_conversation)
    splits = build_splits(
        conversation_ids=conversation_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input_path": str(args.input),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "total_snapshot_count": len(rows),
        "total_conversation_count": len(conversation_ids),
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        split_ids = splits[split_name]
        split_rows = []
        for conversation_id in split_ids:
            split_rows.extend(rows_by_conversation[conversation_id])

        output_path = args.output_dir / f"{args.input.stem}.{split_name}.jsonl"
        write_jsonl(output_path, split_rows)

        metadata["splits"][split_name] = {
            **summarize_split(split_name, split_rows, split_ids),
            "output_path": str(output_path),
        }

    metadata_path = args.output_dir / f"{args.input.stem}.split_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"input_path\t{args.input}")
    print(f"output_dir\t{args.output_dir}")
    print(f"seed\t{args.seed}")
    print(f"total_snapshots\t{len(rows)}")
    print(f"total_conversations\t{len(conversation_ids)}")
    for split_name in ["train", "val", "test"]:
        summary = metadata["splits"][split_name]
        print(
            f"{split_name}\t"
            f"conversations={summary['conversation_count']}\t"
            f"snapshots={summary['snapshot_count']}\t"
            f"path={summary['output_path']}"
        )
    print(f"metadata_path\t{metadata_path}")


if __name__ == "__main__":
    main()
