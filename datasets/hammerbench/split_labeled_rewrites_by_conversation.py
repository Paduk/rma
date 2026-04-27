#!/usr/bin/env python3

import json
import random
from pathlib import Path


HAMMERBENCH_DIR = Path(__file__).resolve().parent
LABEL_BATCH_DIR = HAMMERBENCH_DIR / "labels" / "phase1_batches"
SOURCE_EVAL_PATH = (
    HAMMERBENCH_DIR
    / "en"
    / "multi-turn.phase1_external_mQsA.turn_gt_1.external.eval.jsonl"
)
OUTPUT_ROOT = HAMMERBENCH_DIR / "labels" / "splits"

SPLIT_CONFIGS = [
    {
        "name": "ver1-train7",
        "seed": 180,
        "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
    },
    {
        "name": "ver2-train1",
        "seed": 1292,
        "ratios": {"train": 0.1, "val": 0.1, "test": 0.8},
    },
    {
        "name": "ver3-train3",
        "seed": 3,
        "ratios": {"train": 0.3, "val": 0.0, "test": 0.7},
    },
]


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


def load_source_rows():
    rows = read_jsonl(SOURCE_EVAL_PATH)
    by_id = {}
    for row in rows:
        row_id = row["id"]
        if row_id in by_id:
            raise ValueError(f"Duplicate source id: {row_id}")
        by_id[row_id] = row
    return rows, by_id


def load_label_rows():
    rows = []
    batch_files = sorted(LABEL_BATCH_DIR.glob("batch_*.jsonl"))
    for path in batch_files:
        rows.extend(read_jsonl(path))
    return batch_files, rows


def build_merged_rows(source_by_id, label_rows):
    merged_rows = []
    seen = set()
    for label in label_rows:
        row_id = label["id"]
        if row_id in seen:
            raise ValueError(f"Duplicate labeled id: {row_id}")
        seen.add(row_id)

        if row_id not in source_by_id:
            raise KeyError(f"Labeled id not found in source eval file: {row_id}")
        source = source_by_id[row_id]

        if label["arguments"] != source["gold_call"]["arguments"]:
            raise ValueError(
                f"Labeled arguments do not match source gold_call for id={row_id}"
            )

        merged = dict(source)
        merged["rewrited_query"] = label["rewrited_query"]
        merged["arguments"] = label["arguments"]
        merged_rows.append(merged)
    return merged_rows


def build_splits(conversation_ids, seed, ratios):
    shuffled = list(conversation_ids)
    random.Random(seed).shuffle(shuffled)

    train_count = round(len(shuffled) * ratios["train"])
    val_count = round(len(shuffled) * ratios["val"])
    test_count = len(shuffled) - train_count - val_count

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]

    if len(train_ids) + len(val_ids) + len(test_ids) != len(shuffled):
        raise AssertionError("Split sizes do not cover all conversation_ids")
    if test_count != len(test_ids):
        raise AssertionError("Unexpected test split size")

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def summarize_split(rows, conversation_ids):
    return {
        "snapshot_count": len(rows),
        "conversation_count": len(conversation_ids),
        "conversation_ids": conversation_ids,
    }


def main():
    source_rows, source_by_id = load_source_rows()
    batch_files, label_rows = load_label_rows()
    merged_rows = build_merged_rows(source_by_id, label_rows)

    if len(source_rows) != len(merged_rows):
        raise ValueError(
            f"Source rows ({len(source_rows)}) and labeled rows ({len(merged_rows)}) differ"
        )

    merged_by_id = {row["id"]: row for row in merged_rows}
    ordered_merged_rows = [merged_by_id[row["id"]] for row in source_rows]

    conversation_ids = sorted({row["conversation_id"] for row in ordered_merged_rows})
    rows_by_split_name = {}

    for config in SPLIT_CONFIGS:
        split_dir = OUTPUT_ROOT / config["name"]
        split_dir.mkdir(parents=True, exist_ok=True)

        split_conversation_ids = build_splits(
            conversation_ids=conversation_ids,
            seed=config["seed"],
            ratios=config["ratios"],
        )

        metadata = {
            "split_name": config["name"],
            "seed": config["seed"],
            "ratios": config["ratios"],
            "source_eval_path": str(SOURCE_EVAL_PATH),
            "label_batch_dir": str(LABEL_BATCH_DIR),
            "label_batch_files": [str(path) for path in batch_files],
            "total_snapshot_count": len(ordered_merged_rows),
            "total_conversation_count": len(conversation_ids),
            "splits": {},
        }

        for split_name in ["train", "val", "test"]:
            conversation_id_set = set(split_conversation_ids[split_name])
            split_rows = [
                row
                for row in ordered_merged_rows
                if row["conversation_id"] in conversation_id_set
            ]
            output_path = split_dir / f"rewrite_labeled.{split_name}.jsonl"
            write_jsonl(output_path, split_rows)

            metadata["splits"][split_name] = {
                **summarize_split(split_rows, split_conversation_ids[split_name]),
                "output_path": str(output_path),
            }

        metadata_path = split_dir / "split_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")

        rows_by_split_name[config["name"]] = metadata

    print(f"source_eval_path\t{SOURCE_EVAL_PATH}")
    print(f"label_batch_dir\t{LABEL_BATCH_DIR}")
    print(f"label_batch_files\t{len(batch_files)}")
    print(f"total_labeled_rows\t{len(ordered_merged_rows)}")
    print(f"total_conversations\t{len(conversation_ids)}")
    for config in SPLIT_CONFIGS:
        metadata = rows_by_split_name[config["name"]]
        print(f"split_version\t{config['name']}\tseed={config['seed']}")
        for split_name in ["train", "val", "test"]:
            summary = metadata["splits"][split_name]
            print(
                f"{config['name']}.{split_name}\t"
                f"conversations={summary['conversation_count']}\t"
                f"snapshots={summary['snapshot_count']}\t"
                f"path={summary['output_path']}"
            )


if __name__ == "__main__":
    main()
