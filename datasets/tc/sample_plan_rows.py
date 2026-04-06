#!/usr/bin/env python3

from __future__ import annotations

import ast
import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/home/hj153lee/RMA/datasets/tc")
INPUT_FILES = [
    "it2_nonNR_tc.tsv",
    "it3_nonNR_tc.tsv",
    "it4_nonNR_tc.tsv",
    "it5_nonNR_tc.tsv",
    "it3_complex_1_tc.tsv",
    "it4_complex_1_tc.tsv",
    "it4_complex_2_tc.tsv",
    "it5_complex_1_tc.tsv",
    "it5_complex_2_tc.tsv",
    "it5_complex_3_tc.tsv",
]
MERGED_OUTPUT = BASE_DIR / "merged_tc.tsv"
SAMPLED_OUTPUT = BASE_DIR / "plan_sampled_10_per_plan.tsv"
SAMPLE_SIZE = 10
RANDOM_SEED = 42


def extract_plan(answer_value: str) -> str:
    parsed = ast.literal_eval(answer_value)
    return parsed["plan"]


def extract_source_group(file_name: str) -> str:
    return Path(file_name).stem.split("_")[0]


def parse_quota_map(quota_items: list[str] | None) -> dict[str, int]:
    if not quota_items:
        return {}

    quota_map: dict[str, int] = {}
    for item in quota_items:
        key, value = item.split("=", 1)
        quota_map[key] = int(value)
    return quota_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", default=INPUT_FILES)
    parser.add_argument("--merged-output")
    parser.add_argument("--sampled-output", default=str(SAMPLED_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--source-quotas", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampled_output = Path(args.sampled_output)
    quota_map = parse_quota_map(args.source_quotas)

    frames = []
    for file_name in args.input_files:
        frame = pd.read_csv(BASE_DIR / file_name, sep="\t")
        frame["source_file"] = file_name
        frame["source_group"] = extract_source_group(file_name)
        frames.append(frame)

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["plan"] = merged_df["answer"].map(extract_plan)
    if args.merged_output:
        merged_output = Path(args.merged_output)
        merged_df.to_csv(merged_output, sep="\t", index=False)

    sampled_frames = []
    for _, group in merged_df.groupby("plan"):
        if quota_map:
            quota_samples = []
            for source_group, quota in quota_map.items():
                source_rows = group[group["source_group"] == source_group]
                if len(source_rows) == 0:
                    continue
                quota_samples.append(
                    source_rows.sample(n=min(len(source_rows), quota), random_state=RANDOM_SEED)
                )

            if not quota_samples:
                continue

            sampled_group = pd.concat(quota_samples, ignore_index=False)
            if len(sampled_group) > args.sample_size:
                sampled_group = sampled_group.sample(n=args.sample_size, random_state=RANDOM_SEED)
        else:
            sampled_group = group.sample(n=min(len(group), args.sample_size), random_state=RANDOM_SEED)

        sampled_frames.append(sampled_group)

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    sampled_df.to_csv(sampled_output, sep="\t", index=False)

    print(f"Merged rows: {len(merged_df)}")
    print(f"Sampled rows: {len(sampled_df)}")
    if args.merged_output:
        print(f"Merged output: {merged_output}")
    print(f"Sampled output: {sampled_output}")


if __name__ == "__main__":
    main()
