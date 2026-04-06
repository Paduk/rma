import argparse
import ast
from pathlib import Path

import pandas as pd


PRIORITY_OFFENDERS = [
    {"plan": "send_message", "section": "turn_trend"},
    {"plan": "ACTION_EDIT_ALARM", "section": "turn3_bucket_offenders"},
    {"plan": "ACTION_VIEW_CONTACT", "section": "turn3_bucket_offenders"},
    {"plan": "ACTION_VIEW_CONTACT", "section": "turn5_bucket_offenders"},
    {"plan": "send_email", "section": "turn4_bucket_offenders"},
    {"plan": "ACTION_OPEN_CONTENT", "section": "turn5_bucket_offenders"},
    {"plan": "ACTION_EDIT_CONTACT", "section": "turn3_bucket_offenders"},
    {"plan": "send_message", "section": "turn4_bucket_offenders"},
    {"plan": "search_location", "section": "turn5_bucket_offenders"},
    {"plan": "ACTION_NAVIGATE_TO_LOCATION", "section": "turn_trend"},
]

TURN_BUCKET_RULES = {
    "turn3_bucket_offenders": {"turn": 3, "buckets": ["complex_1", "nonNR"]},
    "turn4_bucket_offenders": {
        "turn": 4,
        "buckets": ["complex_1", "complex_2", "nonNR"],
    },
    "turn5_bucket_offenders": {
        "turn": 5,
        "buckets": ["complex_1", "complex_2", "complex_3", "nonNR"],
    },
}


def safe_parse_gt_plan(value):
    if pd.isnull(value):
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed.get("plan")
    return None


def get_file_bucket(file_name):
    if pd.isnull(file_name):
        return None

    import re

    complex_match = re.search(r"complex_(\d+)", file_name)
    if complex_match:
        return f"complex_{complex_match.group(1)}"
    if "nonNR" in file_name:
        return "nonNR"
    return None


def load_result_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    df["gt_plan"] = df["gt"].apply(safe_parse_gt_plan)
    df["bucket"] = df["file"].apply(get_file_bucket)
    df["is_fail"] = df["all"].str.lower() != "pass"
    return df


def sanitize_name(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def filter_offender_rows(df, offender):
    plan = offender["plan"]
    section = offender["section"]

    subset = df[df["gt_plan"] == plan].copy()
    if section == "turn_trend":
        return subset[subset["turn"].isin([2, 3, 4, 5])].copy()

    rule = TURN_BUCKET_RULES[section]
    return subset[
        (subset["turn"] == rule["turn"]) & (subset["bucket"].isin(rule["buckets"]))
    ].copy()


def sample_pass_rows(df, max_pass_samples):
    if df.empty:
        return df

    group_cols = [col for col in ["turn", "bucket"] if col in df.columns]
    if not group_cols:
        return df.head(max_pass_samples).copy()

    sampled = []
    for _, group in df.groupby(group_cols, dropna=False):
        sampled.append(group.head(max_pass_samples))
    return pd.concat(sampled, ignore_index=True)


def build_review_df(df, offender, max_pass_samples):
    subset = filter_offender_rows(df, offender)
    if subset.empty:
        return subset

    fail_df = subset[subset["is_fail"]].copy()
    pass_df = subset[~subset["is_fail"]].copy()
    pass_sample_df = sample_pass_rows(pass_df, max_pass_samples=max_pass_samples)

    fail_df["review_label"] = "fail"
    pass_sample_df["review_label"] = "pass_sample"

    review_df = pd.concat([fail_df, pass_sample_df], ignore_index=True)
    review_df["review_section"] = offender["section"]
    review_df["review_plan"] = offender["plan"]
    review_df["source_model"] = Path(args.input).name

    sort_cols = [col for col in ["turn", "bucket", "review_label"] if col in review_df.columns]
    if sort_cols:
        review_df = review_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return review_df


def write_review_files(df, output_dir, max_pass_samples):
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for offender in PRIORITY_OFFENDERS:
        review_df = build_review_df(df, offender, max_pass_samples=max_pass_samples)
        if review_df.empty:
            continue

        file_stem = (
            f"{sanitize_name(offender['plan'])}__{sanitize_name(offender['section'])}"
        )
        out_path = output_dir / f"{file_stem}.tsv"
        review_df.to_csv(out_path, sep="\t", index=False)

        manifest_rows.append(
            {
                "plan": offender["plan"],
                "section": offender["section"],
                "output_file": str(out_path),
                "fail_count": int((review_df["review_label"] == "fail").sum()),
                "pass_sample_count": int(
                    (review_df["review_label"] == "pass_sample").sum()
                ),
                "total_rows": len(review_df),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "review_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path, manifest_df


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract review TSVs for a fixed set of priority offender "
            "plan/section pairs."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single result TSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write review TSVs.",
    )
    parser.add_argument(
        "--max-pass-samples",
        type=int,
        default=5,
        help=(
            "Maximum number of pass samples to keep per turn/bucket group. "
            "Default: 5."
        ),
    )
    return parser


def main():
    global args
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = load_result_tsv(input_path)
    manifest_path, manifest_df = write_review_files(
        df, output_dir=output_dir, max_pass_samples=args.max_pass_samples
    )

    print(f"Review files written to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if not manifest_df.empty:
        print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
