import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDENTITY_FILE = ROOT / "datasets/tc/scale/query_eq_rewrited_query_rows.tsv"
SCORE_COLUMNS = ["plan", "arguments", "all"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Match TC rows where query == rewrited_query against a pipeline result TSV "
            "and report their evaluation results."
        )
    )
    parser.add_argument("input_tsv", type=Path, help="Pipeline result TSV file.")
    parser.add_argument(
        "--identity-tsv",
        type=Path,
        default=DEFAULT_IDENTITY_FILE,
        help=f"TSV made from TC rows where query == rewrited_query. Default: {DEFAULT_IDENTITY_FILE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path. Default: next to input TSV with .identity_eval.tsv suffix.",
    )
    parser.add_argument(
        "--print-rows",
        action="store_true",
        help="Also print matched rows to stdout.",
    )
    parser.add_argument(
        "--match-without-history",
        action="store_true",
        help=(
            "Match only on source_file/file, query, and rewrited_query/gt_rewrited_query. "
            "By default, conversation_history is also used to avoid duplicate key fan-out."
        ),
    )
    return parser.parse_args()


def require_columns(df, path, columns):
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")


def normalize_score(value):
    value = str(value).strip().lower()
    if value == "pass":
        return True
    if value == "fail":
        return False
    return pd.NA


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}.identity_eval.tsv")


def main():
    args = parse_args()
    input_path = args.input_tsv.resolve()
    identity_path = args.identity_tsv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(input_path)

    identity_df = pd.read_csv(identity_path, sep="\t", dtype=str, keep_default_na=False)
    result_df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)

    identity_columns = ["source_file", "query", "rewrited_query"]
    result_columns = ["file", "query", "gt_rewrited_query", "gt", *SCORE_COLUMNS]
    if not args.match_without_history:
        identity_columns.append("conversation_history")
        result_columns.append("conversation_history")

    require_columns(identity_df, identity_path, identity_columns)
    require_columns(result_df, input_path, result_columns)

    identity_df = identity_df.reset_index(names="identity_row_index")
    result_df = result_df.reset_index(names="result_row_index")

    left_keys = ["source_file", "query", "rewrited_query"]
    right_keys = ["file", "query", "gt_rewrited_query"]
    if not args.match_without_history:
        left_keys.insert(1, "conversation_history")
        right_keys.insert(1, "conversation_history")

    merged = identity_df.merge(
        result_df,
        how="left",
        left_on=left_keys,
        right_on=right_keys,
        suffixes=("_identity", "_result"),
        indicator=True,
    )

    for column in SCORE_COLUMNS:
        merged[f"{column}_ok"] = merged[column].map(normalize_score)

    output_columns = [
        "source_file",
        "identity_row_index",
        "result_row_index",
        "conversation_history",
        "query",
        "rewrited_query",
        "generated_rewrited_query",
        "gt",
        "plan",
        "plan_ok",
        "arguments",
        "arguments_ok",
        "all",
        "all_ok",
        "test_key",
        "turn",
        "file",
        "_merge",
    ]
    output_columns = [column for column in output_columns if column in merged.columns]
    report_df = merged[output_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, sep="\t", index=False)

    total_identity_rows = len(identity_df)
    matched_rows = int((merged["_merge"] == "both").sum())
    unmatched_rows = int((merged["_merge"] == "left_only").sum())
    duplicate_matches = max(0, len(merged) - total_identity_rows)

    print(f"identity_file={identity_path}")
    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"match_keys={left_keys} -> {right_keys}")
    print(f"identity_rows={total_identity_rows}")
    print(f"matched_rows={matched_rows}")
    print(f"unmatched_rows={unmatched_rows}")
    print(f"duplicate_matches={duplicate_matches}")

    if matched_rows:
        print("score_summary:")
        for column in SCORE_COLUMNS:
            counts = merged.loc[merged["_merge"] == "both", column].value_counts(dropna=False)
            pass_count = int(counts.get("pass", 0))
            fail_count = int(counts.get("fail", 0))
            other_count = matched_rows - pass_count - fail_count
            print(f"  {column}: pass={pass_count} fail={fail_count} other={other_count}")

        all_fail_counts = (
            merged.loc[(merged["_merge"] == "both") & (merged["all"] == "fail"), "source_file"]
            .value_counts()
            .sort_index()
        )
        print("all_fail_by_source_file:")
        if all_fail_counts.empty:
            print("  none")
        else:
            for source_file, fail_count in all_fail_counts.items():
                print(f"  {source_file}: {fail_count}")

    if unmatched_rows:
        print("unmatched_rows_detail:")
        for row in merged.loc[merged["_merge"] == "left_only", ["source_file", "query", "rewrited_query"]].itertuples(index=False):
            print(f"  source_file={row.source_file!r} query={row.query!r} rewrited_query={row.rewrited_query!r}")

    if args.print_rows:
        print()
        print(report_df.to_csv(sep="\t", index=False), end="")


if __name__ == "__main__":
    main()
