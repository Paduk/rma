import argparse
import ast
from pathlib import Path

import pandas as pd


TURN_COLUMNS = [2, 3, 4, 5]


def safe_parse_plan(value):
    if pd.isnull(value):
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed.get("plan")
    return None


def build_plan_turn_summary(df, metric_col="all"):
    work_df = df.copy()
    work_df["turn"] = pd.to_numeric(work_df["turn"], errors="coerce")
    work_df["gt_plan"] = work_df["gt"].apply(safe_parse_plan)
    work_df["correct"] = work_df[metric_col].str.lower() == "pass"
    work_df = work_df.dropna(subset=["gt_plan", "turn"]).copy()

    accuracy = (
        work_df.groupby(["gt_plan", "turn"])["correct"]
        .mean()
        .unstack("turn")
        .reindex(columns=TURN_COLUMNS)
        .rename(columns=lambda turn: f"turn{turn}_accuracy")
    )
    support = (
        work_df.groupby(["gt_plan", "turn"]).size()
        .unstack("turn")
        .reindex(columns=TURN_COLUMNS)
        .rename(columns=lambda turn: f"turn{turn}_support")
    )

    summary = pd.concat([accuracy, support], axis=1).reset_index()
    summary = summary.rename(columns={"gt_plan": "plan"})
    ordered_columns = [
        "plan",
        "turn2_accuracy",
        "turn2_support",
        "turn3_accuracy",
        "turn3_support",
        "turn4_accuracy",
        "turn4_support",
        "turn5_accuracy",
        "turn5_support",
    ]
    return summary[ordered_columns].sort_values("plan").reset_index(drop=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export a plan-by-turn performance summary TSV from a result file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the result TSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Optional output TSV path. Default: "
            "<input_dir>/<input_stem>_plan_turn_summary.tsv"
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_plan_turn_summary.tsv"
    )

    df = pd.read_csv(input_path, sep="\t", dtype=str)
    required_columns = {"gt", "all", "turn"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input TSV is missing required columns: {sorted(missing)}")

    summary_df = build_plan_turn_summary(df)
    summary_df.to_csv(output_path, sep="\t", index=False)
    print(f"Plan turn summary exported to: {output_path}")


if __name__ == "__main__":
    main()
