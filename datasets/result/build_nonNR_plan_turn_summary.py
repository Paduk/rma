import argparse
from pathlib import Path

import pandas as pd


TURN_COLUMNS = [2, 3, 4, 5]


def build_summary_table(df):
    non_nr_df = df[df["bucket"] == "nonNR"].copy()
    if non_nr_df.empty:
        return pd.DataFrame(
            columns=[
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
        )

    accuracy = (
        non_nr_df.pivot_table(
            index="plan", columns="turn", values="accuracy", aggfunc="first"
        )
        .reindex(columns=TURN_COLUMNS)
        .rename(columns=lambda turn: f"turn{turn}_accuracy")
    )
    support = (
        non_nr_df.pivot_table(
            index="plan", columns="turn", values="support", aggfunc="first"
        )
        .reindex(columns=TURN_COLUMNS)
        .rename(columns=lambda turn: f"turn{turn}_support")
    )

    summary = pd.concat([accuracy, support], axis=1).reset_index()
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
        description=(
            "Build a plan-level nonNR summary from plan_turn_file_performance.tsv."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to plan_turn_file_performance.tsv.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Optional output TSV path. Default: "
            "<input_dir>/plan_nonNR_turn_summary.tsv"
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "plan_nonNR_turn_summary.tsv"
    )

    df = pd.read_csv(input_path, sep="\t")
    required_columns = {"turn", "bucket", "plan", "accuracy", "support"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input TSV is missing required columns: {sorted(missing)}")

    summary_df = build_summary_table(df)
    summary_df.to_csv(output_path, sep="\t", index=False)
    print(f"nonNR plan-turn summary exported to: {output_path}")


if __name__ == "__main__":
    main()
