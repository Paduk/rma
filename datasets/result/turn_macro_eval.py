import argparse
import ast
from pathlib import Path

import pandas as pd


TURN_WEIGHT_SCHEMES = {
    "curriculum": {
        2: {"nonNR": 1},
        3: {"complex_1": 7, "nonNR": 3},
        4: {"complex_1": 7, "complex_2": 2, "nonNR": 1},
        5: {"complex_1": 4, "complex_2": 3, "complex_3": 2, "nonNR": 1},
    }
}

REVERSED_TURN_WEIGHT_SCHEMES = {
    "curriculum": {
        2: {"nonNR": 1},
        3: {"complex_1": 3, "nonNR": 7},
        4: {"complex_1": 2, "complex_2": 3, "nonNR": 5},
        5: {"complex_1": 1, "complex_2": 2, "complex_3": 3, "nonNR": 4},
    }
}


def maybe_reverse_bucket_weights(bucket_weights, reverse=False):
    if not reverse:
        return bucket_weights
    bucket_names = list(bucket_weights.keys())
    reversed_values = list(bucket_weights.values())[::-1]
    return dict(zip(bucket_names, reversed_values))


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


def compute_plan_macro_accuracy(df, metric_col="all"):
    if df.empty:
        return float("nan")

    work_df = df.copy()
    if "gt_plan" not in work_df.columns:
        work_df["gt_plan"] = work_df["gt"].apply(safe_parse_gt_plan)

    work_df["correct"] = work_df[metric_col].str.lower() == "pass"
    macro_by_plan = (
        work_df.dropna(subset=["gt_plan"])
        .groupby("gt_plan")["correct"]
        .mean()
    )
    return macro_by_plan.mean()


def extract_turn_from_filename(file_name):
    import re

    match = re.search(r"it(\d+)", file_name)
    if not match:
        return None
    return int(match.group(1))


def get_turn_file_sort_key(file_name):
    import re

    turn = extract_turn_from_filename(file_name)
    complex_match = re.search(r"complex_(\d+)", file_name)
    if complex_match:
        return (turn if turn is not None else 999, 0, int(complex_match.group(1)))
    return (turn if turn is not None else 999, 1, file_name)


def get_plan_view_file_sort_key(file_name):
    import re

    turn = extract_turn_from_filename(file_name)
    if "nonNR" in str(file_name):
        return (0, 0, str(file_name))

    complex_match = re.search(r"complex_(\d+)", str(file_name))
    if complex_match:
        return (1, int(complex_match.group(1)), str(file_name))

    return (2, 999, str(file_name))


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


def compute_weighted_turn_macro_accuracy(
    turn_df, scheme_name, metric_col="all", reverse=False
):
    if not scheme_name:
        return float("nan")

    if reverse:
        scheme = REVERSED_TURN_WEIGHT_SCHEMES[scheme_name]
    else:
        scheme = TURN_WEIGHT_SCHEMES[scheme_name]
    turn_values = []

    for turn, bucket_weights in scheme.items():
        turn_slice = turn_df[turn_df["turn"] == turn]
        if turn_slice.empty:
            continue

        weighted_sum = 0.0
        weight_total = 0

        for bucket_name, bucket_weight in bucket_weights.items():
            bucket_df = turn_slice[
                turn_slice["file"].apply(get_file_bucket) == bucket_name
            ]
            if bucket_df.empty:
                continue
            bucket_macro = compute_plan_macro_accuracy(bucket_df, metric_col=metric_col)
            weighted_sum += bucket_macro * bucket_weight
            weight_total += bucket_weight

        if weight_total == 0:
            turn_values.append((turn, float("nan")))
        else:
            turn_values.append((turn, weighted_sum / weight_total))

    return turn_values


def prepare_analysis_df(df, metric_col="all"):
    work_df = df.copy()
    if "gt_plan" not in work_df.columns:
        work_df["gt_plan"] = work_df["gt"].apply(safe_parse_gt_plan)
    work_df["correct"] = work_df[metric_col].str.lower() == "pass"
    if "bucket" not in work_df.columns:
        work_df["bucket"] = work_df["file"].apply(get_file_bucket)
    return work_df.dropna(subset=["gt_plan"]).copy()


def compute_common_plan_turn_macro(work_df):
    turn_plan = (
        work_df.groupby(["turn", "gt_plan"])["correct"]
        .mean()
        .unstack("turn")
        .sort_index(axis=1)
    )
    required_turns = [turn for turn in [2, 3, 4, 5] if turn in turn_plan.columns]
    common_plan = turn_plan.dropna(subset=required_turns)
    return common_plan.mean(), common_plan


def format_turn_series(series):
    parts = []
    for turn, value in series.items():
        parts.append(f"T{int(turn)}={format_pct(value)}")
    return ", ".join(parts)


def print_turn_progress_report(work_df, top_k, min_support):
    turn_plan_acc = work_df.groupby(["gt_plan", "turn"])["correct"].mean().unstack("turn")
    turn_plan_support = work_df.groupby(["gt_plan", "turn"]).size().unstack("turn")

    required_turns = [turn for turn in [2, 3, 4, 5] if turn in turn_plan_acc.columns]
    comparable_mask = (
        turn_plan_acc[required_turns].notna().all(axis=1)
        & turn_plan_support[required_turns].fillna(0).ge(min_support).all(axis=1)
    )
    comparable = turn_plan_acc[comparable_mask].copy()
    if comparable.empty:
        print(
            f"No plans meet the turn comparison support threshold "
            f"(min_support={min_support})."
        )
        return

    comparable["t5_minus_t2"] = comparable[5] - comparable[2]
    print(
        f"Common-plan balanced turn macro ({len(comparable)} plans, "
        f"min_support={min_support}) : "
        f"{format_turn_series(comparable[[2, 3, 4, 5]].mean())}"
    )

    monotonic_down = comparable[[2, 3, 4, 5]].apply(
        lambda row: row[2] >= row[3] >= row[4] >= row[5], axis=1
    )
    print(
        f"Plans matching T2>=T3>=T4>=T5 : "
        f"{int(monotonic_down.sum())}/{len(comparable)}"
    )

    improved = comparable.sort_values("t5_minus_t2", ascending=False).head(top_k)
    degraded = comparable.sort_values("t5_minus_t2", ascending=True).head(top_k)

    print(f"Top {top_k} plans improving from Turn 2 to Turn 5:")
    for plan, row in improved.iterrows():
        print(
            f"  {plan}: {format_turn_series(row[[2, 3, 4, 5]])}, "
            f"delta={format_pct(row['t5_minus_t2'])}"
        )

    print(f"Top {top_k} plans degrading from Turn 2 to Turn 5:")
    for plan, row in degraded.iterrows():
        print(
            f"  {plan}: {format_turn_series(row[[2, 3, 4, 5]])}, "
            f"delta={format_pct(row['t5_minus_t2'])}"
        )


def print_turn5_bucket_report(work_df, top_k, min_support):
    turn5_df = work_df[work_df["turn"] == 5].copy()
    if turn5_df.empty:
        print("Turn 5 bucket analysis skipped: no Turn 5 rows.")
        return

    pivot_acc = (
        turn5_df.groupby(["gt_plan", "bucket"])["correct"]
        .mean()
        .unstack("bucket")
    )
    pivot_support = turn5_df.groupby(["gt_plan", "bucket"]).size().unstack("bucket")
    bucket_order = ["complex_1", "complex_2", "complex_3", "nonNR"]
    available_buckets = [bucket for bucket in bucket_order if bucket in pivot_acc.columns]
    if len(available_buckets) < 2:
        print("Turn 5 bucket analysis skipped: not enough bucket types.")
        return

    comparable_mask = (
        pivot_acc[available_buckets].notna().all(axis=1)
        & pivot_support[available_buckets].fillna(0).ge(min_support).all(axis=1)
    )
    comparable = pivot_acc[comparable_mask].copy()
    if comparable.empty:
        print(
            f"No Turn 5 plans meet the bucket comparison support threshold "
            f"(min_support={min_support})."
        )
        return

    monotonic_up = comparable[available_buckets].apply(
        lambda row: all(
            row[available_buckets[idx]] <= row[available_buckets[idx + 1]]
            for idx in range(len(available_buckets) - 1)
        ),
        axis=1,
    )
    print(
        f"Turn 5 plans matching {'<='.join(available_buckets)} : "
        f"{int(monotonic_up.sum())}/{len(comparable)}"
    )

    if {"complex_3", "nonNR"}.issubset(comparable.columns):
        comparable["nonNR_minus_complex_3"] = (
            comparable["nonNR"] - comparable["complex_3"]
        )
        worst_non_nr = comparable.sort_values(
            "nonNR_minus_complex_3", ascending=True
        ).head(top_k)
        print(f"Top {top_k} plans where nonNR underperforms complex_3:")
        for plan, row in worst_non_nr.iterrows():
            print(
                f"  {plan}: complex_3={format_pct(row['complex_3'])}, "
                f"nonNR={format_pct(row['nonNR'])}, "
                f"delta={format_pct(row['nonNR_minus_complex_3'])}"
            )

    if {"complex_2", "complex_3"}.issubset(comparable.columns):
        comparable["complex_3_minus_complex_2"] = (
            comparable["complex_3"] - comparable["complex_2"]
        )
        worst_c3 = comparable.sort_values(
            "complex_3_minus_complex_2", ascending=True
        ).head(top_k)
        print(f"Top {top_k} plans where complex_3 underperforms complex_2:")
        for plan, row in worst_c3.iterrows():
            print(
                f"  {plan}: complex_2={format_pct(row['complex_2'])}, "
                f"complex_3={format_pct(row['complex_3'])}, "
                f"delta={format_pct(row['complex_3_minus_complex_2'])}"
            )


def print_analysis_report(df, metric_col, top_k, min_support):
    print("=== Analysis Report ===")


def build_plan_turn_performance_table(work_df):
    grouped = (
        work_df.groupby(["turn", "gt_plan"])["correct"]
        .agg(["mean", "size", "sum"])
        .reset_index()
        .rename(
            columns={
                "gt_plan": "plan",
                "mean": "accuracy",
                "size": "support",
                "sum": "correct_count",
            }
        )
    )
    grouped["incorrect_count"] = grouped["support"] - grouped["correct_count"]
    return grouped[
        ["turn", "plan", "accuracy", "support", "correct_count", "incorrect_count"]
    ].sort_values(["turn", "plan"]).reset_index(drop=True)


def build_plan_turn_file_performance_table(work_df):
    grouped = (
        work_df.groupby(["turn", "file", "bucket", "gt_plan"])["correct"]
        .agg(["mean", "size", "sum"])
        .reset_index()
        .rename(
            columns={
                "gt_plan": "plan",
                "mean": "accuracy",
                "size": "support",
                "sum": "correct_count",
            }
        )
    )
    grouped["incorrect_count"] = grouped["support"] - grouped["correct_count"]
    grouped[["file_bucket_rank", "file_bucket_index", "file_name_key"]] = pd.DataFrame(
        grouped["file"].apply(get_plan_view_file_sort_key).tolist(),
        index=grouped.index,
    )
    ordered = grouped[
        [
            "turn",
            "file",
            "bucket",
            "plan",
            "accuracy",
            "support",
            "correct_count",
            "incorrect_count",
            "file_bucket_rank",
            "file_bucket_index",
            "file_name_key",
        ]
    ].sort_values(
        ["turn", "plan", "file_bucket_rank", "file_bucket_index", "file_name_key"]
    ).reset_index(drop=True)
    return ordered.drop(
        columns=["file_bucket_rank", "file_bucket_index", "file_name_key"]
    )


def export_plan_performance_reports(df, output_dir, metric_col="all"):
    work_df = prepare_analysis_df(df, metric_col=metric_col)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_turn_df = build_plan_turn_performance_table(work_df)
    plan_turn_df.to_csv(
        output_dir / "plan_turn_performance.tsv", sep="\t", index=False
    )

    plan_turn_file_df = build_plan_turn_file_performance_table(work_df)
    plan_turn_file_df.to_csv(
        output_dir / "plan_turn_file_performance.tsv", sep="\t", index=False
    )


def build_human_readable_summary_rows(
    df, metric_col="all", turn_weight_scheme=None, reverse=False
):
    turn_metrics = {}
    file_metric_rows = []

    for turn in [2, 3, 4, 5]:
        turn_df = df[df["turn"] == turn]
        turn_metrics[turn] = {"turn_macro": compute_plan_macro_accuracy(turn_df, metric_col)}

        if "file" not in turn_df.columns:
            continue

        turn_files = sorted(
            turn_df["file"].dropna().unique().tolist(),
            key=get_turn_file_sort_key,
        )
        for file_name in turn_files:
            file_df = turn_df[turn_df["file"] == file_name]
            file_macro = compute_plan_macro_accuracy(file_df, metric_col=metric_col)
            turn_metrics[turn][file_name] = file_macro
            if file_name not in file_metric_rows:
                file_metric_rows.append(file_name)

    weighted_by_turn = {}
    total_weighted_macro = float("nan")
    weighted_column_name = ""
    if turn_weight_scheme:
        weighted_turn_values = compute_weighted_turn_macro_accuracy(
            df,
            turn_weight_scheme,
            metric_col=metric_col,
            reverse=reverse,
        )
        weighted_by_turn = dict(weighted_turn_values)
        weighted_column_name = f"weighted_macro_{turn_weight_scheme}"
        if reverse:
            weighted_column_name = f"{weighted_column_name}_reverse"
        valid_weighted_values = [
            value for value in weighted_by_turn.values() if pd.notnull(value)
        ]
        if valid_weighted_values:
            total_weighted_macro = sum(valid_weighted_values) / len(valid_weighted_values)

    columns = ["turn", "turn_macro"] + file_metric_rows
    if weighted_column_name:
        columns.append(weighted_column_name)

    rows = []
    for turn in [2, 3, 4, 5]:
        if turn not in turn_metrics:
            continue
        row = {"turn": turn, "turn_macro": turn_metrics[turn].get("turn_macro")}
        for file_name in file_metric_rows:
            row[file_name] = turn_metrics[turn].get(file_name)
        if weighted_column_name:
            row[weighted_column_name] = weighted_by_turn.get(turn)
        rows.append(row)

    total_row = {"turn": "TOTAL", "turn_macro": compute_plan_macro_accuracy(df, metric_col)}
    for file_name in file_metric_rows:
        total_row[file_name] = None
    if weighted_column_name:
        total_row[weighted_column_name] = total_weighted_macro
    rows.append(total_row)

    return pd.DataFrame(rows, columns=columns)


def resolve_summary_output_path(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}.macro_summary.tsv")


def export_human_readable_summary(
    df, input_path, metric_col="all", turn_weight_scheme=None, reverse=False
):
    output_path = resolve_summary_output_path(input_path)
    summary_df = build_human_readable_summary_rows(
        df,
        metric_col=metric_col,
        turn_weight_scheme=turn_weight_scheme,
        reverse=reverse,
    )
    summary_df = format_summary_df_for_export(summary_df)
    summary_df.to_csv(output_path, sep="\t", index=False)
    return output_path


def format_pct(value):
    return f"{value * 100:.2f}%"


def format_summary_value(value):
    if pd.isnull(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{value * 100:.2f}"


def format_summary_df_for_export(summary_df):
    formatted_df = summary_df.copy()
    for column in formatted_df.columns:
        if column == "turn":
            continue
        formatted_df[column] = formatted_df[column].apply(format_summary_value)
    return formatted_df


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compute turn-wise and total plan macro accuracy from a result TSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the result TSV file.",
    )
    parser.add_argument(
        "--turn-weight-scheme",
        choices=sorted(TURN_WEIGHT_SCHEMES.keys()),
        help=(
            "Optional turn-level weighting scheme applied to file-level macro scores. "
            "`curriculum` uses Turn3=7:3, Turn4=7:2:1, Turn5=4:3:2:1."
        ),
    )
    parser.add_argument(
        "-reverse",
        "--reverse",
        action="store_true",
        help=(
            "Use the reverse weighting preset. "
            "For `curriculum`, this becomes Turn3=3:7, Turn4=2:2:6, Turn5=1:2:3:4."
        ),
    )
    parser.add_argument(
        "--analysis-report",
        action="store_true",
        help="Print plan-level diagnostics for turn progression and Turn 5 bucket ordering.",
    )
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=10,
        help="Number of plans to show per analysis section. Default: 10.",
    )
    parser.add_argument(
        "--analysis-min-support",
        type=int,
        default=10,
        help=(
            "Minimum row count required in each compared turn/bucket for a plan to "
            "appear in the analysis report. Default: 10."
        ),
    )
    parser.add_argument(
        "--export-plan-performance-dir",
        help=(
            "Optional directory to export plan-level TSVs: "
            "plan_turn_performance.tsv and plan_turn_file_performance.tsv."
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    df = pd.read_csv(args.input, sep="\t", dtype=str)

    if "turn" not in df.columns:
        raise ValueError("The input TSV must contain a `turn` column.")

    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")

    for turn in [2, 3, 4, 5]:
        turn_df = df[df["turn"] == turn]
        turn_macro = compute_plan_macro_accuracy(turn_df, metric_col="all")
        print(f"Turn {turn} Macro Accuracy : {format_pct(turn_macro)}")

        if "file" in turn_df.columns:
            turn_files = sorted(
                turn_df["file"].dropna().unique().tolist(),
                key=get_turn_file_sort_key,
            )
            for file_name in turn_files:
                file_df = turn_df[turn_df["file"] == file_name]
                file_macro = compute_plan_macro_accuracy(file_df, metric_col="all")
                print(f"Turn {turn} {file_name} Macro Accuracy : {format_pct(file_macro)}")

    if args.turn_weight_scheme:
        weighted_turn_values = compute_weighted_turn_macro_accuracy(
            df,
            args.turn_weight_scheme,
            metric_col="all",
            reverse=args.reverse,
        )
        weight_scheme_label = args.turn_weight_scheme
        if args.reverse:
            weight_scheme_label = f"{weight_scheme_label}, reverse"
        for turn, weighted_macro in weighted_turn_values:
            print(
                f"Turn {turn} Weighted Macro Accuracy ({weight_scheme_label}) : "
                f"{format_pct(weighted_macro)}"
            )

        valid_weighted_values = [
            weighted_macro
            for _, weighted_macro in weighted_turn_values
            if pd.notnull(weighted_macro)
        ]
        if valid_weighted_values:
            total_weighted_macro = sum(valid_weighted_values) / len(valid_weighted_values)
            print(
                f"Total Weighted Macro Accuracy ({weight_scheme_label}) : "
                f"{format_pct(total_weighted_macro)}"
            )

    total_macro = compute_plan_macro_accuracy(df, metric_col="all")
    print(f"Total Macro Accuracy : {format_pct(total_macro)}")

    summary_output_path = export_human_readable_summary(
        df,
        input_path=args.input,
        metric_col="all",
        turn_weight_scheme=args.turn_weight_scheme,
        reverse=args.reverse,
    )
    print(f"Summary TSV exported to: {summary_output_path}")

    if args.analysis_report:
        print_analysis_report(
            df,
            metric_col="all",
            top_k=args.analysis_top_k,
            min_support=args.analysis_min_support,
        )

    if args.export_plan_performance_dir:
        output_dir = Path(args.export_plan_performance_dir)
        export_plan_performance_reports(df, output_dir, metric_col="all")
        print(f"Plan performance TSVs exported to: {output_dir}")


if __name__ == "__main__":
    main()
