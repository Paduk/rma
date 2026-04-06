import argparse
import ast
from pathlib import Path

import pandas as pd


TURN_BUCKET_RULES = {
    3: ["complex_1", "nonNR"],
    4: ["complex_1", "complex_2", "nonNR"],
    5: ["complex_1", "complex_2", "complex_3", "nonNR"],
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


def format_pct(value):
    return f"{value * 100:.2f}%"


def load_result_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "turn" not in df.columns:
        raise ValueError(f"{path} is missing required column `turn`.")

    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    df["gt_plan"] = df["gt"].apply(safe_parse_gt_plan)
    df["correct"] = df["all"].str.lower() == "pass"
    df["bucket"] = df["file"].apply(get_file_bucket)
    return df.dropna(subset=["gt_plan"]).copy()


def find_turn_trend_offenders(df, min_support):
    acc = df.groupby(["gt_plan", "turn"])["correct"].mean().unstack("turn")
    support = df.groupby(["gt_plan", "turn"]).size().unstack("turn")
    required_turns = [2, 3, 4, 5]

    mask = (
        acc[required_turns].notna().all(axis=1)
        & support[required_turns].fillna(0).ge(min_support).all(axis=1)
    )
    comparable = acc[mask].copy()
    comparable_support = support[mask].copy()

    offenders = []
    for plan, row in comparable.iterrows():
        is_monotonic = row[2] >= row[3] >= row[4] >= row[5]
        if is_monotonic:
            continue
        offenders.append(
            {
                "plan": plan,
                "series": row[[2, 3, 4, 5]].to_dict(),
                "support": comparable_support.loc[plan, [2, 3, 4, 5]].to_dict(),
                "t5_minus_t2": row[5] - row[2],
            }
        )

    offenders.sort(key=lambda item: item["t5_minus_t2"], reverse=True)
    return offenders, len(comparable)


def find_bucket_offenders(df, turn, min_support):
    expected = TURN_BUCKET_RULES[turn]
    turn_df = df[df["turn"] == turn]
    acc = turn_df.groupby(["gt_plan", "bucket"])["correct"].mean().unstack("bucket")
    support = turn_df.groupby(["gt_plan", "bucket"]).size().unstack("bucket")

    mask = (
        acc[expected].notna().all(axis=1)
        & support[expected].fillna(0).ge(min_support).all(axis=1)
    )
    comparable = acc[mask].copy()
    comparable_support = support[mask].copy()

    offenders = []
    for plan, row in comparable.iterrows():
        is_monotonic = all(
            row[expected[idx]] <= row[expected[idx + 1]]
            for idx in range(len(expected) - 1)
        )
        if is_monotonic:
            continue
        pair_deltas = {}
        for idx in range(len(expected) - 1):
            left = expected[idx]
            right = expected[idx + 1]
            pair_deltas[f"{right}_minus_{left}"] = row[right] - row[left]
        offenders.append(
            {
                "plan": plan,
                "series": row[expected].to_dict(),
                "support": comparable_support.loc[plan, expected].to_dict(),
                "worst_delta": min(pair_deltas.values()),
                "pair_deltas": pair_deltas,
            }
        )

    offenders.sort(key=lambda item: item["worst_delta"])
    return offenders, len(comparable)


def summarize_common_offenders(model_reports, section_key, top_k):
    rows = {}
    for model_name, report in model_reports.items():
        for offender in report[section_key]:
            plan = offender["plan"]
            entry = rows.setdefault(
                plan,
                {
                    "models": [],
                    "score_sum": 0.0,
                },
            )
            entry["models"].append(model_name)
            if section_key == "turn_offenders":
                entry["score_sum"] += offender["t5_minus_t2"]
            else:
                entry["score_sum"] += -offender["worst_delta"]

    ranked = sorted(
        rows.items(),
        key=lambda item: (len(item[1]["models"]), item[1]["score_sum"]),
        reverse=True,
    )
    return ranked[:top_k]


def build_common_offender_table(model_reports, section_key):
    rows = []
    for plan, row in summarize_common_offenders(
        model_reports, section_key, top_k=10**9
    ):
        rows.append(
            {
                "plan": plan,
                "model_count": len(row["models"]),
                "models": ",".join(row["models"]),
                "aggregate_score": row["score_sum"],
                "section": section_key,
            }
        )
    return pd.DataFrame(rows)


def build_turn_detail_table(model_reports):
    rows = []
    for model_name, report in model_reports.items():
        for offender in report["turn_offenders"]:
            for turn, accuracy in offender["series"].items():
                rows.append(
                    {
                        "section": "turn_trend",
                        "model": model_name,
                        "plan": offender["plan"],
                        "axis": f"turn_{turn}",
                        "accuracy": accuracy,
                        "support": int(offender["support"][turn]),
                        "summary_score": offender["t5_minus_t2"],
                    }
                )
    return pd.DataFrame(rows)


def build_bucket_detail_table(model_reports, turn):
    rows = []
    section_key = f"turn{turn}_bucket_offenders"
    for model_name, report in model_reports.items():
        for offender in report[section_key]:
            for bucket, accuracy in offender["series"].items():
                rows.append(
                    {
                        "section": section_key,
                        "model": model_name,
                        "plan": offender["plan"],
                        "axis": bucket,
                        "accuracy": accuracy,
                        "support": int(offender["support"][bucket]),
                        "summary_score": offender["worst_delta"],
                    }
                )
    return pd.DataFrame(rows)


def build_full_turn_bucket_metrics_table(model_dfs):
    rows = []
    for model_name, df in model_dfs.items():
        turn_acc = (
            df.groupby(["gt_plan", "turn"])["correct"]
            .mean()
            .rename("accuracy")
            .reset_index()
        )
        turn_support = (
            df.groupby(["gt_plan", "turn"])
            .size()
            .rename("support")
            .reset_index()
        )
        turn_metrics = turn_acc.merge(turn_support, on=["gt_plan", "turn"], how="inner")
        for row in turn_metrics.itertuples(index=False):
            rows.append(
                {
                    "section": "turn_all",
                    "model": model_name,
                    "plan": row.gt_plan,
                    "turn": int(row.turn),
                    "bucket": "all",
                    "accuracy": row.accuracy,
                    "support": int(row.support),
                }
            )

        bucket_df = df.dropna(subset=["bucket"]).copy()
        bucket_acc = (
            bucket_df.groupby(["gt_plan", "turn", "bucket"])["correct"]
            .mean()
            .rename("accuracy")
            .reset_index()
        )
        bucket_support = (
            bucket_df.groupby(["gt_plan", "turn", "bucket"])
            .size()
            .rename("support")
            .reset_index()
        )
        bucket_metrics = bucket_acc.merge(
            bucket_support, on=["gt_plan", "turn", "bucket"], how="inner"
        )
        for row in bucket_metrics.itertuples(index=False):
            rows.append(
                {
                    "section": "turn_bucket",
                    "model": model_name,
                    "plan": row.gt_plan,
                    "turn": int(row.turn),
                    "bucket": row.bucket,
                    "accuracy": row.accuracy,
                    "support": int(row.support),
                }
            )

    full_df = pd.DataFrame(rows)
    if full_df.empty:
        return full_df
    return full_df.sort_values(
        by=["model", "section", "plan", "turn", "bucket"],
        kind="stable",
    ).reset_index(drop=True)


def export_csv_reports(model_reports, model_dfs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    common_tables = [
        build_common_offender_table(model_reports, "turn_offenders"),
        build_common_offender_table(model_reports, "turn3_bucket_offenders"),
        build_common_offender_table(model_reports, "turn4_bucket_offenders"),
        build_common_offender_table(model_reports, "turn5_bucket_offenders"),
    ]
    common_df = pd.concat(common_tables, ignore_index=True)
    common_df.to_csv(output_dir / "common_offenders.tsv", sep="\t", index=False)

    detail_tables = [
        build_turn_detail_table(model_reports),
        build_bucket_detail_table(model_reports, 3),
        build_bucket_detail_table(model_reports, 4),
        build_bucket_detail_table(model_reports, 5),
    ]
    detail_df = pd.concat(detail_tables, ignore_index=True)
    detail_df.to_csv(output_dir / "offender_details.tsv", sep="\t", index=False)

    full_metrics_df = build_full_turn_bucket_metrics_table(model_dfs)
    full_metrics_df.to_csv(
        output_dir / "full_turn_bucket_metrics.tsv", sep="\t", index=False
    )


def print_series(series):
    return ", ".join(f"{key}={format_pct(value)}" for key, value in series.items())


def print_support(support):
    return ", ".join(f"{key}={int(value)}" for key, value in support.items())


def print_model_report(model_name, report, top_k):
    print(f"=== {model_name} ===")
    print(
        f"Turn-trend offenders: {len(report['turn_offenders'])}/"
        f"{report['turn_comparable_count']} comparable plans"
    )
    for offender in report["turn_offenders"][:top_k]:
        print(
            f"  {offender['plan']}: {print_series(offender['series'])}, "
            f"delta(T5-T2)={format_pct(offender['t5_minus_t2'])}, "
            f"support[{print_support(offender['support'])}]"
        )

    for turn in [3, 4, 5]:
        offenders = report[f"turn{turn}_bucket_offenders"]
        comparable_count = report[f"turn{turn}_bucket_comparable_count"]
        print(
            f"Turn {turn} bucket-order offenders: {len(offenders)}/"
            f"{comparable_count} comparable plans"
        )
        for offender in offenders[:top_k]:
            print(
                f"  {offender['plan']}: {print_series(offender['series'])}, "
                f"worst_delta={format_pct(offender['worst_delta'])}, "
                f"support[{print_support(offender['support'])}]"
            )
    print()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Identify plan-level offenders that break expected turn trends "
            "or turn-specific bucket ordering."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more result TSV paths.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum row count required in each compared turn/bucket. Default: 2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of offenders to show per section. Default: 10.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional directory to export TSV files: common_offenders.tsv, "
            "offender_details.tsv, and full_turn_bucket_metrics.tsv."
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    model_reports = {}
    model_dfs = {}
    for input_path in args.inputs:
        path = Path(input_path)
        model_name = path.name
        df = load_result_tsv(path)
        model_dfs[model_name] = df

        turn_offenders, turn_comparable_count = find_turn_trend_offenders(
            df, args.min_support
        )
        turn3_offenders, turn3_comparable_count = find_bucket_offenders(
            df, 3, args.min_support
        )
        turn4_offenders, turn4_comparable_count = find_bucket_offenders(
            df, 4, args.min_support
        )
        turn5_offenders, turn5_comparable_count = find_bucket_offenders(
            df, 5, args.min_support
        )

        model_reports[model_name] = {
            "turn_offenders": turn_offenders,
            "turn_comparable_count": turn_comparable_count,
            "turn3_bucket_offenders": turn3_offenders,
            "turn3_bucket_comparable_count": turn3_comparable_count,
            "turn4_bucket_offenders": turn4_offenders,
            "turn4_bucket_comparable_count": turn4_comparable_count,
            "turn5_bucket_offenders": turn5_offenders,
            "turn5_bucket_comparable_count": turn5_comparable_count,
        }

    for model_name, report in model_reports.items():
        print_model_report(model_name, report, args.top_k)

    print("=== Common Offenders ===")
    common_turn = summarize_common_offenders(model_reports, "turn_offenders", args.top_k)
    print("Turn-trend offenders shared across models:")
    for plan, row in common_turn:
        print(f"  {plan}: {len(row['models'])} models [{', '.join(row['models'])}]")

    for turn in [3, 4, 5]:
        section_key = f"turn{turn}_bucket_offenders"
        common_bucket = summarize_common_offenders(model_reports, section_key, args.top_k)
        print(f"Turn {turn} bucket-order offenders shared across models:")
        for plan, row in common_bucket:
            print(f"  {plan}: {len(row['models'])} models [{', '.join(row['models'])}]")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        export_csv_reports(model_reports, model_dfs, output_dir)
        print(f"\nTSV exported to: {output_dir}")


if __name__ == "__main__":
    main()
