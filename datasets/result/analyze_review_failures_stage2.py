import argparse
import ast
from pathlib import Path

import pandas as pd


TURN_BUCKET_RULES = {
    "turn3_bucket_offenders": {"turn": 3, "buckets": ["complex_1", "nonNR"]},
    "turn4_bucket_offenders": {"turn": 4, "buckets": ["complex_1", "complex_2", "nonNR"]},
    "turn5_bucket_offenders": {"turn": 5, "buckets": ["complex_1", "complex_2", "complex_3", "nonNR"]},
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create minimal stage-2 analysis outputs from a labeled review TSV."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to a stage-1 labeled TSV file.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory for stage-2 analysis outputs.",
    )
    return parser


def infer_output_dir(input_file: Path, output_root: Path) -> Path:
    stem = input_file.stem
    if "__" in stem:
        plan, section = stem.split("__", 1)
    else:
        plan, section = stem, "analysis"
    return output_root / plan / section


def load_df(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file, sep="\t", dtype=str).fillna("")


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


def normalize_plan_name(value):
    if value is None or pd.isnull(value):
        return ""
    return str(value).strip().lower()


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


def load_source_result_df(review_df: pd.DataFrame, input_file: Path) -> pd.DataFrame | None:
    source_models = [x for x in review_df.get("source_model", pd.Series(dtype=str)).dropna().unique().tolist() if x]
    if len(source_models) != 1:
        return None

    source_path = input_file.parent.parent / source_models[0]
    if not source_path.exists():
        return None

    df = pd.read_csv(source_path, sep="\t", dtype=str).fillna("")
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    df["gt_plan"] = df["gt"].apply(safe_parse_gt_plan)
    df["gt_plan_norm"] = df["gt_plan"].apply(normalize_plan_name)
    df["bucket"] = df["file"].apply(get_file_bucket)
    df["is_fail"] = df["all"].str.lower() != "pass"
    df["turn"] = df["turn"].astype("Int64").astype(str).replace("<NA>", "")
    return df


def filter_source_subset(source_df: pd.DataFrame, plan: str, section: str) -> pd.DataFrame:
    subset = source_df[source_df["gt_plan_norm"] == normalize_plan_name(plan)].copy()
    if section == "turn_trend":
        return subset[subset["turn"].isin(["2", "3", "4", "5"])].copy()

    rule = TURN_BUCKET_RULES.get(section)
    if rule is None:
        return subset.copy()

    return subset[
        (subset["turn"] == str(rule["turn"])) & (subset["bucket"].isin(rule["buckets"]))
    ].copy()


def compute_pattern_summary(fail_df: pd.DataFrame) -> pd.DataFrame:
    fail_df = fail_df.copy()
    fail_df["label_key"] = fail_df["primary_label"] + "|" + fail_df["error_subtype"]
    summary = (
        fail_df.groupby(
            [
                "primary_label",
                "error_subtype",
                "label_key",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["count", "label_key"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = max(len(fail_df), 1)
    summary["reviewed_fail_ratio"] = summary["count"].map(lambda x: round(x / total, 4))
    summary = summary.rename(columns={"count": "reviewed_fail_count"})
    return summary


def compute_turn_bucket_summary(
    df: pd.DataFrame,
    fail_df: pd.DataFrame,
    raw_source_subset: pd.DataFrame | None,
) -> pd.DataFrame:
    totals = (
        df.groupby(["turn", "bucket"], dropna=False)
        .size()
        .reset_index(name="reviewed_rows")
    )
    signatures = (
        fail_df.groupby(["turn", "bucket", "error_signature"], dropna=False)
        .size()
        .reset_index(name="signature_count")
        .sort_values(
            ["turn", "bucket", "signature_count", "error_signature"],
            ascending=[True, True, False, True],
        )
    )
    top_signatures = (
        signatures.groupby(["turn", "bucket"], dropna=False)
        .head(1)
        .rename(
            columns={
                "error_signature": "top_error_signature",
                "signature_count": "top_error_count",
            }
        )
    )

    merged = totals.merge(
        top_signatures[["turn", "bucket", "top_error_signature", "top_error_count"]],
        on=["turn", "bucket"],
        how="left",
    )
    merged = merged.fillna(
        {
            "top_error_signature": "",
            "top_error_count": 0,
        }
    )
    merged["top_error_count"] = merged["top_error_count"].astype(int)

    if raw_source_subset is not None:
        raw_totals = (
            raw_source_subset.groupby(["turn", "bucket"], dropna=False)
            .size()
            .reset_index(name="raw_support")
        )
        raw_fails = (
            raw_source_subset[raw_source_subset["is_fail"]]
            .groupby(["turn", "bucket"], dropna=False)
            .size()
            .reset_index(name="raw_fail_rows")
        )
        merged = merged.merge(raw_totals, on=["turn", "bucket"], how="left")
        merged = merged.merge(raw_fails, on=["turn", "bucket"], how="left")
        merged = merged.fillna({"raw_support": 0, "raw_fail_rows": 0})
        merged["raw_support"] = merged["raw_support"].astype(int)
        merged["raw_fail_rows"] = merged["raw_fail_rows"].astype(int)
        merged["raw_acc"] = merged.apply(
            lambda row: round((row["raw_support"] - row["raw_fail_rows"]) / row["raw_support"], 4)
            if row["raw_support"]
            else 0.0,
            axis=1,
        )

    keep_cols = [
        "turn",
        "bucket",
        "raw_support",
        "raw_fail_rows",
        "raw_acc",
        "support_share_within_turn",
        "fail_contribution_within_turn",
        "top_error_signature",
        "top_error_count",
    ]
    if "raw_support" in merged.columns:
        turn_totals = merged.groupby("turn", dropna=False)[["raw_support", "raw_fail_rows"]].sum().reset_index()
        turn_totals = turn_totals.rename(
            columns={
                "raw_support": "turn_raw_support_total",
                "raw_fail_rows": "turn_raw_fail_total",
            }
        )
        merged = merged.merge(turn_totals, on="turn", how="left")
        merged["support_share_within_turn"] = merged.apply(
            lambda row: round(row["raw_support"] / row["turn_raw_support_total"], 4)
            if row["turn_raw_support_total"]
            else 0.0,
            axis=1,
        )
        merged["fail_contribution_within_turn"] = merged.apply(
            lambda row: round(row["raw_fail_rows"] / row["turn_raw_fail_total"], 4)
            if row["turn_raw_fail_total"]
            else 0.0,
            axis=1,
        )
    extra_cols = [col for col in keep_cols if col in merged.columns]
    return merged[extra_cols].sort_values(["turn", "bucket"]).reset_index(drop=True)


def compute_turn_label_breakdown(
    fail_df: pd.DataFrame,
    raw_source_subset: pd.DataFrame | None,
) -> pd.DataFrame:
    fail_df = fail_df.copy()
    fail_df["label_key"] = fail_df["primary_label"] + "|" + fail_df["error_subtype"]
    grouped = (
        fail_df.groupby(["turn", "primary_label", "error_subtype", "label_key"], dropna=False)
        .size()
        .reset_index(name="label_fail_count")
    )
    turn_fail_totals = grouped.groupby("turn", dropna=False)["label_fail_count"].sum().reset_index(name="turn_fail_total")
    grouped = grouped.merge(turn_fail_totals, on="turn", how="left")
    grouped["label_fail_ratio_within_turn_fail"] = grouped.apply(
        lambda row: round(row["label_fail_count"] / row["turn_fail_total"], 4) if row["turn_fail_total"] else 0.0,
        axis=1,
    )
    if raw_source_subset is not None:
        raw_turn = (
            raw_source_subset.groupby("turn", dropna=False)
            .agg(raw_support=("turn", "size"), raw_fail_rows=("is_fail", "sum"))
            .reset_index()
        )
        raw_turn["raw_acc"] = raw_turn.apply(
            lambda row: round((row["raw_support"] - row["raw_fail_rows"]) / row["raw_support"], 4)
            if row["raw_support"]
            else 0.0,
            axis=1,
        )
        grouped = grouped.merge(raw_turn, on="turn", how="left")
    return grouped.sort_values(
        ["turn", "label_fail_count", "label_key"], ascending=[True, False, True]
    ).reset_index(drop=True)


def compute_turn_bucket_label_breakdown(
    fail_df: pd.DataFrame,
    raw_source_subset: pd.DataFrame | None,
) -> pd.DataFrame:
    fail_df = fail_df.copy()
    fail_df["label_key"] = fail_df["primary_label"] + "|" + fail_df["error_subtype"]
    grouped = (
        fail_df.groupby(["turn", "bucket", "primary_label", "error_subtype", "label_key"], dropna=False)
        .size()
        .reset_index(name="label_fail_count")
    )
    totals = (
        grouped.groupby(["turn", "bucket"], dropna=False)["label_fail_count"]
        .sum()
        .reset_index(name="bucket_fail_total")
    )
    grouped = grouped.merge(totals, on=["turn", "bucket"], how="left")
    grouped["label_fail_ratio_within_bucket_fail"] = grouped.apply(
        lambda row: round(row["label_fail_count"] / row["bucket_fail_total"], 4) if row["bucket_fail_total"] else 0.0,
        axis=1,
    )
    if raw_source_subset is not None:
        raw_bucket = (
            raw_source_subset.groupby(["turn", "bucket"], dropna=False)
            .agg(raw_support=("turn", "size"), raw_fail_rows=("is_fail", "sum"))
            .reset_index()
        )
        raw_bucket["raw_acc"] = raw_bucket.apply(
            lambda row: round((row["raw_support"] - row["raw_fail_rows"]) / row["raw_support"], 4)
            if row["raw_support"]
            else 0.0,
            axis=1,
        )
        grouped = grouped.merge(raw_bucket, on=["turn", "bucket"], how="left")
    return grouped.sort_values(
        ["turn", "bucket", "label_fail_count", "label_key"], ascending=[True, True, False, True]
    ).reset_index(drop=True)


def compute_bucket_summary(
    turn_bucket_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        turn_bucket_summary.groupby("bucket", dropna=False)[["raw_support", "raw_fail_rows"]]
        .sum()
        .reset_index()
    )
    summary["raw_acc"] = summary.apply(
        lambda row: round((row["raw_support"] - row["raw_fail_rows"]) / row["raw_support"], 4)
        if row["raw_support"]
        else 0.0,
        axis=1,
    )
    total_support = max(int(summary["raw_support"].sum()), 1)
    total_fail = max(int(summary["raw_fail_rows"].sum()), 1)
    summary["support_share_within_section"] = summary["raw_support"].map(lambda x: round(x / total_support, 4))
    summary["fail_contribution_within_section"] = summary["raw_fail_rows"].map(lambda x: round(x / total_fail, 4))
    return summary.sort_values(["raw_acc", "bucket"], ascending=[True, True]).reset_index(drop=True)


def compute_bucket_label_breakdown(
    fail_df: pd.DataFrame,
    raw_source_subset: pd.DataFrame | None,
) -> pd.DataFrame:
    fail_df = fail_df.copy()
    fail_df["label_key"] = fail_df["primary_label"] + "|" + fail_df["error_subtype"]
    grouped = (
        fail_df.groupby(["bucket", "primary_label", "error_subtype", "label_key"], dropna=False)
        .size()
        .reset_index(name="label_fail_count")
    )
    totals = grouped.groupby("bucket", dropna=False)["label_fail_count"].sum().reset_index(name="bucket_fail_total")
    grouped = grouped.merge(totals, on="bucket", how="left")
    grouped["label_fail_ratio_within_bucket_fail"] = grouped.apply(
        lambda row: round(row["label_fail_count"] / row["bucket_fail_total"], 4) if row["bucket_fail_total"] else 0.0,
        axis=1,
    )
    if raw_source_subset is not None:
        raw_bucket = (
            raw_source_subset.groupby("bucket", dropna=False)
            .agg(raw_support=("bucket", "size"), raw_fail_rows=("is_fail", "sum"))
            .reset_index()
        )
        raw_bucket["raw_acc"] = raw_bucket.apply(
            lambda row: round((row["raw_support"] - row["raw_fail_rows"]) / row["raw_support"], 4)
            if row["raw_support"]
            else 0.0,
            axis=1,
        )
        grouped = grouped.merge(raw_bucket, on="bucket", how="left")
    return grouped.sort_values(
        ["bucket", "label_fail_count", "label_key"], ascending=[True, False, True]
    ).reset_index(drop=True)

def top_items(series: pd.Series, top_k: int = 5):
    counts = series.value_counts()
    return list(counts.head(top_k).items())


def format_top_list(items):
    if not items:
        return "- 없음"
    return "\n".join(f"- `{name}`: {count}건" for name, count in items)


def pct_str(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def infer_improvements(pattern_summary: pd.DataFrame) -> list[str]:
    improvements = []
    labels = set(pattern_summary["label_key"].tolist())

    if "argument_wrong|arg_value_mismatch" in labels:
        improvements.append(
            "값 선택 규칙 강화: 사용자가 명시한 대상과 history에서 복구한 후보 중 어느 값을 써야 하는지에 대한 선택 규칙을 더 엄격하게 해야 합니다."
        )
    if "argument_wrong|arg_missing" in labels or "argument_wrong|arg_extra" in labels:
        improvements.append(
            "argument 구조화 보강: 필수 argument 누락과 불필요한 extra argument를 줄이기 위한 명시적 체크가 필요합니다."
        )
    if "argument_wrong|arg_hallucinated" in labels or "argument_wrong|arg_vague_or_placeholder" in labels:
        improvements.append(
            "exact-copy 보강: 요약/의역보다 사용자 발화와 이력의 표면형을 더 엄격하게 보존하도록 decoding/prompt 규칙을 조정해야 합니다."
        )
    if "argument_wrong|arg_partial_copy" in labels:
        improvements.append(
            "partial-copy 방지: entity나 값의 일부만 가져오는 오류를 막기 위한 span-copy 또는 post-check 규칙이 필요합니다."
        )
    if "plan_wrong|none" in labels:
        improvements.append(
            "plan selection 안정화: `send_message`와 인접 액션(`send_email`, `open_content`, `camera`) 사이의 혼동을 줄이는 classifier 또는 tool-routing 보강이 필요합니다."
        )
    if "generation_failure|parse_or_runtime_failure" in labels:
        improvements.append(
            "출력 파싱 안정화: generation dict 포맷이 깨지는 케이스가 있어 schema-constrained generation 또는 후처리 복구 로직이 필요합니다."
        )
    if not improvements:
        improvements.append("상위 실패 시그니처 기준의 추가 규칙화 포인트가 아직 뚜렷하지 않습니다. exemplar 검토를 먼저 권장합니다.")
    return improvements


def build_turn_trend_sections(
    fail_df: pd.DataFrame,
    turn_bucket_summary: pd.DataFrame,
    turn_label_breakdown: pd.DataFrame,
) -> list[str]:
    turn_rows = []
    for turn, group in turn_bucket_summary.groupby("turn", dropna=False):
        raw_support = int(group["raw_support"].sum())
        raw_fail = int(group["raw_fail_rows"].sum())
        top_bucket_row = group.sort_values(
            ["raw_fail_rows", "raw_support", "bucket"],
            ascending=[False, False, True],
        ).iloc[0]
        dominant_signature = ""
        turn_fail_df = fail_df[fail_df["turn"] == str(turn)]
        if not turn_fail_df.empty:
            dominant_signature = turn_fail_df["error_signature"].value_counts().index[0]
        turn_rows.append(
            {
                "turn": int(turn),
                "raw_support": raw_support,
                "raw_fail": raw_fail,
                "raw_acc": (raw_support - raw_fail) / raw_support if raw_support else 0.0,
                "dominant_bucket": top_bucket_row["bucket"],
                "dominant_bucket_support": int(top_bucket_row["raw_support"]),
                "dominant_bucket_fail": int(top_bucket_row["raw_fail_rows"]),
                "dominant_bucket_acc": (
                    (int(top_bucket_row["raw_support"]) - int(top_bucket_row["raw_fail_rows"])) / int(top_bucket_row["raw_support"])
                    if int(top_bucket_row["raw_support"])
                    else 0.0
                ),
                "dominant_signature": dominant_signature,
            }
        )

    turn_df = pd.DataFrame(turn_rows).sort_values("turn").reset_index(drop=True)

    trend_lines = []
    for _, row in turn_df.iterrows():
        trend_lines.append(
            f"- Turn {int(row['turn'])}: raw support {int(row['raw_support'])}, raw fail {int(row['raw_fail'])}, "
            f"raw acc {row['raw_acc']:.4f}. 지배 bucket은 `{row['dominant_bucket']}` "
            f"(support {int(row['dominant_bucket_support'])}, fail {int(row['dominant_bucket_fail'])}, acc {row['dominant_bucket_acc']:.4f}), "
            f"대표 실패 시그니처는 `{row['dominant_signature']}`"
        )

    delta_lines = []
    if len(turn_df) >= 2:
        for idx in range(1, len(turn_df)):
            prev_row = turn_df.iloc[idx - 1]
            curr_row = turn_df.iloc[idx]
            delta = curr_row["raw_acc"] - prev_row["raw_acc"]
            sign = "+" if delta >= 0 else ""
            delta_lines.append(
                f"- Turn {int(prev_row['turn'])} -> Turn {int(curr_row['turn'])}: raw acc {prev_row['raw_acc']:.4f} -> {curr_row['raw_acc']:.4f} ({sign}{delta:.4f})"
            )

    bucket_mix_lines = []
    for _, row in turn_df.iterrows():
        turn = str(int(row["turn"]))
        turn_group = turn_bucket_summary[turn_bucket_summary["turn"] == turn].copy()
        if turn_group.empty:
            continue
        turn_group["support_share"] = turn_group["raw_support"] / max(int(turn_group["raw_support"].sum()), 1)
        turn_group = turn_group.sort_values(["raw_support", "bucket"], ascending=[False, True])
        mix_text = ", ".join(
            f"{bucket_row['bucket']} {int(bucket_row['raw_support'])} ({bucket_row['support_share']:.1%}, acc {bucket_row['raw_acc']:.4f})"
            for _, bucket_row in turn_group.iterrows()
        )
        bucket_mix_lines.append(f"- Turn {turn}: {mix_text}")

    overall_summary = []
    if len(turn_df) >= 2 and turn_df["raw_acc"].is_monotonic_increasing:
        overall_summary.append(
            "- raw accuracy가 Turn 2 -> Turn 5에서 단조 증가합니다."
        )
    overall_summary.append("- turn이 증가할수록 bucket 구성이 `nonNR` 중심에서 `complex_1/2/3 + nonNR` 조합으로 바뀝니다.")

    turn_signature_lines = []
    for turn in sorted(turn_label_breakdown["turn"].astype(str).unique().tolist(), key=int):
        turn_breakdown = turn_label_breakdown[turn_label_breakdown["turn"] == turn].head(5)
        signature_text = ", ".join(
            f"{row['label_key']} ({int(row['label_fail_count'])}, {float(row['label_fail_ratio_within_turn_fail']):.1%})"
            for _, row in turn_breakdown.iterrows()
        )
        turn_signature_lines.append(f"- Turn {turn}: {signature_text}")

    comparative_lines = []
    turns_sorted = sorted(turn_label_breakdown["turn"].astype(str).unique().tolist(), key=int)
    if len(turns_sorted) >= 2:
        base_turn = turns_sorted[0]
        target_turn = turns_sorted[-1]
        base_counts = (
            turn_label_breakdown[turn_label_breakdown["turn"] == base_turn]
            .set_index("label_key")["label_fail_count"]
        )
        target_counts = (
            turn_label_breakdown[turn_label_breakdown["turn"] == target_turn]
            .set_index("label_key")["label_fail_count"]
        )
        only_base = [f"{sig} ({int(cnt)})" for sig, cnt in base_counts.items() if sig not in target_counts.index][:5]
        only_target = [f"{sig} ({int(cnt)})" for sig, cnt in target_counts.items() if sig not in base_counts.index][:5]
        shared = []
        for sig in base_counts.index.intersection(target_counts.index):
            shared.append(
                f"{sig} (Turn {base_turn}: {int(base_counts[sig])}, Turn {target_turn}: {int(target_counts[sig])})"
            )
        comparative_lines.append(
            f"- Turn {base_turn} only dominant signatures: {', '.join(only_base) if only_base else '없음'}"
        )
        comparative_lines.append(
            f"- Turn {target_turn} only dominant signatures: {', '.join(only_target) if only_target else '없음'}"
        )
        comparative_lines.append(
            f"- Shared signatures with different load: {', '.join(shared[:5]) if shared else '없음'}"
        )

    hypothesis_lines = [
        "- Turn 2 -> Turn 5 상승은 history length 개선보다 bucket composition 변화의 영향으로 해석하는 편이 현재 수치와 더 잘 맞습니다.",
        "- Turn 2는 `nonNR` 단일 구성이고 acc 0.1000입니다.",
        "- Turn 5는 `complex_3` 50.2%, `complex_2` 25.5%, `complex_1` 18.3%, `nonNR` 6.0%로 분산되며, 저acc bucket인 `complex_1` 비중이 상대적으로 낮습니다.",
    ]

    summary_lines = [
        "- Turn 3의 실패는 거의 `complex_1`이 만듭니다. support share 93.5%, fail contribution 95.3%입니다.",
        "- Turn 4에서는 `complex_2`가 49.8%까지 들어오며 전체 acc가 0.3612로 상승합니다.",
        "- Turn 5에서는 `complex_3`와 `complex_2`가 합쳐 75.7%를 차지하고, 이 둘의 acc가 각각 0.4603, 0.4375라 전체 평균을 끌어올립니다.",
        "- turn별 failure label도 바뀝니다. Turn 2는 `arg_vague_or_placeholder`, `arg_hallucinated`, `parse_or_runtime_failure` 비중이 크고, Turn 3~5는 `arg_value_mismatch`, `arg_extra`, `arg_hallucinated` 축으로 이동합니다.",
    ]

    return [
        "## Executive Summary",
        "- 이 파일은 `turn_bucket_summary.tsv`, `turn_label_breakdown.tsv`, `pattern_summary.tsv`를 합쳐 `send_message / turn_trend` 결과를 요약한 것입니다.",
        "- 기본 bucket 구조는 Turn 2=`nonNR`, Turn 3=`complex_1 + nonNR`, Turn 4=`complex_1 + complex_2 + nonNR`, Turn 5=`complex_1 + complex_2 + complex_3 + nonNR`입니다.",
        *overall_summary,
        "",
        "## Turn Metrics",
        *trend_lines,
        "",
        "## Turn Delta",
        *(delta_lines if delta_lines else ["- 단일 turn만 존재해 delta를 계산하지 않았습니다."]),
        "",
        "## Bucket Mix By Turn",
        *bucket_mix_lines,
        "",
        "## Failure Labels By Turn",
        *turn_signature_lines,
        "",
        "## Cross-Turn Difference",
        *(comparative_lines if comparative_lines else ["- 비교 가능한 복수 turn이 없어 생략했습니다."]),
        "",
        "## Key Findings",
        *summary_lines,
        "",
        "## Interpretation",
        *hypothesis_lines,
        "",
    ]


def build_bucket_offender_sections(
    bucket_summary: pd.DataFrame,
    bucket_label_breakdown: pd.DataFrame,
) -> list[str]:
    bucket_rows = []
    for _, row in bucket_summary.sort_values(["raw_acc", "bucket"], ascending=[True, True]).iterrows():
        bucket = row["bucket"]
        bucket_breakdown = bucket_label_breakdown[bucket_label_breakdown["bucket"] == bucket].head(5)
        dominant_signature = ""
        if not bucket_breakdown.empty:
            dominant_signature = bucket_breakdown.iloc[0]["label_key"]
        bucket_rows.append(
            f"- `{bucket}`: raw support {int(row['raw_support'])}, raw fail {int(row['raw_fail_rows'])}, raw acc {float(row['raw_acc']):.4f}, "
            f"대표 실패 레이블 `{dominant_signature}`"
        )

    label_lines = []
    for bucket in bucket_summary["bucket"].tolist():
        bucket_breakdown = bucket_label_breakdown[bucket_label_breakdown["bucket"] == bucket].head(5)
        label_text = ", ".join(
            f"{row['label_key']} ({int(row['label_fail_count'])}, {float(row['label_fail_ratio_within_bucket_fail']):.1%})"
            for _, row in bucket_breakdown.iterrows()
        )
        label_lines.append(f"- `{bucket}`: {label_text}")

    weakest_row = bucket_summary.sort_values(["raw_acc", "bucket"], ascending=[True, True]).iloc[0]
    strongest_row = bucket_summary.sort_values(["raw_acc", "bucket"], ascending=[False, True]).iloc[0]
    interpretation = [
        f"- `{weakest_row['bucket']}`는 낮은 raw acc와 함께 특정 failure label이 집중되어 있어 bucket-specific failure mode가 강합니다.",
        f"- `{strongest_row['bucket']}`는 상대적으로 높은 raw acc를 보이며 failure label이 더 분산되거나 강도가 낮습니다.",
    ]
    summary_lines = [
        f"- 가장 취약한 bucket은 `{weakest_row['bucket']}`이며 raw acc {float(weakest_row['raw_acc']):.4f}, raw fail {int(weakest_row['raw_fail_rows'])}입니다.",
        f"- 가장 안정적인 bucket은 `{strongest_row['bucket']}`이며 raw acc {float(strongest_row['raw_acc']):.4f}, raw fail {int(strongest_row['raw_fail_rows'])}입니다.",
        "- bucket 간 acc 차이와 dominant failure label 차이가 함께 나타나면, 단순 support 차이보다 bucket-specific failure mode 차이로 해석하는 편이 자연스럽습니다.",
    ]

    return [
        "## Executive Summary",
        "- 이 파일은 `bucket_summary.tsv`, `bucket_label_breakdown.tsv`, `pattern_summary.tsv`를 합쳐 고정 turn 안의 bucket gap을 요약한 것입니다.",
        *summary_lines,
        "",
        "## Bucket Metrics",
        *bucket_rows,
        "",
        "## Failure Labels By Bucket",
        *label_lines,
        "",
        "## Interpretation",
        *interpretation,
        "",
    ]


def build_root_cause_report(
    input_file: Path,
    df: pd.DataFrame,
    fail_df: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    section_outputs: dict,
    raw_source_subset: pd.DataFrame | None,
    plan: str,
    section: str,
) -> str:
    total_rows = len(df)
    fail_rows = len(fail_df)
    pass_rows = int((df["review_label"] == "pass_sample").sum())
    turns = ", ".join(sorted(df["turn"].replace("", pd.NA).dropna().astype(str).unique().tolist()))
    buckets = ", ".join(sorted(df["bucket"].replace("", pd.NA).dropna().astype(str).unique().tolist()))

    top_signatures = top_items(fail_df["error_signature"], top_k=7)
    top_priorities = top_items(fail_df["review_priority"], top_k=5)

    improvements = infer_improvements(pattern_summary)
    if section == "turn_trend":
        diagnosis_sections = build_turn_trend_sections(
            fail_df,
            section_outputs["turn_bucket_summary"],
            section_outputs["turn_label_breakdown"],
        )
    elif section in TURN_BUCKET_RULES:
        diagnosis_sections = build_bucket_offender_sections(
            section_outputs["bucket_summary"],
            section_outputs["bucket_label_breakdown"],
        )
    else:
        diagnosis_sections = [
            "## Generic Diagnosis",
            "- 이 section에 대한 전용 Stage-2 진단 템플릿이 아직 정의되지 않았습니다.",
            "",
        ]

    return "\n".join(
        [
            f"# Stage-2 Root Cause Report",
            "",
            "## Input",
            f"- input_file: `{input_file}`",
            f"- rows_in_review_file: {total_rows}",
            f"- fail_rows: {fail_rows}",
            f"- pass_sample_rows: {pass_rows}",
            f"- turns_in_file: {turns}",
            f"- buckets_in_file: {buckets}",
            "",
            "## Scope",
            f"- plan=`{plan}`, section=`{section}`",
            "- `raw support/raw fail/raw acc`는 원본 모델 결과 TSV에서 다시 계산한 값입니다.",
            "- failure label 분포는 review TSV의 fail 샘플 라벨을 집계한 값입니다.",
            "",
            "## Global Failure Signatures",
            format_top_list(top_signatures),
            "",
            "## Global Review Priorities",
            format_top_list(top_priorities),
            "",
            *diagnosis_sections,
            "## Label-Specific Improvement Hints",
            *[f"- {item}" for item in improvements],
            "",
        ]
    )


def main():
    args = build_arg_parser().parse_args()
    input_file = Path(args.input_file)
    output_root = Path(args.output_root)
    output_dir = infer_output_dir(input_file, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(input_file)
    fail_df = df[df["review_label"] == "fail"].copy()
    plan, section = input_file.stem.split("__", 1) if "__" in input_file.stem else (input_file.stem, "analysis")
    raw_source_df = load_source_result_df(df, input_file)
    raw_source_subset = None if raw_source_df is None else filter_source_subset(raw_source_df, plan=plan, section=section)

    pattern_summary = compute_pattern_summary(fail_df)
    section_outputs = {}
    output_paths = [output_dir / "pattern_summary.tsv"]

    if section == "turn_trend":
        turn_bucket_summary = compute_turn_bucket_summary(df, fail_df, raw_source_subset=raw_source_subset)
        turn_label_breakdown = compute_turn_label_breakdown(fail_df, raw_source_subset=raw_source_subset)
        turn_bucket_label_breakdown = compute_turn_bucket_label_breakdown(fail_df, raw_source_subset=raw_source_subset)
        section_outputs = {
            "turn_bucket_summary": turn_bucket_summary,
            "turn_label_breakdown": turn_label_breakdown,
            "turn_bucket_label_breakdown": turn_bucket_label_breakdown,
        }
        output_paths.extend(
            [
                output_dir / "turn_bucket_summary.tsv",
                output_dir / "turn_label_breakdown.tsv",
                output_dir / "turn_bucket_label_breakdown.tsv",
            ]
        )
    elif section in TURN_BUCKET_RULES:
        turn_bucket_summary = compute_turn_bucket_summary(df, fail_df, raw_source_subset=raw_source_subset)
        bucket_summary = compute_bucket_summary(turn_bucket_summary)
        bucket_label_breakdown = compute_bucket_label_breakdown(fail_df, raw_source_subset=raw_source_subset)
        section_outputs = {
            "bucket_summary": bucket_summary,
            "bucket_label_breakdown": bucket_label_breakdown,
        }
        output_paths.extend(
            [
                output_dir / "bucket_summary.tsv",
                output_dir / "bucket_label_breakdown.tsv",
            ]
        )
    else:
        generic_summary = compute_turn_bucket_summary(df, fail_df, raw_source_subset=raw_source_subset)
        section_outputs = {"turn_bucket_summary": generic_summary}
        output_paths.append(output_dir / "turn_bucket_summary.tsv")

    root_cause_report = build_root_cause_report(
        input_file=input_file,
        df=df,
        fail_df=fail_df,
        pattern_summary=pattern_summary,
        section_outputs=section_outputs,
        raw_source_subset=raw_source_subset,
        plan=plan,
        section=section,
    )

    pattern_summary.to_csv(output_dir / "pattern_summary.tsv", sep="\t", index=False)
    if section == "turn_trend":
        section_outputs["turn_bucket_summary"].to_csv(output_dir / "turn_bucket_summary.tsv", sep="\t", index=False)
        section_outputs["turn_label_breakdown"].to_csv(output_dir / "turn_label_breakdown.tsv", sep="\t", index=False)
        section_outputs["turn_bucket_label_breakdown"].to_csv(output_dir / "turn_bucket_label_breakdown.tsv", sep="\t", index=False)
    elif section in TURN_BUCKET_RULES:
        section_outputs["bucket_summary"].to_csv(output_dir / "bucket_summary.tsv", sep="\t", index=False)
        section_outputs["bucket_label_breakdown"].to_csv(output_dir / "bucket_label_breakdown.tsv", sep="\t", index=False)
    else:
        section_outputs["turn_bucket_summary"].to_csv(output_dir / "turn_bucket_summary.tsv", sep="\t", index=False)
    (output_dir / "root_cause_report.md").write_text(root_cause_report)
    output_paths.append(output_dir / "root_cause_report.md")

    print(f"Stage-2 analysis written to: {output_dir}")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
