import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TOOLS_PATH = Path(__file__).resolve().parent / "apis" / "simple_api.json"
NONE_PLANS = {"none", "null", ""}


def get_arg_parse():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate JSON/tool/argument schema compliance for one baseline TSV output."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Inference result TSV to evaluate.")
    input_group.add_argument("--no_schema_tsv", dest="input", help=argparse.SUPPRESS)
    input_group.add_argument("--schema_tsv", dest="input", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tools",
        default=str(DEFAULT_TOOLS_PATH),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--out", default=None, help="Output TSV for the grouped compliance summary.")
    parser.add_argument("--summary_out", dest="out", help=argparse.SUPPRESS)
    parser.add_argument("--detail_out", default=None, help="Optional debug TSV for row-level diagnostics.")
    parser.add_argument("--comparison_out", dest="out", help=argparse.SUPPRESS)
    none_group = parser.add_mutually_exclusive_group()
    none_group.add_argument(
        "--allow_none_plan",
        action="store_true",
        help="Treat plan=None/null/empty as valid no-op. Default treats it as invalid.",
    )
    none_group.add_argument(
        "--disallow_none_plan",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if is_blank(value):
        return None

    text = str(value).strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(text)
        except Exception:
            pass

    if "```" in text:
        text = text.replace("```json", "```")
        parts = text.split("```")
        if len(parts) >= 3:
            return parse_jsonish(parts[1].strip())

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        for parser in (ast.literal_eval, json.loads):
            try:
                return parser(candidate)
            except Exception:
                pass

    return None


def normalize_plan(plan: Any) -> str:
    if plan is None:
        return ""
    return str(plan).strip()


def is_none_plan(plan: Any) -> bool:
    return normalize_plan(plan).lower() in NONE_PLANS


def read_tools(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as file:
        tools = json.load(file)
    if not isinstance(tools, dict):
        raise ValueError(f"Tool schema must be a dict: {path}")
    return {str(plan): list(args) for plan, args in tools.items()}


def load_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def get_prediction(row: pd.Series) -> tuple[dict, bool, str]:
    for column in ("generation", "raw_generation"):
        if column not in row.index:
            continue
        parsed = parse_jsonish(row[column])
        if isinstance(parsed, dict) and "error" not in parsed:
            return parsed, True, column
    return {}, False, ""


def evaluate_row(
    row: pd.Series,
    *,
    tools: dict[str, list[str]],
    allow_none_plan: bool,
) -> dict[str, Any]:
    prediction, generation_parseable, parsed_from = get_prediction(row)
    parse_error_empty = "parse_error" not in row.index or is_blank(row["parse_error"])
    parse_ok = bool(parse_error_empty or generation_parseable)

    plan = normalize_plan(prediction.get("plan"))
    arguments = prediction.get("arguments")
    arguments_is_object = isinstance(arguments, dict)
    argument_keys = list(arguments.keys()) if arguments_is_object else []

    plan_is_none = is_none_plan(plan)
    none_ok = bool(allow_none_plan and plan_is_none)
    plan_allowed = none_ok or plan in tools

    if plan_is_none:
        allowed_argument_keys = []
    else:
        allowed_argument_keys = tools.get(plan, [])

    invalid_argument_keys = [key for key in argument_keys if key not in allowed_argument_keys]
    argument_keys_subset = arguments_is_object and not invalid_argument_keys
    schema_compliance = bool(parse_ok and generation_parseable and plan_allowed and argument_keys_subset)

    return {
        "parse_error_empty": parse_error_empty,
        "generation_parseable": generation_parseable,
        "parsed_from": parsed_from,
        "parse_ok": parse_ok,
        "predicted_plan": plan,
        "predicted_arguments_keys": json.dumps(argument_keys, ensure_ascii=False),
        "arguments_is_object": arguments_is_object,
        "plan_allowed": plan_allowed,
        "plan_allowed_scope": "simple_api",
        "allowed_argument_keys": json.dumps(allowed_argument_keys, ensure_ascii=False),
        "invalid_argument_keys": json.dumps(invalid_argument_keys, ensure_ascii=False),
        "argument_keys_subset": argument_keys_subset,
        "schema_compliance": schema_compliance,
    }


def build_detail(
    path: str | Path,
    *,
    label: str,
    mode: str,
    tools: dict[str, list[str]],
    allow_none_plan: bool,
) -> pd.DataFrame:
    df = load_tsv(path)
    rows = []
    source_tsv = str(path)

    for row_idx, row in df.iterrows():
        evaluated = evaluate_row(row, tools=tools, allow_none_plan=allow_none_plan)
        base = {
            "label": label,
            "mode": mode,
            "source_tsv": source_tsv,
            "row_idx": row_idx,
            "test_key": row.get("test_key"),
            "file": row.get("file"),
            "turn": row.get("turn"),
            "existing_plan": row.get("plan"),
            "existing_arguments": row.get("arguments"),
            "existing_all": row.get("all"),
            "parse_error": row.get("parse_error"),
        }
        base.update(evaluated)
        rows.append(base)

    return pd.DataFrame(rows)


def pass_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return series.eq("pass").mean() * 100


def bool_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return series.fillna(False).astype(bool).mean() * 100


def summarize_one_group(df: pd.DataFrame, group_cols: list[str], summary_scope: str) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        parse_ok_pct = bool_rate(sub["parse_ok"])
        plan_allowed_pct = bool_rate(sub["plan_allowed"])
        argument_keys_subset_pct = bool_rate(sub["argument_keys_subset"])
        schema_compliance_pct = bool_rate(sub["schema_compliance"])
        row = {column: value for column, value in zip(group_cols, key)}
        row.update(
            {
                "summary_scope": summary_scope,
                "samples": len(sub),
                "parse_ok_pct": parse_ok_pct,
                "plan_allowed_pct": plan_allowed_pct,
                "argument_keys_subset_pct": argument_keys_subset_pct,
                "schema_compliance_pct": schema_compliance_pct,
                "existing_all_accuracy_pct": pass_rate(sub["existing_all"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    common = ["label", "mode", "source_tsv"]
    frames = [
        summarize_one_group(detail, common, "overall"),
        summarize_one_group(detail, common + ["test_key"], "test_key"),
        summarize_one_group(detail, common + ["test_key", "file", "turn"], "file"),
    ]
    summary = pd.concat(frames, ignore_index=True, sort=False)

    metric_cols = [column for column in summary.columns if column.endswith("_pct")]
    summary[metric_cols] = summary[metric_cols].round(2)

    scope_order = {"overall": 0, "test_key": 1, "file": 2}
    summary["_scope_order"] = summary["summary_scope"].map(scope_order).fillna(99)
    sort_cols = [
        column
        for column in ("_scope_order", "test_key", "turn", "file")
        if column in summary.columns
    ]
    summary = summary.sort_values(sort_cols, na_position="first").drop(columns=["_scope_order"])

    return summary


def write_tsv(df: pd.DataFrame, path: str | Path):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


def print_metric_legend():
    print("\n# Metric guide")
    print("parse_ok_pct: generation/raw_generation parsed as a dict, or parse_error is empty.")
    print("plan_allowed_pct: predicted plan exists in apis/simple_api.json; None/null/empty fail by default.")
    print("argument_keys_subset_pct: predicted argument keys are all allowed for the predicted plan; values are not checked.")
    print("schema_compliance_pct: parse_ok, dict parsing, plan_allowed, and argument_keys_subset all passed.")
    print("existing_all_accuracy_pct: existing TSV all-column exact-match accuracy against GT plan+arguments.")


def main(args):
    tools = read_tools(args.tools)
    allow_none_plan = bool(args.allow_none_plan and not args.disallow_none_plan)

    input_tsv = args.input
    mode = "api_tools"
    label = Path(input_tsv).name

    details = build_detail(
        input_tsv,
        label=label,
        mode=mode,
        tools=tools,
        allow_none_plan=allow_none_plan,
    )
    summary = build_summary(details)

    display_cols = [
        "summary_scope",
        "test_key",
        "file",
        "turn",
        "samples",
        "parse_ok_pct",
        "plan_allowed_pct",
        "argument_keys_subset_pct",
        "schema_compliance_pct",
        "existing_all_accuracy_pct",
    ]
    display_cols = [column for column in display_cols if column in summary.columns]
    print(summary[display_cols].to_string(index=False))
    print_metric_legend()

    if args.out:
        write_tsv(summary, args.out)
        print(f"\nWrote summary: {args.out}")
    if args.detail_out:
        write_tsv(details, args.detail_out)
        print(f"Wrote detail: {args.detail_out}")


if __name__ == "__main__":
    main(get_arg_parse())
