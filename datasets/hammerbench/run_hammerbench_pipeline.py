#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

from build_eval_prompts import (
    build_baseline_prompt,
    build_planner_from_rewrite_prompt,
    build_rewrite_prompt,
    build_rewrite_fewshot_examples,
)


HAMMERBENCH_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "http://localhost:11436"
DEFAULT_EXTERNAL_EVAL = (
    HAMMERBENCH_DIR
    / "en"
    / "multi-turn.phase1_external_mQsA.turn_gt_1.external.eval.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the HammerBench phase1 pipeline end to end with a single command. "
            "Default dataset is the External subset."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "ours"],
        required=True,
        help="Pipeline mode: baseline(full history -> planner) or ours(rewrite -> planner).",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=DEFAULT_EXTERNAL_EVAL,
        help="Canonical eval JSONL path. Defaults to the External eval set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for prompts, predictions, merged evals, and summaries. "
            "If omitted, outputs are routed under the default runs path with "
            "provider/model subdirectories."
        ),
    )
    parser.add_argument(
        "--planner-model",
        required=True,
        help="Planner model name or alias.",
    )
    parser.add_argument(
        "--baseline-prompt-key",
        choices=["default", "strong", "hammer"],
        default="default",
        help=(
            "Baseline prompt variant. `default` keeps the original baseline; "
            "`strong` uses rewrite-style planning instructions without running a "
            "rewrite stage; `hammer` uses an Appendix D-inspired HammerBench prompt "
            "adapted to the local evaluator."
        ),
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="Rewrite model name or alias. Required when --mode ours.",
    )
    parser.add_argument(
        "--rewrite-prompt-key",
        choices=[
            "legacy",
            "default",
            "evidence",
            "evidence_fewshot",
            "evidence_fewshot_light",
            "evidence_fewshot_full",
            "structured_fewshot_full_5",
        ],
        default="default",
        help=(
            "Rewrite prompt variant for --mode ours. `default` uses exact-span "
            "preservation rules; `legacy` preserves the original prompt; `evidence` "
            "is an explicit alias for default; `evidence_fewshot_light` adds two "
            "compact gold-call-derived examples; `evidence_fewshot_full` adds two "
            "full-format examples; `structured_fewshot_full_5` emits rewritten_query "
            "and confirmed_arguments with five full-format examples. "
            "`evidence_fewshot` is kept as an alias for `evidence_fewshot_light`."
        ),
    )
    parser.add_argument(
        "--planner-backend",
        choices=["ollama", "cloud"],
        default="ollama",
        help="Backend for the planner stage.",
    )
    parser.add_argument(
        "--rewrite-backend",
        choices=["ollama", "cloud"],
        default="ollama",
        help="Backend for the rewrite stage.",
    )
    parser.add_argument(
        "--planner-host",
        default=DEFAULT_HOST,
        help="Ollama host for the planner model.",
    )
    parser.add_argument(
        "--rewrite-host",
        default=DEFAULT_HOST,
        help="Ollama host for the rewrite model. Ignored for cloud backend.",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=0.0,
        help="Planner generation temperature. Ignored for cloud backend.",
    )
    parser.add_argument(
        "--rewrite-temperature",
        type=float,
        default=0.0,
        help="Rewrite generation temperature. Ignored for cloud backend.",
    )
    parser.add_argument(
        "--planner-num-predict",
        type=int,
        default=512,
        help="Planner max generation tokens.",
    )
    parser.add_argument(
        "--rewrite-num-predict",
        type=int,
        default=256,
        help="Rewrite max generation tokens. Ignored for cloud backend.",
    )
    parser.add_argument(
        "--planner-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Planner reasoning effort for cloud OpenAI models.",
    )
    parser.add_argument(
        "--rewrite-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Rewrite reasoning effort for cloud OpenAI models.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=4,
        help="Concurrent Ollama requests for planner/rewrite stages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of records to run through the model stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Do not call models. Instead, print dataset counts and the first prompt(s) "
            "that would be sent to the model."
        ),
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=1,
        help="Number of example prompt records to print in --preview-only mode.",
    )
    parser.add_argument(
        "--planner-preview-query",
        default="<MODEL_GENERATED_REWRITTEN_QUERY>",
        help=(
            "Placeholder rewritten query used to preview the planner prompt in "
            "--mode ours when --preview-only is set."
        ),
    )
    return parser.parse_args()


def default_output_dir_for_eval(eval_path):
    if eval_path == DEFAULT_EXTERNAL_EVAL:
        return HAMMERBENCH_DIR / "runs" / "external"
    return HAMMERBENCH_DIR / "runs" / eval_path.stem


def sanitize_path_component(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return sanitized.strip("._") or "unknown"


def provider_dir_for_model(backend, model):
    if backend == "ollama":
        return "ollama"
    lowered = str(model or "").lower()
    if "gemini" in lowered:
        return "google"
    if "claude" in lowered:
        return "claude"
    return "chatgpt"


def default_run_output_dir(args):
    base_dir = default_output_dir_for_eval(args.eval)

    if args.mode == "baseline":
        return (
            base_dir
            / provider_dir_for_model(args.planner_backend, args.planner_model)
            / sanitize_path_component(args.planner_model)
        )

    planner_dir = (
        provider_dir_for_model(args.planner_backend, args.planner_model),
        sanitize_path_component(args.planner_model),
    )
    rewrite_dir = (
        provider_dir_for_model(args.rewrite_backend, args.rewrite_model),
        sanitize_path_component(args.rewrite_model),
    )

    if planner_dir == rewrite_dir:
        return base_dir / planner_dir[0] / planner_dir[1]

    return (
        base_dir
        / "planner"
        / planner_dir[0]
        / planner_dir[1]
        / "rewrite"
        / rewrite_dir[0]
        / rewrite_dir[1]
    )


def run_cmd(cmd, dry_run=False):
    printable = shlex.join(str(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def script_path(name):
    return HAMMERBENCH_DIR / name


def build_prompt_cmd(
    eval_path,
    mode,
    output_path,
    rewritten_query_field=None,
    confirmed_arguments_field=None,
    baseline_prompt_key="default",
    rewrite_prompt_key="default",
):
    cmd = [
        sys.executable,
        str(script_path("build_eval_prompts.py")),
        "--input",
        str(eval_path),
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    if rewritten_query_field:
        cmd.extend(["--rewritten-query-field", rewritten_query_field])
    if confirmed_arguments_field:
        cmd.extend(["--confirmed-arguments-field", confirmed_arguments_field])
    if mode == "baseline" and baseline_prompt_key != "default":
        cmd.extend(["--baseline-prompt-key", baseline_prompt_key])
    if mode == "rewrite" and rewrite_prompt_key != "default":
        cmd.extend(["--rewrite-prompt-key", rewrite_prompt_key])
    return cmd


def build_run_cmd(
    prompt_path,
    output_path,
    model,
    backend,
    host,
    temperature,
    num_predict,
    num_parallel,
    limit,
    reasoning_effort=None,
):
    if backend == "ollama":
        cmd = [
            sys.executable,
            str(script_path("run_prompt_jsonl_ollama.py")),
            "--input",
            str(prompt_path),
            "--output",
            str(output_path),
            "--model",
            model,
            "--host",
            host,
            "--temperature",
            str(temperature),
            "--num-predict",
            str(num_predict),
            "--num-parallel",
            str(num_parallel),
        ]
    elif backend == "cloud":
        cmd = [
            sys.executable,
            str(script_path("run_prompt_jsonl_cloud.py")),
            "--input",
            str(prompt_path),
            "--output",
            str(output_path),
            "--model",
            model,
            "--num-parallel",
            str(num_parallel),
        ]
        if reasoning_effort is not None:
            cmd.extend(["--reasoning-effort", reasoning_effort])
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def build_eval_cmd(pred_path, summary_path, rows_path):
    return [
        sys.executable,
        str(script_path("evaluate_predictions.py")),
        "--input",
        str(pred_path),
        "--summary-out",
        str(summary_path),
        "--rows-out",
        str(rows_path),
    ]


def build_audit_cmd(
    mode,
    output_path,
    planner_prompt_path,
    planner_pred_path,
    rows_path,
    rewrite_prompt_path=None,
    rewrite_pred_path=None,
):
    cmd = [
        sys.executable,
        str(script_path("build_run_audit.py")),
        "--mode",
        mode,
        "--planner-prompts",
        str(planner_prompt_path),
        "--planner-preds",
        str(planner_pred_path),
        "--rows",
        str(rows_path),
        "--output",
        str(output_path),
    ]
    if rewrite_prompt_path is not None:
        cmd.extend(["--rewrite-prompts", str(rewrite_prompt_path)])
    if rewrite_pred_path is not None:
        cmd.extend(["--rewrite-preds", str(rewrite_pred_path)])
    return cmd


def build_merge_cmd(
    eval_path,
    rewrite_pred_path,
    output_path,
    field_name,
    confirmed_arguments_field_name="model_confirmed_arguments",
):
    return [
        sys.executable,
        str(script_path("merge_rewrite_predictions.py")),
        "--eval",
        str(eval_path),
        "--rewrite-preds",
        str(rewrite_pred_path),
        "--output",
        str(output_path),
        "--field-name",
        field_name,
        "--confirmed-arguments-field-name",
        confirmed_arguments_field_name,
    ]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_preview(args):
    rows = read_jsonl(args.eval)
    effective_rows = rows[: args.limit] if args.limit is not None else rows
    counts = Counter(row.get("data_type", "unknown") for row in effective_rows)

    print("preview_only\tTrue")
    print(f"eval_records_total\t{len(rows)}")
    print(f"eval_records_effective\t{len(effective_rows)}")
    if args.limit is not None:
        print(f"limit\t{args.limit}")
    for data_type in sorted(counts):
        print(f"data_type_count\t{data_type}\t{counts[data_type]}")

    preview_rows = effective_rows[: args.preview_limit]
    rewrite_fewshot_examples = ""
    if args.mode == "ours":
        rewrite_fewshot_examples = build_rewrite_fewshot_examples(
            rows,
            args.rewrite_prompt_key,
        )
    for idx, record in enumerate(preview_rows, start=1):
        print(f"preview_index\t{idx}")
        print(f"id\t{record['id']}")
        print(f"data_type\t{record['data_type']}")
        print(f"turn_id\t{record['turn_id']}")

        if args.mode == "baseline":
            system_instruction, user_input = build_baseline_prompt(
                record,
                prompt_key=args.baseline_prompt_key,
            )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input},
            ]
            print(f"baseline_prompt_key\t{args.baseline_prompt_key}")
            print("baseline_prompt_messages")
            print(json.dumps(messages, ensure_ascii=True, indent=2))
            print("---")
            continue

        rewrite_system, rewrite_user = build_rewrite_prompt(
            record,
            prompt_key=args.rewrite_prompt_key,
            fewshot_examples=rewrite_fewshot_examples,
        )
        rewrite_messages = [
            {"role": "system", "content": rewrite_system},
            {"role": "user", "content": rewrite_user},
        ]
        planner_system, planner_user = build_planner_from_rewrite_prompt(
            record,
            args.planner_preview_query,
            confirmed_arguments={"example_key": "<MODEL_GENERATED_VALUE>"},
        )
        planner_messages = [
            {"role": "system", "content": planner_system},
            {"role": "user", "content": planner_user},
        ]
        print(f"rewrite_prompt_key\t{args.rewrite_prompt_key}")
        print("rewrite_prompt_messages")
        print(json.dumps(rewrite_messages, ensure_ascii=True, indent=2))
        print("planner_prompt_messages")
        print(json.dumps(planner_messages, ensure_ascii=True, indent=2))
        print("---")


def main():
    args = parse_args()

    if not args.eval.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval}")
    if args.mode == "ours" and not args.rewrite_model:
        raise ValueError("--rewrite-model is required when --mode ours")

    output_dir = args.output_dir or default_run_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.eval.stem

    print(f"mode\t{args.mode}")
    print(f"eval\t{args.eval}")
    print(f"output_dir\t{output_dir}")
    print(f"default_external_eval\t{DEFAULT_EXTERNAL_EVAL}")
    print(f"planner_backend\t{args.planner_backend}")
    if args.mode == "baseline":
        print(f"baseline_prompt_key\t{args.baseline_prompt_key}")
    if args.mode == "ours":
        print(f"rewrite_backend\t{args.rewrite_backend}")
        print(f"rewrite_prompt_key\t{args.rewrite_prompt_key}")

    if args.preview_only:
        print_preview(args)
        return

    if args.mode == "baseline":
        baseline_tag = (
            "baseline"
            if args.baseline_prompt_key == "default"
            else f"baseline.{args.baseline_prompt_key}"
        )
        prompt_path = output_dir / f"{dataset_tag}.{baseline_tag}.prompt.jsonl"
        pred_path = output_dir / f"{dataset_tag}.{baseline_tag}.pred.jsonl"
        summary_path = output_dir / f"{dataset_tag}.{baseline_tag}.summary.json"
        rows_path = output_dir / f"{dataset_tag}.{baseline_tag}.rows.jsonl"
        audit_path = output_dir / f"{dataset_tag}.{baseline_tag}.audit.jsonl"

        run_cmd(
            build_prompt_cmd(
                args.eval,
                mode="baseline",
                output_path=prompt_path,
                baseline_prompt_key=args.baseline_prompt_key,
            ),
            dry_run=args.dry_run,
        )
        run_cmd(
            build_run_cmd(
                prompt_path=prompt_path,
                output_path=pred_path,
                model=args.planner_model,
                backend=args.planner_backend,
                host=args.planner_host,
                temperature=args.planner_temperature,
                num_predict=args.planner_num_predict,
                num_parallel=args.num_parallel,
                limit=args.limit,
                reasoning_effort=args.planner_reasoning_effort,
            ),
            dry_run=args.dry_run,
        )
        run_cmd(
            build_eval_cmd(pred_path, summary_path, rows_path),
            dry_run=args.dry_run,
        )
        run_cmd(
            build_audit_cmd(
                mode="baseline",
                output_path=audit_path,
                planner_prompt_path=prompt_path,
                planner_pred_path=pred_path,
                rows_path=rows_path,
            ),
            dry_run=args.dry_run,
        )
        return

    ours_tag = (
        "ours"
        if args.rewrite_prompt_key == "default"
        else f"ours.{args.rewrite_prompt_key}"
    )
    rewrite_prompt_path = output_dir / f"{dataset_tag}.{ours_tag}.rewrite.prompt.jsonl"
    rewrite_pred_path = output_dir / f"{dataset_tag}.{ours_tag}.rewrite.pred.jsonl"
    merged_eval_path = output_dir / f"{dataset_tag}.{ours_tag}.with_model_rewrite.eval.jsonl"
    planner_prompt_path = output_dir / f"{dataset_tag}.{ours_tag}.planner_from_rewrite.prompt.jsonl"
    pred_path = output_dir / f"{dataset_tag}.{ours_tag}.pred.jsonl"
    summary_path = output_dir / f"{dataset_tag}.{ours_tag}.summary.json"
    rows_path = output_dir / f"{dataset_tag}.{ours_tag}.rows.jsonl"
    audit_path = output_dir / f"{dataset_tag}.{ours_tag}.audit.jsonl"

    run_cmd(
        build_prompt_cmd(
            args.eval,
            mode="rewrite",
            output_path=rewrite_prompt_path,
            rewrite_prompt_key=args.rewrite_prompt_key,
        ),
        dry_run=args.dry_run,
    )
    run_cmd(
        build_run_cmd(
            prompt_path=rewrite_prompt_path,
            output_path=rewrite_pred_path,
            model=args.rewrite_model,
            backend=args.rewrite_backend,
            host=args.rewrite_host,
            temperature=args.rewrite_temperature,
            num_predict=args.rewrite_num_predict,
            num_parallel=args.num_parallel,
            limit=args.limit,
            reasoning_effort=args.rewrite_reasoning_effort,
        ),
        dry_run=args.dry_run,
    )
    run_cmd(
        build_merge_cmd(
            eval_path=args.eval,
            rewrite_pred_path=rewrite_pred_path,
            output_path=merged_eval_path,
            field_name="model_rewritten_query",
            confirmed_arguments_field_name="model_confirmed_arguments",
        ),
        dry_run=args.dry_run,
    )
    run_cmd(
        build_prompt_cmd(
            merged_eval_path,
            mode="planner_from_rewrite",
            output_path=planner_prompt_path,
            rewritten_query_field="model_rewritten_query",
            confirmed_arguments_field="model_confirmed_arguments",
        ),
        dry_run=args.dry_run,
    )
    run_cmd(
        build_run_cmd(
            prompt_path=planner_prompt_path,
            output_path=pred_path,
            model=args.planner_model,
            backend=args.planner_backend,
            host=args.planner_host,
            temperature=args.planner_temperature,
            num_predict=args.planner_num_predict,
            num_parallel=args.num_parallel,
            limit=args.limit,
            reasoning_effort=args.planner_reasoning_effort,
        ),
        dry_run=args.dry_run,
    )
    run_cmd(
        build_eval_cmd(pred_path, summary_path, rows_path),
        dry_run=args.dry_run,
    )
    run_cmd(
        build_audit_cmd(
            mode="ours",
            output_path=audit_path,
            planner_prompt_path=planner_prompt_path,
            planner_pred_path=pred_path,
            rows_path=rows_path,
            rewrite_prompt_path=rewrite_prompt_path,
            rewrite_pred_path=rewrite_pred_path,
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
