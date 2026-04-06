#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path
import json
from ast import literal_eval
import random
import math
import re

DEFAULT_DATA_PATHS = [
    Path(__file__).resolve().parent / "datasets/manual/ad/rma-qwen25.tsv",
    #Path(__file__).resolve().parent / "datasets/manual/ad/rewrite-qwen25-base.tsv",
]


TURN_PATTERN = re.compile(r"^turn(\d+)")


def parse_plan(raw_plan):
    if raw_plan is None:
        return None
    text = str(raw_plan).strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = literal_eval(text)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return data.get("plan")


def extract_turn(fname: str):
    stem = Path(fname).stem
    match = TURN_PATTERN.match(stem)
    if match:
        return match.group(1)
    return None


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                turn = extract_turn(row["file"])
                if turn is None:
                    continue
                plan = parse_plan(row.get("gt"))
                if plan is None:
                    continue
                is_pass = str(row["all"]).strip().lower() == "pass"
                rows.append((turn, plan, is_pass))
    return rows


def macro_accuracy(plan_samples):
    per_plan_acc = []
    for samples in plan_samples.values():
        if not samples:
            continue
        per_plan_acc.append(sum(samples) / len(samples))
    if not per_plan_acc:
        return None, 0
    return sum(per_plan_acc) / len(per_plan_acc), len(per_plan_acc)


def percentile(values, p):
    if not values:
        return None
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def bootstrap_macro_ci(plan_samples, iterations=2000, alpha=0.05, seed=42):
    keys = list(plan_samples.keys())
    if not keys:
        return None, None
    rng = random.Random(seed)
    scores = []
    for _ in range(iterations):
        per_plan = []
        for key in keys:
            samples = plan_samples[key]
            if not samples:
                continue
            resample = [samples[rng.randrange(len(samples))] for _ in range(len(samples))]
            per_plan.append(sum(resample) / len(resample))
        if per_plan:
            scores.append(sum(per_plan) / len(per_plan))
    if not scores:
        return None, None
    scores.sort()
    lower = percentile(scores, alpha / 2)
    upper = percentile(scores, 1 - alpha / 2)
    return lower, upper


def format_with_ci(acc, lower, upper):
    if lower is None or upper is None:
        return f"{acc:.2%}"
    margin = max(acc - lower, upper - acc)
    return f"{acc:.2%} \u00b1 {margin:.2%}"


def main(paths):
    data = load_rows(paths)
    per_turn_plan_samples = defaultdict(lambda: defaultdict(list))
    overall_plan_samples = defaultdict(list)

    for turn, plan, is_pass in data:
        per_turn_plan_samples[turn][plan].append(1 if is_pass else 0)
        overall_plan_samples[plan].append(1 if is_pass else 0)

    for turn in sorted(per_turn_plan_samples, key=int):
        acc, _ = macro_accuracy(per_turn_plan_samples[turn])
        if acc is None:
            print("no valid rows")
            continue
        lower, upper = bootstrap_macro_ci(per_turn_plan_samples[turn])
        print(format_with_ci(acc, lower, upper))

    overall_acc, _ = macro_accuracy(overall_plan_samples)
    if overall_acc is None:
        print("no valid rows")
    else:
        lower, upper = bootstrap_macro_ci(overall_plan_samples)
        print(format_with_ci(overall_acc, lower, upper))


if __name__ == "__main__":
    # 예: python statistic_turn.py RMA/datasets/manual/ad/rewrite-qwen25.tsv RMA/datasets/manual/ad/rewrite-qwen25-base.tsv
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = [str(p) for p in DEFAULT_DATA_PATHS]
    main(paths)
