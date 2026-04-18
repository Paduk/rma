from pathlib import Path

import pandas as pd


TSV_PATH = Path(__file__).with_name("main-results.macro_summary.tsv")
SCORE_COL = "weighted_macro_curriculum_reverse"

MODEL_ORDER = [
    ("qwen3", "Qwen3"),
    ("qwen2.5", "Qwen2.5"),
    ("phi4", "Phi-4"),
    ("llama3", "Llama3.2"),
    ("smollm3", "SmolLM3"),
    ("granite", "Granite3.3"),
    ("glm", "GLM-edge"),
]

METHOD_ORDER = [
    ("baseline", "Baseline"),
    ("baseline_resample", "Baseline + Resampling"),
    ("baseline_retrieval", "Baseline + Retrieval"),
    ("ours", "Ours"),
    ("ours_resample", "Ours + Resampling"),
    ("ours_oracle", "Oracle Rewrite"),
]

MODEL_ALIASES = {
    "qwen3": "qwen3",
    "qwen2.5": "qwen2.5",
    "qwen25": "qwen2.5",
    "phi4": "phi4",
    "phi-4": "phi4",
    "llama3": "llama3",
    "llama3.2": "llama3",
    "smollm3": "smollm3",
    "granite": "granite",
    "granite3.3": "granite",
    "glm": "glm",
    "glm-1.5": "glm",
}

METHOD_ALIASES = {
    "baseline": "baseline",
    "baseline_resample": "baseline_resample",
    "baseline-re-sample": "baseline_resample",
    "baseline_retrieval": "baseline_retrieval",
    "baseline-retrieval": "baseline_retrieval",
    "ours": "ours",
    "ours_resample": "ours_resample",
    "ours-resample": "ours_resample",
    "ours_oracle": "ours_oracle",
    "oracle": "ours_oracle",
    "oracle_rewrite": "ours_oracle",
}


def norm(value):
    return str(value).strip().lower()


def find_column(df, candidates):
    normalized_columns = {norm(column): column for column in df.columns}
    for candidate in candidates:
        column = normalized_columns.get(norm(candidate))
        if column is not None:
            return column
    raise KeyError(f"Missing required column. Expected one of: {candidates}")


def canonical_model(value):
    key = norm(value)
    return MODEL_ALIASES.get(key, key)


def canonical_method(value):
    key = norm(value)
    return METHOD_ALIASES.get(key, key)


def to_score(value):
    if pd.isnull(value):
        return pd.NA
    return pd.to_numeric(str(value).strip().rstrip("%"), errors="coerce")


def format_score(value):
    if pd.isnull(value):
        return "--"
    return f"{float(value):.1f}"


def build_total_score_pivot(tsv_path=TSV_PATH):
    df = pd.read_csv(tsv_path, sep="\t")

    model_col = find_column(df, ["Model name", "Model Name", "model"])
    method_col = find_column(df, ["Method", "method"])
    turn_col = find_column(df, ["turn", "Turn"])

    if SCORE_COL not in df.columns:
        raise KeyError(f"Missing required score column: {SCORE_COL}")

    total_df = df[df[turn_col].astype(str).str.upper() == "TOTAL"].copy()
    total_df["model_key"] = total_df[model_col].map(canonical_model)
    total_df["method_key"] = total_df[method_col].map(canonical_method)
    total_df[SCORE_COL] = total_df[SCORE_COL].map(to_score)

    return total_df.pivot_table(
        index="method_key",
        columns="model_key",
        values=SCORE_COL,
        aggfunc="mean",
    )


def build_latex_table(pivot):
    model_keys = [model_key for model_key, _ in MODEL_ORDER]
    model_headers = [model_label for _, model_label in MODEL_ORDER]

    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Method} ",
    ]
    lines.extend([f"& \\textbf{{{header}}} " for header in model_headers])
    lines.append(r"& \textbf{Avg} \\")
    lines.append(r"\midrule")

    for idx, (method_key, method_label) in enumerate(METHOD_ORDER):
        values = []
        numeric_values = []

        for model_key in model_keys:
            value = pd.NA
            if method_key in pivot.index and model_key in pivot.columns:
                value = pivot.loc[method_key, model_key]
            values.append(format_score(value))
            if pd.notnull(value):
                numeric_values.append(float(value))

        avg_value = sum(numeric_values) / len(numeric_values) if numeric_values else pd.NA
        row_values = values + [format_score(avg_value)]
        lines.append(f"{method_label:<25} & " + " & ".join(row_values) + r" \\")

        if idx == 2:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main():
    pivot = build_total_score_pivot()
    print(build_latex_table(pivot))


if __name__ == "__main__":
    main()
