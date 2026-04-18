import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TSV_PATH = Path(__file__).with_name("main-results.macro_summary.tsv")
MODEL_OUTPUT_PATH = Path(__file__).with_name("turn_performance_core3_models.pdf")
AVERAGE_OUTPUT_PATH = Path(__file__).with_name("turn_performance_core3_average.pdf")
SCORE_COL = "weighted_macro_curriculum_reverse"

MODEL_ORDER = [
    ("qwen3", "Qwen3 (4B)"),
    ("qwen2.5", "Qwen2.5 (3B)"),
    ("phi4", "Phi-4 (4B)"),
    ("llama3", "Llama3.2 (3B)"),
    ("smollm3", "SmolLM3 (3B)"),
    ("granite", "Granite3.3 (2B)"),
    ("glm", "GLM-edge (1.5B)"),
]

CORE_METHOD_ORDER = [
    ("baseline", "Baseline"),
    ("ours", "Ours"),
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
    "glm-edge": "glm",
}

METHOD_ALIASES = {
    "baseline": "baseline",
    "schema-aware history planner": "baseline",
    "baseline_resample": "baseline_resample",
    "baseline-resample": "baseline_resample",
    "baseline + resampling": "baseline_resample",
    "baseline+resampling": "baseline_resample",
    "schema-aware history planner + constraint-guided resampling": "baseline_resample",
    "baseline_retrieval": "baseline_retrieval",
    "baseline-retrieval": "baseline_retrieval",
    "baseline + retrieval": "baseline_retrieval",
    "baseline+retrieval": "baseline_retrieval",
    "retrieval-augmented schema-aware history planner": "baseline_retrieval",
    "ours": "ours",
    "explicit rewrite then plan": "ours",
    "explicit rewrite-then-plan": "ours",
    "rewrite-then-plan": "ours",
    "ours_resample": "ours_resample",
    "ours-resample": "ours_resample",
    "ours + resampling": "ours_resample",
    "ours+resampling": "ours_resample",
    "rewrite-then-plan + constraint-guided resampling": "ours_resample",
    "ours_oracle": "ours_oracle",
    "oracle": "ours_oracle",
    "oracle_rewrite": "ours_oracle",
    "oracle rewrite": "ours_oracle",
    "oracle rewrite + planner": "ours_oracle",
}


def norm(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


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


def load_plot_df(tsv_path=TSV_PATH):
    df = pd.read_csv(tsv_path, sep="\t")

    model_col = find_column(df, ["Model name", "Model Name", "model"])
    method_col = find_column(df, ["Method", "method"])
    turn_col = find_column(df, ["turn", "Turn"])

    if SCORE_COL not in df.columns:
        raise KeyError(f"Missing required score column: {SCORE_COL}")

    df = df[[model_col, method_col, turn_col, SCORE_COL]].copy()
    df = df[df[turn_col].astype(str).str.upper() != "TOTAL"].copy()
    df["model_key"] = df[model_col].map(canonical_model)
    df["method_key"] = df[method_col].map(canonical_method)
    df["turn"] = pd.to_numeric(df[turn_col], errors="coerce")
    df[SCORE_COL] = df[SCORE_COL].map(to_score)

    core_method_keys = [method_key for method_key, _ in CORE_METHOD_ORDER]
    df = df[df["method_key"].isin(core_method_keys)].copy()
    df = df.dropna(subset=["turn", SCORE_COL])
    df["turn"] = df["turn"].astype(int)
    df = df[df["turn"].isin([2, 3, 4, 5])].copy()

    return (
        df.groupby(["method_key", "model_key", "turn"], as_index=False)[SCORE_COL]
        .mean()
        .copy()
    )


def build_average_df(plot_df, model_keys):
    return (
        plot_df[plot_df["model_key"].isin(model_keys)]
        .groupby(["method_key", "turn"], as_index=False)[SCORE_COL]
        .mean()
    )


def get_model_keys(plot_df):
    available_models = set(plot_df["model_key"].unique())
    return [model_key for model_key, _ in MODEL_ORDER if model_key in available_models]


def get_method_keys(plot_df, method_order):
    available_methods = set(plot_df["method_key"].unique())
    return [
        method_key
        for method_key, _ in method_order
        if method_key in available_methods
    ]


def get_style_map():
    return {
        "baseline": {
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "s",
        },
        "baseline_resample": {
            "color": "#17becf",
            "linestyle": ":",
            "marker": "D",
        },
        "baseline_retrieval": {
            "color": "#9467bd",
            "linestyle": "-.",
            "marker": "v",
        },
        "ours": {
            "color": "#ff7f0e",
            "linestyle": "-",
            "marker": "o",
        },
        "ours_resample": {
            "color": "#d62728",
            "linestyle": ":",
            "marker": "P",
        },
        "ours_oracle": {
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "^",
        },
    }


def plot_method_lines(
    ax,
    sub,
    method_keys,
    method_label_by_key,
    style_map,
    legend_handles,
    linewidth=2.0,
    markersize=5.0,
):
    for method_key in method_keys:
        ms = sub[sub["method_key"] == method_key].sort_values("turn")
        if ms.empty:
            continue

        line, = ax.plot(
            ms["turn"],
            ms[SCORE_COL],
            linewidth=linewidth,
            markersize=markersize,
            label=method_label_by_key[method_key],
            **style_map.get(method_key, {}),
        )

        if method_key not in legend_handles:
            legend_handles[method_key] = line


def style_axis(ax, title, turn_order, title_fontsize=12, tick_fontsize=10):
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(turn_order)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)


def plot_model_panels(plot_df, output_path=MODEL_OUTPUT_PATH):
    model_label_by_key = dict(MODEL_ORDER)
    method_label_by_key = dict(CORE_METHOD_ORDER)
    model_keys = get_model_keys(plot_df)
    method_keys = get_method_keys(plot_df, CORE_METHOD_ORDER)

    if not model_keys:
        raise ValueError("No matching models found for plotting.")
    if not method_keys:
        raise ValueError("No matching methods found for plotting.")

    style_map = get_style_map()
    turn_order = [2, 3, 4, 5]

    n_panels = len(model_keys)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.2 * ncols, 2.75 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)
    legend_handles = {}

    for ax, model_key in zip(axes, model_keys):
        sub = plot_df[plot_df["model_key"] == model_key].copy()
        plot_method_lines(
            ax,
            sub,
            method_keys,
            method_label_by_key,
            style_map,
            legend_handles,
        )
        style_axis(ax, model_label_by_key[model_key], turn_order)

    for j in range(len(model_keys), len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Turn", fontsize=12, y=0.08)
    fig.supylabel("Macro Accuracy (%)", fontsize=12, x=0.02)

    handles = [legend_handles[m] for m in method_keys if m in legend_handles]
    labels = [method_label_by_key[m] for m in method_keys if m in legend_handles]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(rect=[0.05, 0.12, 1.0, 0.98])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_average_panel(plot_df, output_path=AVERAGE_OUTPUT_PATH):
    method_label_by_key = dict(CORE_METHOD_ORDER)
    model_keys = get_model_keys(plot_df)
    method_keys = get_method_keys(plot_df, CORE_METHOD_ORDER)

    if not model_keys:
        raise ValueError("No matching models found for averaging.")
    if not method_keys:
        raise ValueError("No matching methods found for plotting.")

    average_df = build_average_df(plot_df, model_keys)
    style_map = get_style_map()
    turn_order = [2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    legend_handles = {}
    plot_method_lines(
        ax,
        average_df,
        method_keys,
        method_label_by_key,
        style_map,
        legend_handles,
        linewidth=2.2,
        markersize=5.5,
    )
    style_axis(ax, "Average", turn_order, title_fontsize=13, tick_fontsize=11)
    ax.set_xlabel("Turn", fontsize=12, labelpad=10)
    ax.set_ylabel("Macro Accuracy (%)", fontsize=12)
    ax.set_ylim(62, 85)

    handles = [legend_handles[m] for m in method_keys if m in legend_handles]
    labels = [method_label_by_key[m] for m in method_keys if m in legend_handles]
    ax.legend(
        handles,
        labels,
        loc="lower left",
        ncol=1,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
        fontsize=10,
        handlelength=2.4,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    plot_df = load_plot_df()
    model_output_path = plot_model_panels(plot_df)
    average_output_path = plot_average_panel(plot_df)
    print(f"Saved model figure to: {model_output_path}")
    print(f"Saved average figure to: {average_output_path}")


if __name__ == "__main__":
    main()
