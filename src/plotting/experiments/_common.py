from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.catalog import ExperimentSpec, get_experiment_spec


COLOR_DYNAMIC = "#E69F00"
COLOR_STATIC = "#56B4E9"
COLOR_ACCENT = "#009E73"
COLOR_GRID = "#E5E7EB"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def require_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required artifact not found: {path_obj}")
    return path_obj


def read_csv_validated(path: str | Path, required_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    csv_path = require_path(path)
    df = pd.read_csv(csv_path)
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {', '.join(missing)}")
    return df


def load_json(path: str | Path) -> Any:
    return json.loads(require_path(path).read_text(encoding="utf-8"))


def resolve_bundle_file(input_dir: Path, relative_name: str) -> Path:
    direct = input_dir / relative_name
    if direct.exists():
        return direct
    data_candidate = input_dir / "data" / relative_name
    if data_candidate.exists():
        return data_candidate
    return direct


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(next_prefix, item, out)
        return
    if isinstance(value, list) or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        scalar = float(value)
        if np.isfinite(scalar):
            out[prefix or "value"] = scalar


def _pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if column.endswith("_id") or column in {"trial_id", "pair_id"}:
                continue
            names.append(column)
    return names


def _pick_category_column(df: pd.DataFrame) -> str | None:
    priority = ["condition", "model_type", "layer", "substrate", "seq_len", "stage_k", "mode", "group"]
    for name in priority:
        if name in df.columns and 1 < df[name].nunique() <= 20:
            return name
    for name in df.columns:
        if not pd.api.types.is_numeric_dtype(df[name]) and 1 < df[name].nunique() <= 20:
            return name
    return None


def _pick_x_column(df: pd.DataFrame) -> str | None:
    for name in ["delay_ms", "bin_index", "time_step", "seq_len", "stage_k"]:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    return None


def _plot_summary_axis(ax: plt.Axes, summary_payload: dict[str, Any], *, title: str) -> None:
    flattened: dict[str, float] = {}
    _flatten_numeric("", summary_payload, flattened)
    items = list(flattened.items())[:12]
    if not items:
        ax.text(0.5, 0.5, "No numeric summary fields", ha="center", va="center")
        ax.set_axis_off()
        return
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    ax.barh(np.arange(len(items)), values, color=COLOR_ACCENT, alpha=0.85)
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6, alpha=0.7)


def _plot_csv_axis(ax: plt.Axes, df: pd.DataFrame, spec: ExperimentSpec) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "Primary CSV is empty", ha="center", va="center")
        ax.set_axis_off()
        return
    numeric_cols = _pick_numeric_columns(df)
    if not numeric_cols:
        ax.text(0.5, 0.5, "No numeric columns available", ha="center", va="center")
        ax.set_axis_off()
        return
    x_col = spec.csv_x if spec.csv_x in df.columns else _pick_x_column(df)
    y_cols = [name for name in spec.csv_y if name in df.columns] or numeric_cols[:2]
    group_col = spec.csv_group if spec.csv_group in df.columns else _pick_category_column(df)
    if x_col and y_cols:
        grouped_by = group_col if group_col and group_col != x_col else None
        if grouped_by:
            for idx, (group_name, sub) in enumerate(df.groupby(grouped_by, sort=True)):
                color = [COLOR_DYNAMIC, COLOR_STATIC, COLOR_ACCENT][idx % 3]
                ax.plot(sub[x_col], sub[y_cols[0]], marker="o", linewidth=1.8, label=str(group_name), color=color)
            ax.legend(frameon=False)
        else:
            for idx, y_name in enumerate(y_cols):
                color = [COLOR_DYNAMIC, COLOR_STATIC, COLOR_ACCENT][idx % 3]
                ax.plot(df[x_col], df[y_name], marker="o", linewidth=1.8, label=y_name, color=color)
            if len(y_cols) > 1:
                ax.legend(frameon=False)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_cols[0])
        ax.set_title(Path(spec.primary_csv or "data").stem)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.7)
        return
    if group_col and y_cols:
        grouped = df.groupby(group_col, sort=True)[y_cols[0]].mean().reset_index()
        ax.bar(grouped[group_col].astype(str), grouped[y_cols[0]], color=COLOR_DYNAMIC, alpha=0.85)
        ax.set_xlabel(group_col)
        ax.set_ylabel(y_cols[0])
        ax.set_title(Path(spec.primary_csv or "data").stem)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.7)
        return
    ax.bar(np.arange(min(len(df), 12)), df[y_cols[0]].head(12), color=COLOR_DYNAMIC, alpha=0.85)
    ax.set_ylabel(y_cols[0])
    ax.set_title(Path(spec.primary_csv or "data").stem)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.7)


def save_figure_outputs(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        out[ext] = str(path)
    return out


def build_experiment_figure(spec: ExperimentSpec, input_dir: Path) -> plt.Figure:
    apply_plot_style()
    summary = load_json(input_dir / "summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.suptitle(spec.title)
    _plot_summary_axis(axes[0], summary, title="Summary Metrics")
    if spec.primary_csv is None:
        axes[1].text(0.5, 0.5, "No primary CSV configured", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        df = read_csv_validated(resolve_bundle_file(input_dir, spec.primary_csv))
        _plot_csv_axis(axes[1], df, spec)
    return fig


def build_plot_parser(spec: ExperimentSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Plot-only entrypoint for {spec.title}.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main_for(experiment_id: str) -> int:
    spec = get_experiment_spec(experiment_id)
    parser = build_plot_parser(spec)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    require_path(input_dir / "summary.json")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "figures"
    fig = build_experiment_figure(spec, input_dir)
    try:
        save_figure_outputs(fig, output_dir, "figure_main")
    finally:
        plt.close(fig)
    return 0


__all__ = [
    "apply_plot_style",
    "build_experiment_figure",
    "load_json",
    "main_for",
    "read_csv_validated",
    "require_path",
    "resolve_bundle_file",
    "save_figure_outputs",
]
