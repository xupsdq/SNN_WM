from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.catalog import EXPERIMENT_SPECS
from src.plotting.common.io import apply_publication_style, save_figure_all_formats


SUMMARY_VALUE_COLUMNS = {"metric", "n_networks", "mean", "sd", "sem", "ci_low", "ci_high", "status"}


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _summary_paths(input_dir: Path, summary_payload: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    aggregate_outputs = summary_payload.get("aggregate_outputs", {})
    if isinstance(aggregate_outputs, dict):
        for payload in aggregate_outputs.values():
            if not isinstance(payload, dict):
                continue
            rel = payload.get("network_summary_csv")
            if rel:
                candidate = input_dir / str(rel)
                if candidate.is_file():
                    paths.append(candidate)
    if not paths:
        paths = sorted((input_dir / "metrics").glob("*__network_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No *__network_summary.csv files found in {input_dir / 'metrics'}")
    return paths


def _group_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in SUMMARY_VALUE_COLUMNS]


def _metric_order(df: pd.DataFrame, experiment_id: str | None) -> list[str]:
    metrics = [str(item) for item in pd.unique(df["metric"].astype(str))]
    plottable = [
        metric
        for metric in metrics
        if not (
            metric.lower().startswith("sem_")
            or metric.lower().startswith("sim_bin")
            or metric.lower() in {"n_trials", "bin_center"}
        )
    ]
    preferred: list[str] = []
    if experiment_id and experiment_id in EXPERIMENT_SPECS:
        preferred.extend(EXPERIMENT_SPECS[experiment_id].csv_y)
    preferred.extend([metric for metric in plottable if any(token in metric.lower() for token in ("acc", "drop", "delta", "effect", "difference"))])
    ordered: list[str] = []
    for metric in preferred + plottable:
        if metric in plottable and metric not in ordered:
            ordered.append(metric)
    return ordered[:6]


def _pick_axes(group_cols: Sequence[str], experiment_id: str | None) -> tuple[str | None, str | None]:
    spec = EXPERIMENT_SPECS.get(experiment_id or "")
    if spec is not None:
        x_col = spec.csv_x if spec.csv_x in group_cols else None
        hue_col = spec.csv_group if spec.csv_group in group_cols else None
        if x_col:
            return x_col, hue_col
    if not group_cols:
        return None, None
    return group_cols[0], group_cols[1] if len(group_cols) > 1 else None


def _label_series(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[list(group_cols)].astype(str).agg(" | ".join, axis=1)


def _plot_metric(df: pd.DataFrame, *, metric: str, title: str, experiment_id: str | None) -> plt.Figure:
    sub = df[df["metric"].astype(str) == metric].copy()
    group_cols = _group_columns(sub)
    x_col, hue_col = _pick_axes(group_cols, experiment_id)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    if x_col is None:
        labels = _label_series(sub, group_cols)
        x = np.arange(len(sub))
        ax.bar(x, sub["mean"].to_numpy(dtype=np.float64), yerr=sub["sem"].fillna(0).to_numpy(dtype=np.float64), capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    elif hue_col and hue_col in sub.columns:
        for label, hue_df in sub.groupby(hue_col, sort=True):
            plot_df = hue_df.sort_values(x_col)
            ax.errorbar(
                plot_df[x_col],
                plot_df["mean"].to_numpy(dtype=np.float64),
                yerr=plot_df["sem"].fillna(0).to_numpy(dtype=np.float64),
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=str(label),
            )
        ax.legend(frameon=False)
    else:
        plot_df = sub.sort_values(x_col)
        ax.errorbar(
            plot_df[x_col],
            plot_df["mean"].to_numpy(dtype=np.float64),
            yerr=plot_df["sem"].fillna(0).to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.8,
            capsize=4,
        )
    ax.set_title(title)
    ax.set_xlabel(x_col or "condition")
    ax.set_ylabel(f"{metric} mean +/- SEM")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot mean +/- SEM summaries from a multi-network result bundle.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    summary_payload = _load_json(input_dir / "summary.json")
    experiment_id = str(summary_payload.get("experiment_id") or "")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    saved: dict[str, dict[str, str]] = {}
    for summary_path in _summary_paths(input_dir, summary_payload):
        df = pd.read_csv(summary_path)
        if df.empty or "metric" not in df.columns:
            continue
        table_stem = summary_path.name.replace("__network_summary.csv", "")
        for metric in _metric_order(df, experiment_id):
            fig = _plot_metric(
                df,
                metric=metric,
                title=f"{table_stem}: {metric}",
                experiment_id=experiment_id,
            )
            try:
                saved[f"{table_stem}__{metric}"] = save_figure_all_formats(fig, output_dir / f"{table_stem}__{metric}")
            finally:
                plt.close(fig)
    if not saved:
        raise RuntimeError(f"No multi-network summary figures were generated for {input_dir}")
    log_path = input_dir / "logs" / "multi_network_summary_plot.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"timestamp={_timestamp_now()}",
                f"input_dir={input_dir}",
                f"output_dir={output_dir}",
                f"command={subprocess.list2cmdline(sys.argv)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = input_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    manifest["multi_network_plot_outputs"] = saved
    manifest["multi_network_plot_replayed_at"] = _timestamp_now()
    manifest_path.write_text(json.dumps(_to_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
