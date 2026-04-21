from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common.io import save_figure_all_formats


CONDITION_A_DYNAMIC_BASE = "A_dynamic_base"
CONDITION_B_TRIAL_SHUFFLE_SPIKE = "B_trial_shuffle_spike"
CONDITION_C_TRIAL_SHUFFLE_MEMBRANE = "C_trial_shuffle_membrane"
CONDITION_D_TRIAL_SHUFFLE_UX = "D_trial_shuffle_ux"
CONDITION_E_STATIC_FROZEN = "E_static_frozen"

CONDITION_ORDER = [
    CONDITION_A_DYNAMIC_BASE,
    CONDITION_B_TRIAL_SHUFFLE_SPIKE,
    CONDITION_C_TRIAL_SHUFFLE_MEMBRANE,
    CONDITION_D_TRIAL_SHUFFLE_UX,
    CONDITION_E_STATIC_FROZEN,
]

CONDITION_LABELS = {
    CONDITION_A_DYNAMIC_BASE: "A: dynamic",
    CONDITION_B_TRIAL_SHUFFLE_SPIKE: "B: trial-shuffle spike-state",
    CONDITION_C_TRIAL_SHUFFLE_MEMBRANE: "C: trial-shuffle membrane",
    CONDITION_D_TRIAL_SHUFFLE_UX: "D: trial-shuffle u/x",
    CONDITION_E_STATIC_FROZEN: "E: static frozen",
}

DEFAULT_MANIFEST = {
    "version": 1,
    "experiment_name": "ux_shuffle_memory_collapse",
    "inputs": {
        "metrics_condition_summary": {
            "path": "metrics/metrics_condition_summary.csv",
            "required_columns": [
                "condition",
                "abs_rate_pred_original_sample",
                "abs_rate_pred_change_under_bmap",
            ],
            "purpose": "Primary memory-readout target figure input.",
        }
    },
    "outputs": [
        "figures/memory_readout_target.png",
        "figures/memory_readout_target.pdf",
        "figures/memory_readout_target.svg",
    ],
}


@dataclass(frozen=True)
class UxShufflePlotBundle:
    metrics_condition_df: pd.DataFrame


def write_plot_bundle_manifest(meta_dir: Path) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "plot_bundle_manifest.json"
    out_path.write_text(json.dumps(DEFAULT_MANIFEST, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out_path


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "meta" / "plot_bundle_manifest.json"
    if not manifest_path.exists():
        return DEFAULT_MANIFEST
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_input_path(input_dir: Path, relative_path: str) -> Path:
    candidate = input_dir / relative_path
    if candidate.exists():
        return candidate
    basename = Path(relative_path).name
    for fallback in (
        input_dir / basename,
        input_dir / "data" / basename,
        input_dir / "metrics" / basename,
        input_dir / "meta" / basename,
    ):
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Required artifact not found: {candidate}")


def _read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")
    return df


def load_plot_bundle(input_dir: str | Path) -> UxShufflePlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    spec = manifest["inputs"]["metrics_condition_summary"]
    csv_path = _resolve_input_path(input_path, str(spec["path"]))
    return UxShufflePlotBundle(
        metrics_condition_df=_read_csv(csv_path, list(spec.get("required_columns", ()))),
    )


def _nice_axis_upper(values: np.ndarray) -> float:
    max_value = float(np.nanmax(values)) if values.size > 0 else 0.0
    if max_value <= 0.0:
        return 5.0
    target = max(5.0, max_value * 1.25)
    exponent = np.floor(np.log10(target))
    base = 10.0 ** exponent
    for mult in (1.0, 2.0, 5.0, 10.0):
        candidate = mult * base
        if candidate >= target:
            return float(candidate)
    return float(10.0 * base)


def _nice_tick_step(upper: float) -> float:
    raw = max(1.0, upper / 5.0)
    exponent = np.floor(np.log10(raw))
    base = 10.0 ** exponent
    for mult in (1.0, 2.0, 5.0, 10.0):
        candidate = mult * base
        if candidate >= raw:
            return float(candidate)
    return float(10.0 * base)


def build_memory_readout_target_figure(metrics_condition: pd.DataFrame) -> plt.Figure:
    m = metrics_condition.set_index("condition").loc[CONDITION_ORDER].reset_index()
    x = np.arange(len(CONDITION_ORDER))
    width = 0.38
    red = m["abs_rate_pred_original_sample"].to_numpy(dtype=float)
    green = m["abs_rate_pred_change_under_bmap"].to_numpy(dtype=float)
    upper = _nice_axis_upper(np.concatenate([red, green], axis=0))
    tick_step = _nice_tick_step(upper)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.bar(
        x - width / 2,
        red,
        width=width,
        color="#d62728",
        edgecolor="black",
        alpha=0.9,
        label="Pred = original sample",
    )
    ax.bar(
        x + width / 2,
        green,
        width=width,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.9,
        label="Pred = change (B-map)",
    )
    ax.set_xticks(x, [CONDITION_LABELS[o] for o in CONDITION_ORDER], rotation=10)
    ax.set_ylabel("Absolute Rate (%)")
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0.0, upper + 0.5 * tick_step, tick_step))
    ax.set_title("Memory Readout Target by Shuffled Substrate")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def render_figures(bundle: UxShufflePlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    fig = build_memory_readout_target_figure(bundle.metrics_condition_df)
    try:
        return {"memory_readout_target": save_figure_all_formats(fig, figures_dir / "memory_readout_target")}
    finally:
        plt.close(fig)


__all__ = [
    "DEFAULT_MANIFEST",
    "UxShufflePlotBundle",
    "build_memory_readout_target_figure",
    "load_plot_bundle",
    "render_figures",
    "write_plot_bundle_manifest",
]
