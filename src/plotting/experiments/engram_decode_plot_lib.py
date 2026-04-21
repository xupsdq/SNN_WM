from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting.common.io import save_figure_all_formats


LAYER_KEYS = ("layer1", "layer2", "layer3")
LAYER_DISPLAY_NAMES = {
    "layer1": "Layer1",
    "layer2": "Layer2",
    "layer3": "Layer3",
}
CHANCE_LEVEL = 0.1

DEFAULT_MANIFEST = {
    "version": 1,
    "experiment_name": "engram_decode",
    "inputs": {
        "engram_decode_metrics": {
            "path": "metrics/engram_decode_metrics.csv",
            "required_columns": ["layer", "delay_ms", "acc", "acc_ci_low", "acc_ci_high", "macro_f1"],
            "purpose": "Primary accuracy-vs-delay figure input.",
        }
    },
    "outputs": [
        "figures/accuracy_vs_delay.png",
        "figures/accuracy_vs_delay.pdf",
        "figures/accuracy_vs_delay.svg",
    ],
    "notes": [
        "Optional diagnostic confusion/PCA plots are not replayed by plot-only.",
    ],
}


@dataclass(frozen=True)
class EngramDecodePlotBundle:
    metrics_df: pd.DataFrame


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


def load_plot_bundle(input_dir: str | Path) -> EngramDecodePlotBundle:
    input_path = Path(input_dir)
    manifest = _load_manifest(input_path)
    spec = manifest["inputs"]["engram_decode_metrics"]
    csv_path = _resolve_input_path(input_path, str(spec["path"]))
    return EngramDecodePlotBundle(
        metrics_df=_read_csv(csv_path, list(spec.get("required_columns", ()))),
    )


def build_accuracy_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer_name in LAYER_KEYS:
        part = metrics_df[metrics_df["layer"] == layer_name].sort_values("delay_ms")
        if len(part) == 0:
            continue
        x_vals = part["delay_ms"].to_numpy(dtype=float)
        y_vals = part["acc"].to_numpy(dtype=float)
        y_low = part["acc_ci_low"].to_numpy(dtype=float)
        y_high = part["acc_ci_high"].to_numpy(dtype=float)
        ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=LAYER_DISPLAY_NAMES[layer_name])
        ax.fill_between(x_vals, y_low, y_high, alpha=0.2)

    ax.axhline(CHANCE_LEVEL, color="k", linestyle="--", linewidth=1, label="Chance (10%)")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_title("Accuracy vs Delay")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig


def render_figures(bundle: EngramDecodePlotBundle, *, figures_dir: Path) -> dict[str, dict[str, str]]:
    fig = build_accuracy_figure(bundle.metrics_df)
    try:
        return {"accuracy_vs_delay": save_figure_all_formats(fig, figures_dir / "accuracy_vs_delay")}
    finally:
        plt.close(fig)


__all__ = [
    "DEFAULT_MANIFEST",
    "EngramDecodePlotBundle",
    "build_accuracy_figure",
    "load_plot_bundle",
    "render_figures",
    "write_plot_bundle_manifest",
]
