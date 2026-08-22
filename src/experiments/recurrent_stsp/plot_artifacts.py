"""Plot-only leaves for persisted recurrent STSP run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE
from src.plotting.paper_fig.typography import (
    VECTOR_TEXT_RCPARAMS,
    apply_paper_figure_typography,
)

from .artifact_plot_data import load_plot_inputs
from .recording import atomic_json_dump


_POPULATION_COLORS = (
    NATURE_COMPATIBLE_PALETTE["primary_navy"],
    NATURE_COMPATIBLE_PALETTE["comparison_coral"],
    NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
    NATURE_COMPATIBLE_PALETTE["fused_slate"],
    NATURE_COMPATIBLE_PALETTE["primary_cyan"],
)


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.8)
    axis.grid(False)


def _export(fig, prefix: Path) -> List[str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    for extension in ("png", "pdf", "svg"):
        path = prefix.with_suffix("." + extension)
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append(str(path.resolve()))
    plt.close(fig)
    return outputs


def _cue_windows(run_config: Dict[str, object]):
    protocol = run_config["protocol"]
    duration = float(protocol["cue_duration_ms"])
    total_time = float(protocol["total_time_ms"])
    windows = []
    for item in protocol["item_loading"]:
        start = float(item["origin_ms"])
        stop = min(start + duration, total_time)
        if start < total_time and stop > 0.0:
            windows.append((max(0.0, start), stop))
    return windows


def plot_spike_raster(
    spike_payload: Dict[str, object],
    run_config: Dict[str, object],
    output_prefix: Path,
    *,
    max_points: int = 1_000_000,
) -> Dict[str, object]:
    times = np.asarray(spike_payload["times_ms"])
    senders = np.asarray(spike_payload["sender_ids"])
    populations = np.asarray(spike_payload["population_ids"])
    original_points = times.size
    stride = max(1, (original_points + max_points - 1) // max_points)
    indices = np.arange(0, original_points, stride)
    times = times[indices]
    senders = senders[indices]
    populations = populations[indices]

    with plt.rc_context(VECTOR_TEXT_RCPARAMS):
        fig, axis = plt.subplots(figsize=(3.54, 2.45), constrained_layout=True)
        unique_populations = sorted(set(int(value) for value in populations))
        for population in unique_populations:
            mask = populations == population
            axis.scatter(
                times[mask],
                senders[mask],
                s=0.30,
                marker=".",
                linewidths=0.0,
                rasterized=True,
                color=_POPULATION_COLORS[population % len(_POPULATION_COLORS)],
                label="Pop. {}".format(population),
            )
        for cue_start, cue_stop in _cue_windows(run_config):
            axis.axvspan(
                cue_start,
                cue_stop,
                color=NATURE_COMPATIBLE_PALETTE["neutral_pale"],
                linewidth=0.0,
                zorder=-2,
            )
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Neuron ID")
        axis.set_xlim(0.0, float(run_config["protocol"]["total_time_ms"]))
        recorded_ids = np.asarray(spike_payload["recorded_neuron_ids"])
        if recorded_ids.size:
            axis.set_ylim(
                float(recorded_ids.min()) - 0.5, float(recorded_ids.max()) + 0.5
            )
        if unique_populations:
            axis.legend(frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)
        _style_axis(axis)
        apply_paper_figure_typography(fig)
        outputs = _export(fig, output_prefix)
    return {
        "source_points": original_points,
        "plotted_points": len(times),
        "deterministic_stride": stride,
        "outputs": outputs,
    }


def plot_stsp_ux(
    stsp_payload: Dict[str, object],
    run_config: Dict[str, object],
    output_prefix: Path,
) -> Dict[str, object]:
    times = np.asarray(stsp_payload["times_ms"])
    ux = np.asarray(stsp_payload["ux"])
    populations = np.asarray(stsp_payload["source_population_ids"])
    with plt.rc_context(VECTOR_TEXT_RCPARAMS):
        fig, axis = plt.subplots(figsize=(3.54, 2.45), constrained_layout=True)
        unique_populations = sorted(int(item) for item in np.unique(populations))
        for population in unique_populations:
            selected = populations == population
            if ux.shape[0] == 0 or not bool(selected.any()):
                continue
            mean_ux = ux[:, selected].mean(axis=1)
            axis.plot(
                times,
                mean_ux,
                color=_POPULATION_COLORS[population % len(_POPULATION_COLORS)],
                linestyle=("-", "--", "-.", ":")[population % 4],
                linewidth=1.1,
                label="Pop. {}".format(population),
            )
        for cue_start, cue_stop in _cue_windows(run_config):
            axis.axvspan(
                cue_start,
                cue_stop,
                color=NATURE_COMPATIBLE_PALETTE["neutral_pale"],
                linewidth=0.0,
                zorder=-2,
            )
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Mean $u x$")
        axis.set_xlim(0.0, float(run_config["protocol"]["total_time_ms"]))
        finite_ux = ux[np.isfinite(ux)]
        if finite_ux.size:
            lower = float(finite_ux.min())
            upper = float(finite_ux.max())
            padding = max(0.01, 0.05 * (upper - lower))
            axis.set_ylim(max(0.0, lower - padding), min(1.0, upper + padding))
        if unique_populations:
            axis.legend(frameon=False, ncol=2, handlelength=1.5)
        _style_axis(axis)
        apply_paper_figure_typography(fig)
        outputs = _export(fig, output_prefix)
    return {"probe_edges": populations.size, "outputs": outputs}


def plot_run_artifacts(
    run_directory: Path,
    *,
    output_directory: Optional[Path] = None,
    max_raster_points: int = 1_000_000,
) -> Dict[str, object]:
    """Read persisted outputs and render figures without launching the model."""

    run_directory = Path(run_directory)
    output_directory = output_directory or run_directory / "figures"
    spikes, stsp, run_config, dependencies = load_plot_inputs(run_directory)
    manifest = {
        "schema_version": 1,
        "plot_only": True,
        "dependencies": [
            *(str(path.resolve()) for path in dependencies),
        ],
        "raster": plot_spike_raster(
            spikes,
            run_config,
            output_directory / "spike_raster",
            max_points=max_raster_points,
        ),
        "stsp_ux": plot_stsp_ux(
            stsp, run_config, output_directory / "stsp_ux"
        ),
    }
    atomic_json_dump(manifest, output_directory / "plot_manifest.json")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render recurrent-STSP artifacts from persisted inputs only."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--max-raster-points", type=int, default=1_000_000)
    arguments = parser.parse_args(argv)
    plot_run_artifacts(
        arguments.run_directory,
        output_directory=arguments.output_directory,
        max_raster_points=arguments.max_raster_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["plot_run_artifacts", "plot_spike_raster", "plot_stsp_ux"]
