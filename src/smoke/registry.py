from __future__ import annotations

from dataclasses import dataclass

from src.experiments.catalog import EXPERIMENT_SPECS, ExperimentSpec


@dataclass(frozen=True)
class SmokeSpec:
    experiment_id: str
    runner_module: str
    plot_module: str
    expected_artifacts: tuple[str, ...]
    expected_plot_files: tuple[str, ...]


SMOKE_SPECS: dict[str, SmokeSpec] = {
    experiment_id: SmokeSpec(
        experiment_id=experiment_id,
        runner_module=f"src.experiments.runners.{experiment_id}",
        plot_module=f"src.plotting.experiments.{experiment_id}_plot",
        expected_artifacts=spec.expected_artifacts,
        expected_plot_files=("figure_main.png", "figure_main.pdf", "figure_main.svg"),
    )
    for experiment_id, spec in EXPERIMENT_SPECS.items()
}


def get_smoke_spec(experiment_id: str) -> SmokeSpec:
    return SMOKE_SPECS[experiment_id]


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    return EXPERIMENT_SPECS[experiment_id]


__all__ = ["SMOKE_SPECS", "SmokeSpec", "get_experiment_spec", "get_smoke_spec"]
