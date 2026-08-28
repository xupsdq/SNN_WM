from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FIGURE_PACKAGE_IDS = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6")


@dataclass(frozen=True)
class PaperFigureRegistry:
    fig_id: str
    experiment_id: str
    runner_module: str
    compatibility_module: str
    scope_tasks: Mapping[str, str]
    subexperiment_tasks: Mapping[str, str]
    archived_subexperiments: tuple[str, ...]
    main_subexperiments: tuple[str, ...]
    supplement_subexperiments: tuple[str, ...]

    def task_for_scope(self, scope: str) -> str:
        try:
            return str(self.scope_tasks[scope])
        except KeyError as exc:
            raise ValueError(f"Unsupported scope: {scope}") from exc


def load_registry_module(fig_id: str) -> Any:
    return importlib.import_module(f"src.experiments.paper_figures.{fig_id}.registry")


def load_figure_registry(fig_id: str) -> PaperFigureRegistry:
    module = load_registry_module(fig_id)
    return PaperFigureRegistry(
        fig_id=str(module.FIGURE_ID),
        experiment_id=str(module.EXPERIMENT_ID),
        runner_module=str(module.RUNNER_MODULE),
        compatibility_module=str(module.COMPATIBILITY_MODULE),
        scope_tasks={str(scope): str(task) for scope, task in module.SCOPE_TASKS.items()},
        subexperiment_tasks={
            str(name): str(task)
            for name, task in module.SUBEXPERIMENT_TASKS.items()
        },
        archived_subexperiments=tuple(str(name) for name in module.ARCHIVED_SUBEXPERIMENTS),
        main_subexperiments=tuple(str(name) for name in getattr(module, "MAIN_SUBEXPERIMENTS")),
        supplement_subexperiments=tuple(str(name) for name in getattr(module, "SUPPLEMENT_SUBEXPERIMENTS")),
    )


def load_all_figure_registries(fig_ids: Sequence[str] = FIGURE_PACKAGE_IDS) -> tuple[PaperFigureRegistry, ...]:
    return tuple(load_figure_registry(fig_id) for fig_id in fig_ids)


__all__ = [
    "FIGURE_PACKAGE_IDS",
    "PaperFigureRegistry",
    "load_all_figure_registries",
    "load_figure_registry",
    "load_registry_module",
]
