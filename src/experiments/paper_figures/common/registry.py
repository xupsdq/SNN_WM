from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FIGURE_PACKAGE_IDS = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6")


@dataclass(frozen=True)
class PaperFigureRegistry:
    fig_id: str
    experiment_id: str
    legacy_module: str
    subexperiment_flags: Mapping[str, tuple[str, ...]]
    main_subexperiments: tuple[str, ...]
    supplement_subexperiments: tuple[str, ...]
    both_scope_flags: tuple[str, ...]

    def flags_for_subexperiments(self, names: Sequence[str]) -> tuple[str, ...]:
        flags: list[str] = []
        for name in names:
            if name not in self.subexperiment_flags:
                raise ValueError(f"{self.fig_id}: unknown sub-experiment in scope list: {name}")
            flags.extend(self.subexperiment_flags[name])
        return tuple(flags)

    def flags_for_scope(self, scope: str) -> tuple[str, ...]:
        if scope == "main":
            return self.flags_for_subexperiments(self.main_subexperiments)
        if scope == "supplement":
            return self.flags_for_subexperiments(self.supplement_subexperiments)
        if scope == "both":
            return self.both_scope_flags
        raise ValueError(f"Unsupported scope: {scope}")


def load_registry_module(fig_id: str) -> Any:
    return importlib.import_module(f"src.experiments.paper_figures.{fig_id}.registry")


def load_figure_registry(fig_id: str) -> PaperFigureRegistry:
    module = load_registry_module(fig_id)
    return PaperFigureRegistry(
        fig_id=str(module.FIGURE_ID),
        experiment_id=str(module.EXPERIMENT_ID),
        legacy_module=str(module.LEGACY_MODULE),
        subexperiment_flags={
            str(name): tuple(str(flag) for flag in flags)
            for name, flags in module.SUBEXPERIMENT_FLAGS.items()
        },
        main_subexperiments=tuple(str(name) for name in getattr(module, "MAIN_SUBEXPERIMENTS")),
        supplement_subexperiments=tuple(str(name) for name in getattr(module, "SUPPLEMENT_SUBEXPERIMENTS")),
        both_scope_flags=tuple(str(flag) for flag in getattr(module, "BOTH_SCOPE_FLAGS", ("--run-all",))),
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
