from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any, Sequence

from src.experiments.paper_figures.common.registry import load_registry_module


def main_for_legacy_module(fig_id: str, argv: Sequence[str] | None = None) -> int:
    """Translate one supported legacy selector to the current figure task runner."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    registry = load_registry_module(fig_id)
    if any(value in {"-h", "--help"} for value in raw_args):
        _print_help(fig_id, registry)
        return 0

    dry_run = "--dry-run" in raw_args
    raw_args = [value for value in raw_args if value != "--dry-run"]
    if "--run-all" in raw_args:
        raise SystemExit(
            f"strict archive: archived legacy selector '--run-all' for {fig_id}; "
            f"use python -m {registry.RUNNER_MODULE} --task both_scope instead"
        )

    selected = _selected_subexperiments(raw_args, registry)
    archived = [name for name in selected if name in set(registry.ARCHIVED_SUBEXPERIMENTS)]
    if archived:
        names = ", ".join(archived)
        raise SystemExit(
            f"strict archive: archived subexperiment(s) {names} are outside the current {fig_id} DAG"
        )
    unknown_run_flags = sorted(
        value
        for value in raw_args
        if value.startswith("--run-") and value not in _selector_flags(registry)
    )
    if unknown_run_flags:
        raise SystemExit(f"strict archive: unknown or archived legacy selector(s): {', '.join(unknown_run_flags)}")
    if not selected:
        raise SystemExit(
            f"strict archive: no supported legacy selector was provided for {fig_id}; "
            f"use python -m {registry.RUNNER_MODULE} --task <task>"
        )

    task = _task_for_selection(selected, registry)
    selector_flags = _selector_flags(registry)
    consumed_flags = selector_flags | set(getattr(registry, "LEGACY_NOOP_FLAGS", ()))
    forwarded = [value for value in raw_args if value not in consumed_flags]
    forwarded.extend(["--task", task])
    if dry_run:
        print(subprocess.list2cmdline([sys.executable, "-m", str(registry.RUNNER_MODULE), *forwarded]))
        return 0
    runner = importlib.import_module(str(registry.RUNNER_MODULE))
    return int(runner.main(forwarded) or 0)


def _selected_subexperiments(argv: Sequence[str], registry: Any) -> list[str]:
    selected: list[str] = []
    values = set(argv)
    for name, flags in registry.SUBEXPERIMENT_FLAGS.items():
        selection_flag = next((str(flag) for flag in flags if str(flag).startswith("--run-")), None)
        if selection_flag in values:
            selected.append(str(name))
    return selected


def _selector_flags(registry: Any) -> set[str]:
    return {
        str(flag)
        for flags in registry.SUBEXPERIMENT_FLAGS.values()
        for flag in flags
        if str(flag).startswith("--run-")
    }


def _task_for_selection(selected: Sequence[str], registry: Any) -> str:
    selected_set = set(selected)
    main_set = set(registry.MAIN_SUBEXPERIMENTS)
    supplement_set = set(registry.SUPPLEMENT_SUBEXPERIMENTS)
    both_set = main_set | supplement_set
    if selected_set == both_set:
        return str(registry.SCOPE_TASKS["both"])
    if selected_set == main_set:
        return str(registry.SCOPE_TASKS["main"])
    if selected_set == supplement_set:
        return str(registry.SCOPE_TASKS["supplement"])
    if len(selected) != 1:
        names = ", ".join(selected)
        raise SystemExit(
            f"strict archive: legacy multi-selector combination is not a current DAG scope: {names}; "
            f"run explicit tasks through {registry.RUNNER_MODULE}"
        )
    name = str(selected[0])
    try:
        return str(registry.SUBEXPERIMENT_TASKS[name])
    except KeyError as exc:
        raise SystemExit(f"strict archive: archived subexperiment {name} is outside the current DAG") from exc


def _print_help(fig_id: str, registry: Any) -> None:
    supported = ", ".join(sorted(registry.SUBEXPERIMENT_TASKS))
    archived = ", ".join(sorted(registry.ARCHIVED_SUBEXPERIMENTS)) or "none"
    print(f"Legacy CLI Adapter for {fig_id}.")
    print(f"Current runner: python -m {registry.RUNNER_MODULE} --task <task>")
    print(f"Supported legacy selectors: {supported}")
    print(f"Strictly archived selectors: {archived}")


__all__ = ["main_for_legacy_module"]
