from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.config.paths import DEFAULT_PATH_CONFIG

RESULT_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "baseline_processing": ("fig1_baseline_processing",),
    "silent_memory_effect": ("fig2_silent_memory_effect",),
    "external_input_interrogation": ("fig4_external_input_interrogation",),
    "engram_decode": ("engram_decode_experiment",),
    "ux_shuffle_memory_collapse": ("ux_shuffle_memory_collapse",),
    "ping_impulse_readout": ("ping_impulse_readout",),
    "dual_task_retention": ("dual_task_retention_experiment",),
    "dual_task_similarity_boundary": ("dual_task_similarity_boundary",),
    "silent_substrate_triplet": ("fig5_silent_substrate_triplet",),
    "causal_substrate_dissociation": ("fig5_causal_substrate_dissociation",),
}


@dataclass(frozen=True)
class PanelContext:
    figure_id: str
    panel_id: str
    experiment_key: str


class PanelDataError(RuntimeError):
    def __init__(
        self,
        *,
        figure_id: str,
        panel_id: str,
        experiment_key: str,
        detail: str,
    ) -> None:
        super().__init__(f"{figure_id} / {panel_id} / {experiment_key} / {detail}")
        self.figure_id = figure_id
        self.panel_id = panel_id
        self.experiment_key = experiment_key
        self.detail = detail


def default_results_root() -> Path:
    return DEFAULT_PATH_CONFIG.results_root.resolve()


def default_model_path() -> Path:
    return DEFAULT_PATH_CONFIG.model_path.resolve()


def default_dataset_root() -> Path:
    return DEFAULT_PATH_CONFIG.dataset_root.resolve()


def resolve_result_dir(results_root: Path | str, experiment_key: str) -> Path:
    results_root = Path(results_root).resolve()
    if experiment_key not in RESULT_DIR_ALIASES:
        raise KeyError(f"Unknown experiment key: {experiment_key}")

    aliases = RESULT_DIR_ALIASES[experiment_key]
    exact_matches = [results_root / alias for alias in aliases if (results_root / alias).is_dir()]
    if exact_matches:
        return exact_matches[0]

    prefix_matches = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        if any(_matches_alias(child.name, alias) for alias in aliases):
            prefix_matches.append(child.resolve())

    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        names = ", ".join(path.name for path in prefix_matches)
        raise RuntimeError(f"Ambiguous result directories for {experiment_key}: {names}")
    raise FileNotFoundError(f"No result directory found for {experiment_key}")


def ensure_result_dir(
    results_root: Path | str,
    experiment_key: str,
    *,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> Path:
    del model_path, dataset_root
    return resolve_result_dir(results_root, experiment_key)


def load_required_csv(
    results_root: Path | str,
    context: PanelContext,
    filename: str,
    required_columns: Iterable[str],
    *,
    ensure_materialized: bool = True,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> pd.DataFrame:
    result_dir = (
        ensure_result_dir(results_root, context.experiment_key, model_path=model_path, dataset_root=dataset_root)
        if ensure_materialized
        else resolve_result_dir(results_root, context.experiment_key)
    )
    csv_path = _resolve_result_file(result_dir, filename, preferred_subdir="data")
    if not csv_path.exists():
        raise PanelDataError(
            figure_id=context.figure_id,
            panel_id=context.panel_id,
            experiment_key=context.experiment_key,
            detail=f"{filename} / missing file",
        )

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise PanelDataError(
            figure_id=context.figure_id,
            panel_id=context.panel_id,
            experiment_key=context.experiment_key,
            detail=f"{filename} / missing {', '.join(missing_columns)}",
        )
    return df


def load_required_json(
    results_root: Path | str,
    context: PanelContext,
    filename: str,
    required_keys: Iterable[str] = (),
    *,
    ensure_materialized: bool = True,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> Mapping[str, object]:
    import json

    result_dir = (
        ensure_result_dir(results_root, context.experiment_key, model_path=model_path, dataset_root=dataset_root)
        if ensure_materialized
        else resolve_result_dir(results_root, context.experiment_key)
    )
    json_path = _resolve_result_file(result_dir, filename, preferred_subdir=None)
    if not json_path.exists():
        raise PanelDataError(
            figure_id=context.figure_id,
            panel_id=context.panel_id,
            experiment_key=context.experiment_key,
            detail=f"{filename} / missing file",
        )

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise PanelDataError(
            figure_id=context.figure_id,
            panel_id=context.panel_id,
            experiment_key=context.experiment_key,
            detail=f"{filename} / missing {', '.join(missing_keys)}",
        )
    return payload


def _matches_alias(name: str, alias: str) -> bool:
    return (
        name == alias
        or name.startswith(f"{alias}(")
        or name.startswith(f"{alias}_")
    )


def _resolve_result_file(result_dir: Path, filename: str, preferred_subdir: str | None) -> Path:
    candidates: list[Path] = []
    if preferred_subdir is not None:
        candidates.append(result_dir / preferred_subdir / filename)
    candidates.append(result_dir / filename)
    if preferred_subdir != "data":
        candidates.append(result_dir / "data" / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
