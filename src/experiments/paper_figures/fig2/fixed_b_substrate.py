from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.config.units import ms
from src.core.network import SDNN_Network
from src.data.encoding import DoGSpikeEncoder
from src.experiments.common.dataset import build_class_index
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.fig2.constants import FIGURE_ID, NUM_CLASSES
from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    load_fixed_b_artifact,
)
from src.experiments.paper_figures.fig2.output import (
    prepare_dirs,
    seed_output_dir,
    utc_now,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import (
    _encode_source_rows,
    _history_rows_at_k,
    _load_boundary,
    _run_branch,
    _simulate_history_rows,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext, Fig2Config
from src.experiments.paper_figures.run_paper_figures import discover_checkpoints


@dataclass(frozen=True)
class PairedHistorySlice:
    rows: pd.DataFrame
    boundary: Mapping[str, Mapping[str, np.ndarray]]
    donor_indices: np.ndarray
    family_ids: tuple[int, ...]
    current_time: int


def resolve_fixed_b_model_path(
    model_path: str | None,
    model_path_glob: str,
    network_seed: int,
    *,
    smoke: bool,
) -> Path:
    if model_path:
        return _resolve_repo_path(model_path)
    try:
        checkpoints = discover_checkpoints(str(model_path_glob))
    except FileNotFoundError:
        if smoke:
            return _resolve_repo_path("results/missing_fig2_smoke_model.pth")
        raise
    by_seed = {int(item.seed): item.model_path for item in checkpoints}
    if int(network_seed) not in by_seed:
        if smoke:
            return _resolve_repo_path("results/missing_fig2_smoke_model.pth")
        known = ", ".join(str(seed) for seed in sorted(by_seed))
        raise FileNotFoundError(
            f"No checkpoint for network seed {network_seed} matched --model-path-glob. Known seeds: {known}"
        )
    return by_seed[int(network_seed)]


def build_fixed_b_context(
    cfg: Fig2Config,
    *,
    load_model: bool = True,
) -> ExperimentContext:
    seed_everything(int(cfg.network_seed))
    seed_dir = seed_output_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    max_duration = max(cfg.sample_ms, cfg.second_item_ms, cfg.weak_probe_ms, 100)
    warnings: list[str] = []
    if not load_model:
        net = None
        encoder = None
    elif Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max_duration,
        )
    elif cfg.smoke:
        seed_everything(int(cfg.network_seed))
        net = SDNN_Network(device=str(device)).to(device)
        net.eval()
        encoder = DoGSpikeEncoder(
            dt=cfg.dt,
            max_duration=max_duration * ms,
            device=str(device),
        )
        warnings.append(
            "Model checkpoint missing; smoke mode used an untrained repo SDNN_Network instance. "
            "Functional outputs are real network rollouts, but are not manuscript evidence."
        )
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")
    return ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=warnings,
        output_files={},
        completed_modules={},
        run_log=[
            f"{utc_now()} start {FIGURE_ID} task runner seed={cfg.network_seed} smoke={cfg.smoke} "
            f"model_loaded={bool(load_model)}"
        ],
    )


def load_fixed_b_parent(task_dir: str | Path, *, task_id: str) -> FixedBArtifact:
    task_dir = Path(task_dir)
    cache_path = task_dir / "cache_key.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Required parent cache key is missing: {cache_path}")
    wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
    expected = wrapper.get("cache_key") if isinstance(wrapper, dict) else None
    if not isinstance(expected, dict):
        raise RuntimeError(f"Unreadable cache key at {task_dir}")
    if str(expected.get("task_id")) != str(task_id):
        raise RuntimeError(f"Parent task/cache-key mismatch at {task_dir}")
    return load_fixed_b_artifact(task_dir, expected, task_id=task_id)


def load_paired_history_slice(
    histories: FixedBArtifact,
    *,
    prefix_k: int,
    max_families: int,
) -> PairedHistorySlice:
    rows_at_k = _history_rows_at_k(histories.tables["history_specs"], int(prefix_k))
    selected = rows_at_k.loc[rows_at_k["history_condition"].isin(("A", "C"))].copy()
    families = sorted(int(value) for value in selected["history_family_id"].unique())[
        : int(max_families)
    ]
    selected = selected.loc[selected["history_family_id"].isin(families)].copy()
    _validate_paired_histories(selected)
    row_indices = [int(value) for value in selected.index]
    selected = selected.reset_index(drop=True)
    elapsed = sorted(int(value) for value in selected["elapsed_steps"].unique())
    if len(elapsed) != 1:
        raise RuntimeError(f"Non-unique elapsed_steps for K={prefix_k}: {elapsed}")
    return PairedHistorySlice(
        rows=selected,
        boundary=_load_boundary(histories, int(prefix_k), row_indices=row_indices),
        donor_indices=_paired_history_indices(selected),
        family_ids=tuple(families),
        current_time=int(elapsed[0]),
    )


def encode_fixed_b_sources(
    ctx: ExperimentContext,
    rows: pd.DataFrame,
    *,
    image_column: str,
    seed_column: str,
    steps: int,
) -> np.ndarray:
    return _encode_source_rows(
        ctx,
        rows,
        image_column=image_column,
        seed_column=seed_column,
        steps=steps,
    )


def simulate_fixed_b_histories(
    ctx: ExperimentContext,
    history_specs: pd.DataFrame,
    inputs: FixedBArtifact,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    return _simulate_history_rows(ctx, history_specs, inputs)


def run_fixed_b_branch(
    ctx: ExperimentContext,
    *,
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    input_seq: torch.Tensor,
    current_time: int,
    restore_mode: str,
    branch: str,
    replay_l1_pooled: np.ndarray | None,
    capture_l1_pooled: bool,
    capture_strong_path: bool,
    random_seed: int,
    capture_layer2_presynaptic_trace: bool = False,
    net_override: Any | None = None,
    skip_restore: bool = False,
) -> dict[str, np.ndarray]:
    return _run_branch(
        ctx,
        boundary=boundary,
        input_seq=input_seq,
        current_time=current_time,
        restore_mode=restore_mode,
        branch=branch,
        replay_l1_pooled=replay_l1_pooled,
        capture_l1_pooled=capture_l1_pooled,
        capture_strong_path=capture_strong_path,
        random_seed=random_seed,
        capture_layer2_presynaptic_trace=capture_layer2_presynaptic_trace,
        net_override=net_override,
        skip_restore=skip_restore,
    )


def _validate_paired_histories(selected: pd.DataFrame) -> None:
    if selected.empty:
        raise RuntimeError("No A/C histories were selected")
    counts = selected.groupby(["history_family_id", "history_condition"]).size().unstack(fill_value=0)
    if set(counts.columns) != {"A", "C"} or not counts.eq(1).all().all():
        raise RuntimeError("Every selected history family must contain exactly one A and one C row")


def _paired_history_indices(selected: pd.DataFrame) -> np.ndarray:
    lookup = {
        (int(row.history_family_id), str(row.history_condition)): int(index)
        for index, row in enumerate(selected.itertuples(index=False))
    }
    return np.asarray(
        [
            lookup[
                (
                    int(row.history_family_id),
                    "C" if str(row.history_condition) == "A" else "A",
                )
            ]
            for row in selected.itertuples(index=False)
        ],
        dtype=np.int64,
    )


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve()


__all__ = [
    "PairedHistorySlice",
    "build_fixed_b_context",
    "encode_fixed_b_sources",
    "load_fixed_b_parent",
    "load_paired_history_slice",
    "resolve_fixed_b_model_path",
    "run_fixed_b_branch",
    "simulate_fixed_b_histories",
]
