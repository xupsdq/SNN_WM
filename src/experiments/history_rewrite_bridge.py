from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.common.monitored_dms import (
    build_layer_input_shapes,
    snapshot_boundary_state,
)
from src.experiments.common.results import (
    save_log_lines,
    save_run_config,
    save_summary_json,
)
from src.experiments.common.run_info import (
    build_run_info,
    finalize_run_info,
    write_run_info,
)
from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    load_fixed_b_artifact,
)
from src.experiments.paper_figures.fig2.schemas import (
    TASK_FIXED_B_HISTORY_BANK,
    TASK_FIXED_B_INPUT_BANK,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import (
    FAST_STATE_KEYS,
    STSP_STATE_KEYS,
    _history_rows_at_k,
    _load_boundary,
    _restore_boundary,
    _run_branch,
)
from src.experiments.paper_figures.fig2.types import Fig2Config
from src.experiments.paper_figures.fig2.run_task import (
    _build_context,
    _resolve_model_path,
)
from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
)


EXPERIMENT_ID = "history_rewrite_bridge"
DEVELOPMENT_SEED = 1000
INFERENCE_SEEDS = tuple(range(1001, 1020))
PREFIX_TO_PROGRESSIVE_STAGE = {1: 2, 5: 6}
PRIMARY_ENDPOINTS = (
    "layer1_to_layer2_update_donor_transfer",
    "layer1_to_early_class_score_donor_transfer",
)
ALL_BOUNDARY_KEYS = FAST_STATE_KEYS + STSP_STATE_KEYS
NEAR_ZERO = 1e-12


@dataclass(frozen=True)
class BridgeConfig:
    output_dir: str = "results/paper_figure_multi_seed/history_rewrite_bridge"
    parent_root: str = (
        "results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory"
    )
    fixed_b_aggregate_root: str = (
        "results/paper_figure_multi_seed/"
        "fig2_fixed_b_mechanism_confirmatory/aggregate"
    )
    progressive_root: str = (
        "results/paper_figure_multi_seed/fig3_multiitem_peak_landscape"
    )
    dataset_root: str = DEFAULT_DATASET_ROOT
    model_path_glob: str = DEFAULT_MODEL_PATH_GLOB
    device: str = "auto"
    prefixes: tuple[int, ...] = (1, 5)
    anchors_per_chunk: int = 5
    max_anchors: int = 0
    max_history_families: int = 0
    smoke: bool = False


def run_boundary_analysis(
    cfg: BridgeConfig,
    *,
    seeds: Sequence[int] = INFERENCE_SEEDS,
    command: str | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    output_root = _resolve(repo_root, cfg.output_dir)
    analysis_root = output_root / "boundary_analysis"
    dirs = _prepare_bundle_dirs(analysis_root)
    source_records: list[dict[str, Any]] = []
    logs: list[str] = []
    run_info = build_run_info(
        experiment_name=f"{EXPERIMENT_ID}.boundary_analysis",
        output_dir=analysis_root,
        entry_script="src.experiments.runners.history_rewrite_bridge",
        seed=20260727,
        dataset=str(_resolve(repo_root, cfg.progressive_root)),
        command=command,
    )
    write_run_info(dirs["meta"], run_info)
    try:
        identity_rows, checkpoint_rows = _collect_existing_identity_audit(
            cfg,
            repo_root=repo_root,
            seeds=seeds,
            source_records=source_records,
        )
        _write_csv(dirs["metrics"] / "boundary_identity_audit.csv", identity_rows)
        _write_csv(dirs["metrics"] / "checkpoint_mapping.csv", checkpoint_rows)

        displacement = _collect_progressive_boundary_scalars(
            cfg,
            repo_root=repo_root,
            seeds=seeds,
            source_records=source_records,
        )
        _write_csv(
            dirs["metrics"] / "boundary_displacement_network_scalars.csv",
            displacement,
        )

        fixed_scalars_path = (
            _resolve(repo_root, cfg.fixed_b_aggregate_root)
            / "fixed_b_confirmatory_network_scalars.csv"
        )
        fixed_scalars = pd.read_csv(fixed_scalars_path)
        _record_source(
            source_records,
            fixed_scalars_path,
            bundle="fixed_b_confirmatory",
            network_seed=-1,
        )
        fixed_scalars = fixed_scalars.loc[
            fixed_scalars["network_seed"].astype(int).isin(tuple(int(v) for v in seeds))
        ].copy()
        inference, conjunction = _unified_boundary_inference(
            displacement,
            fixed_scalars,
            seeds=tuple(int(v) for v in seeds),
        )
        _write_csv(dirs["metrics"] / "boundary_transition_inference.csv", inference)
        _write_csv(
            dirs["metrics"] / "boundary_transition_conjunction.csv",
            conjunction,
        )

        source_manifest = pd.DataFrame(source_records).sort_values(
            ["bundle", "network_seed", "path"], kind="stable"
        )
        _write_csv(dirs["meta"] / "source_manifest.csv", source_manifest)

        identity_summary = {
            "n_rows": int(len(identity_rows)),
            "n_networks": int(identity_rows["network_seed"].nunique()),
            "max_normalized_restoration_error": float(
                identity_rows["normalized_restoration_error"].max()
            ),
            "max_abs_layer2_ux_error": float(
                identity_rows["max_abs_layer2_ux_error"].max()
            ),
            "all_prediction_equal": bool(
                identity_rows["prediction_equal"].eq(1).all()
            ),
            "all_spike_counts_equal": bool(
                identity_rows["spike_counts_equal"].eq(1).all()
            ),
            "all_restoration_gates_pass": bool(
                identity_rows["restoration_margin_pass"].eq(1).all()
            ),
        }
        conjunction_pass = bool(conjunction["conjunction_pass"].eq(1).all())
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "component": "boundary_analysis",
            "status": "completed",
            "promotion_status": "not_promoted",
            "seeds": [int(v) for v in seeds],
            "identity": identity_summary,
            "conjunction_pass_at_both_depths": conjunction_pass,
            "claim_boundary": (
                "Post-hoc network-level bridge analysis of existing outputs; "
                "it is not a new untouched confirmatory cohort."
            ),
            "output_files": {
                "boundary_identity_audit": "data/metrics/boundary_identity_audit.csv",
                "checkpoint_mapping": "data/metrics/checkpoint_mapping.csv",
                "boundary_displacement_network_scalars": (
                    "data/metrics/boundary_displacement_network_scalars.csv"
                ),
                "boundary_transition_inference": (
                    "data/metrics/boundary_transition_inference.csv"
                ),
                "boundary_transition_conjunction": (
                    "data/metrics/boundary_transition_conjunction.csv"
                ),
                "source_manifest": "meta/source_manifest.csv",
            },
        }
        save_run_config(asdict(cfg), analysis_root)
        save_summary_json(summary, analysis_root)
        logs.append(
            f"identity rows={len(identity_rows)} conjunction_pass={conjunction_pass}"
        )
        save_log_lines(logs, dirs["logs"])
        finalize_run_info(dirs["meta"], run_info, status="completed")
        _write_artifact_manifest(
            analysis_root,
            title="Boundary identity and transition analysis",
        )
        return summary
    except Exception:
        logs.append("boundary analysis failed")
        save_log_lines(logs, dirs["logs"])
        finalize_run_info(dirs["meta"], run_info, status="failed")
        raise


def run_bridge_seed(
    cfg: BridgeConfig,
    *,
    network_seed: int,
    command: str | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    output_root = _resolve(repo_root, cfg.output_dir)
    bridge_root = output_root / "bridge"
    parent_seed_root = (
        _resolve(repo_root, cfg.parent_root)
        / f"seed_{int(network_seed)}"
        / "data"
        / "intermediates"
    )
    input_dir = parent_seed_root / TASK_FIXED_B_INPUT_BANK
    history_dir = parent_seed_root / TASK_FIXED_B_HISTORY_BANK
    before_hashes = _parent_file_hashes(
        {
            TASK_FIXED_B_INPUT_BANK: input_dir,
            TASK_FIXED_B_HISTORY_BANK: history_dir,
        }
    )
    inputs = _load_parent(input_dir, TASK_FIXED_B_INPUT_BANK)
    histories = _load_parent(history_dir, TASK_FIXED_B_HISTORY_BANK)

    model_path = _resolve_model_path(
        None,
        str(cfg.model_path_glob),
        int(network_seed),
        smoke=False,
    )
    fig_cfg = Fig2Config(
        model_path=str(model_path),
        dataset_root=str(_resolve(repo_root, cfg.dataset_root)),
        output_root=str(bridge_root),
        network_seed=int(network_seed),
        device=str(cfg.device),
        fixed_b_prefix_depths=tuple(int(v) for v in cfg.prefixes),
        smoke=False,
    )
    ctx = _build_context(fig_cfg, load_model=True)
    dirs = _prepare_bundle_dirs(ctx.seed_dir)
    run_info = build_run_info(
        experiment_name=f"{EXPERIMENT_ID}.seed",
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.runners.history_rewrite_bridge",
        seed=int(network_seed),
        dataset=str(cfg.dataset_root),
        command=command,
        model_path=str(model_path),
    )
    write_run_info(dirs["meta"], run_info)
    logs: list[str] = []
    try:
        c_map = build_c_anchor_mapping(histories.tables["b_anchor_specs"])
        _write_csv(dirs["trial_specs"] / "c_anchor_mapping.csv", c_map)
        cell_frames: list[pd.DataFrame] = []
        identity_frames: list[pd.DataFrame] = []
        for prefix_k in tuple(int(v) for v in cfg.prefixes):
            cells, identity = _run_prefix_bridge(
                ctx,
                inputs=inputs,
                histories=histories,
                c_map=c_map,
                prefix_k=prefix_k,
                anchors_per_chunk=max(1, int(cfg.anchors_per_chunk)),
                max_anchors=int(cfg.max_anchors),
                max_history_families=int(cfg.max_history_families),
            )
            cell_frames.append(cells)
            identity_frames.append(identity)
            logs.append(
                f"prefix={prefix_k} cells={len(cells)} identity_rows={len(identity)}"
            )
        cells = pd.concat(cell_frames, ignore_index=True)
        identity = pd.concat(identity_frames, ignore_index=True)
        network_scalars = _bridge_network_scalars(cells, identity)
        _write_csv(dirs["metrics"] / "bridge_cell_metrics.csv", cells)
        _write_csv(dirs["metrics"] / "bridge_checkpoint_identity.csv", identity)
        _write_csv(dirs["metrics"] / "bridge_network_scalars.csv", network_scalars)

        after_hashes = _parent_file_hashes(
            {
                TASK_FIXED_B_INPUT_BANK: input_dir,
                TASK_FIXED_B_HISTORY_BANK: history_dir,
            }
        )
        parent_audit = before_hashes.merge(
            after_hashes,
            on=["parent_task", "relative_file"],
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        parent_audit["unchanged"] = (
            parent_audit["sha256_before"].eq(parent_audit["sha256_after"])
            & parent_audit["size_bytes_before"].eq(
                parent_audit["size_bytes_after"]
            )
        ).astype(int)
        _write_csv(dirs["meta"] / "parent_hash_audit.csv", parent_audit)

        summary = {
            "experiment_id": EXPERIMENT_ID,
            "component": "bridge_seed",
            "status": "completed",
            "promotion_status": "not_promoted",
            "network_seed": int(network_seed),
            "seed_role": (
                "development_engineering"
                if int(network_seed) == DEVELOPMENT_SEED
                else "frozen_protocol_inference_network"
            ),
            "n_cells": int(len(cells)),
            "prefixes": [int(v) for v in sorted(cells["prefix_k"].unique())],
            "n_history_families": int(cells["history_family_id"].nunique()),
            "n_b_anchors": int(cells["b_anchor_id"].nunique()),
            "all_parent_files_unchanged": bool(
                parent_audit["unchanged"].eq(1).all()
            ),
            "all_checkpoint_identity_gates_pass": bool(
                identity["identity_pass"].eq(1).all()
            ),
            "minimum_primary_coverage": float(
                network_scalars[
                    [
                        "layer2_update_valid_coverage",
                        "early_class_score_valid_coverage",
                    ]
                ].min().min()
            ),
            "output_files": {
                "bridge_cell_metrics": "data/metrics/bridge_cell_metrics.csv",
                "bridge_checkpoint_identity": (
                    "data/metrics/bridge_checkpoint_identity.csv"
                ),
                "bridge_network_scalars": (
                    "data/metrics/bridge_network_scalars.csv"
                ),
                "c_anchor_mapping": "data/trial_specs/c_anchor_mapping.csv",
                "parent_hash_audit": "meta/parent_hash_audit.csv",
            },
        }
        save_run_config(asdict(cfg), ctx.seed_dir)
        save_summary_json(summary, ctx.seed_dir)
        save_log_lines(logs, dirs["logs"])
        finalize_run_info(dirs["meta"], run_info, status="completed")
        _write_artifact_manifest(
            ctx.seed_dir,
            title="Post-B/passive to same-C bridge",
        )
        return summary
    except Exception:
        logs.append("bridge seed failed")
        save_log_lines(logs, dirs["logs"])
        finalize_run_info(dirs["meta"], run_info, status="failed")
        raise


def aggregate_bridge_cohort(
    cfg: BridgeConfig,
    *,
    seeds: Sequence[int] = INFERENCE_SEEDS,
    command: str | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    output_root = _resolve(repo_root, cfg.output_dir)
    bridge_root = output_root / "bridge"
    aggregate_root = bridge_root / "aggregate"
    dirs = _prepare_bundle_dirs(aggregate_root)
    run_info = build_run_info(
        experiment_name=f"{EXPERIMENT_ID}.aggregate",
        output_dir=aggregate_root,
        entry_script="src.experiments.runners.history_rewrite_bridge",
        seed=20260727,
        dataset=str(bridge_root),
        command=command,
    )
    write_run_info(dirs["meta"], run_info)
    tables = []
    seed_summaries = []
    for seed in tuple(int(v) for v in seeds):
        seed_root = bridge_root / f"seed_{seed}"
        scalar_path = seed_root / "data" / "metrics" / "bridge_network_scalars.csv"
        summary_path = seed_root / "summary.json"
        if not scalar_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"Missing bridge output for seed {seed}: {scalar_path}"
            )
        tables.append(pd.read_csv(scalar_path))
        seed_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    scalars = pd.concat(tables, ignore_index=True)
    expected = set(int(v) for v in seeds)
    found = set(scalars["network_seed"].astype(int))
    if found != expected:
        raise RuntimeError(
            f"Bridge cohort mismatch: expected={sorted(expected)}, found={sorted(found)}"
        )
    inference = _bridge_inference(scalars)
    engineering_pass = bool(
        all(
            bool(item["all_parent_files_unchanged"])
            and bool(item["all_checkpoint_identity_gates_pass"])
            and float(item["minimum_primary_coverage"]) >= 0.95
            for item in seed_summaries
        )
    )
    primary_pass = bool(
        len(inference) == 4
        and inference["mean"].gt(0).all()
        and inference["holm_adjusted_p"].lt(0.05).all()
    )
    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "promotion_status": "not_promoted",
        "n_networks": int(scalars["network_seed"].nunique()),
        "network_seeds": sorted(int(v) for v in found),
        "inference_unit": "independently_trained_network",
        "test": "exact_one_sided_sign_flip",
        "multiplicity": "Holm_across_4_prespecified_primary_effects",
        "engineering_pass": engineering_pass,
        "all_primary_effects_positive_and_holm_significant": primary_pass,
        "bridge_core_pass": bool(engineering_pass and primary_pass),
        "verdict": (
            "bridge_core_pass"
            if engineering_pass and primary_pass
            else "bridge_core_fail"
        ),
        "claim_boundary": (
            "Directly tests whether the post-B Layer1 u/x component biases "
            "same-C processing at K=1 and K=5; no promotion is implied."
        ),
    }
    _write_csv(dirs["metrics"] / "bridge_cohort_network_scalars.csv", scalars)
    _write_csv(dirs["metrics"] / "bridge_cohort_inference.csv", inference)
    (aggregate_root / "bridge_cohort_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_summary_json(verdict, aggregate_root)
    finalize_run_info(dirs["meta"], run_info, status="completed")
    _write_artifact_manifest(aggregate_root, title="History-rewrite bridge cohort")
    return verdict


def build_c_anchor_mapping(b_specs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "b_anchor_id",
        "B_image_id",
        "B_label",
        "B_replicate_id",
    }
    missing = sorted(required.difference(b_specs.columns))
    if missing:
        raise ValueError(f"b_anchor_specs missing columns: {missing}")
    specs = b_specs.sort_values("b_anchor_id").reset_index(drop=True)
    lookup = {
        (int(row.B_label), int(row.B_replicate_id)): row
        for row in specs.itertuples(index=False)
    }
    rows = []
    for row in specs.itertuples(index=False):
        target_label = (int(row.B_label) + 1) % 10
        key = (target_label, int(row.B_replicate_id))
        if key not in lookup:
            raise ValueError(f"Missing deterministic C anchor for key={key}")
        target = lookup[key]
        rows.append(
            {
                "b_anchor_id": int(row.b_anchor_id),
                "B_image_id": int(row.B_image_id),
                "B_label": int(row.B_label),
                "B_replicate_id": int(row.B_replicate_id),
                "c_anchor_id": int(target.b_anchor_id),
                "C_image_id": int(target.B_image_id),
                "C_label": int(target.B_label),
                "mapping_rule": "cyclic_next_class_same_replicate",
            }
        )
    mapping = pd.DataFrame(rows)
    if mapping["c_anchor_id"].nunique() != len(mapping):
        raise RuntimeError("C anchor mapping is not one-to-one")
    if mapping["B_label"].eq(mapping["C_label"]).any():
        raise RuntimeError("C anchor mapping contains same-label pairs")
    return mapping


def donor_transfer(
    swap: np.ndarray,
    receiver: np.ndarray,
    donor: np.ndarray,
    *,
    eps: float = NEAR_ZERO,
) -> tuple[np.ndarray, np.ndarray]:
    swap_array = np.asarray(swap, dtype=np.float32)
    receiver_array = np.asarray(receiver, dtype=np.float32)
    donor_array = np.asarray(donor, dtype=np.float32)
    if (
        swap_array.shape != receiver_array.shape
        or donor_array.shape != receiver_array.shape
        or receiver_array.ndim < 2
    ):
        raise ValueError(
            "donor_transfer expects equally shaped arrays with one row per cell"
        )
    donor_delta = (donor_array - receiver_array).reshape(len(receiver_array), -1)
    swap_delta = (swap_array - receiver_array).reshape(len(receiver_array), -1)
    denominator = np.sum(
        donor_delta * donor_delta,
        axis=1,
        dtype=np.float64,
    )
    numerator = np.sum(
        swap_delta * donor_delta,
        axis=1,
        dtype=np.float64,
    )
    valid = denominator > float(eps)
    values = np.full(len(receiver_array), np.nan, dtype=np.float64)
    values[valid] = numerator[valid] / denominator[valid]
    return values, valid


def exact_one_sided_sign_flip_p(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan")
    if len(data) > 24:
        raise ValueError(
            "Exact sign-flip enumeration is intentionally bounded at 24 units"
        )
    observed_sum = float(data.sum())
    tolerance = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.abs(data).sum()))
        * 32.0
    )
    total = 1 << len(data)
    count = 0
    bit_positions = np.arange(len(data), dtype=np.uint64)
    chunk_size = 1 << min(16, len(data))
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = (
            (
                (masks >> bit_positions[None, :])
                & np.uint64(1)
            ).astype(np.float64)
            * 2.0
            - 1.0
        )
        signed_sums = signs @ data
        count += int(
            np.count_nonzero(signed_sums >= observed_sum - tolerance)
        )
    return float(count / total)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("Holm adjustment expects a one-dimensional array")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    total = len(p)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(p[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def mix_layer1_stsp(
    donor: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    receiver: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for layer in LAYER_KEYS:
        output[layer] = {}
        for state, value in receiver[layer].items():
            source = (
                donor[layer][state]
                if layer == "layer1" and state in STSP_STATE_KEYS
                else value
            )
            output[layer][state] = _to_numpy(source).copy()
    return output


def _collect_existing_identity_audit(
    cfg: BridgeConfig,
    *,
    repo_root: Path,
    seeds: Sequence[int],
    source_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_frames: list[pd.DataFrame] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for seed in tuple(int(value) for value in seeds):
        parent_root = (
            _resolve(repo_root, cfg.parent_root)
            / f"seed_{seed}"
            / "data"
            / "intermediates"
        )
        input_dir = parent_root / TASK_FIXED_B_INPUT_BANK
        history_dir = parent_root / TASK_FIXED_B_HISTORY_BANK
        inputs = _load_parent(input_dir, TASK_FIXED_B_INPUT_BANK)
        histories = _load_parent(history_dir, TASK_FIXED_B_HISTORY_BANK)
        _record_source(
            source_records,
            input_dir / "manifest.csv",
            bundle=TASK_FIXED_B_INPUT_BANK,
            network_seed=seed,
        )
        _record_source(
            source_records,
            history_dir / "manifest.csv",
            bundle=TASK_FIXED_B_HISTORY_BANK,
            network_seed=seed,
        )
        restoration = histories.tables.get("restoration_audit")
        if restoration is None:
            raise KeyError(
                f"{history_dir} does not contain restoration_audit"
            )
        required = {
            "network_seed",
            "prefix_k",
            "normalized_restoration_error",
            "max_abs_layer2_ux_error",
            "prediction_equal",
            "spike_counts_equal",
            "restoration_margin_pass",
        }
        missing = sorted(required.difference(restoration.columns))
        if missing:
            raise ValueError(
                f"restoration_audit for seed {seed} is missing {missing}"
            )
        audit_frames.append(restoration.copy())
        history_specs = histories.tables["history_specs"]
        for prefix_k in sorted(
            int(value) for value in history_specs["prefix_k"].unique()
        ):
            elapsed = sorted(
                int(value)
                for value in history_specs.loc[
                    history_specs["prefix_k"].eq(prefix_k),
                    "elapsed_steps",
                ].unique()
            )
            if len(elapsed) != 1:
                raise RuntimeError(
                    f"Non-unique history elapsed steps at seed={seed}, "
                    f"K={prefix_k}: {elapsed}"
                )
            checkpoint_rows.append(
                {
                    "network_seed": seed,
                    "prefix_k": prefix_k,
                    "history_boundary_elapsed_steps": elapsed[0],
                    "history_boundary_definition": (
                        "after each history item and its matched 200-step delay"
                    ),
                    "post_B_elapsed_steps": elapsed[0] + 400,
                    "post_passive_elapsed_steps": elapsed[0] + 400,
                    "C_early_elapsed_steps": elapsed[0] + 420,
                    "C_end_elapsed_steps": elapsed[0] + 600,
                    "C_post_elapsed_steps": elapsed[0] + 800,
                    "fixed_B_anchor_count": int(
                        len(inputs.tables["input_manifest"])
                    ),
                    "history_row_count": int(
                        history_specs["prefix_k"].eq(prefix_k).sum()
                    ),
                }
            )
    identity = pd.concat(audit_frames, ignore_index=True)
    found = set(identity["network_seed"].astype(int).unique())
    expected = set(int(value) for value in seeds)
    if found != expected:
        raise RuntimeError(
            "Restoration audit cohort mismatch: "
            f"expected={sorted(expected)}, found={sorted(found)}"
        )
    return identity, pd.DataFrame(checkpoint_rows)


def _collect_progressive_boundary_scalars(
    cfg: BridgeConfig,
    *,
    repo_root: Path,
    seeds: Sequence[int],
    source_records: list[dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    stages = set(PREFIX_TO_PROGRESSIVE_STAGE.values())
    for seed in tuple(int(value) for value in seeds):
        path = (
            _resolve(repo_root, cfg.progressive_root)
            / f"seed_{seed}"
            / "data"
            / "metrics"
            / "panel_b_progressive_update_metrics.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        _record_source(
            source_records,
            path,
            bundle="fig3_progressive_update",
            network_seed=seed,
        )
        frame = pd.read_csv(path)
        selected = frame.loc[
            frame["condition_id"].eq("K10_D200")
            & frame["layer"].eq("layer2")
            & frame["state_variable"].isin(STSP_STATE_KEYS)
            & frame["stage_k"].isin(stages)
        ].copy()
        if selected.empty:
            raise RuntimeError(
                f"No progressive Layer2 u/x rows found for seed {seed}"
            )
        frames.append(selected)
    data = pd.concat(frames, ignore_index=True)
    variables = (
        data.groupby(
            ["network_seed", "stage_k", "state_variable"],
            as_index=False,
        )
        .agg(
            value=("observed_minus_natural_decay", "mean"),
            n_sequences=("sequence_id", "nunique"),
        )
    )
    stage_to_prefix = {
        stage: prefix for prefix, stage in PREFIX_TO_PROGRESSIVE_STAGE.items()
    }
    variables["prefix_k"] = variables["stage_k"].map(stage_to_prefix).astype(int)
    variables["endpoint"] = (
        "input_driven_boundary_displacement_"
        + variables["state_variable"].astype(str)
    )
    joint = (
        variables.groupby(
            ["network_seed", "stage_k", "prefix_k"],
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
            n_sequences=("n_sequences", "min"),
        )
    )
    joint["state_variable"] = "joint_u_x"
    joint["endpoint"] = "joint_ux_input_driven_boundary_displacement"
    output = pd.concat(
        [
            variables[
                [
                    "network_seed",
                    "prefix_k",
                    "stage_k",
                    "state_variable",
                    "endpoint",
                    "value",
                    "n_sequences",
                ]
            ],
            joint[
                [
                    "network_seed",
                    "prefix_k",
                    "stage_k",
                    "state_variable",
                    "endpoint",
                    "value",
                    "n_sequences",
                ]
            ],
        ],
        ignore_index=True,
    )
    return output.sort_values(
        ["network_seed", "prefix_k", "state_variable"],
        kind="stable",
    ).reset_index(drop=True)


def _unified_boundary_inference(
    displacement: pd.DataFrame,
    fixed_scalars: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_displacement = displacement.loc[
        displacement["endpoint"].eq(
            "joint_ux_input_driven_boundary_displacement"
        ),
        ["network_seed", "prefix_k", "endpoint", "value"],
    ].copy()
    fixed_endpoints = (
        "processing_residual_gamma_energy_fraction",
        "layer1_only_layer2_update_donor_transfer",
        "layer1_only_early_class_score_donor_transfer",
    )
    selected_fixed = fixed_scalars.loc[
        fixed_scalars["endpoint"].isin(fixed_endpoints),
        ["network_seed", "prefix_k", "endpoint", "value"],
    ].copy()
    long = pd.concat(
        [selected_displacement, selected_fixed],
        ignore_index=True,
    )
    expected_seeds = set(int(value) for value in seeds)
    rows: list[dict[str, Any]] = []
    for (endpoint, prefix_k), part in long.groupby(
        ["endpoint", "prefix_k"],
        sort=True,
    ):
        part = part.sort_values("network_seed")
        found = set(part["network_seed"].astype(int))
        if found != expected_seeds or len(part) != len(expected_seeds):
            raise RuntimeError(
                f"Incomplete unified endpoint {endpoint} K={prefix_k}: "
                f"expected={sorted(expected_seeds)}, found={sorted(found)}"
            )
        values = part["value"].to_numpy(dtype=np.float64)
        low, high = _bootstrap_ci(
            values,
            seed=_stable_seed(str(endpoint), int(prefix_k)),
        )
        rows.append(
            {
                "endpoint": str(endpoint),
                "prefix_k": int(prefix_k),
                "n_networks": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "ci95_low": low,
                "ci95_high": high,
                "fraction_above_zero": float(np.mean(values > 0.0)),
                "p_one_sided": exact_one_sided_sign_flip_p(values),
            }
        )
    inference = pd.DataFrame(rows)
    inference["holm_adjusted_p"] = holm_adjust(
        inference["p_one_sided"].to_numpy(dtype=np.float64)
    )
    conjunction_rows = []
    for prefix_k, part in inference.groupby("prefix_k", sort=True):
        conjunction_rows.append(
            {
                "prefix_k": int(prefix_k),
                "n_prespecified_links": int(len(part)),
                "all_network_means_positive": int(part["mean"].gt(0).all()),
                "all_links_holm_significant": int(
                    part["holm_adjusted_p"].lt(0.05).all()
                ),
                "conjunction_pass": int(
                    len(part) == 4
                    and part["mean"].gt(0).all()
                    and part["holm_adjusted_p"].lt(0.05).all()
                ),
            }
        )
    return inference, pd.DataFrame(conjunction_rows)


def _run_prefix_bridge(
    ctx: Any,
    *,
    inputs: FixedBArtifact,
    histories: FixedBArtifact,
    c_map: pd.DataFrame,
    prefix_k: int,
    anchors_per_chunk: int,
    max_anchors: int,
    max_history_families: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = _history_rows_at_k(histories.tables["history_specs"], prefix_k)
    selected = all_rows.loc[
        all_rows["history_condition"].isin(("A", "C"))
    ].copy()
    if max_history_families > 0:
        families = sorted(
            int(value) for value in selected["history_family_id"].unique()
        )[:max_history_families]
        selected = selected.loc[
            selected["history_family_id"].isin(families)
        ].copy()
    if selected.empty:
        raise RuntimeError(f"No A/C histories selected for K={prefix_k}")
    row_indices = [int(value) for value in selected.index]
    selected = selected.reset_index(drop=True)
    history_boundary = _load_boundary(
        histories,
        prefix_k,
        row_indices=row_indices,
    )
    elapsed_steps = sorted(
        int(value) for value in selected["elapsed_steps"].unique()
    )
    if len(elapsed_steps) != 1:
        raise RuntimeError(
            f"History rows have non-unique elapsed steps at K={prefix_k}: "
            f"{elapsed_steps}"
        )
    current_time = elapsed_steps[0]
    exact_inputs = np.asarray(inputs.arrays["exact_b_spikes"], dtype=np.bool_)
    spatial_shape = tuple(int(value) for value in exact_inputs.shape[2:])
    history_count = len(selected)
    zero_history = torch.zeros(
        (
            history_count,
            int(ctx.cfg.fixed_b_stimulus_steps),
            *spatial_shape,
        ),
        dtype=torch.bool,
        device=ctx.device,
    )
    _run_branch(
        ctx,
        boundary=history_boundary,
        input_seq=zero_history,
        current_time=current_time,
        restore_mode="full_boundary",
        branch="passive",
        replay_l1_pooled=None,
        capture_l1_pooled=False,
        capture_strong_path=False,
        random_seed=int(ctx.cfg.network_seed)
        + 700_000
        + 10_000 * int(prefix_k),
    )
    post_passive_history = _snapshot_numpy(ctx.net)

    mapping = c_map.sort_values("b_anchor_id").reset_index(drop=True)
    anchor_ids = [int(value) for value in mapping["b_anchor_id"]]
    if max_anchors > 0:
        anchor_ids = anchor_ids[:max_anchors]
    if not anchor_ids:
        raise RuntimeError("No B anchors selected")
    mapping_by_anchor = mapping.set_index("b_anchor_id", drop=False)
    cell_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for chunk_id, start in enumerate(
        range(0, len(anchor_ids), int(anchors_per_chunk))
    ):
        chunk_anchor_ids = anchor_ids[start : start + int(anchors_per_chunk)]
        anchor_count = len(chunk_anchor_ids)
        cell_count = anchor_count * history_count
        repeated_history = _repeat_boundary(history_boundary, anchor_count)
        b_input = np.repeat(
            exact_inputs[np.asarray(chunk_anchor_ids, dtype=np.int64)],
            history_count,
            axis=0,
        )
        _run_branch(
            ctx,
            boundary=repeated_history,
            input_seq=torch.as_tensor(b_input, device=ctx.device),
            current_time=current_time,
            restore_mode="full_boundary",
            branch="free",
            replay_l1_pooled=None,
            capture_l1_pooled=False,
            capture_strong_path=False,
            random_seed=int(ctx.cfg.network_seed)
            + 710_000
            + 10_000 * int(prefix_k)
            + chunk_id,
        )
        post_b = _snapshot_numpy(ctx.net)
        post_passive = _repeat_boundary(
            post_passive_history,
            anchor_count,
        )
        layer1_swap = mix_layer1_stsp(post_b, post_passive)
        own_sham = mix_layer1_stsp(post_b, post_b)
        layer1_mix_pass = _layer1_mix_is_exact(
            layer1_swap,
            donor=post_b,
            receiver=post_passive,
        )
        own_sham_pass = _boundary_exact_equal(own_sham, post_b)

        conditions = {
            "post_B": post_b,
            "post_passive": post_passive,
            "layer1_B_into_passive": layer1_swap,
        }
        audits = {
            name: _audit_stsp_isolated_restore(
                ctx,
                boundary,
                input_shape=spatial_shape,
            )
            for name, boundary in conditions.items()
        }
        all_fast_hashes = {
            value
            for audit in audits.values()
            for value in audit["fast_hashes"]
        }
        common_fast_state = len(all_fast_hashes) == 1

        c_anchor_ids = [
            int(mapping_by_anchor.loc[anchor_id, "c_anchor_id"])
            for anchor_id in chunk_anchor_ids
        ]
        c_input = np.repeat(
            exact_inputs[np.asarray(c_anchor_ids, dtype=np.int64)],
            history_count,
            axis=0,
        )
        combined_boundary = _concatenate_boundaries(
            [
                conditions["post_B"],
                conditions["post_passive"],
                conditions["layer1_B_into_passive"],
            ]
        )
        combined_c = np.concatenate([c_input, c_input, c_input], axis=0)
        c_hashes = [
            _array_sha256(
                combined_c[index * cell_count : (index + 1) * cell_count]
            )
            for index in range(3)
        ]
        c_tensor_identical = len(set(c_hashes)) == 1
        c_result = _run_branch(
            ctx,
            boundary=combined_boundary,
            input_seq=torch.as_tensor(combined_c, device=ctx.device),
            current_time=current_time
            + int(ctx.cfg.fixed_b_stimulus_steps)
            + int(ctx.cfg.fixed_b_post_steps),
            restore_mode="stsp_only",
            branch="free",
            replay_l1_pooled=None,
            capture_l1_pooled=False,
            capture_strong_path=False,
            random_seed=int(ctx.cfg.network_seed)
            + 720_000
            + 10_000 * int(prefix_k)
            + chunk_id,
        )
        zero_result = _run_branch(
            ctx,
            boundary=combined_boundary,
            input_seq=torch.zeros_like(
                torch.as_tensor(combined_c, device=ctx.device)
            ),
            current_time=current_time
            + int(ctx.cfg.fixed_b_stimulus_steps)
            + int(ctx.cfg.fixed_b_post_steps),
            restore_mode="stsp_only",
            branch="passive",
            replay_l1_pooled=None,
            capture_l1_pooled=False,
            capture_strong_path=False,
            random_seed=int(ctx.cfg.network_seed)
            + 721_000
            + 10_000 * int(prefix_k)
            + chunk_id,
        )
        corrected_l2, corrected_scores = _corrected_c_effects(
            c_result,
            zero_result,
        )
        slices = {
            "post_B": slice(0, cell_count),
            "post_passive": slice(cell_count, 2 * cell_count),
            "layer1_B_into_passive": slice(2 * cell_count, 3 * cell_count),
        }
        effects_l2 = {
            name: corrected_l2[indexer]
            for name, indexer in slices.items()
        }
        effects_scores = {
            name: corrected_scores[indexer]
            for name, indexer in slices.items()
        }
        layer2_transfer, layer2_valid = donor_transfer(
            effects_l2["layer1_B_into_passive"],
            effects_l2["post_passive"],
            effects_l2["post_B"],
        )
        score_transfer, score_valid = donor_transfer(
            effects_scores["layer1_B_into_passive"],
            effects_scores["post_passive"],
            effects_scores["post_B"],
        )
        layer2_donor_delta = (
            effects_l2["post_B"] - effects_l2["post_passive"]
        )
        layer2_swap_delta = (
            effects_l2["layer1_B_into_passive"]
            - effects_l2["post_passive"]
        )
        score_donor_delta = (
            effects_scores["post_B"] - effects_scores["post_passive"]
        )
        score_swap_delta = (
            effects_scores["layer1_B_into_passive"]
            - effects_scores["post_passive"]
        )
        written_l1 = _flatten_stsp(post_b, layer="layer1") - _flatten_stsp(
            post_passive,
            layer="layer1",
        )
        b_specs = histories.tables["b_anchor_specs"].set_index(
            "b_anchor_id",
            drop=False,
        )
        for anchor_offset, anchor_id in enumerate(chunk_anchor_ids):
            map_row = mapping_by_anchor.loc[anchor_id]
            b_row = b_specs.loc[anchor_id]
            for history_offset, history in enumerate(
                selected.itertuples(index=False)
            ):
                index = anchor_offset * history_count + history_offset
                c_label = int(map_row["C_label"])
                cell_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "history_row_id": int(history.history_row_id),
                        "history_family_id": int(history.history_family_id),
                        "history_condition": str(history.history_condition),
                        "history_fold": int(history.history_fold),
                        "b_anchor_id": int(anchor_id),
                        "B_image_id": int(b_row["B_image_id"]),
                        "B_label": int(b_row["B_label"]),
                        "B_replicate_id": int(b_row["B_replicate_id"]),
                        "c_anchor_id": int(map_row["c_anchor_id"]),
                        "C_image_id": int(map_row["C_image_id"]),
                        "C_label": c_label,
                        "c_tensor_sha256": _array_sha256(c_input[index]),
                        "layer1_written_ux_norm": float(
                            np.linalg.norm(written_l1[index])
                        ),
                        "layer2_update_donor_receiver_norm": float(
                            np.linalg.norm(layer2_donor_delta[index])
                        ),
                        "layer2_update_swap_receiver_norm": float(
                            np.linalg.norm(layer2_swap_delta[index])
                        ),
                        "layer1_to_layer2_update_donor_transfer": float(
                            layer2_transfer[index]
                        ),
                        "layer2_update_transfer_valid": int(
                            layer2_valid[index]
                        ),
                        "early_class_score_donor_receiver_norm": float(
                            np.linalg.norm(score_donor_delta[index])
                        ),
                        "early_class_score_swap_receiver_norm": float(
                            np.linalg.norm(score_swap_delta[index])
                        ),
                        "layer1_to_early_class_score_donor_transfer": float(
                            score_transfer[index]
                        ),
                        "early_class_score_transfer_valid": int(
                            score_valid[index]
                        ),
                        "post_B_C_target_early_score": float(
                            effects_scores["post_B"][index, c_label]
                        ),
                        "post_passive_C_target_early_score": float(
                            effects_scores["post_passive"][index, c_label]
                        ),
                        "layer1_swap_C_target_early_score": float(
                            effects_scores["layer1_B_into_passive"][
                                index,
                                c_label,
                            ]
                        ),
                    }
                )

        for condition, audit in audits.items():
            identity_pass = bool(
                audit["all_stsp_exact"]
                and common_fast_state
                and c_tensor_identical
                and layer1_mix_pass
                and own_sham_pass
            )
            identity_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "prefix_k": int(prefix_k),
                    "chunk_id": int(chunk_id),
                    "condition": str(condition),
                    "row_count": int(cell_count),
                    "capture_stsp_sha256": str(
                        audit["expected_stsp_digest"]
                    ),
                    "restored_stsp_sha256": str(
                        audit["restored_stsp_digest"]
                    ),
                    "post_boundary_restore_exact": int(
                        audit["all_stsp_exact"]
                    ),
                    "unique_fast_state_hashes_across_conditions": int(
                        len(all_fast_hashes)
                    ),
                    "fast_state_equalized": int(common_fast_state),
                    "C_tensor_sha256": str(c_hashes[0]),
                    "C_tensor_identical_across_conditions": int(
                        c_tensor_identical
                    ),
                    "layer1_donor_stsp_applied": int(layer1_mix_pass),
                    "receiver_layer2_3_stsp_preserved": int(
                        layer1_mix_pass
                    ),
                    "own_state_sham_exact": int(own_sham_pass),
                    "identity_pass": int(identity_pass),
                }
            )
    cells = pd.DataFrame(cell_rows)
    identity = pd.DataFrame(identity_rows)
    if cells.empty or identity.empty:
        raise RuntimeError(f"Bridge produced no rows for K={prefix_k}")
    return cells, identity


def _bridge_network_scalars(
    cells: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix_k, part in cells.groupby("prefix_k", sort=True):
        layer2_valid = part["layer2_update_transfer_valid"].eq(1)
        score_valid = part["early_class_score_transfer_valid"].eq(1)
        identity_part = identity.loc[identity["prefix_k"].eq(prefix_k)]
        rows.append(
            {
                "network_seed": int(part["network_seed"].iloc[0]),
                "prefix_k": int(prefix_k),
                "n_cells": int(len(part)),
                "n_history_families": int(
                    part["history_family_id"].nunique()
                ),
                "n_b_anchors": int(part["b_anchor_id"].nunique()),
                "layer1_to_layer2_update_donor_transfer": float(
                    part.loc[
                        layer2_valid,
                        "layer1_to_layer2_update_donor_transfer",
                    ].mean()
                ),
                "layer2_update_valid_coverage": float(layer2_valid.mean()),
                "layer1_to_early_class_score_donor_transfer": float(
                    part.loc[
                        score_valid,
                        "layer1_to_early_class_score_donor_transfer",
                    ].mean()
                ),
                "early_class_score_valid_coverage": float(
                    score_valid.mean()
                ),
                "mean_layer1_written_ux_norm": float(
                    part["layer1_written_ux_norm"].mean()
                ),
                "all_checkpoint_identity_gates_pass": int(
                    not identity_part.empty
                    and identity_part["identity_pass"].eq(1).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _bridge_inference(scalars: pd.DataFrame) -> pd.DataFrame:
    endpoints = (
        "layer1_to_layer2_update_donor_transfer",
        "layer1_to_early_class_score_donor_transfer",
    )
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for prefix_k, part in scalars.groupby("prefix_k", sort=True):
            values = part[endpoint].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Non-finite network scalar for {endpoint}, K={prefix_k}"
                )
            low, high = _bootstrap_ci(
                values,
                seed=_stable_seed(endpoint, int(prefix_k)),
            )
            coverage_column = (
                "layer2_update_valid_coverage"
                if endpoint == "layer1_to_layer2_update_donor_transfer"
                else "early_class_score_valid_coverage"
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "prefix_k": int(prefix_k),
                    "n_networks": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "fraction_above_zero": float(np.mean(values > 0.0)),
                    "minimum_cell_coverage": float(
                        part[coverage_column].min()
                    ),
                    "p_one_sided": exact_one_sided_sign_flip_p(values),
                }
            )
    inference = pd.DataFrame(rows)
    inference["holm_adjusted_p"] = holm_adjust(
        inference["p_one_sided"].to_numpy(dtype=np.float64)
    )
    return inference


def _load_parent(task_dir: Path, task_id: str) -> FixedBArtifact:
    cache_path = Path(task_dir) / "cache_key.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Required parent cache key is missing: {cache_path}"
        )
    wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
    expected = wrapper.get("cache_key")
    if not isinstance(expected, dict):
        raise ValueError(f"Malformed cache key wrapper: {cache_path}")
    if str(expected.get("task_id")) != str(task_id):
        raise RuntimeError(
            f"Parent task mismatch at {task_dir}: "
            f"expected={task_id}, found={expected.get('task_id')}"
        )
    return load_fixed_b_artifact(
        Path(task_dir),
        expected,
        task_id=str(task_id),
    )


def _parent_file_hashes(
    parents: Mapping[str, Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, root in sorted(parents.items()):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Required parent directory missing: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append(
                {
                    "parent_task": str(task),
                    "relative_file": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def _corrected_c_effects(
    c_result: Mapping[str, np.ndarray],
    zero_result: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    c_displacement = (
        np.asarray(c_result["layer2_ux_post"], dtype=np.float32)
        - np.asarray(c_result["layer2_ux_pre"], dtype=np.float32)
    )
    zero_displacement = (
        np.asarray(zero_result["layer2_ux_post"], dtype=np.float32)
        - np.asarray(zero_result["layer2_ux_pre"], dtype=np.float32)
    )
    layer2 = c_displacement - zero_displacement
    scores = (
        np.asarray(c_result["class_scores_early"], dtype=np.float32)
        - np.asarray(zero_result["class_scores_early"], dtype=np.float32)
    )
    return layer2, scores


def _snapshot_numpy(net: Any) -> dict[str, dict[str, np.ndarray]]:
    snapshot = snapshot_boundary_state(net)
    return {
        layer: {
            state: _to_numpy(value).copy()
            for state, value in layer_values.items()
        }
        for layer, layer_values in snapshot.items()
    }


def _repeat_boundary(
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    repeats: int,
) -> dict[str, dict[str, np.ndarray]]:
    if int(repeats) < 1:
        raise ValueError("Boundary repeat count must be positive")
    return {
        layer: {
            state: np.concatenate(
                [_to_numpy(value)] * int(repeats),
                axis=0,
            )
            for state, value in layer_values.items()
        }
        for layer, layer_values in boundary.items()
    }


def _concatenate_boundaries(
    boundaries: Sequence[
        Mapping[str, Mapping[str, np.ndarray | torch.Tensor]]
    ],
) -> dict[str, dict[str, np.ndarray]]:
    if not boundaries:
        raise ValueError("At least one boundary is required")
    return {
        layer: {
            state: np.concatenate(
                [_to_numpy(boundary[layer][state]) for boundary in boundaries],
                axis=0,
            )
            for state in boundaries[0][layer]
        }
        for layer in boundaries[0]
    }


def _audit_stsp_isolated_restore(
    ctx: Any,
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    *,
    input_shape: Sequence[int],
) -> dict[str, Any]:
    batch_size = int(_to_numpy(next(iter(boundary["layer1"].values()))).shape[0])
    shapes = build_layer_input_shapes(
        ctx.net,
        batch_size,
        *[int(value) for value in input_shape],
    )
    expected_hashes = _boundary_row_hashes(
        boundary,
        state_keys=STSP_STATE_KEYS,
    )
    _restore_boundary(
        ctx.net,
        boundary,
        shapes,
        mode="stsp_only",
        device=ctx.device,
    )
    restored = _snapshot_numpy(ctx.net)
    restored_hashes = _boundary_row_hashes(
        restored,
        state_keys=STSP_STATE_KEYS,
    )
    fast_hashes = _boundary_row_hashes(
        restored,
        state_keys=FAST_STATE_KEYS,
    )
    return {
        "expected_stsp_digest": _hash_string_sequence(expected_hashes),
        "restored_stsp_digest": _hash_string_sequence(restored_hashes),
        "all_stsp_exact": expected_hashes == restored_hashes,
        "fast_hashes": fast_hashes,
    }


def _boundary_row_hashes(
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    *,
    state_keys: Sequence[str],
) -> list[str]:
    batch_sizes = {
        int(_to_numpy(value).shape[0])
        for layer in LAYER_KEYS
        for state, value in boundary[layer].items()
        if state in state_keys
    }
    if len(batch_sizes) != 1:
        raise RuntimeError(
            f"Boundary batch sizes are inconsistent: {sorted(batch_sizes)}"
        )
    batch_size = next(iter(batch_sizes))
    output: list[str] = []
    for row_index in range(batch_size):
        hasher = hashlib.sha256()
        for layer in LAYER_KEYS:
            for state in state_keys:
                if state not in boundary[layer]:
                    continue
                value = np.ascontiguousarray(
                    _to_numpy(boundary[layer][state])[row_index]
                )
                hasher.update(layer.encode("utf-8"))
                hasher.update(b"\0")
                hasher.update(state.encode("utf-8"))
                hasher.update(b"\0")
                hasher.update(str(value.dtype).encode("ascii"))
                hasher.update(b"\0")
                hasher.update(
                    json.dumps(list(value.shape), separators=(",", ":")).encode(
                        "ascii"
                    )
                )
                hasher.update(b"\0")
                hasher.update(value.tobytes(order="C"))
        output.append(hasher.hexdigest())
    return output


def _hash_string_sequence(values: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(str(value).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _layer1_mix_is_exact(
    mixed: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    *,
    donor: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    receiver: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
) -> bool:
    for layer in LAYER_KEYS:
        for state in ALL_BOUNDARY_KEYS:
            if state not in mixed[layer]:
                continue
            expected = (
                donor[layer][state]
                if layer == "layer1" and state in STSP_STATE_KEYS
                else receiver[layer][state]
            )
            if not _arrays_bitwise_equal(
                mixed[layer][state],
                expected,
            ):
                return False
    return True


def _boundary_exact_equal(
    first: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    second: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
) -> bool:
    if set(first) != set(second):
        return False
    for layer in first:
        if set(first[layer]) != set(second[layer]):
            return False
        for state in first[layer]:
            if not _arrays_bitwise_equal(
                first[layer][state],
                second[layer][state],
            ):
                return False
    return True


def _flatten_stsp(
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    *,
    layer: str,
) -> np.ndarray:
    u = _to_numpy(boundary[layer]["u"]).reshape(
        _to_numpy(boundary[layer]["u"]).shape[0],
        -1,
    )
    x = _to_numpy(boundary[layer]["x"]).reshape(
        _to_numpy(boundary[layer]["x"]).shape[0],
        -1,
    )
    return np.concatenate([u, x], axis=1).astype(
        np.float32,
        copy=False,
    )


def _bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(data), size=(int(draws), len(data)))
    means = data[indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _stable_seed(endpoint: str, prefix_k: int) -> int:
    payload = f"{endpoint}:{int(prefix_k)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _prepare_bundle_dirs(root: Path) -> dict[str, Path]:
    root = Path(root)
    dirs = {
        "root": root,
        "data": root / "data",
        "metrics": root / "data" / "metrics",
        "trial_specs": root / "data" / "trial_specs",
        "figures": root / "figures",
        "logs": root / "logs",
        "meta": root / "meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _record_source(
    records: list[dict[str, Any]],
    path: Path,
    *,
    bundle: str,
    network_seed: int,
) -> None:
    path = Path(path)
    records.append(
        {
            "bundle": str(bundle),
            "network_seed": int(network_seed),
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_artifact_manifest(root: Path, *, title: str) -> Path:
    root = Path(root)
    path = root / "artifact_manifest.json"
    files = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if item == path:
            continue
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size_bytes": int(item.stat().st_size),
                "sha256": _sha256_file(item),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "title": str(title),
        "promotion_status": "not_promoted",
        "files": files,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _array_sha256(value: np.ndarray | torch.Tensor) -> str:
    array = np.ascontiguousarray(_to_numpy(value))
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(b"\n")
    hasher.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    hasher.update(b"\n")
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _arrays_bitwise_equal(
    first: np.ndarray | torch.Tensor,
    second: np.ndarray | torch.Tensor,
) -> bool:
    left = np.ascontiguousarray(_to_numpy(first))
    right = np.ascontiguousarray(_to_numpy(second))
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


__all__ = [
    "ALL_BOUNDARY_KEYS",
    "BridgeConfig",
    "DEVELOPMENT_SEED",
    "EXPERIMENT_ID",
    "INFERENCE_SEEDS",
    "aggregate_bridge_cohort",
    "build_c_anchor_mapping",
    "donor_transfer",
    "exact_one_sided_sign_flip_p",
    "holm_adjust",
    "mix_layer1_stsp",
    "run_boundary_analysis",
    "run_bridge_seed",
]
