from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.common.monitored_dms import (
    build_layer_input_shapes,
    snapshot_boundary_state,
)
from src.experiments.common.ping_common import LAYER_KEYS
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
from src.experiments.common.runtime import seed_everything
from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    load_fixed_b_artifact,
)
from src.experiments.paper_figures.fig2.run_task import (
    _build_context,
    _resolve_model_path,
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
    _network_step,
    _restore_boundary,
    _run_branch,
)
from src.experiments.paper_figures.fig2.types import Fig2Config
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
)


EXPERIMENT_ID = "c5_l2_successor_closure"
DEVELOPMENT_SEED = 1000
PRIMARY_ENDPOINTS = (
    "early_layer2_event_map_donor_transfer",
    "layer3_successor_ux_donor_transfer",
)
ALL_BOUNDARY_KEYS = FAST_STATE_KEYS + STSP_STATE_KEYS
NEAR_ZERO = 1e-12


@dataclass(frozen=True)
class C5Config:
    output_dir: str = "results/causal_closure_single_seed_20260803/c5_l2_successor"
    parent_root: str = "results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory"
    dataset_root: str = DEFAULT_DATASET_ROOT
    model_path_glob: str = DEFAULT_MODEL_PATH_GLOB
    device: str = "auto"
    prefixes: tuple[int, ...] = (1, 5)
    anchors_per_chunk: int = 5
    max_anchors: int = 20
    max_history_families: int = 6
    bootstrap_draws: int = 5000
    minimum_valid_coverage: float = 0.80
    minimum_positive_fraction: float = 0.55
    minimum_mean_transfer: float = 0.10
    smoke: bool = False


def run_c5_seed(
    cfg: C5Config,
    *,
    network_seed: int = DEVELOPMENT_SEED,
    command: str | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    output_root = _resolve(repo_root, cfg.output_dir)
    parent_seed_root = (
        _resolve(repo_root, cfg.parent_root)
        / f"seed_{int(network_seed)}"
        / "data"
        / "intermediates"
    )
    input_dir = parent_seed_root / TASK_FIXED_B_INPUT_BANK
    history_dir = parent_seed_root / TASK_FIXED_B_HISTORY_BANK
    parents = {
        TASK_FIXED_B_INPUT_BANK: input_dir,
        TASK_FIXED_B_HISTORY_BANK: history_dir,
    }
    before_hashes = _parent_file_hashes(parents)
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
        output_root=str(output_root),
        network_seed=int(network_seed),
        device=str(cfg.device),
        fixed_b_prefix_depths=tuple(int(value) for value in cfg.prefixes),
        smoke=False,
    )
    ctx = _build_context(fig_cfg, load_model=True)
    dirs = _prepare_bundle_dirs(ctx.seed_dir)
    run_info = build_run_info(
        experiment_name=EXPERIMENT_ID,
        output_dir=ctx.seed_dir,
        entry_script="src.experiments.runners.c5_l2_successor_closure",
        seed=int(network_seed),
        dataset=str(cfg.dataset_root),
        command=command,
        model_path=str(model_path),
    )
    write_run_info(dirs["meta"], run_info)
    logs: list[str] = []
    try:
        protocol = _protocol_payload(cfg, network_seed=int(network_seed))
        _write_json(ctx.seed_dir / "protocol_freeze.json", protocol)
        c_map = build_c_anchor_mapping(histories.tables["b_anchor_specs"])
        _write_csv(dirs["trial_specs"] / "c_anchor_mapping.csv", c_map)

        cell_frames: list[pd.DataFrame] = []
        audit_frames: list[pd.DataFrame] = []
        for prefix_k in tuple(int(value) for value in cfg.prefixes):
            cells, audit = _run_prefix(
                ctx,
                inputs=inputs,
                histories=histories,
                c_map=c_map,
                prefix_k=prefix_k,
                anchors_per_chunk=max(1, int(cfg.anchors_per_chunk)),
                max_anchors=max(1, int(cfg.max_anchors)),
                max_history_families=max(1, int(cfg.max_history_families)),
            )
            cell_frames.append(cells)
            audit_frames.append(audit)
            logs.append(
                f"prefix={prefix_k} cells={len(cells)} chunks={audit['chunk_id'].nunique()}"
            )

        cells = pd.concat(cell_frames, ignore_index=True)
        identity = pd.concat(audit_frames, ignore_index=True)
        endpoint_summary = summarize_c5_endpoints(cells, cfg)
        verdict = screening_verdict(endpoint_summary, identity)

        _write_csv(dirs["metrics"] / "c5_cell_metrics.csv", cells)
        _write_csv(dirs["metrics"] / "c5_endpoint_summary.csv", endpoint_summary)
        _write_csv(dirs["metrics"] / "c5_identity_audit.csv", identity)

        after_hashes = _parent_file_hashes(parents)
        parent_audit = before_hashes.merge(
            after_hashes,
            on=["parent_task", "relative_file"],
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        parent_audit["unchanged"] = (
            parent_audit["sha256_before"].eq(parent_audit["sha256_after"])
            & parent_audit["size_bytes_before"].eq(parent_audit["size_bytes_after"])
        ).astype(int)
        _write_csv(dirs["meta"] / "parent_hash_audit.csv", parent_audit)

        summary = {
            "experiment_id": EXPERIMENT_ID,
            "status": "completed",
            "promotion_status": "exploratory_single_seed_not_manuscript_evidence",
            "network_seed": int(network_seed),
            "seed_role": "development_engineering" if int(network_seed) == DEVELOPMENT_SEED else "exploratory",
            "n_cells": int(len(cells)),
            "n_history_families": int(cells["history_family_id"].nunique()),
            "n_b_anchors": int(cells["b_anchor_id"].nunique()),
            "prefixes": [int(value) for value in sorted(cells["prefix_k"].unique())],
            "all_identity_gates_pass": bool(identity["identity_pass"].eq(1).all()),
            "all_parent_files_unchanged": bool(parent_audit["unchanged"].eq(1).all()),
            "screening": verdict,
            "claim_boundary": (
                "Within one development network, tests whether selectively transplanting the post-B "
                "Layer-2 u/x successor moves identical-C Layer-2 processing and the Layer-3 u/x successor "
                "toward the paired donor history. It tests model-internal causal sufficiency, not population "
                "inference, necessity, complete mediation, or uniqueness."
            ),
            "output_files": {
                "protocol_freeze": "protocol_freeze.json",
                "c_anchor_mapping": "data/trial_specs/c_anchor_mapping.csv",
                "c5_cell_metrics": "data/metrics/c5_cell_metrics.csv",
                "c5_endpoint_summary": "data/metrics/c5_endpoint_summary.csv",
                "c5_identity_audit": "data/metrics/c5_identity_audit.csv",
                "parent_hash_audit": "meta/parent_hash_audit.csv",
            },
        }
        save_run_config(asdict(cfg), ctx.seed_dir)
        save_summary_json(summary, ctx.seed_dir)
        save_log_lines(logs, dirs["logs"])
        finalize_run_info(dirs["meta"], run_info, status="completed")
        _write_artifact_manifest(ctx.seed_dir)
        return summary
    except Exception:
        logs.append("C5 Layer-2 successor closure run failed")
        save_log_lines(logs, dirs["logs"])
        finalize_run_info(dirs["meta"], run_info, status="failed")
        raise


def build_c_anchor_mapping(b_specs: pd.DataFrame) -> pd.DataFrame:
    required = {"b_anchor_id", "B_image_id", "B_label", "B_replicate_id"}
    missing = sorted(required.difference(b_specs.columns))
    if missing:
        raise ValueError(f"b_anchor_specs missing columns: {missing}")
    specs = b_specs.sort_values("b_anchor_id").reset_index(drop=True)
    lookup = {
        (int(row.B_label), int(row.B_replicate_id)): row
        for row in specs.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for row in specs.itertuples(index=False):
        c_key = ((int(row.B_label) + 1) % 10, int(row.B_replicate_id))
        if c_key not in lookup:
            raise ValueError(f"Missing deterministic C anchor for key={c_key}")
        target = lookup[c_key]
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
    if swap_array.shape != receiver_array.shape or donor_array.shape != receiver_array.shape:
        raise ValueError("donor_transfer expects equally shaped arrays")
    if receiver_array.ndim < 2:
        raise ValueError("donor_transfer expects one feature vector per row")
    donor_delta = (donor_array - receiver_array).reshape(len(receiver_array), -1)
    swap_delta = (swap_array - receiver_array).reshape(len(receiver_array), -1)
    denominator = np.sum(donor_delta * donor_delta, axis=1, dtype=np.float64)
    numerator = np.sum(swap_delta * donor_delta, axis=1, dtype=np.float64)
    valid = denominator > float(eps)
    values = np.full(len(receiver_array), np.nan, dtype=np.float64)
    values[valid] = numerator[valid] / denominator[valid]
    return values, valid


def summarize_c5_endpoints(cells: pd.DataFrame, cfg: C5Config) -> pd.DataFrame:
    definitions = {
        "early_layer2_event_map_donor_transfer": "early_layer2_event_map_transfer_valid",
        "layer3_successor_ux_donor_transfer": "layer3_successor_ux_transfer_valid",
    }
    rows: list[dict[str, Any]] = []
    for endpoint, valid_column in definitions.items():
        for prefix_k, part in cells.groupby("prefix_k", sort=True):
            valid = part[valid_column].eq(1) & np.isfinite(part[endpoint])
            selected = part.loc[valid].copy()
            values = selected[endpoint].to_numpy(dtype=np.float64)
            ci_low, ci_high = _crossed_bootstrap_mean_ci(
                selected,
                endpoint,
                draws=int(cfg.bootstrap_draws),
                seed=_stable_seed(endpoint, int(prefix_k)),
            )
            mean = float(values.mean()) if len(values) else float("nan")
            positive_fraction = float(np.mean(values > 0.0)) if len(values) else float("nan")
            screening_pass = bool(
                float(valid.mean()) >= float(cfg.minimum_valid_coverage)
                and np.isfinite(mean)
                and mean >= float(cfg.minimum_mean_transfer)
                and np.isfinite(ci_low)
                and ci_low > 0.0
                and positive_fraction >= float(cfg.minimum_positive_fraction)
            )
            rows.append(
                {
                    "network_seed": int(part["network_seed"].iloc[0]),
                    "prefix_k": int(prefix_k),
                    "endpoint": endpoint,
                    "n_cells": int(len(part)),
                    "n_valid_cells": int(valid.sum()),
                    "valid_coverage": float(valid.mean()),
                    "mean_transfer": mean,
                    "median_transfer": float(np.median(values)) if len(values) else float("nan"),
                    "positive_fraction": positive_fraction,
                    "crossed_bootstrap_ci95_low": ci_low,
                    "crossed_bootstrap_ci95_high": ci_high,
                    "minimum_valid_coverage": float(cfg.minimum_valid_coverage),
                    "minimum_positive_fraction": float(cfg.minimum_positive_fraction),
                    "minimum_mean_transfer": float(cfg.minimum_mean_transfer),
                    "screening_pass": int(screening_pass),
                    "inference_scope": "within_network_crossed_history_family_by_anchor_stability_only",
                }
            )
    return pd.DataFrame(rows)


def screening_verdict(endpoint_summary: pd.DataFrame, identity: pd.DataFrame) -> dict[str, Any]:
    identity_pass = bool(not identity.empty and identity["identity_pass"].eq(1).all())
    complete = bool(
        len(endpoint_summary) == 4
        and set(endpoint_summary["endpoint"]) == set(PRIMARY_ENDPOINTS)
        and set(endpoint_summary["prefix_k"].astype(int)) == {1, 5}
    )
    strong = bool(complete and identity_pass and endpoint_summary["screening_pass"].eq(1).all())
    directionally_positive = bool(
        complete
        and identity_pass
        and endpoint_summary["mean_transfer"].gt(0.0).all()
        and endpoint_summary["crossed_bootstrap_ci95_low"].gt(0.0).all()
    )
    if strong:
        verdict = "supported_in_development_seed"
    elif directionally_positive:
        verdict = "directionally_supported_below_prespecified_strength_gate"
    elif complete and identity_pass and endpoint_summary["mean_transfer"].gt(0.0).any():
        verdict = "mixed_single_seed_evidence"
    else:
        verdict = "not_supported_in_development_seed"
    return {
        "verdict": verdict,
        "all_identity_gates_pass": identity_pass,
        "all_four_endpoint_depth_cells_present": complete,
        "all_prespecified_screening_gates_pass": strong,
        "all_crossed_bootstrap_intervals_above_zero": directionally_positive,
        "inference_unit_warning": (
            "History-family and anchor resampling quantifies within-network stability only; "
            "independently trained networks are required for manuscript-level inference."
        ),
    }


def _run_prefix(
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
    all_rows = _history_rows_at_k(histories.tables["history_specs"], int(prefix_k))
    selected = all_rows.loc[all_rows["history_condition"].isin(("A", "C"))].copy()
    families = sorted(int(value) for value in selected["history_family_id"].unique())[: int(max_history_families)]
    selected = selected.loc[selected["history_family_id"].isin(families)].copy()
    _validate_history_pairs(selected)
    row_indices = [int(value) for value in selected.index]
    selected = selected.reset_index(drop=True)
    history_boundary = _load_boundary(histories, int(prefix_k), row_indices=row_indices)
    elapsed = sorted(int(value) for value in selected["elapsed_steps"].unique())
    if len(elapsed) != 1:
        raise RuntimeError(f"Non-unique elapsed_steps for K={prefix_k}: {elapsed}")
    current_time = int(elapsed[0])

    exact_inputs = np.asarray(inputs.arrays["exact_b_spikes"], dtype=np.bool_)
    spatial_shape = tuple(int(value) for value in exact_inputs.shape[2:])
    mapping = c_map.sort_values("b_anchor_id").reset_index(drop=True)
    anchor_ids = [int(value) for value in mapping["b_anchor_id"]][: int(max_anchors)]
    mapping_by_anchor = mapping.set_index("b_anchor_id", drop=False)
    history_count = int(len(selected))
    local_donor_indices = _paired_history_indices(selected)

    cell_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for chunk_id, start in enumerate(range(0, len(anchor_ids), int(anchors_per_chunk))):
        chunk_anchor_ids = anchor_ids[start : start + int(anchors_per_chunk)]
        anchor_count = int(len(chunk_anchor_ids))
        cell_count = int(anchor_count * history_count)
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
            random_seed=int(ctx.cfg.network_seed) + 810_000 + 10_000 * int(prefix_k) + chunk_id,
        )
        post_b = _snapshot_numpy(ctx.net)
        donor_indices = np.concatenate(
            [local_donor_indices + anchor_index * history_count for anchor_index in range(anchor_count)]
        ).astype(np.int64, copy=False)
        identity_indices = np.arange(cell_count, dtype=np.int64)
        l2_swap = _mix_layer2_stsp_by_index(post_b, donor_indices)
        own_sham = _mix_layer2_stsp_by_index(post_b, identity_indices)
        mix_exact = _layer2_mix_is_exact(l2_swap, post_b, donor_indices)
        sham_boundary_exact = _boundary_exact_equal(own_sham, post_b)

        conditions = _concatenate_boundaries([post_b, l2_swap, own_sham])
        restore_audit = _audit_stsp_only_restore(ctx, conditions, input_shape=spatial_shape)
        c_anchor_ids = [int(mapping_by_anchor.loc[anchor_id, "c_anchor_id"]) for anchor_id in chunk_anchor_ids]
        c_input = np.repeat(
            exact_inputs[np.asarray(c_anchor_ids, dtype=np.int64)],
            history_count,
            axis=0,
        )
        combined_c = np.concatenate([c_input, c_input, c_input], axis=0)
        c_hashes = [
            _array_sha256(combined_c[index * cell_count : (index + 1) * cell_count])
            for index in range(3)
        ]
        c_tensor_identical = len(set(c_hashes)) == 1
        probe_time = current_time + int(ctx.cfg.fixed_b_stimulus_steps) + int(ctx.cfg.fixed_b_post_steps)
        c_result = _run_transition_capture(
            ctx,
            boundary=conditions,
            input_seq=torch.as_tensor(combined_c, device=ctx.device),
            current_time=probe_time,
            passive=False,
            random_seed=int(ctx.cfg.network_seed) + 820_000 + 10_000 * int(prefix_k) + chunk_id,
        )
        zero_result = _run_transition_capture(
            ctx,
            boundary=conditions,
            input_seq=torch.as_tensor(combined_c, device=ctx.device),
            current_time=probe_time,
            passive=True,
            random_seed=int(ctx.cfg.network_seed) + 821_000 + 10_000 * int(prefix_k) + chunk_id,
        )
        corrected = _passive_corrected_effects(c_result, zero_result)
        slices = {
            "native": slice(0, cell_count),
            "layer2_swap": slice(cell_count, 2 * cell_count),
            "own_sham": slice(2 * cell_count, 3 * cell_count),
        }
        native_l2 = corrected["early_layer2_event_map"][slices["native"]]
        swap_l2 = corrected["early_layer2_event_map"][slices["layer2_swap"]]
        sham_l2 = corrected["early_layer2_event_map"][slices["own_sham"]]
        native_l3 = corrected["layer3_successor_ux"][slices["native"]]
        swap_l3 = corrected["layer3_successor_ux"][slices["layer2_swap"]]
        sham_l3 = corrected["layer3_successor_ux"][slices["own_sham"]]
        donor_l2 = native_l2[donor_indices]
        donor_l3 = native_l3[donor_indices]

        l2_transfer, l2_valid = donor_transfer(swap_l2, native_l2, donor_l2)
        l3_transfer, l3_valid = donor_transfer(swap_l3, native_l3, donor_l3)
        l2_cosine = _row_cosine(swap_l2 - native_l2, donor_l2 - native_l2)
        l3_cosine = _row_cosine(swap_l3 - native_l3, donor_l3 - native_l3)
        sham_l2_max = float(np.max(np.abs(sham_l2 - native_l2)))
        sham_l3_max = float(np.max(np.abs(sham_l3 - native_l3)))
        sham_output_exact = bool(sham_l2_max == 0.0 and sham_l3_max == 0.0)
        identity_pass = bool(
            mix_exact
            and sham_boundary_exact
            and restore_audit["all_stsp_exact"]
            and restore_audit["fast_state_uniform"]
            and c_tensor_identical
            and sham_output_exact
        )
        audit_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "prefix_k": int(prefix_k),
                "chunk_id": int(chunk_id),
                "row_count": int(cell_count),
                "layer2_only_mix_exact": int(mix_exact),
                "own_sham_boundary_exact": int(sham_boundary_exact),
                "stsp_restore_exact": int(restore_audit["all_stsp_exact"]),
                "fast_state_uniform_after_reset": int(restore_audit["fast_state_uniform"]),
                "C_tensor_identical_across_conditions": int(c_tensor_identical),
                "own_sham_output_exact": int(sham_output_exact),
                "own_sham_l2_max_abs": sham_l2_max,
                "own_sham_l3_max_abs": sham_l3_max,
                "identity_pass": int(identity_pass),
            }
        )

        starting_l2 = _flatten_ux(post_b, layer="layer2")
        for local_anchor_index, b_anchor_id in enumerate(chunk_anchor_ids):
            map_row = mapping_by_anchor.loc[int(b_anchor_id)]
            for history_index, history in selected.iterrows():
                index = int(local_anchor_index * history_count + int(history_index))
                donor_history_index = int(local_donor_indices[int(history_index)])
                donor_history = selected.iloc[donor_history_index]
                l2_donor_delta = donor_l2[index] - native_l2[index]
                l2_swap_delta = swap_l2[index] - native_l2[index]
                l3_donor_delta = donor_l3[index] - native_l3[index]
                l3_swap_delta = swap_l3[index] - native_l3[index]
                cell_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "history_family_id": int(history["history_family_id"]),
                        "receiver_history_condition": str(history["history_condition"]),
                        "donor_history_condition": str(donor_history["history_condition"]),
                        "receiver_history_row_id": int(history["history_row_id"]),
                        "donor_history_row_id": int(donor_history["history_row_id"]),
                        "b_anchor_id": int(b_anchor_id),
                        "B_image_id": int(map_row["B_image_id"]),
                        "B_label": int(map_row["B_label"]),
                        "c_anchor_id": int(map_row["c_anchor_id"]),
                        "C_image_id": int(map_row["C_image_id"]),
                        "C_label": int(map_row["C_label"]),
                        "C_tensor_sha256": _array_sha256(c_input[index]),
                        "starting_layer2_ux_donor_receiver_norm": float(
                            np.linalg.norm(starting_l2[donor_indices[index]] - starting_l2[index])
                        ),
                        "early_layer2_event_map_donor_receiver_norm": float(np.linalg.norm(l2_donor_delta)),
                        "early_layer2_event_map_swap_receiver_norm": float(np.linalg.norm(l2_swap_delta)),
                        "early_layer2_event_map_donor_transfer": float(l2_transfer[index]),
                        "early_layer2_event_map_transfer_cosine": float(l2_cosine[index]),
                        "early_layer2_event_map_transfer_valid": int(l2_valid[index]),
                        "layer3_successor_ux_donor_receiver_norm": float(np.linalg.norm(l3_donor_delta)),
                        "layer3_successor_ux_swap_receiver_norm": float(np.linalg.norm(l3_swap_delta)),
                        "layer3_successor_ux_donor_transfer": float(l3_transfer[index]),
                        "layer3_successor_ux_transfer_cosine": float(l3_cosine[index]),
                        "layer3_successor_ux_transfer_valid": int(l3_valid[index]),
                    }
                )
    cells = pd.DataFrame(cell_rows)
    audit = pd.DataFrame(audit_rows)
    if cells.empty or audit.empty:
        raise RuntimeError(f"C5 produced no outputs for K={prefix_k}")
    return cells, audit


def _run_transition_capture(
    ctx: Any,
    *,
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    input_seq: torch.Tensor,
    current_time: int,
    passive: bool,
    random_seed: int,
) -> dict[str, np.ndarray]:
    seed_everything(int(random_seed))
    batch_size, stimulus_steps, channels, height, width = input_seq.shape
    shapes = build_layer_input_shapes(ctx.net, batch_size, channels, height, width)
    _restore_boundary(ctx.net, boundary, shapes, mode="stsp_only", device=ctx.device)
    ctx.net.layer3.reset_decision_state()
    with torch.no_grad():
        ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
        ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
    l3_pre = _layer_ux_checkpoint(ctx.net.layer3, batch_size)
    zero = torch.zeros((batch_size, channels, height, width), dtype=torch.bool, device=ctx.device)
    early_map: torch.Tensor | None = None
    total_steps = int(stimulus_steps + ctx.cfg.fixed_b_post_steps)
    early_cutoff = min(int(ctx.cfg.fixed_b_early_window_steps), int(stimulus_steps))
    with torch.no_grad():
        for local_step in range(total_steps):
            external = input_seq[:, local_step] if local_step < stimulus_steps and not passive else zero
            _, _, s2, _, _, _ = _network_step(
                ctx.net,
                external,
                current_time=int(current_time) + local_step,
                layer2_replay_input=None,
                layer3_time=local_step,
                monitor_layer1=False,
            )
            if local_step < early_cutoff:
                value = s2.detach().to(torch.float32)
                early_map = value.clone() if early_map is None else early_map + value
    if early_map is None:
        raise RuntimeError("Early Layer-2 capture window was empty")
    return {
        "early_layer2_event_map": early_map.cpu().numpy().astype(np.float32, copy=False),
        "layer3_ux_pre": l3_pre,
        "layer3_ux_post": _layer_ux_checkpoint(ctx.net.layer3, batch_size),
    }


def _passive_corrected_effects(
    active: Mapping[str, np.ndarray],
    passive: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    active_l3 = np.asarray(active["layer3_ux_post"], dtype=np.float32) - np.asarray(
        active["layer3_ux_pre"], dtype=np.float32
    )
    passive_l3 = np.asarray(passive["layer3_ux_post"], dtype=np.float32) - np.asarray(
        passive["layer3_ux_pre"], dtype=np.float32
    )
    return {
        "early_layer2_event_map": np.asarray(active["early_layer2_event_map"], dtype=np.float32)
        - np.asarray(passive["early_layer2_event_map"], dtype=np.float32),
        "layer3_successor_ux": active_l3 - passive_l3,
    }


def _mix_layer2_stsp_by_index(
    receiver: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    donor_indices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    donor_indices = np.asarray(donor_indices, dtype=np.int64)
    output: dict[str, dict[str, np.ndarray]] = {}
    for layer in LAYER_KEYS:
        output[layer] = {}
        for state, value in receiver[layer].items():
            array = _to_numpy(value)
            source = array[donor_indices] if layer == "layer2" and state in STSP_STATE_KEYS else array
            output[layer][state] = source.copy()
    return output


def _layer2_mix_is_exact(
    mixed: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    receiver: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    donor_indices: np.ndarray,
) -> bool:
    for layer in LAYER_KEYS:
        for state, value in mixed[layer].items():
            receiver_value = _to_numpy(receiver[layer][state])
            expected = (
                receiver_value[np.asarray(donor_indices, dtype=np.int64)]
                if layer == "layer2" and state in STSP_STATE_KEYS
                else receiver_value
            )
            if not _arrays_bitwise_equal(value, expected):
                return False
    return True


def _audit_stsp_only_restore(
    ctx: Any,
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    *,
    input_shape: Sequence[int],
) -> dict[str, bool]:
    batch_size = int(_to_numpy(boundary["layer1"]["u"]).shape[0])
    shapes = build_layer_input_shapes(ctx.net, batch_size, *[int(value) for value in input_shape])
    _restore_boundary(ctx.net, boundary, shapes, mode="stsp_only", device=ctx.device)
    restored = _snapshot_numpy(ctx.net)
    all_stsp_exact = all(
        _arrays_bitwise_equal(boundary[layer][state], restored[layer][state])
        for layer in LAYER_KEYS
        for state in STSP_STATE_KEYS
        if state in boundary[layer]
    )
    fast_state_uniform = True
    for layer in LAYER_KEYS:
        for state in FAST_STATE_KEYS:
            value = restored[layer][state]
            if len(value) > 1 and not _arrays_bitwise_equal(value, np.repeat(value[:1], len(value), axis=0)):
                fast_state_uniform = False
    return {"all_stsp_exact": bool(all_stsp_exact), "fast_state_uniform": bool(fast_state_uniform)}


def _validate_history_pairs(selected: pd.DataFrame) -> None:
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
    output = []
    for row in selected.itertuples(index=False):
        donor_condition = "C" if str(row.history_condition) == "A" else "A"
        output.append(lookup[(int(row.history_family_id), donor_condition)])
    return np.asarray(output, dtype=np.int64)


def _crossed_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    families = sorted(int(value) for value in frame["history_family_id"].unique())
    anchors = sorted(int(value) for value in frame["b_anchor_id"].unique())
    family_index = {value: index for index, value in enumerate(families)}
    anchor_index = {value: index for index, value in enumerate(anchors)}
    f_rows = frame["history_family_id"].astype(int).map(family_index).to_numpy(dtype=np.int64)
    a_rows = frame["b_anchor_id"].astype(int).map(anchor_index).to_numpy(dtype=np.int64)
    values = frame[value_column].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        f_sample = rng.integers(0, len(families), size=len(families))
        a_sample = rng.integers(0, len(anchors), size=len(anchors))
        f_weight = np.bincount(f_sample, minlength=len(families))[f_rows]
        a_weight = np.bincount(a_sample, minlength=len(anchors))[a_rows]
        weights = f_weight * a_weight
        denominator = float(weights.sum())
        means[draw] = float(np.sum(values * weights) / denominator) if denominator > 0 else float("nan")
    finite = means[np.isfinite(means)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _row_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float32).reshape(len(first), -1)
    b = np.asarray(second, dtype=np.float32).reshape(len(second), -1)
    numerator = np.sum(a * b, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan, dtype=np.float64)
    valid = denominator > NEAR_ZERO
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _layer_ux_checkpoint(layer: Any, batch_size: int) -> np.ndarray:
    u = layer.u_pre.detach().reshape(batch_size, -1).cpu().numpy()
    x = layer.x_pre.detach().reshape(batch_size, -1).cpu().numpy()
    return np.concatenate([u, x], axis=1).astype(np.float32, copy=False)


def _flatten_ux(
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    *,
    layer: str,
) -> np.ndarray:
    u = _to_numpy(boundary[layer]["u"]).reshape(_to_numpy(boundary[layer]["u"]).shape[0], -1)
    x = _to_numpy(boundary[layer]["x"]).reshape(_to_numpy(boundary[layer]["x"]).shape[0], -1)
    return np.concatenate([u, x], axis=1).astype(np.float32, copy=False)


def _snapshot_numpy(net: Any) -> dict[str, dict[str, np.ndarray]]:
    snapshot = snapshot_boundary_state(net)
    return {
        layer: {state: _to_numpy(value).copy() for state, value in layer_values.items()}
        for layer, layer_values in snapshot.items()
    }


def _repeat_boundary(
    boundary: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    repeats: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        layer: {
            state: np.concatenate([_to_numpy(value)] * int(repeats), axis=0)
            for state, value in layer_values.items()
        }
        for layer, layer_values in boundary.items()
    }


def _concatenate_boundaries(
    boundaries: Sequence[Mapping[str, Mapping[str, np.ndarray | torch.Tensor]]],
) -> dict[str, dict[str, np.ndarray]]:
    if not boundaries:
        raise ValueError("At least one boundary is required")
    return {
        layer: {
            state: np.concatenate([_to_numpy(boundary[layer][state]) for boundary in boundaries], axis=0)
            for state in boundaries[0][layer]
        }
        for layer in boundaries[0]
    }


def _boundary_exact_equal(
    first: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
    second: Mapping[str, Mapping[str, np.ndarray | torch.Tensor]],
) -> bool:
    if set(first) != set(second):
        return False
    return all(
        set(first[layer]) == set(second[layer])
        and all(_arrays_bitwise_equal(first[layer][state], second[layer][state]) for state in first[layer])
        for layer in first
    )


def _load_parent(task_dir: Path, task_id: str) -> FixedBArtifact:
    cache_path = Path(task_dir) / "cache_key.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Required parent cache key is missing: {cache_path}")
    wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
    expected = wrapper.get("cache_key")
    if not isinstance(expected, dict) or str(expected.get("task_id")) != str(task_id):
        raise RuntimeError(f"Parent task/cache-key mismatch at {task_dir}")
    return load_fixed_b_artifact(Path(task_dir), expected, task_id=str(task_id))


def _parent_file_hashes(parents: Mapping[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, root in sorted(parents.items()):
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Required parent directory missing: {root}")
        for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
            rows.append(
                {
                    "parent_task": str(task),
                    "relative_file": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def _protocol_payload(cfg: C5Config, *, network_seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_seed": int(network_seed),
        "seed_role": "development_engineering",
        "causal_question": (
            "Does the post-B Layer-2 u/x successor become a causal inherited state for the next identical C transition?"
        ),
        "intervention": (
            "Within each A/C history pair and B anchor, transplant only post-B Layer-2 u/x from the paired donor; "
            "preserve receiver Layer-1/3 u/x, reset all fast variables, and present an identical C spike tensor."
        ),
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "passive_correction": "active C displacement minus duration-matched zero-input displacement",
        "screening_gates": {
            "minimum_valid_coverage": float(cfg.minimum_valid_coverage),
            "minimum_positive_fraction": float(cfg.minimum_positive_fraction),
            "minimum_mean_transfer": float(cfg.minimum_mean_transfer),
            "crossed_bootstrap_ci95_low": "> 0",
            "identity_audits": "all pass",
        },
        "design": {
            "prefixes": [int(value) for value in cfg.prefixes],
            "history_families": int(cfg.max_history_families),
            "B_to_C_anchors": int(cfg.max_anchors),
            "directions": ["A_to_C", "C_to_A"],
            "early_layer2_window_ms": 20,
        },
        "claim_boundary": "single-network exploratory screen; not manuscript-level population inference",
    }


def _prepare_bundle_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": Path(root),
        "metrics": Path(root) / "data" / "metrics",
        "trial_specs": Path(root) / "data" / "trial_specs",
        "figures": Path(root) / "figures",
        "logs": Path(root) / "logs",
        "meta": Path(root) / "meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact_manifest(root: Path) -> None:
    path = Path(root) / "artifact_manifest.json"
    files = []
    for item in sorted(candidate for candidate in Path(root).rglob("*") if candidate.is_file() and candidate != path):
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size_bytes": int(item.stat().st_size),
                "sha256": _sha256_file(item),
            }
        )
    _write_json(
        path,
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "promotion_status": "exploratory_single_seed_not_manuscript_evidence",
            "files": files,
        },
    )


def _array_sha256(value: np.ndarray | torch.Tensor) -> str:
    array = np.ascontiguousarray(_to_numpy(value))
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_seed(endpoint: str, prefix_k: int) -> int:
    payload = f"{endpoint}:{int(prefix_k)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


__all__ = [
    "C5Config",
    "DEVELOPMENT_SEED",
    "PRIMARY_ENDPOINTS",
    "build_c_anchor_mapping",
    "donor_transfer",
    "run_c5_seed",
    "screening_verdict",
    "summarize_c5_endpoints",
]
