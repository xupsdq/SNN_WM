from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.experiments.common.dataset import encode_images
from src.experiments.common.monitored_dms import build_layer_input_shapes, snapshot_boundary_state
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state
from src.experiments.common.runtime import seed_everything
from src.experiments.paper_figures.fig2.fixed_b_artifacts import FixedBArtifact, array_hash
from src.experiments.paper_figures.fig2.fixed_b_protocol import select_history_families
from src.experiments.paper_figures.fig2.successor_replay import (
    FAST_STATE_KEYS,
    STSP_STATE_KEYS,
    advance_network_step,
    layer_ux_checkpoint,
    restore_boundary_state,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import (
    FIXED_B_SCHEMA_VERSION,
    materialize_selected_specs,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext

def build_exact_b_input_bank(
    ctx: ExperimentContext,
    specs: FixedBArtifact,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    b_specs = specs.tables["b_anchor_specs"].sort_values("b_anchor_id").reset_index(drop=True)
    history_specs = specs.tables["history_input_specs"].sort_values("history_input_id").reset_index(drop=True)
    b_spikes = _encode_source_rows(
        ctx,
        b_specs,
        image_column="B_image_id",
        seed_column="encoding_seed",
        steps=int(ctx.cfg.fixed_b_stimulus_steps),
    )
    history_spikes = _encode_source_rows(
        ctx,
        history_specs,
        image_column="image_id",
        seed_column="encoding_seed",
        steps=int(ctx.cfg.fixed_b_item_steps),
    )
    b_rows = []
    for index, row in enumerate(b_specs.itertuples(index=False)):
        local = np.ascontiguousarray(b_spikes[index])
        b_rows.append(
            {
                "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                "b_anchor_id": int(row.b_anchor_id),
                "B_image_id": int(row.B_image_id),
                "B_label": int(row.B_label),
                "B_image_sha256": str(row.B_image_sha256),
                "encoding_seed": int(row.encoding_seed),
                "storage_key": "exact_b_spikes",
                "row_index": int(index),
                "shape": "x".join(str(value) for value in local.shape),
                "dtype": str(local.dtype),
                "tensor_sha256": array_hash(local),
                "spike_count": int(local.sum()),
                "active_site_count": int(local.any(axis=0).sum()),
                "input_energy": float(local.sum()),
            }
        )
    history_rows = []
    for index, row in enumerate(history_specs.itertuples(index=False)):
        local = np.ascontiguousarray(history_spikes[index])
        history_rows.append(
            {
                "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                "history_input_id": int(row.history_input_id),
                "image_id": int(row.image_id),
                "label": int(row.label),
                "image_sha256": str(row.image_sha256),
                "encoding_seed": int(row.encoding_seed),
                "storage_key": "history_spikes",
                "row_index": int(index),
                "shape": "x".join(str(value) for value in local.shape),
                "dtype": str(local.dtype),
                "tensor_sha256": array_hash(local),
                "spike_count": int(local.sum()),
            }
        )
    return (
        {
            "input_manifest": pd.DataFrame(b_rows),
            "history_input_manifest": pd.DataFrame(history_rows),
        },
        {
            "exact_b_spikes": b_spikes.astype(np.bool_, copy=False),
            "history_spikes": history_spikes.astype(np.bool_, copy=False),
        },
        {
            "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
            "identity_rule": "one frozen encoded tensor per source image reused byte-for-byte",
            "encoding_is_repeated": False,
            "B_stimulus_steps": int(ctx.cfg.fixed_b_stimulus_steps),
            "history_item_steps": int(ctx.cfg.fixed_b_item_steps),
        },
    )


def build_history_boundary_bank(
    ctx: ExperimentContext,
    specs: FixedBArtifact,
    inputs: FixedBArtifact,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    if "candidate_history_specs" not in specs.tables:
        selected_tables = {
            name: specs.tables[name].copy()
            for name in (
                "history_families",
                "history_specs",
                "b_anchor_specs",
                "cell_specs",
                "fold_specs",
                "branch_specs",
                "swap_specs",
                "null_specs",
            )
        }
        frozen_audits = {
            name: specs.tables[name].copy()
            for name in (
                "selection_audit",
                "source_balance",
                "candidate_overlap",
            )
            if name in specs.tables
        }
        arrays, selected_features = _simulate_history_rows(ctx, selected_tables["history_specs"], inputs)
        restoration = _restoration_audit(ctx, selected_tables["history_specs"], arrays, inputs.arrays["exact_b_spikes"][0])
        tables = dict(selected_tables)
        tables.update(
            {
                "prestate_features": selected_features,
                "restoration_audit": restoration,
                **frozen_audits,
            }
        )
        return tables, arrays, {"source_selection": {"loaded_from_frozen_protocol": True}}

    candidate_specs = specs.tables["candidate_history_specs"].copy()
    candidate_arrays, candidate_features = _simulate_history_rows(ctx, candidate_specs, inputs)
    candidate_overlap = _candidate_overlap_table(ctx, candidate_specs, candidate_arrays, inputs)
    selected_ids, selection_audit, source_balance, selection_summary = select_history_families(
        ctx,
        candidate_features,
        candidate_overlap,
        specs.tables["candidate_history_families"],
    )
    selected_tables = materialize_selected_specs(ctx, specs.tables, selected_ids)
    selected_arrays, selected_features = _simulate_history_rows(ctx, selected_tables["history_specs"], inputs)
    restoration = _restoration_audit(
        ctx,
        selected_tables["history_specs"],
        selected_arrays,
        inputs.arrays["exact_b_spikes"][0],
    )
    tables = dict(selected_tables)
    tables.update(
        {
            "candidate_prestate_features": candidate_features,
            "candidate_overlap": candidate_overlap,
            "selection_audit": selection_audit,
            "source_balance": source_balance,
            "prestate_features": selected_features,
            "restoration_audit": restoration,
        }
    )
    selection_summary = dict(selection_summary)
    selection_summary.update(
        {
            "selector_network_seed": int(ctx.cfg.network_seed),
            "history_timing": "each item stimulus followed by matched zero-input delay",
            "S0_timing": "zero input for identical elapsed steps",
            "candidate_boundary_arrays_persisted": False,
        }
    )
    return tables, selected_arrays, {"source_selection": selection_summary}


def build_replay_bank(
    ctx: ExperimentContext,
    specs: FixedBArtifact,
    inputs: FixedBArtifact,
    histories: FixedBArtifact,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    del specs
    history_specs = histories.tables["history_specs"]
    exact_b_spikes = inputs.arrays["exact_b_spikes"]
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for prefix_k in sorted(int(value) for value in history_specs["prefix_k"].unique()):
        rows_k = _history_rows_at_k(history_specs, prefix_k)
        source_index = int(np.flatnonzero(rows_k["history_condition"].eq("S0").to_numpy())[0])
        source_row = rows_k.iloc[source_index]
        boundary = _load_boundary(histories, prefix_k, row_indices=[source_index])
        replay_chunks: list[np.ndarray] = []
        for anchor_id in range(len(exact_b_spikes)):
            result = _run_branch(
                ctx,
                boundary=boundary,
                input_seq=torch.as_tensor(exact_b_spikes[anchor_id : anchor_id + 1], device=ctx.device),
                current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                restore_mode="stsp_only",
                branch="free",
                replay_l1_pooled=None,
                capture_l1_pooled=True,
                capture_strong_path=False,
                random_seed=int(ctx.cfg.fixed_b_protocol_seed) + 400_000 + 1000 * prefix_k + anchor_id,
            )
            replay = result["l1_pooled_spikes"]
            replay_chunks.append(replay[0])
            count_map = replay[0, : int(ctx.cfg.fixed_b_stimulus_steps)].sum(axis=0).astype(np.float32)
            feature_rows.extend(_event_feature_rows(ctx, prefix_k, anchor_id, count_map))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "prefix_k": int(prefix_k),
                    "b_anchor_id": int(anchor_id),
                    "source_history_condition": "S0",
                    "source_history_row_id": int(source_row["history_row_id"]),
                    "storage_key": f"replay_k{prefix_k}",
                    "row_index": int(anchor_id),
                    "exact_b_tensor_sha256": str(inputs.tables["input_manifest"].iloc[anchor_id]["tensor_sha256"]),
                    "replay_tensor_sha256": array_hash(replay[0]),
                    "replay_event_count": int(replay[0].sum()),
                }
            )
        arrays[f"replay_k{prefix_k}"] = np.stack(replay_chunks, axis=0).astype(np.bool_, copy=False)
    return (
        {"replay_manifest": pd.DataFrame(rows), "b_event_features": pd.DataFrame(feature_rows)},
        arrays,
        {
            "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
            "replay_definition": "Layer1 pooled event raster captured from elapsed-time-matched S0 plus exact B",
            "replay_application": "Layer1 processes exact B while Layer2 receives the fixed S0+B pooled raster",
        },
    )


def build_rollout_bank(
    ctx: ExperimentContext,
    specs: FixedBArtifact,
    inputs: FixedBArtifact,
    histories: FixedBArtifact,
    replay: FixedBArtifact,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    del specs
    history_specs = histories.tables["history_specs"]
    b_specs = histories.tables["b_anchor_specs"].sort_values("b_anchor_id").reset_index(drop=True)
    exact_b_spikes = inputs.arrays["exact_b_spikes"]
    output_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    event_manifest_rows: list[dict[str, Any]] = []
    arrays_by_name: dict[str, list[np.ndarray]] = {
        "delta_layer1_ux": [],
        "delta_layer2_ux": [],
        "delta_layer2_g": [],
        "delta_layer3_ux": [],
        "class_scores": [],
        "layer1_drive_features": [],
        "layer1_voltage_features": [],
        "layer1_inhibition_features": [],
        "layer1_event_features": [],
    }
    packed_event_rows: list[np.ndarray] = []

    for prefix_k in sorted(int(value) for value in history_specs["prefix_k"].unique()):
        rows_k = _history_rows_at_k(history_specs, prefix_k)
        boundary = _load_boundary(histories, prefix_k)
        batch_size = len(rows_k)
        zero_seq = torch.zeros(
            (batch_size, int(ctx.cfg.fixed_b_stimulus_steps), *exact_b_spikes.shape[2:]),
            dtype=torch.bool,
            device=ctx.device,
        )
        passive_by_track: dict[str, dict[str, np.ndarray]] = {}
        for track, restore_mode in (("natural", "full_boundary"), ("stsp_isolated", "stsp_only")):
            passive_by_track[track] = _run_branch(
                ctx,
                boundary=boundary,
                input_seq=zero_seq,
                current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                restore_mode=restore_mode,
                branch="passive",
                replay_l1_pooled=None,
                capture_l1_pooled=False,
                capture_strong_path=True,
                random_seed=int(ctx.cfg.network_seed) + 500_000 + 1000 * prefix_k + (0 if track == "natural" else 1),
            )

        for anchor_id in range(len(exact_b_spikes)):
            repeated_b = torch.as_tensor(np.repeat(exact_b_spikes[anchor_id : anchor_id + 1], batch_size, axis=0), device=ctx.device)
            for track, restore_mode, branches in (
                ("natural", "full_boundary", ("free",)),
                ("stsp_isolated", "stsp_only", ("free", "replay")),
            ):
                for branch in branches:
                    replay_input = None
                    if branch == "replay":
                        source = replay.arrays[f"replay_k{prefix_k}"][anchor_id : anchor_id + 1]
                        replay_input = np.repeat(source, batch_size, axis=0)
                    result = _run_branch(
                        ctx,
                        boundary=boundary,
                        input_seq=repeated_b,
                        current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                        restore_mode=restore_mode,
                        branch=branch,
                        replay_l1_pooled=replay_input,
                        capture_l1_pooled=False,
                        capture_strong_path=True,
                        capture_layer2_presynaptic_trace=(
                            track == "stsp_isolated" and branch == "free"
                        ),
                        random_seed=int(ctx.cfg.network_seed) + 510_000 + 10_000 * prefix_k + 100 * anchor_id + (0 if track == "natural" else 10) + (0 if branch == "free" else 1),
                    )
                    passive = passive_by_track[track]
                    deltas = {
                        "delta_layer1_ux": result["layer1_ux"] - passive["layer1_ux"],
                        "delta_layer2_ux": result["layer2_ux"] - passive["layer2_ux"],
                        "delta_layer2_g": result["layer2_g"] - passive["layer2_g"],
                        "delta_layer3_ux": result["layer3_ux"] - passive["layer3_ux"],
                    }
                    corrected_path = {
                        "layer1_drive_features": result["layer1_drive_features"] - passive["layer1_drive_features"],
                        "layer1_voltage_features": result["layer1_voltage_features"] - passive["layer1_voltage_features"],
                        "layer1_inhibition_features": result["layer1_inhibition_features"] - passive["layer1_inhibition_features"],
                        "layer1_event_features": result["layer1_event_features"] - passive["layer1_event_features"],
                    }
                    storage_start = len(output_rows)
                    for name, values in {**deltas, **corrected_path, "class_scores": result["class_scores"]}.items():
                        arrays_by_name[name].extend(np.asarray(values, dtype=np.float32))
                    terminal_displacement = result["layer2_ux_post"] - result["layer2_ux_pre"]
                    passive_terminal_displacement = passive["layer2_ux_post"] - passive["layer2_ux_pre"]
                    terminal_corrected = terminal_displacement - passive_terminal_displacement
                    b_active = exact_b_spikes[anchor_id].any(axis=0).reshape(-1)
                    b_spec = b_specs.iloc[anchor_id]
                    for local_index, history in enumerate(rows_k.itertuples(index=False)):
                        rollout_row_id = int(storage_start + local_index)
                        l1_delta = deltas["delta_layer1_ux"][local_index].reshape(2, -1)
                        active_energy = float(np.linalg.norm(l1_delta[:, b_active])) if bool(b_active.any()) else 0.0
                        total_energy = float(np.linalg.norm(l1_delta))
                        scores = result["class_scores_post"][local_index]
                        target, competitor, margin = _target_score_metrics(scores, int(b_spec["B_label"]))
                        output_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "rollout_row_id": rollout_row_id,
                                "prefix_k": int(prefix_k),
                                "history_row_id": int(history.history_row_id),
                                "history_family_id": int(history.history_family_id),
                                "history_condition": str(history.history_condition),
                                "history_fold": int(history.history_fold),
                                "b_anchor_id": int(anchor_id),
                                "b_fold": int(b_spec["b_fold"]),
                                "B_label": int(b_spec["B_label"]),
                                "b_class_fold": int(b_spec["b_class_fold"]),
                                "B_replicate_id": int(b_spec["B_replicate_id"]),
                                "track": track,
                                "branch": branch,
                                "exact_b_tensor_sha256": str(inputs.tables["input_manifest"].iloc[anchor_id]["tensor_sha256"]),
                                "delta_l2_ux_norm": float(np.linalg.norm(deltas["delta_layer2_ux"][local_index])),
                                "delta_l1_ux_norm": total_energy,
                                "delta_l1_active_fraction": min(active_energy / max(total_energy, 1e-12), 1.0),
                                "layer1_spike_count": int(result["layer1_spike_count"][local_index]),
                                "layer2_spike_count": int(result["layer2_spike_count"][local_index]),
                                "layer3_spike_count": int(result["layer3_spike_count"][local_index]),
                                "layer1_early_spike_count": int(result["layer1_early_spike_count"][local_index]),
                                "layer2_early_spike_count": int(result["layer2_early_spike_count"][local_index]),
                                "prediction": int(result["prediction"][local_index]),
                                "target_class_score": target,
                                "strongest_competitor_score": competitor,
                                "target_margin": margin,
                                "replay_l1_mismatch_count": int(result["replay_l1_mismatch_count"][local_index]),
                            }
                        )
                        if track == "stsp_isolated" and branch == "free" and str(history.history_condition) in {"A", "C"}:
                            packed_event_rows.append(
                                result["layer2_presynaptic_event_bits"][local_index]
                            )
                            event_manifest_rows.append(
                                {
                                    "network_seed": int(ctx.cfg.network_seed),
                                    "event_row_id": len(event_manifest_rows),
                                    "rollout_row_id": rollout_row_id,
                                    "prefix_k": int(prefix_k),
                                    "history_family_id": int(history.history_family_id),
                                    "history_condition": str(history.history_condition),
                                    "b_anchor_id": int(anchor_id),
                                    "trace_window_steps": int(
                                        ctx.cfg.fixed_b_trace_window_steps
                                    ),
                                    "unpacked_shape": str(
                                        result["layer2_presynaptic_event_shape"]
                                    ),
                                    "packed_bytes": int(
                                        result["layer2_presynaptic_event_bits"].shape[1]
                                    ),
                                    "content_sha256": array_hash(
                                        result["layer2_presynaptic_event_bits"][local_index]
                                    ),
                                }
                            )
                        _append_trajectory_rows(
                            ctx,
                            trajectory_rows,
                            rollout_row_id=rollout_row_id,
                            prefix_k=prefix_k,
                            history=history,
                            b_spec=b_spec,
                            track=track,
                            branch=branch,
                            result=result,
                            passive=passive,
                            terminal_displacement=terminal_displacement[local_index],
                            terminal_corrected=terminal_corrected[local_index],
                            local_index=local_index,
                        )

    arrays = {
        name: np.stack(values, axis=0).astype(np.float32, copy=False)
        for name, values in arrays_by_name.items()
    }
    arrays["layer2_presynaptic_event_bits"] = (
        np.stack(packed_event_rows, axis=0).astype(np.uint8, copy=False)
        if packed_event_rows
        else np.empty((0, 0), dtype=np.uint8)
    )
    return (
        {
            "rollout_rows": pd.DataFrame(output_rows),
            "state_trajectory_rows": pd.DataFrame(trajectory_rows),
            "layer2_event_manifest": pd.DataFrame(event_manifest_rows),
        },
        arrays,
        {
            "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
            "delta_definition": "branch endpoint minus matched passive endpoint",
            "state_trajectory_checkpoints": ["pre", "early", "b_end", "post"],
            "event_storage": (
                "bit-packed actual Layer2-presynaptic pooled Layer1 event raster "
                "over the full locked B trace window for stsp-isolated free A/C rows"
            ),
            "full_voltage_movie_saved": False,
        },
    )


def build_swap_bank(
    ctx: ExperimentContext,
    specs: FixedBArtifact,
    inputs: FixedBArtifact,
    histories: FixedBArtifact,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    del specs
    swap_specs = histories.tables["swap_specs"]
    history_specs = histories.tables["history_specs"]
    b_specs = histories.tables["b_anchor_specs"].sort_values("b_anchor_id").reset_index(drop=True)
    exact_b_spikes = inputs.arrays["exact_b_spikes"]
    output_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []
    array_lists: dict[str, list[np.ndarray]] = {
        "delta_layer2_ux": [],
        "class_scores": [],
        "class_scores_early": [],
        "class_scores_b_end": [],
        "class_scores_post": [],
        "layer1_voltage_features": [],
        "layer1_event_features": [],
        "layer1_drive_features": [],
    }
    for prefix_k in sorted(int(value) for value in swap_specs["prefix_k"].unique()):
        histories_k = _history_rows_at_k(history_specs, prefix_k)
        native_boundary = _load_boundary(histories, prefix_k)
        index_by_family_condition = {
            (int(row.history_family_id), str(row.history_condition)): index
            for index, row in enumerate(histories_k.itertuples(index=False))
        }
        for scope in ("all_layers", "layer1_only"):
            specs_part = swap_specs.loc[
                swap_specs["prefix_k"].eq(prefix_k) & swap_specs["swap_scope"].eq(scope)
            ].sort_values("swap_spec_id").reset_index(drop=True)
            mixed_boundary = _mixed_swap_boundary(native_boundary, specs_part, index_by_family_condition, scope=scope)
            fast_hashes, stsp_hashes = _prestate_hashes(ctx, mixed_boundary, exact_b_spikes.shape[2:])
            layer1_donor_applied = True
            receiver_l23_preserved = True
            for local_index, swap in enumerate(specs_part.itertuples(index=False)):
                receiver_index = index_by_family_condition[
                    (int(swap.history_family_id), str(swap.receiver_condition))
                ]
                donor_index = index_by_family_condition[
                    (int(swap.history_family_id), str(swap.donor_condition))
                ]
                for state in STSP_STATE_KEYS:
                    layer1_donor_applied = layer1_donor_applied and np.array_equal(
                        mixed_boundary["layer1"][state][local_index],
                        native_boundary["layer1"][state][donor_index],
                    )
                    if scope == "layer1_only":
                        for layer_name in ("layer2", "layer3"):
                            receiver_l23_preserved = (
                                receiver_l23_preserved
                                and np.array_equal(
                                    mixed_boundary[layer_name][state][local_index],
                                    native_boundary[layer_name][state][receiver_index],
                                )
                            )
            isolation_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "prefix_k": int(prefix_k),
                    "swap_scope": scope,
                    "recipient_rows": int(len(specs_part)),
                    "unique_fast_state_hashes": int(len(set(fast_hashes))),
                    "unique_stsp_payload_hashes": int(len(set(stsp_hashes))),
                    "common_fast_state_sha256": str(fast_hashes[0]),
                    "fast_state_equalized": int(len(set(fast_hashes)) == 1),
                    "layer1_donor_stsp_applied": int(layer1_donor_applied),
                    "receiver_layer2_3_stsp_preserved": int(
                        receiver_l23_preserved
                    ),
                }
            )
            batch_size = len(specs_part)
            zero_seq = torch.zeros(
                (batch_size, int(ctx.cfg.fixed_b_stimulus_steps), *exact_b_spikes.shape[2:]),
                dtype=torch.bool,
                device=ctx.device,
            )
            passive = _run_branch(
                ctx,
                boundary=mixed_boundary,
                input_seq=zero_seq,
                current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                restore_mode="stsp_only",
                branch="passive",
                replay_l1_pooled=None,
                capture_l1_pooled=False,
                capture_strong_path=True,
                random_seed=int(ctx.cfg.network_seed) + 600_000 + 1000 * prefix_k + (0 if scope == "all_layers" else 1),
            )
            for anchor_id in range(len(exact_b_spikes)):
                result = _run_branch(
                    ctx,
                    boundary=mixed_boundary,
                    input_seq=torch.as_tensor(np.repeat(exact_b_spikes[anchor_id : anchor_id + 1], batch_size, axis=0), device=ctx.device),
                    current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                    restore_mode="stsp_only",
                    branch="free",
                    replay_l1_pooled=None,
                    capture_l1_pooled=False,
                    capture_strong_path=True,
                    random_seed=int(ctx.cfg.network_seed) + 610_000 + 10_000 * prefix_k + 100 * anchor_id + (0 if scope == "all_layers" else 1),
                )
                corrected = {
                    "delta_layer2_ux": result["layer2_ux"] - passive["layer2_ux"],
                    "class_scores": result["class_scores"],
                    "class_scores_early": result["class_scores_early"],
                    "class_scores_b_end": result["class_scores_b_end"],
                    "class_scores_post": result["class_scores_post"],
                    "layer1_voltage_features": result["layer1_voltage_features"] - passive["layer1_voltage_features"],
                    "layer1_event_features": result["layer1_event_features"] - passive["layer1_event_features"],
                    "layer1_drive_features": result["layer1_drive_features"] - passive["layer1_drive_features"],
                }
                storage_start = len(output_rows)
                for name, values in corrected.items():
                    array_lists[name].extend(np.asarray(values, dtype=np.float32))
                b_spec = b_specs.iloc[anchor_id]
                for local_index, swap in enumerate(specs_part.itertuples(index=False)):
                    output_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "swap_row_id": int(storage_start + local_index),
                            "swap_spec_id": int(swap.swap_spec_id),
                            "prefix_k": int(prefix_k),
                            "history_family_id": int(swap.history_family_id),
                            "history_fold": int(swap.history_fold),
                            "receiver_condition": str(swap.receiver_condition),
                            "donor_condition": str(swap.donor_condition),
                            "swap_scope": scope,
                            "is_own_sham": int(swap.is_own_sham),
                            "b_anchor_id": int(anchor_id),
                            "b_fold": int(b_spec["b_fold"]),
                            "delta_l2_ux_norm": float(np.linalg.norm(corrected["delta_layer2_ux"][local_index])),
                            "layer1_spike_count": int(result["layer1_spike_count"][local_index]),
                            "layer2_spike_count": int(result["layer2_spike_count"][local_index]),
                            "prediction": int(result["prediction"][local_index]),
                        }
                    )
    arrays = {name: np.stack(values, axis=0).astype(np.float32, copy=False) for name, values in array_lists.items()}
    return (
        {"swap_rows": pd.DataFrame(output_rows), "swap_isolation_audit": pd.DataFrame(isolation_rows)},
        arrays,
        {
            "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
            "swap_track": "STSP isolated with all non-STSP fast variables reset to an identical baseline",
            "swap_scopes": {
                "all_layers": "donor u/x restored at Layers 1-3",
                "layer1_only": "donor Layer1 u/x with receiver Layer2/3 u/x",
            },
        },
    )


def pack_event_bits(events: np.ndarray) -> np.ndarray:
    values = np.asarray(events, dtype=np.bool_)
    if values.ndim < 2:
        raise ValueError("Event arrays must have a row axis and at least one event axis")
    return np.packbits(values.reshape(values.shape[0], -1), axis=1, bitorder="little")


def unpack_event_bits(packed: np.ndarray, event_shape: Sequence[int]) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    bit_count = int(np.prod(tuple(int(value) for value in event_shape)))
    unpacked = np.unpackbits(values, axis=1, count=bit_count, bitorder="little")
    return unpacked.reshape(values.shape[0], *tuple(int(value) for value in event_shape)).astype(np.bool_, copy=False)


def _encode_source_rows(
    ctx: ExperimentContext,
    rows: pd.DataFrame,
    *,
    image_column: str,
    seed_column: str,
    steps: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    batch_size = max(1, int(ctx.cfg.batch_size))
    for start in range(0, len(rows), batch_size):
        part = rows.iloc[start : start + batch_size]
        seed_everything(int(part.iloc[0][seed_column]))
        images = torch.stack(
            [ctx.dataset[int(image_id)][0].detach().cpu().to(torch.float32) for image_id in part[image_column]],
            dim=0,
        ).to(ctx.device)
        chunks.append(encode_images(ctx.encoder, images, int(steps)).detach().cpu().to(torch.bool).numpy())
    return np.concatenate(chunks, axis=0).astype(np.bool_, copy=False)


def _simulate_history_rows(
    ctx: ExperimentContext,
    history_specs: pd.DataFrame,
    inputs: FixedBArtifact,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    id_column = "history_row_id" if "history_row_id" in history_specs.columns else "candidate_history_row_id"
    manifest = inputs.tables["history_input_manifest"].set_index("image_id")
    encoded_history = inputs.arrays["history_spikes"]
    arrays: dict[str, np.ndarray] = {}
    feature_rows: list[dict[str, Any]] = []
    for prefix_k in sorted(int(value) for value in history_specs["prefix_k"].unique()):
        rows_k = history_specs.loc[history_specs["prefix_k"].eq(prefix_k)].sort_values(id_column).reset_index(drop=True)
        state_chunks: dict[tuple[str, str], list[np.ndarray]] = {
            (layer, state): [] for layer in LAYER_KEYS for state in FAST_STATE_KEYS + STSP_STATE_KEYS
        }
        for start in range(0, len(rows_k), max(1, int(ctx.cfg.batch_size))):
            part = rows_k.iloc[start : start + max(1, int(ctx.cfg.batch_size))].reset_index(drop=True)
            first_shape = encoded_history.shape[2:]
            prepare_network_state(ctx.net, len(part), *[int(value) for value in first_shape])
            counts = {layer: np.zeros(len(part), dtype=np.int64) for layer in LAYER_KEYS}
            current_time = 0
            seed_everything(int(ctx.cfg.fixed_b_protocol_seed) + 300_000 + 1000 * prefix_k + start)
            for position in range(prefix_k):
                item_spikes = torch.zeros(
                    (len(part), int(ctx.cfg.fixed_b_item_steps), *first_shape),
                    dtype=torch.bool,
                    device=ctx.device,
                )
                for local_index, row in enumerate(part.itertuples(index=False)):
                    if str(row.history_condition) == "S0":
                        continue
                    image_ids = [int(value) for value in json.loads(str(row.sequence_image_ids))]
                    source_index = int(manifest.loc[image_ids[position], "row_index"])
                    item_spikes[local_index] = torch.as_tensor(encoded_history[source_index], device=ctx.device)
                for step in range(int(item_spikes.shape[1])):
                    s1, _, s2, _, s3, _ = advance_network_step(
                        ctx.net,
                        item_spikes[:, step],
                        current_time=current_time,
                    )
                    counts["layer1"] += s1.reshape(len(part), -1).sum(dim=1).cpu().numpy().astype(np.int64)
                    counts["layer2"] += s2.reshape(len(part), -1).sum(dim=1).cpu().numpy().astype(np.int64)
                    counts["layer3"] += s3.reshape(len(part), -1).sum(dim=1).cpu().numpy().astype(np.int64)
                    current_time += 1
                zero = torch.zeros((len(part), *first_shape), dtype=torch.bool, device=ctx.device)
                for _ in range(int(ctx.cfg.fixed_b_inter_delay_steps)):
                    s1, _, s2, _, s3, _ = advance_network_step(
                        ctx.net,
                        zero,
                        current_time=current_time,
                    )
                    counts["layer1"] += s1.reshape(len(part), -1).sum(dim=1).cpu().numpy().astype(np.int64)
                    counts["layer2"] += s2.reshape(len(part), -1).sum(dim=1).cpu().numpy().astype(np.int64)
                    counts["layer3"] += s3.reshape(len(part), -1).sum(dim=1).cpu().numpy().astype(np.int64)
                    current_time += 1
            boundary = snapshot_boundary_state(ctx.net)
            for layer in LAYER_KEYS:
                for state in FAST_STATE_KEYS + STSP_STATE_KEYS:
                    state_chunks[(layer, state)].append(boundary[layer][state].detach().cpu().numpy())
            feature_rows.extend(_prestate_feature_rows(ctx, part, boundary, counts, id_column=id_column))
        for (layer, state), chunks in state_chunks.items():
            arrays[_boundary_key(prefix_k, layer, state)] = np.concatenate(chunks, axis=0)
    return arrays, pd.DataFrame(feature_rows)


def _candidate_overlap_table(
    ctx: ExperimentContext,
    candidate_specs: pd.DataFrame,
    candidate_arrays: Mapping[str, np.ndarray],
    inputs: FixedBArtifact,
) -> pd.DataFrame:
    b_counts = torch.as_tensor(inputs.arrays["exact_b_spikes"].sum(axis=1), dtype=torch.float32, device=ctx.device)
    kernel = ctx.net.layer1.kernels.detach()
    baseline = float(ctx.net.layer1.stsp_U)
    rows: list[dict[str, Any]] = []
    for prefix_k in sorted(int(value) for value in candidate_specs["prefix_k"].unique()):
        part = candidate_specs.loc[candidate_specs["prefix_k"].eq(prefix_k)].sort_values("candidate_history_row_id").reset_index(drop=True)
        u = candidate_arrays[_boundary_key(prefix_k, "layer1", "u")]
        x = candidate_arrays[_boundary_key(prefix_k, "layer1", "x")]
        for local_index, history in enumerate(part.itertuples(index=False)):
            gain_delta = torch.as_tensor(u[local_index] * x[local_index] - baseline, dtype=torch.float32, device=ctx.device)
            weighted = b_counts * gain_delta.unsqueeze(0)
            projected = F.conv2d(weighted, kernel, stride=ctx.net.layer1.stride, padding=ctx.net.layer1.padding)
            flat = projected.reshape(len(projected), -1)
            norms = torch.linalg.vector_norm(flat, dim=1).detach().cpu().numpy()
            signed = flat.sum(dim=1).detach().cpu().numpy()
            for anchor_id in range(len(projected)):
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "candidate_family_id": int(history.candidate_family_id),
                        "history_condition": str(history.history_condition),
                        "prefix_k": int(prefix_k),
                        "b_anchor_id": int(anchor_id),
                        "projected_overlap": float(norms[anchor_id]),
                        "projected_signed_mass": float(signed[anchor_id]),
                    }
                )
    return pd.DataFrame(rows)


def _run_branch(
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
    net = ctx.net if net_override is None else net_override
    seed_everything(int(random_seed))
    batch_size, stimulus_steps, channels, height, width = input_seq.shape
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    if not skip_restore:
        restore_boundary_state(net, boundary, layer_input_shapes, mode=restore_mode, device=ctx.device)
    net.layer3.reset_decision_state()
    with torch.no_grad():
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
    zero = torch.zeros((batch_size, channels, height, width), dtype=torch.bool, device=ctx.device)
    total_steps = int(stimulus_steps + ctx.cfg.fixed_b_post_steps)
    early_cutoff = min(int(ctx.cfg.fixed_b_early_window_steps), int(stimulus_steps))
    trace_cutoff = min(
        int(ctx.cfg.fixed_b_trace_window_steps),
        int(stimulus_steps),
    )
    counts = {layer: torch.zeros(batch_size, dtype=torch.int64, device=ctx.device) for layer in LAYER_KEYS}
    l1_early = torch.zeros(batch_size, dtype=torch.int64, device=ctx.device)
    l2_early = torch.zeros_like(l1_early)
    mismatch = torch.zeros_like(l1_early)
    replay_tensor = None if replay_l1_pooled is None else torch.as_tensor(replay_l1_pooled, device=ctx.device)
    captured_pooled: list[torch.Tensor] = []
    captured_early: list[torch.Tensor] = []
    captured_layer2_presynaptic: list[torch.Tensor] = []
    strong_maps: dict[str, torch.Tensor | None] = {name: None for name in ("drive", "voltage", "inhibition", "events")}
    checkpoint_ux = {"pre": layer_ux_checkpoint(net.layer2, batch_size)}
    checkpoint_scores = {"pre": _class_scores_checkpoint(net)}
    checkpoint_counts = {"pre": tuple(np.zeros(batch_size, dtype=np.int64) for _ in range(3))}
    layer1_pre = layer_ux_checkpoint(net.layer1, batch_size)
    layer3_pre = layer_ux_checkpoint(net.layer3, batch_size)
    for local_step in range(total_steps):
        external = input_seq[:, local_step] if local_step < stimulus_steps and branch != "passive" else zero
        replay_step = None if replay_tensor is None else replay_tensor[:, local_step].to(torch.float32)
        monitor_now = bool(capture_strong_path and local_step < early_cutoff)
        s1, s1_p, s2, _, s3, monitor = advance_network_step(
            net,
            external,
            current_time=current_time + local_step,
            layer2_replay_input=replay_step,
            layer3_time=local_step,
            monitor_layer1=monitor_now,
        )
        local_counts = {
            "layer1": s1.reshape(batch_size, -1).sum(dim=1).to(torch.int64),
            "layer2": s2.reshape(batch_size, -1).sum(dim=1).to(torch.int64),
            "layer3": s3.reshape(batch_size, -1).sum(dim=1).to(torch.int64),
        }
        for layer in LAYER_KEYS:
            counts[layer] += local_counts[layer]
        if local_step < early_cutoff:
            l1_early += local_counts["layer1"]
            l2_early += local_counts["layer2"]
            if capture_strong_path:
                captured_early.append(s1.detach().to(torch.bool).cpu())
                gain = monitor.get("stsp_gain")
                effective = external.float() if gain is None else external.float() * gain
                drive = F.conv2d(effective, net.layer1.kernels, stride=net.layer1.stride, padding=net.layer1.padding)
                values = {
                    "drive": drive,
                    "voltage": monitor["v_effective"],
                    "inhibition": monitor["inh_before"],
                    "events": s1.float(),
                }
                for name, value in values.items():
                    strong_maps[name] = value.detach().clone() if strong_maps[name] is None else strong_maps[name] + value.detach()
        if replay_tensor is not None:
            mismatch += (s1_p.to(torch.bool) != replay_tensor[:, local_step].to(torch.bool)).reshape(batch_size, -1).sum(dim=1)
        if capture_l1_pooled:
            captured_pooled.append(s1_p.detach().to(torch.bool).cpu())
        if capture_layer2_presynaptic_trace and local_step < trace_cutoff:
            layer2_input = s1_p if replay_step is None else replay_step
            captured_layer2_presynaptic.append(
                layer2_input.detach().to(torch.bool).cpu()
            )
        completed_steps = local_step + 1
        if completed_steps == early_cutoff:
            _capture_branch_checkpoint(net, batch_size, "early", checkpoint_ux, checkpoint_scores, checkpoint_counts, counts)
        if completed_steps == stimulus_steps:
            _capture_branch_checkpoint(net, batch_size, "b_end", checkpoint_ux, checkpoint_scores, checkpoint_counts, counts)
    _capture_branch_checkpoint(net, batch_size, "post", checkpoint_ux, checkpoint_scores, checkpoint_counts, counts)
    layer1_post = layer_ux_checkpoint(net.layer1, batch_size)
    layer3_post = layer_ux_checkpoint(net.layer3, batch_size)
    layer2_g = _gain_from_ux(checkpoint_ux["post"])
    firing_times = net.layer3.firing_times.detach()
    fired = torch.isfinite(firing_times).any(dim=1)
    _, indices = firing_times.min(dim=1)
    prediction = (indices // int(net.layer3.neurons_per_class)).to(torch.int64)
    prediction[~fired] = -1
    out: dict[str, Any] = {
        "layer1_ux": layer1_post,
        "layer2_ux": checkpoint_ux["post"],
        "layer2_g": layer2_g,
        "layer3_ux": layer3_post,
        "layer1_ux_pre": layer1_pre,
        "layer3_ux_pre": layer3_pre,
        "class_scores": checkpoint_scores["post"],
        "prediction": prediction.cpu().numpy(),
        "layer1_spike_count": checkpoint_counts["post"][0],
        "layer2_spike_count": checkpoint_counts["post"][1],
        "layer3_spike_count": checkpoint_counts["post"][2],
        "layer1_early_spike_count": l1_early.cpu().numpy(),
        "layer2_early_spike_count": l2_early.cpu().numpy(),
        "replay_l1_mismatch_count": mismatch.cpu().numpy(),
    }
    for checkpoint in ("pre", "early", "b_end", "post"):
        out[f"layer2_ux_{checkpoint}"] = checkpoint_ux[checkpoint]
        out[f"class_scores_{checkpoint}"] = checkpoint_scores[checkpoint]
        out[f"layer1_spike_count_{checkpoint}"] = checkpoint_counts[checkpoint][0]
        out[f"layer2_spike_count_{checkpoint}"] = checkpoint_counts[checkpoint][1]
        out[f"layer3_spike_count_{checkpoint}"] = checkpoint_counts[checkpoint][2]
    if capture_l1_pooled:
        out["l1_pooled_spikes"] = torch.stack(captured_pooled, dim=1).numpy()
    if capture_strong_path:
        for name in ("drive", "voltage", "inhibition"):
            out[f"layer1_{name}_features"] = _map_projection_features(strong_maps[name]).cpu().numpy()
        out["layer1_event_features"] = _map_event_features(strong_maps["events"]).cpu().numpy()
        event_array = torch.stack(captured_early, dim=1).numpy()
        out["layer1_early_event_bits"] = pack_event_bits(event_array)
        out["layer1_early_event_shape"] = tuple(int(value) for value in event_array.shape[1:])
    if capture_layer2_presynaptic_trace:
        event_array = torch.stack(captured_layer2_presynaptic, dim=1).numpy()
        out["layer2_presynaptic_event_bits"] = pack_event_bits(event_array)
        out["layer2_presynaptic_event_shape"] = tuple(
            int(value) for value in event_array.shape[1:]
        )
    return out


def _capture_branch_checkpoint(
    net: Any,
    batch_size: int,
    name: str,
    ux: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    checkpoint_counts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    counts: Mapping[str, torch.Tensor],
) -> None:
    ux[name] = layer_ux_checkpoint(net.layer2, batch_size)
    scores[name] = _class_scores_checkpoint(net)
    checkpoint_counts[name] = tuple(counts[layer].detach().cpu().numpy().copy() for layer in LAYER_KEYS)


def _append_trajectory_rows(
    ctx: ExperimentContext,
    target: list[dict[str, Any]],
    *,
    rollout_row_id: int,
    prefix_k: int,
    history: Any,
    b_spec: pd.Series,
    track: str,
    branch: str,
    result: Mapping[str, np.ndarray],
    passive: Mapping[str, np.ndarray],
    terminal_displacement: np.ndarray,
    terminal_corrected: np.ndarray,
    local_index: int,
) -> None:
    checkpoint_steps = {
        "pre": 0,
        "early": min(int(ctx.cfg.fixed_b_early_window_steps), int(ctx.cfg.fixed_b_stimulus_steps)),
        "b_end": int(ctx.cfg.fixed_b_stimulus_steps),
        "post": int(ctx.cfg.fixed_b_stimulus_steps + ctx.cfg.fixed_b_post_steps),
    }
    for checkpoint in ("pre", "early", "b_end", "post"):
        state = result[f"layer2_ux_{checkpoint}"][local_index]
        passive_state = passive[f"layer2_ux_{checkpoint}"][local_index]
        displacement = state - result["layer2_ux_pre"][local_index]
        passive_displacement = passive_state - passive["layer2_ux_pre"][local_index]
        corrected = displacement - passive_displacement
        scores = result[f"class_scores_{checkpoint}"][local_index]
        target_score, competitor, margin = _target_score_metrics(scores, int(b_spec["B_label"]))
        target.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "trajectory_row_id": len(target),
                "rollout_row_id": int(rollout_row_id),
                "prefix_k": int(prefix_k),
                "history_row_id": int(history.history_row_id),
                "history_family_id": int(history.history_family_id),
                "history_condition": str(history.history_condition),
                "history_fold": int(history.history_fold),
                "b_anchor_id": int(b_spec["b_anchor_id"]),
                "b_fold": int(b_spec["b_fold"]),
                "B_label": int(b_spec["B_label"]),
                "track": track,
                "branch": branch,
                "checkpoint": checkpoint,
                "elapsed_steps": int(checkpoint_steps[checkpoint]),
                "layer2_ux_displacement_norm": float(np.linalg.norm(displacement)),
                "layer2_ux_passive_corrected_norm": float(np.linalg.norm(corrected)),
                "layer2_ux_terminal_alignment": _vector_cosine(displacement, terminal_displacement),
                "layer2_ux_corrected_terminal_alignment": _vector_cosine(corrected, terminal_corrected),
                "layer1_cumulative_spike_count": int(result[f"layer1_spike_count_{checkpoint}"][local_index]),
                "layer2_cumulative_spike_count": int(result[f"layer2_spike_count_{checkpoint}"][local_index]),
                "layer3_cumulative_spike_count": int(result[f"layer3_spike_count_{checkpoint}"][local_index]),
                "target_class_score": target_score,
                "strongest_competitor_score": competitor,
                "target_margin": margin,
                "class_score_vector": json.dumps([float(value) for value in scores], separators=(",", ":")),
            }
        )


def _prestate_feature_rows(
    ctx: ExperimentContext,
    rows: pd.DataFrame,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    counts: Mapping[str, np.ndarray],
    *,
    id_column: str,
) -> list[dict[str, Any]]:
    output = []
    for local_index, history in enumerate(rows.itertuples(index=False)):
        record: dict[str, Any] = {
            "network_seed": int(ctx.cfg.network_seed),
            id_column: int(getattr(history, id_column)),
            "candidate_family_id": int(getattr(history, "candidate_family_id")),
            "history_condition": str(history.history_condition),
            "prefix_k": int(history.prefix_k),
        }
        if hasattr(history, "history_family_id"):
            record["history_family_id"] = int(history.history_family_id)
        if hasattr(history, "balance_stratum"):
            record["balance_stratum"] = int(history.balance_stratum)
        for layer in LAYER_KEYS:
            u = boundary[layer]["u"][local_index].detach().cpu().numpy().astype(np.float64, copy=False)
            x = boundary[layer]["x"][local_index].detach().cpu().numpy().astype(np.float64, copy=False)
            g = u * x
            prefix = f"{layer}_"
            record.update(
                {
                    f"{prefix}u_mean": float(u.mean()),
                    f"{prefix}x_mean": float(x.mean()),
                    f"{prefix}g_mean": float(g.mean()),
                    f"{prefix}u_norm": float(np.linalg.norm(u)),
                    f"{prefix}x_norm": float(np.linalg.norm(x)),
                    f"{prefix}g_norm": float(np.linalg.norm(g)),
                    f"{prefix}headroom_mean": float(((1.0 - u) + x).mean()),
                    f"{prefix}distance_to_bounds": float(np.minimum.reduce([u, 1.0 - u, x, 1.0 - x]).mean()),
                    f"{prefix}support_mass": float(np.maximum(g - float(getattr(ctx.net, layer).stsp_U), 0.0).sum()),
                    f"{prefix}prior_spike_count": int(counts[layer][local_index]),
                }
            )
        output.append(record)
    return output


def _restoration_audit(
    ctx: ExperimentContext,
    history_specs: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    exact_b: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix_k in sorted(int(value) for value in history_specs["prefix_k"].unique()):
        rows_k = _history_rows_at_k(history_specs, prefix_k)
        boundary = _load_boundary_from_arrays(arrays, prefix_k)
        batch_size = len(rows_k)
        repeated_b = torch.as_tensor(np.repeat(exact_b[None, ...], batch_size, axis=0), device=ctx.device)
        zero = torch.zeros_like(repeated_b)
        branch_results: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        for branch, sequence in (("free", repeated_b), ("passive", zero)):
            layer_shapes = build_layer_input_shapes(ctx.net, batch_size, *[int(value) for value in exact_b.shape[1:]])
            restore_boundary_state(
                ctx.net,
                boundary,
                layer_shapes,
                mode="full_boundary",
                device=ctx.device,
            )
            native_net = copy.deepcopy(ctx.net)
            native = _run_branch(
                ctx,
                boundary=boundary,
                input_seq=sequence,
                current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                restore_mode="full_boundary",
                branch=branch,
                replay_l1_pooled=None,
                capture_l1_pooled=False,
                capture_strong_path=False,
                random_seed=int(ctx.cfg.network_seed) + 700_000 + 1000 * prefix_k + (0 if branch == "free" else 1),
                net_override=native_net,
                skip_restore=True,
            )
            restored = _run_branch(
                ctx,
                boundary=boundary,
                input_seq=sequence,
                current_time=int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                restore_mode="full_boundary",
                branch=branch,
                replay_l1_pooled=None,
                capture_l1_pooled=False,
                capture_strong_path=False,
                random_seed=int(ctx.cfg.network_seed) + 700_000 + 1000 * prefix_k + (0 if branch == "free" else 1),
            )
            branch_results[branch] = (native, restored)
            del native_net
        native_corrected = branch_results["free"][0]["layer2_ux"] - branch_results["passive"][0]["layer2_ux"]
        separation_by_family: dict[int, float] = {}
        for family_id, part in rows_k.loc[rows_k["history_condition"].isin(["A", "C"])].groupby("history_family_id"):
            lookup = {str(row.history_condition): int(index) for index, row in part.iterrows()}
            separation_by_family[int(family_id)] = float(np.linalg.norm(native_corrected[lookup["A"]] - native_corrected[lookup["C"]]))
        for branch, (native, restored) in branch_results.items():
            error = np.abs(native["layer2_ux"] - restored["layer2_ux"])
            for local_index, history in enumerate(rows_k.itertuples(index=False)):
                separation = separation_by_family.get(int(history.history_family_id), 0.0)
                maximum = float(error[local_index].max())
                ratio = maximum / max(separation, 1e-12)
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "history_row_id": int(history.history_row_id),
                        "history_family_id": int(history.history_family_id),
                        "history_condition": str(history.history_condition),
                        "branch": branch,
                        "max_abs_layer2_ux_error": maximum,
                        "mean_abs_layer2_ux_error": float(error[local_index].mean()),
                        "native_A_C_separation": separation,
                        "normalized_restoration_error": ratio,
                        "prediction_equal": int(native["prediction"][local_index] == restored["prediction"][local_index]),
                        "spike_counts_equal": int(
                            native["layer1_spike_count"][local_index] == restored["layer1_spike_count"][local_index]
                            and native["layer2_spike_count"][local_index] == restored["layer2_spike_count"][local_index]
                        ),
                        "restoration_margin_pass": int(ratio <= 0.10 + 1e-12),
                    }
                )
    return pd.DataFrame(rows)


def _prestate_hashes(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, np.ndarray]],
    input_shape: Sequence[int],
) -> tuple[list[str], list[str]]:
    batch_size = int(next(iter(boundary["layer1"].values())).shape[0])
    layer_shapes = build_layer_input_shapes(ctx.net, batch_size, *[int(value) for value in input_shape])
    restore_boundary_state(
        ctx.net,
        boundary,
        layer_shapes,
        mode="stsp_only",
        device=ctx.device,
    )
    fast_hashes: list[str] = []
    stsp_hashes: list[str] = []
    for row_index in range(batch_size):
        fast = hashlib.sha256()
        stsp = hashlib.sha256()
        for layer_name in LAYER_KEYS:
            layer = getattr(ctx.net, layer_name)
            for state_name in FAST_STATE_KEYS:
                tensor = layer.lateral_inh.inh_trace if state_name == "inh_trace" else getattr(layer, state_name)
                fast.update(np.ascontiguousarray(tensor[row_index].detach().cpu().numpy()).tobytes())
            for state_name in STSP_STATE_KEYS:
                stsp.update(np.ascontiguousarray(getattr(layer, f"{state_name}_pre")[row_index].detach().cpu().numpy()).tobytes())
        fast_hashes.append(fast.hexdigest())
        stsp_hashes.append(stsp.hexdigest())
    return fast_hashes, stsp_hashes


def _load_boundary(
    artifact: FixedBArtifact,
    prefix_k: int,
    row_indices: list[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    return _load_boundary_from_arrays(artifact.arrays, prefix_k, row_indices=row_indices)


def _load_boundary_from_arrays(
    arrays: Mapping[str, np.ndarray],
    prefix_k: int,
    row_indices: list[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    boundary: dict[str, dict[str, np.ndarray]] = {layer: {} for layer in LAYER_KEYS}
    for layer in LAYER_KEYS:
        for state in FAST_STATE_KEYS + STSP_STATE_KEYS:
            key = _boundary_key(prefix_k, layer, state)
            if key not in arrays:
                continue
            value = arrays[key]
            boundary[layer][state] = value if row_indices is None else value[np.asarray(row_indices, dtype=np.int64)]
    return boundary


def _mixed_swap_boundary(
    boundary: Mapping[str, Mapping[str, np.ndarray]],
    specs_part: pd.DataFrame,
    index_by_family_condition: Mapping[tuple[int, str], int],
    *,
    scope: str,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {layer: {} for layer in LAYER_KEYS}
    for layer in LAYER_KEYS:
        for state in FAST_STATE_KEYS + STSP_STATE_KEYS:
            if state not in boundary[layer]:
                continue
            source_rows = []
            for row in specs_part.itertuples(index=False):
                receiver = index_by_family_condition[(int(row.history_family_id), str(row.receiver_condition))]
                donor = index_by_family_condition[(int(row.history_family_id), str(row.donor_condition))]
                use_donor = state in STSP_STATE_KEYS and (scope == "all_layers" or layer == "layer1")
                source_rows.append(boundary[layer][state][donor if use_donor else receiver])
            output[layer][state] = np.stack(source_rows, axis=0)
    return output


def _map_projection_features(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise RuntimeError("Missing bounded projection accumulator")
    flat = value.reshape(value.shape[0], -1)
    scalars = torch.stack(
        [flat.mean(dim=1), torch.linalg.vector_norm(flat, dim=1), flat.amax(dim=1), flat.amin(dim=1)],
        dim=1,
    )
    channel = value.mean(dim=(2, 3))
    spatial = value.mean(dim=1).reshape(value.shape[0], -1)
    return torch.cat([scalars, channel, spatial], dim=1).detach().cpu().to(torch.float32)


def _map_event_features(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise RuntimeError("Missing event accumulator")
    flat = value.reshape(value.shape[0], -1)
    scalars = torch.stack(
        [flat.sum(dim=1), torch.linalg.vector_norm(flat, dim=1), (flat > 0).sum(dim=1).float()],
        dim=1,
    )
    channel = value.sum(dim=(2, 3))
    spatial = value.sum(dim=1).reshape(value.shape[0], -1)
    return torch.cat([scalars, channel, spatial], dim=1).detach().cpu().to(torch.float32)


def _gain_from_ux(ux: np.ndarray) -> np.ndarray:
    midpoint = ux.shape[1] // 2
    return (ux[:, :midpoint] * ux[:, midpoint:]).astype(np.float32, copy=False)


def _class_scores_checkpoint(net: Any) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage().detach()
    return grouped.max(dim=2).values.cpu().numpy().astype(np.float32, copy=False)


def _target_score_metrics(scores: np.ndarray, target_label: int) -> tuple[float, float, float]:
    target = float(scores[int(target_label)])
    competitor = float(np.max(np.concatenate([scores[:target_label], scores[target_label + 1 :]])))
    return target, competitor, float(target - competitor)


def _vector_cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else float("nan")


def _event_feature_rows(
    ctx: ExperimentContext,
    prefix_k: int,
    anchor_id: int,
    count_map: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        {
            "network_seed": int(ctx.cfg.network_seed),
            "prefix_k": int(prefix_k),
            "b_anchor_id": int(anchor_id),
            "feature_name": "total_event_count",
            "feature_index": 0,
            "value": float(count_map.sum()),
        }
    ]
    rows.extend(
        {
            "network_seed": int(ctx.cfg.network_seed),
            "prefix_k": int(prefix_k),
            "b_anchor_id": int(anchor_id),
            "feature_name": "channel_event_count",
            "feature_index": int(index),
            "value": float(value),
        }
        for index, value in enumerate(count_map.sum(axis=(1, 2)))
    )
    rows.extend(
        {
            "network_seed": int(ctx.cfg.network_seed),
            "prefix_k": int(prefix_k),
            "b_anchor_id": int(anchor_id),
            "feature_name": "spatial_event_count",
            "feature_index": int(index),
            "value": float(value),
        }
        for index, value in enumerate(count_map.sum(axis=0).reshape(-1))
    )
    return rows


def _history_rows_at_k(history_specs: pd.DataFrame, prefix_k: int) -> pd.DataFrame:
    return history_specs.loc[history_specs["prefix_k"].eq(prefix_k)].sort_values("history_row_id").reset_index(drop=True)


def _boundary_key(prefix_k: int, layer: str, state: str) -> str:
    return f"k{int(prefix_k)}__{layer}__{state}"


__all__ = [
    "build_exact_b_input_bank",
    "build_history_boundary_bank",
    "build_replay_bank",
    "build_rollout_bank",
    "build_swap_bank",
    "pack_event_bits",
    "unpack_event_bits",
]
