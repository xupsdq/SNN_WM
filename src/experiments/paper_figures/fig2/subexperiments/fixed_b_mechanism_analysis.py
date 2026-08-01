from __future__ import annotations

import ast
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig2.fixed_b_artifacts import (
    FixedBArtifact,
    array_hash,
)
from src.experiments.paper_figures.fig2.fixed_b_protocol import protocol_digest
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import (
    unpack_event_bits,
)
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_specs import (
    FIXED_B_SCHEMA_VERSION,
)
from src.experiments.paper_figures.fig2.types import ExperimentContext


NEAR_ZERO = 1e-12


def analyze_fixed_b_mechanism_single_seed(
    ctx: ExperimentContext,
    specs: FixedBArtifact,
    inputs: FixedBArtifact,
    histories: FixedBArtifact,
    replay: FixedBArtifact,
    rollouts: FixedBArtifact,
    swaps: FixedBArtifact,
    *,
    protocol: FixedBArtifact,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Analyze the mechanism-aligned fixed-B v4 endpoints for one network.

    Network-level inference is deliberately deferred to the untouched cohort.
    Seed 1000 is used only to verify engineering validity and freeze the
    protocol.
    """

    del specs, histories, replay
    decomposition_cells, event_cells = _decomposition_and_event_metrics(
        ctx,
        rollouts,
    )
    decomposition_summary = _summarize_decomposition(decomposition_cells)
    event_summary = _summarize_events(event_cells)
    swap_cells = _swap_metrics(swaps)
    swap_summary = _summarize_swaps(swap_cells)
    engineering = _engineering_gates(
        ctx,
        inputs=inputs,
        rollouts=rollouts,
        swaps=swaps,
        decomposition_cells=decomposition_cells,
        event_cells=event_cells,
        swap_cells=swap_cells,
    )
    network_scalars = _network_scalars(
        decomposition_summary,
        event_summary,
        swap_summary,
        engineering,
    )
    checklist = _prediction_checklist(
        decomposition_summary,
        event_summary,
        swap_summary,
        engineering,
    )
    claim_ledger = _claim_ledger()
    digest = protocol_digest(protocol)
    engineering_valid = bool(engineering["passed"].eq(1).all())
    core_swap_coverage = swap_summary.loc[
        swap_summary["swap_scope"].eq("layer1_only")
        & swap_summary["endpoint"].isin(
            ["layer2_update", "early_class_score"]
        ),
        "valid_coverage",
    ]
    minimum_coverage = float(
        min(
            decomposition_summary["valid_coverage"].min(),
            event_summary["valid_coverage"].min(),
            core_swap_coverage.min(),
        )
    )
    decision = {
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "network_seed": int(ctx.cfg.network_seed),
        "seed_role": (
            "development_protocol_alignment"
            if int(ctx.cfg.network_seed) == 1000
            else "untouched_confirmatory_network"
        ),
        "protocol_digest": digest,
        "engineering_valid": engineering_valid,
        "minimum_valid_coverage": minimum_coverage,
        "network_level_inference_performed": False,
        "optional_stopping": False,
        "outcome_based_exclusions": False,
        "continuation_eligible": engineering_valid and minimum_coverage >= 0.95,
        "verdict": (
            "development_engineering_valid"
            if int(ctx.cfg.network_seed) == 1000 and engineering_valid
            else (
                "development_engineering_invalid"
                if int(ctx.cfg.network_seed) == 1000
                else (
                    "confirmatory_network_complete"
                    if engineering_valid
                    else "confirmatory_network_engineering_invalid"
                )
            )
        ),
        "claim_boundary": (
            "No scientific fixed-B claim is decided within one network; "
            "the independently trained network is the confirmatory inference unit."
        ),
        # Compatibility fields for the transition audit writer.
        "existing_chain_valid": engineering_valid,
        "core_development_pass": False,
        "strong_development_pass": False,
        "eligible_tracks": ["confirmatory_v4"] if engineering_valid else [],
    }
    tables = {
        "fixed_b_decomposition_cell_metrics": decomposition_cells,
        "fixed_b_decomposition_summary": decomposition_summary,
        "fixed_b_event_gamma_cell_metrics": event_cells,
        "fixed_b_event_gamma_summary": event_summary,
        "fixed_b_swap_cell_metrics": swap_cells,
        "fixed_b_swap_summary": swap_summary,
        "fixed_b_engineering_gates": engineering,
        "fixed_b_network_scalars": network_scalars,
        "fixed_b_prediction_checklist": checklist,
        "fixed_b_claim_ledger": claim_ledger,
    }
    return tables, decision


def _decomposition_and_event_metrics(
    ctx: ExperimentContext,
    rollouts: FixedBArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rollouts.tables["rollout_rows"]
    rows = rows.loc[
        rows["track"].eq("stsp_isolated")
        & rows["history_condition"].isin(["A", "C"])
        & rows["branch"].isin(["free", "replay"])
    ].copy()
    key_columns = [
        "prefix_k",
        "history_family_id",
        "b_anchor_id",
        "history_condition",
        "branch",
    ]
    if rows.duplicated(key_columns).any():
        raise ValueError("Fixed-B rollout rows are not unique at the A/C branch-cell level")
    row_lookup = {
        (
            int(row.prefix_k),
            int(row.history_family_id),
            int(row.b_anchor_id),
            str(row.history_condition),
            str(row.branch),
        ): int(row.rollout_row_id)
        for row in rows.itertuples(index=False)
    }
    vectors = np.asarray(rollouts.arrays["delta_layer2_ux"], dtype=np.float64)

    event_manifest = rollouts.tables["layer2_event_manifest"].copy()
    event_lookup = {
        (
            int(row.prefix_k),
            int(row.history_family_id),
            int(row.b_anchor_id),
            str(row.history_condition),
        ): int(row.event_row_id)
        for row in event_manifest.itertuples(index=False)
    }
    packed_events = rollouts.arrays["layer2_presynaptic_event_bits"]
    shapes = {
        tuple(int(value) for value in ast.literal_eval(str(value)))
        for value in event_manifest["unpacked_shape"]
    }
    if len(shapes) != 1:
        raise ValueError(f"Fixed-B full event traces have inconsistent shapes: {sorted(shapes)}")
    event_shape = next(iter(shapes))
    expected_trace_steps = int(ctx.cfg.fixed_b_trace_window_steps)
    if not event_shape or int(event_shape[0]) != expected_trace_steps:
        raise ValueError(
            "Fixed-B full event trace window mismatch: "
            f"expected={expected_trace_steps}, found={event_shape}"
        )

    decomposition_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    prefix_depths = sorted(int(value) for value in rows["prefix_k"].unique())
    history_families = sorted(int(value) for value in rows["history_family_id"].unique())
    anchors = sorted(int(value) for value in rows["b_anchor_id"].unique())
    for prefix_k in prefix_depths:
        for history_family_id in history_families:
            for b_anchor_id in anchors:
                ids = {
                    (condition, branch): row_lookup[
                        (
                            prefix_k,
                            history_family_id,
                            b_anchor_id,
                            condition,
                            branch,
                        )
                    ]
                    for condition in ("A", "C")
                    for branch in ("free", "replay")
                }
                free_a = vectors[ids[("A", "free")]]
                free_c = vectors[ids[("C", "free")]]
                replay_a = vectors[ids[("A", "replay")]]
                replay_c = vectors[ids[("C", "replay")]]
                total = free_a - free_c
                local = replay_a - replay_c
                gamma = (free_a - replay_a) - (free_c - replay_c)
                closure = total - local - gamma
                common_scale = 0.5 * (
                    np.linalg.norm(free_a) + np.linalg.norm(free_c)
                )
                valid = bool(common_scale > NEAR_ZERO and np.isfinite(gamma).all())
                decomposition_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": prefix_k,
                        "history_family_id": history_family_id,
                        "b_anchor_id": b_anchor_id,
                        "same_B_common_update_cosine": _cosine(free_a, free_c),
                        "free_A_update_norm": float(np.linalg.norm(free_a)),
                        "free_C_update_norm": float(np.linalg.norm(free_c)),
                        "total_contrast_norm": float(np.linalg.norm(total)),
                        "local_replay_contrast_norm": float(np.linalg.norm(local)),
                        "processing_residual_gamma_norm": float(np.linalg.norm(gamma)),
                        "total_contrast_fraction": _safe_ratio(
                            np.linalg.norm(total), common_scale
                        ),
                        "local_replay_fraction": _safe_ratio(
                            np.linalg.norm(local), common_scale
                        ),
                        "processing_residual_gamma_energy_fraction": _safe_ratio(
                            np.linalg.norm(gamma), common_scale
                        ),
                        "decomposition_absolute_error": float(
                            np.linalg.norm(closure)
                        ),
                        "decomposition_relative_error": _safe_ratio(
                            np.linalg.norm(closure),
                            max(np.linalg.norm(total), common_scale),
                        ),
                        "valid": int(valid),
                    }
                )

                event_a_id = event_lookup[
                    (prefix_k, history_family_id, b_anchor_id, "A")
                ]
                event_c_id = event_lookup[
                    (prefix_k, history_family_id, b_anchor_id, "C")
                ]
                events = unpack_event_bits(
                    packed_events[[event_a_id, event_c_id]],
                    event_shape,
                )
                changed = np.any(events[0] != events[1], axis=0).reshape(-1)
                gamma_by_coordinate = np.abs(gamma.reshape(2, -1)).mean(axis=0)
                if gamma_by_coordinate.size != changed.size:
                    raise ValueError(
                        "Layer2 u/x coordinate count does not match the "
                        "Layer2-presynaptic event coordinate count"
                    )
                n_changed = int(changed.sum())
                overall_mean = float(gamma_by_coordinate.mean())
                changed_mean = (
                    float(gamma_by_coordinate[changed].mean())
                    if n_changed
                    else float("nan")
                )
                null_mean = _matched_random_coordinate_mean(
                    gamma_by_coordinate,
                    n_changed,
                    seed=(
                        int(ctx.cfg.fixed_b_protocol_seed)
                        + 1_000_003 * prefix_k
                        + 10_007 * history_family_id
                        + 101 * b_anchor_id
                    ),
                    replicates=int(ctx.cfg.fixed_b_null_replicates),
                )
                gamma_matrix = gamma.reshape(2, -1)
                gamma_norm = float(np.linalg.norm(gamma_matrix))
                changed_energy = (
                    float(np.linalg.norm(gamma_matrix[:, changed]))
                    if n_changed
                    else 0.0
                )
                event_valid = bool(
                    valid
                    and n_changed > 0
                    and np.isfinite(changed_mean)
                    and np.isfinite(null_mean)
                )
                event_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": prefix_k,
                        "history_family_id": history_family_id,
                        "b_anchor_id": b_anchor_id,
                        "event_A_row_id": event_a_id,
                        "event_C_row_id": event_c_id,
                        "trace_window_steps": int(event_shape[0]),
                        "event_coordinate_count": int(changed.size),
                        "changed_event_coordinate_count": n_changed,
                        "changed_event_coordinate_fraction": _safe_ratio(
                            n_changed, changed.size
                        ),
                        "changed_coordinate_gamma_mean_abs": changed_mean,
                        "matched_random_gamma_mean_abs": null_mean,
                        "overall_gamma_mean_abs": overall_mean,
                        "event_gamma_enrichment": _safe_ratio(
                            changed_mean - null_mean,
                            overall_mean,
                        ),
                        "event_gamma_enrichment_ratio": _safe_ratio(
                            changed_mean,
                            null_mean,
                        ),
                        "changed_coordinate_gamma_energy_fraction": _safe_ratio(
                            changed_energy,
                            gamma_norm,
                        ),
                        "valid": int(event_valid),
                    }
                )
    return pd.DataFrame(decomposition_rows), pd.DataFrame(event_rows)


def _swap_metrics(swaps: FixedBArtifact) -> pd.DataFrame:
    rows = swaps.tables["swap_rows"].copy()
    key_columns = [
        "prefix_k",
        "history_family_id",
        "b_anchor_id",
        "swap_scope",
        "receiver_condition",
        "donor_condition",
    ]
    if rows.duplicated(key_columns).any():
        raise ValueError("Fixed-B swap rows are not unique at the donor-receiver cell level")
    row_lookup = {
        (
            int(row.prefix_k),
            int(row.history_family_id),
            int(row.b_anchor_id),
            str(row.swap_scope),
            str(row.receiver_condition),
            str(row.donor_condition),
        ): int(row.swap_row_id)
        for row in rows.itertuples(index=False)
    }
    arrays = {
        "layer2_update": np.asarray(swaps.arrays["delta_layer2_ux"], dtype=np.float64),
        "early_class_score": np.asarray(
            swaps.arrays["class_scores_early"], dtype=np.float64
        ),
        "b_end_class_score": np.asarray(
            swaps.arrays["class_scores_b_end"], dtype=np.float64
        ),
        "early_layer1_voltage": np.asarray(
            swaps.arrays["layer1_voltage_features"], dtype=np.float64
        ),
        "early_layer1_event": np.asarray(
            swaps.arrays["layer1_event_features"], dtype=np.float64
        ),
        "early_layer1_drive": np.asarray(
            swaps.arrays["layer1_drive_features"], dtype=np.float64
        ),
    }
    output: list[dict[str, Any]] = []
    for prefix_k in sorted(int(value) for value in rows["prefix_k"].unique()):
        for history_family_id in sorted(
            int(value) for value in rows["history_family_id"].unique()
        ):
            for b_anchor_id in sorted(int(value) for value in rows["b_anchor_id"].unique()):
                for scope in ("layer1_only", "all_layers"):
                    ids = {
                        (receiver, donor): row_lookup[
                            (
                                prefix_k,
                                history_family_id,
                                b_anchor_id,
                                scope,
                                receiver,
                                donor,
                            )
                        ]
                        for receiver, donor in (
                            ("A", "A"),
                            ("C", "C"),
                            ("A", "C"),
                            ("C", "A"),
                        )
                    }
                    for endpoint, values in arrays.items():
                        for receiver, donor in (("A", "C"), ("C", "A")):
                            receiver_own = values[ids[(receiver, receiver)]]
                            donor_own = values[ids[(donor, donor)]]
                            cross = values[ids[(receiver, donor)]]
                            donor_direction = donor_own - receiver_own
                            observed_shift = cross - receiver_own
                            denominator = float(np.dot(donor_direction, donor_direction))
                            valid = bool(
                                denominator > NEAR_ZERO
                                and np.isfinite(observed_shift).all()
                            )
                            transfer = (
                                float(
                                    np.dot(observed_shift, donor_direction)
                                    / denominator
                                )
                                if valid
                                else float("nan")
                            )
                            output.append(
                                {
                                    "prefix_k": prefix_k,
                                    "history_family_id": history_family_id,
                                    "b_anchor_id": b_anchor_id,
                                    "swap_scope": scope,
                                    "endpoint": endpoint,
                                    "receiver_condition": receiver,
                                    "donor_condition": donor,
                                    "donor_transfer_index": transfer,
                                    "donor_direction_norm": float(
                                        np.linalg.norm(donor_direction)
                                    ),
                                    "observed_shift_norm": float(
                                        np.linalg.norm(observed_shift)
                                    ),
                                    "effect_alignment_cosine": _cosine(
                                        observed_shift, donor_direction
                                    ),
                                    "valid": int(valid),
                                }
                            )
    return pd.DataFrame(output)


def _summarize_decomposition(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix_k, part in cells.groupby("prefix_k", sort=True):
        valid = part.loc[part["valid"].eq(1)]
        rows.append(
            {
                "prefix_k": int(prefix_k),
                "n_cells": int(len(part)),
                "n_valid": int(len(valid)),
                "valid_coverage": _safe_ratio(len(valid), len(part)),
                "mean_same_B_common_update_cosine": float(
                    valid["same_B_common_update_cosine"].mean()
                ),
                "mean_processing_residual_gamma_energy_fraction": float(
                    valid["processing_residual_gamma_energy_fraction"].mean()
                ),
                "median_processing_residual_gamma_energy_fraction": float(
                    valid["processing_residual_gamma_energy_fraction"].median()
                ),
                "mean_total_contrast_fraction": float(
                    valid["total_contrast_fraction"].mean()
                ),
                "mean_local_replay_fraction": float(
                    valid["local_replay_fraction"].mean()
                ),
                "max_decomposition_relative_error": float(
                    part["decomposition_relative_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_events(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix_k, part in cells.groupby("prefix_k", sort=True):
        valid = part.loc[part["valid"].eq(1)]
        rows.append(
            {
                "prefix_k": int(prefix_k),
                "n_cells": int(len(part)),
                "n_valid": int(len(valid)),
                "valid_coverage": _safe_ratio(len(valid), len(part)),
                "mean_event_gamma_enrichment": float(
                    valid["event_gamma_enrichment"].mean()
                ),
                "median_event_gamma_enrichment": float(
                    valid["event_gamma_enrichment"].median()
                ),
                "mean_event_gamma_enrichment_ratio": float(
                    valid["event_gamma_enrichment_ratio"].mean()
                ),
                "mean_changed_event_coordinate_fraction": float(
                    valid["changed_event_coordinate_fraction"].mean()
                ),
                "mean_changed_coordinate_gamma_energy_fraction": float(
                    valid["changed_coordinate_gamma_energy_fraction"].mean()
                ),
                "fraction_positive_event_gamma_enrichment": float(
                    valid["event_gamma_enrichment"].gt(0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_swaps(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (prefix_k, scope, endpoint), part in cells.groupby(
        ["prefix_k", "swap_scope", "endpoint"],
        sort=True,
    ):
        valid = part.loc[part["valid"].eq(1)]
        rows.append(
            {
                "prefix_k": int(prefix_k),
                "swap_scope": str(scope),
                "endpoint": str(endpoint),
                "n_cells": int(len(part)),
                "n_valid": int(len(valid)),
                "valid_coverage": _safe_ratio(len(valid), len(part)),
                "mean_donor_transfer_index": float(
                    valid["donor_transfer_index"].mean()
                ),
                "median_donor_transfer_index": float(
                    valid["donor_transfer_index"].median()
                ),
                "fraction_positive": float(
                    valid["donor_transfer_index"].gt(0).mean()
                ),
                "mean_effect_alignment_cosine": float(
                    valid["effect_alignment_cosine"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _engineering_gates(
    ctx: ExperimentContext,
    *,
    inputs: FixedBArtifact,
    rollouts: FixedBArtifact,
    swaps: FixedBArtifact,
    decomposition_cells: pd.DataFrame,
    event_cells: pd.DataFrame,
    swap_cells: pd.DataFrame,
) -> pd.DataFrame:
    rollout_rows = rollouts.tables["rollout_rows"]
    input_hashes = (
        inputs.tables["input_manifest"]
        .set_index("b_anchor_id")["tensor_sha256"]
        .astype(str)
    )
    expected_hashes = rollout_rows["b_anchor_id"].map(input_hashes)
    exact_b_pass = bool(
        rollout_rows["exact_b_tensor_sha256"].astype(str).eq(expected_hashes).all()
    )
    event_manifest = rollouts.tables["layer2_event_manifest"]
    expected_event_rows = rollout_rows.loc[
        rollout_rows["track"].eq("stsp_isolated")
        & rollout_rows["branch"].eq("free")
        & rollout_rows["history_condition"].isin(["A", "C"]),
        "rollout_row_id",
    ].astype(int)
    event_identity_pass = (
        set(event_manifest["rollout_row_id"].astype(int))
        == set(expected_event_rows)
    )
    packed = rollouts.arrays["layer2_presynaptic_event_bits"]
    event_hash_pass = len(packed) == len(event_manifest) and all(
        array_hash(packed[int(row.event_row_id)]) == str(row.content_sha256)
        for row in event_manifest.itertuples(index=False)
    )
    trace_shape_pass = bool(
        event_manifest["trace_window_steps"]
        .astype(int)
        .eq(int(ctx.cfg.fixed_b_trace_window_steps))
        .all()
    )
    isolation = swaps.tables["swap_isolation_audit"]
    swap_contract_pass = bool(
        isolation["fast_state_equalized"].eq(1).all()
        and isolation["layer1_donor_stsp_applied"].eq(1).all()
        and isolation.loc[
            isolation["swap_scope"].eq("layer1_only"),
            "receiver_layer2_3_stsp_preserved",
        ].eq(1).all()
    )
    layer1_update = swap_cells.loc[
        swap_cells["swap_scope"].eq("layer1_only")
        & swap_cells["endpoint"].eq("layer2_update")
    ]
    layer1_score = swap_cells.loc[
        swap_cells["swap_scope"].eq("layer1_only")
        & swap_cells["endpoint"].eq("early_class_score")
    ]
    gates = [
        (
            "exact_B_hash_identity",
            int(exact_b_pass),
            float(exact_b_pass),
            1.0,
        ),
        (
            "T_equals_L_plus_Gamma",
            int(
                decomposition_cells["decomposition_relative_error"]
                .le(1e-6)
                .all()
            ),
            float(
                decomposition_cells["decomposition_relative_error"].max()
            ),
            1e-6,
        ),
        (
            "full_trace_row_identity",
            int(event_identity_pass),
            float(len(event_manifest)),
            float(len(expected_event_rows)),
        ),
        (
            "full_trace_content_hash",
            int(event_hash_pass),
            float(event_hash_pass),
            1.0,
        ),
        (
            "full_trace_window",
            int(trace_shape_pass),
            float(event_manifest["trace_window_steps"].astype(int).min()),
            float(ctx.cfg.fixed_b_trace_window_steps),
        ),
        (
            "layer1_only_swap_isolation",
            int(swap_contract_pass),
            float(swap_contract_pass),
            1.0,
        ),
        (
            "decomposition_valid_coverage",
            int(decomposition_cells["valid"].mean() >= 0.95),
            float(decomposition_cells["valid"].mean()),
            0.95,
        ),
        (
            "event_gamma_valid_coverage",
            int(event_cells["valid"].mean() >= 0.95),
            float(event_cells["valid"].mean()),
            0.95,
        ),
        (
            "layer1_update_donor_valid_coverage",
            int(layer1_update["valid"].mean() >= 0.95),
            float(layer1_update["valid"].mean()),
            0.95,
        ),
        (
            "layer1_early_score_donor_valid_coverage",
            int(layer1_score["valid"].mean() >= 0.95),
            float(layer1_score["valid"].mean()),
            0.95,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "gate": gate,
                "passed": passed,
                "observed": observed,
                "threshold_or_expected": threshold,
            }
            for gate, passed, observed, threshold in gates
        ]
    )


def _network_scalars(
    decomposition: pd.DataFrame,
    events: pd.DataFrame,
    swaps: pd.DataFrame,
    engineering: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        family: str,
        endpoint: str,
        prefix_k: int,
        value: float,
        role: str,
        threshold: float,
    ) -> None:
        rows.append(
            {
                "family": family,
                "endpoint": endpoint,
                "prefix_k": int(prefix_k),
                "value": float(value),
                "role": role,
                "threshold": float(threshold),
            }
        )

    for row in decomposition.itertuples(index=False):
        add(
            "core_threshold",
            "same_B_common_update_cosine",
            row.prefix_k,
            row.mean_same_B_common_update_cosine,
            "common_update_condition",
            0.50,
        )
        add(
            "core_primary",
            "processing_residual_gamma_energy_fraction",
            row.prefix_k,
            row.mean_processing_residual_gamma_energy_fraction,
            "primary",
            0.05,
        )
    for row in events.itertuples(index=False):
        add(
            "core_primary",
            "full_trace_event_gamma_enrichment",
            row.prefix_k,
            row.mean_event_gamma_enrichment,
            "primary",
            0.0,
        )
    for prefix_k in sorted(swaps["prefix_k"].astype(int).unique()):
        for endpoint, scalar_name, family, role in (
            (
                "layer2_update",
                "layer1_only_layer2_update_donor_transfer",
                "core_primary",
                "causal_primary",
            ),
            (
                "early_class_score",
                "layer1_only_early_class_score_donor_transfer",
                "core_primary",
                "functional_support",
            ),
        ):
            value = swaps.loc[
                swaps["prefix_k"].eq(prefix_k)
                & swaps["swap_scope"].eq("layer1_only")
                & swaps["endpoint"].eq(endpoint),
                "mean_donor_transfer_index",
            ].iloc[0]
            add(family, scalar_name, prefix_k, value, role, 0.0)
        all_layer = swaps.loc[
            swaps["prefix_k"].eq(prefix_k)
            & swaps["swap_scope"].eq("all_layers")
            & swaps["endpoint"].eq("layer2_update"),
            "mean_donor_transfer_index",
        ].iloc[0]
        add(
            "plumbing_control",
            "all_layer_layer2_update_donor_transfer",
            prefix_k,
            all_layer,
            "engineering_control",
            0.0,
        )
    add(
        "engineering",
        "all_engineering_gates",
        0,
        float(engineering["passed"].eq(1).all()),
        "hard_gate",
        1.0,
    )
    return pd.DataFrame(rows)


def _prediction_checklist(
    decomposition: pd.DataFrame,
    events: pd.DataFrame,
    swaps: pd.DataFrame,
    engineering: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in decomposition.itertuples(index=False):
        rows.extend(
            [
                {
                    "tier": "engineering",
                    "endpoint": "T_equals_L_plus_Gamma",
                    "prefix_k": int(row.prefix_k),
                    "observed": float(row.max_decomposition_relative_error),
                    "criterion": "<=1e-6",
                    "passed": int(row.max_decomposition_relative_error <= 1e-6),
                },
                {
                    "tier": "core_condition",
                    "endpoint": "same_B_common_update_cosine",
                    "prefix_k": int(row.prefix_k),
                    "observed": float(row.mean_same_B_common_update_cosine),
                    "criterion": ">=0.50",
                    "passed": int(row.mean_same_B_common_update_cosine >= 0.50),
                },
                {
                    "tier": "core_effect",
                    "endpoint": "processing_residual_gamma_energy_fraction",
                    "prefix_k": int(row.prefix_k),
                    "observed": float(
                        row.mean_processing_residual_gamma_energy_fraction
                    ),
                    "criterion": "network inference and cohort mean >=0.05",
                    "passed": -1,
                },
            ]
        )
    for row in events.itertuples(index=False):
        rows.append(
            {
                "tier": "core_effect",
                "endpoint": "full_trace_event_gamma_enrichment",
                "prefix_k": int(row.prefix_k),
                "observed": float(row.mean_event_gamma_enrichment),
                "criterion": "network inference >0",
                "passed": -1,
            }
        )
    for prefix_k in sorted(swaps["prefix_k"].astype(int).unique()):
        for endpoint, label in (
            ("layer2_update", "layer1_only_layer2_update_donor_transfer"),
            (
                "early_class_score",
                "layer1_only_early_class_score_donor_transfer",
            ),
        ):
            observed = float(
                swaps.loc[
                    swaps["prefix_k"].eq(prefix_k)
                    & swaps["swap_scope"].eq("layer1_only")
                    & swaps["endpoint"].eq(endpoint),
                    "mean_donor_transfer_index",
                ].iloc[0]
            )
            rows.append(
                {
                    "tier": "core_effect",
                    "endpoint": label,
                    "prefix_k": int(prefix_k),
                    "observed": observed,
                    "criterion": "network inference >0",
                    "passed": -1,
                }
            )
    rows.append(
        {
            "tier": "engineering",
            "endpoint": "all_engineering_gates",
            "prefix_k": 0,
            "observed": float(engineering["passed"].eq(1).all()),
            "criterion": "all pass",
            "passed": int(engineering["passed"].eq(1).all()),
        }
    )
    return pd.DataFrame(rows)


def _claim_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "The same exact B elicits a dominant common Layer2 STSP update.",
                "required_evidence": "same_B_common_update_cosine >= 0.50 at K1 and K5",
                "status_before_cohort": "descriptive_only",
            },
            {
                "claim": "History-conditioned Layer1 processing adds a nontrivial Layer2 write-back residual.",
                "required_evidence": "Gamma energy fraction >=0.05 plus positive cohort inference",
                "status_before_cohort": "not_decided",
            },
            {
                "claim": "Changed actual Layer2-presynaptic events carry enriched Gamma energy.",
                "required_evidence": "full-trace enrichment positive after network-level Holm correction",
                "status_before_cohort": "not_decided",
            },
            {
                "claim": "Layer1 STSP is causally sufficient to redirect the Layer2 update and early output state.",
                "required_evidence": "Layer1-only donor-transfer endpoints positive after network-level Holm correction",
                "status_before_cohort": "not_decided",
            },
            {
                "claim": "Layer2 alone is the common state or all-layer swaps prove the mechanism.",
                "required_evidence": "not an authorized claim",
                "status_before_cohort": "forbidden",
            },
        ]
    )


def _matched_random_coordinate_mean(
    values: np.ndarray,
    count: int,
    *,
    seed: int,
    replicates: int,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    if count <= 0 or count > len(values):
        return float("nan")
    rng = np.random.default_rng(int(seed))
    means = [
        float(values[rng.choice(len(values), size=count, replace=False)].mean())
        for _ in range(max(1, int(replicates)))
    ]
    return float(np.mean(means))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= NEAR_ZERO:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator) / float(denominator)
        if np.isfinite(denominator) and abs(float(denominator)) > NEAR_ZERO
        else float("nan")
    )


__all__ = ["analyze_fixed_b_mechanism_single_seed"]
