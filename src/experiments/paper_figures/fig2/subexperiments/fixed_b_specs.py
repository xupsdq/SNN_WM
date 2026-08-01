from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.experiments.paper_figures.fig2.types import ExperimentContext


HISTORY_CONDITIONS = ("A", "C", "S0")
BRANCHES = ("passive", "free", "replay")
TRACKS = ("natural", "stsp_isolated")
SWAP_SCOPES = ("all_layers", "layer1_only")
B_FOLD_MODES = ("stratified_within_class",)
CROSSFIT_AXES = ("both", "history_only", "B_image_only", "class_only")
NULL_PURPOSES = (
    "structured_direction",
    "interaction_spatial_alignment",
    "functional_bridge",
    "drive_to_voltage",
    "voltage_to_event",
    "event_to_update",
)
FIXED_B_SCHEMA_VERSION = 4


def build_fixed_b_specs(ctx: ExperimentContext) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build outcome-blind candidate histories, exact-B anchors, folds, and null seeds."""
    cfg = ctx.cfg
    protocol_seed = int(cfg.fixed_b_protocol_seed)
    n_candidates = int(cfg.fixed_b_candidate_families)
    n_selected = int(cfg.fixed_b_history_families)
    n_anchors = int(cfg.fixed_b_anchors)
    prefix_depths = tuple(sorted({int(value) for value in cfg.fixed_b_prefix_depths}))
    n_folds = int(cfg.fixed_b_folds)
    n_null = int(cfg.fixed_b_null_replicates)
    b_fold_mode = str(cfg.fixed_b_b_fold_mode)
    crossfit_axes = tuple(str(value) for value in cfg.fixed_b_crossfit_axes)

    if str(cfg.split) != "test":
        raise ValueError("The corrected fixed-B protocol is locked to the MNIST test split")
    if b_fold_mode not in B_FOLD_MODES:
        raise ValueError(f"Unsupported corrected fixed-B fold mode: {b_fold_mode!r}")
    invalid_axes = sorted(set(crossfit_axes) - set(CROSSFIT_AXES))
    if not crossfit_axes or invalid_axes:
        raise ValueError(f"Unsupported fixed-B crossfit axes: {invalid_axes}")
    if not prefix_depths or min(prefix_depths) < 1:
        raise ValueError("fixed_b_prefix_depths must contain positive integers")
    if n_candidates < n_selected or n_selected < n_folds:
        raise ValueError("fixed-B candidate/selected history counts do not support the locked folds")
    if n_anchors % 10 or n_anchors < 10 * n_folds:
        raise ValueError("fixed-B anchors must be equally stratified over ten classes and every fold")
    anchors_per_class = n_anchors // 10
    if anchors_per_class % n_folds:
        raise ValueError("B anchors per class must be a multiple of fixed_b_folds")
    if n_null < 1:
        raise ValueError("fixed_b_null_replicates must be positive")
    if not bool(cfg.smoke) and (n_candidates, n_selected, n_anchors, prefix_depths, n_folds) != (50, 10, 50, (1, 5), 5):
        raise ValueError("Full corrected fixed-B runs require 50 candidates, 10 selected histories, 50 anchors, K={1,5}, and five folds")

    rng = np.random.default_rng(protocol_seed)
    shuffled_pools = {
        label: rng.permutation(np.asarray(ctx.class_index[label], dtype=np.int64)).tolist()
        for label in range(10)
    }
    cursors = {label: 0 for label in range(10)}

    def take_image(label: int) -> int:
        label = int(label)
        cursor = cursors[label]
        pool = shuffled_pools[label]
        if cursor >= len(pool):
            raise RuntimeError(f"Exhausted MNIST image pool for label {label}")
        cursors[label] = cursor + 1
        return int(pool[cursor])

    max_k = max(prefix_depths)
    family_rows: list[dict[str, Any]] = []
    candidate_history_rows: list[dict[str, Any]] = []
    input_rows_by_image: dict[int, dict[str, Any]] = {}
    for candidate_id in range(n_candidates):
        sequences: dict[str, tuple[list[int], list[int]]] = {}
        for condition, offset, stride in (("A", 0, 3), ("C", 5, 7)):
            labels = [int((candidate_id + offset + stride * position) % 10) for position in range(max_k)]
            image_ids = [take_image(label) for label in labels]
            sequences[condition] = (image_ids, labels)
            for position, (image_id, label) in enumerate(zip(image_ids, labels)):
                if image_id in input_rows_by_image:
                    raise RuntimeError(f"History source image {image_id} was reused")
                image = ctx.dataset[image_id][0].detach().cpu()
                input_rows_by_image[image_id] = {
                    "protocol_seed": protocol_seed,
                    "history_input_id": len(input_rows_by_image),
                    "image_id": int(image_id),
                    "label": int(label),
                    "encoding_seed": protocol_seed + 100_000 + int(image_id),
                    "image_sha256": _tensor_sha256(image.numpy()),
                    "candidate_family_id": int(candidate_id),
                    "history_condition": condition,
                    "sequence_position": int(position),
                }
        family_rows.append(
            {
                "protocol_seed": protocol_seed,
                "candidate_family_id": int(candidate_id),
                "balance_stratum": int(candidate_id % 10),
                "A_full_image_ids": _json_list(sequences["A"][0]),
                "A_full_labels": _json_list(sequences["A"][1]),
                "C_full_image_ids": _json_list(sequences["C"][0]),
                "C_full_labels": _json_list(sequences["C"][1]),
            }
        )
        for prefix_k in prefix_depths:
            for condition in ("A", "C"):
                full_images, full_labels = sequences[condition]
                candidate_history_rows.append(
                    {
                        "protocol_seed": protocol_seed,
                        "candidate_history_row_id": len(candidate_history_rows),
                        "candidate_family_id": int(candidate_id),
                        "balance_stratum": int(candidate_id % 10),
                        "history_condition": condition,
                        "prefix_k": int(prefix_k),
                        "sequence_image_ids": _json_list(full_images[:prefix_k]),
                        "sequence_labels": _json_list(full_labels[:prefix_k]),
                        "sequence_encoding_seeds": _json_list(
                            [protocol_seed + 100_000 + int(image_id) for image_id in full_images[:prefix_k]]
                        ),
                        "elapsed_steps": int(prefix_k * (cfg.fixed_b_item_steps + cfg.fixed_b_inter_delay_steps)),
                    }
                )

    history_image_ids = set(input_rows_by_image)
    b_rows: list[dict[str, Any]] = []
    for anchor_id in range(n_anchors):
        label = int(anchor_id % 10)
        image_id = take_image(label)
        if image_id in history_image_ids:
            raise RuntimeError(f"B anchor image {image_id} overlaps a history source")
        image = ctx.dataset[image_id][0].detach().cpu()
        pixels = image.numpy().astype(np.float64, copy=False)
        replicate_id = int(anchor_id // 10)
        b_rows.append(
            {
                "protocol_seed": protocol_seed,
                "b_anchor_id": int(anchor_id),
                "B_image_id": int(image_id),
                "B_label": label,
                "B_replicate_id": replicate_id,
                "encoding_seed": protocol_seed + 200_000 + int(anchor_id),
                "B_image_sha256": _tensor_sha256(image.numpy()),
                "B_pixel_sum": float(pixels.sum()),
                "B_foreground_area": int((pixels > float(cfg.foreground_threshold)).sum()),
                "b_fold": int(replicate_id % n_folds),
                "b_class_fold": int(label % n_folds),
                "b_fold_mode": b_fold_mode,
                "spike_storage_key": "exact_b_spikes",
                "spike_row_index": int(anchor_id),
            }
        )

    branch_specs = _branch_specs(protocol_seed)
    null_specs = _null_specs(protocol_seed, prefix_depths, n_null)
    endpoint_spec = {
        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
        "protocol_seed": protocol_seed,
        "analysis_status": "frozen_before_untouched_confirmatory_networks",
        "primary_representation": "layer2_concatenated_u_x_passive_corrected_update",
        "mechanism_chain": [
            "exact_B_input",
            "history_conditioned_layer1_STSP_current",
            "layer1_firing",
            "layer2_presynaptic_events",
            "layer2_STSP_writeback",
        ],
        "decomposition": {
            "T": "free_A_minus_free_C",
            "L": "replay_A_minus_replay_C",
            "Gamma": "(free_A_minus_replay_A)_minus_(free_C_minus_replay_C)",
            "identity": "T_equals_L_plus_Gamma",
        },
        "core_endpoints": [
            "same_B_common_update_cosine",
            "processing_residual_gamma_energy_fraction",
            "full_trace_layer2_presynaptic_event_gamma_enrichment",
            "layer1_only_layer2_update_donor_transfer",
            "layer1_only_early_class_score_donor_transfer",
        ],
        "strong_endpoints": [
            "gain_weighted_kernel_drive_to_effective_voltage",
            "effective_voltage_to_actual_layer1_event_composition",
            "actual_layer1_events_to_layer2_update_direction",
            "full_trace_time_binned_event_to_gamma_prediction",
            "all_layer_donor_transfer_plumbing_control",
        ],
        "thresholds": {
            "source_max_abs_smd": float(cfg.fixed_b_source_match_max_smd),
            "near_zero_update_norm": 1e-12,
            "decomposition_relative_error": 1e-6,
            "same_B_common_update_cosine": 0.50,
            "processing_residual_gamma_energy_fraction": 0.05,
            "minimum_valid_coverage": 0.95,
            "one_sided_familywise_alpha": 0.05,
            "restoration_fraction_of_native_separation": 0.10,
        },
        "confirmatory_design": {
            "development_networks": [1000],
            "confirmatory_networks": list(range(1001, 1020)),
            "inference_unit": "independently_trained_network",
            "within_network_aggregation": "10_history_families_x_50_exact_B_anchors",
            "test": "exact_one_sided_sign_flip",
            "multiplicity": "Holm_within_prespecified_core_family",
            "optional_stopping": False,
            "outcome_based_exclusions": False,
        },
        "event_trace": {
            "source": "actual_layer2_presynaptic_input_from_pooled_layer1_spikes",
            "window_ms": int(cfg.fixed_b_trace_window_ms),
            "storage": "bit_packed_full_raster_for_stsp_isolated_free_A_C_rows",
        },
        "causal_swap": {
            "primary_scope": "layer1_only",
            "receiver_layer2_and_layer3_STSP_fixed": True,
            "all_non_STSP_fast_state_equalized": True,
            "all_layer_scope_role": "engineering_plumbing_control_only",
        },
        "b_fold_mode": b_fold_mode,
        "crossfit_axes": list(crossfit_axes),
        "ridge_alphas": [float(value) for value in cfg.fixed_b_ridge_alphas],
        "target_components": int(cfg.fixed_b_target_components),
        "accuracy_role": "secondary_direction_unconstrained",
        "selection_uses_outcomes": False,
    }
    return (
        {
            "candidate_history_families": pd.DataFrame(family_rows),
            "candidate_history_specs": pd.DataFrame(candidate_history_rows),
            "history_input_specs": pd.DataFrame(sorted(input_rows_by_image.values(), key=lambda row: int(row["history_input_id"]))),
            "b_anchor_specs": pd.DataFrame(b_rows),
            "branch_specs": branch_specs,
            "null_specs": null_specs,
        },
        endpoint_spec,
    )


def materialize_selected_specs(
    ctx: ExperimentContext,
    candidate_tables: Mapping[str, pd.DataFrame],
    selected_candidate_ids: Sequence[int],
) -> dict[str, pd.DataFrame]:
    """Materialize the full selected-history × exact-B source-of-truth tables."""
    selected_ids = [int(value) for value in selected_candidate_ids]
    if len(selected_ids) != int(ctx.cfg.fixed_b_history_families) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Selected fixed-B candidate IDs must be unique and match fixed_b_history_families")
    candidate_families = candidate_tables["candidate_history_families"].copy()
    candidate_histories = candidate_tables["candidate_history_specs"].copy()
    known = set(candidate_families["candidate_family_id"].astype(int))
    missing = sorted(set(selected_ids) - known)
    if missing:
        raise ValueError(f"Unknown selected fixed-B candidate families: {missing}")

    by_stratum = candidate_families.set_index("candidate_family_id")["balance_stratum"].astype(int).to_dict()
    selected_ids.sort(key=lambda value: (by_stratum[value], value))
    family_id_by_candidate = {candidate_id: index for index, candidate_id in enumerate(selected_ids)}
    selected_family_rows: list[dict[str, Any]] = []
    for candidate_id in selected_ids:
        source = candidate_families.loc[candidate_families["candidate_family_id"].eq(candidate_id)].iloc[0].to_dict()
        selected_family_rows.append(
            {
                "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                "history_family_id": int(family_id_by_candidate[candidate_id]),
                "candidate_family_id": int(candidate_id),
                "balance_stratum": int(source["balance_stratum"]),
                "A_full_image_ids": str(source["A_full_image_ids"]),
                "A_full_labels": str(source["A_full_labels"]),
                "C_full_image_ids": str(source["C_full_image_ids"]),
                "C_full_labels": str(source["C_full_labels"]),
            }
        )

    n_folds = int(ctx.cfg.fixed_b_folds)
    fold_rng = np.random.default_rng(int(ctx.cfg.fixed_b_protocol_seed) + 301)
    fold_order = fold_rng.permutation(len(selected_ids))
    history_fold_by_family = {
        int(family_id): int(rank % n_folds)
        for rank, family_id in enumerate(fold_order)
    }
    history_rows: list[dict[str, Any]] = []
    prefix_depths = tuple(sorted(int(value) for value in ctx.cfg.fixed_b_prefix_depths))
    for prefix_k in prefix_depths:
        for candidate_id in selected_ids:
            family_id = family_id_by_candidate[candidate_id]
            for condition in HISTORY_CONDITIONS:
                if condition == "S0":
                    image_ids: list[int] = []
                    labels: list[int] = []
                    encoding_seeds: list[int] = []
                else:
                    source = candidate_histories.loc[
                        candidate_histories["candidate_family_id"].eq(candidate_id)
                        & candidate_histories["history_condition"].eq(condition)
                        & candidate_histories["prefix_k"].eq(prefix_k)
                    ]
                    if len(source) != 1:
                        raise RuntimeError("Selected candidate history lookup is not one-to-one")
                    record = source.iloc[0]
                    image_ids = [int(value) for value in json.loads(str(record["sequence_image_ids"]))]
                    labels = [int(value) for value in json.loads(str(record["sequence_labels"]))]
                    encoding_seeds = [int(value) for value in json.loads(str(record["sequence_encoding_seeds"]))]
                history_rows.append(
                    {
                        "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                        "history_row_id": len(history_rows),
                        "history_family_id": int(family_id),
                        "candidate_family_id": int(candidate_id),
                        "history_condition": condition,
                        "prefix_k": int(prefix_k),
                        "sequence_image_ids": _json_list(image_ids),
                        "sequence_labels": _json_list(labels),
                        "sequence_encoding_seeds": _json_list(encoding_seeds),
                        "elapsed_steps": int(prefix_k * (ctx.cfg.fixed_b_item_steps + ctx.cfg.fixed_b_inter_delay_steps)),
                        "history_fold": int(history_fold_by_family[family_id]),
                    }
                )

    history_specs = pd.DataFrame(history_rows)
    b_anchor_specs = candidate_tables["b_anchor_specs"].copy().reset_index(drop=True)
    cell_rows: list[dict[str, Any]] = []
    for history in history_specs.itertuples(index=False):
        for anchor in b_anchor_specs.itertuples(index=False):
            labels = {int(value) for value in json.loads(str(history.sequence_labels))}
            same_fold = int(history.history_fold) == int(anchor.b_fold)
            cell_rows.append(
                {
                    "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                    "cell_id": len(cell_rows),
                    "history_row_id": int(history.history_row_id),
                    "history_family_id": int(history.history_family_id),
                    "candidate_family_id": int(history.candidate_family_id),
                    "history_condition": str(history.history_condition),
                    "prefix_k": int(history.prefix_k),
                    "b_anchor_id": int(anchor.b_anchor_id),
                    "B_image_id": int(anchor.B_image_id),
                    "B_label": int(anchor.B_label),
                    "B_replicate_id": int(anchor.B_replicate_id),
                    "history_contains_B_label": int(int(anchor.B_label) in labels),
                    "history_fold": int(history.history_fold),
                    "b_fold": int(anchor.b_fold),
                    "b_class_fold": int(anchor.b_class_fold),
                    "outer_test_fold": int(history.history_fold) if same_fold else -1,
                    "cell_role": "test" if same_fold else "guard_or_train",
                }
            )

    fold_rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        fold_rows.append(
            {
                "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                "fold": int(fold),
                "test_history_families": _json_list(
                    sorted(family for family, value in history_fold_by_family.items() if value == fold)
                ),
                "test_b_anchors": _json_list(
                    sorted(
                        int(value)
                        for value in b_anchor_specs.loc[b_anchor_specs["b_fold"].eq(fold), "b_anchor_id"]
                    )
                ),
                "test_B_labels": _json_list(
                    sorted(int(value) for value in b_anchor_specs.loc[b_anchor_specs["b_fold"].eq(fold), "B_label"].unique())
                ),
                "training_rule": "H_not_f_x_B_not_f",
                "guard_rule": "H_f_x_B_not_f union H_not_f_x_B_f",
                "test_rule": "H_f_x_B_f",
            }
        )

    swap_rows: list[dict[str, Any]] = []
    for prefix_k in prefix_depths:
        for family_id in range(len(selected_ids)):
            for scope in SWAP_SCOPES:
                for receiver, donor in (("A", "A"), ("C", "C"), ("A", "C"), ("C", "A")):
                    swap_rows.append(
                        {
                            "protocol_seed": int(ctx.cfg.fixed_b_protocol_seed),
                            "swap_spec_id": len(swap_rows),
                            "prefix_k": int(prefix_k),
                            "history_family_id": int(family_id),
                            "history_fold": int(history_fold_by_family[family_id]),
                            "receiver_condition": receiver,
                            "donor_condition": donor,
                            "swap_scope": scope,
                            "is_own_sham": int(receiver == donor),
                            "fast_state_rule": "all_layers_reset_to_identical_baseline",
                            "stsp_rule": "donor_all_layers" if scope == "all_layers" else "donor_layer1_receiver_other_layers",
                        }
                    )

    tables = {
        "history_families": pd.DataFrame(selected_family_rows),
        "history_specs": history_specs,
        "b_anchor_specs": b_anchor_specs,
        "cell_specs": pd.DataFrame(cell_rows),
        "fold_specs": pd.DataFrame(fold_rows),
        "branch_specs": candidate_tables["branch_specs"].copy(),
        "swap_specs": pd.DataFrame(swap_rows),
        "null_specs": candidate_tables["null_specs"].copy(),
    }
    validate_selected_source_tables(ctx, tables)
    return tables


def validate_selected_source_tables(ctx: ExperimentContext, tables: Mapping[str, pd.DataFrame]) -> None:
    history_specs = tables["history_specs"]
    b_specs = tables["b_anchor_specs"]
    expected_history_rows = 3 * int(ctx.cfg.fixed_b_history_families) * len(tuple(ctx.cfg.fixed_b_prefix_depths))
    if len(history_specs) != expected_history_rows:
        raise ValueError("Selected fixed-B history_specs row-count mismatch")
    if len(b_specs) != int(ctx.cfg.fixed_b_anchors):
        raise ValueError("Selected fixed-B B-anchor row-count mismatch")
    history_ids: list[int] = []
    for row in history_specs.loc[history_specs["history_condition"].isin(["A", "C"])].itertuples(index=False):
        values = [int(value) for value in json.loads(str(row.sequence_image_ids))]
        if len(values) != int(row.prefix_k):
            raise ValueError("Selected fixed-B history sequence length mismatch")
        if int(row.prefix_k) == max(int(value) for value in ctx.cfg.fixed_b_prefix_depths):
            history_ids.extend(values)
    if len(history_ids) != len(set(history_ids)):
        raise ValueError("Selected A/C histories reuse source images")
    b_ids = set(b_specs["B_image_id"].astype(int))
    if b_ids & set(history_ids):
        raise ValueError("Selected history images overlap exact-B anchors")
    for prefix_k in sorted(int(value) for value in ctx.cfg.fixed_b_prefix_depths):
        for condition in ("A", "C"):
            labels: list[int] = []
            part = history_specs.loc[
                history_specs["prefix_k"].eq(prefix_k) & history_specs["history_condition"].eq(condition)
            ]
            for encoded in part["sequence_labels"]:
                labels.extend(int(value) for value in json.loads(str(encoded)))
            counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=10)
            if int(counts.max() - counts.min()) > 1:
                raise ValueError(f"Selected {condition} class counts are not independently balanced at K={prefix_k}")
    for fold in range(int(ctx.cfg.fixed_b_folds)):
        anchors = b_specs.loc[b_specs["b_fold"].eq(fold)]
        if set(anchors["B_label"].astype(int)) != set(range(10)):
            raise ValueError(f"B fold {fold} is not stratified over all classes")


def _branch_specs(protocol_seed: int) -> pd.DataFrame:
    rows = [
        {
            "protocol_seed": int(protocol_seed),
            "track": track,
            "branch": branch,
            "enabled": int(not (track == "natural" and branch == "replay")),
            "prestate_rule": "full_boundary" if track == "natural" else "stsp_only_fast_state_equalized",
            "input_rule": "zero_input" if branch == "passive" else (
                "exact_B_external_input" if branch == "free" else "exact_B_with_S0_B_internal_L1_replay"
            ),
        }
        for track in TRACKS
        for branch in BRANCHES
    ]
    return pd.DataFrame(rows)


def _null_specs(protocol_seed: int, prefix_depths: Sequence[int], n_replicates: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for purpose_index, purpose in enumerate(NULL_PURPOSES):
        for prefix_k in prefix_depths:
            for replicate in range(int(n_replicates)):
                rows.append(
                    {
                        "fixed_b_schema_version": FIXED_B_SCHEMA_VERSION,
                        "protocol_seed": int(protocol_seed),
                        "null_id": len(rows),
                        "purpose": purpose,
                        "prefix_k": int(prefix_k),
                        "replicate": int(replicate),
                        "random_seed": int(protocol_seed + 1_000_000 + 100_000 * purpose_index + 1_000 * int(prefix_k) + replicate),
                        "permutation_rule": (
                            "within_row_joint_spatial_permutation" if purpose in {"interaction_spatial_alignment", "drive_to_voltage"}
                            else "matched_family_B_block_sign_flip"
                        ),
                        "percentile_gate": 95.0,
                    }
                )
    return pd.DataFrame(rows)


def _tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


__all__ = [
    "B_FOLD_MODES",
    "BRANCHES",
    "CROSSFIT_AXES",
    "FIXED_B_SCHEMA_VERSION",
    "HISTORY_CONDITIONS",
    "NULL_PURPOSES",
    "SWAP_SCOPES",
    "TRACKS",
    "build_fixed_b_specs",
    "materialize_selected_specs",
    "validate_selected_source_tables",
]
