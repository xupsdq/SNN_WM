from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.experiments.paper_figures.fig2.fixed_b_artifacts import FixedBArtifact
from src.experiments.paper_figures.fig2.fixed_b_protocol import protocol_digest
from src.experiments.paper_figures.fig2.subexperiments.fixed_b_runtime import unpack_event_bits
from src.experiments.paper_figures.fig2.types import ExperimentContext


CORE_ENDPOINTS = (
    "structured_direction",
    "structured_interaction",
    "all_layer_donor_transfer",
    "functional_bridge",
)
STRONG_ENDPOINTS = (
    "drive_to_voltage",
    "voltage_to_event",
    "event_to_update",
    "layer1_voltage_donor_transfer",
    "layer1_event_donor_transfer",
    "layer1_update_donor_transfer",
    "free_minus_replay",
)


def analyze_fixed_b_single_seed(
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
    del specs
    interaction = _crossfit_interaction(ctx, histories, replay, rollouts)
    direct_fold, direct_summary, direct_pairs = _direct_direction_metrics(
        ctx,
        histories,
        replay,
        rollouts,
        interaction["base_features"],
    )
    donor_trial, donor_fold, donor_summary = _donor_transfer_metrics(ctx, swaps)
    functional_fold, functional_summary, functional_pairs = _functional_bridge_metrics(ctx, rollouts, histories)
    stage_fold, stage_null, stage_summary = _stagewise_metrics(
        ctx,
        histories,
        replay,
        rollouts,
        interaction["base_features"],
    )
    restoration = _restoration_summary(ctx, histories)
    exact_b = _exact_b_audit(ctx, inputs, rollouts)
    event_audit = _event_recompute_audit(ctx, rollouts)
    engineering = _engineering_gates(
        ctx,
        histories=histories,
        rollouts=rollouts,
        swaps=swaps,
        interaction_audit=interaction["fold_audit"],
        restoration=restoration,
        exact_b=exact_b,
        event_audit=event_audit,
    )
    influence = _influence_diagnostics(
        ctx,
        interaction_rows=interaction["row_metrics"],
        direct_pairs=direct_pairs,
        donor_trials=donor_trial,
        functional_pairs=functional_pairs,
    )
    existing_chain = _existing_chain_audit(ctx)
    checklist, claim_ledger, decision = _tiered_verdict(
        ctx,
        protocol=protocol,
        engineering=engineering,
        existing_chain=existing_chain,
        interaction_summary=interaction["summary"],
        direct_summary=direct_summary,
        donor_summary=donor_summary,
        functional_summary=functional_summary,
        stage_summary=stage_summary,
        influence=influence,
    )
    tables = {
        "fixed_b_primary_crossfit_fold_metrics": interaction["fold_metrics"],
        "fixed_b_primary_crossfit_metrics": interaction["summary"],
        "fixed_b_spatial_null_metrics": interaction["null_metrics"],
        "fixed_b_crossfit_row_metrics": interaction["row_metrics"],
        "fixed_b_fold_preprocessing_audit": interaction["fold_audit"],
        "fixed_b_direct_direction_fold_metrics": direct_fold,
        "fixed_b_direct_direction_metrics": direct_summary,
        "fixed_b_direct_direction_pair_scores": direct_pairs,
        "fixed_b_donor_transfer_trial_metrics": donor_trial,
        "fixed_b_donor_transfer_fold_metrics": donor_fold,
        "fixed_b_donor_transfer_metrics": donor_summary,
        "fixed_b_functional_bridge_fold_metrics": functional_fold,
        "fixed_b_functional_bridge_metrics": functional_summary,
        "fixed_b_functional_bridge_pair_scores": functional_pairs,
        "fixed_b_stagewise_fold_metrics": stage_fold,
        "fixed_b_stagewise_null_metrics": stage_null,
        "fixed_b_stagewise_metrics": stage_summary,
        "fixed_b_influence_diagnostics": influence,
        "fixed_b_restoration_summary": restoration,
        "fixed_b_exact_b_audit": exact_b,
        "fixed_b_event_recompute_audit": event_audit,
        "fixed_b_engineering_gates": engineering,
        "fixed_b_existing_chain_audit": existing_chain,
        "fixed_b_prediction_checklist": checklist,
        "fixed_b_claim_ledger": claim_ledger,
    }
    decision.update(
        {
            "fixed_b_schema_version": 3,
            "protocol_digest": protocol_digest(protocol),
            "network_seed": int(ctx.cfg.network_seed),
            "status": "development_seed_not_manuscript_inference",
            "remaining_seeds_allowed": False,
            "confirmatory_inference_performed": False,
            "claim_boundary": "Seed 1000 is development evidence only; it cannot establish a population-level manuscript claim.",
        }
    )
    return tables, decision


def _crossfit_interaction(
    ctx: ExperimentContext,
    histories: FixedBArtifact,
    replay: FixedBArtifact,
    rollouts: FixedBArtifact,
) -> dict[str, Any]:
    rows = _primary_rollout_rows(rollouts, branches=("free", "replay"))
    vectors = rollouts.arrays["delta_layer2_ux"].astype(np.float64, copy=False)
    fold_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    base_by_row: dict[int, np.ndarray] = {}
    for (prefix_k, branch), part in rows.groupby(["prefix_k", "branch"], sort=True):
        part = part.sort_values(["history_family_id", "history_condition", "b_anchor_id"]).reset_index(drop=True)
        y_raw = vectors[part["rollout_row_id"].to_numpy(dtype=np.int64)]
        valid = np.isfinite(y_raw).all(axis=1) & (np.linalg.norm(y_raw, axis=1) > 1e-12)
        part = part.loc[valid].reset_index(drop=True)
        y_raw = y_raw[valid]
        base, interaction = _feature_matrices(ctx, histories, replay, part, int(prefix_k))
        for row_id, vector in zip(part["rollout_row_id"].astype(int), base):
            base_by_row[int(row_id)] = vector
        for outer_fold in range(int(ctx.cfg.fixed_b_folds)):
            train, test, guard = _two_axis_masks(part, outer_fold)
            train_indices = np.flatnonzero(train)
            test_indices = np.flatnonzero(test)
            if len(test_indices) < 2 or len(train_indices) < 4:
                raise RuntimeError(f"Insufficient fixed-B rows in outer fold {outer_fold}")
            projection = _fit_target_projection(
                y_raw[train_indices],
                y_raw[test_indices],
                int(ctx.cfg.fixed_b_target_components),
                int(ctx.cfg.fixed_b_protocol_seed) + 10_000 * int(prefix_k) + outer_fold,
            )
            alpha_base = _select_alpha(ctx, part, train_indices, base, y_raw, outer_fold)
            alpha_interaction = _select_alpha(ctx, part, train_indices, interaction, y_raw, outer_fold)
            base_fit = _fit_feature_model(
                base[train_indices],
                projection["train"],
                base[test_indices],
                projection["test"],
                alpha_base,
            )
            interaction_fit = _fit_feature_model(
                interaction[train_indices],
                projection["train"],
                interaction[test_indices],
                projection["test"],
                alpha_interaction,
            )
            relative = (base_fit["mse"] - interaction_fit["mse"]) / max(base_fit["mse"], 1e-12)
            delta_r2 = float(interaction_fit["r2"] - base_fit["r2"])
            fold_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "prefix_k": int(prefix_k),
                    "branch": str(branch),
                    "outer_fold": int(outer_fold),
                    "n_train": int(len(train_indices)),
                    "n_test": int(len(test_indices)),
                    "n_guard": int(guard.sum()),
                    "target_components": int(projection["train"].shape[1]),
                    "base_feature_count": int(base.shape[1]),
                    "interaction_feature_count": int(interaction.shape[1] - base.shape[1]),
                    "alpha_base": float(alpha_base),
                    "alpha_interaction": float(alpha_interaction),
                    "base_supported_feature_count": int(
                        base_fit["supported_feature_count"]
                    ),
                    "interaction_supported_feature_count": int(
                        interaction_fit["supported_feature_count"]
                    ),
                    "base_clipped_test_values": int(
                        base_fit["clipped_test_value_count"]
                    ),
                    "interaction_clipped_test_values": int(
                        interaction_fit["clipped_test_value_count"]
                    ),
                    "mse_separable": float(base_fit["mse"]),
                    "mse_interaction": float(interaction_fit["mse"]),
                    "r2_separable": float(base_fit["r2"]),
                    "r2_interaction": float(interaction_fit["r2"]),
                    "delta_r2": delta_r2,
                    "relative_mse_reduction": float(relative),
                }
            )
            test_part = part.iloc[test_indices].reset_index(drop=True)
            row_advantage = np.mean(
                (projection["test"] - base_fit["prediction"]) ** 2
                - (projection["test"] - interaction_fit["prediction"]) ** 2,
                axis=1,
            )
            for local_index, row in enumerate(test_part.itertuples(index=False)):
                row_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "branch": str(branch),
                        "outer_fold": int(outer_fold),
                        "rollout_row_id": int(row.rollout_row_id),
                        "history_family_id": int(row.history_family_id),
                        "b_anchor_id": int(row.b_anchor_id),
                        "history_condition": str(row.history_condition),
                        "row_mse_advantage": float(row_advantage[local_index]),
                    }
                )
            audit_rows.append(
                _fold_audit_row(
                    ctx,
                    part,
                    train_indices,
                    test_indices,
                    np.flatnonzero(guard),
                    prefix_k=int(prefix_k),
                    branch=str(branch),
                    fold=outer_fold,
                    base=base,
                    interaction=interaction,
                    base_fit=base_fit,
                    interaction_fit=interaction_fit,
                )
            )
            if str(branch) == "free":
                null_specs = histories.tables["null_specs"].loc[
                    histories.tables["null_specs"]["purpose"].eq("interaction_spatial_alignment")
                    & histories.tables["null_specs"]["prefix_k"].eq(prefix_k)
                ]
                for null_spec in null_specs.itertuples(index=False):
                    _, null_interaction = _feature_matrices(
                        ctx,
                        histories,
                        replay,
                        part,
                        int(prefix_k),
                        spatial_seed=int(null_spec.random_seed),
                    )
                    null_fit = _fit_feature_model(
                        null_interaction[train_indices],
                        projection["train"],
                        null_interaction[test_indices],
                        projection["test"],
                        alpha_interaction,
                    )
                    null_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "prefix_k": int(prefix_k),
                            "outer_fold": int(outer_fold),
                            "replicate": int(null_spec.replicate),
                            "random_seed": int(null_spec.random_seed),
                            "delta_r2": float(null_fit["r2"] - base_fit["r2"]),
                            "relative_mse_reduction": float(
                                (base_fit["mse"] - null_fit["mse"]) / max(base_fit["mse"], 1e-12)
                            ),
                        }
                    )
    fold_metrics = pd.DataFrame(fold_rows)
    null_metrics = pd.DataFrame(null_rows)
    summary_rows = []
    for (prefix_k, branch), part in fold_metrics.groupby(["prefix_k", "branch"], sort=True):
        null_part = null_metrics.loc[null_metrics["prefix_k"].eq(prefix_k)] if str(branch) == "free" else pd.DataFrame()
        null_grouped = (
            null_part.groupby("replicate")[["delta_r2", "relative_mse_reduction"]].mean()
            if not null_part.empty
            else pd.DataFrame()
        )
        delta_null95 = float(np.percentile(null_grouped["delta_r2"], 95)) if not null_grouped.empty else float("nan")
        relative_null95 = float(np.percentile(null_grouped["relative_mse_reduction"], 95)) if not null_grouped.empty else float("nan")
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "prefix_k": int(prefix_k),
                "branch": str(branch),
                "n_folds": int(len(part)),
                "mean_r2_separable": float(part["r2_separable"].mean()),
                "mean_r2_interaction": float(part["r2_interaction"].mean()),
                "mean_delta_r2": float(part["delta_r2"].mean()),
                "mean_relative_mse_reduction": float(part["relative_mse_reduction"].mean()),
                "spatial_null_delta_r2_p95": delta_null95,
                "spatial_null_relative_mse_p95": relative_null95,
                "delta_r2_null_excess": float(part["delta_r2"].mean() - delta_null95) if np.isfinite(delta_null95) else float("nan"),
                "relative_mse_null_excess": float(part["relative_mse_reduction"].mean() - relative_null95) if np.isfinite(relative_null95) else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    replay_map = summary.loc[summary["branch"].eq("replay")].set_index("prefix_k")["mean_delta_r2"].to_dict()
    summary["free_minus_replay_delta_r2"] = [
        float(row.mean_delta_r2 - replay_map.get(int(row.prefix_k), np.nan)) if str(row.branch) == "free" else float("nan")
        for row in summary.itertuples(index=False)
    ]
    return {
        "fold_metrics": fold_metrics,
        "summary": summary,
        "null_metrics": null_metrics,
        "row_metrics": pd.DataFrame(row_rows),
        "fold_audit": pd.DataFrame(audit_rows),
        "base_features": base_by_row,
    }


def _direct_direction_metrics(
    ctx: ExperimentContext,
    histories: FixedBArtifact,
    replay: FixedBArtifact,
    rollouts: FixedBArtifact,
    base_features: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del replay
    rows = _primary_rollout_rows(rollouts, branches=("free",))
    vectors = rollouts.arrays["delta_layer2_ux"].astype(np.float64, copy=False)
    fold_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for prefix_k, part in rows.groupby("prefix_k", sort=True):
        part = part.sort_values(["history_family_id", "history_condition", "b_anchor_id"]).reset_index(drop=True)
        y_raw = vectors[part["rollout_row_id"].to_numpy(dtype=np.int64)]
        base = np.stack([base_features[int(value)] for value in part["rollout_row_id"]])
        for fold in range(int(ctx.cfg.fixed_b_folds)):
            train, test, _ = _two_axis_masks(part, fold)
            train_indices = np.flatnonzero(train)
            test_indices = np.flatnonzero(test)
            projection = _fit_target_projection(
                y_raw[train_indices],
                y_raw[test_indices],
                int(ctx.cfg.fixed_b_target_components),
                int(ctx.cfg.fixed_b_protocol_seed) + 30_000 + 100 * int(prefix_k) + fold,
            )
            nuisance = _fit_feature_model(
                base[train_indices],
                projection["train"],
                base[test_indices],
                projection["test"],
                float(ctx.cfg.fixed_b_diagnostic_alpha),
            )
            train_prediction = nuisance["train_prediction"]
            train_residual = projection["train"] - train_prediction
            test_residual = projection["test"] - nuisance["prediction"]
            train_part = part.iloc[train_indices].reset_index(drop=True)
            templates = {
                condition: _unit_vector(train_residual[train_part["history_condition"].eq(condition).to_numpy()].mean(axis=0))
                for condition in ("A", "C")
            }
            test_part = part.iloc[test_indices].reset_index(drop=True)
            pair_scores = []
            for (family_id, anchor_id), pair in test_part.groupby(["history_family_id", "b_anchor_id"], sort=True):
                local = {str(row.history_condition): index for index, row in pair.iterrows()}
                if set(local) != {"A", "C"}:
                    continue
                a_score = float(np.dot(_unit_vector(test_residual[local["A"]]), templates["A"] - templates["C"]))
                c_score = float(np.dot(_unit_vector(test_residual[local["C"]]), templates["C"] - templates["A"]))
                score = 0.5 * (a_score + c_score)
                pair_scores.append(score)
                pair_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "outer_fold": int(fold),
                        "history_family_id": int(family_id),
                        "b_anchor_id": int(anchor_id),
                        "structured_direction_score": score,
                    }
                )
            fold_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "prefix_k": int(prefix_k),
                    "outer_fold": int(fold),
                    "n_test_pairs": int(len(pair_scores)),
                    "mean_structured_direction_score": float(np.mean(pair_scores)),
                }
            )
    pair_table = pd.DataFrame(pair_rows)
    summary_rows = []
    for prefix_k, part in pair_table.groupby("prefix_k", sort=True):
        values = part["structured_direction_score"].to_numpy(dtype=np.float64)
        null_values = _sign_flip_nulls(histories, "structured_direction", int(prefix_k), values)
        leave_b = [float(part.loc[~part["b_anchor_id"].eq(value), "structured_direction_score"].mean()) for value in sorted(part["b_anchor_id"].unique())]
        leave_h = [float(part.loc[~part["history_family_id"].eq(value), "structured_direction_score"].mean()) for value in sorted(part["history_family_id"].unique())]
        observed = float(values.mean())
        null95 = float(np.percentile(null_values, 95))
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "prefix_k": int(prefix_k),
                "observed_score": observed,
                "null_p95": null95,
                "null_excess": observed - null95,
                "min_leave_one_B_score": float(min(leave_b)),
                "min_leave_one_history_score": float(min(leave_h)),
                "passed": int(observed > 0 and observed > null95 and min(leave_b) > 0 and min(leave_h) > 0),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows), pair_table


def _donor_transfer_metrics(
    ctx: ExperimentContext,
    swaps: FixedBArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = swaps.tables["swap_rows"].copy()
    endpoints = {
        "update": swaps.arrays["delta_layer2_ux"].astype(np.float64, copy=False),
        "voltage": swaps.arrays["layer1_voltage_features"].astype(np.float64, copy=False),
        "event": swaps.arrays["layer1_event_features"].astype(np.float64, copy=False),
    }
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for (prefix_k, scope), part in rows.groupby(["prefix_k", "swap_scope"], sort=True):
        allowed_endpoints = ("update",) if str(scope) == "all_layers" else ("update", "voltage", "event")
        for endpoint in allowed_endpoints:
            vectors = endpoints[endpoint]
            for fold in range(int(ctx.cfg.fixed_b_folds)):
                train = ~part["history_fold"].eq(fold) & ~part["b_fold"].eq(fold) & part["is_own_sham"].eq(1)
                test = part["history_fold"].eq(fold) & part["b_fold"].eq(fold)
                train_part = part.loc[train]
                templates = {}
                for condition in ("A", "C"):
                    condition_rows = train_part.loc[
                        train_part["receiver_condition"].eq(condition) & train_part["donor_condition"].eq(condition)
                    ]
                    templates[condition] = _unit_vector(
                        np.mean(vectors[condition_rows["swap_row_id"].to_numpy(dtype=np.int64)], axis=0)
                    )
                values = []
                for (family_id, anchor_id), cell in part.loc[test].groupby(["history_family_id", "b_anchor_id"], sort=True):
                    lookup = {
                        (str(row.receiver_condition), str(row.donor_condition)): int(row.swap_row_id)
                        for row in cell.itertuples(index=False)
                    }
                    for receiver, donor in (("A", "C"), ("C", "A")):
                        vector = _unit_vector(vectors[lookup[(receiver, donor)]])
                        dti = float(np.dot(vector, templates[donor]) - np.dot(vector, templates[receiver]))
                        values.append(dti)
                        trial_rows.append(
                            {
                                "network_seed": int(ctx.cfg.network_seed),
                                "prefix_k": int(prefix_k),
                                "swap_scope": str(scope),
                                "endpoint": endpoint,
                                "outer_fold": int(fold),
                                "history_family_id": int(family_id),
                                "b_anchor_id": int(anchor_id),
                                "receiver_condition": receiver,
                                "donor_condition": donor,
                                "donor_transfer_index": dti,
                                "template_fit_rule": "H_not_f_x_B_not_f_own_shams_only",
                            }
                        )
                fold_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "swap_scope": str(scope),
                        "endpoint": endpoint,
                        "outer_fold": int(fold),
                        "n_train_templates": int(len(train_part)),
                        "n_test_swaps": int(len(values)),
                        "mean_donor_transfer_index": float(np.mean(values)),
                    }
                )
    trial = pd.DataFrame(trial_rows)
    fold_table = pd.DataFrame(fold_rows)
    summary = (
        trial.groupby(["prefix_k", "swap_scope", "endpoint"], sort=True)
        .agg(
            n_swaps=("donor_transfer_index", "size"),
            mean_donor_transfer_index=("donor_transfer_index", "mean"),
            median_donor_transfer_index=("donor_transfer_index", "median"),
            positive_fraction=("donor_transfer_index", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    summary.insert(0, "network_seed", int(ctx.cfg.network_seed))
    return trial, fold_table, summary


def _functional_bridge_metrics(
    ctx: ExperimentContext,
    rollouts: FixedBArtifact,
    histories: FixedBArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory = rollouts.tables["state_trajectory_rows"].copy()
    trajectory = trajectory.loc[
        trajectory["track"].eq("stsp_isolated")
        & trajectory["branch"].eq("free")
        & trajectory["history_condition"].isin(["A", "C"])
        & trajectory["checkpoint"].isin(["early", "b_end", "post"])
    ]
    vectors_by_rollout: dict[int, np.ndarray] = {}
    for rollout_id, part in trajectory.groupby("rollout_row_id", sort=False):
        ordered = part.set_index("checkpoint").loc[["early", "b_end", "post"]]
        vectors_by_rollout[int(rollout_id)] = np.concatenate(
            [np.asarray(json.loads(str(value)), dtype=np.float64) for value in ordered["class_score_vector"]]
        )
    rows = _primary_rollout_rows(rollouts, branches=("free",))
    fold_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for prefix_k, part in rows.groupby("prefix_k", sort=True):
        for fold in range(int(ctx.cfg.fixed_b_folds)):
            train, test, _ = _two_axis_masks(part, fold)
            train_pairs = _paired_contrasts(part.loc[train], vectors_by_rollout)
            test_pairs = _paired_contrasts(part.loc[test], vectors_by_rollout)
            template = _unit_vector(np.mean([value[2] for value in train_pairs], axis=0))
            values = []
            for family_id, anchor_id, contrast in test_pairs:
                score = float(np.dot(contrast, template))
                values.append(score)
                pair_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "outer_fold": int(fold),
                        "history_family_id": int(family_id),
                        "b_anchor_id": int(anchor_id),
                        "trajectory_alignment_score": score,
                    }
                )
            fold_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "prefix_k": int(prefix_k),
                    "outer_fold": int(fold),
                    "n_train_pairs": int(len(train_pairs)),
                    "n_test_pairs": int(len(test_pairs)),
                    "mean_trajectory_alignment_score": float(np.mean(values)),
                }
            )
    pair_table = pd.DataFrame(pair_rows)
    summary_rows = []
    for prefix_k, part in pair_table.groupby("prefix_k", sort=True):
        values = part["trajectory_alignment_score"].to_numpy(dtype=np.float64)
        null_values = _sign_flip_nulls(histories, "functional_bridge", int(prefix_k), values)
        leave_b = [float(part.loc[~part["b_anchor_id"].eq(value), "trajectory_alignment_score"].mean()) for value in sorted(part["b_anchor_id"].unique())]
        leave_h = [float(part.loc[~part["history_family_id"].eq(value), "trajectory_alignment_score"].mean()) for value in sorted(part["history_family_id"].unique())]
        observed = float(values.mean())
        null95 = float(np.percentile(null_values, 95))
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "prefix_k": int(prefix_k),
                "observed_score": observed,
                "null_p95": null95,
                "null_excess": observed - null95,
                "min_leave_one_B_score": float(min(leave_b)),
                "min_leave_one_history_score": float(min(leave_h)),
                "passed": int(observed > 0 and observed > null95 and min(leave_b) > 0 and min(leave_h) > 0),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows), pair_table


def _stagewise_metrics(
    ctx: ExperimentContext,
    histories: FixedBArtifact,
    replay: FixedBArtifact,
    rollouts: FixedBArtifact,
    base_features: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = _primary_rollout_rows(rollouts, branches=("free",))
    rollout_ids = rows["rollout_row_id"].to_numpy(dtype=np.int64)
    base = np.stack([base_features[int(value)] for value in rollout_ids])
    drive = rollouts.arrays["layer1_drive_features"][rollout_ids].astype(np.float64, copy=False)
    voltage = rollouts.arrays["layer1_voltage_features"][rollout_ids].astype(np.float64, copy=False)
    events = rollouts.arrays["layer1_event_features"][rollout_ids].astype(np.float64, copy=False)
    update = rollouts.arrays["delta_layer2_ux"][rollout_ids].astype(np.float64, copy=False)
    replay_features = _matched_replay_features(rollouts, rows)
    stage_defs = {
        "drive_to_voltage": (base, drive, voltage),
        "voltage_to_event": (np.concatenate([base, _compact_vector_features(replay_features)], axis=1), voltage, events),
        "event_to_update": (np.concatenate([base, _compact_vector_features(replay_features)], axis=1), events, update),
    }
    fold_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for prefix_k, part_indices in rows.groupby("prefix_k", sort=True).groups.items():
        local_indices = np.asarray(sorted(part_indices), dtype=np.int64)
        part = rows.iloc[local_indices].reset_index(drop=True)
        for endpoint, (base_all, added_all, target_all) in stage_defs.items():
            base_local = base_all[local_indices]
            added_local = added_all[local_indices]
            target_local = target_all[local_indices]
            for fold in range(int(ctx.cfg.fixed_b_folds)):
                train, test, _ = _two_axis_masks(part, fold)
                train_idx = np.flatnonzero(train)
                test_idx = np.flatnonzero(test)
                projection = _fit_target_projection(
                    target_local[train_idx],
                    target_local[test_idx],
                    int(ctx.cfg.fixed_b_target_components),
                    int(ctx.cfg.fixed_b_protocol_seed) + 50_000 + 1000 * int(prefix_k) + fold,
                )
                reduced = _fit_feature_model(
                    base_local[train_idx],
                    projection["train"],
                    base_local[test_idx],
                    projection["test"],
                    float(ctx.cfg.fixed_b_diagnostic_alpha),
                )
                extended_matrix = np.concatenate([base_local, added_local], axis=1)
                extended = _fit_feature_model(
                    extended_matrix[train_idx],
                    projection["train"],
                    extended_matrix[test_idx],
                    projection["test"],
                    float(ctx.cfg.fixed_b_diagnostic_alpha),
                )
                fold_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "prefix_k": int(prefix_k),
                        "endpoint": endpoint,
                        "outer_fold": int(fold),
                        "delta_r2": float(extended["r2"] - reduced["r2"]),
                        "relative_mse_reduction": float((reduced["mse"] - extended["mse"]) / max(reduced["mse"], 1e-12)),
                    }
                )
                null_specs = histories.tables["null_specs"].loc[
                    histories.tables["null_specs"]["purpose"].eq(endpoint)
                    & histories.tables["null_specs"]["prefix_k"].eq(prefix_k)
                ]
                for spec in null_specs.itertuples(index=False):
                    null_added = _null_added_features(part, added_local, endpoint, int(spec.random_seed))
                    null_matrix = np.concatenate([base_local, null_added], axis=1)
                    null_fit = _fit_feature_model(
                        null_matrix[train_idx],
                        projection["train"],
                        null_matrix[test_idx],
                        projection["test"],
                        float(ctx.cfg.fixed_b_diagnostic_alpha),
                    )
                    null_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "prefix_k": int(prefix_k),
                            "endpoint": endpoint,
                            "outer_fold": int(fold),
                            "replicate": int(spec.replicate),
                            "delta_r2": float(null_fit["r2"] - reduced["r2"]),
                        }
                    )
    fold_table = pd.DataFrame(fold_rows)
    null_table = pd.DataFrame(null_rows)
    summary_rows = []
    for (prefix_k, endpoint), part in fold_table.groupby(["prefix_k", "endpoint"], sort=True):
        null = null_table.loc[null_table["prefix_k"].eq(prefix_k) & null_table["endpoint"].eq(endpoint)]
        null_means = null.groupby("replicate")["delta_r2"].mean().to_numpy(dtype=np.float64)
        observed = float(part["delta_r2"].mean())
        null95 = float(np.percentile(null_means, 95))
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "prefix_k": int(prefix_k),
                "endpoint": str(endpoint),
                "mean_delta_r2": observed,
                "mean_relative_mse_reduction": float(part["relative_mse_reduction"].mean()),
                "null_delta_r2_p95": null95,
                "null_excess": observed - null95,
                "passed": int(observed > 0 and observed > null95),
            }
        )
    return fold_table, null_table, pd.DataFrame(summary_rows)


def _feature_matrices(
    ctx: ExperimentContext,
    histories: FixedBArtifact,
    replay: FixedBArtifact,
    rows: pd.DataFrame,
    prefix_k: int,
    *,
    spatial_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_table = histories.tables["prestate_features"].loc[
        histories.tables["prestate_features"]["prefix_k"].eq(prefix_k)
    ].sort_values("history_row_id").reset_index(drop=True)
    feature_index = {int(row.history_row_id): index for index, row in enumerate(feature_table.itertuples(index=False))}
    metadata = {
        "network_seed",
        "history_row_id",
        "history_family_id",
        "candidate_family_id",
        "history_condition",
        "prefix_k",
        "balance_stratum",
    }
    scalar_columns = [
        column
        for column in feature_table.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(feature_table[column])
    ]
    history_scalars = feature_table[scalar_columns].to_numpy(dtype=np.float64)
    b_specs = histories.tables["b_anchor_specs"].sort_values("b_anchor_id").reset_index(drop=True)
    replay_spikes = replay.arrays[f"replay_k{prefix_k}"][:, : int(ctx.cfg.fixed_b_stimulus_steps)]
    event_counts = replay_spikes.sum(axis=1).astype(np.float64)
    u = histories.arrays[f"k{prefix_k}__layer2__u"].astype(np.float64, copy=False)
    x = histories.arrays[f"k{prefix_k}__layer2__x"].astype(np.float64, copy=False)
    gain = u * x - float(ctx.net.layer2.stsp_U)
    rng = np.random.default_rng(spatial_seed) if spatial_seed is not None else None
    base_rows = []
    interaction_rows = []
    for row in rows.itertuples(index=False):
        h_index = feature_index[int(row.history_row_id)]
        anchor_id = int(row.b_anchor_id)
        b_event = event_counts[anchor_id]
        history_values = history_scalars[h_index]
        b_row = b_specs.iloc[anchor_id]
        b_values = np.concatenate(
            [
                np.asarray(
                    [
                        b_event.sum(),
                        np.linalg.norm(b_event),
                        np.count_nonzero(b_event),
                        float(b_row["B_pixel_sum"]),
                        float(b_row["B_foreground_area"]),
                        float(row.history_condition == "A"),
                        float(row.history_condition == "C"),
                        float(row.B_label),
                    ]
                ),
                b_event.sum(axis=(1, 2)),
            ]
        )
        base_raw = np.concatenate([history_values, b_values])
        base_feature = _bounded_separable_basis(base_raw)
        local_gain = gain[h_index]
        if rng is not None:
            permutation = rng.permutation(local_gain.shape[1] * local_gain.shape[2])
            local_gain = local_gain.reshape(local_gain.shape[0], -1)[:, permutation].reshape(local_gain.shape)
        spatial = local_gain.mean(axis=0) * b_event.sum(axis=0)
        channel = local_gain.mean(axis=(1, 2)) * b_event.sum(axis=(1, 2))
        joint = np.concatenate(
            [
                np.asarray([spatial.sum(), np.abs(spatial).sum(), np.linalg.norm(spatial), channel.sum()]),
                spatial.reshape(-1),
                np.abs(spatial).reshape(-1),
                channel,
                np.abs(channel),
            ]
        )
        base_rows.append(base_feature)
        interaction_rows.append(np.concatenate([base_feature, joint]))
    return np.stack(base_rows), np.stack(interaction_rows)


def _bounded_separable_basis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.maximum(np.abs(finite), 1.0)
    bounded = finite / scale
    return np.concatenate([finite, np.tanh(finite), bounded * bounded, bounded / (1.0 + np.abs(bounded))])


def _fit_target_projection(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    target_components: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    max_components = min(int(target_components), len(train_scaled) - 1, train_scaled.shape[1])
    if max_components < 1:
        raise RuntimeError("Fixed-B target projection has no supported component")
    solver = "randomized" if max_components < min(train_scaled.shape) else "full"
    pca = PCA(n_components=max_components, svd_solver=solver, random_state=int(random_seed)).fit(train_scaled)
    return {
        "train": _normalize_rows(pca.transform(train_scaled)),
        "test": _normalize_rows(pca.transform(test_scaled)),
    }


def _fit_feature_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(x_train)
    train_x = scaler.transform(x_train)
    test_x = scaler.transform(x_test)
    model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
    ).fit(train_x, y_train)
    prediction = np.asarray(model.predict(test_x))
    train_prediction = np.asarray(model.predict(train_x))
    if prediction.ndim == 1:
        prediction = prediction[:, None]
        train_prediction = train_prediction[:, None]
    mse = float(np.mean((y_test - prediction) ** 2))
    baseline = float(
        np.mean(
            (y_test - y_train.mean(axis=0, keepdims=True)) ** 2
        )
    )
    max_abs_test_z = (
        float(np.max(np.abs(test_x)))
        if test_x.size
        else 0.0
    )
    return {
        "prediction": prediction,
        "train_prediction": train_prediction,
        "mse": mse,
        "r2": float(1.0 - mse / max(baseline, 1e-12)),
        "supported_feature_count": int(x_train.shape[1]),
        "clipped_test_value_count": 0,
        "max_abs_test_z_before_clip": max_abs_test_z,
        "max_abs_test_z_after_clip": max_abs_test_z,
        "test_scaling_rule": (
            "unclipped_train_fitted_standard_scaler_transform"
        ),
    }


def _select_alpha(
    ctx: ExperimentContext,
    rows: pd.DataFrame,
    outer_train_indices: np.ndarray,
    features: np.ndarray,
    target_raw: np.ndarray,
    outer_fold: int,
) -> float:
    alphas = tuple(float(value) for value in ctx.cfg.fixed_b_ridge_alphas)
    if len(alphas) == 1:
        return alphas[0]
    outer_rows = rows.iloc[outer_train_indices].reset_index(drop=True)
    scores = {alpha: [] for alpha in alphas}
    for inner_fold in range(int(ctx.cfg.fixed_b_folds)):
        if inner_fold == int(outer_fold):
            continue
        train, test, _ = _two_axis_masks(outer_rows, inner_fold)
        inner_train = outer_train_indices[np.flatnonzero(train)]
        inner_test = outer_train_indices[np.flatnonzero(test)]
        if len(inner_train) < 4 or len(inner_test) < 2:
            continue
        projection = _fit_target_projection(
            target_raw[inner_train],
            target_raw[inner_test],
            int(ctx.cfg.fixed_b_target_components),
            int(ctx.cfg.fixed_b_protocol_seed) + 70_000 + 100 * outer_fold + inner_fold,
        )
        for alpha in alphas:
            fit = _fit_feature_model(
                features[inner_train],
                projection["train"],
                features[inner_test],
                projection["test"],
                alpha,
            )
            scores[alpha].append(float(fit["mse"]))
    available = {alpha: float(np.mean(values)) for alpha, values in scores.items() if values}
    if not available:
        return min(alphas, key=lambda value: abs(np.log10(max(value, 1e-12))))
    return min(available, key=lambda value: (available[value], value))


def _two_axis_masks(rows: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_test = rows["history_fold"].eq(fold).to_numpy()
    b_test = rows["b_fold"].eq(fold).to_numpy()
    test = history_test & b_test
    train = ~history_test & ~b_test
    guard = history_test ^ b_test
    if bool(np.any(train & test) or np.any(train & guard) or np.any(test & guard)):
        raise RuntimeError("Two-axis blocked fold masks overlap")
    return train, test, guard


def _fold_audit_row(
    ctx: ExperimentContext,
    rows: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    guard_indices: np.ndarray,
    *,
    prefix_k: int,
    branch: str,
    fold: int,
    base: np.ndarray,
    interaction: np.ndarray,
    base_fit: Mapping[str, Any],
    interaction_fit: Mapping[str, Any],
) -> dict[str, Any]:
    train_keys = _row_identity_hash(rows.iloc[train_indices])
    test_keys = _row_identity_hash(rows.iloc[test_indices])
    guard_keys = _row_identity_hash(rows.iloc[guard_indices])
    train_families = set(
        rows.iloc[train_indices]["history_family_id"].astype(int)
    )
    test_families = set(
        rows.iloc[test_indices]["history_family_id"].astype(int)
    )
    train_anchors = set(
        rows.iloc[train_indices]["b_anchor_id"].astype(int)
    )
    test_anchors = set(
        rows.iloc[test_indices]["b_anchor_id"].astype(int)
    )
    leakage = bool(
        train_families & test_families
        or train_anchors & test_anchors
    )
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "prefix_k": int(prefix_k),
        "branch": branch,
        "outer_fold": int(fold),
        "train_row_hash": train_keys,
        "test_row_hash": test_keys,
        "guard_row_hash": guard_keys,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "guard_rows": int(len(guard_indices)),
        "base_feature_count": int(base.shape[1]),
        "interaction_feature_count": int(interaction.shape[1]),
        "base_supported_feature_count": int(
            base_fit["supported_feature_count"]
        ),
        "interaction_supported_feature_count": int(
            interaction_fit["supported_feature_count"]
        ),
        "base_clipped_test_values": int(
            base_fit["clipped_test_value_count"]
        ),
        "interaction_clipped_test_values": int(
            interaction_fit["clipped_test_value_count"]
        ),
        "base_max_abs_test_z_before_clip": float(
            base_fit["max_abs_test_z_before_clip"]
        ),
        "base_max_abs_test_z_after_clip": float(
            base_fit["max_abs_test_z_after_clip"]
        ),
        "interaction_max_abs_test_z_before_clip": float(
            interaction_fit["max_abs_test_z_before_clip"]
        ),
        "interaction_max_abs_test_z_after_clip": float(
            interaction_fit["max_abs_test_z_after_clip"]
        ),
        "test_scaling_rule": str(base_fit["test_scaling_rule"]),
        "train_only_preprocessing": 1,
        "identity_leakage": int(leakage),
        "passed": int(not leakage),
    }


def _engineering_gates(
    ctx: ExperimentContext,
    *,
    histories: FixedBArtifact,
    rollouts: FixedBArtifact,
    swaps: FixedBArtifact,
    interaction_audit: pd.DataFrame,
    restoration: pd.DataFrame,
    exact_b: pd.DataFrame,
    event_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(gate: str, passed: bool, value: Any, threshold: str) -> None:
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "gate": gate,
                "value": value,
                "threshold": threshold,
                "passed": int(bool(passed)),
            }
        )

    add("exact_B_identity", bool(exact_b["passed"].eq(1).all()), int(exact_b["passed"].min()), "all anchors pass")
    add("restoration_exact_and_margin", bool(restoration["passed"].eq(1).all()), int(restoration["passed"].min()), "exact and <=0.10 native separation")
    isolation = swaps.tables["swap_isolation_audit"]
    add("fast_state_isolation", bool(isolation["fast_state_equalized"].eq(1).all()), int(isolation["fast_state_equalized"].min()), "one fast-state hash per recipient batch")
    add("fold_identity_leakage", bool(interaction_audit["passed"].eq(1).all()), int(interaction_audit["passed"].min()), "zero train/test/guard leakage")
    finite = np.isfinite(rollouts.arrays["delta_layer2_ux"]).all()
    add("finite_updates", bool(finite), int(finite), "all finite")
    add("event_round_trip", bool(event_audit["passed"].eq(1).all()), int(event_audit["passed"].min()), "packed events reproduce early counts")
    source_balance = histories.tables["source_balance"]
    max_smd = float(source_balance["abs_standardized_mean_difference"].max()) if len(source_balance) else float("nan")
    add(
        "source_balance",
        np.isfinite(max_smd)
        and max_smd <= float(ctx.cfg.fixed_b_source_match_max_smd) + 1e-12,
        max_smd,
        f"<={float(ctx.cfg.fixed_b_source_match_max_smd):.6g}",
    )
    return pd.DataFrame(rows)


def _restoration_summary(ctx: ExperimentContext, histories: FixedBArtifact) -> pd.DataFrame:
    audit = histories.tables["restoration_audit"].copy()
    rows = []
    for (prefix_k, branch), part in audit.groupby(["prefix_k", "branch"], sort=True):
        exact = bool(
            float(part["max_abs_layer2_ux_error"].max()) == 0.0
            and part["prediction_equal"].eq(1).all()
            and part["spike_counts_equal"].eq(1).all()
        )
        margin = bool(part["restoration_margin_pass"].eq(1).all())
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "prefix_k": int(prefix_k),
                "branch": str(branch),
                "max_abs_layer2_ux_error": float(part["max_abs_layer2_ux_error"].max()),
                "max_normalized_restoration_error": float(part["normalized_restoration_error"].max()),
                "exact_equivalence": int(exact),
                "margin_pass": int(margin),
                "passed": int(exact and margin),
            }
        )
    return pd.DataFrame(rows)


def _exact_b_audit(ctx: ExperimentContext, inputs: FixedBArtifact, rollouts: FixedBArtifact) -> pd.DataFrame:
    rows = rollouts.tables["rollout_rows"]
    manifest = inputs.tables["input_manifest"].set_index("b_anchor_id")
    output = []
    for anchor_id, part in rows.groupby("b_anchor_id", sort=True):
        expected = str(manifest.loc[int(anchor_id), "tensor_sha256"])
        hashes = set(str(value) for value in part["exact_b_tensor_sha256"])
        output.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "b_anchor_id": int(anchor_id),
                "rows": int(len(part)),
                "unique_tensor_hashes": int(len(hashes)),
                "expected_tensor_sha256": expected,
                "observed_tensor_sha256": next(iter(hashes)) if len(hashes) == 1 else "",
                "input_energy": float(manifest.loc[int(anchor_id), "input_energy"]),
                "passed": int(hashes == {expected}),
            }
        )
    return pd.DataFrame(output)


def _event_recompute_audit(ctx: ExperimentContext, rollouts: FixedBArtifact) -> pd.DataFrame:
    manifest = rollouts.tables["layer1_event_manifest"].copy()
    if manifest.empty:
        return pd.DataFrame([{"network_seed": int(ctx.cfg.network_seed), "rows": 0, "max_count_error": float("inf"), "passed": 0}])
    shapes = {tuple(int(value) for value in ast.literal_eval(str(value))) for value in manifest["unpacked_shape"]}
    if len(shapes) != 1:
        raise ValueError("Packed Layer1 event rows have inconsistent shapes")
    unpacked = unpack_event_bits(rollouts.arrays["layer1_early_event_bits"], next(iter(shapes)))
    counts = unpacked.reshape(len(unpacked), -1).sum(axis=1)
    rollout_counts = rollouts.tables["rollout_rows"].set_index("rollout_row_id")["layer1_early_spike_count"]
    expected = manifest["rollout_row_id"].map(rollout_counts).to_numpy(dtype=np.int64)
    error = np.abs(counts.astype(np.int64) - expected)
    return pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "rows": int(len(manifest)),
                "max_count_error": int(error.max()),
                "packed_array_sha256": _array_sha256(rollouts.arrays["layer1_early_event_bits"]),
                "passed": int(int(error.max()) == 0),
            }
        ]
    )


def _existing_chain_audit(ctx: ExperimentContext) -> pd.DataFrame:
    root = Path("results/paper_figure_multi_seed/statistics")
    audit_path = root / "manuscript_stats_audit.csv"
    long_path = root / "manuscript_stats_long.csv"
    task_ids = ("Q16", "Q21", "Q22", "Q22E")
    if not audit_path.exists() or not long_path.exists():
        return pd.DataFrame(
            [
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "task_id": task_id,
                    "status": "missing_statistics_input",
                    "usable_seeds": "",
                    "source_files": "",
                    "required_columns": "",
                    "row_count_check": "",
                    "statistics_rows": 0,
                    "statistics_network_counts": "",
                    "source_file_count": 0,
                    "source_manifest_sha256": "",
                    "audit_file_sha256": "",
                    "statistics_file_sha256": "",
                    "claim_role": "existing_chain_only_not_fixed_B_core",
                    "passed": 0,
                }
                for task_id in task_ids
            ]
        )
    audit = pd.read_csv(audit_path, keep_default_na=False)
    statistics = pd.read_csv(long_path, keep_default_na=False)
    output = []
    for task_id in task_ids:
        part = audit.loc[audit["task_id"].eq(task_id)]
        if len(part) != 1:
            raise RuntimeError(f"Existing-chain audit row is not unique for {task_id}")
        row = part.iloc[0]
        stats = statistics.loc[statistics["task_id"].eq(task_id)]
        source_files = [value.strip() for value in str(row["source_files"]).split(";") if value.strip()]
        required = _required_source_columns(str(row["required_columns"]))
        source_records: list[str] = []
        source_seeds: set[int] = set()
        schema_valid = True
        for source_file in source_files:
            matches = sorted(root.parent.glob(f"*/seed_*/{source_file}"))
            for path in matches:
                seed = int(path.parents[2].name.removeprefix("seed_"))
                source_seeds.add(seed)
                columns = set(pd.read_csv(path, nrows=0).columns)
                schema_valid = schema_valid and set(required.get(source_file, ())).issubset(columns)
                source_records.append(
                    f"{path.relative_to(root.parent).as_posix()}|{_file_sha256(path)}"
                )
        expected_seeds = set(range(1000, 1020))
        expected_file_count = 20 * len(source_files)
        network_counts = sorted(set(stats["n_networks"].astype(int))) if len(stats) else []
        passed = bool(
            str(row["status"]) == "ok"
            and str(row["usable_seeds"]) == "1000..1019"
            and str(row["calculable"]).lower() == "true"
            and len(stats) > 0
            and network_counts == [20]
            and len(source_records) == expected_file_count
            and source_seeds == expected_seeds
            and schema_valid
        )
        manifest_payload = "\n".join(sorted(source_records)).encode("utf-8")
        output.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "task_id": task_id,
                "status": str(row["status"]),
                "usable_seeds": str(row["usable_seeds"]),
                "source_files": str(row["source_files"]),
                "required_columns": str(row["required_columns"]),
                "row_count_check": str(row["row_count_check"]),
                "statistics_rows": int(len(stats)),
                "statistics_network_counts": json.dumps(network_counts),
                "source_file_count": int(len(source_records)),
                "source_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "audit_file_sha256": _file_sha256(audit_path),
                "statistics_file_sha256": _file_sha256(long_path),
                "claim_role": "existing_chain_support_only_not_fixed_B_core",
                "passed": int(passed),
            }
        )
    return pd.DataFrame(output)


def _required_source_columns(encoded: str) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for part in encoded.split(" | "):
        source, separator, columns = part.partition(": ")
        if separator:
            output[source.strip()] = tuple(
                value.strip() for value in columns.split(",") if value.strip()
            )
    return output


def _influence_diagnostics(
    ctx: ExperimentContext,
    *,
    interaction_rows: pd.DataFrame,
    direct_pairs: pd.DataFrame,
    donor_trials: pd.DataFrame,
    functional_pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    definitions = [
        ("structured_interaction", interaction_rows.loc[interaction_rows["branch"].eq("free")], "row_mse_advantage"),
        ("structured_direction", direct_pairs, "structured_direction_score"),
        ("all_layer_donor_transfer", donor_trials.loc[donor_trials["swap_scope"].eq("all_layers") & donor_trials["endpoint"].eq("update")], "donor_transfer_index"),
        ("functional_bridge", functional_pairs, "trajectory_alignment_score"),
    ]
    for endpoint, table, value_column in definitions:
        for prefix_k, part in table.groupby("prefix_k", sort=True):
            overall = float(part[value_column].mean())
            leave_b = [float(part.loc[~part["b_anchor_id"].eq(value), value_column].mean()) for value in sorted(part["b_anchor_id"].unique())]
            leave_h = [float(part.loc[~part["history_family_id"].eq(value), value_column].mean()) for value in sorted(part["history_family_id"].unique())]
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "endpoint": endpoint,
                    "prefix_k": int(prefix_k),
                    "overall_value": overall,
                    "min_leave_one_B_value": float(min(leave_b)),
                    "min_leave_one_history_value": float(min(leave_h)),
                    "sign_stable": int(overall > 0 and min(leave_b) > 0 and min(leave_h) > 0),
                }
            )
    return pd.DataFrame(rows)


def _tiered_verdict(
    ctx: ExperimentContext,
    *,
    protocol: FixedBArtifact,
    engineering: pd.DataFrame,
    existing_chain: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    direct_summary: pd.DataFrame,
    donor_summary: pd.DataFrame,
    functional_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
    influence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(tier: str, endpoint: str, prefix_k: int | str, value: float, threshold: str, passed: bool) -> None:
        checks.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "tier": tier,
                "endpoint": endpoint,
                "prefix_k": prefix_k,
                "value": float(value),
                "threshold": threshold,
                "passed": int(bool(passed)),
            }
        )

    engineering_pass = bool(engineering["passed"].eq(1).all())
    add("engineering", "all_engineering_gates", "all", float(engineering_pass), "==1", engineering_pass)
    existing_chain_pass = bool(existing_chain["passed"].eq(1).all())
    add(
        "engineering",
        "existing_chain_lineage",
        "all",
        float(existing_chain_pass),
        "Q16/Q21/Q22/Q22E all pass lineage, schema, row, seed, and hash audit",
        existing_chain_pass,
    )
    for prefix_k in sorted(int(value) for value in ctx.cfg.fixed_b_prefix_depths):
        direct = direct_summary.loc[direct_summary["prefix_k"].eq(prefix_k)].iloc[0]
        add("core", "structured_direction", prefix_k, float(direct["null_excess"]), ">0 with all leave-one scores >0", bool(direct["passed"]))
        interaction = interaction_summary.loc[
            interaction_summary["prefix_k"].eq(prefix_k) & interaction_summary["branch"].eq("free")
        ].iloc[0]
        interaction_pass = bool(
            float(interaction["mean_delta_r2"]) > 0
            and float(interaction["mean_relative_mse_reduction"]) >= 0.01
            and float(interaction["delta_r2_null_excess"]) > 0
            and float(interaction["relative_mse_null_excess"]) > 0
        )
        add("core", "structured_interaction", prefix_k, float(interaction["mean_delta_r2"]), ">0; relative MSE >=0.01; both above spatial null", interaction_pass)
        donor = donor_summary.loc[
            donor_summary["prefix_k"].eq(prefix_k)
            & donor_summary["swap_scope"].eq("all_layers")
            & donor_summary["endpoint"].eq("update")
        ].iloc[0]
        add("core", "all_layer_donor_transfer", prefix_k, float(donor["mean_donor_transfer_index"]), ">0", float(donor["mean_donor_transfer_index"]) > 0)
        bridge = functional_summary.loc[functional_summary["prefix_k"].eq(prefix_k)].iloc[0]
        add("core", "functional_bridge", prefix_k, float(bridge["null_excess"]), ">0 with all leave-one scores >0", bool(bridge["passed"]))
        for endpoint in ("drive_to_voltage", "voltage_to_event", "event_to_update"):
            stage = stage_summary.loc[
                stage_summary["prefix_k"].eq(prefix_k) & stage_summary["endpoint"].eq(endpoint)
            ].iloc[0]
            add("strong", endpoint, prefix_k, float(stage["null_excess"]), ">0 and observed delta R2 above 95th null", bool(stage["passed"]))
        for endpoint, label in (
            ("voltage", "layer1_voltage_donor_transfer"),
            ("event", "layer1_event_donor_transfer"),
            ("update", "layer1_update_donor_transfer"),
        ):
            donor = donor_summary.loc[
                donor_summary["prefix_k"].eq(prefix_k)
                & donor_summary["swap_scope"].eq("layer1_only")
                & donor_summary["endpoint"].eq(endpoint)
            ].iloc[0]
            value = float(donor["mean_donor_transfer_index"])
            add("strong", label, prefix_k, value, ">0", value > 0)
        free_minus_replay = float(interaction["free_minus_replay_delta_r2"])
        add("strong", "free_minus_replay", prefix_k, free_minus_replay, ">0", free_minus_replay > 0)
    influence_pass = bool(influence["sign_stable"].eq(1).all())
    add("core", "influence_stability", "all", float(influence_pass), "all core endpoint leave-one signs positive", influence_pass)
    checklist = pd.DataFrame(checks)
    core_pass = (
        engineering_pass
        and existing_chain_pass
        and bool(
            checklist.loc[
                checklist["tier"].eq("core"),
                "passed",
            ].eq(1).all()
        )
    )
    strong_pass = core_pass and bool(checklist.loc[checklist["tier"].eq("strong"), "passed"].eq(1).all())
    if not engineering_pass:
        verdict = "engineering_invalid"
    elif not core_pass:
        verdict = "core_fail"
    elif not strong_pass:
        verdict = "core_pass_strong_fail"
    else:
        verdict = "core_pass_strong_pass"
    claim_rows = [
        {
            "network_seed": int(ctx.cfg.network_seed),
            "claim_id": "core_history_conditioned_updating",
            "tier": "core",
            "supported_development": int(core_pass),
            "allowed_wording": "development evidence for history-conditioned passive-corrected successor-state updating" if core_pass else "not supported by seed-1000 development evidence",
            "forbidden_wording": "population-level gap closure; chunk or fusion claim",
        },
        {
            "network_seed": int(ctx.cfg.network_seed),
            "claim_id": "strong_firing_redirected_writeback",
            "tier": "strong",
            "supported_development": int(strong_pass),
            "allowed_wording": "development evidence for the firing/write-back path" if strong_pass else "strong firing/write-back wording is not supported",
            "forbidden_wording": "population-level gap closure; confirmatory mechanism claim",
        },
    ]
    decision = {
        "verdict": verdict,
        "engineering_valid": engineering_pass,
        "existing_chain_valid": existing_chain_pass,
        "core_development_pass": core_pass,
        "strong_development_pass": strong_pass,
        "continuation_eligible": bool(core_pass),
        "eligible_tracks": (["core_only", "strong_track"] if strong_pass else (["core_only"] if core_pass else [])),
        "protocol_digest": protocol_digest(protocol),
        "prefix_depths": [int(value) for value in ctx.cfg.fixed_b_prefix_depths],
        "remaining_seeds_allowed": False,
    }
    return checklist, pd.DataFrame(claim_rows), decision


def _primary_rollout_rows(
    rollouts: FixedBArtifact,
    *,
    branches: Sequence[str],
) -> pd.DataFrame:
    rows = rollouts.tables["rollout_rows"]
    return (
        rows.loc[
            rows["track"].eq("stsp_isolated")
            & rows["history_condition"].isin(["A", "C"])
            & rows["branch"].isin(list(branches))
        ]
        .copy()
        .reset_index(drop=True)
    )


def _paired_contrasts(rows: pd.DataFrame, vectors_by_rollout: Mapping[int, np.ndarray]) -> list[tuple[int, int, np.ndarray]]:
    output = []
    for (family_id, anchor_id), pair in rows.groupby(["history_family_id", "b_anchor_id"], sort=True):
        lookup = {str(row.history_condition): int(row.rollout_row_id) for row in pair.itertuples(index=False)}
        if set(lookup) != {"A", "C"}:
            continue
        output.append(
            (
                int(family_id),
                int(anchor_id),
                np.asarray(vectors_by_rollout[lookup["A"]] - vectors_by_rollout[lookup["C"]], dtype=np.float64),
            )
        )
    return output


def _matched_replay_features(rollouts: FixedBArtifact, free_rows: pd.DataFrame) -> np.ndarray:
    rows = rollouts.tables["rollout_rows"]
    replay = rows.loc[
        rows["track"].eq("stsp_isolated")
        & rows["branch"].eq("replay")
        & rows["history_condition"].isin(["A", "C"])
    ]
    lookup = {
        (int(row.prefix_k), int(row.history_family_id), str(row.history_condition), int(row.b_anchor_id)): int(row.rollout_row_id)
        for row in replay.itertuples(index=False)
    }
    indices = [
        lookup[(int(row.prefix_k), int(row.history_family_id), str(row.history_condition), int(row.b_anchor_id))]
        for row in free_rows.itertuples(index=False)
    ]
    return rollouts.arrays["delta_layer2_ux"][np.asarray(indices, dtype=np.int64)].astype(np.float64, copy=False)


def _compact_vector_features(values: np.ndarray, bins: int = 32) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    edges = np.linspace(0, array.shape[1], min(int(bins), array.shape[1]) + 1, dtype=np.int64)
    chunks = [array[:, edges[index] : edges[index + 1]].mean(axis=1) for index in range(len(edges) - 1)]
    return np.column_stack(
        [array.mean(axis=1), np.linalg.norm(array, axis=1), array.std(axis=1), *chunks]
    )


def _null_added_features(rows: pd.DataFrame, added: np.ndarray, endpoint: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    output = np.asarray(added, dtype=np.float64).copy()
    if endpoint == "drive_to_voltage":
        scalar_count = min(36, output.shape[1])
        for row_index in range(len(output)):
            output[row_index, scalar_count:] = output[row_index, scalar_count:][rng.permutation(output.shape[1] - scalar_count)]
        return output
    for _, pair_indices in rows.groupby(["history_family_id", "b_anchor_id"], sort=False).groups.items():
        indices = np.asarray(list(pair_indices), dtype=np.int64)
        if len(indices) == 2 and bool(rng.integers(0, 2)):
            output[indices] = output[indices[::-1]]
    return output


def _sign_flip_nulls(histories: FixedBArtifact, purpose: str, prefix_k: int, values: np.ndarray) -> np.ndarray:
    specs = histories.tables["null_specs"].loc[
        histories.tables["null_specs"]["purpose"].eq(purpose)
        & histories.tables["null_specs"]["prefix_k"].eq(prefix_k)
    ]
    output = []
    for row in specs.itertuples(index=False):
        rng = np.random.default_rng(int(row.random_seed))
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values), replace=True)
        output.append(float(np.mean(values * signs)))
    return np.asarray(output, dtype=np.float64)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(output, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-12)
    output[valid] /= norms[valid, None]
    output[~valid] = 0.0
    return output


def _unit_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-12 else np.zeros_like(array)


def _row_identity_hash(rows: pd.DataFrame) -> str:
    columns = ["rollout_row_id", "history_family_id", "history_condition", "b_anchor_id"]
    encoded = rows.loc[:, columns].sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["analyze_fixed_b_single_seed"]
