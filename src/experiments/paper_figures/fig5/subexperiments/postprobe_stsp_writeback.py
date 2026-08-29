from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.experiments.common.dataset import encode_images
from src.experiments.common.monitored_dms import snapshot_boundary_state
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.paper_figures.fig5.artifacts import (
    ProbeStspUpdateArtifact,
    finalize_probe_stsp_update_artifact,
    iter_probe_stsp_update_shards,
    prepare_probe_stsp_update_artifact_dir,
    write_probe_stsp_update_shard,
)
from src.experiments.paper_figures.fig5.cache_keys import dataframe_hash
from src.experiments.paper_figures.fig5.constants import (
    L1_STSP_PERTURBATION_CONDITIONS,
    MAIN_CONDITIONS,
    PRIMARY_LAYER,
    REFERENCE_CONDITIONS,
    SUPP_CONDITIONS,
)
from src.experiments.paper_figures.fig5.output import save_csv as _save_csv
from src.experiments.paper_figures.fig5.output import utc_now
from src.experiments.paper_figures.fig5.schemas import (
    POSTPROBE_L1_FIRING_BRIDGE_COLUMNS,
    POSTPROBE_L2_BY_NETWORK_COLUMNS,
    POSTPROBE_L2_BY_TRIAL_COLUMNS,
    POSTPROBE_L2_HISTORY_BY_TRIAL_COLUMNS,
    POSTPROBE_L2_HISTORY_COMPOSITION_COLUMNS,
    POSTPROBE_L2_MEMORY_OVERLAP_COLUMNS,
    POSTPROBE_L2_SUMMARY_COLUMNS,
    POSTPROBE_MAGNITUDE_QC_COLUMNS,
    PROBE_STSP_CONDITION_MANIFEST_COLUMNS,
    SNAPSHOT_MANIFEST_COLUMNS,
)
from src.experiments.paper_figures.fig5.subexperiments.helpers import (
    _apply_l1_stsp_perturbation,
    _fig5d_condition_label,
    _images_for_ids,
    _iter_batches,
    _l1_stsp_perturbation_mode,
    _layer_stsp_baseline_u,
    _progress,
    _resize_mask,
    _restore_boundary_state,
    _slice_boundary,
    _step_network_once,
)
from src.experiments.paper_figures.fig5.types import ExperimentContext, LocalSupportCompetitionBank

SNAPSHOT_LAYER = "layer2_presynaptic"
STSP_STATE_LAYER = "layer2"
L1_BRIDGE_LAYER = "layer1"
MEMORY_CONTROL_CONDITION = "static_frozen"
SNAPSHOT_VARIABLE_SETS = (
    "u_pre",
    "x_pre",
    "G_pre",
    "u_post",
    "x_post",
    "G_post",
    "prior_l2_update_event",
    "probe_l2_update_event",
    "probe_l2_update_opportunity",
    "prior_l2_retained_memory",
    "prior_l1_fire",
    "probe_l1_fire",
    "early_l1_fire",
)


@dataclass(frozen=True)
class BranchSnapshot:
    arrays: dict[str, np.ndarray]


def probe_stsp_update_conditions() -> tuple[str, ...]:
    return tuple(dict.fromkeys(MAIN_CONDITIONS + REFERENCE_CONDITIONS + SUPP_CONDITIONS))


def probe_stsp_update_layers() -> tuple[str, ...]:
    return (SNAPSHOT_LAYER,)


def probe_stsp_update_variable_sets() -> tuple[str, ...]:
    return SNAPSHOT_VARIABLE_SETS


def unit_group_digest(unit_groups: pd.DataFrame) -> str:
    return dataframe_hash(unit_groups.reset_index(drop=True).copy())


def build_and_save_probe_stsp_update_artifact(
    ctx: ExperimentContext,
    bank: LocalSupportCompetitionBank,
    *,
    task_dir: Path,
    cache_key: Mapping[str, Any],
    trial_hash: str,
    parent_support_bank_digest: str,
) -> ProbeStspUpdateArtifact:
    prepare_probe_stsp_update_artifact_dir(task_dir)
    tables, snapshot_manifest = _build_probe_stsp_update_payload(
        ctx,
        bank,
        task_dir=task_dir,
        trial_hash=trial_hash,
        parent_support_bank_digest=parent_support_bank_digest,
    )
    return finalize_probe_stsp_update_artifact(
        task_dir,
        tables=tables,
        snapshot_manifest=snapshot_manifest,
        cache_key=cache_key,
        load_payloads=False,
    )


def write_postprobe_stsp_update_metrics(ctx: ExperimentContext, artifact: ProbeStspUpdateArtifact) -> None:
    event_df, magnitude_qc_df, l1_bridge_df, history_by_trial_df = _event_count_metrics_from_artifact(ctx, artifact)
    memory_overlap_df = event_df.loc[:, list(POSTPROBE_L2_MEMORY_OVERLAP_COLUMNS)].copy()
    summary_df = _summary_metrics(event_df)
    by_network_df = _by_network_metrics(summary_df)
    history_composition_df = _history_composition_summary(history_by_trial_df)

    _save_csv(ctx, summary_df.loc[:, list(POSTPROBE_L2_SUMMARY_COLUMNS)], ctx.metrics_dir / "panel_postprobe_l2_stsp_writeback_summary.csv")
    _save_csv(ctx, history_composition_df.loc[:, list(POSTPROBE_L2_HISTORY_COMPOSITION_COLUMNS)], ctx.metrics_dir / "panel_postprobe_l2_reupdate_history_composition.csv")
    _save_csv(ctx, event_df.loc[:, list(POSTPROBE_L2_BY_TRIAL_COLUMNS)], ctx.metrics_dir / "supp_postprobe_l2_writeback_by_trial.csv")
    _save_csv(ctx, history_by_trial_df.loc[:, list(POSTPROBE_L2_HISTORY_BY_TRIAL_COLUMNS)], ctx.metrics_dir / "supp_postprobe_l2_reupdate_history_by_trial.csv")
    _save_csv(ctx, memory_overlap_df, ctx.metrics_dir / "supp_postprobe_l2_writeback_memory_overlap.csv")
    _save_csv(ctx, magnitude_qc_df.loc[:, list(POSTPROBE_MAGNITUDE_QC_COLUMNS)], ctx.metrics_dir / "supp_postprobe_l2_writeback_magnitude_qc.csv")
    _save_csv(ctx, l1_bridge_df.loc[:, list(POSTPROBE_L1_FIRING_BRIDGE_COLUMNS)], ctx.metrics_dir / "supp_postprobe_l1_firing_bridge.csv")
    _save_csv(ctx, by_network_df.loc[:, list(POSTPROBE_L2_BY_NETWORK_COLUMNS)], ctx.metrics_dir / "supp_postprobe_l2_writeback_by_network.csv")
    ctx.completed_modules["postprobe_stsp_update"] = True
    ctx.availability["postprobe_stsp_update_available"] = bool(not summary_df.empty)
    ctx.availability["postprobe_l2_reupdate_history_available"] = bool(not history_composition_df.empty)
    ctx.availability["postprobe_stsp_update_artifact_digest"] = str(artifact.digest)


def condition_manifest() -> pd.DataFrame:
    rows = []
    for condition in probe_stsp_update_conditions():
        perturbation_mode = _l1_stsp_perturbation_mode(condition) if condition in L1_STSP_PERTURBATION_CONDITIONS else "none"
        rows.append(
            {
                "condition": str(condition),
                "condition_label": _fig5d_condition_label(condition),
                "stsp_mode": "static_frozen" if condition == "static_frozen" else "dynamic",
                "perturbation_mode": perturbation_mode,
                "perturbed_layer": PRIMARY_LAYER if condition in L1_STSP_PERTURBATION_CONDITIONS else "none",
                "perturbed_variables": "u_pre;x_pre" if condition in L1_STSP_PERTURBATION_CONDITIONS else "none",
                "branch_role": _branch_role(condition),
            }
        )
    return pd.DataFrame(rows, columns=list(PROBE_STSP_CONDITION_MANIFEST_COLUMNS))


def _build_probe_stsp_update_payload(
    ctx: ExperimentContext,
    bank: LocalSupportCompetitionBank,
    *,
    task_dir: Path,
    trial_hash: str,
    parent_support_bank_digest: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if ctx.net is None or ctx.encoder is None:
        raise RuntimeError("Fig.5 post-probe STSP update requires a loaded real network and encoder.")
    conditions = probe_stsp_update_conditions()
    _record_branch_batch_mode(ctx)
    manifest_rows: list[dict[str, Any]] = []

    batches = _iter_batches(bank.trials, ctx.cfg.batch_size)
    total_batches = int(np.ceil(len(bank.trials) / max(1, int(ctx.cfg.batch_size))))
    for trial_chunk_id, batch in enumerate(
        _progress(batches, total=total_batches, desc="fig5 post-probe STSP snapshots", enabled=ctx.cfg.show_progress)
    ):
        storage_file = _storage_file(SNAPSHOT_LAYER, trial_chunk_id)
        shard_payload: dict[str, np.ndarray] = {}
        boundary, probe_spikes, prior_l1_fire, prior_l2_update = _preprobe_boundary_and_probe_spikes(ctx, batch)
        trial_rows = list(batch.reset_index(drop=True).itertuples(index=False))
        for local_idx, trial in enumerate(trial_rows):
            trial_id = int(trial.trial_id)
            single_boundary = _slice_boundary(boundary, local_idx)
            single_probe = probe_spikes[local_idx : local_idx + 1]
            single_prior_fire = prior_l1_fire[local_idx]
            single_prior_l2_update = prior_l2_update[local_idx]
            for condition in conditions:
                snapshot = _run_branch_snapshot(
                    ctx,
                    single_boundary,
                    single_probe,
                    single_prior_fire,
                    single_prior_l2_update,
                    str(condition),
                )
                for variable_set in SNAPSHOT_VARIABLE_SETS:
                    key = _storage_key(trial_id, str(condition), SNAPSHOT_LAYER, variable_set)
                    arr = np.asarray(snapshot.arrays[variable_set])
                    shard_payload[key] = arr
                    manifest_rows.append(
                        {
                            "snapshot_id": f"trial_{trial_id}__{condition}__{SNAPSHOT_LAYER}",
                            "network_seed": int(ctx.cfg.network_seed),
                            "trial_id": trial_id,
                            "trial_chunk_id": int(trial_chunk_id),
                            "condition": str(condition),
                            "layer": SNAPSHOT_LAYER,
                            "storage_file": storage_file,
                            "storage_key": key,
                            "variable_set": variable_set,
                            "shape": _shape_text(arr.shape),
                            "dtype": str(arr.dtype),
                            "sha256": "",
                            "n_units": int(arr.size),
                            "parent_trial_hash": str(trial_hash),
                            "parent_support_bank_digest": str(parent_support_bank_digest),
                        }
                    )
        write_probe_stsp_update_shard(task_dir, storage_file, shard_payload)
        shard_payload.clear()

    snapshot_manifest = pd.DataFrame(manifest_rows, columns=list(SNAPSHOT_MANIFEST_COLUMNS))
    tables = {
        "trials": bank.trials.reset_index(drop=True).copy(),
        "unit_groups": bank.unit_groups.reset_index(drop=True).copy(),
        "condition_manifest": condition_manifest(),
    }
    return tables, snapshot_manifest


def _preprobe_boundary_and_probe_spikes(ctx: ExperimentContext, batch: pd.DataFrame):
    sample_images = _images_for_ids(ctx.dataset, batch["sample_image_id"].to_numpy()).to(ctx.device)
    probe_images = _images_for_ids(ctx.dataset, batch["probe_image_id"].to_numpy()).to(ctx.device)
    sample_spikes = encode_images(ctx.encoder, sample_images, ctx.cfg.sample_steps)
    probe_spikes = encode_images(ctx.encoder, probe_images, ctx.cfg.probe_steps)
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
    current_time = 0
    prior_fire: torch.Tensor | None = None
    prior_l2_update: torch.Tensor | None = None
    with torch.no_grad():
        for t in range(ctx.cfg.sample_steps):
            s1, _ = ctx.net.layer1.forward_step(sample_spikes[:, t], current_time, training=False, monitor=False, stsp_mode="dynamic")
            spatial = s1.detach().to(torch.bool).any(dim=1)
            prior_fire = spatial if prior_fire is None else torch.logical_or(prior_fire, spatial)
            s1p = ctx.net.pool1(s1.float())
            l2_input_event = s1p.detach().to(torch.bool)
            prior_l2_update = l2_input_event if prior_l2_update is None else torch.logical_or(prior_l2_update, l2_input_event)
            s2, _ = ctx.net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
            s2p = ctx.net.pool2(s2.float())
            ctx.net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
            current_time += 1
        zero = torch.zeros((batch_size, channels, height, width), device=ctx.device)
        for _ in range(ctx.cfg.delay_steps):
            current_time = _step_network_once(ctx.net, zero, current_time, stsp_mode="dynamic")
    prior_np = (
        np.zeros((batch_size, height, width), dtype=bool)
        if prior_fire is None
        else prior_fire.detach().cpu().numpy()
    )
    if prior_l2_update is None:
        layer2_shape = tuple(int(value) for value in ctx.net.layer2.u_pre.shape)
        prior_l2_np = np.zeros(layer2_shape, dtype=bool)
    else:
        prior_l2_np = prior_l2_update.detach().cpu().numpy()
    return snapshot_boundary_state(ctx.net), probe_spikes, prior_np, prior_l2_np


def _run_branch_snapshot(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    prior_l1_fire: np.ndarray,
    prior_l2_update: np.ndarray,
    condition: str,
) -> BranchSnapshot:
    batch_size, _, channels, height, width = probe_spikes.shape
    prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
    _restore_boundary_state(ctx.net, boundary)
    if condition in L1_STSP_PERTURBATION_CONDITIONS:
        _apply_l1_stsp_perturbation(
            ctx.net,
            condition,
            attenuation_factor=float(ctx.cfg.perturbation_attenuation_factor),
        )
    elif condition not in {"dynamic_intact", "static_frozen", "sham_perturbation"}:
        raise RuntimeError(f"Unsupported Fig.5 post-probe STSP update condition: {condition}")

    pre = snapshot_boundary_state(ctx.net)
    stsp_mode = "static_frozen" if condition == "static_frozen" else "dynamic"
    probe_fire: torch.Tensor | None = None
    early_fire: torch.Tensor | None = None
    probe_l2_opportunity: torch.Tensor | None = None
    with torch.no_grad():
        ctx.net.layer3.reset_decision_state()
        ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
        ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
        for t in range(int(probe_spikes.shape[1])):
            s1, _ = ctx.net.layer1.forward_step(probe_spikes[:, t], t, training=False, monitor=False, stsp_mode=stsp_mode)
            spatial = s1.detach().to(torch.bool).any(dim=1)[0]
            probe_fire = spatial if probe_fire is None else torch.logical_or(probe_fire, spatial)
            if t < int(ctx.cfg.early_window_steps):
                early_fire = spatial if early_fire is None else torch.logical_or(early_fire, spatial)
            s1p = ctx.net.pool1(s1.float())
            l2_input_event = s1p.detach().to(torch.bool)[0]
            probe_l2_opportunity = l2_input_event if probe_l2_opportunity is None else torch.logical_or(probe_l2_opportunity, l2_input_event)
            s2, _ = ctx.net.layer2.forward_step(s1p, t, training=False, monitor=False, stsp_mode=stsp_mode)
            s2p = ctx.net.pool2(s2.float())
            ctx.net.layer3.forward_step(s2p, t, training=False, monitor=False, stsp_mode=stsp_mode)
    post = snapshot_boundary_state(ctx.net)
    u_pre = _stsp_map(pre, STSP_STATE_LAYER, "u")
    x_pre = _stsp_map(pre, STSP_STATE_LAYER, "x")
    u_post = _stsp_map(post, STSP_STATE_LAYER, "u")
    x_post = _stsp_map(post, STSP_STATE_LAYER, "x")
    l1_spatial_shape = tuple(int(value) for value in np.asarray(prior_l1_fire).shape[-2:])
    prior_fire_np = _spatial_mask_to_shape(np.asarray(prior_l1_fire, dtype=bool), l1_spatial_shape)
    probe_fire_np = (
        np.zeros(l1_spatial_shape, dtype=bool)
        if probe_fire is None
        else _spatial_mask_to_shape(probe_fire.detach().cpu().numpy(), l1_spatial_shape)
    )
    prior_l2_update_np = _stsp_event_mask_to_shape(np.asarray(prior_l2_update, dtype=bool), u_pre.shape)
    probe_l2_opportunity_np = (
        np.zeros_like(prior_l2_update_np, dtype=bool)
        if probe_l2_opportunity is None
        else _stsp_event_mask_to_shape(probe_l2_opportunity.detach().cpu().numpy(), u_pre.shape)
    )
    probe_l2_event_np = probe_l2_opportunity_np if stsp_mode == "dynamic" else np.zeros_like(probe_l2_opportunity_np, dtype=bool)
    prior_retained_np = _prior_retained_memory_mask(ctx, u_pre, x_pre, layer_name=STSP_STATE_LAYER)
    early_fire_np = (
        np.zeros(l1_spatial_shape, dtype=bool)
        if early_fire is None
        else _spatial_mask_to_shape(early_fire.detach().cpu().numpy(), l1_spatial_shape)
    )
    return BranchSnapshot(
        arrays={
            "u_pre": u_pre.astype(np.float32),
            "x_pre": x_pre.astype(np.float32),
            "G_pre": (u_pre * x_pre).astype(np.float32),
            "u_post": u_post.astype(np.float32),
            "x_post": x_post.astype(np.float32),
            "G_post": (u_post * x_post).astype(np.float32),
            "prior_l2_update_event": prior_l2_update_np.astype(np.float32),
            "probe_l2_update_event": probe_l2_event_np.astype(np.float32),
            "probe_l2_update_opportunity": probe_l2_opportunity_np.astype(np.float32),
            "prior_l2_retained_memory": prior_retained_np.astype(np.float32),
            "prior_l1_fire": prior_fire_np.astype(np.float32),
            "probe_l1_fire": probe_fire_np.astype(np.float32),
            "early_l1_fire": early_fire_np.astype(np.float32),
        }
    )


def _event_count_metrics_from_artifact(ctx: ExperimentContext, artifact: ProbeStspUpdateArtifact) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_groups = artifact.tables["unit_groups"].copy()
    event_rows: list[dict[str, Any]] = []
    magnitude_rows: list[dict[str, Any]] = []
    l1_bridge_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    shard_count = 0
    for _filename, shard_manifest, shard_payload in iter_probe_stsp_update_shards(artifact.root, artifact.snapshot_manifest):
        shard_count += 1
        records = shard_manifest[
            shard_manifest["variable_set"].astype(str).eq("probe_l2_update_event")
        ].loc[:, ["trial_id", "condition", "layer"]].drop_duplicates()
        for rec in records.itertuples(index=False):
            trial_id = int(rec.trial_id)
            condition = str(rec.condition)
            layer = str(rec.layer)
            arrays = _artifact_arrays_from_payload(shard_manifest, shard_payload, trial_id, condition, layer)
            static_arrays = _artifact_arrays_from_payload(shard_manifest, shard_payload, trial_id, MEMORY_CONTROL_CONDITION, layer)
            if condition == "dynamic_intact":
                history_rows.append(_history_by_trial_row(ctx, trial_id, layer, arrays, static_arrays))
            trial_groups = _trial_unit_groups(unit_groups, trial_id)
            for group_name, group_df in trial_groups.groupby("unit_group", sort=False):
                l2_idx = _pooled_l1_group_indices(trial_groups, group_df, arrays["probe_l2_update_event"].shape)
                if l2_idx.size == 0:
                    continue
                l2_event_names = (
                    "prior_l2_update_event",
                    "probe_l2_update_event",
                    "probe_l2_update_opportunity",
                    "prior_l2_retained_memory",
                )
                values = {name: arrays[name].reshape(-1)[l2_idx] for name in l2_event_names}
                static_values = {name: static_arrays[name].reshape(-1)[l2_idx] for name in l2_event_names}

                prior_update = values["prior_l2_update_event"] > 0
                retained = values["prior_l2_retained_memory"] > 0
                probe_update_dynamic = values["probe_l2_update_event"] > 0
                static_opportunity = static_values["probe_l2_update_opportunity"] > 0
                static_actual = static_values["probe_l2_update_event"] > 0
                reupdate_dynamic = prior_update & probe_update_dynamic
                reupdate_static = prior_update & static_opportunity
                memory_enabled_update = probe_update_dynamic & ~static_opportunity
                memory_enabled_reupdate = prior_update & memory_enabled_update

                n_total = int(l2_idx.size)
                n_prior = int(prior_update.sum())
                n_probe_dynamic = int(probe_update_dynamic.sum())
                n_static_opportunity = int(static_opportunity.sum())
                n_static_actual = int(static_actual.sum())
                n_reupdate_dynamic = int(reupdate_dynamic.sum())
                n_reupdate_static = int(reupdate_static.sum())
                n_memory_enabled = int(memory_enabled_update.sum())
                n_memory_enabled_reupdate = int(memory_enabled_reupdate.sum())
                frac_dynamic = _safe_div(n_reupdate_dynamic, n_probe_dynamic)
                frac_static = _safe_div(n_reupdate_static, n_static_opportunity)
                event_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": trial_id,
                        "condition": condition,
                        "layer": SNAPSHOT_LAYER,
                        "unit_group": str(group_name),
                        "memory_control_condition": MEMORY_CONTROL_CONDITION,
                        "n_l2_total_elements": n_total,
                        "n_l2_prior_updated": n_prior,
                        "n_l2_prior_retained_memory": int(retained.sum()),
                        "n_l2_probe_update_dynamic": n_probe_dynamic,
                        "n_l2_probe_update_static_opportunity": n_static_opportunity,
                        "n_l2_probe_update_static_actual": n_static_actual,
                        "n_l2_reupdate_dynamic": n_reupdate_dynamic,
                        "n_l2_reupdate_static_opportunity": n_reupdate_static,
                        "n_memory_enabled_l2_update": n_memory_enabled,
                        "n_memory_enabled_l2_reupdate": n_memory_enabled_reupdate,
                        "frac_prior_among_probe_updates_dynamic": frac_dynamic,
                        "frac_prior_among_probe_updates_static": frac_static,
                        "dynamic_minus_static_frac_prior": _finite_diff(frac_dynamic, frac_static),
                        "frac_reupdate_dynamic_among_prior": _safe_div(n_reupdate_dynamic, n_prior),
                        "frac_reupdate_static_among_prior": _safe_div(n_reupdate_static, n_prior),
                        "frac_memory_enabled_reupdate_among_prior": _safe_div(n_memory_enabled_reupdate, n_prior),
                        "prior_update_base_rate": _safe_div(n_prior, n_total),
                    }
                )

                qc_values = {name: arr.reshape(-1)[l2_idx] for name, arr in arrays.items() if name in {"u_pre", "x_pre", "G_pre", "u_post", "x_post", "G_post"}}
                delta = qc_values["G_post"] - qc_values["G_pre"]
                l1_early = _l1_group_values(arrays, group_df, "early_l1_fire") > 0
                magnitude_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": trial_id,
                        "condition": condition,
                        "layer": SNAPSHOT_LAYER,
                        "unit_group": str(group_name),
                        "n_total_stsp_elements": int(l2_idx.size),
                        "mean_u_pre": _nanmean(qc_values["u_pre"]),
                        "mean_x_pre": _nanmean(qc_values["x_pre"]),
                        "mean_G_pre": _nanmean(qc_values["G_pre"]),
                        "mean_u_post": _nanmean(qc_values["u_post"]),
                        "mean_x_post": _nanmean(qc_values["x_post"]),
                        "mean_G_post": _nanmean(qc_values["G_post"]),
                        "mean_delta_G": _nanmean(delta),
                        "mean_abs_delta_G": _nanmean(np.abs(delta)),
                        "early_fire_fraction": float(l1_early.mean()) if l1_early.size else float("nan"),
                    }
                )
                bridge_row = _l1_firing_bridge_row(ctx, trial_id, condition, group_name, group_df, arrays, static_arrays)
                if bridge_row is not None:
                    l1_bridge_rows.append(bridge_row)
        shard_payload.clear()
    ctx.availability["postprobe_stsp_update_metric_shards_processed"] = int(shard_count)
    ctx.availability["postprobe_stsp_update_metric_processing"] = "trial_chunk_streaming_l2_writeback_counts"
    event_df = pd.DataFrame(event_rows)
    for column in POSTPROBE_L2_BY_TRIAL_COLUMNS:
        if column not in event_df.columns:
            event_df[column] = pd.Series(dtype=float)
    magnitude_df = pd.DataFrame(magnitude_rows)
    for column in POSTPROBE_MAGNITUDE_QC_COLUMNS:
        if column not in magnitude_df.columns:
            magnitude_df[column] = pd.Series(dtype=float)
    l1_bridge_df = pd.DataFrame(l1_bridge_rows)
    for column in POSTPROBE_L1_FIRING_BRIDGE_COLUMNS:
        if column not in l1_bridge_df.columns:
            l1_bridge_df[column] = pd.Series(dtype=float)
    history_df = pd.DataFrame(history_rows)
    for column in POSTPROBE_L2_HISTORY_BY_TRIAL_COLUMNS:
        if column not in history_df.columns:
            history_df[column] = pd.Series(dtype=float)
    return (
        event_df.sort_values(
            ["network_seed", "trial_id", "condition", "layer", "unit_group"],
            kind="mergesort",
        ),
        magnitude_df.sort_values(
            ["network_seed", "trial_id", "condition", "layer", "unit_group"],
            kind="mergesort",
        ),
        l1_bridge_df.sort_values(
            ["network_seed", "trial_id", "condition", "layer", "unit_group"],
            kind="mergesort",
        ),
        history_df.sort_values(
            ["network_seed", "trial_id", "layer"],
            kind="mergesort",
        ),
    )


def _history_by_trial_row(
    ctx: ExperimentContext,
    trial_id: int,
    layer: str,
    arrays: Mapping[str, np.ndarray],
    static_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    prior_update = np.asarray(arrays["prior_l2_update_event"] > 0, dtype=bool).reshape(-1)
    nonprior_update = ~prior_update
    probe_update_dynamic = np.asarray(arrays["probe_l2_update_event"] > 0, dtype=bool).reshape(-1)
    static_opportunity = np.asarray(static_arrays["probe_l2_update_opportunity"] > 0, dtype=bool).reshape(-1)

    n_total = int(prior_update.size)
    n_prior = int(prior_update.sum())
    n_nonprior = int(nonprior_update.sum())
    n_dynamic = int(probe_update_dynamic.sum())
    n_static = int(static_opportunity.sum())
    n_dynamic_prior = int((prior_update & probe_update_dynamic).sum())
    n_dynamic_nonprior = int((nonprior_update & probe_update_dynamic).sum())
    n_static_prior = int((prior_update & static_opportunity).sum())
    n_static_nonprior = int((nonprior_update & static_opportunity).sum())

    dynamic_prior_fraction = _safe_div(n_dynamic_prior, n_dynamic)
    dynamic_nonprior_fraction = _safe_div(n_dynamic_nonprior, n_dynamic)
    static_prior_fraction = _safe_div(n_static_prior, n_static)
    static_nonprior_fraction = _safe_div(n_static_nonprior, n_static)
    dynamic_conditional_delta = _finite_diff(
        _safe_div(n_dynamic_prior, n_prior),
        _safe_div(n_dynamic_nonprior, n_nonprior),
    )
    static_conditional_delta = _finite_diff(
        _safe_div(n_static_prior, n_prior),
        _safe_div(n_static_nonprior, n_nonprior),
    )

    return {
        "network_seed": int(ctx.cfg.network_seed),
        "trial_id": int(trial_id),
        "layer": str(layer),
        "memory_control_condition": MEMORY_CONTROL_CONDITION,
        "n_l2_total_elements": n_total,
        "n_l2_prior_updated": n_prior,
        "n_l2_not_prior_updated": n_nonprior,
        "n_l2_probe_update_dynamic": n_dynamic,
        "n_l2_probe_update_static_opportunity": n_static,
        "n_l2_dynamic_prior_update": n_dynamic_prior,
        "n_l2_dynamic_nonprior_update": n_dynamic_nonprior,
        "n_l2_static_prior_opportunity": n_static_prior,
        "n_l2_static_nonprior_opportunity": n_static_nonprior,
        "dynamic_prior_fraction_among_updates": dynamic_prior_fraction,
        "dynamic_nonprior_fraction_among_updates": dynamic_nonprior_fraction,
        "static_prior_fraction_among_opportunities": static_prior_fraction,
        "static_nonprior_fraction_among_opportunities": static_nonprior_fraction,
        "dynamic_minus_static_prior_fraction": _finite_diff(dynamic_prior_fraction, static_prior_fraction),
        "dynamic_minus_static_nonprior_fraction": _finite_diff(dynamic_nonprior_fraction, static_nonprior_fraction),
        "p_dynamic_update_given_prior": _safe_div(n_dynamic_prior, n_prior),
        "p_dynamic_update_given_nonprior": _safe_div(n_dynamic_nonprior, n_nonprior),
        "p_static_opportunity_given_prior": _safe_div(n_static_prior, n_prior),
        "p_static_opportunity_given_nonprior": _safe_div(n_static_nonprior, n_nonprior),
        "dynamic_conditional_prior_minus_nonprior": dynamic_conditional_delta,
        "static_conditional_prior_minus_nonprior": static_conditional_delta,
        "conditional_difference_in_differences": _finite_diff(dynamic_conditional_delta, static_conditional_delta),
    }


def _history_composition_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if history_df.empty:
        return pd.DataFrame(rows, columns=list(POSTPROBE_L2_HISTORY_COMPOSITION_COLUMNS))

    grouped = history_df.groupby(["network_seed", "layer"], sort=False)
    for keys, part in grouped:
        network_seed, layer = keys
        total = int(pd.to_numeric(part["n_l2_total_elements"], errors="coerce").sum())
        prior = int(pd.to_numeric(part["n_l2_prior_updated"], errors="coerce").sum())
        nonprior = int(pd.to_numeric(part["n_l2_not_prior_updated"], errors="coerce").sum())
        dynamic_total = int(pd.to_numeric(part["n_l2_probe_update_dynamic"], errors="coerce").sum())
        static_total = int(pd.to_numeric(part["n_l2_probe_update_static_opportunity"], errors="coerce").sum())
        dynamic_prior = int(pd.to_numeric(part["n_l2_dynamic_prior_update"], errors="coerce").sum())
        dynamic_nonprior = int(pd.to_numeric(part["n_l2_dynamic_nonprior_update"], errors="coerce").sum())
        static_prior = int(pd.to_numeric(part["n_l2_static_prior_opportunity"], errors="coerce").sum())
        static_nonprior = int(pd.to_numeric(part["n_l2_static_nonprior_opportunity"], errors="coerce").sum())

        dynamic_prior_fraction = _safe_div(dynamic_prior, dynamic_total)
        static_prior_fraction = _safe_div(static_prior, static_total)
        dynamic_conditional_delta = _finite_diff(
            _safe_div(dynamic_prior, prior),
            _safe_div(dynamic_nonprior, nonprior),
        )
        static_conditional_delta = _finite_diff(
            _safe_div(static_prior, prior),
            _safe_div(static_nonprior, nonprior),
        )
        common = {
            "network_seed": int(network_seed),
            "memory_control_condition": MEMORY_CONTROL_CONDITION,
            "layer": str(layer),
            "n_trials": int(part["trial_id"].nunique()),
            "n_l2_total_elements": total,
            "dynamic_minus_static_prior_fraction": _finite_diff(dynamic_prior_fraction, static_prior_fraction),
            "dynamic_conditional_prior_minus_nonprior": dynamic_conditional_delta,
            "static_conditional_prior_minus_nonprior": static_conditional_delta,
            "conditional_difference_in_differences": _finite_diff(dynamic_conditional_delta, static_conditional_delta),
            "denominator_definition": "fraction among Layer2 update sites for the listed condition; static uses update opportunity",
        }
        row_specs = (
            ("dynamic_intact", "Dynamic", 0, "dynamic_intact", "prior_updated", "Prior-updated", 0, prior, dynamic_prior, dynamic_total),
            ("dynamic_intact", "Dynamic", 0, "dynamic_intact", "not_prior_updated", "Not prior-updated", 1, nonprior, dynamic_nonprior, dynamic_total),
            ("static_opportunity", "Static", 1, MEMORY_CONTROL_CONDITION, "prior_updated", "Prior-updated", 0, prior, static_prior, static_total),
            ("static_opportunity", "Static", 1, MEMORY_CONTROL_CONDITION, "not_prior_updated", "Not prior-updated", 1, nonprior, static_nonprior, static_total),
        )
        for condition, label, condition_order, source_condition, history_status, history_label, history_order, n_history, n_updated, n_updated_total in row_specs:
            rows.append(
                {
                    **common,
                    "condition": condition,
                    "condition_label": label,
                    "condition_order": int(condition_order),
                    "source_condition": source_condition,
                    "history_status": history_status,
                    "history_label": history_label,
                    "history_order": int(history_order),
                    "n_l2_history_sites": int(n_history),
                    "n_l2_updated_sites": int(n_updated),
                    "n_l2_total_updated_sites": int(n_updated_total),
                    "fraction_among_updates": _safe_div(n_updated, n_updated_total),
                    "update_probability_given_history": _safe_div(n_updated, n_history),
                }
            )
    out = pd.DataFrame(rows, columns=list(POSTPROBE_L2_HISTORY_COMPOSITION_COLUMNS))
    if not out.empty:
        out = out.sort_values(
            ["network_seed", "layer", "condition_order", "history_order"],
            kind="mergesort",
        )
    return out


def _summary_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = event_df.groupby(["network_seed", "condition", "layer", "unit_group"], sort=False)
    for keys, part in grouped:
        network_seed, condition, layer, unit_group = keys
        total = int(pd.to_numeric(part["n_l2_total_elements"], errors="coerce").sum())
        prior = int(pd.to_numeric(part["n_l2_prior_updated"], errors="coerce").sum())
        probe_dynamic = int(pd.to_numeric(part["n_l2_probe_update_dynamic"], errors="coerce").sum())
        static_opportunity = int(pd.to_numeric(part["n_l2_probe_update_static_opportunity"], errors="coerce").sum())
        reupdate_dynamic = int(pd.to_numeric(part["n_l2_reupdate_dynamic"], errors="coerce").sum())
        reupdate_static = int(pd.to_numeric(part["n_l2_reupdate_static_opportunity"], errors="coerce").sum())
        memory_enabled_reupdate = int(pd.to_numeric(part["n_memory_enabled_l2_reupdate"], errors="coerce").sum())
        frac_dynamic = _safe_div(reupdate_dynamic, probe_dynamic)
        frac_static = _safe_div(reupdate_static, static_opportunity)
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "memory_control_condition": MEMORY_CONTROL_CONDITION,
                "layer": str(layer),
                "unit_group": str(unit_group),
                "n_trials": int(part["trial_id"].nunique()),
                "n_l2_total_elements": total,
                "n_l2_prior_updated": prior,
                "n_l2_prior_retained_memory": int(pd.to_numeric(part["n_l2_prior_retained_memory"], errors="coerce").sum()),
                "n_l2_probe_update_dynamic": probe_dynamic,
                "n_l2_probe_update_static_opportunity": static_opportunity,
                "n_l2_probe_update_static_actual": int(pd.to_numeric(part["n_l2_probe_update_static_actual"], errors="coerce").sum()),
                "n_l2_reupdate_dynamic": reupdate_dynamic,
                "n_l2_reupdate_static_opportunity": reupdate_static,
                "n_memory_enabled_l2_update": int(pd.to_numeric(part["n_memory_enabled_l2_update"], errors="coerce").sum()),
                "n_memory_enabled_l2_reupdate": memory_enabled_reupdate,
                "frac_prior_among_probe_updates_dynamic": frac_dynamic,
                "frac_prior_among_probe_updates_static": frac_static,
                "dynamic_minus_static_frac_prior": _finite_diff(frac_dynamic, frac_static),
                "frac_reupdate_dynamic_among_prior": _safe_div(reupdate_dynamic, prior),
                "frac_reupdate_static_among_prior": _safe_div(reupdate_static, prior),
                "frac_memory_enabled_reupdate_among_prior": _safe_div(memory_enabled_reupdate, prior),
                "prior_update_base_rate": _safe_div(prior, total),
            }
        )
    out = pd.DataFrame(rows, columns=list(POSTPROBE_L2_SUMMARY_COLUMNS))
    if not out.empty:
        out = out.sort_values(["network_seed", "condition", "layer", "unit_group"], kind="mergesort")
    return out


def _by_network_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "n_l2_reupdate_dynamic",
        "n_l2_reupdate_static_opportunity",
        "n_memory_enabled_l2_reupdate",
        "frac_prior_among_probe_updates_dynamic",
        "frac_prior_among_probe_updates_static",
        "dynamic_minus_static_frac_prior",
        "frac_memory_enabled_reupdate_among_prior",
    )
    for row in summary_df.itertuples(index=False):
        for metric in metrics:
            rows.append(
                {
                    "network_seed": int(row.network_seed),
                    "condition": str(row.condition),
                    "layer": str(row.layer),
                    "unit_group": str(row.unit_group),
                    "metric": metric,
                    "value": float(getattr(row, metric)),
                    "n_trials": int(row.n_trials),
                    "n_l2_total_elements": int(row.n_l2_total_elements),
                }
            )
    return pd.DataFrame(rows, columns=list(POSTPROBE_L2_BY_NETWORK_COLUMNS)).sort_values(
        ["network_seed", "condition", "layer", "unit_group", "metric"],
        kind="mergesort",
    )


def _artifact_arrays(artifact: ProbeStspUpdateArtifact, trial_id: int, condition: str, layer: str) -> dict[str, np.ndarray]:
    part = artifact.snapshot_manifest[
        artifact.snapshot_manifest["trial_id"].astype(int).eq(int(trial_id))
        & artifact.snapshot_manifest["condition"].astype(str).eq(str(condition))
        & artifact.snapshot_manifest["layer"].astype(str).eq(str(layer))
    ]
    storage_files = {str(value) for value in part["storage_file"].astype(str).tolist()}
    if len(storage_files) != 1:
        raise RuntimeError(
            f"Fig.5 post-probe STSP artifact expected one shard for trial={trial_id} "
            f"condition={condition} layer={layer}, found={sorted(storage_files)}"
        )
    storage_file = next(iter(storage_files))
    if artifact.payloads:
        return _artifact_arrays_from_payload(part, artifact.payloads[storage_file], trial_id, condition, layer)
    full_shard_manifest = artifact.snapshot_manifest[
        artifact.snapshot_manifest["storage_file"].astype(str).eq(storage_file)
    ]
    for filename, shard_manifest, shard_payload in iter_probe_stsp_update_shards(artifact.root, full_shard_manifest):
        if str(filename) != storage_file:
            continue
        try:
            return _artifact_arrays_from_payload(shard_manifest, shard_payload, trial_id, condition, layer)
        finally:
            shard_payload.clear()
    raise RuntimeError(f"Fig.5 post-probe STSP artifact shard not found for trial={trial_id} condition={condition} layer={layer}")


def _artifact_arrays_from_payload(
    manifest_part: pd.DataFrame,
    payload: Mapping[str, np.ndarray],
    trial_id: int,
    condition: str,
    layer: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    part = manifest_part[
        manifest_part["trial_id"].astype(int).eq(int(trial_id))
        & manifest_part["condition"].astype(str).eq(str(condition))
        & manifest_part["layer"].astype(str).eq(str(layer))
    ]
    for row in part.itertuples(index=False):
        out[str(row.variable_set)] = np.asarray(payload[str(row.storage_key)])
    missing = set(SNAPSHOT_VARIABLE_SETS).difference(out)
    if missing:
        raise RuntimeError(f"Fig.5 post-probe STSP artifact missing variables for trial={trial_id} condition={condition}: {sorted(missing)}")
    return out


def _trial_unit_groups(unit_groups: pd.DataFrame, trial_id: int) -> pd.DataFrame:
    groups = unit_groups[unit_groups["trial_id"].astype(int).eq(int(trial_id))].copy()
    if "layer" not in groups.columns:
        return groups
    layer_groups = groups[groups["layer"].astype(str).eq(L1_BRIDGE_LAYER)].copy()
    return layer_groups if not layer_groups.empty else groups


def _pooled_l1_group_indices(trial_groups: pd.DataFrame, group_df: pd.DataFrame, shape: tuple[int, ...]) -> np.ndarray:
    shape = tuple(int(value) for value in shape)
    if len(shape) < 2:
        return np.array([], dtype=np.int64)
    channels = int(np.prod(shape[:-2])) if len(shape) >= 3 else 1
    height, width = int(shape[-2]), int(shape[-1])
    rows = pd.to_numeric(group_df["row"], errors="coerce").to_numpy(dtype=int)
    cols = pd.to_numeric(group_df["col"], errors="coerce").to_numpy(dtype=int)
    if rows.size == 0 or cols.size == 0:
        return np.array([], dtype=np.int64)
    source_h = int(pd.to_numeric(trial_groups["row"], errors="coerce").max()) + 1
    source_w = int(pd.to_numeric(trial_groups["col"], errors="coerce").max()) + 1
    source_h = max(1, source_h)
    source_w = max(1, source_w)
    pooled_rows = np.floor(rows.astype(float) * float(height) / float(source_h)).astype(np.int64)
    pooled_cols = np.floor(cols.astype(float) * float(width) / float(source_w)).astype(np.int64)
    valid = (pooled_rows >= 0) & (pooled_cols >= 0) & (pooled_rows < height) & (pooled_cols < width)
    if not valid.any():
        return np.array([], dtype=np.int64)
    spatial = np.unique((pooled_rows[valid] * width + pooled_cols[valid]).astype(np.int64))
    offsets = (np.arange(channels, dtype=np.int64) * height * width).reshape(-1, 1)
    return (offsets + spatial.reshape(1, -1)).reshape(-1)


def _l1_group_values(arrays: Mapping[str, np.ndarray], group_df: pd.DataFrame, variable_set: str) -> np.ndarray:
    arr = np.asarray(arrays[variable_set])
    idx = _row_col_indices(group_df, arr.shape)
    if idx.size == 0:
        return np.array([], dtype=arr.dtype)
    return arr.reshape(-1)[idx]


def _l1_firing_bridge_row(
    ctx: ExperimentContext,
    trial_id: int,
    condition: str,
    group_name: Any,
    group_df: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    static_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any] | None:
    prior_fire = _l1_group_values(arrays, group_df, "prior_l1_fire") > 0
    probe_fire_memory = _l1_group_values(arrays, group_df, "probe_l1_fire") > 0
    probe_fire_nomemory = _l1_group_values(static_arrays, group_df, "probe_l1_fire") > 0
    early = _l1_group_values(arrays, group_df, "early_l1_fire") > 0
    if prior_fire.size == 0:
        return None
    memory_enabled = probe_fire_memory & ~probe_fire_nomemory
    memory_suppressed = probe_fire_nomemory & ~probe_fire_memory
    changed = np.logical_xor(probe_fire_memory, probe_fire_nomemory)
    changed_prior = changed & prior_fire
    n_total = int(prior_fire.size)
    n_prior = int(prior_fire.sum())
    n_changed = int(changed.sum())
    n_changed_prior = int(changed_prior.sum())
    frac_prior_among_changed = _safe_div(n_changed_prior, n_changed)
    prior_base_rate = _safe_div(n_prior, n_total)
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "trial_id": int(trial_id),
        "condition": str(condition),
        "layer": L1_BRIDGE_LAYER,
        "unit_group": str(group_name),
        "memory_control_condition": MEMORY_CONTROL_CONDITION,
        "n_total_l1_units": n_total,
        "n_prior_fired": n_prior,
        "n_probe_fire_memory": int(probe_fire_memory.sum()),
        "n_probe_fire_nomemory": int(probe_fire_nomemory.sum()),
        "n_memory_enabled_fire": int(memory_enabled.sum()),
        "n_memory_suppressed_fire": int(memory_suppressed.sum()),
        "n_changed_fire": n_changed,
        "n_changed_prior_fired": n_changed_prior,
        "n_early_fired": int(early.sum()),
        "frac_changed": _safe_div(n_changed, n_total),
        "frac_prior_among_changed": frac_prior_among_changed,
        "frac_changed_among_prior": _safe_div(n_changed_prior, n_prior),
        "prior_fire_base_rate": prior_base_rate,
        "enrichment_vs_prior_base_rate": _finite_diff(frac_prior_among_changed, prior_base_rate),
    }


def _row_col_indices(group_df: pd.DataFrame, shape: tuple[int, ...]) -> np.ndarray:
    shape = tuple(int(value) for value in shape)
    rows = pd.to_numeric(group_df["row"], errors="coerce").to_numpy(dtype=int)
    cols = pd.to_numeric(group_df["col"], errors="coerce").to_numpy(dtype=int)
    if len(shape) >= 3:
        channels, height, width = int(np.prod(shape[:-2])), int(shape[-2]), int(shape[-1])
        valid = (rows >= 0) & (cols >= 0) & (rows < height) & (cols < width)
        spatial = (rows[valid] * width + cols[valid]).astype(np.int64)
        if spatial.size == 0:
            return np.array([], dtype=np.int64)
        offsets = (np.arange(channels, dtype=np.int64) * height * width).reshape(-1, 1)
        return (offsets + spatial.reshape(1, -1)).reshape(-1)
    if len(shape) == 2:
        height, width = shape
        valid = (rows >= 0) & (cols >= 0) & (rows < height) & (cols < width)
        return (rows[valid] * width + cols[valid]).astype(np.int64)
    if len(shape) == 1:
        valid = (rows >= 0) & (rows < shape[0])
        return rows[valid].astype(np.int64)
    return np.array([], dtype=np.int64)


def _storage_key(trial_id: int, condition: str, layer: str, variable_set: str) -> str:
    return f"trial_{int(trial_id)}__{condition}__{layer}__{variable_set}"


def _storage_file(layer: str, trial_chunk_id: int) -> str:
    return f"stsp_update_{str(layer)}_trialchunk{int(trial_chunk_id)}.npz"


def _record_branch_batch_mode(ctx: ExperimentContext) -> None:
    requested = bool(getattr(ctx.cfg, "enable_branch_batch", False))
    fallback_reason = (
        "probe_stsp_update_bank captures pre/post STSP boundary tensors after stateful probe execution; "
        "branch-batched boundary capture has not been validated for this artifact, so conditions execute serially."
    )
    setattr(ctx, "probe_stsp_update_branch_batch_requested", requested)
    setattr(ctx, "probe_stsp_update_branch_batch_effective", False)
    setattr(ctx, "probe_stsp_update_branch_batch_execution", "serial_fallback" if requested else "serial")
    setattr(ctx, "probe_stsp_update_branch_batch_fallback_reason", fallback_reason if requested else "")
    if requested:
        message = f"Fig.5 probe_stsp_update_bank enable_branch_batch requested; using serial_fallback. {fallback_reason}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        if message not in ctx.warnings:
            ctx.warnings.append(message)
        ctx.run_log.append(f"{utc_now()} warning {message}")


def _stsp_map(boundary: Mapping[str, Mapping[str, torch.Tensor]], layer: str, key: str) -> np.ndarray:
    if layer not in boundary or key not in boundary[layer]:
        raise RuntimeError(f"Fig.5 post-probe STSP snapshot missing {layer}/{key}.")
    arr = boundary[layer][key].detach().to(torch.float32).cpu().numpy()
    if arr.ndim == 4:
        arr2 = arr[0]
    elif arr.ndim == 3:
        arr2 = arr
    elif arr.ndim == 2:
        arr2 = arr
    else:
        arr2 = arr.reshape(-1, 1)
    return np.asarray(arr2, dtype=np.float32)


def _spatial_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_shape = tuple(int(value) for value in target_shape)
    if len(target_shape) != 2:
        raise ValueError(f"Layer1 firing masks must be spatial HxW arrays, got target_shape={target_shape}")
    src = np.asarray(mask, dtype=bool)
    if src.shape == target_shape:
        return src
    if src.ndim >= 3 and src.shape[-2:] == target_shape:
        return src.reshape(-1, *target_shape).any(axis=0)
    if src.ndim >= 2:
        return _resize_mask(src, target_shape[0], target_shape[1])
    return np.resize(src.astype(bool), target_shape)


def _stsp_event_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    target_shape = tuple(int(value) for value in target_shape)
    src = np.asarray(mask, dtype=bool)
    if src.shape == target_shape:
        return src
    if src.ndim == len(target_shape) + 1 and src.shape[0] == 1 and src.shape[1:] == target_shape:
        return src[0]
    if src.ndim >= 3 and src.shape[-3:] == target_shape[-3:]:
        return src.reshape(-1, *target_shape[-3:]).any(axis=0)
    if len(target_shape) >= 3 and src.ndim >= 2:
        spatial = _spatial_mask_to_shape(src, tuple(target_shape[-2:]))
        channels = int(np.prod(target_shape[:-2]))
        return np.broadcast_to(spatial.reshape(1, *spatial.shape), (channels, *spatial.shape)).reshape(target_shape)
    return np.resize(src.astype(bool), target_shape)


def _prior_retained_memory_mask(ctx: ExperimentContext, u_pre: np.ndarray, x_pre: np.ndarray, *, layer_name: str) -> np.ndarray:
    layer = getattr(ctx.net, str(layer_name), None)
    baseline_u = _layer_stsp_baseline_u(layer, torch.as_tensor(u_pre)) if layer is not None else float(np.nanmean(u_pre))
    return (np.abs(np.asarray(u_pre, dtype=float) - float(baseline_u)) > 1e-6) | (
        np.abs(np.asarray(x_pre, dtype=float) - 1.0) > 1e-6
    )


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _branch_role(condition: str) -> str:
    if condition == "dynamic_intact":
        return "dynamic_reference"
    if condition == "static_frozen":
        return "static_control"
    if condition in L1_STSP_PERTURBATION_CONDITIONS:
        return "stsp_control"
    if condition == "sham_perturbation":
        return "sham_control"
    return "condition"


def _nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _update_direction(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return "zero"
    return "positive" if value > 0 else "negative"


def _safe_div(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    return float(numerator) / denom if np.isfinite(denom) and abs(denom) > 0.0 else float("nan")


def _finite_diff(value: float, baseline: float) -> float:
    return float(value - baseline) if np.isfinite(value) and np.isfinite(baseline) else float("nan")


__all__ = [
    "SNAPSHOT_LAYER",
    "SNAPSHOT_VARIABLE_SETS",
    "build_and_save_probe_stsp_update_artifact",
    "condition_manifest",
    "probe_stsp_update_conditions",
    "probe_stsp_update_layers",
    "probe_stsp_update_variable_sets",
    "unit_group_digest",
    "write_postprobe_stsp_update_metrics",
]
