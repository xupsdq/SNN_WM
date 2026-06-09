from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.common.dataset import encode_images
from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as legacy
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import (
    _run_ping_multi_boundary_batch,
    _run_ping_from_boundary,
    _weak_probe_memory_specs_for_target,
    concat_named_boundaries,
    run_probe_readout_from_boundary,
)
from src.experiments.paper_figures.fig3.subexperiments.weak_probe import _make_weak_probe_spikes_encoded_dropout
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank


def run_weak_cue_access(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    access_jobs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    weak_jobs = access_jobs[access_jobs["job_type"].astype(str).eq("weak_cue")].copy()
    if bool(ctx.cfg.enable_condition_batch) and float(ctx.cfg.weak_probe_noise) == 0.0:
        raw_rows = _run_weak_cue_access_batched(ctx, bank, weak_jobs)
    else:
        raw_rows = _run_weak_cue_access_serial(ctx, bank, weak_jobs)
    raw = pd.DataFrame(raw_rows)
    metrics = _weak_cue_metrics(raw)
    gain = _weak_cue_gain(metrics)
    boundary = _functional_boundary(gain, threshold=0.0)
    legacy._save_csv(ctx, raw, ctx.raw_dir / "panel_d_weak_cue_item_readout.csv")
    legacy._save_csv(ctx, metrics, ctx.metrics_dir / "panel_d_weak_cue_item_metrics.csv")
    legacy._save_csv(ctx, gain, ctx.metrics_dir / "panel_d_item_functional_gain.csv")
    legacy._save_csv(ctx, boundary, ctx.metrics_dir / "panel_d_functional_boundary_metrics.csv")
    ctx.completed_modules["weak_cue_access"] = True
    return {
        "weak_cue_item_readout": raw,
        "weak_cue_item_metrics": metrics,
        "item_functional_gain": gain,
        "functional_boundary_metrics": boundary,
    }


def _run_weak_cue_access_serial(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    weak_jobs: pd.DataFrame,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for _, job in weak_jobs.iterrows():
        seq_id = int(job["sequence_id"])
        seq_len = int(job["seq_len"])
        condition_id = str(job["condition_id"])
        delay_ms = int(job["delay_ms"])
        target_position = int(job["target_position"])
        target_label = int(job["target_label"])
        image = ctx.dataset[int(job["target_image_id"])][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
        full_spikes = encode_images(ctx.encoder, image, ctx.cfg.weak_probe_steps).to(ctx.device)
        memory_specs = _weak_probe_memory_specs_for_target(ctx, bank, seq_id, target_position, condition_id=condition_id, delay_ms=delay_ms)
        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
            full_spikes,
            float(job["keep_prob"]),
            seed=int(job["mask_seed"]),
            same_mask_count=len(memory_specs),
            use_same_mask_across_states=True,
            device=ctx.device,
        )
        boundary = concat_named_boundaries([spec[2] for spec in memory_specs], device=ctx.device)
        predictions, fire_times = run_probe_readout_from_boundary(
            ctx,
            boundary,
            weak_spikes,
            probe_scale=float(ctx.cfg.weak_probe_scale),
            probe_noise=float(ctx.cfg.weak_probe_noise),
            seed=int(job["mask_seed"]) + 17,
        )
        for idx, (state_condition, memory_condition, _) in enumerate(memory_specs):
            pred = int(predictions[idx])
            fire = int(fire_times[idx])
            raw_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "job_id": int(job["job_id"]),
                    "condition_id": condition_id,
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "delay_ms": delay_ms,
                    "target_position": target_position,
                    "target_image_id": int(job["target_image_id"]),
                    "target_label": target_label,
                    "keep_prob": float(job["keep_prob"]),
                    "repeat_id": int(job["repeat_id"]),
                    "mask_seed": int(job["mask_seed"]),
                    "state_condition": str(state_condition),
                    "memory_condition": str(memory_condition),
                    "prediction": pred,
                    "pred_is_target": int(pred == target_label),
                    "silent": int(pred < 0),
                    "first_fire_time_ms": fire,
                    "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
                    "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
                }
            )
    return raw_rows


def _run_weak_cue_access_batched(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    weak_jobs: pd.DataFrame,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    pending_boundaries: list[Any] = []
    pending_spikes: list[torch.Tensor] = []
    pending_rows: list[dict[str, Any]] = []
    encoded_cache: dict[int, torch.Tensor] = {}
    batch_size = max(1, int(ctx.cfg.batch_size))
    for _, job in weak_jobs.iterrows():
        seq_id = int(job["sequence_id"])
        seq_len = int(job["seq_len"])
        condition_id = str(job["condition_id"])
        delay_ms = int(job["delay_ms"])
        target_position = int(job["target_position"])
        target_label = int(job["target_label"])
        full_spikes = _encoded_job_spikes(ctx, int(job["target_image_id"]), encoded_cache)
        memory_specs = _weak_probe_memory_specs_for_target(ctx, bank, seq_id, target_position, condition_id=condition_id, delay_ms=delay_ms)
        weak_spikes, mask_info = _make_weak_probe_spikes_encoded_dropout(
            full_spikes,
            float(job["keep_prob"]),
            seed=int(job["mask_seed"]),
            same_mask_count=len(memory_specs),
            use_same_mask_across_states=True,
            device=ctx.device,
        )
        for idx, (state_condition, memory_condition, boundary) in enumerate(memory_specs):
            pending_boundaries.append(boundary)
            pending_spikes.append(weak_spikes[idx : idx + 1])
            pending_rows.append(
                _weak_cue_base_row(
                    ctx,
                    job,
                    seq_id=seq_id,
                    seq_len=seq_len,
                    condition_id=condition_id,
                    delay_ms=delay_ms,
                    target_position=target_position,
                    target_label=target_label,
                    state_condition=str(state_condition),
                    memory_condition=str(memory_condition),
                    mask_info=mask_info,
                )
            )
        if len(pending_rows) >= batch_size:
            raw_rows.extend(_flush_weak_cue_access_batch(ctx, pending_boundaries, pending_spikes, pending_rows))
            pending_boundaries = []
            pending_spikes = []
            pending_rows = []
    if pending_rows:
        raw_rows.extend(_flush_weak_cue_access_batch(ctx, pending_boundaries, pending_spikes, pending_rows))
    return raw_rows


def _flush_weak_cue_access_batch(
    ctx: ExperimentContext,
    boundaries: list[Any],
    spikes: list[torch.Tensor],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    batched_boundary = concat_named_boundaries(boundaries, device=ctx.device)
    batched_spikes = torch.cat(spikes, dim=0).contiguous()
    predictions, fire_times = run_probe_readout_from_boundary(
        ctx,
        batched_boundary,
        batched_spikes,
        probe_scale=float(ctx.cfg.weak_probe_scale),
        probe_noise=0.0,
        seed=0,
    )
    out: list[dict[str, Any]] = []
    for row, pred_value, fire_value in zip(rows, predictions, fire_times):
        pred = int(pred_value)
        enriched = dict(row)
        enriched.update(
            {
                "prediction": pred,
                "pred_is_target": int(pred == int(row["target_label"])),
                "silent": int(pred < 0),
                "first_fire_time_ms": int(fire_value),
            }
        )
        out.append({column: enriched[column] for column in _WEAK_CUE_RAW_COLUMNS})
    return out


def _weak_cue_base_row(
    ctx: ExperimentContext,
    job: pd.Series,
    *,
    seq_id: int,
    seq_len: int,
    condition_id: str,
    delay_ms: int,
    target_position: int,
    target_label: int,
    state_condition: str,
    memory_condition: str,
    mask_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "job_id": int(job["job_id"]),
        "condition_id": condition_id,
        "sequence_id": seq_id,
        "seq_len": seq_len,
        "delay_ms": delay_ms,
        "target_position": target_position,
        "target_image_id": int(job["target_image_id"]),
        "target_label": target_label,
        "keep_prob": float(job["keep_prob"]),
        "repeat_id": int(job["repeat_id"]),
        "mask_seed": int(job["mask_seed"]),
        "state_condition": state_condition,
        "memory_condition": memory_condition,
        "realized_keep_fraction": float(mask_info["realized_keep_fraction"]),
        "weak_spike_fraction": float(mask_info["weak_spike_fraction"]),
    }


def _encoded_job_spikes(ctx: ExperimentContext, image_id: int, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    image_id = int(image_id)
    if image_id not in cache:
        image = ctx.dataset[image_id][0].detach().to(ctx.device, dtype=torch.float32).unsqueeze(0)
        cache[image_id] = encode_images(ctx.encoder, image, ctx.cfg.weak_probe_steps).to(ctx.device)
    return cache[image_id]


_WEAK_CUE_RAW_COLUMNS = [
    "network_seed",
    "job_id",
    "condition_id",
    "sequence_id",
    "seq_len",
    "delay_ms",
    "target_position",
    "target_image_id",
    "target_label",
    "keep_prob",
    "repeat_id",
    "mask_seed",
    "state_condition",
    "memory_condition",
    "prediction",
    "pred_is_target",
    "silent",
    "first_fire_time_ms",
    "realized_keep_fraction",
    "weak_spike_fraction",
]


def run_neutral_ping_access(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    access_jobs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ping_jobs = access_jobs[access_jobs["job_type"].astype(str).eq("neutral_ping")].copy()
    if bool(ctx.cfg.enable_condition_batch):
        raw_rows = _run_neutral_ping_access_batched(ctx, bank, ping_jobs)
    else:
        raw_rows = _run_neutral_ping_access_serial(ctx, bank, ping_jobs)
    raw = pd.DataFrame(raw_rows)
    position = _neutral_ping_position_distribution(raw)
    summary = _neutral_ping_summary(raw)
    legacy._save_csv(ctx, raw, ctx.raw_dir / "panel_c_neutral_ping_access_readout.csv")
    legacy._save_csv(ctx, position, ctx.metrics_dir / "panel_c_neutral_ping_position_distribution.csv")
    legacy._save_csv(ctx, summary, ctx.metrics_dir / "panel_c_neutral_ping_access_summary.csv")
    ctx.completed_modules["neutral_ping_access"] = True
    return {
        "neutral_ping_access_readout": raw,
        "neutral_ping_position_distribution": position,
        "neutral_ping_access_summary": summary,
    }


def _run_neutral_ping_access_serial(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    ping_jobs: pd.DataFrame,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for _, job in ping_jobs.iterrows():
        seq_id = int(job["sequence_id"])
        seq_len = int(job["seq_len"])
        condition_id = str(job["condition_id"])
        delay_ms = int(job["delay_ms"])
        labels = _labels_for_sequence(bank, seq_id, condition_id=condition_id, delay_ms=delay_ms)
        for state_condition in ctx.cfg.ping_main_state_conditions:
            state_key = "S_final" if str(state_condition) == "S_final" else "S0"
            pred, fire, ping_energy, ping_spike_count, restore_info = _run_ping_from_boundary(ctx, bank.boundary_for(seq_id, state_key, condition_id=condition_id, delay_ms=delay_ms))
            position = labels.index(pred) + 1 if pred in labels else -1
            raw_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "job_id": int(job["job_id"]),
                    "condition_id": condition_id,
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "delay_ms": delay_ms,
                    "ping_repeat": int(job["ping_repeat"]),
                    "state_condition": str(state_condition),
                    "memory_condition": "sequence_state" if state_key == "S_final" else "cue_only",
                    "predicted_label": int(pred),
                    "predicted_position": int(position),
                    "serial_bin": f"pos_{position}" if 1 <= int(position) <= seq_len else ("silent" if pred < 0 else "other"),
                    "pred_is_seen_item": int(position > 0),
                    "silent": int(pred < 0),
                    "first_fire_time_ms": int(fire),
                    "ping_energy": float(ping_energy),
                    "ping_spike_count": float(ping_spike_count),
                    "restore_ok": int(restore_info.get("restore_ok", 1)),
                }
            )
    return raw_rows


def _run_neutral_ping_access_batched(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    ping_jobs: pd.DataFrame,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    batch_size = max(1, int(ctx.cfg.batch_size))
    for _, job in ping_jobs.iterrows():
        seq_id = int(job["sequence_id"])
        seq_len = int(job["seq_len"])
        condition_id = str(job["condition_id"])
        delay_ms = int(job["delay_ms"])
        labels = _labels_for_sequence(bank, seq_id, condition_id=condition_id, delay_ms=delay_ms)
        for state_condition in ctx.cfg.ping_main_state_conditions:
            state_key = "S_final" if str(state_condition) == "S_final" else "S0"
            pending.append(
                {
                    "job": job,
                    "boundary": bank.boundary_for(seq_id, state_key, condition_id=condition_id, delay_ms=delay_ms),
                    "seq_id": seq_id,
                    "seq_len": seq_len,
                    "condition_id": condition_id,
                    "delay_ms": delay_ms,
                    "labels": labels,
                    "state_condition": str(state_condition),
                    "memory_condition": "sequence_state" if state_key == "S_final" else "cue_only",
                }
            )
        if len(pending) >= batch_size:
            raw_rows.extend(_flush_neutral_ping_batch(ctx, pending))
            pending = []
    if pending:
        raw_rows.extend(_flush_neutral_ping_batch(ctx, pending))
    return raw_rows


def _flush_neutral_ping_batch(ctx: ExperimentContext, pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = _run_ping_multi_boundary_batch(ctx, [item["boundary"] for item in pending])
    rows: list[dict[str, Any]] = []
    for item, (pred, fire, ping_energy, ping_spike_count, restore_info) in zip(pending, results):
        job = item["job"]
        labels = item["labels"]
        seq_len = int(item["seq_len"])
        position = labels.index(pred) + 1 if pred in labels else -1
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "job_id": int(job["job_id"]),
                "condition_id": str(item["condition_id"]),
                "sequence_id": int(item["seq_id"]),
                "seq_len": seq_len,
                "delay_ms": int(item["delay_ms"]),
                "ping_repeat": int(job["ping_repeat"]),
                "state_condition": str(item["state_condition"]),
                "memory_condition": str(item["memory_condition"]),
                "predicted_label": int(pred),
                "predicted_position": int(position),
                "serial_bin": f"pos_{position}" if 1 <= int(position) <= seq_len else ("silent" if pred < 0 else "other"),
                "pred_is_seen_item": int(position > 0),
                "silent": int(pred < 0),
                "first_fire_time_ms": int(fire),
                "ping_energy": float(ping_energy),
                "ping_spike_count": float(ping_spike_count),
                "restore_ok": int(restore_info.get("restore_ok", 1)),
            }
        )
    return rows


def _weak_cue_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "condition_id",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "target_position",
        "memory_condition",
        "keep_prob",
        "P_target",
        "P_silent",
        "n_trials",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "target_position", "memory_condition", "keep_prob"]
    for keys, part in raw.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        denom = max(1, len(part))
        row["P_target"] = float(part["pred_is_target"].sum() / denom)
        row["P_silent"] = float(part["silent"].sum() / denom)
        row["n_trials"] = int(len(part))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _weak_cue_gain(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "condition_id",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "target_position",
        "keep_prob",
        "P_target_sequence_state",
        "P_target_single_item_memory",
        "P_target_cue_only",
        "G_i",
        "U_i",
        "G_i_norm",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "target_position", "keep_prob"]
    for keys, part in metrics.groupby(group_cols, sort=True):
        values = {str(row["memory_condition"]): float(row["P_target"]) for _, row in part.iterrows()}
        seq = values.get("sequence_state", float("nan"))
        single = values.get("single_item_memory", float("nan"))
        cue = values.get("cue_only", float("nan"))
        g_i = seq - cue if np.isfinite(seq) and np.isfinite(cue) else float("nan")
        u_i = single - cue if np.isfinite(single) and np.isfinite(cue) else float("nan")
        denom = u_i if np.isfinite(u_i) and abs(u_i) > 1e-9 else float("nan")
        if np.isfinite(g_i) and np.isfinite(denom):
            g_norm = float(np.clip(g_i / denom, 0.0, 1.0))
        elif np.isfinite(g_i) and abs(g_i) <= 1e-9:
            g_norm = 0.0
        else:
            g_norm = float("nan")
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "P_target_sequence_state": seq,
                "P_target_single_item_memory": single,
                "P_target_cue_only": cue,
                "G_i": g_i,
                "U_i": u_i,
                "G_i_norm": g_norm,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _functional_boundary(gain: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    columns = [
        "network_seed",
        "condition_id",
        "sequence_id",
        "seq_len",
        "delay_ms",
        "keep_prob",
        "accessible_item_count",
        "singleton_access_count",
        "sequence_access_count",
        "rescued_count",
        "singleton_access_fraction",
        "sequence_access_fraction",
        "rescued_fraction",
        "functional_retention_index",
        "mean_G_i",
        "mean_U_i",
        "mean_G_i_norm",
    ]
    if gain.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "keep_prob"]
    for keys, part in gain.groupby(group_cols, sort=True):
        g_i = pd.to_numeric(part["G_i"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        u_i = pd.to_numeric(part["U_i"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pos = np.maximum(g_i, 0.0)
        seq_len = int(keys[3])
        denom = float(seq_len) if seq_len > 0 else float(len(part))
        if pos.sum() > 1e-12:
            mass = pos / pos.sum()
            n_eff = float(1.0 / np.sum(mass * mass))
            retention = float((n_eff - 1.0) / max(seq_len - 1, 1))
        else:
            retention = 0.0
        singleton_access = u_i > float(threshold)
        sequence_access = g_i > float(threshold)
        rescued = (u_i <= float(threshold)) & (g_i > float(threshold))
        singleton_count = int(np.sum(singleton_access))
        sequence_count = int(np.sum(sequence_access))
        rescued_count = int(np.sum(rescued))
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "accessible_item_count": sequence_count,
                "singleton_access_count": singleton_count,
                "sequence_access_count": sequence_count,
                "rescued_count": rescued_count,
                "singleton_access_fraction": float(singleton_count / denom) if denom > 0 else float("nan"),
                "sequence_access_fraction": float(sequence_count / denom) if denom > 0 else float("nan"),
                "rescued_fraction": float(rescued_count / denom) if denom > 0 else float("nan"),
                "functional_retention_index": retention,
                "mean_G_i": float(np.nanmean(g_i)) if g_i.size else float("nan"),
                "mean_U_i": float(np.nanmean(u_i)) if u_i.size else float("nan"),
                "mean_G_i_norm": float(pd.to_numeric(part["G_i_norm"], errors="coerce").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _neutral_ping_position_distribution(raw: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "state_condition", "serial_bin", "serial_position", "readout_mass", "n_trials"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "state_condition"]
    for keys, part in raw.groupby(group_cols, sort=True):
        seq_len = int(keys[3])
        bins = [f"pos_{idx}" for idx in range(1, seq_len + 1)] + ["other", "silent"]
        for serial_bin in bins:
            pos = int(serial_bin.split("_")[1]) if serial_bin.startswith("pos_") else -1
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "serial_bin": serial_bin,
                    "serial_position": pos,
                    "readout_mass": float((part["serial_bin"].astype(str) == serial_bin).mean()),
                    "n_trials": int(len(part)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _neutral_ping_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "state_condition"]
    for keys, part in raw.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        positions = pd.to_numeric(part["predicted_position"], errors="coerce")
        seq_len = int(row["seq_len"])
        row.update(
            {
                "P_seen_item": float(part["pred_is_seen_item"].mean()),
                "P_silent": float(part["silent"].mean()),
                "latest_item_mass": float((positions == seq_len).mean()),
                "earlier_item_mass": float(((positions > 0) & (positions < seq_len)).mean()),
                "n_trials": int(len(part)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _labels_for_sequence(bank: MultiItemSequenceLandscapeBank, seq_id: int, condition_id: str | None = None, delay_ms: int | None = None) -> list[int]:
    try:
        row = bank.sequence_meta_row(int(seq_id), condition_id=condition_id, delay_ms=delay_ms)
    except KeyError:
        return []
    return [int(v) for v in str(row.get("ordered_item_labels", "")).split(";") if str(v) != ""]


__all__ = ["run_neutral_ping_access", "run_weak_cue_access"]
