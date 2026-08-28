from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry as _save_csv
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import (
    _layer_input_shapes_for_batch,
    _progress,
    _region_ping_amp_sweep_latency,
    _region_ping_amp_sweep_summary,
    _region_ping_contrast,
    _region_ping_current_matching,
    _region_ping_current_matching_status,
    _region_ping_position_distribution,
    _region_ping_summary,
    _step_network_once,
    concat_named_boundaries,
    restore_condition_state_for_functional_readout,
)
from src.experiments.paper_figures.fig3.subexperiments.structural_weak_cue_supplement import _main_sequence_meta
from src.experiments.paper_figures.fig3.types import ExperimentContext, MultiItemSequenceLandscapeBank

def run_region_gated_ping_readout(ctx: ExperimentContext, bank: MultiItemSequenceLandscapeBank) -> None:
    if not bank.landscapes:
        raise RuntimeError("run_region_ping requires state-bank landscapes; run the state-bank path first.")
    raw = _region_gated_ping_trial_rows(
        ctx,
        bank,
        amp_values=(float(ctx.cfg.ping_amp),),
        include_s0=bool(ctx.cfg.run_region_ping_s0_control),
        repeats=int(ctx.cfg.region_ping_repeats),
        desc="fig3 region ping sequences",
    )
    _save_csv(ctx, raw, ctx.raw_dir / "panel_f_region_ping_trial_readout.csv")
    position = _region_ping_position_distribution(ctx.cfg.network_seed, raw)
    summary = _region_ping_summary(ctx.cfg.network_seed, raw)
    contrast = _region_ping_contrast(ctx.cfg.network_seed, raw)
    matching = _region_ping_current_matching(ctx.cfg.network_seed, raw)
    _save_csv(ctx, position, ctx.metrics_dir / "panel_f_region_ping_position_distribution.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_f_region_ping_summary.csv")
    _save_csv(ctx, contrast, ctx.metrics_dir / "panel_f_region_ping_contrast.csv")
    _save_csv(ctx, matching, ctx.metrics_dir / "panel_f_region_ping_current_matching.csv")
    if _region_ping_current_matching_status(matching) != "passed":
        ctx.warnings.append("Fig.3F region ping current matching failed; see panel_f_region_ping_current_matching.csv.")
    if bool(ctx.cfg.run_region_ping_amp_sweep):
        sweep_raw = _region_gated_ping_trial_rows(
            ctx,
            bank,
            amp_values=tuple(float(v) for v in ctx.cfg.region_ping_amp_sweep),
            include_s0=bool(ctx.cfg.run_region_ping_s0_control),
            repeats=int(ctx.cfg.region_ping_repeats),
            desc="fig3 region ping amp sweep",
        )
        _save_csv(ctx, sweep_raw, ctx.raw_dir / "supp_region_ping_amp_sweep_trial_readout.csv")
        _save_csv(ctx, _region_ping_amp_sweep_summary(ctx.cfg.network_seed, sweep_raw), ctx.metrics_dir / "supp_region_ping_amp_sweep_summary.csv")
        _save_csv(ctx, _region_ping_amp_sweep_latency(ctx.cfg.network_seed, sweep_raw), ctx.metrics_dir / "supp_region_ping_amp_sweep_latency.csv")
    ctx.completed_modules["region_ping"] = True

def _region_gated_ping_trial_rows(
    ctx: ExperimentContext,
    bank: MultiItemSequenceLandscapeBank,
    *,
    amp_values: Sequence[float],
    include_s0: bool,
    repeats: int,
    desc: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 1301)
    main_meta = _main_sequence_meta(ctx, bank)
    states = [("S_final", "sequence_state")]
    if include_s0:
        states.append(("S0", "no_memory"))
    for _, meta in _progress(main_meta.iterrows(), total=len(main_meta), desc=desc, enabled=ctx.cfg.show_progress):
        seq_id = int(meta["sequence_id"])
        seq_len = int(meta["seq_len"])
        labels = [int(v) for v in str(meta["ordered_item_labels"]).split(";")]
        landscape = bank.landscapes.get(seq_id)
        if landscape is None:
            raise RuntimeError(f"Missing landscape for sequence_id={seq_id}; region ping cannot run.")
        support_metric = str(ctx.cfg.region_ping_support_metric)
        if support_metric not in {"gain_ratio_map", "delta_gain_map", "G_final"}:
            raise ValueError(f"Unsupported region_ping_support_metric={support_metric!r}; expected gain_ratio_map, delta_gain_map, or G_final.")
        support_map = np.asarray(landscape[support_metric], dtype=np.float32)
        if np.any(~np.isfinite(support_map)):
            ctx.warnings.append(f"Region ping excluded non-finite support values for sequence_id={seq_id}.")
        for ping_repeat in range(int(repeats)):
            ping_seed = int(ctx.cfg.network_seed) * 1000000 + seq_id * 1000 + ping_repeat
            masks = _make_region_ping_masks(
                support_map,
                float(ctx.cfg.region_ping_q),
                np.random.default_rng(ping_seed),
                ctx.cfg.region_ping_conditions,
            )
            for ping_amp in amp_values:
                jobs: list[dict[str, Any]] = []
                for state_condition, memory_condition in states:
                    for region_condition in ctx.cfg.region_ping_conditions:
                        jobs.append(
                            {
                                "state_condition": state_condition,
                                "memory_condition": memory_condition,
                                "region_condition": str(region_condition),
                                "boundary": bank.boundaries[seq_id][state_condition],
                                "mask": masks[str(region_condition)],
                            }
                        )
                if bool(ctx.cfg.enable_condition_batch) and len(jobs) > 1:
                    results = _run_masked_ping_batch_from_boundaries(
                        ctx,
                        [job["boundary"] for job in jobs],
                        [job["mask"] for job in jobs],
                        float(ping_amp),
                        int(ctx.cfg.ping_steps),
                    )
                else:
                    results = [
                        _run_masked_ping_from_boundary(
                            ctx,
                            job["boundary"],
                            job["mask"],
                            float(ping_amp),
                            int(ctx.cfg.ping_steps),
                        )
                        for job in jobs
                    ]
                for job, result in zip(jobs, results):
                    pred, fire_ms, total_current, active_units, restore_info = result
                    region_condition = str(job["region_condition"])
                    state_condition = str(job["state_condition"])
                    memory_condition = str(job["memory_condition"])
                    mask = job["mask"]
                    silent = int(pred < 0)
                    positions = [idx + 1 for idx, label in enumerate(labels) if int(label) == int(pred)]
                    ambiguous = int(len(positions) > 1)
                    predicted_position = int(positions[0]) if positions else -1
                    serial_bin = "silent" if silent else (f"pos_{predicted_position}" if predicted_position > 0 else "other")
                    rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "ordered_item_labels": ";".join(str(v) for v in labels),
                            "region_condition": region_condition,
                            "support_metric": support_metric,
                            "region_q": float(ctx.cfg.region_ping_q),
                            "region_unit_count": int(np.asarray(mask, dtype=bool).sum()),
                            "ping_repeat": int(ping_repeat),
                            "ping_seed": int(ping_seed),
                            "ping_amp": float(ping_amp),
                            "ping_ms": int(ctx.cfg.ping_ms),
                            "active_unit_count": float(active_units),
                            "total_ping_current": float(total_current),
                            "state_condition": state_condition,
                            "memory_condition": memory_condition,
                            "predicted_label": int(pred),
                            "predicted_position": int(predicted_position),
                            "serial_bin": serial_bin,
                            "pred_is_seen_item": int((not silent) and predicted_position > 0),
                            "pred_is_latest_item": int((not silent) and predicted_position == seq_len),
                            "pred_is_recent_item": int((not silent) and predicted_position >= seq_len - 2 and predicted_position < seq_len),
                            "pred_is_earlier_item": int((not silent) and predicted_position > 0 and predicted_position < seq_len - 2),
                            "pred_is_unseen": int((not silent) and predicted_position < 0),
                            "silent": silent,
                            "label_is_ambiguous": ambiguous,
                            "first_fire_time_ms": float(fire_ms),
                            "restore_mode": str(ctx.cfg.functional_restore_mode),
                            "stsp_only_restore": int(str(ctx.cfg.functional_restore_mode) == "stsp_only"),
                            "restore_ok": int(restore_info.get("restore_ok", 0)),
                        }
                    )
    return pd.DataFrame(rows)

def _make_region_ping_masks(
    support_map: np.ndarray,
    region_q: float,
    rng: np.random.Generator,
    conditions: Sequence[str],
) -> dict[str, np.ndarray]:
    support = np.asarray(support_map, dtype=np.float64)
    if support.ndim < 2:
        raise ValueError(f"region ping support_map must be at least 2D, got shape={support.shape}")
    valid_flat = np.flatnonzero(np.isfinite(support).reshape(-1))
    if valid_flat.size == 0:
        raise ValueError("region ping support_map has no finite units.")
    q = float(np.clip(float(region_q), 0.0, 1.0))
    count = max(1, int(round(q * valid_flat.size)))
    support_flat = support.reshape(-1)
    order = valid_flat[np.argsort(support_flat[valid_flat], kind="mergesort")]
    valley_idx = order[:count]
    peak_idx = order[-count:]
    random_idx = rng.choice(valid_flat, size=count, replace=valid_flat.size < count)
    index_by_condition = {"peak": peak_idx, "valley": valley_idx, "random": random_idx}
    masks: dict[str, np.ndarray] = {}
    for condition in conditions:
        cond = str(condition)
        if cond not in index_by_condition:
            raise ValueError(f"Unsupported region ping condition={cond!r}; expected peak, valley, or random.")
        out = np.zeros(support.size, dtype=bool)
        out[index_by_condition[cond]] = True
        masks[cond] = out.reshape(support.shape)
    counts = {cond: int(mask.sum()) for cond, mask in masks.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Region ping masks are not count-matched: {counts}")
    return masks

def _run_masked_ping_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    region_mask: np.ndarray,
    ping_amp: float,
    ping_steps: int,
) -> tuple[int, int, float, float, dict[str, object]]:
    batch_size = 1
    restore_info = restore_condition_state_for_functional_readout(ctx, boundary, batch_size=batch_size)
    input_shape = _layer_input_shapes_for_batch(boundary, batch_size)["layer1"]
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    mask_arr = np.asarray(region_mask, dtype=np.float32)
    if tuple(mask_arr.shape) == tuple(input_shape[1:]):
        mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32, device=ctx.device).unsqueeze(0)
    elif len(input_shape) == 4 and tuple(mask_arr.shape) == tuple(input_shape[2:]):
        mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32, device=ctx.device).unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.expand(batch_size, input_shape[1], input_shape[2], input_shape[3]).contiguous()
    else:
        raise ValueError(f"region_mask shape {mask_arr.shape} is incompatible with layer1 input shape {input_shape}")
    ping_drive = torch.as_tensor(float(ping_amp), dtype=torch.float32, device=ctx.device) * mask_tensor
    active_unit_count = float((ping_drive > 0).detach().to(torch.float32).sum().item())
    total_ping_current = float(ping_amp) * active_unit_count * int(ping_steps)
    with torch.no_grad():
        for t_idx in range(int(ping_steps)):
            _step_network_once(ctx.net, zero, int(t_idx), ping_drive=ping_drive)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    fire_step = int(fire[0].item())
    fire_ms = int(fire_step * ctx.cfg.dt / ms) if fire_step >= 0 else -1
    return int(pred[0].item()), fire_ms, total_ping_current, active_unit_count, restore_info

def _run_masked_ping_batch_from_boundaries(
    ctx: ExperimentContext,
    boundaries: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    region_masks: Sequence[np.ndarray],
    ping_amp: float,
    ping_steps: int,
) -> list[tuple[int, int, float, float, dict[str, object]]]:
    if len(boundaries) != len(region_masks):
        raise ValueError("Fig.3 region-ping batch requires one mask per boundary.")
    if not boundaries:
        return []
    batch_size = int(len(boundaries))
    batched_boundary = concat_named_boundaries(boundaries, device=ctx.device)
    restore_info = restore_condition_state_for_functional_readout(ctx, batched_boundary, batch_size=batch_size)
    input_shape = _layer_input_shapes_for_batch(batched_boundary, batch_size)["layer1"]
    zero = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    mask_tensors: list[torch.Tensor] = []
    for region_mask in region_masks:
        mask_arr = np.asarray(region_mask, dtype=np.float32)
        if tuple(mask_arr.shape) == tuple(input_shape[1:]):
            mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32, device=ctx.device)
        elif len(input_shape) == 4 and tuple(mask_arr.shape) == tuple(input_shape[2:]):
            mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32, device=ctx.device).unsqueeze(0)
            mask_tensor = mask_tensor.expand(input_shape[1], input_shape[2], input_shape[3]).contiguous()
        else:
            raise ValueError(f"region_mask shape {mask_arr.shape} is incompatible with layer1 input shape {input_shape}")
        mask_tensors.append(mask_tensor)
    mask_batch = torch.stack(mask_tensors, dim=0)
    ping_drive = torch.as_tensor(float(ping_amp), dtype=torch.float32, device=ctx.device) * mask_batch
    active_unit_counts = (ping_drive > 0).detach().to(torch.float32).flatten(1).sum(dim=1).cpu().numpy().astype(np.float64, copy=False)
    total_currents = active_unit_counts * float(ping_amp) * int(ping_steps)
    with torch.no_grad():
        for t_idx in range(int(ping_steps)):
            _step_network_once(ctx.net, zero, int(t_idx), ping_drive=ping_drive)
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size)
    out: list[tuple[int, int, float, float, dict[str, object]]] = []
    for idx in range(batch_size):
        fire_step = int(fire[idx].item())
        fire_ms = int(fire_step * ctx.cfg.dt / ms) if fire_step >= 0 else -1
        out.append(
            (
                int(pred[idx].item()),
                fire_ms,
                float(total_currents[idx]),
                float(active_unit_counts[idx]),
                restore_info,
            )
        )
    return out
