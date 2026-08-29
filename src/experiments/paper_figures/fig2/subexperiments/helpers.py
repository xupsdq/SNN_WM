from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.monitored_dms import (
    boundary_state_to_restore_ux_by_layer,
    restore_functional_probe_state_in_place,
    snapshot_boundary_state,
)
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state, snapshot_ux_state
from src.experiments.paper_figures.fig2.constants import STATE_CONDITIONS
from src.experiments.paper_figures.fig2.types import ExperimentContext, FunctionalReadout, PairEpisodeStateBank

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _maybe_int(value: Any) -> int | float:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return float("nan")
    return int(value)

def _make_weak_probe_spikes_encoded_dropout(
    full_probe_spikes: torch.Tensor,
    keep_prob: float,
    *,
    seed: int,
    same_mask_count: int,
    use_same_mask_across_states: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    full = full_probe_spikes.to(device=device, dtype=torch.float32)
    keep_prob = float(np.clip(float(keep_prob), 0.0, 1.0))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if use_same_mask_across_states:
        mask = (torch.rand(full.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = (full * mask).repeat(int(same_mask_count), 1, 1, 1, 1)
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    else:
        expanded = full.repeat(int(same_mask_count), 1, 1, 1, 1)
        mask = (torch.rand(expanded.shape, generator=gen, device=device) < keep_prob).to(torch.float32)
        weak = expanded * mask
        realized_keep_fraction = float(mask.mean().detach().cpu().item())
    full_spike_count = float(full.sum().detach().cpu().item())
    weak_spike_count = float(weak.sum().detach().cpu().item())
    denom = full_spike_count * float(same_mask_count)
    return weak.contiguous(), {
        "keep_prob": keep_prob,
        "mask_space": "encoded_spikes",
        "realized_keep_fraction": realized_keep_fraction,
        "full_spike_count": full_spike_count,
        "weak_spike_count": weak_spike_count,
        "weak_spike_fraction": float(weak_spike_count / denom) if denom > 0.0 else 0.0,
        "same_mask_used_across_states": bool(use_same_mask_across_states),
        "mask_seed": int(seed),
    }

def _make_weak_probe_spikes_image_foreground(
    ctx: ExperimentContext,
    target_image_id: int,
    target_item: str,
    keep_prob: float,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _ = target_item
    target_image = ctx.dataset[int(target_image_id)][0].detach().cpu().to(torch.float32).squeeze()
    foreground = target_image.numpy() > float(ctx.cfg.foreground_threshold)
    target_foreground_count = int(foreground.sum())
    mask_rng = np.random.default_rng(int(seed))
    keep = mask_rng.random(foreground.shape) < float(keep_prob)
    mask = foreground & keep
    partial = (target_image.numpy() * mask.astype(np.float32)).astype(np.float32)
    partial_tensor = torch.as_tensor(partial, dtype=torch.float32).view(1, 1, *partial.shape)
    partial_spikes = encode_images(ctx.encoder, partial_tensor.to(ctx.device), ctx.cfg.weak_probe_steps).to(ctx.device)
    encoded_spike_count = float(partial_spikes.sum().detach().cpu().item())
    cue_pixel_count = int(mask.sum())
    cue_energy = float(partial.sum())
    cue_fraction_actual = float(cue_pixel_count / max(1, target_foreground_count))
    return partial_spikes.contiguous(), {
        "keep_prob": float(keep_prob),
        "mask_space": "image_foreground",
        "cue_pixel_count": cue_pixel_count,
        "target_foreground_count": target_foreground_count,
        "cue_fraction_actual": cue_fraction_actual,
        "cue_energy": cue_energy,
        "encoded_spike_count": encoded_spike_count,
        "weak_spike_count": encoded_spike_count,
        "weak_spike_fraction": float("nan"),
        "same_mask_used_across_states": True,
        "mask_seed": int(seed),
    }

def _weak_probe_mask_row(
    ctx: ExperimentContext,
    *,
    mask_id: int,
    pair_id: int,
    target_item: str,
    target_label: int,
    keep_prob: float,
    repeat_id: int,
    mask_seed: int,
    mask_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "mask_id": int(mask_id),
        "pair_id": int(pair_id),
        "target_item": str(target_item),
        "target_label": int(target_label),
        "keep_prob": float(keep_prob),
        "repeat_id": int(repeat_id),
        "mask_seed": int(mask_seed),
        "mask_space": str(mask_info.get("mask_space", ctx.cfg.weak_probe_mask_space)),
        "same_mask_used_across_states": bool(mask_info.get("same_mask_used_across_states", ctx.cfg.weak_probe_use_same_mask_across_states)),
        "weak_probe_scale": float(ctx.cfg.weak_probe_scale),
        "weak_probe_noise": float(ctx.cfg.weak_probe_noise),
        "realized_keep_fraction": _maybe_float(mask_info.get("realized_keep_fraction")),
        "full_spike_count": _maybe_float(mask_info.get("full_spike_count")),
        "weak_spike_count": _maybe_float(mask_info.get("weak_spike_count")),
        "weak_spike_fraction": _maybe_float(mask_info.get("weak_spike_fraction")),
        "cue_pixel_count": _maybe_int(mask_info.get("cue_pixel_count")),
        "target_foreground_count": _maybe_int(mask_info.get("target_foreground_count")),
        "cue_fraction_actual": _maybe_float(mask_info.get("cue_fraction_actual")),
        "cue_energy": _maybe_float(mask_info.get("cue_energy")),
        "encoded_spike_count": _maybe_float(mask_info.get("encoded_spike_count")),
    }

def _capture_pair_batch(
    ctx: ExperimentContext,
    a_spikes: torch.Tensor,
    b_spikes: torch.Tensor,
    *,
    delay2_steps: int | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Mapping[str, Mapping[str, torch.Tensor]]]]:
    cfg = ctx.cfg
    n, _, channels, height, width = a_spikes.shape
    conditions = len(STATE_CONDITIONS)
    prepare_network_state(ctx.net, n * conditions, channels, height, width)
    zero = torch.zeros((n * conditions, channels, height, width), device=ctx.device)
    current_time = 0

    def expand_phase(phase: str, t: int) -> torch.Tensor:
        x = zero.clone()
        if phase == "A":
            x[n : 2 * n] = a_spikes[:, t, ...]
            x[3 * n : 4 * n] = a_spikes[:, t, ...]
        elif phase == "B":
            x[2 * n : 3 * n] = b_spikes[:, t, ...]
            x[3 * n : 4 * n] = b_spikes[:, t, ...]
        return x

    with torch.no_grad():
        for t in range(cfg.sample_steps):
            current_time = _step_network_once(ctx.net, expand_phase("A", t), current_time)
        for _ in range(cfg.delay1_steps):
            current_time = _step_network_once(ctx.net, zero, current_time)
        for t in range(cfg.second_item_steps):
            current_time = _step_network_once(ctx.net, expand_phase("B", t), current_time)
        for _ in range(cfg.delay2_steps if delay2_steps is None else int(delay2_steps)):
            current_time = _step_network_once(ctx.net, zero, current_time)
        snapshot = snapshot_ux_state(ctx.net, batch_size=n * conditions)
        boundary = snapshot_boundary_state(ctx.net)

    bank: dict[str, dict[str, dict[str, np.ndarray]]] = {cond: {layer: {} for layer in LAYER_KEYS} for cond in STATE_CONDITIONS}
    boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    for cidx, cond in enumerate(STATE_CONDITIONS):
        sl = slice(cidx * n, (cidx + 1) * n)
        for layer in LAYER_KEYS:
            u = snapshot[layer]["u"][sl].astype(np.float32, copy=False)
            x = snapshot[layer]["x"][sl].astype(np.float32, copy=False)
            bank[cond][layer]["u"] = u
            bank[cond][layer]["x"] = x
            bank[cond][layer]["g"] = (u * x).astype(np.float32, copy=False)
        boundaries[cond] = _slice_boundary_state(boundary, sl)
    return bank, boundaries

def _step_network_once(net, input_t: torch.Tensor, current_time: int, *, stsp_mode: str = "dynamic", ping_drive: torch.Tensor | None = None) -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode, ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1

def _fit_mixture_models(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> dict[str, dict[str, Any]]:
    x_a = np.asarray(x_a, dtype=np.float64).reshape(-1)
    x_b = np.asarray(x_b, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    target_norm = float(np.linalg.norm(y))
    models = {
        "A_only": _linear_model_metrics(y, _fit_single(x_a, y)),
        "B_only": _linear_model_metrics(y, _fit_single(x_b, y)),
        "mean_AB": _fixed_model_metrics(y, 0.5 * x_a + 0.5 * x_b),
        "sum_AB": _fixed_model_metrics(y, x_a + x_b),
        "unconstrained_AB": _linear_model_metrics(y, _fit_two(x_a, x_b, y)),
        "convex_AB": _fixed_model_metrics(y, _convex_prediction(x_a, x_b, y)),
    }
    for name, metrics in models.items():
        metrics["target_norm"] = target_norm
        metrics["cv_r2"] = _cv_r2(name, x_a, x_b, y, cv_folds, seed)
        if name == "convex_AB":
            w_a = _convex_weight(x_a, x_b, y)
            metrics["convex_weight_A"] = w_a
            metrics["convex_weight_B"] = 1.0 - w_a
        if name == "unconstrained_AB":
            beta = _fit_two_coeffs(x_a, x_b, y)
            metrics["beta_A"] = beta[0]
            metrics["beta_B"] = beta[1]
            metrics["intercept"] = beta[2]
        if name == "A_only":
            beta, intercept = _fit_single_coeffs(x_a, y)
            metrics["beta_A"] = beta
            metrics["intercept"] = intercept
        if name == "B_only":
            beta, intercept = _fit_single_coeffs(x_b, y)
            metrics["beta_B"] = beta
            metrics["intercept"] = intercept
    return models

def _linear_model_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return _fixed_model_metrics(y, pred)

def _fixed_model_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    residual = y - pred
    residual_norm = float(np.linalg.norm(residual))
    target_norm = float(np.linalg.norm(y))
    return {
        "prediction": pred,
        "r2": _r2(y, pred),
        "residual_norm": residual_norm,
        "target_norm": target_norm,
        "residual_norm_ratio": float(residual_norm / max(target_norm, 1e-12)),
    }

def _fit_single(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, intercept = _fit_single_coeffs(x, y)
    return beta * x + intercept

def _fit_single_coeffs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mat = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coef[0]), float(coef[1])

def _fit_two(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta_a, beta_b, intercept = _fit_two_coeffs(x_a, x_b, y)
    return beta_a * x_a + beta_b * x_b + intercept

def _fit_two_coeffs(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mat = np.column_stack([x_a, x_b, np.ones_like(x_a)])
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])

def _convex_weight(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> float:
    d = x_a - x_b
    denom = float(np.dot(d, d))
    if denom <= 1e-12:
        return 0.5
    return float(np.clip(np.dot(y - x_b, d) / denom, 0.0, 1.0))

def _convex_prediction(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = _convex_weight(x_a, x_b, y)
    return w * x_a + (1.0 - w) * x_b

def _cv_r2(model_name: str, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> float:
    n = len(y)
    k = max(2, min(int(cv_folds), n))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    scores = []
    for fold in np.array_split(perm, k):
        if len(fold) == 0 or len(fold) == n:
            continue
        train = np.setdiff1d(perm, fold, assume_unique=False)
        pred = _predict_model(model_name, x_a, x_b, y, train, fold)
        scores.append(_r2(y[fold], pred))
    return float(np.nanmean(scores)) if scores else float("nan")

def _predict_model(model_name: str, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    if model_name == "A_only":
        beta, intercept = _fit_single_coeffs(x_a[train], y[train])
        return beta * x_a[test] + intercept
    if model_name == "B_only":
        beta, intercept = _fit_single_coeffs(x_b[train], y[train])
        return beta * x_b[test] + intercept
    if model_name == "mean_AB":
        return 0.5 * x_a[test] + 0.5 * x_b[test]
    if model_name == "sum_AB":
        return x_a[test] + x_b[test]
    if model_name == "convex_AB":
        w = _convex_weight(x_a[train], x_b[train], y[train])
        return w * x_a[test] + (1.0 - w) * x_b[test]
    beta_a, beta_b, intercept = _fit_two_coeffs(x_a[train], x_b[train], y[train])
    return beta_a * x_a[test] + beta_b * x_b[test] + intercept

def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)

def _access_scores(bank: PairEpisodeStateBank, pair_id: int, condition: str) -> dict[str, float]:
    idx = int(pair_id)
    layer = "layer3"
    variable = "g"
    state = bank.get(condition, layer, variable)[idx : idx + 1]
    sim_a = float(_row_centered_cosine(state, bank.get("S_A", layer, variable)[idx : idx + 1])[0])
    sim_b = float(_row_centered_cosine(state, bank.get("S_B", layer, variable)[idx : idx + 1])[0])
    sim0 = float(_row_centered_cosine(state, bank.get("S0", layer, variable)[idx : idx + 1])[0])
    a = max(0.0, (sim_a - sim0 + 1.0) / 2.0)
    b = max(0.0, (sim_b - sim0 + 1.0) / 2.0)
    return {"A": float(np.clip(a, 0.0, 1.0)), "B": float(np.clip(b, 0.0, 1.0))}

def _prediction_from_scores(scores: Mapping[str, float], a_label: int, b_label: int, seed: int) -> int:
    threshold = 0.15
    if max(scores["A"], scores["B"]) < threshold:
        return -1
    if abs(scores["A"] - scores["B"]) < 1e-9:
        return a_label if int(seed) % 2 == 0 else b_label
    return a_label if scores["A"] > scores["B"] else b_label

def _partial_cue_metrics(network_seed: int, raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped_stats: dict[tuple[str, float, str], dict[str, float]] = {}
    for (target, keep, cond), part in raw_df.groupby(["target_item", "keep_prob", "state_condition"], sort=False):
        denom = max(1, len(part))
        grouped_stats[(str(target), float(keep), str(cond))] = {
            "P_target": float(part["pred_is_target"].sum() / denom),
            "P_other_pair_member": float(part["pred_is_other_pair_member"].sum() / denom),
            "P_silent": float(part["silent"].sum() / denom),
        }
    for keys, part in raw_df.groupby(["state_condition", "target_item", "keep_prob"], sort=False):
        cond, target, keep = str(keys[0]), str(keys[1]), float(keys[2])
        denom = max(1, len(part))
        relevant = "S_A" if target == "A" else "S_B"
        irrelevant = "S_B" if target == "A" else "S_A"
        p = grouped_stats
        p_cond = p.get((target, keep, cond), {})
        p_sab = p.get((target, keep, "S_AB"), {})
        p_s0 = p.get((target, keep, "S0"), {})
        p_rel = p.get((target, keep, relevant), {})
        p_irrel = p.get((target, keep, irrelevant), {})
        rows.append(
            {
                "network_seed": int(network_seed),
                "state_condition": cond,
                "target_item": target,
                "keep_prob": keep,
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_A": float(part["pred_is_A"].sum() / denom),
                "P_B": float(part["pred_is_B"].sum() / denom),
                "P_pair_member": float(part["pred_is_pair_member"].sum() / denom),
                "P_other_pair_member": float(part["pred_is_other_pair_member"].sum() / denom),
                "P_other_class": float(part["pred_is_other_class"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "P_relevant_single_target": float(p_rel.get("P_target", np.nan)),
                "P_irrelevant_single_target": float(p_irrel.get("P_target", np.nan)),
                "target_recovery_gain_vs_S0": float(p_sab.get("P_target", np.nan) - p_s0.get("P_target", np.nan)),
                "target_recovery_gain_vs_relevant_single": float(p_sab.get("P_target", np.nan) - p_rel.get("P_target", np.nan)),
                "target_recovery_gain_vs_irrelevant_single": float(p_sab.get("P_target", np.nan) - p_irrel.get("P_target", np.nan)),
                "other_pair_intrusion_change_vs_relevant_single": float(
                    p_sab.get("P_other_pair_member", np.nan) - p_rel.get("P_other_pair_member", np.nan)
                ),
                "silent_reduction_vs_S0": float(p_s0.get("P_silent", np.nan) - p_sab.get("P_silent", np.nan)),
                "relevant_single_condition": relevant,
                "irrelevant_single_condition": irrelevant,
                "weak_probe_metric_mode": _mode_value(part, "weak_probe_metric_mode", "fig4_compat"),
                "weak_probe_mask_space": _mode_value(part, "mask_space", ""),
                "mean_first_fire_time_ms": float(pd.to_numeric(part["first_fire_time_ms"], errors="coerce").replace(-1, np.nan).mean()),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _ping_sweep_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in trial_df.groupby(["sweep_type", "ping_amp", "ping_ms", "state_condition"], sort=False):
        sweep_type, ping_amp, ping_ms, condition = str(keys[0]), float(keys[1]), int(keys[2]), str(keys[3])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "sweep_type": sweep_type,
                "ping_amp": ping_amp,
                "ping_ms": ping_ms,
                "state_condition": condition,
                "pair_member_readout_rate": float(part["pred_is_pair_member"].sum() / denom),
                "A_readout_rate": float(part["pred_is_A"].sum() / denom),
                "B_readout_rate": float(part["pred_is_B"].sum() / denom),
                "other_readout_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "sweep_type",
            "ping_amp",
            "ping_ms",
            "state_condition",
            "pair_member_readout_rate",
            "A_readout_rate",
            "B_readout_rate",
            "other_readout_rate",
            "silent_rate",
            "n_trials",
        ],
    )

def _completion_delay_sweep_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in trial_df.groupby(["delay2_ms", "state_condition", "keep_prob"], sort=False):
        delay2_ms, condition, keep_prob = int(keys[0]), str(keys[1]), float(keys[2])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay2_ms": delay2_ms,
                "state_condition": condition,
                "keep_prob": keep_prob,
                "target_recovery_rate": float(part["correct_target"].sum() / denom),
                "A_readout_rate": float(part["pred_is_A"].sum() / denom),
                "B_readout_rate": float(part["pred_is_B"].sum() / denom),
                "other_readout_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "delay2_ms",
            "state_condition",
            "keep_prob",
            "target_recovery_rate",
            "A_readout_rate",
            "B_readout_rate",
            "other_readout_rate",
            "silent_rate",
            "n_trials",
        ],
    )

def _completion_delay_sweep_contrast(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in metrics.groupby(["delay2_ms", "keep_prob"], sort=True):
        delay2_ms, keep_prob = int(keys[0]), float(keys[1])
        by_cond = {str(row["state_condition"]): row for row in part.to_dict("records")}
        recovery_sab = _maybe_float(by_cond.get("S_AB", {}).get("target_recovery_rate"))
        recovery_sb = _maybe_float(by_cond.get("S_B", {}).get("target_recovery_rate"))
        recovery_s0 = _maybe_float(by_cond.get("S0", {}).get("target_recovery_rate"))
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay2_ms": delay2_ms,
                "keep_prob": keep_prob,
                "recovery_SAB": recovery_sab,
                "recovery_SB": recovery_sb,
                "recovery_S0": recovery_s0,
                "completion_gain_SAB_minus_SB": _nan_diff(recovery_sab, recovery_sb),
                "completion_gain_SAB_minus_S0": _nan_diff(recovery_sab, recovery_s0),
                "n_trials_SAB": _maybe_int(by_cond.get("S_AB", {}).get("n_trials")),
                "n_trials_SB": _maybe_int(by_cond.get("S_B", {}).get("n_trials")),
                "n_trials_S0": _maybe_int(by_cond.get("S0", {}).get("n_trials")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "delay2_ms",
            "keep_prob",
            "recovery_SAB",
            "recovery_SB",
            "recovery_S0",
            "completion_gain_SAB_minus_SB",
            "completion_gain_SAB_minus_S0",
            "n_trials_SAB",
            "n_trials_SB",
            "n_trials_S0",
        ],
    )

def _stable_sweep_seed(network_seed: int, sweep_type: str, ping_amp: float, ping_ms: int, condition: str) -> int:
    token = f"{int(network_seed)}|{sweep_type}|{float(ping_amp):.6f}|{int(ping_ms)}|{condition}"
    value = 2166136261
    for ch in token:
        value = (value ^ ord(ch)) * 16777619
        value &= 0x7FFFFFFF
    return int(value)

def _partial_cue_auc_metrics(network_seed: int, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_item, target_part in metrics.groupby("target_item", sort=False):
        auc_by_condition: dict[str, float] = {}
        p50_by_condition: dict[str, float] = {}
        legacy_threshold_by_condition: dict[str, float] = {}
        for condition, part in target_part.groupby("state_condition", sort=False):
            ordered = part.sort_values("keep_prob")
            x = ordered["keep_prob"].to_numpy(dtype=float)
            y = ordered["P_target"].to_numpy(dtype=float)
            auc = _normalized_auc(x, y)
            auc_by_condition[str(condition)] = auc
            p50_by_condition[str(condition)] = _p50_from_curve(x, y, threshold=0.5)
            threshold_rows = ordered[ordered["P_target"] >= 0.5]
            legacy_threshold_by_condition[str(condition)] = float(threshold_rows["keep_prob"].iloc[0]) if not threshold_rows.empty else float("nan")
        for condition, part in target_part.groupby("state_condition", sort=False):
            ordered = part.sort_values("keep_prob")
            auc = auc_by_condition[str(condition)]
            relevant = "S_A" if target_item == "A" else "S_B"
            irrelevant = "S_B" if target_item == "A" else "S_A"
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "state_condition": str(condition),
                    "target_item": str(target_item),
                    "normalized_auc_target_recovery": auc,
                    "p50_target_recovery_keep_prob": p50_by_condition[str(condition)],
                    "legacy_threshold_keep_prob": legacy_threshold_by_condition[str(condition)],
                    "SAB_vs_S0_auc_gain": float(auc_by_condition.get("S_AB", 0.0) - auc_by_condition.get("S0", 0.0)),
                    "SAB_vs_relevant_single_auc_gain": float(auc_by_condition.get("S_AB", 0.0) - auc_by_condition.get(relevant, 0.0)),
                    "SAB_vs_irrelevant_single_auc_gain": float(auc_by_condition.get("S_AB", 0.0) - auc_by_condition.get(irrelevant, 0.0)),
                    "SAB_vs_relevant_single_upper_bound_gap": float(auc_by_condition.get(relevant, 0.0) - auc_by_condition.get("S_AB", 0.0)),
                    "SAB_vs_relevant_single_p50_shift": _nan_diff(p50_by_condition.get("S_AB"), p50_by_condition.get(relevant)),
                    "SAB_vs_S0_p50_shift": _nan_diff(p50_by_condition.get("S_AB"), p50_by_condition.get("S0")),
                    "low_cue_gain": _cue_gain(target_part, target_item, max_keep=0.1),
                    "mid_cue_gain": _cue_gain(target_part, target_item, min_keep=0.1, max_keep=0.3),
                    "high_cue_gain": _cue_gain(target_part, target_item, min_keep=0.3),
                    "weak_probe_metric_mode": _mode_value(part, "weak_probe_metric_mode", "fig4_compat"),
                    "weak_probe_mask_space": _mode_value(part, "weak_probe_mask_space", ""),
                    "n_trials": int(part["n_trials"].sum()) if "n_trials" in part.columns else int(len(part)),
                }
            )
    return pd.DataFrame(rows)

def _partial_cue_pair_metrics(network_seed: int, raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in raw_df.groupby(["pair_id", "target_item", "state_condition", "keep_prob"], sort=False):
        pair_id, target, cond, keep = int(keys[0]), str(keys[1]), str(keys[2]), float(keys[3])
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "pair_id": pair_id,
                "target_item": target,
                "state_condition": cond,
                "keep_prob": keep,
                "P_target": float(part["pred_is_target"].sum() / denom),
                "P_A": float(part["pred_is_A"].sum() / denom),
                "P_B": float(part["pred_is_B"].sum() / denom),
                "P_pair_member": float(part["pred_is_pair_member"].sum() / denom),
                "P_other_pair_member": float(part["pred_is_other_pair_member"].sum() / denom),
                "P_other_class": float(part["pred_is_other_class"].sum() / denom),
                "P_silent": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    return pd.DataFrame(rows)

def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return float(y.mean())
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return float(np.nanmean(y))
    return float(np.trapezoid(y, x) / span)

def _p50_from_curve(x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(y >= float(threshold)):
        first = int(np.argmax(y >= float(threshold)))
        if first == 0:
            return float(x[0])
        x0, x1 = float(x[first - 1]), float(x[first])
        y0, y1 = float(y[first - 1]), float(y[first])
        if abs(y1 - y0) <= 1e-12:
            return x1
        frac = (float(threshold) - y0) / (y1 - y0)
        return float(x0 + frac * (x1 - x0))
    return float("nan")

def _nan_diff(a: Any, b: Any) -> float:
    aa = float(a) if a is not None else float("nan")
    bb = float(b) if b is not None else float("nan")
    return float(aa - bb) if math.isfinite(aa) and math.isfinite(bb) else float("nan")

def _mode_value(part: pd.DataFrame, column: str, default: str) -> str:
    if column not in part.columns or part.empty:
        return str(default)
    values = part[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else str(default)

def _compat_fig4_weak_probe_outputs(
    network_seed: int,
    metrics: pd.DataFrame,
    auc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for _, row in metrics.iterrows():
        target = str(row["target_item"])
        summary_rows.append(
            {
                "network_seed": int(network_seed),
                "target_item": target,
                "fig4_mapping": "A_completion" if target == "A" else "mirrored_B_completion",
                "state_condition": str(row["state_condition"]),
                "baseline_condition": "S0",
                "relevant_single_condition": "S_A" if target == "A" else "S_B",
                "irrelevant_single_condition": "S_B" if target == "A" else "S_A",
                "keep_prob": float(row["keep_prob"]),
                "pred_A": float(row["P_A"]),
                "pred_B": float(row["P_B"]),
                "pred_other": float(row["P_other_class"]),
                "pred_silent": float(row["P_silent"]),
                "P_target": float(row["P_target"]),
                "P_pair": float(row["P_pair_member"]),
                "target_recovery_gain_vs_S0": float(row.get("target_recovery_gain_vs_S0", np.nan)),
                "target_recovery_gain_vs_relevant_single": float(row.get("target_recovery_gain_vs_relevant_single", np.nan)),
                "weak_probe_metric_mode": str(row.get("weak_probe_metric_mode", "")),
                "weak_probe_mask_space": str(row.get("weak_probe_mask_space", "")),
            }
        )
    auc_rows = []
    threshold_rows = []
    for _, row in auc.iterrows():
        base = {
            "network_seed": int(network_seed),
            "target_item": str(row["target_item"]),
            "fig4_mapping": "A_completion" if str(row["target_item"]) == "A" else "mirrored_B_completion",
            "state_condition": str(row["state_condition"]),
            "weak_probe_metric_mode": str(row.get("weak_probe_metric_mode", "")),
            "weak_probe_mask_space": str(row.get("weak_probe_mask_space", "")),
        }
        auc_rows.append(
            {
                **base,
                "normalized_auc_target_recovery": float(row.get("normalized_auc_target_recovery", np.nan)),
                "SAB_vs_S0_auc_gain": float(row.get("SAB_vs_S0_auc_gain", np.nan)),
                "SAB_vs_relevant_single_auc_gain": float(row.get("SAB_vs_relevant_single_auc_gain", np.nan)),
                "SAB_vs_irrelevant_single_auc_gain": float(row.get("SAB_vs_irrelevant_single_auc_gain", np.nan)),
                "low_cue_gain": float(row.get("low_cue_gain", np.nan)),
                "mid_cue_gain": float(row.get("mid_cue_gain", np.nan)),
                "high_cue_gain": float(row.get("high_cue_gain", np.nan)),
            }
        )
        threshold_rows.append(
            {
                **base,
                "p50_target_recovery_keep_prob": float(row.get("p50_target_recovery_keep_prob", np.nan)),
                "SAB_vs_relevant_single_p50_shift": float(row.get("SAB_vs_relevant_single_p50_shift", np.nan)),
                "SAB_vs_S0_p50_shift": float(row.get("SAB_vs_S0_p50_shift", np.nan)),
                "legacy_threshold_keep_prob": float(row.get("legacy_threshold_keep_prob", np.nan)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(auc_rows), pd.DataFrame(threshold_rows)

def _cue_gain(target_part: pd.DataFrame, target_item: str, *, min_keep: float = -np.inf, max_keep: float = np.inf) -> float:
    _ = target_item
    part = target_part[(pd.to_numeric(target_part["keep_prob"], errors="coerce") > float(min_keep)) & (pd.to_numeric(target_part["keep_prob"], errors="coerce") <= float(max_keep))]
    if part.empty:
        return float("nan")
    pivot = part.pivot_table(index="keep_prob", columns="state_condition", values="P_target", aggfunc="mean")
    if not {"S_AB", "S0"}.issubset(pivot.columns):
        return float("nan")
    return float((pivot["S_AB"] - pivot["S0"]).mean())

def run_ping_readout_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    ping_seed: int,
    ping_amp: float | None = None,
    ping_steps: int | None = None,
    record_trace: bool = False,
) -> FunctionalReadout:
    batch_size = int(next(iter(next(iter(boundary.values())).values())).shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    input_shape = _layer_input_shapes_for_batch(boundary, batch_size)["layer1"]
    zero_input = torch.zeros(input_shape, dtype=torch.float32, device=ctx.device)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(int(ping_seed))
    traces: dict[str, list[torch.Tensor]] = {"layer3_spikes": []}
    amp = float(ctx.cfg.ping_amp if ping_amp is None else ping_amp)
    steps = int(ctx.cfg.ping_steps if ping_steps is None else ping_steps)
    with torch.no_grad():
        for t_idx in range(steps):
            if ctx.cfg.ping_mode == "bernoulli_drive":
                ping_drive = (torch.rand(zero_input.shape, generator=gen, device=ctx.device) < amp).to(torch.float32)
            else:
                ping_drive = torch.full_like(zero_input, amp)
            if float(ctx.cfg.ping_noise) > 0.0:
                ping_drive = torch.clamp(ping_drive + torch.randn(ping_drive.shape, generator=gen, device=ctx.device) * float(ctx.cfg.ping_noise), min=0.0)
            s3 = _forward_three_layers_with_optional_trace(ctx.net, zero_input, t_idx, ping_drive=ping_drive)
            if record_trace:
                traces["layer3_spikes"].append(s3.detach().to(torch.float32).clone())
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=batch_size)
    trace = _pack_trace(traces) if record_trace else None
    return FunctionalReadout(
        prediction=pred.numpy().astype(np.int64, copy=False),
        first_fire_time_ms=fire_t.numpy().astype(np.float32, copy=False) * float(ctx.cfg.dt / ms),
        silent=(pred.numpy() < 0),
        readout_margin_A=_readout_margin_for_class(ctx, 0),
        readout_margin_B=_readout_margin_for_class(ctx, 1),
        trace=trace,
    )

def run_probe_readout_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    *,
    probe_scale: float = 1.0,
    probe_noise: float = 0.0,
    seed: int = 0,
    record_trace: bool = False,
) -> FunctionalReadout:
    batch_size = int(probe_spikes.shape[0])
    restore_condition_state_for_functional_readout(ctx, boundary, batch_size)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(int(seed))
    traces: dict[str, list[torch.Tensor]] = {"layer3_spikes": []}
    with torch.no_grad():
        for t_idx in range(int(probe_spikes.shape[1])):
            input_t = probe_spikes[:, t_idx].to(ctx.device, dtype=torch.float32) * float(probe_scale)
            if float(probe_noise) > 0.0:
                input_t = torch.clamp(
                    input_t + torch.randn(input_t.shape, generator=gen, device=ctx.device) * float(probe_noise),
                    min=0.0,
                )
            s3 = _forward_three_layers_with_optional_trace(ctx.net, input_t, t_idx)
            if record_trace:
                traces["layer3_spikes"].append(s3.detach().to(torch.float32).clone())
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(ctx.net, batch_size=batch_size)
    trace = _pack_trace(traces) if record_trace else None
    return FunctionalReadout(
        prediction=pred.numpy().astype(np.int64, copy=False),
        first_fire_time_ms=fire_t.numpy().astype(np.float32, copy=False) * float(ctx.cfg.dt / ms),
        silent=(pred.numpy() < 0),
        readout_margin_A=_readout_margin_for_class(ctx, 0),
        readout_margin_B=_readout_margin_for_class(ctx, 1),
        trace=trace,
    )

def restore_condition_state_for_functional_readout(ctx: ExperimentContext, boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, object]:
    layer_input_shapes = _layer_input_shapes_for_batch(boundary, int(batch_size))
    return restore_functional_probe_state_in_place(
        ctx.net,
        layer_input_shapes,
        boundary,
        mode=str(ctx.cfg.functional_restore_mode),
        device=ctx.device,
    )

def boundary_state_to_restore_ux_by_layer(boundary: Mapping[str, Mapping[str, torch.Tensor]], device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from src.experiments.common.monitored_dms import boundary_state_to_restore_ux_by_layer as _impl

    return _impl(boundary, device)

def slice_boundary_state(boundary_state: Mapping[str, Mapping[str, torch.Tensor]], row_indices: Sequence[int], device: torch.device | None = None) -> dict[str, dict[str, torch.Tensor]]:
    idx = torch.as_tensor(list(row_indices), dtype=torch.long)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary_state.items():
        out[layer_key] = {}
        for key, value in state.items():
            selected = value.index_select(0, idx).detach().clone()
            out[layer_key][key] = selected.to(device) if device is not None else selected
    return out

def concat_condition_boundaries(
    boundary_states_by_condition: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    conditions: Sequence[str],
    row_indices: Sequence[int],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    sliced = [slice_boundary_state(boundary_states_by_condition[condition], row_indices, device) for condition in conditions]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in sliced[0]:
        out[layer_key] = {}
        for key in sliced[0][layer_key]:
            out[layer_key][key] = torch.cat([part[layer_key][key] for part in sliced], dim=0)
    return out

def _forward_three_layers_with_optional_trace(net, input_t: torch.Tensor, t_step: int, *, ping_drive: torch.Tensor | None = None) -> torch.Tensor:
    s1, _ = net.layer1.forward_step(input_t, t_step, training=False, monitor=False, stsp_mode="dynamic", ping_drive=ping_drive)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    s2p = net.pool2(s2.float())
    s3, _ = net.layer3.forward_step(s2p, t_step, training=False, monitor=False, stsp_mode="dynamic")
    return s3

def _layer_input_shapes_from_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    return {layer_key: tuple(state["u"].shape) for layer_key, state in boundary.items() if "u" in state}

def _layer_input_shapes_for_batch(boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, tuple[int, ...]]:
    shapes = _layer_input_shapes_from_boundary(boundary)
    return {layer_key: (int(batch_size),) + tuple(shape[1:]) for layer_key, shape in shapes.items()}

def _pack_trace(traces: Mapping[str, list[torch.Tensor]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in traces.items():
        if values:
            out[key] = torch.stack(values, dim=0).cpu().numpy().astype(np.float32, copy=False)
    return out

def _readout_margin_for_class(ctx: ExperimentContext, class_id: int) -> np.ndarray:
    firing = ctx.net.layer3.firing_times.detach().cpu()
    batch_size = firing.shape[0]
    grouped = firing.view(batch_size, ctx.net.layer3.num_classes, ctx.net.layer3.neurons_per_class, -1).reshape(batch_size, ctx.net.layer3.num_classes, -1)
    class_min = grouped.min(dim=2).values
    target = class_min[:, int(class_id)].numpy().astype(np.float32, copy=False)
    others = class_min.clone()
    others[:, int(class_id)] = float("inf")
    other_min = others.min(dim=1).values.numpy().astype(np.float32, copy=False)
    margin = other_min - target
    margin[~np.isfinite(margin)] = np.nan
    return margin

def _readout_margin_value(values: np.ndarray | None, idx: int) -> float:
    if values is None or idx >= len(values):
        return float("nan")
    return float(values[idx])

def _ping_spike_count(ctx: ExperimentContext, ping_seed: int) -> float:
    shape = tuple(ctx.net.layer1.output_shape)
    if ctx.cfg.ping_mode == "bernoulli_drive":
        gen = torch.Generator(device=ctx.device)
        gen.manual_seed(int(ping_seed))
        return float(sum((torch.rand(shape, generator=gen, device=ctx.device) < float(ctx.cfg.ping_amp)).sum().item() for _ in range(int(ctx.cfg.ping_steps))))
    return float(np.prod(shape) * int(ctx.cfg.ping_steps)) if float(ctx.cfg.ping_amp) > 0.0 else 0.0

def _ping_energy(ctx: ExperimentContext, ping_seed: int) -> float:
    _ = ping_seed
    shape = tuple(ctx.net.layer1.output_shape)
    if ctx.cfg.ping_mode == "bernoulli_drive":
        return float(_ping_spike_count(ctx, ping_seed))
    return float(np.prod(shape) * float(ctx.cfg.ping_amp) * int(ctx.cfg.ping_steps))

def _neutral_ping_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup: dict[str, dict[str, float]] = {}
    for condition, part in trial_df.groupby("state_condition", sort=False):
        denom = max(1, len(part))
        row = {
            "network_seed": int(network_seed),
            "state_condition": condition,
            "P_A": float(part["pred_is_A"].sum() / denom),
            "P_B": float(part["pred_is_B"].sum() / denom),
            "P_pair": float(part["pred_is_pair_member"].sum() / denom),
            "P_other": float(part["pred_is_other"].sum() / denom),
            "P_silent": float(part["silent"].sum() / denom),
            "P_A_minus_B": float((part["pred_is_A"].sum() - part["pred_is_B"].sum()) / denom),
            "pair_access_gain_SAB_vs_S0": 0.0,
            "old_item_rescue_SAB_vs_SB": 0.0,
            "new_item_rescue_SAB_vs_SA": 0.0,
            "dual_access_balance": 0.0,
            "n_trials": int(len(part)),
        }
        lookup[str(condition)] = row
        rows.append(row)
    p_a_sab = lookup.get("S_AB", {}).get("P_A", 0.0)
    p_b_sab = lookup.get("S_AB", {}).get("P_B", 0.0)
    for row in rows:
        row["pair_access_gain_SAB_vs_S0"] = float(lookup.get("S_AB", {}).get("P_pair", 0.0) - lookup.get("S0", {}).get("P_pair", 0.0))
        row["old_item_rescue_SAB_vs_SB"] = float(p_a_sab - lookup.get("S_B", {}).get("P_A", 0.0))
        row["new_item_rescue_SAB_vs_SA"] = float(p_b_sab - lookup.get("S_A", {}).get("P_B", 0.0))
        row["dual_access_balance"] = float(min(p_a_sab, p_b_sab))
    return pd.DataFrame(rows)

def _metric_lookup(path: Path, layer: str, variable: str, metric_col: str) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=float)
    df = pd.read_csv(path)
    part = df[(df["layer"].astype(str) == layer) & (df["state_variable"].astype(str) == variable)].sort_values("pair_id")
    return part[metric_col].to_numpy(dtype=float) if metric_col in part.columns else np.asarray([], dtype=float)

def _linear_metric_lookup(path: Path, layer: str, variable: str, model_name: str, metric_col: str) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=float)
    df = pd.read_csv(path)
    part = df[(df["layer"].astype(str) == layer) & (df["state_variable"].astype(str) == variable) & (df["model_name"].astype(str) == model_name)].sort_values("pair_id")
    return part[metric_col].to_numpy(dtype=float) if metric_col in part.columns else np.asarray([], dtype=float)

def _pair_sampling_audit(network_seed: int, pairs: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("pixel_similarity", "foreground_overlap"):
        values = pd.to_numeric(pairs[col], errors="coerce")
        rows.append({"network_seed": int(network_seed), "audit_type": f"{col}_summary", "label": "mean", "count": int(values.count()), "value": float(values.mean())})
        rows.append({"network_seed": int(network_seed), "audit_type": f"{col}_summary", "label": "std", "count": int(values.count()), "value": float(values.std(ddof=1) if values.count() > 1 else 0.0)})
    for class_pair, count in pairs["class_pair"].value_counts().sort_index().items():
        rows.append({"network_seed": int(network_seed), "audit_type": "class_pair_count", "label": str(class_pair), "count": int(count), "value": float(count)})
    rows.append({"network_seed": int(network_seed), "audit_type": "candidate_pool", "label": "eligible", "count": int(pool.get("eligible", pd.Series(dtype=int)).sum() if not pool.empty else 0), "value": float(len(pool))})
    return pd.DataFrame(rows)

def _trial_condition_audit(network_seed: int, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = [{"network_seed": int(network_seed), "audit_type": "n_pairs", "label": "all", "count": int(len(pairs)), "value": float(len(pairs))}]
    same = int((pairs["A_label"] == pairs["B_label"]).sum())
    rows.append({"network_seed": int(network_seed), "audit_type": "same_label_pairs", "label": "A_label_eq_B_label", "count": same, "value": float(same / max(1, len(pairs)))})
    for col in ("A_label", "B_label"):
        for label, count in pairs[col].value_counts().sort_index().items():
            rows.append({"network_seed": int(network_seed), "audit_type": f"{col}_count", "label": int(label), "count": int(count), "value": float(count)})
    return pd.DataFrame(rows)

def _image_similarity_and_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    sim = _centered_cosine(a, b)
    fa = a > 0.1
    fb = b > 0.1
    union = np.logical_or(fa, fb).sum()
    overlap = float(np.logical_and(fa, fb).sum() / max(1, union))
    return float(sim), overlap

def _selection_bin(sim: float) -> str:
    if sim < 0.25:
        return "low_similarity"
    if sim < 0.55:
        return "mid_similarity"
    return "high_similarity"

def _row_centered_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = a.astype(np.float64, copy=False) - a.astype(np.float64, copy=False).mean(axis=1, keepdims=True)
    b2 = b.astype(np.float64, copy=False) - b.astype(np.float64, copy=False).mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    return np.sum(a2 * b2, axis=1) / np.maximum(denom, 1e-12)

def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-12))

def _slice_boundary_state(boundary: Mapping[str, Mapping[str, torch.Tensor]], sl: slice) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        out[layer_key] = {key: value[sl].detach().cpu().clone() for key, value in state.items()}
    return out

def _concat_boundary_states(a: Mapping[str, Mapping[str, torch.Tensor]], b: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key in a:
        out[layer_key] = {}
        for key in a[layer_key]:
            out[layer_key][key] = torch.cat([a[layer_key][key], b[layer_key][key]], dim=0)
    return out

def _images_for_ids(dataset, image_ids: Iterable[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)

def _encode_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, *, cache: dict[tuple[Any, ...], torch.Tensor]) -> torch.Tensor:
    ids = tuple(int(v) for v in image_ids)
    key = (ids, int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]

def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), int(batch_size)):
        yield df.iloc[start : start + int(batch_size)].reset_index(drop=True)

__all__ = ('_make_weak_probe_spikes_encoded_dropout', '_make_weak_probe_spikes_image_foreground', '_weak_probe_mask_row', '_capture_pair_batch', '_step_network_once', '_fit_mixture_models', '_linear_model_metrics', '_fixed_model_metrics', '_fit_single', '_fit_single_coeffs', '_fit_two', '_fit_two_coeffs', '_convex_weight', '_convex_prediction', '_cv_r2', '_predict_model', '_r2', '_access_scores', '_prediction_from_scores', '_partial_cue_metrics', '_ping_sweep_metrics', '_completion_delay_sweep_metrics', '_completion_delay_sweep_contrast', '_stable_sweep_seed', '_partial_cue_auc_metrics', '_partial_cue_pair_metrics', '_normalized_auc', '_p50_from_curve', '_nan_diff', '_mode_value', '_compat_fig4_weak_probe_outputs', '_cue_gain', 'run_ping_readout_from_boundary', 'run_probe_readout_from_boundary', 'restore_condition_state_for_functional_readout', 'boundary_state_to_restore_ux_by_layer', 'slice_boundary_state', 'concat_condition_boundaries', '_forward_three_layers_with_optional_trace', '_layer_input_shapes_from_boundary', '_layer_input_shapes_for_batch', '_pack_trace', '_readout_margin_for_class', '_readout_margin_value', '_ping_spike_count', '_ping_energy', '_neutral_ping_metrics', '_metric_lookup', '_linear_metric_lookup', '_pair_sampling_audit', '_trial_condition_audit', '_image_similarity_and_overlap', '_selection_bin', '_row_centered_cosine', '_slice_boundary_state', '_concat_boundary_states', '_images_for_ids', '_encode_cached', '_iter_batches', '_progress', '_ms_to_steps', '_maybe_float', '_maybe_int')
