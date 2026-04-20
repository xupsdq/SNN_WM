from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.experiments.common.dataset import encode_images
from src.experiments.common.monitored_dms import run_dms_snapshot_rollout
from src.experiments.distractor.shared.config import EPS


def center_grouped_voltage(v: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr - np.mean(arr, axis=-1, keepdims=True)


def compute_delta_v(v_condition: np.ndarray | torch.Tensor, v_static: np.ndarray | torch.Tensor) -> np.ndarray:
    return center_grouped_voltage(v_condition) - center_grouped_voltage(v_static)


def safe_normalize(v: np.ndarray | torch.Tensor, eps: float = EPS) -> np.ndarray | None:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= float(eps):
        return None
    return arr / norm


def safe_cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, eps: float = EPS) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(aa))
    norm_b = float(np.linalg.norm(bb))
    if norm_a <= float(eps) or norm_b <= float(eps):
        return float("nan")
    return float(np.dot(aa, bb) / (norm_a * norm_b))


def safe_angle_deg(cosine_value: float) -> float:
    if not np.isfinite(float(cosine_value)):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(cosine_value), -1.0, 1.0))))


def compute_reference_direction_from_delays(
    delta_v_by_delay: Mapping[float, np.ndarray],
    *,
    eps: float = EPS,
) -> dict[str, object]:
    ordered_items = sorted(
        ((float(delay_ms), np.asarray(vec, dtype=np.float64).reshape(-1)) for delay_ms, vec in delta_v_by_delay.items()),
        key=lambda item: item[0],
    )
    if not ordered_items:
        return {"status": "empty", "u_ref": None}
    stacked = np.stack([item[1] for item in ordered_items], axis=0)
    summed = stacked.sum(axis=0)
    u_ref = safe_normalize(summed, eps=eps)
    if u_ref is not None:
        return {
            "status": "mean_direction",
            "u_ref": u_ref,
            "summed_norm": float(np.linalg.norm(summed)),
            "fallback_delay_ms": None,
        }
    norms = np.linalg.norm(stacked, axis=1)
    valid = np.flatnonzero(norms > float(eps))
    if valid.size > 0:
        best_idx = int(valid[np.argmax(norms[valid])])
        fallback_vec = stacked[best_idx]
        u_fallback = safe_normalize(fallback_vec, eps=eps)
        if u_fallback is not None:
            return {
                "status": "fallback_max_norm_delay",
                "u_ref": u_fallback,
                "summed_norm": float(np.linalg.norm(summed)),
                "fallback_delay_ms": float(ordered_items[best_idx][0]),
            }
    return {
        "status": "skip_all_zero",
        "u_ref": None,
        "summed_norm": float(np.linalg.norm(summed)),
        "fallback_delay_ms": None,
    }


def compute_strength_metrics(delta_v: np.ndarray, u_ref: np.ndarray, *, eps: float = EPS) -> dict[str, float]:
    vec = np.asarray(delta_v, dtype=np.float64).reshape(-1)
    ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    magnitude = float(np.linalg.norm(vec))
    effective = float(np.dot(vec, ref))
    cos_theta = safe_cosine(vec, ref, eps=eps)
    return {
        "M": magnitude,
        "A": effective,
        "cos_theta": cos_theta,
        "theta_deg": safe_angle_deg(cos_theta),
    }


def _prepare_single_source_spike_batch(
    images: torch.Tensor,
    batch_df: pd.DataFrame,
    *,
    encoder,
    preceding_steps: int,
    probe_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    preceding_ids = batch_df["preceding_id"].astype(int).tolist()
    probe_ids = batch_df["probe_id"].astype(int).tolist()
    unique_preceding_ids = list(dict.fromkeys(preceding_ids))
    unique_probe_ids = list(dict.fromkeys(probe_ids))
    preceding_encoded = encode_images(
        encoder,
        images[[int(idx) for idx in unique_preceding_ids]].to(device=device, dtype=torch.float32),
        steps=int(preceding_steps),
    )
    probe_encoded = encode_images(
        encoder,
        images[[int(idx) for idx in unique_probe_ids]].to(device=device, dtype=torch.float32),
        steps=int(probe_steps),
    )
    preceding_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_preceding_ids)}
    probe_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_probe_ids)}
    preceding_select = torch.tensor([preceding_lookup[int(idx)] for idx in preceding_ids], dtype=torch.long, device=device)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], dtype=torch.long, device=device)
    return preceding_encoded.index_select(0, preceding_select), probe_encoded.index_select(0, probe_select)


def _extract_grouped_voltage_vector(net, voltage_snapshot: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(voltage_snapshot.to(torch.float32))
    return grouped.mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)


def run_single_source_preceding_item_task(
    net,
    preceding_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    gap_steps: int,
    stsp_mode: str,
    readout_step: int,
    phase_reset: bool = True,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        out = run_dms_snapshot_rollout(
            net=net,
            sample_spikes=preceding_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(gap_steps),
            stsp_mode=str(stsp_mode),
            phase_reset=bool(phase_reset),
            intervention_plan=None,
            readout_step=int(readout_step),
            snapshot_state_names=("v_mem",),
            record_full_trace_state_names=(),
        )
    prediction_probe = out["predictions"]["prediction_probe"].numpy().astype(np.int64, copy=False)
    grouped_voltage = _extract_grouped_voltage_vector(net, out["readout_snapshots"]["layer3"]["v_mem"])
    return {
        "grouped_voltage": grouped_voltage,
        "prediction_probe": prediction_probe,
    }


def _project_vectors_to_2d(vectors: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    centered = arr - arr.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return np.concatenate([centered, np.zeros((centered.shape[0], 1), dtype=np.float64)], axis=1)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((centered.shape[0], 2), dtype=np.float64)
    projected = centered @ vh[:2].T
    if projected.shape[1] == 1:
        projected = np.concatenate([projected, np.zeros((projected.shape[0], 1), dtype=np.float64)], axis=1)
    return projected[:, :2]


def _normalize_projection_for_plot(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if not np.isfinite(max_abs) or max_abs <= EPS:
        return np.zeros_like(arr)
    return arr / max_abs


def compute_spearman_summary(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> dict[str, object]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 3:
        return {"status": "insufficient_samples", "n": int(xx.size), "rho": None, "p_value": None}
    if float(np.std(xx)) <= EPS or float(np.std(yy)) <= EPS:
        return {"status": "zero_variance", "n": int(xx.size), "rho": None, "p_value": None}
    result = stats.spearmanr(xx, yy)
    return {
        "status": "ok",
        "n": int(xx.size),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


__all__ = [
    "_normalize_projection_for_plot",
    "_prepare_single_source_spike_batch",
    "_project_vectors_to_2d",
    "compute_delta_v",
    "compute_reference_direction_from_delays",
    "compute_spearman_summary",
    "compute_strength_metrics",
    "run_single_source_preceding_item_task",
]
