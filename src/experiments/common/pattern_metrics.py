from __future__ import annotations

import numpy as np
import torch


def extract_grouped_voltage_vector(net, voltage_snapshot: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(voltage_snapshot.to(torch.float32))
    return grouped.mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)


def normalize_pattern_vector(x: np.ndarray | torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    centered = arr - float(arr.mean())
    norm = float(np.linalg.norm(centered, ord=2))
    return centered / float(norm + float(eps))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)))


def compute_trace_pattern_similarity(
    cond_trace: np.ndarray | torch.Tensor,
    ref_trace_dyn: np.ndarray | torch.Tensor,
    ref_trace_sta: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    cond = np.asarray(cond_trace, dtype=np.float64)
    ref_dyn = np.asarray(ref_trace_dyn, dtype=np.float64)
    ref_sta = np.asarray(ref_trace_sta, dtype=np.float64)
    if cond.shape != ref_dyn.shape or cond.shape != ref_sta.shape:
        raise ValueError("Trace shapes must match for pattern similarity.")
    s_dyn = np.zeros(cond.shape[0], dtype=np.float64)
    s_sta = np.zeros(cond.shape[0], dtype=np.float64)
    for t_step in range(cond.shape[0]):
        cond_norm = normalize_pattern_vector(cond[t_step], eps=eps)
        dyn_norm = normalize_pattern_vector(ref_dyn[t_step], eps=eps)
        sta_norm = normalize_pattern_vector(ref_sta[t_step], eps=eps)
        s_dyn[t_step] = _cosine(cond_norm, dyn_norm)
        s_sta[t_step] = _cosine(cond_norm, sta_norm)
    return s_dyn, s_sta, float(np.mean(s_dyn - s_sta))


def compute_final_pattern_similarity(
    v_cond: np.ndarray | torch.Tensor,
    v_full_dyn: np.ndarray | torch.Tensor,
    v_full_sta: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    cond_norm = normalize_pattern_vector(v_cond, eps=eps)
    dyn_norm = normalize_pattern_vector(v_full_dyn, eps=eps)
    sta_norm = normalize_pattern_vector(v_full_sta, eps=eps)
    s_dyn = _cosine(cond_norm, dyn_norm)
    s_sta = _cosine(cond_norm, sta_norm)
    return float(s_dyn), float(s_sta), float(s_dyn - s_sta)
