from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.config.units import ms


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float
    delay_ms: float = 0.0

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round((self.delay_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def build_dataset_arrays(dataset) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    images = torch.stack([dataset[idx][0] for idx in range(len(dataset))], dim=0).cpu().to(torch.float32)
    labels = np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)
    flat = images.view(len(dataset), -1).numpy().astype(np.float32, copy=False)
    return images, labels, flat


def bootstrap_rate_ci(values: Iterable[float], *, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boot = np.zeros(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        sample_idx = rng.integers(0, arr.size, size=arr.size)
        boot[idx] = 100.0 * float(arr[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def rank_correlation(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    x_rank = pd.Series(x_arr[mask]).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_arr[mask]).rank(method="average").to_numpy(dtype=np.float64)
    if np.allclose(x_rank, x_rank[0]) or np.allclose(y_rank, y_rank[0]):
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def compute_region_specific_overlap(
    *,
    probe_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    diagnostic_mask: np.ndarray,
    nondiagnostic_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probe_arr = np.asarray(probe_vector, dtype=np.float64).reshape(-1)
    candidate_arr = np.asarray(candidate_matrix, dtype=np.float64)
    if candidate_arr.ndim != 2:
        raise ValueError("candidate_matrix must be 2D")
    diag_mask_flat = np.asarray(diagnostic_mask, dtype=bool).reshape(-1)
    nond_mask_flat = np.asarray(nondiagnostic_mask, dtype=bool).reshape(-1)
    if candidate_arr.shape[1] != probe_arr.shape[0]:
        raise ValueError("candidate_matrix width must match flattened probe_vector")

    if diag_mask_flat.any():
        diagnostic_scores = np.einsum("ij,j->i", candidate_arr[:, diag_mask_flat], probe_arr[diag_mask_flat], optimize=True)
    else:
        diagnostic_scores = np.zeros(candidate_arr.shape[0], dtype=np.float64)

    if nond_mask_flat.any():
        nondiagnostic_scores = np.einsum("ij,j->i", candidate_arr[:, nond_mask_flat], probe_arr[nond_mask_flat], optimize=True)
    else:
        nondiagnostic_scores = np.zeros(candidate_arr.shape[0], dtype=np.float64)

    return diagnostic_scores.astype(np.float64, copy=False), nondiagnostic_scores.astype(np.float64, copy=False)


def select_sample_types_for_probe(
    *,
    probe_id: int,
    probe_label: int,
    image_matrix_flat: np.ndarray,
    dataset_labels: np.ndarray,
    diagnostic_mask: np.ndarray,
    nondiagnostic_mask: np.ndarray,
) -> Dict[str, object]:
    candidate_ids = np.arange(len(dataset_labels), dtype=np.int64)
    keep_mask = candidate_ids != int(probe_id)
    filtered_ids = candidate_ids[keep_mask]
    filtered_labels = np.asarray(dataset_labels, dtype=np.int64)[keep_mask]
    filtered_matrix = np.asarray(image_matrix_flat, dtype=np.float32)[keep_mask]
    diagnostic_scores, nondiagnostic_scores = compute_region_specific_overlap(
        probe_vector=np.asarray(image_matrix_flat, dtype=np.float32)[int(probe_id)],
        candidate_matrix=filtered_matrix,
        diagnostic_mask=diagnostic_mask,
        nondiagnostic_mask=nondiagnostic_mask,
    )
    df = pd.DataFrame(
        {
            "candidate_id": filtered_ids.astype(np.int64, copy=False),
            "candidate_label": filtered_labels.astype(np.int64, copy=False),
            "diagnostic_overlap_score": diagnostic_scores.astype(np.float64, copy=False),
            "nondiagnostic_overlap_score": nondiagnostic_scores.astype(np.float64, copy=False),
        }
    )
    df["label_relation"] = np.where(df["candidate_label"] == int(probe_label), "same_label", "different_label")
    df["diagnostic_margin"] = df["diagnostic_overlap_score"] - df["nondiagnostic_overlap_score"]
    df["nondiagnostic_margin"] = df["nondiagnostic_overlap_score"] - df["diagnostic_overlap_score"]

    if df.empty:
        return {
            "probe_id": int(probe_id),
            "probe_label": int(probe_label),
            "selection_status": "excluded",
            "selection_exclusion_reason": "no_candidate_samples",
        }

    relation_scores = df.groupby("label_relation", sort=True)["diagnostic_margin"].max().sort_values(ascending=False, kind="stable")
    label_relation = str(relation_scores.index[0]) if not relation_scores.empty else "different_label"
    subset = df[df["label_relation"] == label_relation].copy()
    if subset.empty:
        return {
            "probe_id": int(probe_id),
            "probe_label": int(probe_label),
            "selection_status": "excluded",
            "selection_exclusion_reason": "no_candidates_for_label_relation",
        }

    diagnostic_pick = subset.sort_values(
        ["diagnostic_margin", "diagnostic_overlap_score", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    nond_subset = subset[subset["candidate_id"] != int(diagnostic_pick["candidate_id"])].copy()
    if nond_subset.empty:
        return {
            "probe_id": int(probe_id),
            "probe_label": int(probe_label),
            "selection_status": "excluded",
            "selection_exclusion_reason": "no_distinct_nondiagnostic_candidate",
        }
    nondiagnostic_pick = nond_subset.sort_values(
        ["nondiagnostic_margin", "nondiagnostic_overlap_score", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    return {
        "probe_id": int(probe_id),
        "probe_label": int(probe_label),
        "selection_status": "selected",
        "selection_exclusion_reason": "",
        "label_relation": str(label_relation),
        "diagnostic_sample_id": int(diagnostic_pick["candidate_id"]),
        "diagnostic_sample_label": int(diagnostic_pick["candidate_label"]),
        "diagnostic_overlap_score": float(diagnostic_pick["diagnostic_overlap_score"]),
        "diagnostic_nondiagnostic_overlap_score": float(diagnostic_pick["nondiagnostic_overlap_score"]),
        "diagnostic_margin": float(diagnostic_pick["diagnostic_margin"]),
        "nondiagnostic_sample_id": int(nondiagnostic_pick["candidate_id"]),
        "nondiagnostic_sample_label": int(nondiagnostic_pick["candidate_label"]),
        "nondiagnostic_overlap_score": float(nondiagnostic_pick["diagnostic_overlap_score"]),
        "nondiagnostic_nondiagnostic_overlap_score": float(nondiagnostic_pick["nondiagnostic_overlap_score"]),
        "nondiagnostic_margin": float(nondiagnostic_pick["nondiagnostic_margin"]),
    }
