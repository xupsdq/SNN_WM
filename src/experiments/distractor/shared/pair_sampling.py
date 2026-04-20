from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import build_dataset_arrays as shared_build_dataset_arrays
from src.experiments.common.dataset import encode_images
from src.experiments.common.pattern_metrics import extract_grouped_voltage_vector as shared_extract_grouped_voltage_vector
from src.experiments.common.seed import mix_seed


@dataclass(frozen=True)
class PairExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def build_dataset_arrays(dataset) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    return shared_build_dataset_arrays(dataset)


def assign_bins_from_values(values: np.ndarray, num_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=object)
    q = max(1, min(int(num_bins), int(arr.size)))
    labels = [f"bin_{idx + 1}" for idx in range(q)]
    if q == 1:
        return np.asarray([labels[0]] * arr.size, dtype=object)
    ranks = pd.Series(arr).rank(method="first")
    try:
        return pd.qcut(ranks, q=q, labels=labels).astype("object").to_numpy()
    except ValueError:
        order = np.argsort(arr, kind="stable")
        raw = np.linspace(0, q - 1, num=arr.size)
        out = np.empty(arr.size, dtype=object)
        for pos, idx in enumerate(order.tolist()):
            out[idx] = labels[int(round(raw[pos]))]
        return out


def select_probe_ids_balanced(
    class_index: Mapping[int, Sequence[int]],
    max_probes: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(int(seed))
    per_class: Dict[int, List[int]] = {}
    for class_label in sorted(class_index):
        ids = np.asarray([int(idx) for idx in class_index[int(class_label)]], dtype=np.int64)
        per_class[int(class_label)] = rng.permutation(ids).tolist()
    selected: List[int] = []
    while len(selected) < int(max_probes):
        made_progress = False
        for class_label in sorted(per_class):
            if not per_class[class_label]:
                continue
            selected.append(int(per_class[class_label].pop(0)))
            made_progress = True
            if len(selected) >= int(max_probes):
                break
        if not made_progress:
            break
    return selected


def _take_evenly_from_sorted(df: pd.DataFrame, count: int) -> List[int]:
    if count <= 0 or df.empty:
        return []
    base_index = df["index"].to_numpy(dtype=np.int64, copy=False) if "index" in df.columns else df.index.to_numpy(dtype=np.int64, copy=False)
    if len(df) <= int(count):
        return base_index.astype(int).tolist()
    positions = np.linspace(0, len(df) - 1, num=int(count))
    taken = sorted({int(round(pos)) for pos in positions.tolist()})
    while len(taken) < int(count):
        for idx in range(len(df)):
            if idx not in taken:
                taken.append(idx)
            if len(taken) >= int(count):
                break
    taken = sorted(taken[: int(count)])
    return base_index[np.asarray(taken, dtype=np.int64)].astype(int).tolist()


def select_probe_samples_from_candidates(
    df_candidates: pd.DataFrame,
    samples_per_probe: int,
    num_bins: int,
) -> pd.DataFrame:
    if df_candidates.empty:
        return df_candidates.iloc[:0].copy()
    ordered = df_candidates.sort_values(["similarity_public_or_initial", "sample_id"], kind="stable").reset_index(drop=True)
    ordered["candidate_bin"] = assign_bins_from_values(
        ordered["similarity_public_or_initial"].to_numpy(dtype=np.float64, copy=False),
        num_bins=num_bins,
    )
    unique_bins = pd.unique(ordered["candidate_bin"]).tolist()
    desired_bins = np.floor(np.linspace(0, max(len(unique_bins) - 1, 0), num=max(1, int(samples_per_probe)))).astype(np.int64)
    bin_counter = Counter(int(idx) for idx in desired_bins.tolist())
    selected_idx: List[int] = []
    for bin_position, bin_label in enumerate(unique_bins):
        take = int(bin_counter.get(int(bin_position), 0))
        if take <= 0:
            continue
        sub = ordered[ordered["candidate_bin"] == bin_label].copy().reset_index()
        selected_idx.extend(_take_evenly_from_sorted(sub, take))
    selected_idx = sorted(dict.fromkeys(int(idx) for idx in selected_idx))
    if len(selected_idx) < int(samples_per_probe):
        leftovers = ordered.drop(index=selected_idx, errors="ignore")
        missing = int(samples_per_probe) - len(selected_idx)
        selected_idx.extend(_take_evenly_from_sorted(leftovers.reset_index(), missing))
    selected = ordered.iloc[sorted(dict.fromkeys(selected_idx))].copy().reset_index(drop=True)
    return selected.iloc[: int(samples_per_probe)].copy()


def _rebalance_global_pairs(df_pairs: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    if len(df_pairs) <= int(max_pairs):
        return df_pairs.copy().reset_index(drop=True)
    by_probe = [sub.copy().reset_index(drop=True) for _, sub in df_pairs.groupby("probe_id", sort=True)]
    selected_parts: List[pd.DataFrame] = []
    cursor = 0
    while len(selected_parts) < int(max_pairs):
        made_progress = False
        for group in by_probe:
            if cursor >= len(group):
                continue
            selected_parts.append(group.iloc[[cursor]].copy())
            made_progress = True
            if len(selected_parts) >= int(max_pairs):
                break
        if not made_progress:
            break
        cursor += 1
    return pd.concat(selected_parts, axis=0, ignore_index=True).reset_index(drop=True)


def build_pair_specs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    samples_per_probe: int,
    num_bins: int,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    del images
    probe_ids = select_probe_ids_balanced(class_index=class_index, max_probes=max_probes, seed=mix_seed(seed, 31))
    rows: List[Dict[str, object]] = []
    all_ids = np.arange(len(labels), dtype=np.int64)
    for probe_rank, probe_id in enumerate(probe_ids):
        probe_id_int = int(probe_id)
        sims = flat_normalized @ flat_normalized[probe_id_int]
        mask = all_ids != probe_id_int
        candidate_ids = all_ids[mask]
        candidate_sims = sims[mask]
        df_candidates = pd.DataFrame(
            {
                "sample_id": candidate_ids.astype(np.int64, copy=False),
                "sample_label": labels[candidate_ids].astype(np.int64, copy=False),
                "probe_id": int(probe_id_int),
                "probe_label": int(labels[probe_id_int]),
                "probe_rank": int(probe_rank),
                "similarity_public_or_initial": candidate_sims.astype(np.float64, copy=False),
            }
        )
        selected = select_probe_samples_from_candidates(
            df_candidates=df_candidates,
            samples_per_probe=int(samples_per_probe),
            num_bins=int(num_bins),
        )
        rows.extend(selected.to_dict("records"))
    if not rows:
        raise RuntimeError("No sample-probe pairs were generated.")
    df_pairs = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "probe_id"], keep="first").reset_index(drop=True)
    df_pairs = _rebalance_global_pairs(df_pairs, max_pairs=max_pairs)
    df_pairs["similarity_bin"] = assign_bins_from_values(
        df_pairs["similarity_public_or_initial"].to_numpy(dtype=np.float64, copy=False),
        num_bins=num_bins,
    )
    label_order = {label: idx for idx, label in enumerate(pd.unique(df_pairs["similarity_bin"]).tolist())}
    df_pairs["similarity_bin_index"] = df_pairs["similarity_bin"].map(label_order).astype(np.int64)
    df_pairs = df_pairs.sort_values(
        ["probe_rank", "similarity_bin_index", "similarity_public_or_initial", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)
    df_pairs["pair_id"] = np.arange(len(df_pairs), dtype=np.int64)
    return df_pairs


def _stack_encoded_batch(
    image_ids: Sequence[int],
    images: torch.Tensor,
    encoder,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    batch_images = images[[int(idx) for idx in image_ids]].to(device=device, dtype=torch.float32)
    return encode_images(encoder, batch_images, steps=int(steps))


def prepare_pair_spike_batch(
    images: torch.Tensor,
    batch_df: pd.DataFrame,
    encoder,
    spec: PairExperimentSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_ids = batch_df["sample_id"].astype(int).tolist()
    probe_ids = batch_df["probe_id"].astype(int).tolist()
    unique_sample_ids = list(dict.fromkeys(sample_ids))
    unique_probe_ids = list(dict.fromkeys(probe_ids))
    sample_encoded = _stack_encoded_batch(unique_sample_ids, images=images, encoder=encoder, steps=spec.sample_steps, device=device)
    probe_encoded = _stack_encoded_batch(unique_probe_ids, images=images, encoder=encoder, steps=spec.probe_steps, device=device)
    sample_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_sample_ids)}
    probe_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_probe_ids)}
    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], dtype=torch.long, device=device)
    return sample_encoded.index_select(0, sample_select), probe_encoded.index_select(0, probe_select)


def extract_grouped_voltage_vector(net, voltage_snapshot: torch.Tensor) -> np.ndarray:
    return shared_extract_grouped_voltage_vector(net, voltage_snapshot)
