from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.encoding import build_mnist_skeleton_loader
from src.experiments.common.dataset import encode_images
from src.experiments.common.seed import mix_seed


def load_mnist_dataset(dataset_root: str, split: str):
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=1,
        input_size=28,
        num_workers=0,
    )
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_loader.dataset
    if split_name == "test":
        return test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


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
) -> list[int]:
    rng = np.random.default_rng(int(seed))
    per_class: dict[int, list[int]] = {}
    for class_label in sorted(class_index):
        ids = np.asarray([int(idx) for idx in class_index[int(class_label)]], dtype=np.int64)
        per_class[int(class_label)] = rng.permutation(ids).tolist()
    selected: list[int] = []
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


def _take_evenly_from_sorted(df: pd.DataFrame, count: int) -> list[int]:
    if count <= 0 or df.empty:
        return []
    base_index = (
        df["index"].to_numpy(dtype=np.int64, copy=False)
        if "index" in df.columns
        else df.index.to_numpy(dtype=np.int64, copy=False)
    )
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
    desired_bins = np.floor(
        np.linspace(0, max(len(unique_bins) - 1, 0), num=max(1, int(samples_per_probe)))
    ).astype(np.int64)
    bin_counts = {int(idx): int((desired_bins == idx).sum()) for idx in np.unique(desired_bins)}
    selected_idx: list[int] = []
    for bin_position, bin_label in enumerate(unique_bins):
        take = int(bin_counts.get(int(bin_position), 0))
        if take <= 0:
            continue
        sub = ordered[ordered["candidate_bin"] == bin_label].copy().reset_index()
        selected_idx.extend(_take_evenly_from_sorted(sub, take))
    selected_idx = sorted(dict.fromkeys(int(idx) for idx in selected_idx))
    if len(selected_idx) < int(samples_per_probe):
        leftovers = ordered.drop(index=selected_idx, errors="ignore")
        selected_idx.extend(_take_evenly_from_sorted(leftovers.reset_index(), int(samples_per_probe) - len(selected_idx)))
    selected = ordered.iloc[sorted(dict.fromkeys(selected_idx))].copy().reset_index(drop=True)
    return selected.iloc[: int(samples_per_probe)].copy()


def _rebalance_global_triplets(df_triplets: pd.DataFrame, max_triplets: int) -> pd.DataFrame:
    if len(df_triplets) <= int(max_triplets):
        return df_triplets.copy().reset_index(drop=True)
    by_probe = [sub.copy().reset_index(drop=True) for _, sub in df_triplets.groupby("probe_id", sort=True)]
    selected_parts: list[pd.DataFrame] = []
    cursor = 0
    while len(selected_parts) < int(max_triplets):
        made_progress = False
        for group in by_probe:
            if cursor >= len(group):
                continue
            selected_parts.append(group.iloc[[cursor]].copy())
            made_progress = True
            if len(selected_parts) >= int(max_triplets):
                break
        if not made_progress:
            break
        cursor += 1
    return pd.concat(selected_parts, axis=0, ignore_index=True).reset_index(drop=True)


def _select_distractor_for_sample_probe(
    *,
    sample_id: int,
    sample_label: int,
    probe_id: int,
    probe_label: int,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    all_ids: np.ndarray,
    num_bins: int,
    sample_position: int,
    total_samples_for_probe: int,
) -> dict[str, object]:
    sims_to_probe = flat_normalized @ flat_normalized[int(probe_id)]
    sims_to_sample = flat_normalized @ flat_normalized[int(sample_id)]
    mask = (
        (all_ids != int(sample_id))
        & (all_ids != int(probe_id))
        & (labels[all_ids] != int(sample_label))
        & (labels[all_ids] != int(probe_label))
    )
    candidate_ids = all_ids[mask]
    if candidate_ids.size <= 0:
        raise RuntimeError("No valid distractor candidate found for the requested sample/probe pair.")
    df_candidates = pd.DataFrame(
        {
            "distractor_id": candidate_ids.astype(np.int64, copy=False),
            "distractor_label": labels[candidate_ids].astype(np.int64, copy=False),
            "dp_similarity": sims_to_probe[candidate_ids].astype(np.float64, copy=False),
            "sd_similarity": sims_to_sample[candidate_ids].astype(np.float64, copy=False),
        }
    ).sort_values(["dp_similarity", "sd_similarity", "distractor_id"], kind="stable").reset_index(drop=True)
    df_candidates["dp_bin"] = assign_bins_from_values(
        df_candidates["dp_similarity"].to_numpy(dtype=np.float64, copy=False),
        num_bins=int(num_bins),
    )
    unique_bins = pd.unique(df_candidates["dp_bin"]).tolist()
    target_positions = np.floor(
        np.linspace(0, max(len(unique_bins) - 1, 0), num=max(1, int(total_samples_for_probe)))
    ).astype(np.int64)
    target_position = int(target_positions[min(int(sample_position), len(target_positions) - 1)])
    target_bin = str(unique_bins[target_position])
    sub = df_candidates[df_candidates["dp_bin"] == target_bin].copy().reset_index(drop=True)
    if sub.empty:
        sub = df_candidates.copy().reset_index(drop=True)
        target_bin = str(sub.iloc[0]["dp_bin"])
    chosen = sub.iloc[min(int(sample_position), len(sub) - 1)]
    return {
        "distractor_id": int(chosen["distractor_id"]),
        "distractor_label": int(chosen["distractor_label"]),
        "dp_similarity": float(chosen["dp_similarity"]),
        "sd_similarity": float(chosen["sd_similarity"]),
        "dp_bin": str(target_bin),
    }


def build_triplet_specs(
    images: torch.Tensor,
    labels: np.ndarray,
    flat_normalized: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    samples_per_probe: int,
    num_bins: int,
    max_triplets: int,
    seed: int,
) -> pd.DataFrame:
    del images
    probe_ids = select_probe_ids_balanced(class_index=class_index, max_probes=max_probes, seed=mix_seed(seed, 31))
    rows: list[dict[str, object]] = []
    all_ids = np.arange(len(labels), dtype=np.int64)
    for probe_rank, probe_id in enumerate(probe_ids):
        probe_id_int = int(probe_id)
        probe_label = int(labels[probe_id_int])
        sims_to_probe = flat_normalized @ flat_normalized[probe_id_int]
        mask = (all_ids != probe_id_int) & (labels[all_ids] != probe_label)
        candidate_ids = all_ids[mask]
        df_sample_candidates = pd.DataFrame(
            {
                "sample_id": candidate_ids.astype(np.int64, copy=False),
                "sample_label": labels[candidate_ids].astype(np.int64, copy=False),
                "probe_id": probe_id_int,
                "probe_label": probe_label,
                "probe_rank": int(probe_rank),
                "similarity_public_or_initial": sims_to_probe[candidate_ids].astype(np.float64, copy=False),
            }
        )
        selected_samples = select_probe_samples_from_candidates(
            df_candidates=df_sample_candidates,
            samples_per_probe=int(samples_per_probe),
            num_bins=int(num_bins),
        )
        if selected_samples.empty:
            continue
        selected_samples = selected_samples.rename(
            columns={"similarity_public_or_initial": "sp_similarity"}
        ).reset_index(drop=True)
        selected_samples["sp_bin"] = assign_bins_from_values(
            selected_samples["sp_similarity"].to_numpy(dtype=np.float64, copy=False),
            num_bins=int(num_bins),
        )
        for sample_position, row in enumerate(selected_samples.itertuples(index=False)):
            distractor = _select_distractor_for_sample_probe(
                sample_id=int(row.sample_id),
                sample_label=int(row.sample_label),
                probe_id=probe_id_int,
                probe_label=probe_label,
                labels=labels,
                flat_normalized=flat_normalized,
                all_ids=all_ids,
                num_bins=int(num_bins),
                sample_position=int(sample_position),
                total_samples_for_probe=int(len(selected_samples)),
            )
            rows.append(
                {
                    "sample_id": int(row.sample_id),
                    "sample_label": int(row.sample_label),
                    "distractor_id": int(distractor["distractor_id"]),
                    "distractor_label": int(distractor["distractor_label"]),
                    "probe_id": probe_id_int,
                    "probe_label": probe_label,
                    "probe_rank": int(probe_rank),
                    "sp_similarity": float(row.sp_similarity),
                    "dp_similarity": float(distractor["dp_similarity"]),
                    "sd_similarity": float(distractor["sd_similarity"]),
                    "sp_bin": str(row.sp_bin),
                    "dp_bin": str(distractor["dp_bin"]),
                }
            )
    if not rows:
        raise RuntimeError("No distractor triplets were generated.")
    df_triplets = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "distractor_id", "probe_id"], keep="first").reset_index(drop=True)
    df_triplets = _rebalance_global_triplets(df_triplets, max_triplets=int(max_triplets))
    sp_label_order = {label: idx for idx, label in enumerate(pd.unique(df_triplets["sp_bin"]).tolist())}
    dp_label_order = {label: idx for idx, label in enumerate(pd.unique(df_triplets["dp_bin"]).tolist())}
    df_triplets["sp_bin_index"] = df_triplets["sp_bin"].map(sp_label_order).astype(np.int64)
    df_triplets["dp_bin_index"] = df_triplets["dp_bin"].map(dp_label_order).astype(np.int64)
    df_triplets = df_triplets.sort_values(
        ["probe_rank", "sp_bin_index", "dp_bin_index", "sp_similarity", "dp_similarity", "sample_id", "distractor_id"],
        kind="stable",
    ).reset_index(drop=True)
    df_triplets["triplet_id"] = np.arange(len(df_triplets), dtype=np.int64)
    return df_triplets


def _stack_encoded_batch(
    image_ids: Sequence[int],
    *,
    images: torch.Tensor,
    encoder,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    batch_images = images[[int(idx) for idx in image_ids]].to(device=device, dtype=torch.float32)
    return encode_images(encoder, batch_images, steps=int(steps))


def prepare_triplet_spike_batch(
    images: torch.Tensor,
    batch_df: pd.DataFrame,
    encoder,
    *,
    sample_steps: int,
    distractor_steps: int,
    probe_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_ids = batch_df["sample_id"].astype(int).tolist()
    distractor_ids = batch_df["distractor_id"].astype(int).tolist()
    probe_ids = batch_df["probe_id"].astype(int).tolist()
    unique_sample_ids = list(dict.fromkeys(sample_ids))
    unique_distractor_ids = list(dict.fromkeys(distractor_ids))
    unique_probe_ids = list(dict.fromkeys(probe_ids))
    sample_encoded = _stack_encoded_batch(
        unique_sample_ids,
        images=images,
        encoder=encoder,
        steps=int(sample_steps),
        device=device,
    )
    distractor_encoded = _stack_encoded_batch(
        unique_distractor_ids,
        images=images,
        encoder=encoder,
        steps=int(distractor_steps),
        device=device,
    )
    probe_encoded = _stack_encoded_batch(
        unique_probe_ids,
        images=images,
        encoder=encoder,
        steps=int(probe_steps),
        device=device,
    )
    sample_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_sample_ids)}
    distractor_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_distractor_ids)}
    probe_lookup = {int(image_id): pos for pos, image_id in enumerate(unique_probe_ids)}
    sample_select = torch.tensor([sample_lookup[int(idx)] for idx in sample_ids], dtype=torch.long, device=device)
    distractor_select = torch.tensor([distractor_lookup[int(idx)] for idx in distractor_ids], dtype=torch.long, device=device)
    probe_select = torch.tensor([probe_lookup[int(idx)] for idx in probe_ids], dtype=torch.long, device=device)
    return (
        sample_encoded.index_select(0, sample_select),
        distractor_encoded.index_select(0, distractor_select),
        probe_encoded.index_select(0, probe_select),
    )


__all__ = [
    "assign_bins_from_values",
    "build_triplet_specs",
    "load_mnist_dataset",
    "prepare_triplet_spike_batch",
    "select_probe_ids_balanced",
    "select_probe_samples_from_candidates",
]
