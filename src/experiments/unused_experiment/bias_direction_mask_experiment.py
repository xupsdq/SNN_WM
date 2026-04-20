from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import run_dms_snapshot_rollout
from src.experiments.common.ping_common import prepare_network_state
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.diagnostic_mask_utils import (
    PatchSpec,
    build_patch_grid,
    project_patch_values_to_image,
)
from src.experiments.common.voltage_readout import resolve_readout_step
from src.plotting.common.io import (
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_SCATTER_LIGHT,
    CMAP_ACTIVATION,
    CMAP_IMAGE_GRAY,
    CMAP_OVERLAP,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_BROWN,
    FIGSIZE_THREE_PANEL,
    GRID_ALPHA,
    LAYER_ROUTE_COLORS,
    apply_standard_legend,
    case_grid_figsize,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/bias_direction_mask_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY_MS = 500.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_PAIRS = 240
DEFAULT_NUM_SIM_BINS = 5
DEFAULT_L1_PATCH_SIZE = 5
DEFAULT_L1_STRIDE = 4
DEFAULT_L3_MASK_MODE = "1x1"
DEFAULT_L3_TEMPORAL_POOL = "mean"
DEFAULT_SAVE_CASE_COUNT = 4


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


@dataclass(frozen=True)
class L3RegionSpec:
    region_id: int
    row_index: int
    col_index: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int


def mix_seed(base_seed: int, *parts: int) -> int:
    value = int(base_seed) & 0xFFFFFFFF
    for idx, part in enumerate(parts, start=1):
        value = (value * 1664525 + 1013904223 + int(part) * (374761393 + idx * 97)) & 0xFFFFFFFF
    return int(value)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    return value


def _save_json(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_dataset(dataset_root: str, split: str):
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


def build_dataset_arrays(dataset) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    images = torch.stack([dataset[idx][0].detach().cpu().to(torch.float32) for idx in range(len(dataset))], dim=0)
    labels = np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)
    flat = images.view(len(dataset), -1).numpy().astype(np.float64, copy=False)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return images, labels, flat / norms


def _cosine_similarity_1d(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), eps)
    return float(np.dot(aa, bb) / denom)


def _cosine_similarity_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    aa = a.detach().cpu().to(torch.float32).reshape(-1)
    bb = b.detach().cpu().to(torch.float32).reshape(-1)
    denom = max(float(torch.norm(aa) * torch.norm(bb)), eps)
    return float(torch.dot(aa, bb) / denom)


def _assign_bins_from_values(values: np.ndarray, num_bins: int) -> np.ndarray:
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


def build_l3_regions(height: int, width: int, mask_mode: str) -> List[L3RegionSpec]:
    mode = str(mask_mode).strip().lower()
    if mode not in {"1x1", "2x2"}:
        raise ValueError(f"Unsupported --l3-mask-mode: {mask_mode}")
    block = 1 if mode == "1x1" else 2
    row_positions = list(range(0, max(1, height - block + 1), block))
    col_positions = list(range(0, max(1, width - block + 1), block))
    if not row_positions or row_positions[-1] != max(0, height - block):
        row_positions.append(max(0, height - block))
    if not col_positions or col_positions[-1] != max(0, width - block):
        col_positions.append(max(0, width - block))
    row_positions = sorted(dict.fromkeys(int(v) for v in row_positions))
    col_positions = sorted(dict.fromkeys(int(v) for v in col_positions))
    regions: List[L3RegionSpec] = []
    region_id = 0
    for row_idx, row_start in enumerate(row_positions):
        row_end = min(height, row_start + block)
        for col_idx, col_start in enumerate(col_positions):
            col_end = min(width, col_start + block)
            regions.append(
                L3RegionSpec(
                    region_id=int(region_id),
                    row_index=int(row_idx),
                    col_index=int(col_idx),
                    row_start=int(row_start),
                    row_end=int(row_end),
                    col_start=int(col_start),
                    col_end=int(col_end),
                )
            )
            region_id += 1
    return regions


def _region_values_to_grid(regions: Sequence[L3RegionSpec], values: Sequence[float]) -> np.ndarray:
    if not regions:
        return np.zeros((0, 0), dtype=np.float64)
    n_rows = max(region.row_index for region in regions) + 1
    n_cols = max(region.col_index for region in regions) + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for region, value in zip(regions, values):
        grid[int(region.row_index), int(region.col_index)] = float(value)
    return grid


def _build_l3_mask_tensor(regions: Sequence[L3RegionSpec], height: int, width: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros((len(regions), height, width), dtype=torch.bool, device=device)
    for idx, region in enumerate(regions):
        masks[idx, region.row_start:region.row_end, region.col_start:region.col_end] = True
    return masks


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


def _select_probe_samples_from_candidates(
    df_candidates: pd.DataFrame,
    samples_per_probe: int,
    num_bins: int,
) -> pd.DataFrame:
    if df_candidates.empty:
        return df_candidates.iloc[:0].copy()
    ordered = df_candidates.sort_values(["similarity_public_or_initial", "sample_id"], kind="stable").reset_index(drop=True)
    ordered["candidate_bin"] = _assign_bins_from_values(
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
        selected = _select_probe_samples_from_candidates(
            df_candidates=df_candidates,
            samples_per_probe=int(samples_per_probe),
            num_bins=int(num_bins),
        )
        rows.extend(selected.to_dict("records"))
    if not rows:
        raise RuntimeError("No sample-probe pairs were generated.")
    df_pairs = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "probe_id"], keep="first").reset_index(drop=True)
    df_pairs = _rebalance_global_pairs(df_pairs, max_pairs=max_pairs)
    df_pairs["similarity_bin"] = _assign_bins_from_values(
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
    spec: ExperimentSpec,
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


def _apply_probe_input_mask(input_t: torch.Tensor, probe_input_mask: torch.Tensor | None) -> torch.Tensor:
    if probe_input_mask is None:
        return input_t
    mask = probe_input_mask.to(device=input_t.device, dtype=torch.bool)
    while mask.ndim < input_t.ndim:
        mask = mask.unsqueeze(1)
    return input_t.masked_fill(mask, 0.0)


def _apply_l3_spatial_mask(s2_p: torch.Tensor, l3_spatial_mask: torch.Tensor | None) -> torch.Tensor:
    if l3_spatial_mask is None:
        return s2_p
    mask = l3_spatial_mask.to(device=s2_p.device, dtype=torch.bool)
    if mask.ndim != 3:
        raise ValueError(f"l3_spatial_mask must have shape [B, H, W], got {tuple(mask.shape)}")
    return s2_p.masked_fill(mask.unsqueeze(1), 0.0)


def run_maskable_dms_rollout(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    delay_steps: int,
    *,
    stsp_mode: str,
    readout_step: int,
    phase_reset: bool = True,
    probe_input_mask: torch.Tensor | None = None,
    l3_spatial_mask: torch.Tensor | None = None,
    record_sample_s2p: bool = False,
    record_probe_s2p: bool = False,
) -> Dict[str, object]:
    if sample_spikes.ndim != 5 or probe_spikes.ndim != 5:
        raise ValueError("sample_spikes and probe_spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = sample_spikes.shape
    if int(probe_spikes.shape[0]) != int(batch_size):
        raise ValueError("sample_spikes and probe_spikes must share batch size")
    prepare_network_state(net, batch_size, channels, height, width)
    current_time = 0
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    readout_snapshot = None
    sample_s2p_trace: List[torch.Tensor] = []
    probe_s2p_trace: List[torch.Tensor] = []

    def step_network(input_t: torch.Tensor, *, phase: str, phase_step: int, force_l3_time: int | None = None) -> None:
        nonlocal current_time, readout_snapshot
        input_local = _apply_probe_input_mask(input_t, probe_input_mask if phase == "probe" else None)
        s1, _ = net.layer1.forward_step(input_local, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        if phase == "sample" and record_sample_s2p:
            sample_s2p_trace.append(s2_p.detach().cpu().to(torch.float32))
        if phase == "probe" and record_probe_s2p:
            probe_s2p_trace.append(s2_p.detach().cpu().to(torch.float32))
        s2_p_for_l3 = _apply_l3_spatial_mask(s2_p, l3_spatial_mask if phase == "probe" else None)
        l3_time = current_time if force_l3_time is None else force_l3_time
        _, m3 = net.layer3.forward_step(
            s2_p_for_l3,
            l3_time,
            training=False,
            monitor=(phase == "probe" and int(phase_step) == int(readout_step)),
            stsp_mode=stsp_mode,
        )
        if phase == "probe" and int(phase_step) == int(readout_step):
            if "v_mem_snapshot" not in m3:
                raise RuntimeError("Layer-3 readout snapshot was not captured.")
            readout_snapshot = m3["v_mem_snapshot"].detach().cpu().to(torch.float32)
        current_time += 1

    with torch.no_grad():
        for t_step in range(int(sample_spikes.shape[1])):
            step_network(sample_spikes[:, t_step, ...], phase="sample", phase_step=t_step)
        for _ in range(int(delay_steps)):
            step_network(zero_input, phase="delay", phase_step=0)
        net.layer3.reset_decision_state()
        if phase_reset:
            net.layer3.v_mem.fill_(net.layer3.V_L)
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)
        for t_step in range(int(probe_spikes.shape[1])):
            step_network(probe_spikes[:, t_step, ...], phase="probe", phase_step=t_step, force_l3_time=t_step if phase_reset else None)

    if readout_snapshot is None:
        raise RuntimeError("Requested probe readout snapshot was not produced.")
    flat_times = net.layer3.firing_times
    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    prediction_probe = (min_indices // net.layer3.neurons_per_class).long()
    prediction_probe[~has_fired] = -1
    first_fire_t_probe = min_times.clone()
    first_fire_t_probe[~has_fired] = -1
    first_fire_t_probe = first_fire_t_probe.to(torch.long)
    return {
        "readout_snapshot": readout_snapshot,
        "readout_step": int(readout_step),
        "predictions": {
            "prediction_probe": prediction_probe.detach().cpu(),
            "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        },
        "sample_s2p_trace": None if not sample_s2p_trace else torch.stack(sample_s2p_trace, dim=0),
        "probe_s2p_trace": None if not probe_s2p_trace else torch.stack(probe_s2p_trace, dim=0),
    }


def extract_grouped_voltage_vector(net, voltage_snapshot: torch.Tensor) -> np.ndarray:
    grouped = net.layer3.get_grouped_voltage(voltage_snapshot.to(torch.float32))
    return grouped.mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)


def run_reference_mode_batch(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay_steps: int,
    stsp_mode: str,
    readout_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = run_dms_snapshot_rollout(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(delay_steps),
            stsp_mode=str(stsp_mode),
            phase_reset=True,
            intervention_plan=None,
            readout_step=int(readout_step),
            snapshot_state_names=("v_mem",),
            record_full_trace_state_names=(),
        )
    pred = out["predictions"]["prediction_probe"].numpy().astype(np.int64, copy=False)
    vector = extract_grouped_voltage_vector(net, out["readout_snapshots"]["layer3"]["v_mem"])
    return pred, vector


def extract_s2p_feature_aggregate(
    net,
    spikes: torch.Tensor,
    *,
    stsp_mode: str,
    temporal_pool: str,
) -> torch.Tensor:
    if spikes.ndim != 5:
        raise ValueError("spikes must have shape [B, T, C, H, W]")
    batch_size, _, channels, height, width = spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    trace: List[torch.Tensor] = []
    current_time = 0
    with torch.no_grad():
        for t_step in range(int(spikes.shape[1])):
            s1, _ = net.layer1.forward_step(spikes[:, t_step, ...], current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s1_p = net.pool1(s1.float())
            s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s2_p = net.pool2(s2.float())
            trace.append(s2_p.detach().cpu().to(torch.float32))
            current_time += 1
    stacked = torch.stack(trace, dim=0)
    if str(temporal_pool).strip().lower() == "mean":
        return stacked.mean(dim=0)
    if str(temporal_pool).strip().lower() == "sum":
        return stacked.sum(dim=0)
    raise ValueError(f"Unsupported temporal pool: {temporal_pool}")


def _safe_vector_similarity(pred_vec: np.ndarray, target_vec: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_vec, dtype=np.float64)
    target = np.asarray(target_vec, dtype=np.float64)
    metrics = {
        "cosine": _cosine_similarity_1d(pred, target),
        "pearson": float("nan"),
        "spearman": float("nan"),
    }
    if pred.size >= 2 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        metrics["pearson"] = float(stats.pearsonr(pred, target).statistic)
        metrics["spearman"] = float(stats.spearmanr(pred, target).statistic)
    return metrics


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64), ord=2))


def _build_blank_sample(spec: ExperimentSpec, channels: int, height: int, width: int) -> torch.Tensor:
    return torch.zeros((spec.sample_steps, channels, height, width), dtype=torch.float32)


def compute_probe_evidence_matrices(
    net,
    probe_spikes_single: torch.Tensor,
    blank_sample_single: torch.Tensor,
    *,
    delay_steps: int,
    readout_step: int,
    l1_patches: Sequence[PatchSpec],
    l3_regions: Sequence[L3RegionSpec],
    device: torch.device,
    mask_batch_size: int,
) -> Dict[str, object]:
    probe_spikes = probe_spikes_single.unsqueeze(0).to(device=device, dtype=torch.float32)
    blank_sample = blank_sample_single.unsqueeze(0).to(device=device, dtype=torch.float32)
    base = run_maskable_dms_rollout(
        net=net,
        sample_spikes=blank_sample,
        probe_spikes=probe_spikes,
        delay_steps=delay_steps,
        stsp_mode="static_frozen",
        readout_step=readout_step,
        phase_reset=True,
    )
    v_orig = extract_grouped_voltage_vector(net, base["readout_snapshot"])[0]
    probe_height = int(probe_spikes.shape[-2])
    probe_width = int(probe_spikes.shape[-1])
    l1_masks = torch.zeros((len(l1_patches), probe_height, probe_width), dtype=torch.bool, device=device)
    for idx, patch in enumerate(l1_patches):
        l1_masks[idx, patch.row_start:patch.row_end, patch.col_start:patch.col_end] = True
    c_l1_rows: List[np.ndarray] = []
    for start in range(0, len(l1_patches), int(mask_batch_size)):
        end = min(len(l1_patches), start + int(mask_batch_size))
        current_masks = l1_masks[start:end]
        batch_probe = probe_spikes.repeat(end - start, 1, 1, 1, 1)
        batch_sample = blank_sample.repeat(end - start, 1, 1, 1, 1)
        out = run_maskable_dms_rollout(
            net=net,
            sample_spikes=batch_sample,
            probe_spikes=batch_probe,
            delay_steps=delay_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
            phase_reset=True,
            probe_input_mask=current_masks,
        )
        masked_vectors = extract_grouped_voltage_vector(net, out["readout_snapshot"])
        c_l1_rows.append(v_orig[None, :] - masked_vectors)
    c_l1 = np.concatenate(c_l1_rows, axis=0) if c_l1_rows else np.zeros((0, len(v_orig)), dtype=np.float64)

    feature_height = int(net.layer3.kernels.shape[-2])
    feature_width = int(net.layer3.kernels.shape[-1])
    l3_masks = _build_l3_mask_tensor(l3_regions, feature_height, feature_width, device=device)
    c_l3_rows: List[np.ndarray] = []
    for start in range(0, len(l3_regions), int(mask_batch_size)):
        end = min(len(l3_regions), start + int(mask_batch_size))
        current_masks = l3_masks[start:end]
        batch_probe = probe_spikes.repeat(end - start, 1, 1, 1, 1)
        batch_sample = blank_sample.repeat(end - start, 1, 1, 1, 1)
        out = run_maskable_dms_rollout(
            net=net,
            sample_spikes=batch_sample,
            probe_spikes=batch_probe,
            delay_steps=delay_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
            phase_reset=True,
            l3_spatial_mask=current_masks,
        )
        masked_vectors = extract_grouped_voltage_vector(net, out["readout_snapshot"])
        c_l3_rows.append(v_orig[None, :] - masked_vectors)
    c_l3 = np.concatenate(c_l3_rows, axis=0) if c_l3_rows else np.zeros((0, len(v_orig)), dtype=np.float64)

    if np.allclose(c_l1, 0.0):
        warnings.warn("C_L1 is all zeros for a probe; check checkpoint or masking semantics.", RuntimeWarning)
    if np.allclose(c_l3, 0.0):
        warnings.warn("C_L3 is all zeros for a probe; check checkpoint or masking semantics.", RuntimeWarning)
    return {
        "v_orig": v_orig,
        "c_l1": c_l1,
        "c_l3": c_l3,
    }


def summarize_similarity_bins(df_pairs: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for bin_label, sub in df_pairs.groupby("similarity_bin", sort=False):
        rows.append(
            {
                "similarity_bin": str(bin_label),
                "count": int(len(sub)),
                "min_similarity": _safe_float(sub["similarity_public_or_initial"].min()),
                "max_similarity": _safe_float(sub["similarity_public_or_initial"].max()),
                "mean_similarity": _safe_float(sub["similarity_public_or_initial"].mean()),
            }
        )
    return rows


def select_case_pairs(df_pairs: pd.DataFrame, save_case_count: int) -> pd.DataFrame:
    if df_pairs.empty:
        return df_pairs.copy()
    selected_ids: List[int] = []
    for _, sub in df_pairs.groupby("probe_id", sort=False):
        if len(sub) < 2:
            continue
        probe_sorted = sub.sort_values(["similarity_public_or_initial", "bias_mag"], ascending=[True, False], kind="stable")
        best_pair = None
        best_score = None
        rows = list(probe_sorted.itertuples(index=False))
        for idx_a in range(len(rows)):
            for idx_b in range(idx_a + 1, len(rows)):
                row_a = rows[idx_a]
                row_b = rows[idx_b]
                if int(row_a.k_star) == int(row_b.k_star):
                    continue
                sim_gap = abs(float(row_a.similarity_public_or_initial) - float(row_b.similarity_public_or_initial))
                score = (sim_gap, -float(row_a.bias_mag + row_b.bias_mag))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (int(row_a.pair_id), int(row_b.pair_id))
        if best_pair is not None:
            for pair_id in best_pair:
                if pair_id not in selected_ids:
                    selected_ids.append(pair_id)
                if len(selected_ids) >= int(save_case_count):
                    break
        if len(selected_ids) >= int(save_case_count):
            break
    if len(selected_ids) < int(save_case_count):
        fallback = df_pairs.copy()
        fallback["prediction_gap"] = (fallback["direction_match_l1"] != fallback["direction_match_l3"]).astype(np.int64)
        fallback = fallback.sort_values(["prediction_gap", "bias_mag"], ascending=[False, False], kind="stable")
        for pair_id in fallback["pair_id"].astype(int).tolist():
            if pair_id not in selected_ids:
                selected_ids.append(pair_id)
            if len(selected_ids) >= int(save_case_count):
                break
    return df_pairs[df_pairs["pair_id"].isin(selected_ids)].copy().sort_values(["probe_id", "pair_id"], kind="stable").reset_index(drop=True)


def plot_similarity_distribution(df_pairs: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    sims = df_pairs["similarity_public_or_initial"].to_numpy(dtype=np.float64, copy=False)
    axes[0].hist(sims, bins=min(20, max(5, len(df_pairs) // 2)), color=COLOR_ACCENT_BLUE, alpha=ALPHA_BAR, edgecolor="white")
    axes[0].set_xlabel("Public similarity")
    axes[0].set_ylabel("Pair count")
    axes[0].set_title("Selected pair similarity distribution")
    axes[0].grid(alpha=GRID_ALPHA)

    bin_summary = (
        df_pairs.groupby("similarity_bin", sort=False)
        .agg(count=("pair_id", "size"))
        .reset_index()
    )
    x = np.arange(len(bin_summary), dtype=np.float64)
    axes[1].bar(x, bin_summary["count"].to_numpy(dtype=np.float64), color=COLOR_ACCENT_BROWN)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_summary["similarity_bin"].astype(str).tolist(), rotation=0)
    axes[1].set_xlabel("Similarity bin")
    axes[1].set_ylabel("Pair count")
    axes[1].set_title("Per-bin coverage")
    axes[1].grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_case_studies(
    df_cases: pd.DataFrame,
    images: torch.Tensor,
    probe_c_l1: np.ndarray,
    probe_c_l3: np.ndarray,
    pair_overlap_l1: np.ndarray,
    pair_overlap_l3: np.ndarray,
    pair_vectors: Mapping[str, np.ndarray],
    probe_id_to_index: Mapping[int, int],
    pair_id_to_index: Mapping[int, int],
    l1_patches: Sequence[PatchSpec],
    l3_regions: Sequence[L3RegionSpec],
) -> plt.Figure:
    apply_publication_style()
    n_cases = max(1, len(df_cases))
    fig, axes = plt.subplots(n_cases, 7, figsize=case_grid_figsize(n_cases, width=19.0, row_height=3.6), squeeze=False)
    for row_idx, row in enumerate(df_cases.itertuples(index=False)):
        probe_index = int(probe_id_to_index[int(row.probe_id)])
        pair_index = int(pair_id_to_index[int(row.pair_id)])
        k_star = int(row.k_star)
        probe_image = images[int(row.probe_id), 0].numpy().astype(np.float64, copy=False)
        c_l1_map = project_patch_values_to_image(
            height=probe_image.shape[0],
            width=probe_image.shape[1],
            patches=l1_patches,
            values=probe_c_l1[probe_index, :, k_star],
        )
        o_l1_map = project_patch_values_to_image(
            height=probe_image.shape[0],
            width=probe_image.shape[1],
            patches=l1_patches,
            values=pair_overlap_l1[pair_index],
        )
        c_l3_grid = _region_values_to_grid(l3_regions, probe_c_l3[probe_index, :, k_star])
        o_l3_grid = _region_values_to_grid(l3_regions, pair_overlap_l3[pair_index])
        delta_v = pair_vectors["delta_v"][pair_index]
        p_l1 = pair_vectors["P_L1"][pair_index]
        p_l3 = pair_vectors["P_L3"][pair_index]
        ax = axes[row_idx]
        ax[0].imshow(probe_image, cmap=CMAP_IMAGE_GRAY, vmin=0.0, vmax=max(1e-6, float(probe_image.max())))
        ax[0].set_title(f"Probe {int(row.probe_id)}\nlabel={int(row.probe_label)}")
        ax[0].axis("off")

        im1 = ax[1].imshow(c_l1_map, cmap=CMAP_ACTIVATION)
        ax[1].set_title(f"C_L1 @ k*={k_star}")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(o_l1_map, cmap=CMAP_OVERLAP)
        ax[2].set_title("O_L1")
        ax[2].axis("off")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        im3 = ax[3].imshow(c_l3_grid, cmap=CMAP_ACTIVATION)
        ax[3].set_title(f"C_L3 @ k*={k_star}")
        ax[3].axis("off")
        fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        im4 = ax[4].imshow(o_l3_grid, cmap=CMAP_OVERLAP)
        ax[4].set_title("O_L3")
        ax[4].axis("off")
        fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

        class_x = np.arange(len(delta_v), dtype=np.float64)
        width = 0.38
        ax[5].bar(class_x - width / 2.0, delta_v, width=width, color=COLOR_ACCENT_BLUE, label="delta_v")
        ax[5].bar(class_x + width / 2.0, p_l1, width=width, color=LAYER_ROUTE_COLORS["L1"], label="P_L1")
        ax[5].set_title("delta_v vs P_L1")
        ax[5].set_xlabel("Class")
        ax[5].grid(alpha=GRID_ALPHA, axis="y")
        if row_idx == 0:
            apply_standard_legend(ax[5])

        ax[6].bar(class_x - width / 2.0, delta_v, width=width, color=COLOR_ACCENT_BLUE, label="delta_v")
        ax[6].bar(class_x + width / 2.0, p_l3, width=width, color=LAYER_ROUTE_COLORS["L3"], label="P_L3")
        ax[6].set_title(
            f"P_L3 | sim={float(row.similarity_public_or_initial):.2f}\n"
            f"k_hat=({int(row.k_hat_l1)},{int(row.k_hat_l3)})"
        )
        ax[6].set_xlabel("Class")
        ax[6].grid(alpha=GRID_ALPHA, axis="y")
        if row_idx == 0:
            apply_standard_legend(ax[6])
    fig.tight_layout()
    return fig


def plot_direction_accuracy_compare(df_pairs: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    overall = pd.DataFrame(
        {
            "route": ["L1", "L3"],
            "accuracy": [
                float(df_pairs["direction_match_l1"].mean()),
                float(df_pairs["direction_match_l3"].mean()),
            ],
        }
    )
    axes[0].bar(overall["route"], overall["accuracy"], color=[LAYER_ROUTE_COLORS["L1"], LAYER_ROUTE_COLORS["L3"]])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Direction accuracy")
    axes[0].set_title("Overall direction match")
    axes[0].grid(alpha=GRID_ALPHA, axis="y")

    bin_summary = (
        df_pairs.groupby("similarity_bin", sort=False)[["direction_match_l1", "direction_match_l3"]]
        .mean()
        .reset_index()
    )
    x = np.arange(len(bin_summary), dtype=np.float64)
    axes[1].plot(x, bin_summary["direction_match_l1"].to_numpy(dtype=np.float64), marker="o", color=LAYER_ROUTE_COLORS["L1"], label="L1")
    axes[1].plot(x, bin_summary["direction_match_l3"].to_numpy(dtype=np.float64), marker="o", color=LAYER_ROUTE_COLORS["L3"], label="L3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_summary["similarity_bin"].astype(str).tolist())
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Similarity bin")
    axes[1].set_ylabel("Direction accuracy")
    axes[1].set_title("Direction accuracy by similarity")
    axes[1].grid(alpha=GRID_ALPHA)
    apply_standard_legend(axes[1])
    fig.tight_layout()
    return fig


def plot_vector_similarity_compare(df_pairs: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL)
    metrics = [
        ("cos_l1_delta", "cos_l3_delta", "Cosine"),
        ("pearson_l1_delta", "pearson_l3_delta", "Pearson"),
        ("spearman_l1_delta", "spearman_l3_delta", "Spearman"),
    ]
    for ax, (l1_col, l3_col, title) in zip(axes, metrics):
        data = [
            df_pairs[l1_col].to_numpy(dtype=np.float64, copy=False),
            df_pairs[l3_col].to_numpy(dtype=np.float64, copy=False),
        ]
        box = ax.boxplot(data, tick_labels=["L1", "L3"], patch_artist=True)
        for patch, color in zip(box["boxes"], [LAYER_ROUTE_COLORS["L1"], LAYER_ROUTE_COLORS["L3"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_similarity_binned_compare(df_pairs: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    df_bin = (
        df_pairs.groupby("similarity_bin", sort=False)
        .agg(
            mean_similarity=("similarity_public_or_initial", "mean"),
            direction_accuracy_l1=("direction_match_l1", "mean"),
            direction_accuracy_l3=("direction_match_l3", "mean"),
            cosine_l1=("cos_l1_delta", "mean"),
            cosine_l3=("cos_l3_delta", "mean"),
        )
        .reset_index()
    )
    x = np.arange(len(df_bin), dtype=np.float64)
    axes[0].plot(x, df_bin["direction_accuracy_l1"].to_numpy(dtype=np.float64), marker="o", color=LAYER_ROUTE_COLORS["L1"], label="L1")
    axes[0].plot(x, df_bin["direction_accuracy_l3"].to_numpy(dtype=np.float64), marker="o", color=LAYER_ROUTE_COLORS["L3"], label="L3")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_bin["similarity_bin"].astype(str).tolist())
    axes[0].set_xlabel("Similarity bin")
    axes[0].set_ylabel("Direction accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Binned direction accuracy")
    axes[0].grid(alpha=GRID_ALPHA)
    apply_standard_legend(axes[0])

    axes[1].plot(x, df_bin["cosine_l1"].to_numpy(dtype=np.float64), marker="o", color=LAYER_ROUTE_COLORS["L1"], label="L1")
    axes[1].plot(x, df_bin["cosine_l3"].to_numpy(dtype=np.float64), marker="o", color=LAYER_ROUTE_COLORS["L3"], label="L3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_bin["similarity_bin"].astype(str).tolist())
    axes[1].set_xlabel("Similarity bin")
    axes[1].set_ylabel("Mean cosine(P, delta_v)")
    axes[1].set_title("Binned vector alignment")
    axes[1].grid(alpha=GRID_ALPHA)
    apply_standard_legend(axes[1])
    fig.tight_layout()
    return fig


def plot_vector_scatter_compare(df_pairs: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    sims = df_pairs["similarity_public_or_initial"].to_numpy(dtype=np.float64, copy=False)
    axes[0].scatter(sims, df_pairs["cos_l1_delta"].to_numpy(dtype=np.float64), s=24, alpha=ALPHA_SCATTER_LIGHT, color=LAYER_ROUTE_COLORS["L1"], label="L1")
    axes[0].scatter(sims, df_pairs["cos_l3_delta"].to_numpy(dtype=np.float64), s=24, alpha=ALPHA_SCATTER_LIGHT, color=LAYER_ROUTE_COLORS["L3"], label="L3")
    axes[0].set_xlabel("Public similarity")
    axes[0].set_ylabel("Cosine(P, delta_v)")
    axes[0].set_title("Vector alignment vs similarity")
    axes[0].grid(alpha=GRID_ALPHA)
    apply_standard_legend(axes[0])

    axes[1].scatter(
        df_pairs["bias_mag"].to_numpy(dtype=np.float64),
        df_pairs["cos_l1_delta"].to_numpy(dtype=np.float64),
        s=24,
        alpha=ALPHA_SCATTER_LIGHT,
        color=LAYER_ROUTE_COLORS["L1"],
        label="L1",
    )
    axes[1].scatter(
        df_pairs["bias_mag"].to_numpy(dtype=np.float64),
        df_pairs["cos_l3_delta"].to_numpy(dtype=np.float64),
        s=24,
        alpha=ALPHA_SCATTER_LIGHT,
        color=LAYER_ROUTE_COLORS["L3"],
        label="L3",
    )
    axes[1].set_xlabel("Bias magnitude")
    axes[1].set_ylabel("Cosine(P, delta_v)")
    axes[1].set_title("Vector alignment vs bias magnitude")
    axes[1].grid(alpha=GRID_ALPHA)
    apply_standard_legend(axes[1])
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STSP-induced bias direction mask experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--l1-patch-size", type=int, default=DEFAULT_L1_PATCH_SIZE)
    parser.add_argument("--l1-stride", type=int, default=DEFAULT_L1_STRIDE)
    parser.add_argument("--l3-mask-mode", type=str, default=DEFAULT_L3_MASK_MODE, choices=["1x1", "2x2"])
    parser.add_argument("--l3-temporal-pool", type=str, default=DEFAULT_L3_TEMPORAL_POOL, choices=["mean", "sum"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_probes) <= 0:
        raise ValueError("--max-probes must be positive.")
    if int(args.samples_per_probe) <= 0:
        raise ValueError("--samples-per-probe must be positive.")
    if int(args.max_pairs) <= 0:
        raise ValueError("--max-pairs must be positive.")
    if int(args.num_sim_bins) <= 1:
        raise ValueError("--num-sim-bins must be greater than 1.")
    if int(args.save_case_count) <= 0:
        raise ValueError("--save-case-count must be positive.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe durations must resolve to positive steps.")
    delay_steps = int(round((float(args.delay_ms) * ms) / spec.dt))
    if delay_steps < 0:
        raise ValueError("--delay-ms must be non-negative.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    output_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    dataset = _load_dataset(dataset_root=args.dataset_root, split=args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = len(set(int(label) for label in labels.tolist()))
    class_index = build_class_index(dataset, num_classes=num_classes)
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms)),
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    df_pairs = build_pair_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(args.num_sim_bins),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
    )

    voltage_dynamic = np.zeros((len(df_pairs), num_classes), dtype=np.float64)
    voltage_static = np.zeros((len(df_pairs), num_classes), dtype=np.float64)
    pred_dynamic = np.full(len(df_pairs), -1, dtype=np.int64)
    pred_static = np.full(len(df_pairs), -1, dtype=np.int64)
    for start in tqdm(range(0, len(df_pairs), int(args.batch_size)), desc="BaselineDynamicStatic"):
        batch = df_pairs.iloc[start:start + int(args.batch_size)].copy().reset_index(drop=True)
        sample_spikes, probe_spikes = prepare_pair_spike_batch(images=images, batch_df=batch, encoder=encoder, spec=spec, device=device)
        pred_d, vec_d = run_reference_mode_batch(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=delay_steps,
            stsp_mode="dynamic",
            readout_step=readout_step,
        )
        pred_s, vec_s = run_reference_mode_batch(
            net=net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=delay_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
        )
        pair_ids = batch["pair_id"].to_numpy(dtype=np.int64, copy=False)
        voltage_dynamic[pair_ids] = vec_d
        voltage_static[pair_ids] = vec_s
        pred_dynamic[pair_ids] = pred_d
        pred_static[pair_ids] = pred_s

    delta_v = (voltage_dynamic - voltage_dynamic.mean(axis=1, keepdims=True)) - (
        voltage_static - voltage_static.mean(axis=1, keepdims=True)
    )
    k_star = np.argmax(delta_v, axis=1).astype(np.int64, copy=False)
    bias_mag = np.linalg.norm(delta_v, axis=1).astype(np.float64, copy=False)

    probe_height = int(images.shape[-2])
    probe_width = int(images.shape[-1])
    encoder_probe = encoder.forward(images[[0]].to(device=device, dtype=torch.float32))
    channels = int(encoder_probe.shape[2])
    blank_sample = _build_blank_sample(spec, channels=channels, height=probe_height, width=probe_width)
    l1_patches = build_patch_grid(
        height=probe_height,
        width=probe_width,
        patch_size=int(args.l1_patch_size),
        stride=int(args.l1_stride),
    )
    l3_regions = build_l3_regions(
        height=int(net.layer3.kernels.shape[-2]),
        width=int(net.layer3.kernels.shape[-1]),
        mask_mode=str(args.l3_mask_mode),
    )

    sample_spike_cache: Dict[int, torch.Tensor] = {}
    probe_spike_cache: Dict[int, torch.Tensor] = {}
    sample_l3_feature_cache: Dict[int, torch.Tensor] = {}
    probe_l3_feature_cache: Dict[int, torch.Tensor] = {}
    probe_evidence_cache: Dict[int, Dict[str, object]] = {}

    def get_sample_spikes(image_id: int) -> torch.Tensor:
        key = int(image_id)
        if key not in sample_spike_cache:
            sample_spike_cache[key] = _stack_encoded_batch([key], images=images, encoder=encoder, steps=spec.sample_steps, device=device)[0].cpu()
        return sample_spike_cache[key]

    def get_probe_spikes(image_id: int) -> torch.Tensor:
        key = int(image_id)
        if key not in probe_spike_cache:
            probe_spike_cache[key] = _stack_encoded_batch([key], images=images, encoder=encoder, steps=spec.probe_steps, device=device)[0].cpu()
        return probe_spike_cache[key]

    def get_sample_l3_feature(image_id: int) -> torch.Tensor:
        key = int(image_id)
        if key not in sample_l3_feature_cache:
            spikes = get_sample_spikes(key).unsqueeze(0).to(device=device, dtype=torch.float32)
            sample_l3_feature_cache[key] = extract_s2p_feature_aggregate(
                net=net,
                spikes=spikes,
                stsp_mode="static_frozen",
                temporal_pool=str(args.l3_temporal_pool),
            )[0]
        return sample_l3_feature_cache[key]

    def get_probe_l3_feature(image_id: int) -> torch.Tensor:
        key = int(image_id)
        if key not in probe_l3_feature_cache:
            spikes = get_probe_spikes(key).unsqueeze(0).to(device=device, dtype=torch.float32)
            probe_l3_feature_cache[key] = extract_s2p_feature_aggregate(
                net=net,
                spikes=spikes,
                stsp_mode="static_frozen",
                temporal_pool=str(args.l3_temporal_pool),
            )[0]
        return probe_l3_feature_cache[key]

    unique_probe_ids = sorted(df_pairs["probe_id"].astype(int).unique().tolist())
    probe_c_l1_rows: List[np.ndarray] = []
    probe_c_l3_rows: List[np.ndarray] = []
    probe_v_orig_rows: List[np.ndarray] = []
    probe_meta_rows: List[Dict[str, object]] = []
    for probe_id in tqdm(unique_probe_ids, desc="ProbeEvidence"):
        evidence = compute_probe_evidence_matrices(
            net=net,
            probe_spikes_single=get_probe_spikes(probe_id),
            blank_sample_single=blank_sample,
            delay_steps=delay_steps,
            readout_step=readout_step,
            l1_patches=l1_patches,
            l3_regions=l3_regions,
            device=device,
            mask_batch_size=max(1, int(args.batch_size)),
        )
        probe_evidence_cache[int(probe_id)] = evidence
        probe_c_l1_rows.append(np.asarray(evidence["c_l1"], dtype=np.float64))
        probe_c_l3_rows.append(np.asarray(evidence["c_l3"], dtype=np.float64))
        probe_v_orig_rows.append(np.asarray(evidence["v_orig"], dtype=np.float64))
        probe_meta_rows.append(
            {
                "probe_id": int(probe_id),
                "probe_label": int(labels[int(probe_id)]),
                "num_l1_regions": int(len(l1_patches)),
                "num_l3_regions": int(len(l3_regions)),
                "l1_all_zero": int(np.allclose(evidence["c_l1"], 0.0)),
                "l3_all_zero": int(np.allclose(evidence["c_l3"], 0.0)),
            }
        )

    pair_overlap_l1 = np.zeros((len(df_pairs), len(l1_patches)), dtype=np.float64)
    pair_overlap_l3 = np.zeros((len(df_pairs), len(l3_regions)), dtype=np.float64)
    pred_l1 = np.zeros((len(df_pairs), num_classes), dtype=np.float64)
    pred_l3 = np.zeros((len(df_pairs), num_classes), dtype=np.float64)
    k_hat_l1 = np.zeros(len(df_pairs), dtype=np.int64)
    k_hat_l3 = np.zeros(len(df_pairs), dtype=np.int64)

    pair_rows: List[Dict[str, object]] = []
    for row in tqdm(df_pairs.itertuples(index=False), total=len(df_pairs), desc="PairPrediction"):
        pair_id = int(row.pair_id)
        sample_id = int(row.sample_id)
        probe_id = int(row.probe_id)
        sample_spikes = get_sample_spikes(sample_id)
        probe_spikes = get_probe_spikes(probe_id)
        sample_l1_feature = sample_spikes.mean(dim=0)
        probe_l1_feature = probe_spikes.mean(dim=0)
        for patch_idx, patch in enumerate(l1_patches):
            pair_overlap_l1[pair_id, patch_idx] = _cosine_similarity_tensor(
                sample_l1_feature[:, patch.row_start:patch.row_end, patch.col_start:patch.col_end],
                probe_l1_feature[:, patch.row_start:patch.row_end, patch.col_start:patch.col_end],
            )

        sample_l3_feature = get_sample_l3_feature(sample_id)
        probe_l3_feature = get_probe_l3_feature(probe_id)
        for region_idx, region in enumerate(l3_regions):
            pair_overlap_l3[pair_id, region_idx] = _cosine_similarity_tensor(
                sample_l3_feature[:, region.row_start:region.row_end, region.col_start:region.col_end],
                probe_l3_feature[:, region.row_start:region.row_end, region.col_start:region.col_end],
            )

        c_l1 = np.asarray(probe_evidence_cache[probe_id]["c_l1"], dtype=np.float64)
        c_l3 = np.asarray(probe_evidence_cache[probe_id]["c_l3"], dtype=np.float64)
        pred_l1[pair_id] = pair_overlap_l1[pair_id] @ c_l1
        pred_l3[pair_id] = pair_overlap_l3[pair_id] @ c_l3
        k_hat_l1[pair_id] = int(np.argmax(pred_l1[pair_id]))
        k_hat_l3[pair_id] = int(np.argmax(pred_l3[pair_id]))

        sim_l1 = _safe_vector_similarity(pred_l1[pair_id], delta_v[pair_id])
        sim_l3 = _safe_vector_similarity(pred_l3[pair_id], delta_v[pair_id])
        pair_rows.append(
            {
                "pair_id": pair_id,
                "probe_id": probe_id,
                "sample_id": sample_id,
                "probe_label": int(row.probe_label),
                "sample_label": int(row.sample_label),
                "similarity_public_or_initial": float(row.similarity_public_or_initial),
                "similarity_bin": str(row.similarity_bin),
                "similarity_bin_index": int(row.similarity_bin_index),
                "pred_label_dynamic": int(pred_dynamic[pair_id]),
                "pred_label_static": int(pred_static[pair_id]),
                "bias_mag": float(bias_mag[pair_id]),
                "delta_v_norm": float(_norm(delta_v[pair_id])),
                "P_l1_norm": float(_norm(pred_l1[pair_id])),
                "P_l3_norm": float(_norm(pred_l3[pair_id])),
                "k_star": int(k_star[pair_id]),
                "k_hat_l1": int(k_hat_l1[pair_id]),
                "k_hat_l3": int(k_hat_l3[pair_id]),
                "direction_match_l1": int(k_hat_l1[pair_id] == k_star[pair_id]),
                "direction_match_l3": int(k_hat_l3[pair_id] == k_star[pair_id]),
                "cos_l1_delta": float(sim_l1["cosine"]),
                "cos_l3_delta": float(sim_l3["cosine"]),
                "pearson_l1_delta": float(sim_l1["pearson"]),
                "pearson_l3_delta": float(sim_l3["pearson"]),
                "spearman_l1_delta": float(sim_l1["spearman"]),
                "spearman_l3_delta": float(sim_l3["spearman"]),
                "mean_overlap_l1": float(np.mean(pair_overlap_l1[pair_id])),
                "mean_overlap_l3": float(np.mean(pair_overlap_l3[pair_id])),
            }
        )

    df_results = pd.DataFrame(pair_rows).sort_values(["pair_id"], kind="stable").reset_index(drop=True)
    df_cases = select_case_pairs(df_results, save_case_count=int(args.save_case_count))

    probe_id_to_index = {int(probe_id): idx for idx, probe_id in enumerate(unique_probe_ids)}
    pair_id_to_index = {int(pair_id): idx for idx, pair_id in enumerate(df_results["pair_id"].astype(int).tolist())}
    bin_summary_df = (
        df_results.groupby("similarity_bin", sort=False)
        .agg(
            count=("pair_id", "size"),
            mean_similarity=("similarity_public_or_initial", "mean"),
            direction_accuracy_l1=("direction_match_l1", "mean"),
            direction_accuracy_l3=("direction_match_l3", "mean"),
            cosine_l1=("cos_l1_delta", "mean"),
            cosine_l3=("cos_l3_delta", "mean"),
            pearson_l1=("pearson_l1_delta", "mean"),
            pearson_l3=("pearson_l3_delta", "mean"),
            spearman_l1=("spearman_l1_delta", "mean"),
            spearman_l3=("spearman_l3_delta", "mean"),
        )
        .reset_index()
    )

    summary_metrics = {
        "overall": {
            "num_pairs": int(len(df_results)),
            "num_probes": int(len(unique_probe_ids)),
            "direction_accuracy_l1": float(df_results["direction_match_l1"].mean()),
            "direction_accuracy_l3": float(df_results["direction_match_l3"].mean()),
            "mean_cosine_l1": float(df_results["cos_l1_delta"].mean()),
            "mean_cosine_l3": float(df_results["cos_l3_delta"].mean()),
            "mean_pearson_l1": float(df_results["pearson_l1_delta"].mean(skipna=True)),
            "mean_pearson_l3": float(df_results["pearson_l3_delta"].mean(skipna=True)),
            "mean_spearman_l1": float(df_results["spearman_l1_delta"].mean(skipna=True)),
            "mean_spearman_l3": float(df_results["spearman_l3_delta"].mean(skipna=True)),
            "mean_bias_mag": float(df_results["bias_mag"].mean()),
        },
        "similarity_bins": bin_summary_df.to_dict("records"),
        "selection_similarity_bins": summarize_similarity_bins(df_pairs),
        "case_pair_ids": df_cases["pair_id"].astype(int).tolist(),
        "assumptions": {
            "repo_directory": "src/experiments",
            "voltage_readout": "layer3.get_grouped_voltage(v_mem_snapshot).mean(-1) at decision_offset pre-decision snapshot",
            "l3_overlap_mode": "sensory_only_static_frozen",
            "l3_temporal_pool": str(args.l3_temporal_pool),
            "public_similarity": "cosine on raw input image flatten vectors",
        },
    }

    pair_csv = save_tidy_csv(df_results, output_dir / "pair_results.csv", sort_by=["pair_id"])
    probe_meta_csv = save_tidy_csv(pd.DataFrame(probe_meta_rows), output_dir / "probe_metadata.csv", sort_by=["probe_id"])
    case_pairs_csv = save_tidy_csv(df_cases, output_dir / "case_pairs.csv", sort_by=["probe_id", "pair_id"])
    pair_vectors_npz = output_dir / "pair_vectors.npz"
    np.savez_compressed(
        pair_vectors_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        v_dyn=voltage_dynamic,
        v_sta=voltage_static,
        delta_v=delta_v,
        P_L1=pred_l1,
        P_L3=pred_l3,
    )
    probe_c_l1_npz = output_dir / "probe_C_l1.npz"
    np.savez_compressed(
        probe_c_l1_npz,
        probe_id=np.asarray(unique_probe_ids, dtype=np.int64),
        probe_label=labels[np.asarray(unique_probe_ids, dtype=np.int64)],
        C_L1=np.stack(probe_c_l1_rows, axis=0),
        V_orig=np.stack(probe_v_orig_rows, axis=0),
    )
    probe_c_l3_npz = output_dir / "probe_C_l3.npz"
    np.savez_compressed(
        probe_c_l3_npz,
        probe_id=np.asarray(unique_probe_ids, dtype=np.int64),
        probe_label=labels[np.asarray(unique_probe_ids, dtype=np.int64)],
        C_L3=np.stack(probe_c_l3_rows, axis=0),
        V_orig=np.stack(probe_v_orig_rows, axis=0),
    )
    pair_overlap_l1_npz = output_dir / "pair_overlap_l1.npz"
    np.savez_compressed(
        pair_overlap_l1_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        O_L1=pair_overlap_l1,
    )
    pair_overlap_l3_npz = output_dir / "pair_overlap_l3.npz"
    np.savez_compressed(
        pair_overlap_l3_npz,
        pair_id=df_results["pair_id"].to_numpy(dtype=np.int64, copy=False),
        O_L3=pair_overlap_l3,
    )
    summary_json = _save_json(summary_metrics, output_dir / "summary_metrics.json")

    fig1 = plot_similarity_distribution(df_results)
    fig1_paths = save_figure_all_formats(fig1, figures_dir / "figure_1_similarity_distribution")
    plt.close(fig1)

    fig2 = plot_case_studies(
        df_cases=df_cases,
        images=images,
        probe_c_l1=np.stack(probe_c_l1_rows, axis=0),
        probe_c_l3=np.stack(probe_c_l3_rows, axis=0),
        pair_overlap_l1=pair_overlap_l1,
        pair_overlap_l3=pair_overlap_l3,
        pair_vectors={"delta_v": delta_v, "P_L1": pred_l1, "P_L3": pred_l3},
        probe_id_to_index=probe_id_to_index,
        pair_id_to_index=pair_id_to_index,
        l1_patches=l1_patches,
        l3_regions=l3_regions,
    )
    fig2_paths = save_figure_all_formats(fig2, figures_dir / "figure_2_case_studies")
    plt.close(fig2)

    fig3 = plot_direction_accuracy_compare(df_results)
    fig3_paths = save_figure_all_formats(fig3, figures_dir / "figure_3_direction_accuracy_compare")
    plt.close(fig3)

    fig4 = plot_vector_similarity_compare(df_results)
    fig4_paths = save_figure_all_formats(fig4, figures_dir / "figure_4_vector_similarity_compare")
    plt.close(fig4)

    scatter_fig = plot_vector_scatter_compare(df_results)
    scatter_paths = save_figure_all_formats(scatter_fig, figures_dir / "figure_4b_vector_scatter_compare")
    plt.close(scatter_fig)

    fig5 = plot_similarity_binned_compare(df_results)
    fig5_paths = save_figure_all_formats(fig5, figures_dir / "figure_5_similarity_binned_compare")
    plt.close(fig5)

    run_config_path = save_run_config(
        {
            "model_path": str(Path(args.model_path).resolve()),
            "config_argument": args.config,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": str(args.split),
            "output_dir": str(result_root.resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "probe_ms": float(args.probe_ms),
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_pairs": int(args.max_pairs),
            "num_sim_bins": int(args.num_sim_bins),
            "save_case_count": int(args.save_case_count),
            "l1_patch_size": int(args.l1_patch_size),
            "l1_stride": int(args.l1_stride),
            "l3_mask_mode": str(args.l3_mask_mode),
            "l3_temporal_pool": str(args.l3_temporal_pool),
            "num_classes": int(num_classes),
            "readout_step": int(readout_step),
            "outputs": {
                "pair_results_csv": str(Path(pair_csv).resolve()),
                "probe_metadata_csv": str(Path(probe_meta_csv).resolve()),
                "case_pairs_csv": str(Path(case_pairs_csv).resolve()),
                "pair_vectors_npz": str(pair_vectors_npz.resolve()),
                "probe_C_l1_npz": str(probe_c_l1_npz.resolve()),
                "probe_C_l3_npz": str(probe_c_l3_npz.resolve()),
                "pair_overlap_l1_npz": str(pair_overlap_l1_npz.resolve()),
                "pair_overlap_l3_npz": str(pair_overlap_l3_npz.resolve()),
                "summary_metrics_json": str(summary_json.resolve()),
                "figure_1_png": fig1_paths["png"],
                "figure_2_png": fig2_paths["png"],
                "figure_3_png": fig3_paths["png"],
                "figure_4_png": fig4_paths["png"],
                "figure_4b_png": scatter_paths["png"],
                "figure_5_png": fig5_paths["png"],
            },
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "bias_direction_mask_experiment",
            "pair_count": int(len(df_results)),
            "probe_count": int(len(unique_probe_ids)),
            "direction_accuracy_l1": float(df_results["direction_match_l1"].mean()),
            "direction_accuracy_l3": float(df_results["direction_match_l3"].mean()),
            "mean_cos_l1_delta": float(df_results["cos_l1_delta"].mean()),
            "mean_cos_l3_delta": float(df_results["cos_l3_delta"].mean()),
            "artifacts": {
                "data_summary_metrics_json": str(summary_json.resolve()),
                "run_config_json": str(run_config_path.resolve()),
            },
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=bias_direction_mask_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"pairs={len(df_results)}",
            f"probes={len(unique_probe_ids)}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )

    print("\n=== Bias Direction Mask Experiment Summary ===")
    print(f"Pairs: {len(df_results)}")
    print(f"Probes: {len(unique_probe_ids)}")
    print(f"Direction accuracy L1: {float(df_results['direction_match_l1'].mean()):.4f}")
    print(f"Direction accuracy L3: {float(df_results['direction_match_l3'].mean()):.4f}")
    print(f"Mean cosine(P, delta_v) L1: {float(df_results['cos_l1_delta'].mean()):.4f}")
    print(f"Mean cosine(P, delta_v) L3: {float(df_results['cos_l3_delta'].mean()):.4f}")
    print(f"Saved outputs under: {result_root.resolve()}")
    print(f"Saved: {pair_csv}")
    print(f"Saved: {pair_vectors_npz}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {run_config_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {run_log_path}")


if __name__ == "__main__":
    main()
