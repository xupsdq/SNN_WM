from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, build_dataset_arrays, encode_images
from src.experiments.common.input_masks import (
    define_overlap_probe_only_masks,
    foreground_mask_from_image,
    spatial_mask_to_channel_mask,
)
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import (
    reset_all_state_restore_selected_stsp_in_place,
    run_monitored_dms_rollout,
)
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_SINGLE_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)

EXPERIMENT_NAME = "dms_overlap_ux_support_mechanism_experiment"
DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = f"results/{EXPERIMENT_NAME}"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY_MS = 400.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_PROBES = 1000
DEFAULT_MAX_PAIRS = 1000
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_MIN_OVERLAP_AREA = 4
DEFAULT_MIN_PROBE_ONLY_AREA = 4
DEFAULT_MEDIUM_Q_LOW = 0.35
DEFAULT_MEDIUM_Q_HIGH = 0.65
DEFAULT_EARLY_WINDOW_MS = 15.0
DEFAULT_DRIVE_SCORE_THRESHOLD = 0.05
DEFAULT_SAVE_CASE_COUNT = 0
DT = 1.0 * ms
EPS = 1e-12
EVENT_ALIGN_PRE_STEPS = 8
EVENT_ALIGN_POST_STEPS = 12
CHAIN_PRE_SPIKE_STEPS = 4
CHAIN_POST_SPIKE_STEPS = 6

PANEL_FILENAMES = {
    "panel_a_overlap_definition": "fig4_panel_a_overlap_definition",
    "panel_b_preprobe_ux_overlap_vs_probeonly": "fig4_panel_b_preprobe_ux_overlap_vs_probeonly",
    "panel_c_support_area": "fig4_panel_c_support_area",
    "panel_d_mean_ux_on_overlap": "fig4_panel_d_mean_ux_on_overlap",
    "panel_e_total_memory_support": "fig4_panel_e_total_memory_support",
    "panel_f_p_advance": "fig4_panel_f_p_advance",
    "panel_g_p_recruit": "fig4_panel_g_p_recruit",
    "panel_h_p_loss": "fig4_panel_h_p_loss",
    "panel_i_delta_early_spike_count": "fig4_panel_i_delta_early_spike_count",
    "panel_j_delta_first_spike_latency": "fig4_panel_j_delta_first_spike_latency",
    "panel_k_overlap_input_gain": "fig4_panel_k_overlap_input_gain",
    "panel_l_probe_only_input_gain": "fig4_panel_l_probe_only_input_gain",
    "panel_m_input_selectivity_gain": "fig4_panel_m_input_selectivity_gain",
    "panel_n_lost_spike_delta_inhibition": "fig4_panel_n_lost_spike_delta_inhibition",
    "panel_n1_n_lost_spike_units": "fig4_panel_n1_n_lost_spike_units",
    "panel_o_local_winner_loser_voltage_trace": "fig4_panel_o_local_winner_loser_voltage_trace",
    "panel_p_local_winner_support_rate": "fig4_panel_p_local_winner_support_rate",
    "panel_q_winner_loser_contrast_shift": "fig4_panel_q_winner_loser_contrast_shift",
    "panel_r_event_time_mechanism": "fig4_panel_r_event_time_mechanism",
    "panel_s_causal_chain_prevalence": "fig4_panel_s_causal_chain_prevalence",
}
GROUP_ORDER = ["all_units", "overlap_dominant", "probe_only_dominant"]
GROUP_COLORS = {
    "all_units": COLOR_DYNAMIC,
    "overlap_dominant": COLOR_DYNAMIC,
    "probe_only_dominant": COLOR_STATIC,
}
GROUP_DISPLAY_NAMES = {
    "all_units": "all receiving",
    "overlap_dominant": "overlap-biased",
    "probe_only_dominant": "probe-only-biased",
}
LOCAL_KERNEL_RADIUS = 2


@dataclass(frozen=True)
class MediumTrial:
    trial_id: int
    probe_id: int
    probe_label: int
    sample_id: int
    sample_label: int
    overlap_mask: np.ndarray
    probe_only_mask: np.ndarray
    probe_foreground_mask: np.ndarray
    overlap_area: int
    probe_only_area: int
    overlap_quantile: float
    candidate_count: int


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        draws[idx] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return float(arr.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _mask_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    return float(np.asarray(arr, dtype=np.float64)[mask_bool].mean()) if bool(mask_bool.any()) else float("nan")


def _mask_sum(arr: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    return float(np.asarray(arr, dtype=np.float64)[mask_bool].sum()) if bool(mask_bool.any()) else 0.0


def _mask_to_coord_list(mask: np.ndarray) -> list[list[int]]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    return [[int(r), int(c)] for r, c in coords.tolist()]


def _pixel_gain(boundary_state: Mapping[str, torch.Tensor], batch_idx: int) -> np.ndarray:
    u = np.asarray(boundary_state["u"][batch_idx], dtype=np.float64)
    x = np.asarray(boundary_state["x"][batch_idx], dtype=np.float64)
    return (u * x).mean(axis=0)


def _balanced_probe_ids(class_index: Mapping[int, Sequence[int]], max_probes: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    buckets = {int(k): rng.permutation(np.asarray(v, dtype=np.int64)).tolist() for k, v in class_index.items()}
    out: list[int] = []
    while len(out) < int(max_probes):
        progressed = False
        for key in sorted(buckets):
            if not buckets[key]:
                continue
            out.append(int(buckets[key].pop(0)))
            progressed = True
            if len(out) >= int(max_probes):
                break
        if not progressed:
            break
    return out


def _stack_encoded(images: torch.Tensor, ids: Sequence[int], *, encoder, steps: int, device: torch.device) -> torch.Tensor:
    batch = images[[int(i) for i in ids]].to(device=device, dtype=torch.float32)
    return encode_images(encoder, batch, steps=int(steps))


def _first_latency(spikes: np.ndarray, silent_fill: int) -> np.ndarray:
    has_fire = spikes.any(axis=0)
    out = np.full(spikes.shape[1], int(silent_fill), dtype=np.int64)
    if bool(has_fire.any()):
        out[has_fire] = np.argmax(spikes[:, has_fire], axis=0).astype(np.int64)
    return out


def _ensure_dataframe(rows: list[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=list(columns))


def construct_medium_trials(
    images: torch.Tensor,
    labels: np.ndarray,
    class_index: Mapping[int, Sequence[int]],
    *,
    max_probes: int,
    max_pairs: int,
    foreground_threshold: float,
    min_overlap_area: int,
    min_probe_only_area: int,
    q_low: float,
    q_high: float,
    seed: int,
) -> list[MediumTrial]:
    # Fig4 no longer uses a high/low pair. It fixes a medium-overlap trial and asks
    # whether dynamic Layer1 STSP alone changes probe-time firing relative to static.
    fg_masks = [foreground_mask_from_image(images[i], threshold=foreground_threshold) for i in range(len(images))]
    all_ids = np.arange(len(images), dtype=np.int64)
    trials: list[MediumTrial] = []
    for probe_id in _balanced_probe_ids(class_index, max_probes=max_probes, seed=mix_seed(seed, 11)):
        probe_fg = np.asarray(fg_masks[int(probe_id)], dtype=bool)
        probe_label = int(labels[int(probe_id)])
        probe_area = int(probe_fg.sum())
        cand_ids = all_ids[(all_ids != int(probe_id)) & (labels[all_ids] != probe_label)]
        if cand_ids.size <= 0:
            continue
        overlap = np.asarray([int((fg_masks[int(i)] & probe_fg).sum()) for i in cand_ids.tolist()], dtype=np.int64)
        probe_only = np.asarray(probe_area - overlap, dtype=np.int64)
        valid = (overlap >= int(min_overlap_area)) & (probe_only >= int(min_probe_only_area))
        if not bool(valid.any()):
            continue
        cand_ids = cand_ids[valid]
        overlap = overlap[valid]
        probe_only = probe_only[valid]
        lo = float(np.quantile(overlap, float(q_low)))
        hi = float(np.quantile(overlap, float(q_high)))
        mid = (overlap >= lo) & (overlap <= hi)
        if not bool(mid.any()):
            dist = np.abs(overlap.astype(np.float64) - float(np.median(overlap)))
            mid = dist == float(dist.min())
        mid_ids = cand_ids[mid]
        mid_overlap = overlap[mid]
        mid_probe_only = probe_only[mid]
        order = np.lexsort(
            (-mid_probe_only.astype(np.float64), np.abs(mid_overlap.astype(np.float64) - float(np.median(overlap))))
        )
        sample_id = int(mid_ids[int(order[0])])
        overlap_mask, probe_only_mask, probe_fg_mask = define_overlap_probe_only_masks(
            images[sample_id], images[int(probe_id)], threshold=foreground_threshold
        )
        if int(probe_only_mask.sum()) < int(min_probe_only_area):
            continue
        overlap_area = int(overlap_mask.sum())
        quantile = float((np.sum(overlap <= overlap_area) - 0.5) / max(int(overlap.size), 1))
        trials.append(
            MediumTrial(
                trial_id=len(trials),
                probe_id=int(probe_id),
                probe_label=probe_label,
                sample_id=sample_id,
                sample_label=int(labels[sample_id]),
                overlap_mask=np.asarray(overlap_mask, dtype=bool),
                probe_only_mask=np.asarray(probe_only_mask, dtype=bool),
                probe_foreground_mask=np.asarray(probe_fg_mask, dtype=bool),
                overlap_area=overlap_area,
                probe_only_area=int(probe_only_mask.sum()),
                overlap_quantile=quantile,
                candidate_count=int(cand_ids.size),
            )
        )
        if len(trials) >= int(max_pairs):
            break
    if not trials:
        raise RuntimeError("No medium-overlap trials were found.")
    return trials


def trial_metadata_table(trials: Sequence[MediumTrial]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": [t.trial_id for t in trials],
            "probe_id": [t.probe_id for t in trials],
            "probe_label": [t.probe_label for t in trials],
            "sample_id": [t.sample_id for t in trials],
            "sample_label": [t.sample_label for t in trials],
            "overlap_area": [t.overlap_area for t in trials],
            "probe_only_area": [t.probe_only_area for t in trials],
            "probe_foreground_area": [int(t.probe_foreground_mask.sum()) for t in trials],
            "overlap_quantile": [t.overlap_quantile for t in trials],
            "candidate_count": [t.candidate_count for t in trials],
        }
    ).sort_values(["trial_id"], kind="stable").reset_index(drop=True)


def trial_mask_payload(trials: Sequence[MediumTrial]) -> dict[str, object]:
    return {
        "trials": [
            {
                "trial_id": int(t.trial_id),
                "probe_id": int(t.probe_id),
                "sample_id": int(t.sample_id),
                "overlap_coords": _mask_to_coord_list(t.overlap_mask),
                "probe_only_coords": _mask_to_coord_list(t.probe_only_mask),
            }
            for t in trials
        ]
    }


def summarize_preprobe_stsp(*, trial: MediumTrial, boundary_state: Mapping[str, torch.Tensor], batch_idx: int, model_type: str) -> dict[str, object]:
    gain = _pixel_gain(boundary_state, batch_idx)
    return {
        "trial_id": int(trial.trial_id),
        "model_type": str(model_type),
        "ux_overlap_pre": _mask_mean(gain, trial.overlap_mask),
        "ux_probe_only_pre": _mask_mean(gain, trial.probe_only_mask),
        "support_area": int(trial.overlap_mask.sum()),
        "mean_ux_on_overlap": _mask_mean(gain, trial.overlap_mask),
        "total_memory_support": _mask_sum(gain, trial.overlap_mask),
    }


def compute_l1_drive_scores(trial: MediumTrial, kernels_cpu: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = int(kernels_cpu.shape[1])
    overlap = torch.from_numpy(spatial_mask_to_channel_mask(trial.overlap_mask, channels).astype(np.float32)).unsqueeze(0)
    probe_only = torch.from_numpy(spatial_mask_to_channel_mask(trial.probe_only_mask, channels).astype(np.float32)).unsqueeze(0)
    w_overlap = F.conv2d(overlap, kernels_cpu, stride=1, padding=2).squeeze(0).numpy().reshape(-1)
    w_probe_only = F.conv2d(probe_only, kernels_cpu, stride=1, padding=2).squeeze(0).numpy().reshape(-1)
    drive = (w_overlap - w_probe_only) / (w_overlap + w_probe_only + EPS)
    return drive.astype(np.float64), w_overlap.astype(np.float64), w_probe_only.astype(np.float64)


def _classify_drive_group(drive: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(drive.shape, "balanced", dtype=object)
    labels[drive > float(threshold)] = "overlap_dominant"
    labels[drive < -float(threshold)] = "probe_only_dominant"
    return labels


def _receiving_input_mask(w_overlap: np.ndarray, w_probe_only: np.ndarray) -> np.ndarray:
    # Transition composition excludes units with zero probe feedforward drive from
    # both overlap-defined and probe-only-defined probe pixels.
    return (np.asarray(w_overlap, dtype=np.float64) + np.asarray(w_probe_only, dtype=np.float64)) > EPS


def _classify_transition(dyn_count: np.ndarray, sta_count: np.ndarray, dyn_lat: np.ndarray, sta_lat: np.ndarray) -> np.ndarray:
    labels = np.full(dyn_count.shape, "unchanged", dtype=object)
    sta_silent = sta_count <= 0.0
    dyn_silent = dyn_count <= 0.0
    both = (~sta_silent) & (~dyn_silent)
    labels[sta_silent & (~dyn_silent)] = "recruit"
    labels[(~sta_silent) & dyn_silent] = "loss"
    labels[both & (dyn_lat < sta_lat)] = "advance"
    return labels


def compute_effective_input_trace(
    *,
    probe_spikes: torch.Tensor,
    kernels_cpu: torch.Tensor,
    gain_trace: torch.Tensor | None,
    static_gain: float | None,
    source_mask: torch.Tensor | None,
) -> np.ndarray:
    # Effective input is explicitly split into overlap-source and probe-only-source
    # terms so the mechanism is explained in Layer1 rather than a downstream axis.
    if (gain_trace is None) == (static_gain is None):
        raise ValueError("Exactly one of gain_trace or static_gain must be provided.")
    presyn = probe_spikes.to(dtype=torch.float32)
    if source_mask is not None:
        presyn = presyn * source_mask.to(dtype=torch.float32, device=presyn.device).unsqueeze(0)
    scaled = presyn * gain_trace.to(dtype=torch.float32) if gain_trace is not None else presyn * float(static_gain)
    syn = F.conv2d(scaled, kernels_cpu.to(device=presyn.device, dtype=torch.float32), stride=1, padding=2)
    return syn.detach().cpu().numpy().reshape(syn.shape[0], -1).astype(np.float64, copy=False)


def _build_panel_a_case_payload(
    *,
    trial: MediumTrial,
    dynamic_boundary_state: Mapping[str, torch.Tensor],
    static_boundary_state: Mapping[str, torch.Tensor],
    batch_idx: int,
) -> dict[str, object]:
    return {
        "trial_id": np.asarray([int(trial.trial_id)], dtype=np.int64),
        "sample_id": np.asarray([int(trial.sample_id)], dtype=np.int64),
        "probe_id": np.asarray([int(trial.probe_id)], dtype=np.int64),
        "overlap_mask": np.asarray(trial.overlap_mask, dtype=np.uint8),
        "probe_only_mask": np.asarray(trial.probe_only_mask, dtype=np.uint8),
        "ux_map_pre_dynamic": _pixel_gain(dynamic_boundary_state, batch_idx).astype(np.float32, copy=False),
        "ux_map_pre_static": _pixel_gain(static_boundary_state, batch_idx).astype(np.float32, copy=False),
    }


def summarize_trial(
    *,
    trial: MediumTrial,
    dynamic_output: Mapping[str, object],
    static_output: Mapping[str, object],
    batch_idx: int,
    early_window_steps: int,
    drive_score_threshold: float,
    kernels_cpu: torch.Tensor,
    probe_spikes_cpu: torch.Tensor,
    static_gain: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    dyn_l1 = dynamic_output["state_traces"]["layer1"]
    sta_l1 = static_output["state_traces"]["layer1"]
    dyn_spk = dyn_l1["spikes"][:early_window_steps, batch_idx].numpy().reshape(early_window_steps, -1)
    sta_spk = sta_l1["spikes"][:early_window_steps, batch_idx].numpy().reshape(early_window_steps, -1)
    dyn_inh = dyn_l1["inh_before"][:early_window_steps, batch_idx].numpy().reshape(early_window_steps, -1)
    sta_inh = sta_l1["inh_before"][:early_window_steps, batch_idx].numpy().reshape(early_window_steps, -1)
    drive, w_overlap, w_probe_only = compute_l1_drive_scores(trial, kernels_cpu)
    receives_probe_input = _receiving_input_mask(w_overlap, w_probe_only)
    unit_group = _classify_drive_group(drive, threshold=drive_score_threshold)
    dyn_count = dyn_spk.sum(axis=0).astype(np.float64)
    sta_count = sta_spk.sum(axis=0).astype(np.float64)
    dyn_lat = _first_latency(dyn_spk.astype(bool), silent_fill=early_window_steps + 1).astype(np.float64)
    sta_lat = _first_latency(sta_spk.astype(bool), silent_fill=early_window_steps + 1).astype(np.float64)
    transition = _classify_transition(dyn_count, sta_count, dyn_lat, sta_lat)

    channels = int(kernels_cpu.shape[1])
    overlap_mask = torch.from_numpy(spatial_mask_to_channel_mask(trial.overlap_mask, channels).astype(np.float32))
    probe_only_mask = torch.from_numpy(spatial_mask_to_channel_mask(trial.probe_only_mask, channels).astype(np.float32))
    probe_window = probe_spikes_cpu[batch_idx, :early_window_steps]
    dyn_overlap = compute_effective_input_trace(probe_spikes=probe_window, kernels_cpu=kernels_cpu, gain_trace=dyn_l1["gain"][:early_window_steps, batch_idx], static_gain=None, source_mask=overlap_mask)
    sta_overlap = compute_effective_input_trace(probe_spikes=probe_window, kernels_cpu=kernels_cpu, gain_trace=None, static_gain=static_gain, source_mask=overlap_mask)
    dyn_probe_only = compute_effective_input_trace(probe_spikes=probe_window, kernels_cpu=kernels_cpu, gain_trace=dyn_l1["gain"][:early_window_steps, batch_idx], static_gain=None, source_mask=probe_only_mask)
    sta_probe_only = compute_effective_input_trace(probe_spikes=probe_window, kernels_cpu=kernels_cpu, gain_trace=None, static_gain=static_gain, source_mask=probe_only_mask)

    drive_rows = [
        {
            "trial_id": int(trial.trial_id),
            "unit_idx": int(i),
            "w_overlap": float(w_overlap[i]),
            "w_probe_only": float(w_probe_only[i]),
            "drive_score": float(drive[i]),
            "receives_probe_input": bool(receives_probe_input[i]),
            "unit_group": str(unit_group[i]),
        }
        for i in range(drive.shape[0])
    ]
    groups = {
        "all_units": receives_probe_input,
        "overlap_dominant": receives_probe_input & (unit_group == "overlap_dominant"),
        "probe_only_dominant": receives_probe_input & (unit_group == "probe_only_dominant"),
    }
    winner_mask = np.isin(transition, np.asarray(["advance", "recruit"], dtype=object))
    winner_latency = int(np.min(dyn_lat[winner_mask])) if bool(winner_mask.any()) else None
    firing_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    for name, mask in groups.items():
        n_units = int(mask.sum())
        labels = transition[mask]
        firing_rows.append(
            {
                "trial_id": int(trial.trial_id),
                "aggregation_scope": "per_trial",
                "unit_group": str(name),
                "n_units": n_units,
                "n_advance": int(np.sum(labels == "advance")),
                "n_recruit": int(np.sum(labels == "recruit")),
                "n_loss": int(np.sum(labels == "loss")),
                "n_unchanged": int(np.sum(labels == "unchanged")),
                "P_advance": float(np.mean(labels == "advance")) if n_units > 0 else float("nan"),
                "P_recruit": float(np.mean(labels == "recruit")) if n_units > 0 else float("nan"),
                "P_loss": float(np.mean(labels == "loss")) if n_units > 0 else float("nan"),
                "P_unchanged": float(np.mean(labels == "unchanged")) if n_units > 0 else float("nan"),
                "delta_early_spike_count": float(np.mean(dyn_count[mask] - sta_count[mask])) if n_units > 0 else float("nan"),
                "delta_first_spike_latency": float(np.mean(dyn_lat[mask] - sta_lat[mask])) if n_units > 0 else float("nan"),
            }
        )
        selected = mask & np.isin(transition, np.asarray(["advance", "recruit"], dtype=object))
        n_selected = int(selected.sum())
        overlap_gain = float(np.mean(dyn_overlap[:, selected] - sta_overlap[:, selected])) if n_selected > 0 else float("nan")
        probe_only_gain = float(np.mean(dyn_probe_only[:, selected] - sta_probe_only[:, selected])) if n_selected > 0 else float("nan")
        input_rows.append(
            {
                "trial_id": int(trial.trial_id),
                "aggregation_scope": "per_trial",
                "unit_group": str(name),
                "transition_focus": "advance_or_recruit",
                "n_units_selected": n_selected,
                "overlap_input_dynamic": float(np.mean(dyn_overlap[:, selected])) if n_selected > 0 else float("nan"),
                "overlap_input_static": float(np.mean(sta_overlap[:, selected])) if n_selected > 0 else float("nan"),
                "probe_only_input_dynamic": float(np.mean(dyn_probe_only[:, selected])) if n_selected > 0 else float("nan"),
                "probe_only_input_static": float(np.mean(sta_probe_only[:, selected])) if n_selected > 0 else float("nan"),
                "overlap_input_gain": overlap_gain,
                "probe_only_input_gain": probe_only_gain,
                "input_selectivity_gain": overlap_gain - probe_only_gain if n_selected > 0 else float("nan"),
            }
        )
        loss_sel = mask & (transition == "loss")
        n_loss = int(loss_sel.sum())
        delta_inh = dyn_inh[:, loss_sel] - sta_inh[:, loss_sel] if n_loss > 0 else None
        loss_rows.append(
            {
                "trial_id": int(trial.trial_id),
                "aggregation_scope": "per_trial",
                "unit_group": str(name),
                "n_lost_spike_units": n_loss,
                "lost_spike_delta_inh": float(np.mean(delta_inh)) if n_loss > 0 else float("nan"),
                "winner_loser_latency_gap": float(np.mean(sta_lat[loss_sel] - float(winner_latency))) if n_loss > 0 and winner_latency is not None else float("nan"),
                "post_winner_inhibition_rise": float(np.mean(delta_inh[max(0, min(int(winner_latency), early_window_steps - 1)) :, :])) if n_loss > 0 and winner_latency is not None else float("nan"),
            }
        )
    return drive_rows, firing_rows, input_rows, loss_rows


def _unit_positions_from_shape(shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels, height, width = [int(v) for v in shape]
    flat = np.arange(channels * height * width, dtype=np.int64)
    channel_idx = flat // (height * width)
    spatial = flat % (height * width)
    row_idx = spatial // width
    col_idx = spatial % width
    return channel_idx, row_idx, col_idx


def _dynamic_first_spike_time(spike_trace: np.ndarray) -> int:
    coords = np.argwhere(np.asarray(spike_trace, dtype=bool))
    if coords.size <= 0:
        return -1
    return int(coords[0, 0])


def _aligned_window(trace: np.ndarray, *, center: int, pre_steps: int, post_steps: int) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64).reshape(-1)
    out = np.full(pre_steps + post_steps + 1, np.nan, dtype=np.float64)
    if arr.size <= 0:
        return out
    start = max(0, int(center) - int(pre_steps))
    stop = min(arr.size, int(center) + int(post_steps) + 1)
    out_start = int(pre_steps) - (int(center) - start)
    out_stop = out_start + (stop - start)
    out[out_start:out_stop] = arr[start:stop]
    return out


def _window_mean(trace: np.ndarray, *, start: int, stop: int) -> float:
    arr = np.asarray(trace, dtype=np.float64).reshape(-1)
    lo = max(0, int(start))
    hi = min(arr.size, int(stop))
    if hi <= lo:
        return float("nan")
    vals = arr[lo:hi]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size > 0 else float("nan")


def _local_winner_priority_key(
    *,
    unit_idx: int,
    transition: np.ndarray,
    overlap_gain: np.ndarray,
    dyn_first_spike: np.ndarray,
) -> tuple[float, float, float, float]:
    is_recruit = 0.0 if str(transition[unit_idx]) == "recruit" else 1.0
    gain_rank = -float(overlap_gain[unit_idx])
    latency = float(dyn_first_spike[unit_idx]) if int(dyn_first_spike[unit_idx]) >= 0 else float("inf")
    return (is_recruit, gain_rank, latency, float(unit_idx))


def build_local_winner_loser_analysis(
    *,
    trial: MediumTrial,
    dynamic_output: Mapping[str, object],
    static_output: Mapping[str, object],
    batch_idx: int,
    early_window_steps: int,
    kernels_cpu: torch.Tensor,
    probe_spikes_cpu: torch.Tensor,
    static_gain: float,
    drive_score_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object] | None]:
    # Local pairs are loser-centered because the mechanism question is whether each loser
    # has a nearby overlap-enhanced rival within the actual 5x5 inhibition neighborhood,
    # not whether a global earliest winner exists somewhere else in the map.
    dyn_l1 = dynamic_output["state_traces"]["layer1"]
    sta_l1 = static_output["state_traces"]["layer1"]
    dyn_spikes_full = dyn_l1["spikes"][:, batch_idx].numpy()
    sta_spikes_full = sta_l1["spikes"][:, batch_idx].numpy()
    dyn_v_effective = dyn_l1["v_effective"][:, batch_idx].numpy().reshape(dyn_spikes_full.shape[0], -1)
    sta_v_effective = sta_l1["v_effective"][:, batch_idx].numpy().reshape(sta_spikes_full.shape[0], -1)
    dyn_v_raw = dyn_l1["v_raw"][:, batch_idx].numpy().reshape(dyn_spikes_full.shape[0], -1)
    sta_v_raw = sta_l1["v_raw"][:, batch_idx].numpy().reshape(sta_spikes_full.shape[0], -1)
    dyn_inh_before = dyn_l1["inh_before"][:, batch_idx].numpy().reshape(dyn_spikes_full.shape[0], -1)
    sta_inh_before = sta_l1["inh_before"][:, batch_idx].numpy().reshape(sta_spikes_full.shape[0], -1)
    dyn_inh_after = dyn_l1["inh_after"][:, batch_idx].numpy().reshape(dyn_spikes_full.shape[0], -1)
    sta_inh_after = sta_l1["inh_after"][:, batch_idx].numpy().reshape(sta_spikes_full.shape[0], -1)
    dyn_spikes_2d = dyn_spikes_full.reshape(dyn_spikes_full.shape[0], -1)
    sta_spikes_2d = sta_spikes_full.reshape(sta_spikes_full.shape[0], -1)

    dyn_count = dyn_spikes_2d[:, :].sum(axis=0).astype(np.float64)
    sta_count = sta_spikes_2d[:, :].sum(axis=0).astype(np.float64)
    dyn_first = np.asarray([_dynamic_first_spike_time(dyn_spikes_2d[:, i]) for i in range(dyn_spikes_2d.shape[1])], dtype=np.int64)
    sta_first = np.asarray([_dynamic_first_spike_time(sta_spikes_2d[:, i]) for i in range(sta_spikes_2d.shape[1])], dtype=np.int64)
    dyn_lat_early = _first_latency(dyn_spikes_2d[:early_window_steps].astype(bool), silent_fill=early_window_steps + 1).astype(np.float64)
    sta_lat_early = _first_latency(sta_spikes_2d[:early_window_steps].astype(bool), silent_fill=early_window_steps + 1).astype(np.float64)
    transition = _classify_transition(dyn_count, sta_count, dyn_lat_early, sta_lat_early)

    drive_score, _, _ = compute_l1_drive_scores(trial, kernels_cpu)
    unit_group = _classify_drive_group(drive_score, threshold=drive_score_threshold)
    channels = int(kernels_cpu.shape[1])
    overlap_mask = torch.from_numpy(spatial_mask_to_channel_mask(trial.overlap_mask, channels).astype(np.float32))
    probe_only_mask = torch.from_numpy(spatial_mask_to_channel_mask(trial.probe_only_mask, channels).astype(np.float32))
    probe_full = probe_spikes_cpu[batch_idx]
    dyn_overlap_full = compute_effective_input_trace(
        probe_spikes=probe_full,
        kernels_cpu=kernels_cpu,
        gain_trace=dyn_l1["gain"][:, batch_idx],
        static_gain=None,
        source_mask=overlap_mask,
    )
    sta_overlap_full = compute_effective_input_trace(
        probe_spikes=probe_full,
        kernels_cpu=kernels_cpu,
        gain_trace=None,
        static_gain=static_gain,
        source_mask=overlap_mask,
    )
    dyn_probe_only_full = compute_effective_input_trace(
        probe_spikes=probe_full,
        kernels_cpu=kernels_cpu,
        gain_trace=dyn_l1["gain"][:, batch_idx],
        static_gain=None,
        source_mask=probe_only_mask,
    )
    sta_probe_only_full = compute_effective_input_trace(
        probe_spikes=probe_full,
        kernels_cpu=kernels_cpu,
        gain_trace=None,
        static_gain=static_gain,
        source_mask=probe_only_mask,
    )
    dyn_overlap_gain_early = np.mean(
        dyn_overlap_full[:early_window_steps, :] - sta_overlap_full[:early_window_steps, :],
        axis=0,
    )
    dyn_probe_only_gain_early = np.mean(
        dyn_probe_only_full[:early_window_steps, :] - sta_probe_only_full[:early_window_steps, :],
        axis=0,
    )

    _, row_idx, col_idx = _unit_positions_from_shape(dyn_spikes_full.shape[1:])
    # Local losers are restricted to units that spike in static but go fully silent in dynamic.
    loser_mask = (sta_count > 0.0) & (dyn_count <= 0.0)
    winner_mask = np.isin(transition, np.asarray(["recruit", "advance"], dtype=object)) & (dyn_overlap_gain_early > 0.0)

    pair_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    chain_rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    exemplar_row: dict[str, object] | None = None
    for loser_idx in np.flatnonzero(loser_mask):
        in_neighborhood = (
            (np.abs(row_idx - row_idx[loser_idx]) <= LOCAL_KERNEL_RADIUS)
            & (np.abs(col_idx - col_idx[loser_idx]) <= LOCAL_KERNEL_RADIUS)
            & winner_mask
        )
        candidate_indices = np.flatnonzero(in_neighborhood)
        supported = int(candidate_indices.size > 0)
        support_rows.append(
            {
                "aggregation_scope": "loser_event",
                "trial_id": int(trial.trial_id),
                "loser_unit_idx": int(loser_idx),
                "loser_row": int(row_idx[loser_idx]),
                "loser_col": int(col_idx[loser_idx]),
                "loser_group": str(unit_group[loser_idx]),
                "supported": supported,
                "local_winner_support_rate": float(supported),
                "n_loser_events": 1,
                "n_supported_events": supported,
            }
        )
        if not supported:
            continue
        # Recruit winners are preferred over advance winners because the clearest local
        # takeover event is a unit that is newly brought into the competition by overlap gain.
        ranked = sorted(
            candidate_indices.tolist(),
            key=lambda idx: _local_winner_priority_key(
                unit_idx=int(idx),
                transition=transition,
                overlap_gain=dyn_overlap_gain_early,
                dyn_first_spike=dyn_first,
            ),
        )
        winner_idx = int(ranked[0])
        t_star = int(dyn_first[winner_idx]) if int(dyn_first[winner_idx]) >= 0 else 0
        t_star = max(0, min(t_star, int(dyn_v_effective.shape[0] - 1)))
        winner_delta_v = dyn_v_effective[:, winner_idx] - sta_v_effective[:, winner_idx]
        loser_delta_v = dyn_v_effective[:, loser_idx] - sta_v_effective[:, loser_idx]
        loser_inh_before_dynamic = dyn_inh_before[:, loser_idx]
        loser_inh_before_static = sta_inh_before[:, loser_idx]
        loser_inh_after_dynamic = dyn_inh_after[:, loser_idx]
        loser_inh_after_static = sta_inh_after[:, loser_idx]
        winner_pre_boost_mean = _window_mean(
            winner_delta_v,
            start=t_star - CHAIN_PRE_SPIKE_STEPS,
            stop=t_star,
        )
        loser_post_delta_v_mean = _window_mean(
            loser_delta_v,
            start=t_star + 1,
            stop=t_star + 1 + CHAIN_POST_SPIKE_STEPS,
        )
        loser_pre_inh_before_mean = _window_mean(
            loser_inh_before_dynamic,
            start=t_star - CHAIN_PRE_SPIKE_STEPS,
            stop=t_star,
        )
        loser_post_inh_before_mean = _window_mean(
            loser_inh_before_dynamic,
            start=t_star + 1,
            stop=t_star + 1 + CHAIN_POST_SPIKE_STEPS,
        )
        winner_pre_spike_boost = bool(np.isfinite(winner_pre_boost_mean) and winner_pre_boost_mean > 0.0)
        winner_spikes_earlier = bool(int(dyn_first[winner_idx]) >= 0 and (int(sta_first[winner_idx]) < 0 or int(dyn_first[winner_idx]) < int(sta_first[winner_idx])))
        loser_post_inh_rise = float(loser_post_inh_before_mean - loser_pre_inh_before_mean) if np.isfinite(loser_post_inh_before_mean) and np.isfinite(loser_pre_inh_before_mean) else float("nan")
        loser_post_winner_suppressed = bool(
            np.isfinite(loser_pre_inh_before_mean)
            and np.isfinite(loser_post_inh_before_mean)
            and loser_post_inh_before_mean > loser_pre_inh_before_mean
        )
        full_chain_satisfied = bool(winner_pre_spike_boost and winner_spikes_earlier and loser_post_winner_suppressed)
        c_dyn = float(dyn_v_effective[t_star, winner_idx] - dyn_v_effective[t_star, loser_idx])
        c_sta = float(sta_v_effective[t_star, winner_idx] - sta_v_effective[t_star, loser_idx])
        row = {
            "trial_id": int(trial.trial_id),
            "winner_unit_idx": int(winner_idx),
            "loser_unit_idx": int(loser_idx),
            "winner_group": str(unit_group[winner_idx]),
            "loser_group": str(unit_group[loser_idx]),
            "winner_transition": str(transition[winner_idx]),
            "loser_transition": "local_loser",
            "winner_row": int(row_idx[winner_idx]),
            "winner_col": int(col_idx[winner_idx]),
            "loser_row": int(row_idx[loser_idx]),
            "loser_col": int(col_idx[loser_idx]),
            "winner_first_spike_dynamic": int(dyn_first[winner_idx]),
            "winner_first_spike_static": int(sta_first[winner_idx]),
            "loser_first_spike_dynamic": int(dyn_first[loser_idx]),
            "loser_first_spike_static": int(sta_first[loser_idx]),
            "winner_overlap_input_gain": float(dyn_overlap_gain_early[winner_idx]),
            "winner_probe_only_input_gain": float(dyn_probe_only_gain_early[winner_idx]),
            "loser_overlap_input_gain": float(dyn_overlap_gain_early[loser_idx]),
            "loser_probe_only_input_gain": float(dyn_probe_only_gain_early[loser_idx]),
            "contrast_time_index": int(t_star),
            "contrast_dynamic": c_dyn,
            "contrast_static": c_sta,
            "winner_loser_contrast_shift": float(c_dyn - c_sta),
            "winner_priority_class": str(transition[winner_idx]),
            "local_radius": int(LOCAL_KERNEL_RADIUS),
        }
        pair_rows.append(row)
        chain_rows.append(
            {
                "trial_id": int(trial.trial_id),
                "winner_unit_idx": int(winner_idx),
                "loser_unit_idx": int(loser_idx),
                "winner_group": str(unit_group[winner_idx]),
                "loser_group": str(unit_group[loser_idx]),
                "align_time_index": int(t_star),
                "winner_first_spike_dynamic": int(dyn_first[winner_idx]),
                "winner_first_spike_static": int(sta_first[winner_idx]),
                "loser_first_spike_dynamic": int(dyn_first[loser_idx]),
                "loser_first_spike_static": int(sta_first[loser_idx]),
                "winner_pre_spike_boost": int(winner_pre_spike_boost),
                "winner_spikes_earlier": int(winner_spikes_earlier),
                "loser_post_winner_suppressed": int(loser_post_winner_suppressed),
                "full_chain_satisfied": int(full_chain_satisfied),
                "winner_pre_spike_delta_v_mean": float(winner_pre_boost_mean),
                "loser_post_winner_delta_v_mean": float(loser_post_delta_v_mean),
                "loser_pre_winner_inh_before_mean": float(loser_pre_inh_before_mean),
                "loser_post_winner_inh_before_mean": float(loser_post_inh_before_mean),
                "loser_post_winner_inh_rise": float(loser_post_inh_rise),
            }
        )
        aligned_rows.append(
            {
                "trial_id": int(trial.trial_id),
                "winner_unit_idx": int(winner_idx),
                "loser_unit_idx": int(loser_idx),
                "winner_delta_v_aligned": _aligned_window(
                    winner_delta_v,
                    center=t_star,
                    pre_steps=EVENT_ALIGN_PRE_STEPS,
                    post_steps=EVENT_ALIGN_POST_STEPS,
                ),
                "loser_delta_v_aligned": _aligned_window(
                    loser_delta_v,
                    center=t_star,
                    pre_steps=EVENT_ALIGN_PRE_STEPS,
                    post_steps=EVENT_ALIGN_POST_STEPS,
                ),
                "loser_inh_before_aligned": _aligned_window(
                    loser_inh_before_dynamic,
                    center=t_star,
                    pre_steps=EVENT_ALIGN_PRE_STEPS,
                    post_steps=EVENT_ALIGN_POST_STEPS,
                ),
                "loser_inh_after_aligned": _aligned_window(
                    loser_inh_after_dynamic,
                    center=t_star,
                    pre_steps=EVENT_ALIGN_PRE_STEPS,
                    post_steps=EVENT_ALIGN_POST_STEPS,
                ),
            }
        )
        exemplar_payload = {
            **row,
            "t_axis": np.arange(int(dyn_v_effective.shape[0]), dtype=np.int64),
            "winner_v_effective_dynamic": dyn_v_effective[:, winner_idx].copy(),
            "winner_v_effective_static": sta_v_effective[:, winner_idx].copy(),
            "loser_v_effective_dynamic": dyn_v_effective[:, loser_idx].copy(),
            "loser_v_effective_static": sta_v_effective[:, loser_idx].copy(),
            "winner_v_raw_dynamic": dyn_v_raw[:, winner_idx].copy(),
            "winner_v_raw_static": sta_v_raw[:, winner_idx].copy(),
            "loser_v_raw_dynamic": dyn_v_raw[:, loser_idx].copy(),
            "loser_v_raw_static": sta_v_raw[:, loser_idx].copy(),
            "winner_inh_before_dynamic": dyn_inh_before[:, winner_idx].copy(),
            "winner_inh_before_static": sta_inh_before[:, winner_idx].copy(),
            "loser_inh_before_dynamic": dyn_inh_before[:, loser_idx].copy(),
            "loser_inh_before_static": sta_inh_before[:, loser_idx].copy(),
            "winner_inh_after_dynamic": dyn_inh_after[:, winner_idx].copy(),
            "winner_inh_after_static": sta_inh_after[:, winner_idx].copy(),
            "loser_inh_after_dynamic": dyn_inh_after[:, loser_idx].copy(),
            "loser_inh_after_static": sta_inh_after[:, loser_idx].copy(),
            "winner_overlap_input_dynamic": dyn_overlap_full[:, winner_idx].copy(),
            "winner_overlap_input_static": sta_overlap_full[:, winner_idx].copy(),
            "winner_probe_only_input_dynamic": dyn_probe_only_full[:, winner_idx].copy(),
            "winner_probe_only_input_static": sta_probe_only_full[:, winner_idx].copy(),
        }
        if exemplar_row is None or float(exemplar_payload["winner_loser_contrast_shift"]) > float(exemplar_row["winner_loser_contrast_shift"]):
            exemplar_row = exemplar_payload
    return pair_rows, support_rows, chain_rows, aligned_rows, exemplar_row


def _append_pooled(df: pd.DataFrame, *, weight_col: str, group_cols: list[str], value_cols: list[str], count_cols: list[str]) -> pd.DataFrame:
    pooled: list[dict[str, object]] = []
    base = df[df["aggregation_scope"] == "per_trial"].copy()
    for keys, sub in base.groupby(group_cols, sort=True):
        row = {"trial_id": -1, "aggregation_scope": "pooled"}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys):
            row[col] = value
        weights = sub[weight_col].to_numpy(dtype=np.float64)
        row[weight_col] = int(np.sum(weights))
        for col in count_cols:
            row[col] = int(sub[col].sum())
        for col in value_cols:
            row[col] = _wmean(sub[col].to_numpy(dtype=np.float64), weights)
        pooled.append(row)
    return pd.concat([df, pd.DataFrame(pooled)], ignore_index=True) if pooled else df


def make_layer1_only_probe_reset_fn() -> Callable:
    # The probe resets all layers and restores only Layer1 u/x so Fig4 isolates the
    # retained Layer1 memory source and deletes the downstream closed loop.
    def probe_reset_fn(net, ctx):
        layer1_u = net.layer1.u_pre.detach().clone()
        layer1_x = net.layer1.x_pre.detach().clone()
        return reset_all_state_restore_selected_stsp_in_place(
            net=net,
            layer_input_shapes=ctx["layer_input_shapes"],
            restore_ux_by_layer={"layer1": (layer1_u, layer1_x)},
        )

    return probe_reset_fn


def run_rollout_batch(*, net, sample_spikes: torch.Tensor, probe_spikes: torch.Tensor, delay_steps: int, stsp_mode: str) -> dict[str, object]:
    # Layer3 grouped voltage, decision vector, and downstream bias are removed.
    rollout = run_monitored_dms_rollout(
        net=net,
        sample_spikes=sample_spikes,
        probe_spikes=probe_spikes,
        delay_steps=delay_steps,
        stsp_mode=stsp_mode,
        phase_reset=True,
        intervention_plan={"probe_reset_fn": make_layer1_only_probe_reset_fn()},
        record_state_names={"layer1": ("spikes", "gain", "inh_before", "inh_after", "v_raw", "v_effective")},
        record_phase_names=("probe",),
    )
    return {"state_traces": rollout["state_traces"], "boundary_states": rollout["boundary_states"]}


def _scatter_with_mean(ax, x: float, values: np.ndarray, color: str) -> None:
    values_arr = np.asarray(values, dtype=np.float64)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size <= 0:
        return
    jitter = np.linspace(-0.08, 0.08, num=values_arr.size) if values_arr.size > 1 else np.asarray([0.0], dtype=np.float64)
    ax.scatter(np.full(values_arr.size, x, dtype=np.float64) + jitter, values_arr, s=20, color=color, alpha=0.65, edgecolors="none")
    sem = _sem(values_arr)
    ax.errorbar([x], [float(values_arr.mean())], yerr=[[sem], [sem]], fmt="o", color="black", capsize=3, linewidth=1.2, markersize=4)


def _strip_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _single_image_panel(image_2d: np.ndarray, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    ax.imshow(np.asarray(image_2d, dtype=np.float64), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.98)
    return fig


def _two_condition_panel(df: pd.DataFrame, metric: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    for i, (label, color) in enumerate((("dynamic", COLOR_DYNAMIC), ("static", COLOR_STATIC))):
        _scatter_with_mean(ax, float(i), df[df["model_type"] == label][metric].to_numpy(dtype=np.float64), color)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["dynamic", "static"])
    ax.set_ylabel(ylabel)
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def _group_panel(df: pd.DataFrame, metric: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    base = df[(df["aggregation_scope"] == "per_trial") & (df["unit_group"].isin(GROUP_ORDER))]
    for i, group in enumerate(GROUP_ORDER):
        _scatter_with_mean(ax, float(i), base[base["unit_group"] == group][metric].to_numpy(dtype=np.float64), GROUP_COLORS[group])
    ax.set_xticks(np.arange(len(GROUP_ORDER), dtype=np.float64))
    ax.set_xticklabels([GROUP_DISPLAY_NAMES[group] for group in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def _build_local_support_summary(df_support_events: pd.DataFrame) -> pd.DataFrame:
    trial_rows: list[dict[str, object]] = []
    event_df = df_support_events[df_support_events["aggregation_scope"] == "loser_event"].copy()
    for trial_id, sub in event_df.groupby("trial_id", sort=True):
        n_events = int(len(sub))
        n_supported = int(sub["supported"].sum())
        trial_rows.append(
            {
                "aggregation_scope": "per_trial",
                "trial_id": int(trial_id),
                "loser_unit_idx": -1,
                "loser_row": -1,
                "loser_col": -1,
                "loser_group": "all_local_losers",
                "supported": float("nan"),
                "local_winner_support_rate": float(n_supported / n_events) if n_events > 0 else float("nan"),
                "n_loser_events": n_events,
                "n_supported_events": n_supported,
            }
        )
    if trial_rows:
        total_events = int(sum(row["n_loser_events"] for row in trial_rows))
        total_supported = int(sum(row["n_supported_events"] for row in trial_rows))
        trial_rows.append(
            {
                "aggregation_scope": "pooled",
                "trial_id": -1,
                "loser_unit_idx": -1,
                "loser_row": -1,
                "loser_col": -1,
                "loser_group": "all_local_losers",
                "supported": float("nan"),
                "local_winner_support_rate": float(total_supported / total_events) if total_events > 0 else float("nan"),
                "n_loser_events": total_events,
                "n_supported_events": total_supported,
            }
        )
    return pd.concat([df_support_events, pd.DataFrame(trial_rows)], ignore_index=True) if trial_rows else df_support_events


def _stack_aligned_event_rows(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    rel_time = np.arange(-EVENT_ALIGN_PRE_STEPS, EVENT_ALIGN_POST_STEPS + 1, dtype=np.int64)
    if not rows:
        width = rel_time.size
        return {
            "relative_time": rel_time,
            "trial_id": np.empty((0,), dtype=np.int64),
            "winner_unit_idx": np.empty((0,), dtype=np.int64),
            "loser_unit_idx": np.empty((0,), dtype=np.int64),
            "winner_delta_v_aligned": np.empty((0, width), dtype=np.float32),
            "loser_delta_v_aligned": np.empty((0, width), dtype=np.float32),
            "loser_inh_before_aligned": np.empty((0, width), dtype=np.float32),
            "loser_inh_after_aligned": np.empty((0, width), dtype=np.float32),
        }
    return {
        "relative_time": rel_time,
        "trial_id": np.asarray([int(row["trial_id"]) for row in rows], dtype=np.int64),
        "winner_unit_idx": np.asarray([int(row["winner_unit_idx"]) for row in rows], dtype=np.int64),
        "loser_unit_idx": np.asarray([int(row["loser_unit_idx"]) for row in rows], dtype=np.int64),
        "winner_delta_v_aligned": np.stack([np.asarray(row["winner_delta_v_aligned"], dtype=np.float32) for row in rows], axis=0),
        "loser_delta_v_aligned": np.stack([np.asarray(row["loser_delta_v_aligned"], dtype=np.float32) for row in rows], axis=0),
        "loser_inh_before_aligned": np.stack([np.asarray(row["loser_inh_before_aligned"], dtype=np.float32) for row in rows], axis=0),
        "loser_inh_after_aligned": np.stack([np.asarray(row["loser_inh_after_aligned"], dtype=np.float32) for row in rows], axis=0),
    }


def _nanmean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for nanmean/sem summary.")
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0).astype(np.float64)
    std = np.nanstd(arr, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 1.0)
    sem[n <= 1.0] = 0.0
    return mean, sem


def _overlap_definition_panel(*, images: torch.Tensor, trial: MediumTrial, panel_a_case: Mapping[str, object] | None) -> plt.Figure:
    fig = plt.figure(figsize=(8.8, 2.9))
    host = fig.add_subplot(1, 1, 1)
    host.set_axis_off()
    left = 0.02
    sample_w = 0.23
    probe_w = 0.23
    heat_w = 0.30
    gap = 0.035
    cbar_w = 0.018
    ax_sample = host.inset_axes([left, 0.12, sample_w, 0.78])
    ax_probe = host.inset_axes([left + sample_w + gap, 0.12, probe_w, 0.78])
    ax_heat = host.inset_axes([left + sample_w + probe_w + 2.0 * gap, 0.12, heat_w, 0.78])
    ax_cbar = host.inset_axes([left + sample_w + probe_w + heat_w + 2.7 * gap, 0.16, cbar_w, 0.70])

    sample = images[int(trial.sample_id)][0].numpy()
    probe = images[int(trial.probe_id)][0].numpy()
    heat = np.asarray(panel_a_case["ux_map_pre_dynamic"], dtype=np.float64) if panel_a_case is not None else np.zeros_like(sample, dtype=np.float64)
    overlap_mask = np.asarray(panel_a_case["overlap_mask"], dtype=np.float64) if panel_a_case is not None else trial.overlap_mask.astype(np.float64)
    probe_only_mask = np.asarray(panel_a_case["probe_only_mask"], dtype=np.float64) if panel_a_case is not None else trial.probe_only_mask.astype(np.float64)

    for ax, image, title in (
        (ax_sample, sample, "Sample"),
        (ax_probe, probe, "Probe"),
    ):
        ax.imshow(np.asarray(image, dtype=np.float64), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title)

    im = ax_heat.imshow(heat, cmap="magma", interpolation="nearest")
    ax_heat.contour(overlap_mask, levels=[0.5], colors=[COLOR_DYNAMIC], linewidths=1.2)
    ax_heat.contour(probe_only_mask, levels=[0.5], colors=[COLOR_STATIC], linewidths=1.1)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Pre-probe u*x")
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("u*x")
    if panel_a_case is not None:
        overlap_mean = _mask_mean(heat, overlap_mask)
        probe_only_mean = _mask_mean(heat, probe_only_mask)
        ax_heat.text(
            0.02,
            0.98,
            f"overlap={overlap_mean:.3f}\nprobe-only={probe_only_mean:.3f}",
            transform=ax_heat.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": (0.0, 0.0, 0.0, 0.38), "edgecolor": "none", "boxstyle": "round,pad=0.18"},
        )
    fig.tight_layout()
    return fig


def _local_voltage_trace_panel(exemplar: Mapping[str, object] | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if exemplar is None:
        ax.text(0.5, 0.5, "No local winner-loser exemplar", ha="center", va="center", transform=ax.transAxes)
        _strip_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    t_axis = np.asarray(exemplar["t_axis"], dtype=np.int64)
    traces = {
        "winner_v_effective_dynamic": 1000.0 * np.asarray(exemplar["winner_v_effective_dynamic"], dtype=np.float64),
        "winner_v_effective_static": 1000.0 * np.asarray(exemplar["winner_v_effective_static"], dtype=np.float64),
        "loser_v_effective_dynamic": 1000.0 * np.asarray(exemplar["loser_v_effective_dynamic"], dtype=np.float64),
        "loser_v_effective_static": 1000.0 * np.asarray(exemplar["loser_v_effective_static"], dtype=np.float64),
    }
    ax.plot(t_axis, traces["winner_v_effective_dynamic"], color=COLOR_DYNAMIC, linewidth=1.8, label="Winner dynamic")
    ax.plot(t_axis, traces["winner_v_effective_static"], color=COLOR_DYNAMIC, linewidth=1.2, linestyle="--", alpha=0.9, label="Winner static")
    ax.plot(t_axis, traces["loser_v_effective_dynamic"], color=COLOR_STATIC, linewidth=1.8, label="Loser dynamic")
    ax.plot(t_axis, traces["loser_v_effective_static"], color=COLOR_STATIC, linewidth=1.2, linestyle="--", alpha=0.9, label="Loser static")
    ax.axhline(-60.0, color="black", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.65)
    for key, series_key, color, fill in (
        ("winner_first_spike_dynamic", "winner_v_effective_dynamic", COLOR_DYNAMIC, True),
        ("winner_first_spike_static", "winner_v_effective_static", COLOR_DYNAMIC, False),
        ("loser_first_spike_dynamic", "loser_v_effective_dynamic", COLOR_STATIC, True),
        ("loser_first_spike_static", "loser_v_effective_static", COLOR_STATIC, False),
    ):
        spike_t = int(exemplar[key])
        if spike_t >= 0 and spike_t < traces[series_key].shape[0]:
            ax.scatter(
                [spike_t],
                [traces[series_key][spike_t]],
                s=18,
                facecolor=color if fill else "white",
                edgecolor=color,
                linewidth=0.8,
                zorder=4,
            )
    ax.set_xlabel("Probe step")
    ax.set_ylabel("$V_{effective}$ (mV)")
    _strip_axis(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def _trial_rate_panel(df_support: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    trial_df = df_support[df_support["aggregation_scope"] == "per_trial"]
    _scatter_with_mean(ax, 0.0, trial_df["local_winner_support_rate"].to_numpy(dtype=np.float64), COLOR_DYNAMIC)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["loser events"])
    ax.set_ylabel("Local winner support rate")
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def _contrast_shift_panel(df_pairs: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    values = 1000.0 * df_pairs["winner_loser_contrast_shift"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size > 0:
        ax.boxplot(
            [values],
            positions=[0.0],
            widths=0.34,
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "#D8F0E6", "edgecolor": COLOR_DYNAMIC, "linewidth": 0.95},
            whiskerprops={"color": COLOR_DYNAMIC, "linewidth": 0.9},
            capprops={"color": COLOR_DYNAMIC, "linewidth": 0.9},
            medianprops={"color": COLOR_DYNAMIC, "linewidth": 1.25},
            flierprops={"markersize": 0},
        )
        jitter = np.linspace(-0.08, 0.08, num=values.size) if values.size > 1 else np.asarray([0.0], dtype=np.float64)
        ax.scatter(np.full(values.size, 0.0) + jitter, values, s=16, color=COLOR_DYNAMIC, alpha=0.26, edgecolors="none", zorder=3)
        mean_val = float(values.mean())
        ci = 1.96 * _sem(values)
        ax.errorbar([0.0], [mean_val], yerr=[[ci], [ci]], fmt="o", color="black", capsize=3, linewidth=1.1, markersize=4.2, zorder=4)
        ax.text(
            0.0,
            0.97,
            f"mean={mean_val:.2f} mV\npositive={100.0 * np.mean(values > 0.0):.0f}%",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5, linestyle=(0, (3, 2)))
    ax.set_xticks([0.0])
    ax.set_xticklabels(["local pairs"])
    ax.set_ylabel("Winner-loser\ncontrast shift (mV)")
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def _event_time_mechanism_panel(aligned_payload: Mapping[str, object]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True, gridspec_kw={"hspace": 0.12})
    rel_t = np.asarray(aligned_payload["relative_time"], dtype=np.int64)
    if np.asarray(aligned_payload["winner_delta_v_aligned"]).shape[0] <= 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No aligned local events", ha="center", va="center", transform=ax.transAxes)
            _strip_axis(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig
    winner_mean, winner_sem = _nanmean_sem(np.asarray(aligned_payload["winner_delta_v_aligned"], dtype=np.float64))
    loser_mean, loser_sem = _nanmean_sem(np.asarray(aligned_payload["loser_delta_v_aligned"], dtype=np.float64))
    loser_inh_before_mean, loser_inh_before_sem = _nanmean_sem(np.asarray(aligned_payload["loser_inh_before_aligned"], dtype=np.float64))

    ax_top, ax_bottom = axes
    ax_top.plot(rel_t, 1000.0 * winner_mean, color=COLOR_DYNAMIC, linewidth=1.8, label="Winner ΔV")
    ax_top.fill_between(rel_t, 1000.0 * (winner_mean - winner_sem), 1000.0 * (winner_mean + winner_sem), color=COLOR_DYNAMIC, alpha=0.18, linewidth=0)
    ax_top.plot(rel_t, 1000.0 * loser_mean, color=COLOR_STATIC, linewidth=1.8, label="Loser ΔV")
    ax_top.fill_between(rel_t, 1000.0 * (loser_mean - loser_sem), 1000.0 * (loser_mean + loser_sem), color=COLOR_STATIC, alpha=0.18, linewidth=0)
    ax_top.axvline(0.0, color="black", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.6)
    ax_top.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax_top.set_ylabel("ΔV_effective (mV)")
    _strip_axis(ax_top)
    ax_top.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    ax_bottom.plot(rel_t, 1000.0 * loser_inh_before_mean, color=COLOR_STATIC, linewidth=1.8, label="Loser inhibition")
    ax_bottom.fill_between(
        rel_t,
        1000.0 * (loser_inh_before_mean - loser_inh_before_sem),
        1000.0 * (loser_inh_before_mean + loser_inh_before_sem),
        color=COLOR_STATIC,
        alpha=0.18,
        linewidth=0,
    )
    ax_bottom.axvline(0.0, color="black", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.6)
    ax_bottom.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax_bottom.set_xlabel("Relative time to winner dynamic first spike")
    ax_bottom.set_ylabel("Loser inhibition (mV)")
    _strip_axis(ax_bottom)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    return fig


def _causal_chain_prevalence_panel(df_chain: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    if df_chain.empty:
        ax.text(0.5, 0.5, "No local chain events", ha="center", va="center", transform=ax.transAxes)
        _strip_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    metrics = [
        ("winner_pre_spike_boost", "winner\nboosted"),
        ("winner_spikes_earlier", "winner\nspikes earlier"),
        ("loser_post_winner_suppressed", "loser\nsuppressed after"),
        ("full_chain_satisfied", "full\nchain"),
    ]
    xpos = np.arange(len(metrics), dtype=np.float64)
    for idx, (metric, label) in enumerate(metrics):
        vals = pd.to_numeric(df_chain[metric], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        mean, lo, hi = _bootstrap_ci(vals, seed=5200 + idx)
        ax.vlines(xpos[idx], 100.0 * lo, 100.0 * hi, color=COLOR_DYNAMIC, linewidth=2.0, zorder=2)
        ax.scatter([xpos[idx]], [100.0 * mean], s=34, color=COLOR_DYNAMIC, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Prevalence (%)")
    ax.set_ylim(0.0, 105.0)
    _strip_axis(ax)
    fig.tight_layout()
    return fig


def save_all_panels(layout, panels: Mapping[str, plt.Figure]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, fig in panels.items():
        out[key] = save_figure_all_formats(fig, layout.figure_base(PANEL_FILENAMES[key]))
        plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DMS Layer1 medium-overlap support mechanism experiment.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--early-window-ms", type=float, default=DEFAULT_EARLY_WINDOW_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--min-overlap-area", type=int, default=DEFAULT_MIN_OVERLAP_AREA)
    parser.add_argument("--min-probe-only-area", type=int, default=DEFAULT_MIN_PROBE_ONLY_AREA)
    parser.add_argument("--medium-overlap-q-low", type=float, default=DEFAULT_MEDIUM_Q_LOW)
    parser.add_argument("--medium-overlap-q-high", type=float, default=DEFAULT_MEDIUM_Q_HIGH)
    parser.add_argument("--min-area-gap", type=int, default=0)
    parser.add_argument("--drive-score-threshold", type=float, default=DEFAULT_DRIVE_SCORE_THRESHOLD)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= float(args.medium_overlap_q_low) < float(args.medium_overlap_q_high) <= 1.0):
        raise ValueError("medium-overlap quantiles must satisfy 0 <= low < high <= 1")

    seed_everything(int(args.seed))
    apply_publication_style()
    device = resolve_device(args.device)
    layout = prepare_result_layout(args.output_dir)
    sample_steps = int(round((float(args.sample_ms) * ms) / DT))
    delay_steps = int(round((float(args.delay_ms) * ms) / DT))
    probe_steps = int(round((float(args.probe_ms) * ms) / DT))
    early_window_steps = max(1, int(round((float(args.early_window_ms) * ms) / DT)))

    dataset = load_mnist_skeleton_dataset(args.dataset_root, args.dataset_split)
    images, labels, _ = build_dataset_arrays(dataset)
    class_index = build_class_index(dataset, num_classes=10)
    trials = construct_medium_trials(
        images,
        labels,
        class_index,
        max_probes=int(args.max_probes),
        max_pairs=int(args.max_pairs),
        foreground_threshold=float(args.foreground_threshold),
        min_overlap_area=int(args.min_overlap_area),
        min_probe_only_area=int(args.min_probe_only_area),
        q_low=float(args.medium_overlap_q_low),
        q_high=float(args.medium_overlap_q_high),
        seed=int(args.seed),
    )

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=DT,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms), 100.0),
    )
    kernels_cpu = net.layer1.kernels.detach().cpu().to(torch.float32)
    static_gain = float(net.layer1.stsp_U)

    save_run_config(
        {
            "experiment": EXPERIMENT_NAME,
            "scientific_target": "Layer1 firing-pattern reordering",
            # smoke experiment should be run in torch_env
            "smoke_note": "smoke experiment should be run in torch_env",
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "probe_ms": float(args.probe_ms),
            "early_window_ms": float(args.early_window_ms),
            "medium_overlap_q_low": float(args.medium_overlap_q_low),
            "medium_overlap_q_high": float(args.medium_overlap_q_high),
            "device": str(device),
            "min_area_gap_ignored": int(args.min_area_gap),
            "save_case_count_unused": int(args.save_case_count),
        },
        layout.root,
    )

    pair_metadata_csv = save_tidy_csv(trial_metadata_table(trials), layout.data_file("pair_metadata.csv"), sort_by=["trial_id"])
    pair_mask_json = layout.data_file("pair_mask_metadata.json")
    pair_mask_json.write_text(json.dumps(trial_mask_payload(trials), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    preprobe_rows: list[dict[str, object]] = []
    drive_rows: list[dict[str, object]] = []
    firing_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    local_pair_rows: list[dict[str, object]] = []
    local_support_event_rows: list[dict[str, object]] = []
    chain_event_rows: list[dict[str, object]] = []
    aligned_event_rows: list[dict[str, object]] = []
    exemplar_pair: dict[str, object] | None = None
    panel_a_case_arrays: dict[str, object] | None = None
    n_batches = math.ceil(len(trials) / int(args.batch_size))
    for start in tqdm(range(0, len(trials), int(args.batch_size)), total=n_batches, desc="Running Fig4 Layer1 medium-overlap"):
        batch = list(trials[start : start + int(args.batch_size)])
        probe_ids = [t.probe_id for t in batch]
        sample_ids = [t.sample_id for t in batch]
        probe_spikes = _stack_encoded(images, probe_ids, encoder=encoder, steps=probe_steps, device=device)
        sample_spikes = _stack_encoded(images, sample_ids, encoder=encoder, steps=sample_steps, device=device)
        probe_spikes_cpu = probe_spikes.detach().cpu()
        med_dynamic = run_rollout_batch(net=net, sample_spikes=sample_spikes, probe_spikes=probe_spikes, delay_steps=delay_steps, stsp_mode="dynamic")
        med_static = run_rollout_batch(net=net, sample_spikes=sample_spikes, probe_spikes=probe_spikes, delay_steps=delay_steps, stsp_mode="static_frozen")
        for batch_idx, trial in enumerate(batch):
            preprobe_rows.append(summarize_preprobe_stsp(trial=trial, boundary_state=med_dynamic["boundary_states"]["pre_intervention"]["layer1"], batch_idx=batch_idx, model_type="dynamic"))
            preprobe_rows.append(summarize_preprobe_stsp(trial=trial, boundary_state=med_static["boundary_states"]["pre_intervention"]["layer1"], batch_idx=batch_idx, model_type="static"))
            if panel_a_case_arrays is None and int(trial.trial_id) == 0:
                panel_a_case_arrays = _build_panel_a_case_payload(
                    trial=trial,
                    dynamic_boundary_state=med_dynamic["boundary_states"]["pre_intervention"]["layer1"],
                    static_boundary_state=med_static["boundary_states"]["pre_intervention"]["layer1"],
                    batch_idx=batch_idx,
                )
            d_rows, f_rows, i_rows, l_rows = summarize_trial(
                trial=trial,
                dynamic_output=med_dynamic,
                static_output=med_static,
                batch_idx=batch_idx,
                early_window_steps=early_window_steps,
                drive_score_threshold=float(args.drive_score_threshold),
                kernels_cpu=kernels_cpu,
                probe_spikes_cpu=probe_spikes_cpu,
                static_gain=static_gain,
            )
            drive_rows.extend(d_rows)
            firing_rows.extend(f_rows)
            input_rows.extend(i_rows)
            loss_rows.extend(l_rows)
            pair_rows, support_rows, chain_rows, aligned_rows, exemplar_candidate = build_local_winner_loser_analysis(
                trial=trial,
                dynamic_output=med_dynamic,
                static_output=med_static,
                batch_idx=batch_idx,
                early_window_steps=early_window_steps,
                kernels_cpu=kernels_cpu,
                probe_spikes_cpu=probe_spikes_cpu,
                static_gain=static_gain,
                drive_score_threshold=float(args.drive_score_threshold),
            )
            local_pair_rows.extend(pair_rows)
            local_support_event_rows.extend(support_rows)
            chain_event_rows.extend(chain_rows)
            aligned_event_rows.extend(aligned_rows)
            if exemplar_candidate is not None and (
                exemplar_pair is None
                or float(exemplar_candidate["winner_loser_contrast_shift"]) > float(exemplar_pair["winner_loser_contrast_shift"])
            ):
                exemplar_pair = exemplar_candidate

    df_preprobe = pd.DataFrame(preprobe_rows).sort_values(["trial_id", "model_type"], kind="stable").reset_index(drop=True)
    df_drive = pd.DataFrame(drive_rows).sort_values(["trial_id", "unit_idx"], kind="stable").reset_index(drop=True)
    df_firing = _append_pooled(
        pd.DataFrame(firing_rows).sort_values(["trial_id", "unit_group"], kind="stable").reset_index(drop=True),
        weight_col="n_units",
        group_cols=["unit_group"],
        value_cols=["P_advance", "P_recruit", "P_loss", "P_unchanged", "delta_early_spike_count", "delta_first_spike_latency"],
        count_cols=["n_advance", "n_recruit", "n_loss", "n_unchanged"],
    )
    df_input = _append_pooled(
        pd.DataFrame(input_rows).sort_values(["trial_id", "unit_group"], kind="stable").reset_index(drop=True),
        weight_col="n_units_selected",
        group_cols=["unit_group", "transition_focus"],
        value_cols=[
            "overlap_input_dynamic",
            "overlap_input_static",
            "probe_only_input_dynamic",
            "probe_only_input_static",
            "overlap_input_gain",
            "probe_only_input_gain",
            "input_selectivity_gain",
        ],
        count_cols=[],
    )
    df_loss = _append_pooled(
        pd.DataFrame(loss_rows).sort_values(["trial_id", "unit_group"], kind="stable").reset_index(drop=True),
        weight_col="n_lost_spike_units",
        group_cols=["unit_group"],
        value_cols=["lost_spike_delta_inh", "winner_loser_latency_gap", "post_winner_inhibition_rise"],
        count_cols=[],
    )
    df_local_pairs = _ensure_dataframe(
        local_pair_rows,
        columns=[
            "trial_id",
            "winner_unit_idx",
            "loser_unit_idx",
            "winner_group",
            "loser_group",
            "winner_transition",
            "loser_transition",
            "winner_row",
            "winner_col",
            "loser_row",
            "loser_col",
            "winner_first_spike_dynamic",
            "winner_first_spike_static",
            "loser_first_spike_dynamic",
            "loser_first_spike_static",
            "winner_overlap_input_gain",
            "winner_probe_only_input_gain",
            "loser_overlap_input_gain",
            "loser_probe_only_input_gain",
            "contrast_time_index",
            "contrast_dynamic",
            "contrast_static",
            "winner_loser_contrast_shift",
            "winner_priority_class",
            "local_radius",
        ],
    ).sort_values(["trial_id", "loser_unit_idx", "winner_unit_idx"], kind="stable").reset_index(drop=True)
    df_chain = _ensure_dataframe(
        chain_event_rows,
        columns=[
            "trial_id",
            "winner_unit_idx",
            "loser_unit_idx",
            "winner_group",
            "loser_group",
            "align_time_index",
            "winner_first_spike_dynamic",
            "winner_first_spike_static",
            "loser_first_spike_dynamic",
            "loser_first_spike_static",
            "winner_pre_spike_boost",
            "winner_spikes_earlier",
            "loser_post_winner_suppressed",
            "full_chain_satisfied",
            "winner_pre_spike_delta_v_mean",
            "loser_post_winner_delta_v_mean",
            "loser_pre_winner_inh_before_mean",
            "loser_post_winner_inh_before_mean",
            "loser_post_winner_inh_rise",
        ],
    ).sort_values(["trial_id", "loser_unit_idx", "winner_unit_idx"], kind="stable").reset_index(drop=True)
    df_local_support = _build_local_support_summary(
        _ensure_dataframe(
            local_support_event_rows,
            columns=[
                "aggregation_scope",
                "trial_id",
                "loser_unit_idx",
                "loser_row",
                "loser_col",
                "loser_group",
                "supported",
                "local_winner_support_rate",
                "n_loser_events",
                "n_supported_events",
            ],
        ).sort_values(["trial_id", "loser_unit_idx"], kind="stable").reset_index(drop=True)
    )
    aligned_event_payload = _stack_aligned_event_rows(aligned_event_rows)

    preprobe_csv = save_tidy_csv(df_preprobe, layout.data_file("preprobe_stsp_summary.csv"), sort_by=["trial_id", "model_type"])
    drive_csv = save_tidy_csv(df_drive, layout.data_file("l1_drive_group_summary.csv"), sort_by=["trial_id", "unit_idx"])
    firing_csv = save_tidy_csv(df_firing, layout.data_file("l1_firing_transition_summary.csv"), sort_by=["aggregation_scope", "trial_id", "unit_group"])
    input_csv = save_tidy_csv(df_input, layout.data_file("l1_input_source_gain_summary.csv"), sort_by=["aggregation_scope", "trial_id", "unit_group", "transition_focus"])
    loss_csv = save_tidy_csv(df_loss, layout.data_file("l1_loss_inhibition_summary.csv"), sort_by=["aggregation_scope", "trial_id", "unit_group"])
    local_pairs_csv = save_tidy_csv(df_local_pairs, layout.data_file("l1_local_winner_loser_pairs.csv"), sort_by=["trial_id", "loser_unit_idx", "winner_unit_idx"])
    causal_chain_csv = save_tidy_csv(df_chain, layout.data_file("l1_local_causal_chain_events.csv"), sort_by=["trial_id", "loser_unit_idx", "winner_unit_idx"])
    local_support_csv = save_tidy_csv(df_local_support, layout.data_file("l1_local_winner_support_summary.csv"), sort_by=["aggregation_scope", "trial_id", "loser_unit_idx"])
    exemplar_trace_npz = None
    panel_a_case_npz = None
    aligned_event_npz = layout.data_file("l1_local_event_time_alignment.npz")
    np.savez_compressed(aligned_event_npz, **aligned_event_payload)
    if exemplar_pair is not None:
        exemplar_trace_npz = layout.data_file("l1_local_winner_loser_exemplar_trace.npz")
        np.savez_compressed(exemplar_trace_npz, **exemplar_pair)
    if panel_a_case_arrays is not None:
        panel_a_case_npz = layout.data_file("l1_panel_a_preprobe_gain_map.npz")
        np.savez_compressed(panel_a_case_npz, **panel_a_case_arrays)

    case = trials[0]
    panels = {
        "panel_a_overlap_definition": _overlap_definition_panel(images=images, trial=case, panel_a_case=panel_a_case_arrays),
    }
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    for model, offset, color in (("dynamic", -0.12, COLOR_DYNAMIC), ("static", 0.12, COLOR_STATIC)):
        sub = df_preprobe[df_preprobe["model_type"] == model]
        for row in sub.itertuples(index=False):
            ax.plot([0.0 + offset, 1.0 + offset], [float(row.ux_overlap_pre), float(row.ux_probe_only_pre)], color=color, alpha=0.25, linewidth=0.9)
        _scatter_with_mean(ax, 0.0 + offset, sub["ux_overlap_pre"].to_numpy(dtype=np.float64), color)
        _scatter_with_mean(ax, 1.0 + offset, sub["ux_probe_only_pre"].to_numpy(dtype=np.float64), color)
    ax.set_xticks([0.0, 1.0]); ax.set_xticklabels(["overlap", "probe-only"]); ax.set_ylabel("Pre-probe u*x"); _strip_axis(ax); fig.tight_layout(); panels["panel_b_preprobe_ux_overlap_vs_probeonly"] = fig
    fig, ax = plt.subplots(figsize=PUBLICATION_SINGLE_COLUMN_FIGSIZE)
    _scatter_with_mean(ax, 0.0, df_preprobe[df_preprobe["model_type"] == "dynamic"]["support_area"].to_numpy(dtype=np.float64), COLOR_DYNAMIC)
    ax.set_xticks([0.0]); ax.set_xticklabels(["medium"]); ax.set_ylabel("Support area"); _strip_axis(ax); fig.tight_layout(); panels["panel_c_support_area"] = fig
    panels["panel_d_mean_ux_on_overlap"] = _two_condition_panel(df_preprobe, "mean_ux_on_overlap", "Mean u*x on overlap")
    panels["panel_e_total_memory_support"] = _two_condition_panel(df_preprobe, "total_memory_support", "Total memory support")
    panels["panel_f_p_advance"] = _group_panel(df_firing, "P_advance", "P(advance)")
    panels["panel_g_p_recruit"] = _group_panel(df_firing, "P_recruit", "P(recruit)")
    panels["panel_h_p_loss"] = _group_panel(df_firing, "P_loss", "P(loss)")
    panels["panel_i_delta_early_spike_count"] = _group_panel(df_firing, "delta_early_spike_count", "Delta early spike count")
    panels["panel_j_delta_first_spike_latency"] = _group_panel(df_firing, "delta_first_spike_latency", "Delta first-spike latency")
    focus = df_input[df_input["transition_focus"] == "advance_or_recruit"]
    panels["panel_k_overlap_input_gain"] = _group_panel(focus, "overlap_input_gain", "Overlap-source input gain")
    panels["panel_l_probe_only_input_gain"] = _group_panel(focus, "probe_only_input_gain", "Probe-only input gain")
    panels["panel_m_input_selectivity_gain"] = _group_panel(focus, "input_selectivity_gain", "Input selectivity gain")
    panels["panel_n_lost_spike_delta_inhibition"] = _group_panel(df_loss, "lost_spike_delta_inh", "Lost-spike delta inhibition")
    panels["panel_n1_n_lost_spike_units"] = _group_panel(df_loss, "n_lost_spike_units", "Lost spike units")
    panels["panel_o_local_winner_loser_voltage_trace"] = _local_voltage_trace_panel(exemplar_pair)
    panels["panel_p_local_winner_support_rate"] = _trial_rate_panel(df_local_support)
    panels["panel_q_winner_loser_contrast_shift"] = _contrast_shift_panel(df_local_pairs)
    panels["panel_r_event_time_mechanism"] = _event_time_mechanism_panel(aligned_event_payload)
    panels["panel_s_causal_chain_prevalence"] = _causal_chain_prevalence_panel(df_chain)
    panel_paths = {} if bool(args.skip_figures) else save_all_panels(layout, panels)

    summary_json = save_summary_json(
        {
            "experiment": EXPERIMENT_NAME,
            "scientific_target": "Layer1 firing-pattern reordering",
            "mechanism_extension": "Local winner-loser chain with overlap-enhanced local winner support.",
            "presentation_note": "Feedforward group labels are presented as overlap-biased versus probe-only-biased Layer1 units. These are intermediate bias-conditioned groups used to bridge overlap-defined latent support to downstream winner-loser competition, not the main competition actors themselves.",
            "trial_selection": {
                "n_trials": int(len(trials)),
                "definition": "Fix probe and choose a medium-overlap sample from the middle overlap quantile band with non-trivial probe-only area.",
            },
            "comparison": {"dynamic_condition": "med_dynamic", "static_condition": "med_static"},
            "mechanism_design": {
                "layer1_only_probe_reset": True,
                "high_low_removed": True,
                "downstream_removed": True,
                "feedforward_bias_grouping": "Drive-score groups retain the original overlap_dominant and probe_only_dominant internal keys, but are presented as overlap-biased and probe-only-biased receiving-input units.",
                "local_winner_definition": "Loser-centered local neighborhood within the 5x5 Layer1 inhibition support.",
                "causal_chain_definition": "winner pre-spike boost -> winner spikes earlier -> loser direct inhibition rises after winner spike",
            },
            "tables": {
                "pair_metadata_csv": pair_metadata_csv,
                "pair_mask_metadata_json": str(pair_mask_json),
                "preprobe_stsp_summary_csv": preprobe_csv,
                "l1_drive_group_summary_csv": drive_csv,
                "l1_firing_transition_summary_csv": firing_csv,
                "l1_input_source_gain_summary_csv": input_csv,
                "l1_loss_inhibition_summary_csv": loss_csv,
                "l1_local_winner_loser_pairs_csv": local_pairs_csv,
                "l1_local_causal_chain_events_csv": causal_chain_csv,
                "l1_local_winner_support_summary_csv": local_support_csv,
                "l1_local_event_time_alignment_npz": str(aligned_event_npz),
                "l1_local_winner_loser_exemplar_trace_npz": str(exemplar_trace_npz) if exemplar_trace_npz is not None else None,
                "l1_panel_a_preprobe_gain_map_npz": str(panel_a_case_npz) if panel_a_case_npz is not None else None,
            },
            "panel_paths": panel_paths,
        },
        layout.root,
    )
    save_log_lines(
        [
            f"experiment={EXPERIMENT_NAME}",
            f"n_trials={len(trials)}",
            f"pair_metadata_csv={pair_metadata_csv}",
            f"preprobe_stsp_summary_csv={preprobe_csv}",
            f"l1_drive_group_summary_csv={drive_csv}",
            f"l1_firing_transition_summary_csv={firing_csv}",
            f"l1_input_source_gain_summary_csv={input_csv}",
            f"l1_loss_inhibition_summary_csv={loss_csv}",
            f"l1_local_winner_loser_pairs_csv={local_pairs_csv}",
            f"l1_local_causal_chain_events_csv={causal_chain_csv}",
            f"l1_local_winner_support_summary_csv={local_support_csv}",
            f"l1_local_event_time_alignment_npz={aligned_event_npz}",
            "smoke_note=smoke experiment should be run in torch_env",
            f"summary_json={summary_json}",
        ],
        layout.log_dir,
    )
    print(f"[Done] Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
