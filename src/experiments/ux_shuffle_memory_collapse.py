from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.common.dataset import build_class_index as shared_build_class_index
from src.experiments.common.dataset import encode_images as shared_encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3 as shared_decode_prediction_and_fire_time_from_layer3
from src.experiments.common.model_io import load_model_and_encoder as shared_load_model_and_encoder
from src.experiments.common.results import prepare_result_layout
from src.experiments.common.runtime import seed_everything
from src.experiments.ping_memory.shared.shuffle_metrics import (
    compute_bias_table as shared_compute_bias_table,
    compute_collapse_summary as shared_compute_collapse_summary,
    compute_condition_metrics as shared_compute_condition_metrics,
)
from src.plotting.common.io import save_figure_all_formats, save_tidy_csv
from src.plotting.experiments.ux_shuffle_memory_collapse_plot_lib import (
    build_memory_readout_target_figure,
    write_plot_bundle_manifest,
)
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import ms


CONDITION_A_DYNAMIC_BASE = "A_dynamic_base"
CONDITION_B_TRIAL_SHUFFLE_SPIKE = "B_trial_shuffle_spike"
CONDITION_C_TRIAL_SHUFFLE_MEMBRANE = "C_trial_shuffle_membrane"
CONDITION_D_TRIAL_SHUFFLE_UX = "D_trial_shuffle_ux"
CONDITION_E_STATIC_FROZEN = "E_static_frozen"

SUBSTRATE_SPIKE = "spike"
SUBSTRATE_MEMBRANE = "membrane"
SUBSTRATE_STSP = "stsp"
SUBSTRATE_ORDER = [SUBSTRATE_SPIKE, SUBSTRATE_MEMBRANE, SUBSTRATE_STSP]

CONDITION_ORDER = [
    CONDITION_A_DYNAMIC_BASE,
    CONDITION_B_TRIAL_SHUFFLE_SPIKE,
    CONDITION_C_TRIAL_SHUFFLE_MEMBRANE,
    CONDITION_D_TRIAL_SHUFFLE_UX,
    CONDITION_E_STATIC_FROZEN,
]

CONDITION_LABELS = {
    CONDITION_A_DYNAMIC_BASE: "A: dynamic",
    CONDITION_B_TRIAL_SHUFFLE_SPIKE: "B: trial-shuffle spike-state",
    CONDITION_C_TRIAL_SHUFFLE_MEMBRANE: "C: trial-shuffle membrane",
    CONDITION_D_TRIAL_SHUFFLE_UX: "D: trial-shuffle u/x",
    CONDITION_E_STATIC_FROZEN: "E: static frozen",
}

LAYER_KEYS = ["layer1", "layer2", "layer3"]
DEFAULT_RESULTS_DIR = os.path.join("results", "ux_shuffle_memory_collapse")
SHUFFLE_CONDITIONS = [
    CONDITION_B_TRIAL_SHUFFLE_SPIKE,
    CONDITION_C_TRIAL_SHUFFLE_MEMBRANE,
    CONDITION_D_TRIAL_SHUFFLE_UX,
]
CONDITION_TO_SUBSTRATE = {
    CONDITION_B_TRIAL_SHUFFLE_SPIKE: SUBSTRATE_SPIKE,
    CONDITION_C_TRIAL_SHUFFLE_MEMBRANE: SUBSTRATE_MEMBRANE,
    CONDITION_D_TRIAL_SHUFFLE_UX: SUBSTRATE_STSP,
}
SUBSTRATE_TO_CONDITION = {v: k for k, v in CONDITION_TO_SUBSTRATE.items()}


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int((self.sample_ms * ms) / self.dt)

    @property
    def delay_steps(self) -> int:
        return int((self.delay_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def setup_logger(log_path: Path, experiment_name: str) -> logging.Logger:
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(data, path: Path, logger: logging.Logger | None = None) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, indent=2, ensure_ascii=False, sort_keys=True)
    if logger is not None:
        logger.info("[Save] JSON saved to %s", path)
    return path


def generate_trial_specs(
    class_index: Dict[int, List[int]],
    num_trials: int,
    num_classes: int,
    rng: random.Random,
) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    classes = list(range(num_classes))
    for trial_id in range(num_trials):
        sample_label = rng.choice(classes)
        probe_candidates = [c for c in classes if c != sample_label]
        probe_label = rng.choice(probe_candidates)
        sample_index = rng.choice(class_index[sample_label])
        probe_index = rng.choice(class_index[probe_label])
        rows.append(
            {
                "trial_id": int(trial_id),
                "sample_index": int(sample_index),
                "sample_label": int(sample_label),
                "probe_index": int(probe_index),
                "probe_label": int(probe_label),
            }
        )
    return pd.DataFrame(rows)


def validate_trial_specs(df_specs: pd.DataFrame, num_classes: int) -> None:
    if df_specs["trial_id"].nunique() != len(df_specs):
        raise ValueError("trial_id must be unique in trial specs")
    if not np.all(df_specs["sample_label"].to_numpy() != df_specs["probe_label"].to_numpy()):
        raise ValueError("Found trial(s) where sample_label == probe_label")
    for col in ["sample_label", "probe_label"]:
        vals = df_specs[col].to_numpy()
        if (vals < 0).any() or (vals >= num_classes).any():
            raise ValueError(f"{col} contains out-of-range class index")


def _batch_index_select(tensor: torch.Tensor, donor_batch_index: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(donor_batch_index, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, idx).contiguous()


def _copy_reordered_in_place(tensor: torch.Tensor, donor_batch_index: np.ndarray) -> None:
    tensor.copy_(_batch_index_select(tensor, donor_batch_index))


def _tensors_match_exact(live: torch.Tensor, saved: torch.Tensor) -> bool:
    if live.dtype.is_floating_point or live.dtype.is_complex:
        return bool(torch.allclose(live, saved, atol=0.0, rtol=0.0, equal_nan=True))
    return bool(torch.equal(live, saved))


def _capture_substrate_state(net: SDNN_Network, target_substrate: str) -> Tuple[Dict[str, Dict[str, torch.Tensor]], int]:
    state_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    restore_ok = 1
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None:
            continue
        captured: Dict[str, torch.Tensor] = {}
        if target_substrate == SUBSTRATE_STSP:
            if not getattr(layer, "enable_stsp", False):
                continue
            if layer.u_pre is None or layer.x_pre is None:
                restore_ok = 0
                continue
            captured["u_pre"] = layer.u_pre.detach().clone()
            captured["x_pre"] = layer.x_pre.detach().clone()
        elif target_substrate == SUBSTRATE_MEMBRANE:
            if layer.v_mem is None:
                restore_ok = 0
                continue
            captured["v_mem"] = layer.v_mem.detach().clone()
        elif target_substrate == SUBSTRATE_SPIKE:
            if layer.g_e is None or layer.res is None:
                restore_ok = 0
                continue
            captured["g_e"] = layer.g_e.detach().clone()
            captured["res"] = layer.res.detach().clone()
            inh_trace = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
            if inh_trace is not None:
                captured["lateral_inh.inh_trace"] = inh_trace.detach().clone()
        else:
            raise ValueError(f"Unsupported target_substrate={target_substrate}")
        if captured:
            state_cache[layer_key] = captured
    if target_substrate == SUBSTRATE_STSP and len(state_cache) == 0:
        restore_ok = 0
    return state_cache, restore_ok


def _restore_substrate_state(
    net: SDNN_Network,
    target_substrate: str,
    state_cache: Dict[str, Dict[str, torch.Tensor]],
) -> int:
    restore_ok = 1
    for layer_key, state_items in state_cache.items():
        layer = getattr(net, layer_key, None)
        if layer is None:
            restore_ok = 0
            continue
        for state_name, saved in state_items.items():
            if state_name == "lateral_inh.inh_trace":
                live = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
            else:
                live = getattr(layer, state_name, None)
            if live is None or tuple(live.shape) != tuple(saved.shape):
                restore_ok = 0
                continue
            live.copy_(saved)
            if not _tensors_match_exact(live, saved):
                restore_ok = 0
    if target_substrate == SUBSTRATE_STSP:
        for layer_key in state_cache:
            layer = getattr(net, layer_key, None)
            if layer is None or layer.u_pre is None or layer.x_pre is None:
                restore_ok = 0
    return restore_ok


def _reset_all_layer_states(net: SDNN_Network, layer_input_shapes: Dict[str, Tuple[int, int, int, int]]) -> None:
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None:
            continue
        layer.reset_state(layer_input_shapes[layer_key])


def _apply_legacy_layer3_probe_phase_reset(net: SDNN_Network) -> None:
    net.layer3.reset_decision_state()
    with torch.no_grad():
        if getattr(net.layer3, "v_mem", None) is not None:
            net.layer3.v_mem.fill_(net.layer3.V_L)
        if hasattr(net.layer3, "lateral_inh"):
            net.layer3.lateral_inh.reset_state(net.layer3.output_shape)


def run_dms_session_with_intervention(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str,
    intervention_fn: Optional[Callable[[SDNN_Network, Dict[str, np.ndarray]], Dict[str, int]]] = None,
    batch_meta: Optional[Dict[str, np.ndarray]] = None,
    pure_substrate_only: bool = False,
    target_substrate: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if batch_meta is None:
        batch_meta = {}

    batch_size, t_sample, c, h, w = sample_spikes.shape
    t_probe = probe_spikes.shape[1]

    net.layer1.reset_state((batch_size, c, h, w))
    h1 = (h + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (w + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    net.layer2.reset_state((batch_size, net.layer1.out_channels, h1_p, w1_p))

    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    net.layer3.reset_state((batch_size, net.layer2.out_channels, h2_p, w2_p))
    layer_input_shapes = {
        "layer1": (batch_size, c, h, w),
        "layer2": (batch_size, net.layer1.out_channels, h1_p, w1_p),
        "layer3": (batch_size, net.layer2.out_channels, h2_p, w2_p),
    }

    current_time = 0
    zero_input = torch.zeros((batch_size, c, h, w), device=sample_spikes.device)
    intervention_record: Dict[str, int] = {}

    def step_network(input_t: torch.Tensor, force_l3_time: Optional[int] = None) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())
        t_for_l3 = current_time if force_l3_time is None else force_l3_time
        net.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)
        current_time += 1

    for t in range(t_sample):
        step_network(sample_spikes[:, t, ...])

    for _ in range(delay_steps):
        step_network(zero_input)

    if intervention_fn is not None:
        intervention_record = intervention_fn(net, batch_meta)

    restore_ok = 1
    reset_applied = 0
    legacy_phase_reset_applied = 0
    if target_substrate is not None and pure_substrate_only:
        with torch.no_grad():
            state_cache, restore_ok = _capture_substrate_state(net, target_substrate)
            _reset_all_layer_states(net, layer_input_shapes)
            restore_ok = min(restore_ok, _restore_substrate_state(net, target_substrate, state_cache))
        reset_applied = 1
        net.layer3.reset_decision_state()
    elif target_substrate is not None:
        net.layer3.reset_decision_state()
    else:
        _apply_legacy_layer3_probe_phase_reset(net)
        legacy_phase_reset_applied = 1

    for t in range(t_probe):
        step_network(probe_spikes[:, t, ...], force_l3_time=t)

    prediction_probe, first_fire_t_probe = shared_decode_prediction_and_fire_time_from_layer3(net, batch_size)
    return {
        "prediction_probe": prediction_probe.detach().cpu(),
        "first_fire_t_probe": first_fire_t_probe.detach().cpu(),
        "intervention_record": intervention_record,
        "restore_ok": torch.tensor(restore_ok, dtype=torch.long),
        "reset_applied": torch.tensor(reset_applied, dtype=torch.long),
        "legacy_phase_reset_applied": torch.tensor(legacy_phase_reset_applied, dtype=torch.long),
    }


def _build_constrained_permutation(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: random.Random,
    require_no_self: bool,
) -> Optional[np.ndarray]:
    n = len(sample_labels)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    candidates: List[List[int]] = []
    for recv_i in range(n):
        cand = [j for j in range(n) if (sample_labels[j] != probe_labels[recv_i]) and (not require_no_self or j != recv_i)]
        candidates.append(cand)
        if len(cand) == 0:
            return None

    order = sorted(range(n), key=lambda i: len(candidates[i]))
    donor_for_recv = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=np.bool_)

    def dfs(depth: int) -> bool:
        if depth == n:
            return True
        recv_i = order[depth]
        cand = candidates[recv_i][:]
        rng.shuffle(cand)
        for donor_j in cand:
            if used[donor_j]:
                continue
            used[donor_j] = True
            donor_for_recv[recv_i] = donor_j
            if dfs(depth + 1):
                return True
            used[donor_j] = False
            donor_for_recv[recv_i] = -1
        return False

    return donor_for_recv if dfs(0) else None


def build_trial_shuffle_plan(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(sample_labels)
    identity = np.arange(n, dtype=np.int64)
    if n <= 1:
        return identity, {"n_self_swap": n, "used_relaxed_rule": 1}

    donor_idx = _build_constrained_permutation(sample_labels, probe_labels, rng, require_no_self=True)
    used_relaxed = 0
    if donor_idx is None:
        donor_idx = _build_constrained_permutation(sample_labels, probe_labels, rng, require_no_self=False)
        used_relaxed = 1
    if donor_idx is None:
        raise RuntimeError("Failed to build a valid trial-shuffle mapping for this batch.")

    donor_sample = sample_labels[donor_idx]
    if np.any(donor_sample == probe_labels):
        raise RuntimeError("Invalid shuffle plan: donor sample label equals receiver probe label.")

    n_self_swap = int(np.sum(donor_idx == identity))
    return donor_idx, {"n_self_swap": n_self_swap, "used_relaxed_rule": used_relaxed}


def apply_trial_shuffle_ux_in_place(net: SDNN_Network, donor_batch_index: np.ndarray) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is None or (not getattr(layer, "enable_stsp", False)):
                continue
            if layer.u_pre is None or layer.x_pre is None:
                continue
            if layer.u_pre.shape[0] != len(donor_batch_index):
                raise ValueError(
                    f"Layer {layer_key} batch mismatch: u_pre.shape[0]={layer.u_pre.shape[0]} "
                    f"vs permutation={len(donor_batch_index)}"
                )
            _copy_reordered_in_place(layer.u_pre, donor_batch_index)
            _copy_reordered_in_place(layer.x_pre, donor_batch_index)


def apply_trial_shuffle_membrane_in_place(net: SDNN_Network, donor_batch_index: np.ndarray) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is None or layer.v_mem is None:
                continue
            if layer.v_mem.shape[0] != len(donor_batch_index):
                raise ValueError(
                    f"Layer {layer_key} batch mismatch: v_mem.shape[0]={layer.v_mem.shape[0]} "
                    f"vs permutation={len(donor_batch_index)}"
                )
            _copy_reordered_in_place(layer.v_mem, donor_batch_index)


def apply_trial_shuffle_spike_state_in_place(net: SDNN_Network, donor_batch_index: np.ndarray) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is None:
                continue
            if layer.g_e is not None:
                if layer.g_e.shape[0] != len(donor_batch_index):
                    raise ValueError(
                        f"Layer {layer_key} batch mismatch: g_e.shape[0]={layer.g_e.shape[0]} "
                        f"vs permutation={len(donor_batch_index)}"
                    )
                _copy_reordered_in_place(layer.g_e, donor_batch_index)
            if layer.res is not None:
                if layer.res.shape[0] != len(donor_batch_index):
                    raise ValueError(
                        f"Layer {layer_key} batch mismatch: res.shape[0]={layer.res.shape[0]} "
                        f"vs permutation={len(donor_batch_index)}"
                    )
                _copy_reordered_in_place(layer.res, donor_batch_index)
            inh_trace = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
            if inh_trace is not None:
                if inh_trace.shape[0] != len(donor_batch_index):
                    raise ValueError(
                        f"Layer {layer_key} batch mismatch: inh_trace.shape[0]={inh_trace.shape[0]} "
                        f"vs permutation={len(donor_batch_index)}"
                    )
                _copy_reordered_in_place(inh_trace, donor_batch_index)


def run_experiment(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
    pure_substrate_only: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    all_records: List[Dict[str, int]] = []
    for start in tqdm(range(0, len(df_specs), batch_size), desc="UXCollapse Batches"):
        batch = df_specs.iloc[start : start + batch_size]
        bsz = len(batch)
        sample_lbl_np = batch["sample_label"].to_numpy(dtype=np.int64)
        probe_lbl_np = batch["probe_label"].to_numpy(dtype=np.int64)
        trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)

        sample_imgs = torch.stack([dataset[int(i)][0] for i in batch["sample_index"].tolist()], dim=0).to(device)
        probe_imgs = torch.stack([dataset[int(i)][0] for i in batch["probe_index"].tolist()], dim=0).to(device)
        sample_spikes = shared_encode_images(encoder, sample_imgs, spec.sample_steps)
        probe_spikes = shared_encode_images(encoder, probe_imgs, spec.probe_steps)

        donor_idx_b, plan_info = build_trial_shuffle_plan(sample_lbl_np, probe_lbl_np, rng)
        batch_meta: Dict[str, np.ndarray] = {
            "trial_id": trial_ids,
            "sample_label": sample_lbl_np,
            "probe_label": probe_lbl_np,
            "donor_batch_index": donor_idx_b,
            "used_relaxed_rule": np.full(bsz, plan_info["used_relaxed_rule"], dtype=np.int64),
        }

        def make_trial_shuffle_intervention(target_substrate: str) -> Callable[[SDNN_Network, Dict[str, np.ndarray]], Dict[str, int]]:
            shuffle_fn = {
                SUBSTRATE_SPIKE: apply_trial_shuffle_spike_state_in_place,
                SUBSTRATE_MEMBRANE: apply_trial_shuffle_membrane_in_place,
                SUBSTRATE_STSP: apply_trial_shuffle_ux_in_place,
            }[target_substrate]

            def _apply(local_net: SDNN_Network, meta: Dict[str, np.ndarray]) -> Dict[str, int]:
                donor_idx = np.asarray(meta["donor_batch_index"], dtype=np.int64)
                shuffle_fn(local_net, donor_idx)
                return {
                    "applied": 1,
                    "n_self_swap": int(np.sum(donor_idx == np.arange(len(donor_idx), dtype=np.int64))),
                    "used_relaxed_rule": int(plan_info["used_relaxed_rule"]),
                }

            return _apply

        condition_runs = [
            (CONDITION_A_DYNAMIC_BASE, "dynamic", None, None, False),
            (
                CONDITION_B_TRIAL_SHUFFLE_SPIKE,
                "dynamic",
                make_trial_shuffle_intervention(SUBSTRATE_SPIKE),
                SUBSTRATE_SPIKE,
                pure_substrate_only,
            ),
            (
                CONDITION_C_TRIAL_SHUFFLE_MEMBRANE,
                "dynamic",
                make_trial_shuffle_intervention(SUBSTRATE_MEMBRANE),
                SUBSTRATE_MEMBRANE,
                pure_substrate_only,
            ),
            (
                CONDITION_D_TRIAL_SHUFFLE_UX,
                "dynamic",
                make_trial_shuffle_intervention(SUBSTRATE_STSP),
                SUBSTRATE_STSP,
                pure_substrate_only,
            ),
            (CONDITION_E_STATIC_FROZEN, "static_frozen", None, None, False),
        ]

        for condition_name, stsp_mode, intervention_fn, target_substrate, pure_this_condition in condition_runs:
            with torch.no_grad():
                out = run_dms_session_with_intervention(
                    net=net,
                    sample_spikes=sample_spikes,
                    probe_spikes=probe_spikes,
                    delay_steps=spec.delay_steps,
                    stsp_mode=stsp_mode,
                    intervention_fn=intervention_fn,
                    batch_meta=batch_meta,
                    pure_substrate_only=pure_this_condition,
                    target_substrate=target_substrate,
                )

            pred_probe = out["prediction_probe"].detach().cpu().long()
            fire_t_probe = out["first_fire_t_probe"].detach().cpu().long()
            restore_ok = int(out["restore_ok"].item()) if "restore_ok" in out else 1
            reset_applied = int(out["reset_applied"].item()) if "reset_applied" in out else 0
            legacy_phase_reset_applied = (
                int(out["legacy_phase_reset_applied"].item()) if "legacy_phase_reset_applied" in out else 0
            )

            for i in range(bsz):
                sample_label = int(sample_lbl_np[i])
                probe_label = int(probe_lbl_np[i])
                donor_batch_i = int(donor_idx_b[i])
                donor_trial_id = int(trial_ids[donor_batch_i])
                donor_sample_label = int(sample_lbl_np[donor_batch_i])
                pred_i = int(pred_probe[i].item())
                fire_i = int(fire_t_probe[i].item())
                donor_distinct = int(donor_sample_label != sample_label)

                all_records.append(
                    {
                        "trial_id": int(trial_ids[i]),
                        "condition": condition_name,
                        "stsp_mode": stsp_mode,
                        "sample_label": sample_label,
                        "probe_label": probe_label,
                        "donor_batch_index": donor_batch_i,
                        "donor_trial_id": donor_trial_id,
                        "donor_sample_label": donor_sample_label,
                        "donor_is_distinct": donor_distinct,
                        "is_self_swap": int(donor_batch_i == i),
                        "donor_probe_conflict": int(donor_sample_label == probe_label),
                        "prediction_probe": pred_i,
                        "first_fire_t_probe": fire_i,
                        "is_correct_probe": int(pred_i == probe_label),
                        "is_silent_probe": int(pred_i == -1),
                        "pred_is_original_sample": int(pred_i == sample_label),
                        "pred_is_donor_sample": int(pred_i == donor_sample_label),
                        "pred_is_donor_shifted_memory": int((pred_i == donor_sample_label) and (donor_distinct == 1)),
                        "pure_substrate_only": int(bool(target_substrate is not None) and bool(pure_this_condition)),
                        "target_substrate": target_substrate if target_substrate is not None else "none",
                        "restore_ok": int((target_substrate is None) or (restore_ok == 1)),
                        "reset_applied": int(reset_applied),
                        "legacy_phase_reset_applied": int(legacy_phase_reset_applied),
                        "used_relaxed_rule": int(plan_info["used_relaxed_rule"]),
                        "b_pure_ux_mode": int((condition_name == CONDITION_D_TRIAL_SHUFFLE_UX) and bool(pure_this_condition)),
                        "non_ux_state_reset_applied": int(reset_applied),
                        "ux_restore_ok": int((condition_name != CONDITION_D_TRIAL_SHUFFLE_UX) or (restore_ok == 1)),
                    }
                )

        if plan_info["used_relaxed_rule"] == 1:
            logger.warning(
                "[Warn] Batch start=%d required relaxed no-self rule; self-swaps=%d/%d",
                start,
                plan_info["n_self_swap"],
                bsz,
            )

    return pd.DataFrame(all_records).sort_values(["trial_id", "condition"]).reset_index(drop=True)


def validate_pairing(df_trials: pd.DataFrame, pure_substrate_only: bool) -> None:
    count_per_trial = df_trials.groupby("trial_id").size()
    expected = len(CONDITION_ORDER)
    if not (count_per_trial == expected).all():
        bad_ids = count_per_trial[count_per_trial != expected].index.tolist()
        raise ValueError(f"Each trial_id must appear exactly {expected} times. Bad ids: {bad_ids[:10]}")
    for col in ["sample_label", "probe_label"]:
        uniq = df_trials.groupby("trial_id")[col].nunique()
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not paired-identical across conditions for ids: {bad_ids[:10]}")
    for col in ["donor_batch_index", "donor_trial_id", "donor_sample_label", "is_self_swap", "used_relaxed_rule"]:
        uniq = df_trials.groupby("trial_id")[col].nunique()
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not donor-paired-identical across conditions for ids: {bad_ids[:10]}")
    shuffle_rows = df_trials[df_trials["condition"].isin(SHUFFLE_CONDITIONS)]
    if len(shuffle_rows) > 0 and bool((shuffle_rows["donor_probe_conflict"] == 1).any()):
        raise ValueError("Found donor_sample_label == probe_label in a trial-shuffle condition.")
    if len(shuffle_rows) > 0 and pure_substrate_only:
        if bool((shuffle_rows["reset_applied"] != 1).any()):
            raise ValueError("Pure substrate mode expected reset_applied=1 for all shuffle rows.")
        if bool((shuffle_rows["restore_ok"] != 1).any()):
            raise ValueError("Pure substrate mode has restore_ok=0 rows.")


def compute_condition_metrics(df_trials: pd.DataFrame) -> pd.DataFrame:
    return shared_compute_condition_metrics(
        df_trials,
        condition_order=CONDITION_ORDER,
        shuffle_condition=CONDITION_D_TRIAL_SHUFFLE_UX,
        static_condition=CONDITION_E_STATIC_FROZEN,
    )


def compute_bias_table(df_trials: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    return shared_compute_bias_table(df_trials, num_classes=num_classes, condition_order=CONDITION_ORDER)


def compute_collapse_summary(
    df_trials: pd.DataFrame,
    metrics_condition: pd.DataFrame,
    metrics_bias: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[pd.DataFrame] = []
    boot_rows: List[pd.DataFrame] = []
    for idx, substrate in enumerate(SUBSTRATE_ORDER):
        shuffle_condition = SUBSTRATE_TO_CONDITION[substrate]
        summary_i, boot_i = shared_compute_collapse_summary(
            df_trials,
            metrics_condition,
            metrics_bias,
            n_boot=n_boot,
            seed=seed + idx * 1000,
            dynamic_condition=CONDITION_A_DYNAMIC_BASE,
            shuffle_condition=shuffle_condition,
            static_condition=CONDITION_E_STATIC_FROZEN,
        )
        summary_i.insert(0, "substrate", substrate)
        summary_i.insert(1, "shuffle_condition", shuffle_condition)
        boot_i.insert(0, "substrate", substrate)
        boot_i.insert(1, "shuffle_condition", shuffle_condition)
        summary_rows.append(summary_i)
        boot_rows.append(boot_i)
    return pd.concat(summary_rows, ignore_index=True), pd.concat(boot_rows, ignore_index=True)


def _nice_axis_upper(values: np.ndarray) -> float:
    max_value = float(np.nanmax(values)) if values.size > 0 else 0.0
    if max_value <= 0.0:
        return 5.0
    target = max(5.0, max_value * 1.25)
    exponent = np.floor(np.log10(target))
    base = 10.0 ** exponent
    for mult in (1.0, 2.0, 5.0, 10.0):
        candidate = mult * base
        if candidate >= target:
            return float(candidate)
    return float(10.0 * base)


def _nice_tick_step(upper: float) -> float:
    raw = max(1.0, upper / 5.0)
    exponent = np.floor(np.log10(raw))
    base = 10.0 ** exponent
    for mult in (1.0, 2.0, 5.0, 10.0):
        candidate = mult * base
        if candidate >= raw:
            return float(candidate)
    return float(10.0 * base)


def build_summary(metrics_condition: pd.DataFrame, collapse_summary: pd.DataFrame, experiment_name: str) -> Dict[str, object]:
    row_cond = metrics_condition.set_index("condition")
    row_sum = collapse_summary.set_index("substrate")
    row_stsp = row_sum.loc[SUBSTRATE_STSP]
    orig_dynamic = float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "abs_rate_pred_original_sample"])
    orig_shuffle = float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "abs_rate_pred_original_sample"])
    orig_static = float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "abs_rate_pred_original_sample"])
    change_dynamic = float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "abs_rate_pred_change_under_bmap"])
    change_shuffle = float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "abs_rate_pred_change_under_bmap"])
    change_static = float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "abs_rate_pred_change_under_bmap"])

    substrate_summary = []
    for substrate in SUBSTRATE_ORDER:
        condition_name = SUBSTRATE_TO_CONDITION[substrate]
        substrate_summary.append(
            {
                "substrate": substrate,
                "condition": condition_name,
                "acc_probe": float(row_cond.loc[condition_name, "acc_probe"]),
                "original_sample_readout": float(row_cond.loc[condition_name, "abs_rate_pred_original_sample"]),
                "changed_memory_readout": float(row_cond.loc[condition_name, "abs_rate_pred_change_under_bmap"]),
                "collapse_toward_static_improvement_pp": float(row_sum.loc[substrate, "collapse_toward_static_improvement_pp"]),
            }
        )

    if (orig_dynamic > orig_shuffle) and (change_shuffle > change_dynamic):
        summary_text = (
            "Delay-end substrate shuffling shifts probe readout away from the original sample; the STSP u/x shuffle "
            "still produces the clearest donor-shift signature."
        )
    else:
        summary_text = (
            "Delay-end substrate shuffling changes memory-readout target statistics across spike, membrane, and STSP states."
        )

    return {
        "experiment_name": experiment_name,
        "primary_figure": os.path.join("figures", "memory_readout_target.png"),
        "acc_probe_dynamic": float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "acc_probe"]),
        "acc_probe_shuffle": float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "acc_probe"]),
        "acc_probe_static": float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "acc_probe"]),
        "original_sample_readout_dynamic": orig_dynamic,
        "original_sample_readout_shuffle": orig_shuffle,
        "original_sample_readout_static": orig_static,
        "changed_memory_readout_dynamic": change_dynamic,
        "changed_memory_readout_shuffle": change_shuffle,
        "changed_memory_readout_static": change_static,
        "ami_drop_A_minus_B_pp": float(row_stsp["ami_drop_A_minus_B_pp"]),
        "sample_pred_rate_drop_A_minus_B_pp": float(row_stsp["sample_pred_rate_drop_A_minus_B_pp"]),
        "paired_bootstrap_p_one_sided_nonpositive": float(row_stsp["paired_bootstrap_p_one_sided_nonpositive"]),
        "paired_bootstrap_p_one_sided_no_donor_gain": float(row_stsp["paired_bootstrap_p_one_sided_no_donor_gain"]),
        "collapse_gain_bootstrap_p_one_sided_nonpositive": float(row_stsp["collapse_gain_bootstrap_p_one_sided_nonpositive"]),
        "substrate_shuffle_summary": substrate_summary,
        "summary_text": summary_text,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shuffled-substrate memory-collapse experiment (DMS)")
    parser.add_argument("--model-path", type=str, default=os.path.join("results", "sdnn_deep_final", "net_final.pth"))
    parser.add_argument("--save-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=500.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--num-boot", type=int, default=5000)
    parser.add_argument(
        "--pure-substrate-only",
        "--b-pure-ux-only",
        dest="pure_substrate_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_classes < 3:
        raise ValueError("--num-classes must be >= 3")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(dt=1.0 * ms, sample_ms=args.sample_ms, delay_ms=args.delay_ms, probe_ms=args.probe_ms)
    for name, steps in [("sample", spec.sample_steps), ("delay", spec.delay_steps), ("probe", spec.probe_steps)]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    layout = prepare_result_layout(args.save_dir)
    experiment_name = Path(args.save_dir).name or "ux_shuffle_memory_collapse"
    logger = setup_logger(layout.log_file(), experiment_name)
    data_dir = layout.data_dir
    metrics_dir = layout.metrics_dir
    meta_dir = layout.meta_dir
    used_deprecated_alias = any(
        arg.startswith("--b-pure-ux-only") or arg.startswith("--no-b-pure-ux-only") for arg in sys.argv[1:]
    )

    logger.info("[Init] Run started at %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("[Init] Save dir: %s", layout.root)
    logger.info("[Init] Device: %s", device)
    logger.info("[Init] Model path: %s", args.model_path)
    logger.info(
        "[Init] DMS timing | sample=%d steps (%.1fms), delay=%d steps (%.1fms), probe=%d steps (%.1fms)",
        spec.sample_steps,
        spec.sample_ms,
        spec.delay_steps,
        spec.delay_ms,
        spec.probe_steps,
        spec.probe_ms,
    )
    logger.info(
        "[Init] trials=%d | batch_size=%d | num_classes=%d | num_boot=%d | pure_substrate_only=%s",
        args.trials,
        args.batch_size,
        args.num_classes,
        args.num_boot,
        bool(args.pure_substrate_only),
    )
    if used_deprecated_alias:
        logger.warning("[Init] Deprecated alias --b-pure-ux-only used; prefer --pure-substrate-only.")

    run_config = {
        "experiment_name": experiment_name,
        "model_path": args.model_path,
        "seed": int(args.seed),
        "trials": int(args.trials),
        "batch_size": int(args.batch_size),
        "num_classes": int(args.num_classes),
        "num_boot": int(args.num_boot),
        "sample_ms": float(args.sample_ms),
        "delay_ms": float(args.delay_ms),
        "probe_ms": float(args.probe_ms),
        "pure_substrate_only": bool(args.pure_substrate_only),
        "b_pure_ux_only": bool(args.pure_substrate_only),
    }
    run_config_path = save_json(run_config, layout.root_file("run_config.json"), logger)
    save_json(run_config, meta_dir / "run_config.snapshot.json", logger)

    net, encoder = shared_load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.probe_ms),
    )

    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = shared_build_class_index(dataset, num_classes=args.num_classes)
    rng = random.Random(args.seed)

    df_specs = generate_trial_specs(class_index, num_trials=args.trials, num_classes=args.num_classes, rng=rng)
    validate_trial_specs(df_specs, num_classes=args.num_classes)
    trial_specs_csv = Path(save_tidy_csv(df_specs, data_dir / "trial_specs.csv", sort_by=["trial_id"]))
    logger.info("[Save] trial_specs_csv=%s", trial_specs_csv)

    df_trials = run_experiment(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        pure_substrate_only=bool(args.pure_substrate_only),
        logger=logger,
    )
    validate_pairing(df_trials, pure_substrate_only=bool(args.pure_substrate_only))
    trial_pred_csv = Path(save_tidy_csv(df_trials, data_dir / "trial_predictions.csv", sort_by=["trial_id", "condition"]))
    logger.info("[Save] trial_predictions_csv=%s", trial_pred_csv)

    metrics_condition = compute_condition_metrics(df_trials)
    metrics_condition_csv = Path(save_tidy_csv(metrics_condition, metrics_dir / "metrics_condition_summary.csv", sort_by=["condition"]))
    logger.info("[Save] metrics_condition_summary_csv=%s", metrics_condition_csv)

    metrics_bias = compute_bias_table(df_trials, num_classes=args.num_classes)
    metrics_bias_csv = Path(save_tidy_csv(metrics_bias, metrics_dir / "metrics_error_bias.csv", sort_by=["condition"]))
    logger.info("[Save] metrics_error_bias_csv=%s", metrics_bias_csv)

    collapse_summary, metrics_boot = compute_collapse_summary(
        df_trials=df_trials,
        metrics_condition=metrics_condition,
        metrics_bias=metrics_bias,
        n_boot=args.num_boot,
        seed=args.seed + 100,
    )
    collapse_summary_csv = Path(save_tidy_csv(collapse_summary, metrics_dir / "metrics_collapse_summary.csv"))
    substrate_summary_csv = Path(
        save_tidy_csv(collapse_summary, metrics_dir / "metrics_substrate_shuffle_summary.csv", sort_by=["substrate"])
    )
    metrics_boot_csv = Path(save_tidy_csv(metrics_boot, metrics_dir / "metrics_bootstrap_tests.csv"))
    write_plot_bundle_manifest(meta_dir)
    logger.info("[Save] metrics_collapse_summary_csv=%s", collapse_summary_csv)
    logger.info("[Save] metrics_substrate_shuffle_summary_csv=%s", substrate_summary_csv)
    logger.info("[Save] metrics_bootstrap_tests_csv=%s", metrics_boot_csv)

    figure_paths = {"png": "", "pdf": "", "svg": ""}
    if not bool(args.skip_figures):
        fig = build_memory_readout_target_figure(metrics_condition)
        figure_paths = save_figure_all_formats(fig, layout.figure_base("memory_readout_target"))
        plt.close(fig)
        logger.info("[Save] Primary figure saved to %s", figure_paths["png"])
        logger.info("[Save] Primary figure saved to %s", figure_paths["pdf"])
        logger.info("[Save] Primary figure saved to %s", figure_paths["svg"])

    summary = build_summary(metrics_condition, collapse_summary, experiment_name)
    summary_path = save_json(summary, layout.root_file("summary.json"), logger)
    save_json(summary, metrics_dir / "summary.json", logger)
    row_cond = metrics_condition.set_index("condition")
    row_sum = collapse_summary.set_index("substrate")
    save_json(
        {
            "experiment_name": experiment_name,
            "dynamic_probe_accuracy": float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "acc_probe"]),
            "static_probe_accuracy": float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "acc_probe"]),
            "ux_shuffle_probe_accuracy": float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "acc_probe"]),
            "condition_metrics_csv": str(metrics_condition_csv.resolve()),
            "collapse_summary_csv": str(collapse_summary_csv.resolve()),
            "bootstrap_tests_csv": str(metrics_boot_csv.resolve()),
        },
        metrics_dir / "main_metrics.json",
        logger,
    )
    logger.info(
        "[Summary] Probe accuracy dynamic/spike/membrane/u-x/static: %.2f / %.2f / %.2f / %.2f / %.2f",
        float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "acc_probe"]),
        float(row_cond.loc[CONDITION_B_TRIAL_SHUFFLE_SPIKE, "acc_probe"]),
        float(row_cond.loc[CONDITION_C_TRIAL_SHUFFLE_MEMBRANE, "acc_probe"]),
        float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "acc_probe"]),
        float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "acc_probe"]),
    )
    logger.info(
        "[Summary] Original-sample readout dynamic/spike/membrane/u-x/static: %.2f / %.2f / %.2f / %.2f / %.2f",
        float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "abs_rate_pred_original_sample"]),
        float(row_cond.loc[CONDITION_B_TRIAL_SHUFFLE_SPIKE, "abs_rate_pred_original_sample"]),
        float(row_cond.loc[CONDITION_C_TRIAL_SHUFFLE_MEMBRANE, "abs_rate_pred_original_sample"]),
        float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "abs_rate_pred_original_sample"]),
        float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "abs_rate_pred_original_sample"]),
    )
    logger.info(
        "[Summary] Changed-memory readout dynamic/spike/membrane/u-x/static: %.2f / %.2f / %.2f / %.2f / %.2f",
        float(row_cond.loc[CONDITION_A_DYNAMIC_BASE, "abs_rate_pred_change_under_bmap"]),
        float(row_cond.loc[CONDITION_B_TRIAL_SHUFFLE_SPIKE, "abs_rate_pred_change_under_bmap"]),
        float(row_cond.loc[CONDITION_C_TRIAL_SHUFFLE_MEMBRANE, "abs_rate_pred_change_under_bmap"]),
        float(row_cond.loc[CONDITION_D_TRIAL_SHUFFLE_UX, "abs_rate_pred_change_under_bmap"]),
        float(row_cond.loc[CONDITION_E_STATIC_FROZEN, "abs_rate_pred_change_under_bmap"]),
    )
    for substrate in SUBSTRATE_ORDER:
        logger.info(
            "[Summary] substrate=%s | ami_drop_A_minus_shuffle_pp=%.2f | sample_pred_rate_drop_pp=%.2f | p_sample=%.4g | p_donor=%.4g | p_collapse=%.4g",
            substrate,
            float(row_sum.loc[substrate, "ami_drop_A_minus_B_pp"]),
            float(row_sum.loc[substrate, "sample_pred_rate_drop_A_minus_B_pp"]),
            float(row_sum.loc[substrate, "paired_bootstrap_p_one_sided_nonpositive"]),
            float(row_sum.loc[substrate, "paired_bootstrap_p_one_sided_no_donor_gain"]),
            float(row_sum.loc[substrate, "collapse_gain_bootstrap_p_one_sided_nonpositive"]),
        )
    logger.info("[Done] summary_json=%s", summary_path)
    logger.info("[Done] run_config_json=%s", run_config_path)


if __name__ == "__main__":
    main()
