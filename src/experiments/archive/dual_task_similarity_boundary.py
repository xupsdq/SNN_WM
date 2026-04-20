"""Supplementary/exploratory script.

This file is no longer part of the main-text figure pipeline.
Use the plot_fig*.py scripts plus figure_utils_common.py for the main figure path.
"""
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# Support direct execution via `python src/experiments/...py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.experiments.common.dataset import build_class_index as shared_build_class_index
from src.experiments.common.dataset import encode_images as shared_encode_images
from src.experiments.common.decoding import decode_accuracy_with_splits as shared_decode_accuracy_with_splits
from src.experiments.common.ping_common import snapshot_ux_layer_means as shared_snapshot_ux_layer_means
from src.experiments.common.model_io import compensate_stsp_gain as shared_compensate_stsp_gain
from src.experiments.common.model_io import load_model_and_encoder as shared_load_model_and_encoder
from src.experiments.common.runtime import seed_everything as shared_seed_everything
from src.platform.legacy_adapters.units import ms


LABEL_CONDITION_ORDER = [
    "clean_reference",
    "high_same_label",
    "high_diff_label",
    "medium_confusion",
    "low_confusion",
]
FAMILY_ORDER = ["same_label", "different_label"]
SIMILARITY_BIN_ORDER = ["low", "medium", "high"]
LAYER_KEYS = ["layer1", "layer2", "layer3"]
STSP_MODES = ["dynamic", "static_frozen"]
NUM_CLASSES_GLOBAL = 10


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool

    @property
    def sample_steps(self) -> int:
        return int((self.sample_ms * ms) / self.dt)

    @property
    def delay1_steps(self) -> int:
        return int((self.delay1_ms * ms) / self.dt)

    @property
    def distractor_steps(self) -> int:
        return int((self.distractor_ms * ms) / self.dt)

    @property
    def delay2_steps(self) -> int:
        return int((self.delay2_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)

    @property
    def clean_delay_steps(self) -> int:
        return self.delay1_steps + self.distractor_steps + self.delay2_steps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    net = SDNN_Network(device=str(device)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    max_duration_ms = max(spec.sample_ms, spec.distractor_ms, spec.probe_ms)
    encoder = DoGSpikeEncoder(dt=spec.dt, max_duration=max_duration_ms * ms, device=str(device))
    return net, encoder


def build_class_index(dataset, num_classes: int) -> Dict[int, List[int]]:
    class_index: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        lbl = int(label)
        if 0 <= lbl < num_classes:
            class_index[lbl].append(idx)

    for cls in range(num_classes):
        if len(class_index[cls]) == 0:
            raise ValueError(f"Class {cls} has no samples in dataset")
    return class_index


def encode_images(encoder: DoGSpikeEncoder, images: torch.Tensor, steps: int) -> torch.Tensor:
    with torch.no_grad():
        spikes = encoder.forward(images)
    return spikes[:, :steps, ...].contiguous()


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.maximum(den, eps)
    return (num / den).astype(np.float64, copy=False)


def extract_prediction_and_fire_time_from_layer3(net: SDNN_Network, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_times = net.layer3.firing_times
    if flat_times.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: firing_times={flat_times.shape[0]}, expected={batch_size}")

    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    pred = (min_indices // net.layer3.neurons_per_class).long()
    pred[~has_fired] = -1

    fire_t = min_times.clone()
    fire_t[~has_fired] = -1
    fire_t = fire_t.to(torch.long)
    return pred.detach().cpu(), fire_t.detach().cpu()


def run_interface_check(net: SDNN_Network, device: torch.device) -> None:
    with torch.no_grad():
        bsz, c, h, w = 2, 2, 28, 28
        sample = (torch.rand((bsz, 20, c, h, w), device=device) > 0.95).float()
        distractor = (torch.rand((bsz, 20, c, h, w), device=device) > 0.95).float()
        probe = (torch.rand((bsz, 20, c, h, w), device=device) > 0.95).float()
        out = net.forward_dual_task_session(
            sample_spikes=sample,
            distractor_spikes=distractor,
            probe_spikes=probe,
            delay1_steps=10,
            delay2_steps=10,
            stsp_mode="static_frozen",
            phase_reset=True,
        )

    required_keys = {
        "prediction_distractor",
        "prediction_probe",
        "first_fire_t_distractor",
        "first_fire_t_probe",
    }
    if set(out.keys()) != required_keys:
        raise ValueError(f"forward_dual_task_session keys mismatch: {set(out.keys())}")

    for key in required_keys:
        tensor = out[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{key} is not a tensor")
        if tensor.shape != (bsz,):
            raise ValueError(f"{key} shape mismatch: {tensor.shape}")
        if tensor.dtype != torch.long:
            raise ValueError(f"{key} dtype mismatch: {tensor.dtype}")


def _prepare_network_state(net: SDNN_Network, bsz: int, c: int, h: int, w: int) -> None:
    net.layer1.reset_state((bsz, c, h, w))

    h1 = (h + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    w1 = (w + 2 * net.layer1.padding - net.layer1.kernel_size) // net.layer1.stride + 1
    h1_p, w1_p = h1 // 2, w1 // 2
    net.layer2.reset_state((bsz, net.layer1.out_channels, h1_p, w1_p))

    h2 = (h1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    w2 = (w1_p + 2 * net.layer2.padding - net.layer2.kernel_size) // net.layer2.stride + 1
    h2_p, w2_p = h2 // 2, w2 // 2
    net.layer3.reset_state((bsz, net.layer2.out_channels, h2_p, w2_p))


def _snapshot_ux_features(net: SDNN_Network, batch_size: int) -> Dict[str, Optional[np.ndarray]]:
    out: Dict[str, Optional[np.ndarray]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key, None)
        if layer is None or getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            out[layer_key] = None
            continue
        gain = (layer.u_pre * layer.x_pre).detach().view(batch_size, -1).cpu().numpy().astype(np.float32, copy=False)
        out[layer_key] = gain
    return out


def _snapshot_ux_layer_means(net: SDNN_Network, batch_size: int) -> Dict[str, np.ndarray]:
    return shared_snapshot_ux_layer_means(net, batch_size=batch_size)


def run_similarity_session(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    spec: ExperimentSpec,
    session_kind: str,
    stsp_mode: str,
) -> Dict[str, object]:
    bsz, t_sample, c, h, w = sample_spikes.shape
    t_distractor = distractor_spikes.shape[1]
    t_probe = probe_spikes.shape[1]

    _prepare_network_state(net, bsz, c, h, w)
    zero_input = torch.zeros((bsz, c, h, w), device=sample_spikes.device)
    current_time = 0
    ux_timecourse: Optional[List[Dict[str, object]]] = None

    def step_network(input_t: torch.Tensor, force_l3_time: Optional[int] = None) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode=stsp_mode)
        s1_p = net.pool1(s1.float())

        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode=stsp_mode)
        s2_p = net.pool2(s2.float())

        t_for_l3 = current_time if force_l3_time is None else force_l3_time
        net.layer3.forward_step(s2_p, t_for_l3, training=False, monitor=False, stsp_mode=stsp_mode)
        current_time += 1

    def reset_decision_window() -> None:
        net.layer3.reset_decision_state()
        if spec.phase_reset:
            with torch.no_grad():
                net.layer3.v_mem.fill_(net.layer3.V_L)
                net.layer3.lateral_inh.reset_state(net.layer3.output_shape)

    def record_ux_timepoint(phase: str, phase_step: int) -> None:
        if ux_timecourse is None:
            return
        layer_means = _snapshot_ux_layer_means(net, bsz)
        ux_timecourse.append(
            {
                "phase": phase,
                "phase_step": int(phase_step),
                "global_time_step": int(current_time - 1),
                "time_ms": float(current_time * spec.dt / ms),
                "layer_means": layer_means,
            }
        )

    for t in range(t_sample):
        step_network(sample_spikes[:, t, ...])
        record_ux_timepoint("sample", t)

    if session_kind == "clean":
        for _ in range(spec.clean_delay_steps):
            step_network(zero_input)
        ux_features = _snapshot_ux_features(net, bsz)
        pred_distractor = torch.full((bsz,), -1, dtype=torch.long)
        fire_t_distractor = torch.full((bsz,), -1, dtype=torch.long)
    elif session_kind == "distracted":
        for _ in range(spec.delay1_steps):
            step_network(zero_input)
            record_ux_timepoint("delay1", _)

        reset_decision_window()
        if stsp_mode == "dynamic":
            ux_timecourse = []
        for t in range(t_distractor):
            force_t = t if spec.phase_reset else None
            step_network(distractor_spikes[:, t, ...], force_l3_time=force_t)
            record_ux_timepoint("distractor", t)
        pred_distractor, fire_t_distractor = extract_prediction_and_fire_time_from_layer3(net, bsz)

        for t in range(spec.delay2_steps):
            step_network(zero_input)
            record_ux_timepoint("delay2", t)
        ux_features = _snapshot_ux_features(net, bsz)
    else:
        raise ValueError(f"Unknown session_kind: {session_kind}")

    reset_decision_window()
    for t in range(t_probe):
        force_t = t if spec.phase_reset else None
        step_network(probe_spikes[:, t, ...], force_l3_time=force_t)
        record_ux_timepoint("probe", t)
    pred_probe, fire_t_probe = extract_prediction_and_fire_time_from_layer3(net, bsz)

    return {
        "prediction_distractor": pred_distractor,
        "prediction_probe": pred_probe,
        "first_fire_t_distractor": fire_t_distractor,
        "first_fire_t_probe": fire_t_probe,
        "ux_features": ux_features,
        "ux_timecourse": ux_timecourse,
    }


def compute_confusion_artifacts(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset_root: str,
    batch_size: int,
    num_classes: int,
    sample_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, test_loader = build_mnist_skeleton_loader(root=dataset_root, batch_size=batch_size)
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    totals = np.zeros(num_classes, dtype=np.int64)
    silent = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="ConfusionEval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            spikes = encode_images(encoder, images, sample_steps)
            _ = net(spikes, layer_idx=3, labels=None, monitor=False)
            pred, _fire_t = extract_prediction_and_fire_time_from_layer3(net, batch_size=len(labels))

            y = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            p = pred.numpy().astype(np.int64, copy=False)
            for yi, pi in zip(y, p):
                totals[int(yi)] += 1
                if pi == -1:
                    silent[int(yi)] += 1
                elif 0 <= pi < num_classes:
                    counts[int(yi), int(pi)] += 1

    norm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for cls in range(num_classes):
        if totals[cls] > 0:
            norm[cls] = counts[cls].astype(np.float64) / float(totals[cls])
    return counts, norm, silent


def build_confusion_matrix_table(
    counts: np.ndarray,
    norm: np.ndarray,
    silent: np.ndarray,
    totals: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    num_classes = counts.shape[0]
    for true_label in range(num_classes):
        row: Dict[str, float] = {
            "true_label": int(true_label),
            "total_count": int(totals[true_label]),
            "pred_silent_count": int(silent[true_label]),
            "pred_silent_norm": float(silent[true_label] / max(1, totals[true_label])),
        }
        for pred_label in range(num_classes):
            row[f"pred_{pred_label}_count"] = int(counts[true_label, pred_label])
            row[f"pred_{pred_label}_norm"] = float(norm[true_label, pred_label])
        rows.append(row)
    return pd.DataFrame(rows)


def build_confusion_pair_catalog(
    counts: np.ndarray,
    norm: np.ndarray,
    totals: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    num_classes = counts.shape[0]
    n_off = num_classes - 1
    mid_rank = (n_off // 2) + 1

    for sample_label in range(num_classes):
        ranked = []
        for distractor_label in range(num_classes):
            if distractor_label == sample_label:
                continue
            ranked.append(
                {
                    "sample_label": int(sample_label),
                    "distractor_label": int(distractor_label),
                    "confusion_count": int(counts[sample_label, distractor_label]),
                    "confusion_prob": float(norm[sample_label, distractor_label]),
                    "row_total_count": int(totals[sample_label]),
                }
            )

        ranked_desc = sorted(
            ranked,
            key=lambda x: (-x["confusion_prob"], -x["confusion_count"], x["distractor_label"]),
        )
        ranked_asc = sorted(
            ranked,
            key=lambda x: (x["confusion_prob"], x["confusion_count"], x["distractor_label"]),
        )
        rank_desc_map = {row["distractor_label"]: i + 1 for i, row in enumerate(ranked_desc)}
        rank_asc_map = {row["distractor_label"]: i + 1 for i, row in enumerate(ranked_asc)}

        for item in ranked:
            rank_desc = rank_desc_map[item["distractor_label"]]
            rank_asc = rank_asc_map[item["distractor_label"]]
            tier = ""
            if rank_desc == 1:
                tier = "high_diff_label"
            elif rank_desc == mid_rank:
                tier = "medium_confusion"
            elif rank_asc == 1:
                tier = "low_confusion"
            item["rank_desc"] = int(rank_desc)
            item["rank_asc"] = int(rank_asc)
            item["tier"] = tier
            rows.append(item)

    return pd.DataFrame(rows).sort_values(["sample_label", "rank_desc", "distractor_label"]).reset_index(drop=True)


def load_confusion_artifacts(
    confusion_csv: str,
    catalog_csv: str,
    num_classes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(confusion_csv):
        raise FileNotFoundError(f"Missing confusion matrix CSV: {confusion_csv}")
    if not os.path.exists(catalog_csv):
        raise FileNotFoundError(f"Missing confusion pair catalog CSV: {catalog_csv}")

    df_confusion = pd.read_csv(confusion_csv)
    df_catalog = pd.read_csv(catalog_csv)
    expected_true = set(range(num_classes))
    if set(df_confusion["true_label"].tolist()) != expected_true:
        raise ValueError("Confusion matrix CSV has unexpected true_label set")
    return df_confusion, df_catalog


def build_confusion_lookup(df_catalog: pd.DataFrame, num_classes: int) -> Dict[int, Dict[str, int]]:
    lookup: Dict[int, Dict[str, int]] = {}
    for sample_label in range(num_classes):
        sub = df_catalog[df_catalog["sample_label"] == sample_label].copy()
        if len(sub) != num_classes - 1:
            raise ValueError(f"Unexpected catalog size for sample_label={sample_label}")

        cond_map: Dict[str, int] = {}
        for tier in ["high_diff_label", "medium_confusion", "low_confusion"]:
            tier_sub = sub[sub["tier"] == tier]
            if len(tier_sub) != 1:
                raise ValueError(f"Expected exactly one {tier} entry for sample_label={sample_label}")
            cond_map[tier] = int(tier_sub.iloc[0]["distractor_label"])

        if len(set(cond_map.values())) != 3:
            raise ValueError(f"Confusion lookup labels are not distinct for sample_label={sample_label}")
        lookup[int(sample_label)] = cond_map
    return lookup


def _random_choice(rng: random.Random, values: Sequence[int]) -> int:
    if len(values) == 0:
        raise ValueError("random choice received empty sequence")
    return int(values[rng.randrange(len(values))])


def generate_trial_specs(
    class_index: Dict[int, List[int]],
    confusion_lookup: Dict[int, Dict[str, int]],
    num_trials: int,
    num_classes: int,
    rng: random.Random,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    trial_id = 0
    classes = list(range(num_classes))

    for base_trial_id in range(num_trials):
        sample_label = _random_choice(rng, classes)
        sample_index = _random_choice(rng, class_index[sample_label])

        same_label_pool = [idx for idx in class_index[sample_label] if idx != sample_index]
        if len(same_label_pool) == 0:
            raise ValueError(f"No same-label distractor candidate available for class {sample_label}")

        high_diff_label = int(confusion_lookup[sample_label]["high_diff_label"])
        medium_label = int(confusion_lookup[sample_label]["medium_confusion"])
        low_label = int(confusion_lookup[sample_label]["low_confusion"])
        diff_labels = [high_diff_label, medium_label, low_label]

        probe_candidates = [c for c in classes if c != sample_label and c not in diff_labels]
        if len(probe_candidates) == 0:
            raise ValueError(f"No valid probe candidate for sample_label={sample_label}")
        probe_label = _random_choice(rng, probe_candidates)
        probe_index = _random_choice(rng, class_index[probe_label])

        condition_items = [
            ("clean_reference", -1, -1),
            ("high_same_label", sample_label, _random_choice(rng, same_label_pool)),
            ("high_diff_label", high_diff_label, _random_choice(rng, class_index[high_diff_label])),
            ("medium_confusion", medium_label, _random_choice(rng, class_index[medium_label])),
            ("low_confusion", low_label, _random_choice(rng, class_index[low_label])),
        ]

        for label_condition, distractor_label, distractor_index in condition_items:
            session_kind = "clean" if label_condition == "clean_reference" else "distracted"
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "base_trial_id": int(base_trial_id),
                    "label_similarity_condition": str(label_condition),
                    "session_kind": session_kind,
                    "sample_index": int(sample_index),
                    "sample_label": int(sample_label),
                    "distractor_index": int(distractor_index),
                    "distractor_label": int(distractor_label),
                    "probe_index": int(probe_index),
                    "probe_label": int(probe_label),
                }
            )
            trial_id += 1

    return pd.DataFrame(rows)


def validate_trial_specs(df_specs: pd.DataFrame, num_classes: int) -> None:
    if df_specs["trial_id"].nunique() != len(df_specs):
        raise ValueError("trial_id must be unique in trial specs")

    expected_conditions = set(LABEL_CONDITION_ORDER)
    grouped = df_specs.groupby("base_trial_id")
    for base_trial_id, sub in grouped:
        conds = set(sub["label_similarity_condition"].tolist())
        if conds != expected_conditions:
            raise ValueError(f"base_trial_id={base_trial_id} has wrong condition set: {conds}")
        if len(sub) != len(LABEL_CONDITION_ORDER):
            raise ValueError(f"base_trial_id={base_trial_id} has wrong number of rows")
        if sub["sample_index"].nunique() != 1 or sub["sample_label"].nunique() != 1:
            raise ValueError(f"base_trial_id={base_trial_id} does not keep sample fixed")
        if sub["probe_index"].nunique() != 1 or sub["probe_label"].nunique() != 1:
            raise ValueError(f"base_trial_id={base_trial_id} does not keep probe fixed")

        clean = sub[sub["label_similarity_condition"] == "clean_reference"]
        if len(clean) != 1:
            raise ValueError(f"base_trial_id={base_trial_id} must have one clean_reference")
        clean_row = clean.iloc[0]
        if clean_row["session_kind"] != "clean":
            raise ValueError("clean_reference row must use session_kind=clean")
        if int(clean_row["distractor_label"]) != -1 or int(clean_row["distractor_index"]) != -1:
            raise ValueError("clean_reference row must use distractor_label/index = -1")

        same_row = sub[sub["label_similarity_condition"] == "high_same_label"].iloc[0]
        if int(same_row["distractor_label"]) != int(same_row["sample_label"]):
            raise ValueError("high_same_label must reuse the sample label")
        if int(same_row["distractor_index"]) == int(same_row["sample_index"]):
            raise ValueError("high_same_label must use a different image index")

        diff_rows = sub[sub["label_similarity_condition"].isin(["high_diff_label", "medium_confusion", "low_confusion"])]
        if not np.all(diff_rows["distractor_label"].to_numpy(dtype=np.int64) != int(clean_row["sample_label"])):
            raise ValueError(f"base_trial_id={base_trial_id} has diff condition equal to sample label")
        if diff_rows["distractor_label"].nunique() != 3:
            raise ValueError(f"base_trial_id={base_trial_id} needs three distinct diff distractor labels")
        if int(clean_row["probe_label"]) in diff_rows["distractor_label"].tolist():
            raise ValueError(f"base_trial_id={base_trial_id} probe collides with diff distractor label")

    for col in ["sample_label", "probe_label"]:
        vals = df_specs[col].to_numpy(dtype=np.int64)
        if np.any((vals < 0) | (vals >= num_classes)):
            raise ValueError(f"{col} contains out-of-range values")


def _input_bin_labels(num_bins: int) -> List[str]:
    if num_bins == 3:
        return ["low", "medium", "high"]
    return [f"bin_{i + 1}" for i in range(num_bins)]


def assign_similarity_bins(df_specs: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    if num_bins <= 1:
        raise ValueError("--input-sim-bins must be > 1")

    labels = _input_bin_labels(num_bins)
    out = df_specs.copy()
    distracted_mask = out["session_kind"] == "distracted"
    for value_col, bin_col in [
        ("pixel_cosine_similarity", "pixel_similarity_bin"),
        ("input_spike_count_cosine_similarity", "spike_similarity_bin"),
    ]:
        ranks = out.loc[distracted_mask, value_col].rank(method="first")
        out.loc[distracted_mask, bin_col] = pd.qcut(ranks, q=num_bins, labels=labels)
        out[bin_col] = out[bin_col].astype("object")
        out.loc[~distracted_mask, bin_col] = np.nan
    return out


def assign_family_similarity_bins(df_specs: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    if num_bins <= 1:
        raise ValueError("--input-sim-bins must be > 1")

    labels = _input_bin_labels(num_bins)
    out = df_specs.copy()
    out["label_relation_family"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    distracted_mask = out["session_kind"] == "distracted"
    same_mask = distracted_mask & (out["sample_label"] == out["distractor_label"])
    diff_mask = distracted_mask & (out["sample_label"] != out["distractor_label"])
    out.loc[same_mask, "label_relation_family"] = "same_label"
    out.loc[diff_mask, "label_relation_family"] = "different_label"
    out["pixel_similarity_within_family_bin"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["pixel_similarity_family_bucket"] = pd.Series([None] * len(out), index=out.index, dtype="object")

    distracted = out[out["session_kind"] == "distracted"].copy()
    for family in FAMILY_ORDER:
        mask = distracted["label_relation_family"] == family
        if int(mask.sum()) == 0:
            continue
        ranks = distracted.loc[mask, "pixel_cosine_similarity"].rank(method="first")
        bins = pd.qcut(ranks, q=num_bins, labels=labels)
        distracted.loc[mask, "pixel_similarity_within_family_bin"] = bins.astype("object")
        distracted.loc[mask, "pixel_similarity_family_bucket"] = [
            f"{family}_{str(v)}" for v in distracted.loc[mask, "pixel_similarity_within_family_bin"].tolist()
        ]

    out = out.drop(columns=["label_relation_family", "pixel_similarity_within_family_bin", "pixel_similarity_family_bucket"], errors="ignore")
    out = out.merge(
        distracted[
            [
                "trial_id",
                "label_relation_family",
                "pixel_similarity_within_family_bin",
                "pixel_similarity_family_bucket",
            ]
        ],
        on="trial_id",
        how="left",
    )
    return out


def run_experiment(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    df_specs: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_records: List[Dict[str, object]] = []
    similarity_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    ux_timecourse_rows: List[Dict[str, object]] = []

    def process_subset(subset_specs: pd.DataFrame, session_kind: str) -> None:
        starts = range(0, len(subset_specs), batch_size)
        for start in tqdm(starts, desc=f"{session_kind.capitalize()}Batches"):
            batch = subset_specs.iloc[start:start + batch_size].copy()
            if len(batch) == 0:
                continue

            sample_imgs = torch.stack([dataset[int(i)][0] for i in batch["sample_index"].tolist()], dim=0).to(device)
            probe_imgs = torch.stack([dataset[int(i)][0] for i in batch["probe_index"].tolist()], dim=0).to(device)
            sample_spikes = encode_images(encoder, sample_imgs, spec.sample_steps)
            probe_spikes = encode_images(encoder, probe_imgs, spec.probe_steps)

            if session_kind == "distracted":
                distractor_imgs = torch.stack([dataset[int(i)][0] for i in batch["distractor_index"].tolist()], dim=0).to(device)
                distractor_spikes = encode_images(encoder, distractor_imgs, spec.distractor_steps)

                pixel_sample = sample_imgs.detach().cpu().numpy().reshape(len(batch), -1).astype(np.float32, copy=False)
                pixel_dist = distractor_imgs.detach().cpu().numpy().reshape(len(batch), -1).astype(np.float32, copy=False)
                spike_sample = sample_spikes.sum(dim=1).detach().cpu().numpy().reshape(len(batch), -1).astype(np.float32, copy=False)
                spike_dist = distractor_spikes.sum(dim=1).detach().cpu().numpy().reshape(len(batch), -1).astype(np.float32, copy=False)
                pixel_sim = cosine_similarity_rows(pixel_sample, pixel_dist)
                spike_sim = cosine_similarity_rows(spike_sample, spike_dist)
            else:
                distractor_spikes = torch.zeros(
                    (len(batch), spec.distractor_steps, sample_spikes.shape[2], sample_spikes.shape[3], sample_spikes.shape[4]),
                    device=device,
                    dtype=sample_spikes.dtype,
                )
                pixel_sim = np.full(len(batch), np.nan, dtype=np.float64)
                spike_sim = np.full(len(batch), np.nan, dtype=np.float64)

            for i, row in enumerate(batch.itertuples(index=False)):
                similarity_rows.append(
                    {
                        "trial_id": int(row.trial_id),
                        "pixel_cosine_similarity": float(pixel_sim[i]),
                        "input_spike_count_cosine_similarity": float(spike_sim[i]),
                    }
                )

            for stsp_mode in STSP_MODES:
                with torch.no_grad():
                    out = run_similarity_session(
                        net=net,
                        sample_spikes=sample_spikes,
                        distractor_spikes=distractor_spikes,
                        probe_spikes=probe_spikes,
                        spec=spec,
                        session_kind=session_kind,
                        stsp_mode=stsp_mode,
                    )

                pred_dist = out["prediction_distractor"].numpy().astype(np.int64, copy=False)
                pred_probe = out["prediction_probe"].numpy().astype(np.int64, copy=False)
                fire_dist = out["first_fire_t_distractor"].numpy().astype(np.int64, copy=False)
                fire_probe = out["first_fire_t_probe"].numpy().astype(np.int64, copy=False)
                ux_features = out["ux_features"]
                ux_timecourse = out["ux_timecourse"]

                for i, row in enumerate(batch.itertuples(index=False)):
                    y_sample = int(row.sample_label)
                    y_dist = int(row.distractor_label)
                    y_probe = int(row.probe_label)
                    pp = int(pred_probe[i])
                    pd_i = int(pred_dist[i])
                    distractor_is_distinct = int(session_kind == "distracted" and y_dist != y_sample)
                    pred_is_distractor = int(distractor_is_distinct and pp == y_dist)
                    pred_is_other = int(
                        (pp >= 0)
                        and (pp < NUM_CLASSES_GLOBAL)
                        and (pp not in {y_sample, y_probe})
                        and ((not distractor_is_distinct) or pp != y_dist)
                    )

                    pred_records.append(
                        {
                            "trial_id": int(row.trial_id),
                            "base_trial_id": int(row.base_trial_id),
                            "label_similarity_condition": str(row.label_similarity_condition),
                            "session_kind": session_kind,
                            "stsp_mode": stsp_mode,
                            "sample_label": y_sample,
                            "distractor_label": y_dist,
                            "probe_label": y_probe,
                            "prediction_distractor": pd_i,
                            "prediction_probe": pp,
                            "first_fire_t_distractor": int(fire_dist[i]),
                            "first_fire_t_probe": int(fire_probe[i]),
                            "is_correct_distractor": float(pd_i == y_dist) if session_kind == "distracted" else np.nan,
                            "is_correct_probe": float(pp == y_probe),
                            "is_silent_distractor": float(pd_i == -1) if session_kind == "distracted" else np.nan,
                            "is_silent_probe": float(pp == -1),
                            "pred_is_sample": float(pp == y_sample),
                            "pred_is_probe": float(pp == y_probe),
                            "pred_is_distractor": float(pred_is_distractor),
                            "pred_is_other": float(pred_is_other),
                            "pred_is_silent": float(pp == -1),
                            "pixel_cosine_similarity": float(pixel_sim[i]),
                            "input_spike_count_cosine_similarity": float(spike_sim[i]),
                        }
                    )

                    for layer_key in LAYER_KEYS:
                        layer_feat = ux_features[layer_key]
                        if layer_feat is None:
                            continue
                        feature_rows.append(
                            {
                                "trial_id": int(row.trial_id),
                                "base_trial_id": int(row.base_trial_id),
                                "label_similarity_condition": str(row.label_similarity_condition),
                                "session_kind": session_kind,
                                "stsp_mode": stsp_mode,
                                "layer": layer_key,
                                "sample_label": y_sample,
                                "distractor_label": y_dist,
                                "feature": layer_feat[i].astype(np.float32, copy=False),
                                "pixel_cosine_similarity": float(pixel_sim[i]),
                                "input_spike_count_cosine_similarity": float(spike_sim[i]),
                            }
                        )

                    if session_kind == "distracted" and stsp_mode == "dynamic" and ux_timecourse is not None:
                        for timepoint in ux_timecourse:
                            for layer_key in LAYER_KEYS:
                                ux_timecourse_rows.append(
                                    {
                                        "trial_id": int(row.trial_id),
                                        "base_trial_id": int(row.base_trial_id),
                                        "layer": layer_key,
                                        "phase": str(timepoint["phase"]),
                                        "phase_step": int(timepoint["phase_step"]),
                                        "time_step": int(timepoint["global_time_step"]),
                                        "time_ms": float(timepoint["time_ms"]),
                                        "ux_value": float(timepoint["layer_means"][layer_key][i]),
                                    }
                                )

    clean_specs = df_specs[df_specs["session_kind"] == "clean"].copy()
    distracted_specs = df_specs[df_specs["session_kind"] == "distracted"].copy()
    process_subset(clean_specs, session_kind="clean")
    process_subset(distracted_specs, session_kind="distracted")

    df_similarity = pd.DataFrame(similarity_rows).drop_duplicates(subset=["trial_id"]).sort_values("trial_id").reset_index(drop=True)
    df_specs_enriched = df_specs.merge(df_similarity, on="trial_id", how="left")
    df_trials = pd.DataFrame(pred_records).sort_values(["trial_id", "stsp_mode"]).reset_index(drop=True)
    df_features = pd.DataFrame(feature_rows)
    df_ux_timecourse = pd.DataFrame(ux_timecourse_rows).sort_values(["trial_id", "layer", "time_step"]).reset_index(drop=True)
    return df_specs_enriched, df_trials, df_features, df_ux_timecourse


def validate_prediction_tables(df_specs: pd.DataFrame, df_trials: pd.DataFrame) -> None:
    expected_rows = len(df_specs) * len(STSP_MODES)
    if len(df_trials) != expected_rows:
        raise ValueError(f"trial_predictions row mismatch: got {len(df_trials)}, expected {expected_rows}")

    counts = df_trials.groupby("trial_id").size()
    if not (counts == len(STSP_MODES)).all():
        raise ValueError("Each trial_id must appear once per stsp mode")

    clean_trials = df_trials[df_trials["session_kind"] == "clean"]
    if not clean_trials["is_correct_distractor"].isna().all():
        raise ValueError("clean rows must keep is_correct_distractor NaN")


def summarize_ux_timecourse(df_ux_timecourse: pd.DataFrame) -> pd.DataFrame:
    if len(df_ux_timecourse) == 0:
        return pd.DataFrame(columns=["layer", "phase", "phase_step", "time_step", "time_ms", "ux_mean", "ux_sem", "n_trials"])
    grouped = (
        df_ux_timecourse.groupby(["layer", "phase", "phase_step", "time_step", "time_ms"], as_index=False)
        .agg(
            ux_mean=("ux_value", "mean"),
            ux_sem=("ux_value", "sem"),
            n_trials=("trial_id", "nunique"),
        )
        .fillna({"ux_sem": 0.0})
        .sort_values(["layer", "time_step"])
        .reset_index(drop=True)
    )
    return grouped

    distracted_trials = df_trials[df_trials["session_kind"] == "distracted"]
    if distracted_trials["is_correct_distractor"].isna().any():
        raise ValueError("distracted rows must define is_correct_distractor")

    row_sum = (
        df_trials["pred_is_sample"].to_numpy(dtype=np.float64)
        + df_trials["pred_is_probe"].to_numpy(dtype=np.float64)
        + df_trials["pred_is_distractor"].to_numpy(dtype=np.float64)
        + df_trials["pred_is_other"].to_numpy(dtype=np.float64)
        + df_trials["pred_is_silent"].to_numpy(dtype=np.float64)
    )
    if np.any(np.abs(row_sum - 1.0) > 1e-8):
        raise ValueError("probe prediction decomposition does not sum to 1")


def validate_family_similarity_columns(df_specs: pd.DataFrame) -> None:
    clean = df_specs[df_specs["session_kind"] == "clean"].copy()
    if len(clean) > 0:
        if clean["label_relation_family"].notna().any():
            raise ValueError("clean rows must not have label_relation_family")
        if clean["pixel_similarity_within_family_bin"].notna().any():
            raise ValueError("clean rows must not have pixel_similarity_within_family_bin")

    distracted = df_specs[df_specs["session_kind"] == "distracted"].copy()
    if len(distracted) == 0:
        raise ValueError("No distracted rows found for family similarity validation")

    expected = {
        "high_same_label": "same_label",
        "high_diff_label": "different_label",
        "medium_confusion": "different_label",
        "low_confusion": "different_label",
    }
    for cond, family in expected.items():
        sub = distracted[distracted["label_similarity_condition"] == cond]
        if len(sub) == 0:
            continue
        got = set(sub["label_relation_family"].dropna().astype(str).tolist())
        if got != {family}:
            raise ValueError(f"{cond} expected family={family}, got={got}")

    for family in FAMILY_ORDER:
        sub = distracted[distracted["label_relation_family"] == family]
        bins = set(sub["pixel_similarity_within_family_bin"].dropna().astype(str).tolist())
        if bins != set(SIMILARITY_BIN_ORDER):
            raise ValueError(f"family={family} missing similarity bins: {bins}")


def validate_confusion_lookup(df_catalog: pd.DataFrame, num_classes: int) -> None:
    expected_tiers = {"high_diff_label", "medium_confusion", "low_confusion"}
    for sample_label in range(num_classes):
        sub = df_catalog[df_catalog["sample_label"] == sample_label]
        tier_set = set(sub[sub["tier"] != ""]["tier"].tolist())
        if tier_set != expected_tiers:
            raise ValueError(f"sample_label={sample_label} does not expose all required tiers")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int, chunk_size: int = 256) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("bootstrap_mean_ci received empty values")
    rng = np.random.default_rng(seed)
    n = vals.size
    boot = np.zeros(n_boot, dtype=np.float64)
    out_start = 0
    while out_start < n_boot:
        block = min(chunk_size, n_boot - out_start)
        idx = rng.integers(0, n, size=(block, n))
        boot[out_start:out_start + block] = vals[idx].mean(axis=1)
        out_start += block
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_ci_scalar(sub: pd.DataFrame, metric_fn, n_boot: int, seed: int) -> Tuple[float, float]:
    rec = sub.to_records(index=False)
    n = len(rec)
    if n == 0:
        raise ValueError("bootstrap_ci_scalar received empty subset")
    rng = np.random.default_rng(seed)
    vals = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(metric_fn(pd.DataFrame.from_records(rec[idx])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_sample_and_noise_bias_ci(
    df_subset: pd.DataFrame,
    num_classes: int,
    has_distractor: bool,
    n_boot: int,
    seed: int,
    chunk_size: int = 128,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    err = df_subset[df_subset["prediction_probe"] != df_subset["probe_label"]]
    if len(err) == 0:
        return (0.0, 0.0), (0.0, 0.0)

    pred = err["prediction_probe"].to_numpy(dtype=np.int64)
    sample = err["sample_label"].to_numpy(dtype=np.int64)
    probe = err["probe_label"].to_numpy(dtype=np.int64)
    distractor = err["distractor_label"].to_numpy(dtype=np.int64) if has_distractor else None

    n = len(err)
    k = num_classes - 3 if has_distractor else num_classes - 2
    if k <= 0:
        raise ValueError("num_classes is too small for noise-bias definition")

    rng = np.random.default_rng(seed)
    sample_boot = np.zeros(n_boot, dtype=np.float64)
    noise_boot = np.zeros(n_boot, dtype=np.float64)
    out_start = 0

    while out_start < n_boot:
        block = min(chunk_size, n_boot - out_start)
        idx = rng.integers(0, n, size=(block, n))
        pred_b = pred[idx]
        sample_b = sample[idx]
        probe_b = probe[idx]

        sample_boot[out_start:out_start + block] = np.mean(pred_b == sample_b, axis=1)

        valid = (pred_b >= 0) & (pred_b < num_classes)
        noise_hit = valid & (pred_b != sample_b) & (pred_b != probe_b)
        if has_distractor and distractor is not None:
            distractor_b = distractor[idx]
            noise_hit = noise_hit & (pred_b != distractor_b)
        noise_boot[out_start:out_start + block] = noise_hit.sum(axis=1) / float(n * k)
        out_start += block

    sample_ci = (float(np.percentile(sample_boot, 2.5)), float(np.percentile(sample_boot, 97.5)))
    noise_ci = (float(np.percentile(noise_boot, 2.5)), float(np.percentile(noise_boot, 97.5)))
    return sample_ci, noise_ci


def compute_sample_and_noise_bias(df_subset: pd.DataFrame, num_classes: int, has_distractor: bool) -> Tuple[float, float]:
    err = df_subset[df_subset["prediction_probe"] != df_subset["probe_label"]]
    if len(err) == 0:
        return 0.0, 0.0

    pred = err["prediction_probe"].to_numpy(dtype=np.int64)
    sample = err["sample_label"].to_numpy(dtype=np.int64)
    probe = err["probe_label"].to_numpy(dtype=np.int64)
    valid = (pred >= 0) & (pred < num_classes)

    bias_sample = float(np.mean(pred == sample))
    noise_hit = valid & (pred != sample) & (pred != probe)
    if has_distractor:
        distractor = err["distractor_label"].to_numpy(dtype=np.int64)
        noise_hit = noise_hit & (pred != distractor)
        k = num_classes - 3
    else:
        k = num_classes - 2
    if k <= 0:
        raise ValueError("num_classes is too small for noise-bias definition")
    bias_noise = float(noise_hit.sum() / float(len(err) * k))
    return bias_sample, bias_noise


def build_stratified_splits(
    labels: np.ndarray,
    n_splits: int,
    test_ratio: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_splits):
        train_idx: List[int] = []
        test_idx: List[int] = []
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) < 2:
                raise ValueError(f"Class {int(cls)} has <2 trials; increase --trials.")
            perm = rng.permutation(cls_idx)
            n_test = max(1, int(round(len(perm) * test_ratio)))
            n_test = min(n_test, len(perm) - 1)
            test_idx.extend(perm[:n_test].tolist())
            train_idx.extend(perm[n_test:].tolist())
        splits.append((np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)))
    return splits


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device: Optional[torch.device] = None,
) -> float:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    if device is None:
        device = torch.device("cpu")

    x_t = torch.as_tensor(x.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y.astype(np.int64, copy=False), dtype=torch.long, device=device)

    accs: List[float] = []
    for train_idx, test_idx in splits:
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)
        x_train = x_t.index_select(0, train_idx_t)
        y_train = y_t.index_select(0, train_idx_t)
        x_test = x_t.index_select(0, test_idx_t)
        y_test = y_t.index_select(0, test_idx_t)

        d = x_t.shape[1]
        counts = torch.bincount(y_train, minlength=num_classes).to(torch.float32)
        valid = counts > 0
        if not torch.any(valid):
            accs.append(0.0)
            continue

        centroids = torch.zeros((num_classes, d), dtype=torch.float32, device=device)
        centroids.index_add_(0, y_train, x_train)
        centroids[valid] = centroids[valid] / counts[valid].unsqueeze(1)

        dist = torch.cdist(x_test, centroids, p=2.0) ** 2
        dist[:, ~valid] = float("inf")
        pred = torch.argmin(dist, dim=1)
        accs.append(float((pred == y_test).to(torch.float32).mean().item()))
    return float(np.mean(accs))


def bootstrap_decode_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    if len(features) != len(labels):
        raise ValueError("Feature/label length mismatch in bootstrap_decode_accuracy")

    by_class = {int(cls): np.where(labels == cls)[0] for cls in np.unique(labels)}
    rng = np.random.default_rng(seed)
    scores = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx_blocks = []
        for cls in sorted(by_class):
            cls_idx = by_class[cls]
            idx_blocks.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
        idx = np.concatenate(idx_blocks, axis=0)
        boot_x = features[idx]
        boot_y = labels[idx]
        splits = build_stratified_splits(
            labels=boot_y,
            n_splits=decode_splits,
            test_ratio=0.3,
            seed=seed + 1000 + b,
        )
        scores[b] = decode_accuracy_with_splits(
            x=boot_x,
            y=boot_y,
            splits=splits,
            num_classes=num_classes,
            device=device,
        )

    return {
        "ci95_lower": float(np.percentile(scores, 2.5)),
        "ci95_upper": float(np.percentile(scores, 97.5)),
        "p_one_sided_gt_chance": float((np.sum(scores <= (1.0 / float(num_classes))) + 1.0) / (len(scores) + 1.0)),
    }


def _binary_metric_with_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return 100.0 * float(values.mean()), 100.0 * lo, 100.0 * hi


def compute_bucket_summary_base(
    df_trials: pd.DataFrame,
    bucket_col: str,
    bucket_values: Sequence[str],
    num_classes: int,
    n_boot: int,
    seed: int,
    clean_dynamic_reference: Optional[pd.DataFrame] = None,
    progress_desc: str = "",
) -> pd.DataFrame:
    if clean_dynamic_reference is None:
        clean_dynamic_all = df_trials[
            (df_trials["label_similarity_condition"] == "clean_reference")
            & (df_trials["stsp_mode"] == "dynamic")
        ].copy()
    else:
        clean_dynamic_all = clean_dynamic_reference.copy()
    rows: List[Dict[str, object]] = []

    bucket_iter = bucket_values
    if progress_desc != "":
        bucket_iter = tqdm(bucket_values, desc=progress_desc, leave=False)

    for bucket_idx, bucket_value in enumerate(bucket_iter):
        bucket_df = df_trials[df_trials[bucket_col] == bucket_value].copy()
        if len(bucket_df) == 0:
            continue

        for mode_idx, stsp_mode in enumerate(STSP_MODES):
            sub = bucket_df[bucket_df["stsp_mode"] == stsp_mode].copy()
            if len(sub) == 0:
                continue

            session_kinds = set(sub["session_kind"].tolist())
            if len(session_kinds) != 1:
                raise ValueError(f"Bucket {bucket_value} mixes session kinds: {session_kinds}")
            has_distractor = "distracted" in session_kinds

            probe_acc = sub["is_correct_probe"].to_numpy(dtype=np.float64)
            probe_acc_mean, probe_acc_lo, probe_acc_hi = _binary_metric_with_ci(
                probe_acc,
                n_boot=n_boot,
                seed=seed + 11 + bucket_idx * 100 + mode_idx * 17,
            )

            if has_distractor:
                distractor_acc = sub["is_correct_distractor"].to_numpy(dtype=np.float64)
                distractor_acc_mean, distractor_acc_lo, distractor_acc_hi = _binary_metric_with_ci(
                    distractor_acc,
                    n_boot=n_boot,
                    seed=seed + 21 + bucket_idx * 100 + mode_idx * 17,
                )
                distractor_silent = sub["is_silent_distractor"].to_numpy(dtype=np.float64)
                distractor_silent_mean, distractor_silent_lo, distractor_silent_hi = _binary_metric_with_ci(
                    distractor_silent,
                    n_boot=n_boot,
                    seed=seed + 31 + bucket_idx * 100 + mode_idx * 17,
                )
            else:
                distractor_acc_mean = distractor_acc_lo = distractor_acc_hi = float("nan")
                distractor_silent_mean = distractor_silent_lo = distractor_silent_hi = float("nan")

            bias_sample, bias_noise = compute_sample_and_noise_bias(sub, num_classes=num_classes, has_distractor=has_distractor)
            sample_bias_ci, noise_bias_ci = bootstrap_sample_and_noise_bias_ci(
                df_subset=sub,
                num_classes=num_classes,
                has_distractor=has_distractor,
                n_boot=n_boot,
                seed=seed + 41 + bucket_idx * 100 + mode_idx * 17,
            )

            matched_base_ids = sub["base_trial_id"].to_numpy(dtype=np.int64)
            clean_matched = clean_dynamic_all[clean_dynamic_all["base_trial_id"].isin(matched_base_ids)].copy()
            clean_sample_bias, _clean_noise_bias = compute_sample_and_noise_bias(
                clean_matched,
                num_classes=num_classes,
                has_distractor=False,
            )
            if bucket_value == "clean_reference" and stsp_mode == "dynamic":
                retention_pct = 100.0 if clean_sample_bias > 0.0 else float("nan")
            else:
                retention_pct = (bias_sample / clean_sample_bias * 100.0) if clean_sample_bias > 0.0 else float("nan")

            rows.append(
                {
                    "bucket_name": str(bucket_value),
                    "bucket_source": str(bucket_col),
                    "session_kind": list(session_kinds)[0],
                    "stsp_mode": stsp_mode,
                    "n_trials": int(len(sub)),
                    "n_error_trials": int((sub["prediction_probe"] != sub["probe_label"]).sum()),
                    "probe_accuracy": float(probe_acc_mean),
                    "probe_accuracy_ci95_lower": float(probe_acc_lo),
                    "probe_accuracy_ci95_upper": float(probe_acc_hi),
                    "distractor_accuracy": float(distractor_acc_mean),
                    "distractor_accuracy_ci95_lower": float(distractor_acc_lo),
                    "distractor_accuracy_ci95_upper": float(distractor_acc_hi),
                    "distractor_silent_rate": float(distractor_silent_mean),
                    "distractor_silent_rate_ci95_lower": float(distractor_silent_lo),
                    "distractor_silent_rate_ci95_upper": float(distractor_silent_hi),
                    "sample_bias": float(bias_sample),
                    "sample_bias_ci95_lower": float(sample_bias_ci[0]),
                    "sample_bias_ci95_upper": float(sample_bias_ci[1]),
                    "noise_bias": float(bias_noise),
                    "noise_bias_ci95_lower": float(noise_bias_ci[0]),
                    "noise_bias_ci95_upper": float(noise_bias_ci[1]),
                    "sample_bias_minus_noise": float(100.0 * (bias_sample - bias_noise)),
                    "sample_bias_retention_pct": float(retention_pct),
                    "matched_clean_dynamic_sample_bias": float(clean_sample_bias),
                    "n_boot": int(n_boot),
                }
            )

    return pd.DataFrame(rows)


def compute_family_similarity_summary(
    df_trials: pd.DataFrame,
    num_classes: int,
    n_boot: int,
    seed: int,
    clean_dynamic_reference: pd.DataFrame,
) -> pd.DataFrame:
    family_bucket_order = [f"{family}_{sim_bin}" for family in FAMILY_ORDER for sim_bin in SIMILARITY_BIN_ORDER]
    base = compute_bucket_summary_base(
        df_trials=df_trials[df_trials["session_kind"] == "distracted"].copy(),
        bucket_col="pixel_similarity_family_bucket",
        bucket_values=family_bucket_order,
        num_classes=num_classes,
        n_boot=n_boot,
        seed=seed,
        clean_dynamic_reference=clean_dynamic_reference,
        progress_desc="FamilySummary",
    )
    if len(base) == 0:
        return base

    parts = base["bucket_name"].astype(str).str.split("_", n=2, expand=True)
    base["label_relation_family"] = parts[0] + "_" + parts[1]
    base["similarity_bin"] = parts[2]
    cols_front = [
        "label_relation_family",
        "similarity_bin",
        "bucket_name",
        "bucket_source",
        "session_kind",
        "stsp_mode",
    ]
    remain = [c for c in base.columns if c not in cols_front]
    return base[cols_front + remain]


def compute_ux_decode_table(
    df_features: pd.DataFrame,
    num_classes: int,
    decode_splits: int,
    n_boot: int,
    seed: int,
    show_progress: bool = False,
    decode_device: Optional[torch.device] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    chance = 1.0 / float(num_classes)
    analysis_specs = [
        ("label_similarity", "label_similarity_condition", LABEL_CONDITION_ORDER, False),
        ("input_similarity_pixel", "pixel_similarity_bin", ["low", "medium", "high"], False),
        ("input_similarity_spike", "spike_similarity_bin", ["low", "medium", "high"], False),
        (
            "family_similarity_pixel",
            "pixel_similarity_family_bucket",
            [f"{family}_{sim_bin}" for family in FAMILY_ORDER for sim_bin in SIMILARITY_BIN_ORDER],
            True,
        ),
    ]

    total_decode_jobs = 0
    for _analysis_source, _bucket_col, bucket_values, _use_dual_targets in analysis_specs:
        target_mult = 2 if _use_dual_targets else 1
        total_decode_jobs += len(bucket_values) * len(STSP_MODES) * len(LAYER_KEYS) * target_mult
    decode_pbar = tqdm(total=total_decode_jobs, desc="UXDecode", leave=False) if show_progress else None

    for analysis_idx, (analysis_source, bucket_col, bucket_values, use_dual_targets) in enumerate(analysis_specs):
        for bucket_idx, bucket_name in enumerate(bucket_values):
            for mode_idx, stsp_mode in enumerate(STSP_MODES):
                sub_mode = df_features[df_features["stsp_mode"] == stsp_mode].copy()
                if analysis_source == "label_similarity":
                    sub = sub_mode[sub_mode[bucket_col] == bucket_name].copy()
                else:
                    sub = sub_mode[
                        (sub_mode["session_kind"] == "distracted")
                        & (sub_mode[bucket_col] == bucket_name)
                    ].copy()
                if len(sub) == 0:
                    continue

                decode_targets = [("sample_label", "sample_label")]
                if use_dual_targets:
                    decode_targets.append(("distractor_label", "distractor_label"))

                for layer_idx, layer in enumerate(LAYER_KEYS):
                    layer_sub = sub[sub["layer"] == layer].copy()
                    if len(layer_sub) == 0:
                        if decode_pbar is not None:
                            decode_pbar.update(len(decode_targets))
                        continue

                    features = np.stack(layer_sub["feature"].tolist(), axis=0).astype(np.float32, copy=False)
                    for target_idx, (decode_target_kind, label_col) in enumerate(decode_targets):
                        if decode_pbar is not None:
                            decode_pbar.update(1)
                        labels = layer_sub[label_col].to_numpy(dtype=np.int64, copy=False)
                        decode_error = ""
                        acc = float("nan")
                        ci_lower = float("nan")
                        ci_upper = float("nan")
                        p_one = float("nan")
                        try:
                            splits = build_stratified_splits(
                                labels=labels,
                                n_splits=decode_splits,
                                test_ratio=0.3,
                                seed=seed + 101 + analysis_idx * 1000 + bucket_idx * 100 + mode_idx * 20 + layer_idx * 3 + target_idx,
                            )
                            acc = decode_accuracy_with_splits(
                                x=features,
                                y=labels,
                                splits=splits,
                                num_classes=num_classes,
                                device=decode_device,
                            )
                            boot = bootstrap_decode_accuracy(
                                features=features,
                                labels=labels,
                                num_classes=num_classes,
                                decode_splits=decode_splits,
                                n_boot=n_boot,
                                seed=seed + 201 + analysis_idx * 1000 + bucket_idx * 100 + mode_idx * 20 + layer_idx * 3 + target_idx,
                                device=decode_device,
                            )
                            ci_lower = float(boot["ci95_lower"])
                            ci_upper = float(boot["ci95_upper"])
                            p_one = float(boot["p_one_sided_gt_chance"])
                        except ValueError as exc:
                            decode_error = str(exc)

                        rows.append(
                            {
                                "analysis_source": analysis_source,
                                "bucket_name": str(bucket_name),
                                "stsp_mode": stsp_mode,
                                "layer": layer,
                                "decode_target_kind": decode_target_kind,
                                "n_trials": int(len(layer_sub)),
                                "ux_decode_acc": float(acc),
                                "ux_decode_acc_pct": 100.0 * float(acc) if np.isfinite(acc) else float("nan"),
                                "ux_decode_acc_ci95_lower": float(ci_lower),
                                "ux_decode_acc_ci95_upper": float(ci_upper),
                                "ux_decode_acc_ci95_lower_pct": 100.0 * float(ci_lower) if np.isfinite(ci_lower) else float("nan"),
                                "ux_decode_acc_ci95_upper_pct": 100.0 * float(ci_upper) if np.isfinite(ci_upper) else float("nan"),
                                "chance_level": float(chance),
                                "chance_level_pct": 100.0 * float(chance),
                                "p_one_sided_gt_chance": float(p_one),
                                "decode_error": decode_error,
                                "decode_splits": int(decode_splits),
                                "n_boot": int(n_boot),
                            }
                        )

    if decode_pbar is not None:
        decode_pbar.close()
    return pd.DataFrame(rows)


def merge_summary_with_decode(
    df_summary_base: pd.DataFrame,
    df_decode: pd.DataFrame,
    analysis_source: str,
    decode_target_kind: str = "sample_label",
) -> pd.DataFrame:
    decode_sub = df_decode[
        (df_decode["analysis_source"] == analysis_source)
        & (df_decode["decode_target_kind"] == decode_target_kind)
    ].copy()
    decode_sub = decode_sub.rename(
        columns={
            "n_trials": "ux_decode_n_trials",
            "n_boot": "ux_decode_n_boot",
            "decode_splits": "ux_decode_splits",
        }
    )
    return df_summary_base.merge(
        decode_sub,
        on=["bucket_name", "stsp_mode"],
        how="left",
    )


def build_ux_decode_competition(df_decode: pd.DataFrame) -> pd.DataFrame:
    sub = df_decode[df_decode["analysis_source"] == "family_similarity_pixel"].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    key_cols = ["bucket_name", "stsp_mode", "layer"]
    sample_sub = sub[sub["decode_target_kind"] == "sample_label"][key_cols + ["ux_decode_acc_pct"]].rename(
        columns={"ux_decode_acc_pct": "sample_decode_acc"}
    )
    dist_sub = sub[sub["decode_target_kind"] == "distractor_label"][key_cols + ["ux_decode_acc_pct"]].rename(
        columns={"ux_decode_acc_pct": "distractor_decode_acc"}
    )
    out = sample_sub.merge(dist_sub, on=key_cols, how="outer")
    parts = out["bucket_name"].astype(str).str.split("_", n=2, expand=True)
    out["label_relation_family"] = parts[0] + "_" + parts[1]
    out["similarity_bin"] = parts[2]
    out["sample_minus_distractor_decode"] = out["sample_decode_acc"] - out["distractor_decode_acc"]
    front = ["label_relation_family", "similarity_bin", "bucket_name", "stsp_mode", "layer"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].sort_values(front).reset_index(drop=True)


def validate_decode_table(df_decode: pd.DataFrame) -> None:
    if len(df_decode) == 0:
        raise ValueError("metrics_ux_decode_by_layer is empty")
    finite = df_decode["ux_decode_acc"].dropna()
    if len(finite) == 0:
        raise ValueError("No finite decode accuracy found in metrics_ux_decode_by_layer")


def plot_label_similarity_summary(df_summary: pd.DataFrame, save_path: str) -> None:
    metric_df = df_summary.drop_duplicates(subset=["bucket_name", "stsp_mode"]).copy()
    order = [x for x in LABEL_CONDITION_ORDER if x in metric_df["bucket_name"].tolist()]

    sns.set(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metric_specs = [
        ("probe_accuracy", "Probe accuracy (%)"),
        ("distractor_accuracy", "Distractor accuracy (%)"),
        ("sample_bias_retention_pct", "Sample-bias retention (%)"),
        ("sample_bias_minus_noise", "Sample bias - noise bias (pp)"),
    ]
    palette = {"dynamic": "#d62728", "static_frozen": "#7f7f7f"}

    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        sns.barplot(data=metric_df, x="bucket_name", y=metric, hue="stsp_mode", order=order, palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        if metric != "sample_bias_minus_noise":
            ax.set_ylim(0, 100)
        ax.legend(title="")

    fig.suptitle("Dual-task distractor similarity boundary: label-level summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_input_similarity_summary_on_axes(axes: Sequence[plt.Axes], df_summary: pd.DataFrame, source_name: str) -> None:
    metric_df = df_summary.drop_duplicates(subset=["bucket_name", "stsp_mode"]).copy()
    order = [x for x in ["low", "medium", "high"] if x in metric_df["bucket_name"].tolist()]
    flat_axes = np.asarray(list(axes), dtype=object).reshape(-1)

    if len(flat_axes) != 4:
        raise ValueError("plot_input_similarity_summary_on_axes expects exactly four axes")
    metric_specs = [
        ("probe_accuracy", "Probe accuracy (%)"),
        ("distractor_accuracy", "Distractor accuracy (%)"),
        ("sample_bias_retention_pct", "Sample-bias retention (%)"),
        ("sample_bias_minus_noise", "Sample bias - noise bias (pp)"),
    ]
    palette = {"dynamic": "#d62728", "static_frozen": "#7f7f7f"}

    for ax, (metric, title) in zip(flat_axes, metric_specs):
        sns.barplot(data=metric_df, x="bucket_name", y=metric, hue="stsp_mode", order=order, palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        if metric != "sample_bias_minus_noise":
            ax.set_ylim(0, 100)
        ax.legend(title="")


def plot_input_similarity_summary(df_summary: pd.DataFrame, source_name: str, save_path: str) -> None:
    sns.set(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    plot_input_similarity_summary_on_axes(list(axes.flatten()), df_summary, source_name)
    fig.suptitle(f"Input-level similarity summary ({source_name})", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_family_similarity_summary(df_summary: pd.DataFrame, save_path: str) -> None:
    metric_df = df_summary.drop_duplicates(subset=["label_relation_family", "similarity_bin", "stsp_mode"]).copy()
    metric_df["family_bin_label"] = metric_df["label_relation_family"].astype(str) + "\n" + metric_df["similarity_bin"].astype(str)
    order = [f"{family}\n{sim_bin}" for family in FAMILY_ORDER for sim_bin in SIMILARITY_BIN_ORDER]

    sns.set(style="whitegrid", font_scale=0.92)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9))
    metric_specs = [
        ("probe_accuracy", "Probe accuracy (%)"),
        ("distractor_accuracy", "Distractor accuracy (%)"),
        ("sample_bias_retention_pct", "Sample-bias retention (%)"),
        ("sample_bias_minus_noise", "Sample bias - noise bias (pp)"),
    ]
    palette = {"dynamic": "#d62728", "static_frozen": "#7f7f7f"}

    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        sns.barplot(
            data=metric_df,
            x="family_bin_label",
            y=metric,
            hue="stsp_mode",
            order=order,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        if metric != "sample_bias_minus_noise":
            ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="")

    fig.suptitle("Family-specific pixel similarity hierarchy", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ux_decode(df_decode: pd.DataFrame, save_path: str) -> None:
    df_decode = df_decode[df_decode["decode_target_kind"] == "sample_label"].copy()
    analysis_order = [
        ("label_similarity", LABEL_CONDITION_ORDER, "Label similarity"),
        ("input_similarity_pixel", ["low", "medium", "high"], "Pixel-sim bins"),
        ("input_similarity_spike", ["low", "medium", "high"], "Spike-sim bins"),
    ]
    layer_colors = {"layer1": "#1f77b4", "layer2": "#ff7f0e", "layer3": "#2ca02c"}
    mode_styles = {"dynamic": "-", "static_frozen": "--"}

    sns.set(style="whitegrid", font_scale=0.92)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (analysis_source, bucket_order, title) in zip(axes, analysis_order):
        sub = df_decode[df_decode["analysis_source"] == analysis_source].copy()
        if len(sub) == 0:
            ax.set_axis_off()
            continue

        x = np.arange(len(bucket_order))
        for layer in LAYER_KEYS:
            for stsp_mode in STSP_MODES:
                g = sub[(sub["layer"] == layer) & (sub["stsp_mode"] == stsp_mode)].copy()
                if len(g) == 0:
                    continue
                g = g.set_index("bucket_name").reindex(bucket_order).reset_index()
                y = g["ux_decode_acc_pct"].to_numpy(dtype=np.float64)
                ax.plot(
                    x,
                    y,
                    linestyle=mode_styles[stsp_mode],
                    color=layer_colors[layer],
                    marker="o",
                    linewidth=2,
                    label=f"{layer}-{stsp_mode}",
                )

        chance = float(sub["chance_level_pct"].dropna().iloc[0]) if sub["chance_level_pct"].notna().any() else 0.0
        ax.axhline(chance, color="black", linewidth=1.2, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_order, rotation=25)
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel("u*x decode accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=True)
    fig.suptitle("Probe-pre u*x sample-label decoding", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ux_decode_competition(df_comp: pd.DataFrame, save_path: str) -> None:
    if len(df_comp) == 0:
        return

    sns.set(style="whitegrid", font_scale=0.94)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)
    layer_colors = {"layer1": "#1f77b4", "layer2": "#ff7f0e", "layer3": "#2ca02c"}
    x = np.arange(len(SIMILARITY_BIN_ORDER))
    legend_handles = []
    legend_labels = []

    for ax, family in zip(axes, FAMILY_ORDER):
        sub = df_comp[df_comp["label_relation_family"] == family].copy()
        if len(sub) == 0:
            ax.set_axis_off()
            continue

        for layer in LAYER_KEYS:
            layer_sub = sub[(sub["layer"] == layer) & (sub["stsp_mode"] == "dynamic")].copy()
            if len(layer_sub) == 0:
                continue
            layer_sub = layer_sub.set_index("similarity_bin").reindex(SIMILARITY_BIN_ORDER).reset_index()
            sample_line, = ax.plot(
                x,
                layer_sub["sample_decode_acc"].to_numpy(dtype=np.float64),
                color=layer_colors[layer],
                linestyle="-",
                marker="o",
                linewidth=2,
                label=f"{layer} sample",
            )
            distractor_line, = ax.plot(
                x,
                layer_sub["distractor_decode_acc"].to_numpy(dtype=np.float64),
                color=layer_colors[layer],
                linestyle="--",
                marker="o",
                linewidth=2,
                label=f"{layer} distractor",
            )
            if family == FAMILY_ORDER[0]:
                legend_handles.extend([sample_line, distractor_line])
                legend_labels.extend([f"{layer} sample (solid)", f"{layer} distractor (dashed)"])

        ax.set_xticks(x)
        ax.set_xticklabels(SIMILARITY_BIN_ORDER)
        ax.set_ylim(0, 100)
        ax.set_title(family)
        ax.set_ylabel("Decode accuracy (%)")

    if len(legend_handles) > 0:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.89),
            ncol=3,
            fontsize=9,
            frameon=True,
        )
    fig.suptitle("u*x competition decode: sample vs distractor", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.74])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ux_timecourse_distracted_dynamic(df_summary: pd.DataFrame, save_path: str) -> None:
    if len(df_summary) == 0:
        return

    sns.set(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    layer_colors = {"layer1": "#1f77b4", "layer2": "#ff7f0e", "layer3": "#2ca02c"}
    phase_order = ["sample", "delay1", "distractor", "delay2", "probe"]
    phase_bounds = (
        df_summary[["phase", "time_ms"]]
        .drop_duplicates()
        .groupby("phase", as_index=False)["time_ms"]
        .agg(["min", "max"])
        .reset_index()
    )
    phase_bounds.columns = ["phase", "start_ms", "end_ms"]
    for ax, layer_key in zip(axes, LAYER_KEYS):
        sub = df_summary[df_summary["layer"] == layer_key].copy().sort_values("time_step")
        if len(sub) == 0:
            ax.set_axis_off()
            continue
        x = sub["time_ms"].to_numpy(dtype=np.float64)
        y = sub["ux_mean"].to_numpy(dtype=np.float64)
        yerr = sub["ux_sem"].to_numpy(dtype=np.float64)
        color = layer_colors[layer_key]
        ax.plot(x, y, color=color, linewidth=2.2)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.16, linewidth=0)
        ax.set_title(layer_key)
        for _, phase_row in phase_bounds.iterrows():
            if phase_row["phase"] not in phase_order:
                continue
            ax.axvspan(float(phase_row["start_ms"]), float(phase_row["end_ms"]), color="black", alpha=0.03)
            ax.axvline(float(phase_row["end_ms"]), color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Session Time (ms)")
        ax.set_ylabel("Mean u*x")
    phase_labels = [
        f"{row.phase}"
        for row in phase_bounds.sort_values("start_ms").itertuples(index=False)
        if row.phase in phase_order
    ]
    if len(phase_labels) > 0:
        fig.text(0.5, 0.01, "Phases: " + " | ".join(phase_labels), ha="center", fontsize=9)
    fig.suptitle("Whole-session u*x time course (dynamic, distracted trials)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-task distractor similarity boundary experiment")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/dual_task_similarity_boundary")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay1-ms", type=float, default=300.0)
    parser.add_argument("--distractor-ms", type=float, default=120.0)
    parser.add_argument("--delay2-ms", type=float, default=380.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--num-boot", type=int, default=100)
    parser.add_argument("--decode-splits", type=int, default=5)
    parser.add_argument("--input-sim-bins", type=int, default=3)
    parser.add_argument("--skip-confusion-recompute", action="store_true")
    parser.add_argument("--skip-interface-check", action="store_true")
    parser.add_argument("--no-phase-reset", action="store_true")
    return parser


seed_everything = shared_seed_everything
build_class_index = shared_build_class_index
encode_images = shared_encode_images


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    shared_compensate_stsp_gain(net, scaling_factor=scaling_factor)


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    return shared_load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max(spec.sample_ms, spec.distractor_ms, spec.probe_ms),
    )


def decode_accuracy_with_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device: Optional[torch.device] = None,
) -> float:
    return shared_decode_accuracy_with_splits(
        x=x,
        y=y,
        splits=splits,
        num_classes=num_classes,
        device=device,
    )


def main() -> None:
    global NUM_CLASSES_GLOBAL

    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_classes < 4:
        raise ValueError("--num-classes must be >= 4")
    if args.num_boot <= 0:
        raise ValueError("--num-boot must be positive")
    if args.decode_splits <= 0:
        raise ValueError("--decode-splits must be positive")
    if args.input_sim_bins != 3:
        raise ValueError("This implementation currently expects --input-sim-bins=3")

    NUM_CLASSES_GLOBAL = int(args.num_classes)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=args.sample_ms,
        delay1_ms=args.delay1_ms,
        distractor_ms=args.distractor_ms,
        delay2_ms=args.delay2_ms,
        probe_ms=args.probe_ms,
        phase_reset=(not args.no_phase_reset),
    )
    for name, steps in [
        ("sample", spec.sample_steps),
        ("delay1", spec.delay1_steps),
        ("distractor", spec.distractor_steps),
        ("delay2", spec.delay2_steps),
        ("probe", spec.probe_steps),
    ]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {args.save_dir}")
    print(
        f"[Init] Timing steps | sample={spec.sample_steps}, delay1={spec.delay1_steps}, "
        f"distractor={spec.distractor_steps}, delay2={spec.delay2_steps}, probe={spec.probe_steps}"
    )

    net, encoder = load_model_and_encoder(args.model_path, device, spec)
    if not args.skip_interface_check:
        run_interface_check(net, device)
        print("[Check] forward_dual_task_session interface check passed.")

    _, _, test_loader = build_mnist_skeleton_loader(root=args.dataset_root, batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)

    confusion_csv = os.path.join(args.save_dir, "confusion_matrix.csv")
    catalog_csv = os.path.join(args.save_dir, "confusion_pair_catalog.csv")
    if args.skip_confusion_recompute:
        df_confusion, df_catalog = load_confusion_artifacts(confusion_csv, catalog_csv, num_classes=args.num_classes)
    else:
        counts, norm, silent = compute_confusion_artifacts(
            net=net,
            encoder=encoder,
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            sample_steps=spec.sample_steps,
            device=device,
        )
        totals = counts.sum(axis=1) + silent
        df_confusion = build_confusion_matrix_table(counts=counts, norm=norm, silent=silent, totals=totals)
        df_catalog = build_confusion_pair_catalog(counts=counts, norm=norm, totals=totals)
        df_confusion.to_csv(confusion_csv, index=False)
        df_catalog.to_csv(catalog_csv, index=False)

    validate_confusion_lookup(df_catalog, num_classes=args.num_classes)
    confusion_lookup = build_confusion_lookup(df_catalog, num_classes=args.num_classes)

    rng = random.Random(args.seed)
    df_specs = generate_trial_specs(
        class_index=class_index,
        confusion_lookup=confusion_lookup,
        num_trials=args.trials,
        num_classes=args.num_classes,
        rng=rng,
    )
    validate_trial_specs(df_specs, num_classes=args.num_classes)

    df_specs, df_trials, df_features, df_ux_timecourse = run_experiment(
        net=net,
        encoder=encoder,
        dataset=dataset,
        df_specs=df_specs,
        spec=spec,
        batch_size=args.batch_size,
        device=device,
    )
    df_specs = assign_similarity_bins(df_specs, num_bins=args.input_sim_bins)
    df_specs = assign_family_similarity_bins(df_specs, num_bins=args.input_sim_bins)
    df_ux_timecourse_summary = summarize_ux_timecourse(df_ux_timecourse)
    df_trials = df_trials.drop(columns=["pixel_cosine_similarity", "input_spike_count_cosine_similarity"], errors="ignore").merge(
        df_specs[
            [
                "trial_id",
                "pixel_cosine_similarity",
                "input_spike_count_cosine_similarity",
                "pixel_similarity_bin",
                "spike_similarity_bin",
                "label_relation_family",
                "pixel_similarity_within_family_bin",
                "pixel_similarity_family_bucket",
            ]
        ],
        on="trial_id",
        how="left",
    )
    df_features = df_features.merge(
        df_specs[
            [
                "trial_id",
                "pixel_similarity_bin",
                "spike_similarity_bin",
                "label_relation_family",
                "pixel_similarity_within_family_bin",
                "pixel_similarity_family_bucket",
            ]
        ],
        on="trial_id",
        how="left",
    )

    validate_family_similarity_columns(df_specs)
    validate_prediction_tables(df_specs, df_trials)

    clean_dynamic_reference = df_trials[
        (df_trials["label_similarity_condition"] == "clean_reference")
        & (df_trials["stsp_mode"] == "dynamic")
    ].copy()
    decode_device = device if device.type == "cuda" else torch.device("cpu")
    print(f"[Post] u*x decode device: {decode_device}")

    post_stages = tqdm(
        total=10,
        desc="PostProcess",
        leave=True,
    )

    label_summary_base = compute_bucket_summary_base(
        df_trials=df_trials,
        bucket_col="label_similarity_condition",
        bucket_values=LABEL_CONDITION_ORDER,
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed + 101,
        clean_dynamic_reference=clean_dynamic_reference,
        progress_desc="LabelSummary",
    )
    post_stages.update(1)
    pixel_summary_base = compute_bucket_summary_base(
        df_trials=df_trials[df_trials["session_kind"] == "distracted"].copy(),
        bucket_col="pixel_similarity_bin",
        bucket_values=["low", "medium", "high"],
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed + 201,
        clean_dynamic_reference=clean_dynamic_reference,
        progress_desc="PixelSummary",
    )
    post_stages.update(1)
    spike_summary_base = compute_bucket_summary_base(
        df_trials=df_trials[df_trials["session_kind"] == "distracted"].copy(),
        bucket_col="spike_similarity_bin",
        bucket_values=["low", "medium", "high"],
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed + 301,
        clean_dynamic_reference=clean_dynamic_reference,
        progress_desc="SpikeSummary",
    )
    post_stages.update(1)
    family_summary_base = compute_family_similarity_summary(
        df_trials=df_trials,
        num_classes=args.num_classes,
        n_boot=args.num_boot,
        seed=args.seed + 351,
        clean_dynamic_reference=clean_dynamic_reference,
    )
    post_stages.update(1)

    df_decode = compute_ux_decode_table(
        df_features=df_features,
        num_classes=args.num_classes,
        decode_splits=args.decode_splits,
        n_boot=args.num_boot,
        seed=args.seed + 401,
        show_progress=True,
        decode_device=decode_device,
    )
    validate_decode_table(df_decode)
    post_stages.update(1)

    label_summary = merge_summary_with_decode(
        df_summary_base=label_summary_base,
        df_decode=df_decode,
        analysis_source="label_similarity",
    )
    pixel_summary = merge_summary_with_decode(
        df_summary_base=pixel_summary_base,
        df_decode=df_decode,
        analysis_source="input_similarity_pixel",
    )
    spike_summary = merge_summary_with_decode(
        df_summary_base=spike_summary_base,
        df_decode=df_decode,
        analysis_source="input_similarity_spike",
    )
    family_summary = merge_summary_with_decode(
        df_summary_base=family_summary_base,
        df_decode=df_decode,
        analysis_source="family_similarity_pixel",
    )
    df_decode_competition = build_ux_decode_competition(df_decode)
    post_stages.update(1)

    trial_specs_csv = os.path.join(args.save_dir, "trial_specs.csv")
    trial_predictions_csv = os.path.join(args.save_dir, "trial_predictions.csv")
    label_summary_csv = os.path.join(args.save_dir, "metrics_label_similarity_summary.csv")
    pixel_summary_csv = os.path.join(args.save_dir, "metrics_input_similarity_summary_pixel.csv")
    spike_summary_csv = os.path.join(args.save_dir, "metrics_input_similarity_summary_spike.csv")
    family_summary_csv = os.path.join(args.save_dir, "metrics_family_similarity_summary.csv")
    decode_csv = os.path.join(args.save_dir, "metrics_ux_decode_by_layer.csv")
    decode_competition_csv = os.path.join(args.save_dir, "metrics_ux_decode_competition.csv")
    ux_timecourse_csv = os.path.join(args.save_dir, "metrics_ux_timecourse_distracted_dynamic.csv")
    run_config_json = os.path.join(args.save_dir, "run_config.json")

    df_specs.to_csv(trial_specs_csv, index=False)
    df_trials.to_csv(trial_predictions_csv, index=False)
    label_summary.to_csv(label_summary_csv, index=False)
    pixel_summary.to_csv(pixel_summary_csv, index=False)
    spike_summary.to_csv(spike_summary_csv, index=False)
    family_summary.to_csv(family_summary_csv, index=False)
    df_decode.to_csv(decode_csv, index=False)
    df_decode_competition.to_csv(decode_competition_csv, index=False)
    df_ux_timecourse_summary.to_csv(ux_timecourse_csv, index=False)
    post_stages.update(1)

    plot_label_similarity_summary(label_summary, os.path.join(args.save_dir, "plot_label_similarity_summary.png"))
    plot_input_similarity_summary(
        pixel_summary,
        source_name="pixel cosine similarity",
        save_path=os.path.join(args.save_dir, "plot_input_similarity_summary_pixel.png"),
    )
    plot_input_similarity_summary(
        spike_summary,
        source_name="input spike-count cosine similarity",
        save_path=os.path.join(args.save_dir, "plot_input_similarity_summary_spike.png"),
    )
    plot_family_similarity_summary(
        family_summary,
        os.path.join(args.save_dir, "plot_family_similarity_summary.png"),
    )
    plot_ux_decode(df_decode, os.path.join(args.save_dir, "plot_ux_decode_by_layer.png"))
    plot_ux_decode_competition(
        df_decode_competition,
        os.path.join(args.save_dir, "plot_ux_decode_competition.png"),
    )
    plot_ux_timecourse_distracted_dynamic(
        df_ux_timecourse_summary,
        os.path.join(args.save_dir, "plot_ux_timecourse_distracted_dynamic.png"),
    )
    post_stages.update(1)

    run_config = {
        "model_path": args.model_path,
        "save_dir": args.save_dir,
        "dataset_root": args.dataset_root,
        "trials": int(args.trials),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "num_classes": int(args.num_classes),
        "sample_ms": float(args.sample_ms),
        "delay1_ms": float(args.delay1_ms),
        "distractor_ms": float(args.distractor_ms),
        "delay2_ms": float(args.delay2_ms),
        "probe_ms": float(args.probe_ms),
        "num_boot": int(args.num_boot),
        "decode_splits": int(args.decode_splits),
        "input_sim_bins": int(args.input_sim_bins),
        "phase_reset": bool(spec.phase_reset),
        "skip_confusion_recompute": bool(args.skip_confusion_recompute),
        "label_condition_order": LABEL_CONDITION_ORDER,
        "family_similarity_enabled": True,
        "family_similarity_source": "pixel",
        "family_order": FAMILY_ORDER,
        "family_similarity_bin_order": SIMILARITY_BIN_ORDER,
        "decode_target_kinds": ["sample_label", "distractor_label"],
        "ux_timecourse_monitoring": {
            "enabled": True,
            "window": "whole_session_sample_to_probe",
            "stsp_mode": "dynamic",
            "statistic": "mean_over_presynaptic_units_then_mean_sem_over_trials",
            "layout": "three_layer_facets",
            "phases": ["sample", "delay1", "distractor", "delay2", "probe"],
        },
        "confusion_lookup": confusion_lookup,
        "device": str(device),
    }
    with open(run_config_json, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    post_stages.update(1)
    post_stages.close()

    print("[Done] Dual-task similarity boundary experiment completed.")
    print(f"[Done] Saved: {trial_specs_csv}")
    print(f"[Done] Saved: {trial_predictions_csv}")
    print(f"[Done] Saved: {confusion_csv}")
    print(f"[Done] Saved: {catalog_csv}")
    print(f"[Done] Saved: {label_summary_csv}")
    print(f"[Done] Saved: {pixel_summary_csv}")
    print(f"[Done] Saved: {spike_summary_csv}")
    print(f"[Done] Saved: {family_summary_csv}")
    print(f"[Done] Saved: {decode_csv}")
    print(f"[Done] Saved: {decode_competition_csv}")
    print(f"[Done] Saved: {ux_timecourse_csv}")
    print(f"[Done] Saved: {run_config_json}")


if __name__ == "__main__":
    main()
