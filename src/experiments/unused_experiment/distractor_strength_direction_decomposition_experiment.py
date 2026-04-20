from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.units import ms
from src.experiments.common.dataset import encode_images
from src.experiments.common.dataset import build_class_index
from src.experiments.common.json_io import save_json_payload as _save_json
from src.experiments.common.monitored_dms import run_dms_snapshot_rollout
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor.shared.config import DEFAULT_TAU_MS
from src.experiments.distractor.shared.pair_sampling import build_dataset_arrays
from src.experiments.distractor.shared.triplets import (
    TripletMaskBundle,
    _augment_triplet_specs_with_mask_metadata,
    _build_condition_mask_batch,
    _load_dataset,
    build_probe_relevant_masks_for_triplet,
    build_triplet_specs,
    prepare_triplet_spike_batch,
    run_overlap_perturbed_distractor_task,
)
from src.plotting.common.io import (
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
)
from src.plotting.common.theme_tokens import (
    ALPHA_BAR,
    ALPHA_FILL,
    ALPHA_GUIDE,
    ALPHA_SCATTER,
    CMAP_OVERLAP,
    DISTRACTOR_MAIN_CONDITION_COLORS,
    FIGSIZE_SINGLE_PANEL_MEDIUM,
    FIGSIZE_SINGLE_PANEL_MEDIUM_TALL,
    FIGSIZE_THREE_PANEL,
    FIGSIZE_THREE_PANEL_WIDE,
    FIGSIZE_TWO_PANEL,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
    MARKER_CIRCLE,
    apply_standard_legend,
    case_grid_figsize,
)

DEFAULT_MODEL_PATH = "results/sdnn_deep_final/net_final.pth"
DEFAULT_OUTPUT_DIR = "results/distractor_strength_direction_decomposition_experiment"
DEFAULT_DATASET_ROOT = "./MNIST"
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_DELAY1_MS = 400.0
DEFAULT_DISTRACTOR_MS = 200.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_DIRECTION_DELAY_MS = 400.0
DEFAULT_DELAY_SWEEP_MS: tuple[float, ...] = (100.0, 150.0, 200.0, 300.0, 400.0, 500.0)
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_PROBES = 20
DEFAULT_SAMPLES_PER_PROBE = 12
DEFAULT_MAX_TRIPLETS = 240
DEFAULT_NUM_SIM_BINS = 4
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_DILATION_RADIUS = 1
DEFAULT_SAVE_CASE_COUNT = 4
DEFAULT_NUM_CONTROL_CANDIDATES = 64
EPS = 1e-12

OVERLAP_CONDITION_ORDER: tuple[str, ...] = (
    "full_dynamic",
    "sample_remove_SPonly",
    "distractor_remove_DPonly",
    "sample_remove_SDP",
    "distractor_remove_SDP",
    "both_remove_SDP",
)

CONDITION_COLORS: dict[str, str] = dict(DISTRACTOR_MAIN_CONDITION_COLORS)

ANCHOR_SIMILARITY_BIN_ORDER: tuple[str, ...] = ("low", "middle", "high")

@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool = True

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay1_steps(self) -> int:
        return int(round((self.delay1_ms * ms) / self.dt))

    @property
    def distractor_steps(self) -> int:
        return int(round((self.distractor_ms * ms) / self.dt))

    @property
    def delay2_steps(self) -> int:
        return int(round((self.delay2_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    stsp_mode: str
    sample_mask_key: str | None
    distractor_mask_key: str | None


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


def _sem(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _condition_spec_table() -> dict[str, ConditionSpec]:
    return {
        "full_dynamic": ConditionSpec("full_dynamic", "dynamic", None, None),
        "full_static": ConditionSpec("full_static", "static_frozen", None, None),
        "sample_remove_SPonly": ConditionSpec("sample_remove_SPonly", "dynamic", "sample_sp_only_mask", None),
        "distractor_remove_DPonly": ConditionSpec("distractor_remove_DPonly", "dynamic", None, "distractor_dp_only_mask"),
        "sample_remove_SDP": ConditionSpec("sample_remove_SDP", "dynamic", "sample_sdp_mask", None),
        "distractor_remove_SDP": ConditionSpec("distractor_remove_SDP", "dynamic", None, "distractor_sdp_mask"),
        "both_remove_SDP": ConditionSpec("both_remove_SDP", "dynamic", "sample_sdp_mask", "distractor_sdp_mask"),
    }


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    if allow_zero:
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _sanitize_delay_sweep(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("--delay-sweep-ms must contain at least one value.")
    delays = sorted(dict.fromkeys(float(v) for v in values))
    for delay_ms in delays:
        if delay_ms < 0.0:
            raise ValueError("Delay sweep values must be non-negative.")
    return delays


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


def compute_direction_rotation_metrics(
    delta_v_by_condition: Mapping[str, np.ndarray],
    *,
    eps: float = EPS,
) -> dict[str, object]:
    if "full_dynamic" not in delta_v_by_condition:
        raise KeyError("delta_v_by_condition must contain full_dynamic.")
    u_full = safe_normalize(delta_v_by_condition["full_dynamic"], eps=eps)
    if u_full is None:
        return {"status": "skip_invalid_full_dynamic", "u_full": None, "metrics": {}}
    metrics: dict[str, dict[str, float]] = {}
    for condition_name, vec in delta_v_by_condition.items():
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        magnitude = float(np.linalg.norm(arr))
        if condition_name == "full_dynamic":
            metrics[str(condition_name)] = {"cos_phi": 1.0, "phi_deg": 0.0, "M_condition": magnitude}
            continue
        cos_phi = safe_cosine(arr, u_full, eps=eps)
        metrics[str(condition_name)] = {
            "cos_phi": cos_phi,
            "phi_deg": safe_angle_deg(cos_phi),
            "M_condition": magnitude,
        }
    return {"status": "ok", "u_full": u_full, "metrics": metrics}


def build_source_anchor_specs_from_triplets(df_triplets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Legacy helper kept for backward compatibility only.
    # The current primary analysis uses triplet-specific personalized anchors,
    # not probe-level averaged anchors produced by this function.
    if df_triplets.empty:
        empty = pd.DataFrame(
            columns=[
                "probe_id",
                "probe_label",
                "preceding_id",
                "preceding_label",
                "anchor_type",
                "pair_key",
                "n_pairs_for_probe",
            ]
        )
        return {"uSP_pairs": empty.copy(), "uDP_pairs": empty.copy()}

    def _build(*, preceding_id_col: str, preceding_label_col: str, anchor_type: str) -> pd.DataFrame:
        out = (
            df_triplets[
                ["probe_id", "probe_label", preceding_id_col, preceding_label_col]
            ]
            .rename(columns={preceding_id_col: "preceding_id", preceding_label_col: "preceding_label"})
            .drop_duplicates(subset=["probe_id", "preceding_id"], keep="first")
            .sort_values(["probe_id", "preceding_id"], kind="stable")
            .reset_index(drop=True)
        )
        out["anchor_type"] = str(anchor_type)
        out["pair_key"] = out["probe_id"].astype(str) + "::" + out["preceding_id"].astype(str)
        counts = out.groupby("probe_id", sort=True)["preceding_id"].size().rename("n_pairs_for_probe").reset_index()
        out = out.merge(counts, on="probe_id", how="left")
        return out[
            [
                "probe_id",
                "probe_label",
                "preceding_id",
                "preceding_label",
                "anchor_type",
                "pair_key",
                "n_pairs_for_probe",
            ]
        ].copy()

    return {
        "uSP_pairs": _build(preceding_id_col="sample_id", preceding_label_col="sample_label", anchor_type="uSP"),
        "uDP_pairs": _build(preceding_id_col="distractor_id", preceding_label_col="distractor_label", anchor_type="uDP"),
    }


def build_triplet_source_anchor_specs(df_triplets: pd.DataFrame) -> pd.DataFrame:
    if df_triplets.empty:
        return pd.DataFrame(
            columns=[
                "triplet_id",
                "probe_id",
                "probe_label",
                "sample_id",
                "sample_label",
                "distractor_id",
                "distractor_label",
            ]
        )
    return (
        df_triplets[
            [
                "triplet_id",
                "probe_id",
                "probe_label",
                "sample_id",
                "sample_label",
                "distractor_id",
                "distractor_label",
            ]
        ]
        .copy()
        .sort_values(["triplet_id"], kind="stable")
        .reset_index(drop=True)
    )


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


def compute_probe_source_anchors(
    *,
    net,
    images: torch.Tensor,
    encoder,
    device: torch.device,
    readout_step: int,
    spec: ExperimentSpec,
    sample_anchor_pairs: pd.DataFrame,
    distractor_anchor_pairs: pd.DataFrame,
    batch_size: int,
    eps: float = EPS,
) -> dict[str, object]:
    # Legacy probe-averaged anchor helper kept for backward compatibility only.
    # The current primary analysis no longer averages anchors across triplets.
    anchor_configs = {
        "uSP": {
            "pairs": sample_anchor_pairs,
            "gap_steps": int(spec.delay1_steps + spec.distractor_steps + spec.delay2_steps),
            "preceding_steps": int(spec.sample_steps),
        },
        "uDP": {
            "pairs": distractor_anchor_pairs,
            "gap_steps": int(spec.delay2_steps),
            "preceding_steps": int(spec.distractor_steps),
        },
    }

    per_pair_delta: dict[str, pd.DataFrame] = {}
    per_probe_vector: dict[str, dict[int, np.ndarray | None]] = {"uSP": {}, "uDP": {}}
    per_probe_status: dict[str, dict[int, str]] = {"uSP": {}, "uDP": {}}
    per_probe_norm: dict[str, dict[int, float]] = {"uSP": {}, "uDP": {}}
    metadata_by_probe: dict[int, dict[str, object]] = {}
    known_probe_ids = sorted(
        set(sample_anchor_pairs["probe_id"].astype(int).tolist() if not sample_anchor_pairs.empty else [])
        | set(distractor_anchor_pairs["probe_id"].astype(int).tolist() if not distractor_anchor_pairs.empty else [])
    )

    for probe_id in known_probe_ids:
        metadata_by_probe[int(probe_id)] = {"probe_id": int(probe_id)}

    for anchor_type, config in anchor_configs.items():
        pair_df = config["pairs"].copy()
        if pair_df.empty:
            per_pair_delta[anchor_type] = pd.DataFrame(
                columns=["probe_id", "preceding_id", "anchor_type", "delta_valid", "delta_norm"]
            )
            continue

        delta_rows: list[dict[str, object]] = []
        for batch_start in range(0, len(pair_df), int(batch_size)):
            batch = pair_df.iloc[batch_start : batch_start + int(batch_size)].copy().reset_index(drop=True)
            preceding_spikes, probe_spikes = _prepare_single_source_spike_batch(
                images=images,
                batch_df=batch,
                encoder=encoder,
                preceding_steps=int(config["preceding_steps"]),
                probe_steps=int(spec.probe_steps),
                device=device,
            )
            dynamic = run_single_source_preceding_item_task(
                net=net,
                preceding_spikes=preceding_spikes,
                probe_spikes=probe_spikes,
                gap_steps=int(config["gap_steps"]),
                stsp_mode="dynamic",
                readout_step=readout_step,
                phase_reset=spec.phase_reset,
            )
            static = run_single_source_preceding_item_task(
                net=net,
                preceding_spikes=preceding_spikes,
                probe_spikes=probe_spikes,
                gap_steps=int(config["gap_steps"]),
                stsp_mode="static_frozen",
                readout_step=readout_step,
                phase_reset=spec.phase_reset,
            )
            delta_v = compute_delta_v(dynamic["grouped_voltage"], static["grouped_voltage"])
            for batch_idx, batch_row in enumerate(batch.itertuples(index=False)):
                vec = np.asarray(delta_v[batch_idx], dtype=np.float64).reshape(-1)
                delta_norm = float(np.linalg.norm(vec))
                delta_rows.append(
                    {
                        "probe_id": int(batch_row.probe_id),
                        "probe_label": int(batch_row.probe_label),
                        "preceding_id": int(batch_row.preceding_id),
                        "preceding_label": int(batch_row.preceding_label),
                        "anchor_type": str(anchor_type),
                        "pair_key": str(batch_row.pair_key),
                        "delta_valid": int(np.isfinite(delta_norm) and delta_norm > float(eps)),
                        "delta_norm": delta_norm,
                        "delta_v": vec,
                    }
                )

        pair_delta_df = pd.DataFrame(delta_rows)
        per_pair_delta[anchor_type] = pair_delta_df
        for probe_id in sorted(pair_df["probe_id"].astype(int).unique().tolist()):
            probe_subset = pair_delta_df[pair_delta_df["probe_id"] == int(probe_id)]
            probe_label = int(pair_df[pair_df["probe_id"] == int(probe_id)]["probe_label"].iloc[0])
            meta = metadata_by_probe.setdefault(int(probe_id), {"probe_id": int(probe_id)})
            meta["probe_label"] = probe_label
            count_key = "n_sample_pairs_for_uSP" if anchor_type == "uSP" else "n_distractor_pairs_for_uDP"
            valid_count_key = "n_valid_sample_pairs_for_uSP" if anchor_type == "uSP" else "n_valid_distractor_pairs_for_uDP"
            status_key = "uSP_status" if anchor_type == "uSP" else "uDP_status"
            norm_key = "uSP_agg_norm" if anchor_type == "uSP" else "uDP_agg_norm"
            meta[count_key] = int(len(probe_subset))
            valid_subset = probe_subset[probe_subset["delta_valid"] > 0]
            meta[valid_count_key] = int(len(valid_subset))
            if probe_subset.empty:
                status = "skip_no_pairs"
                agg_norm = float("nan")
                vector = None
            elif valid_subset.empty:
                status = "skip_no_valid_pair_delta"
                agg_norm = float("nan")
                vector = None
            else:
                stacked = np.stack(valid_subset["delta_v"].tolist(), axis=0)
                aggregated = stacked.mean(axis=0)
                agg_norm = float(np.linalg.norm(aggregated))
                vector = safe_normalize(aggregated, eps=eps)
                status = "ok" if vector is not None else "skip_zero_norm"
            meta[status_key] = str(status)
            meta[norm_key] = float(agg_norm)
            per_probe_status[anchor_type][int(probe_id)] = str(status)
            per_probe_norm[anchor_type][int(probe_id)] = float(agg_norm)
            per_probe_vector[anchor_type][int(probe_id)] = None if vector is None else np.asarray(vector, dtype=np.float64)

    for probe_id, meta in metadata_by_probe.items():
        meta.setdefault("probe_label", -1)
        meta.setdefault("n_sample_pairs_for_uSP", 0)
        meta.setdefault("n_distractor_pairs_for_uDP", 0)
        meta.setdefault("n_valid_sample_pairs_for_uSP", 0)
        meta.setdefault("n_valid_distractor_pairs_for_uDP", 0)
        meta.setdefault("uSP_status", "skip_no_pairs")
        meta.setdefault("uDP_status", "skip_no_pairs")
        meta.setdefault("uSP_agg_norm", float("nan"))
        meta.setdefault("uDP_agg_norm", float("nan"))

    metadata_columns = [
        "probe_id",
        "probe_label",
        "n_sample_pairs_for_uSP",
        "n_distractor_pairs_for_uDP",
        "n_valid_sample_pairs_for_uSP",
        "n_valid_distractor_pairs_for_uDP",
        "uSP_status",
        "uDP_status",
        "uSP_agg_norm",
        "uDP_agg_norm",
    ]
    metadata_rows = [metadata_by_probe[int(probe_id)] for probe_id in sorted(metadata_by_probe.keys())]
    metadata_df = (
        pd.DataFrame(metadata_rows, columns=metadata_columns)
        if metadata_rows
        else pd.DataFrame(columns=metadata_columns)
    )
    return {
        "uSP_by_probe": per_probe_vector["uSP"],
        "uDP_by_probe": per_probe_vector["uDP"],
        "probe_anchor_metadata": metadata_df.sort_values(["probe_id"], kind="stable").reset_index(drop=True),
        "uSP_pair_delta": per_pair_delta["uSP"],
        "uDP_pair_delta": per_pair_delta["uDP"],
    }


def compute_triplet_single_source_anchor_vectors(
    *,
    net,
    images: torch.Tensor,
    encoder,
    device: torch.device,
    readout_step: int,
    spec: ExperimentSpec,
    triplet_anchor_df: pd.DataFrame,
    batch_size: int,
    eps: float = EPS,
) -> dict[str, object]:
    # Primary source-anchor analysis:
    # build personalized anchors for each concrete (sample, distractor, probe) triplet.
    # We intentionally use the same fixed delay2 gap for both uSP and uDP so the two
    # empirical source directions are directly comparable within that triplet.
    # This no longer estimates a probe-level average direction.
    if triplet_anchor_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "triplet_id",
                "probe_id",
                "sample_id",
                "distractor_id",
                "uSP_status",
                "uDP_status",
                "uSP_norm",
                "uDP_norm",
                "anchor_cosine",
            ]
        )
        return {"triplet_anchor_table": empty_df}

    def _run_anchor_pass(
        *,
        anchor_name: str,
        preceding_id_col: str,
        preceding_label_col: str,
        preceding_steps: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for batch_start in range(0, len(triplet_anchor_df), int(batch_size)):
            batch = triplet_anchor_df.iloc[batch_start : batch_start + int(batch_size)].copy().reset_index(drop=True)
            batch["preceding_id"] = batch[preceding_id_col].astype(int)
            batch["preceding_label"] = batch[preceding_label_col].astype(int)
            preceding_spikes, probe_spikes = _prepare_single_source_spike_batch(
                images=images,
                batch_df=batch[["preceding_id", "probe_id"]],
                encoder=encoder,
                preceding_steps=int(preceding_steps),
                probe_steps=int(spec.probe_steps),
                device=device,
            )
            dynamic = run_single_source_preceding_item_task(
                net=net,
                preceding_spikes=preceding_spikes,
                probe_spikes=probe_spikes,
                gap_steps=int(spec.delay2_steps),
                stsp_mode="dynamic",
                readout_step=readout_step,
                phase_reset=spec.phase_reset,
            )
            static = run_single_source_preceding_item_task(
                net=net,
                preceding_spikes=preceding_spikes,
                probe_spikes=probe_spikes,
                gap_steps=int(spec.delay2_steps),
                stsp_mode="static_frozen",
                readout_step=readout_step,
                phase_reset=spec.phase_reset,
            )
            delta_v = compute_delta_v(dynamic["grouped_voltage"], static["grouped_voltage"])
            for batch_idx, batch_row in enumerate(batch.itertuples(index=False)):
                raw = np.asarray(delta_v[batch_idx], dtype=np.float64).reshape(-1)
                norm = float(np.linalg.norm(raw))
                unit = safe_normalize(raw, eps=eps)
                status = "ok" if unit is not None else "skip_zero_norm"
                rows.append(
                    {
                        "triplet_id": int(batch_row.triplet_id),
                        "probe_id": int(batch_row.probe_id),
                        "sample_id": int(batch_row.sample_id),
                        "distractor_id": int(batch_row.distractor_id),
                        f"{anchor_name}_raw": raw,
                        f"{anchor_name}_unit": None if unit is None else np.asarray(unit, dtype=np.float64),
                        f"{anchor_name}_norm": float(norm),
                        f"{anchor_name}_status": str(status),
                    }
                )
        return pd.DataFrame(rows).sort_values(["triplet_id"], kind="stable").reset_index(drop=True)

    sp_df = _run_anchor_pass(
        anchor_name="uSP",
        preceding_id_col="sample_id",
        preceding_label_col="sample_label",
        preceding_steps=int(spec.sample_steps),
    )
    dp_df = _run_anchor_pass(
        anchor_name="uDP",
        preceding_id_col="distractor_id",
        preceding_label_col="distractor_label",
        preceding_steps=int(spec.distractor_steps),
    )
    merged = (
        triplet_anchor_df[["triplet_id", "probe_id", "sample_id", "distractor_id"]]
        .merge(sp_df, on=["triplet_id", "probe_id", "sample_id", "distractor_id"], how="left")
        .merge(dp_df, on=["triplet_id", "probe_id", "sample_id", "distractor_id"], how="left")
        .sort_values(["triplet_id"], kind="stable")
        .reset_index(drop=True)
    )
    anchor_cosines: list[float] = []
    for row in merged.itertuples(index=False):
        anchor_cosines.append(
            safe_cosine(row.uSP_unit, row.uDP_unit, eps=eps)
            if row.uSP_unit is not None and row.uDP_unit is not None
            else float("nan")
        )
    merged["anchor_cosine"] = np.asarray(anchor_cosines, dtype=np.float64)
    return {"triplet_anchor_table": merged}


def compute_triplet_source_anchor_metrics(
    delta_v_condition: np.ndarray,
    uSP_triplet_raw: np.ndarray | None,
    uSP_triplet_unit: np.ndarray | None,
    uDP_triplet_raw: np.ndarray | None,
    uDP_triplet_unit: np.ndarray | None,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    # Compare each overlap-condition DeltaV against that triplet's own personalized
    # source-anchor coordinate system rather than a probe-level averaged anchor.
    vec = np.asarray(delta_v_condition, dtype=np.float64).reshape(-1)
    proj_uSP_unit = float("nan") if uSP_triplet_unit is None else float(np.dot(vec, np.asarray(uSP_triplet_unit, dtype=np.float64).reshape(-1)))
    proj_uDP_unit = float("nan") if uDP_triplet_unit is None else float(np.dot(vec, np.asarray(uDP_triplet_unit, dtype=np.float64).reshape(-1)))
    cos_to_uSP = float("nan") if uSP_triplet_unit is None else safe_cosine(vec, uSP_triplet_unit, eps=eps)
    cos_to_uDP = float("nan") if uDP_triplet_unit is None else safe_cosine(vec, uDP_triplet_unit, eps=eps)
    proj_uSP_raw = float("nan") if uSP_triplet_raw is None else float(np.dot(vec, np.asarray(uSP_triplet_raw, dtype=np.float64).reshape(-1)))
    proj_uDP_raw = float("nan") if uDP_triplet_raw is None else float(np.dot(vec, np.asarray(uDP_triplet_raw, dtype=np.float64).reshape(-1)))
    if uSP_triplet_unit is None or uDP_triplet_unit is None:
        source_bias_raw = float("nan")
        source_bias_norm = float("nan")
    else:
        source_bias_raw = float(proj_uDP_unit - proj_uSP_unit)
        denom = abs(proj_uDP_unit) + abs(proj_uSP_unit) + float(eps)
        source_bias_norm = float(source_bias_raw / denom)
    return {
        "cos_to_uSP": float(cos_to_uSP),
        "cos_to_uDP": float(cos_to_uDP),
        "proj_to_uSP_unit": float(proj_uSP_unit),
        "proj_to_uDP_unit": float(proj_uDP_unit),
        "proj_to_uSP_raw": float(proj_uSP_raw),
        "proj_to_uDP_raw": float(proj_uDP_raw),
        "proj_uSP": float(proj_uSP_unit),
        "proj_uDP": float(proj_uDP_unit),
        "source_bias_raw": float(source_bias_raw),
        "source_bias_norm": float(source_bias_norm),
    }


def summarize_source_anchor_effects(df_source_records: pd.DataFrame) -> dict[str, object]:
    metric_columns = (
        "cos_to_uSP",
        "cos_to_uDP",
        "proj_to_uSP_unit",
        "proj_to_uDP_unit",
        "source_bias_raw",
        "source_bias_norm",
    )
    condition_rows: list[dict[str, object]] = []
    for condition_name in OVERLAP_CONDITION_ORDER:
        subset = df_source_records[df_source_records["condition"] == condition_name].copy()
        row: dict[str, object] = {
            "condition": str(condition_name),
            "n_triplets": int(subset["triplet_id"].nunique()) if not subset.empty else 0,
            "n_valid_uSP": int(np.isfinite(subset["cos_to_uSP"].to_numpy(dtype=np.float64)).sum()) if not subset.empty else 0,
            "n_valid_uDP": int(np.isfinite(subset["cos_to_uDP"].to_numpy(dtype=np.float64)).sum()) if not subset.empty else 0,
            "n_valid_source_bias": int(np.isfinite(subset["source_bias_norm"].to_numpy(dtype=np.float64)).sum()) if not subset.empty else 0,
        }
        for metric_name in metric_columns:
            values = subset[metric_name].to_numpy(dtype=np.float64) if not subset.empty else np.asarray([], dtype=np.float64)
            row[f"mean_{metric_name}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"sem_{metric_name}"] = _sem(values)
        condition_rows.append(row)
    return {
        "condition_summary": condition_rows,
        "triplet_anchor_diagnostic": {
            "n_triplets": int(df_source_records["triplet_id"].nunique()) if not df_source_records.empty else 0,
        },
    }


def assign_anchor_similarity_bins(
    triplet_anchor_df: pd.DataFrame,
    *,
    eps: float = EPS,
) -> dict[str, object]:
    # Supplementary mechanistic sensitivity analysis:
    # stratify triplets by tertiles of triplet-specific cos(uSP, uDP) to test
    # whether source-bias metrics become more discriminative when the two
    # personalized source anchors are more geometrically separable.
    df = triplet_anchor_df.copy()
    if df.empty:
        df["anchor_similarity_bin"] = pd.Series(dtype="object")
        return {
            "triplet_anchor_bins": df,
            "q33": float("nan"),
            "q67": float("nan"),
            "bin_counts": {},
            "invalid_count": 0,
            "warnings": ["no_triplets"],
        }

    valid_mask = (
        df["uSP_status"].astype(str).eq("ok")
        & df["uDP_status"].astype(str).eq("ok")
        & np.isfinite(df["anchor_cosine"].to_numpy(dtype=np.float64))
        & (df["uSP_norm"].to_numpy(dtype=np.float64) > float(eps))
        & (df["uDP_norm"].to_numpy(dtype=np.float64) > float(eps))
    )
    df["anchor_similarity_bin"] = "invalid"
    warnings: list[str] = []
    valid_values = df.loc[valid_mask, "anchor_cosine"].to_numpy(dtype=np.float64)
    if valid_values.size <= 0:
        warnings.append("no_valid_triplet_anchor_cosines")
        q33 = float("nan")
        q67 = float("nan")
    else:
        q33, q67 = np.quantile(valid_values, [1.0 / 3.0, 2.0 / 3.0])
        q33 = float(q33)
        q67 = float(q67)
        low_mask = valid_mask & (df["anchor_cosine"].to_numpy(dtype=np.float64) <= q33)
        mid_mask = valid_mask & (df["anchor_cosine"].to_numpy(dtype=np.float64) > q33) & (df["anchor_cosine"].to_numpy(dtype=np.float64) <= q67)
        high_mask = valid_mask & (df["anchor_cosine"].to_numpy(dtype=np.float64) > q67)
        df.loc[low_mask, "anchor_similarity_bin"] = "low"
        df.loc[mid_mask, "anchor_similarity_bin"] = "middle"
        df.loc[high_mask, "anchor_similarity_bin"] = "high"
    bin_counts = {
        bin_name: int((df["anchor_similarity_bin"] == bin_name).sum())
        for bin_name in ANCHOR_SIMILARITY_BIN_ORDER
    }
    invalid_count = int((df["anchor_similarity_bin"] == "invalid").sum())
    if any(count < 3 for count in bin_counts.values() if count > 0):
        warnings.append("at_least_one_anchor_similarity_bin_has_fewer_than_3_triplets")
    df["anchor_cosine_q33"] = float(q33)
    df["anchor_cosine_q67"] = float(q67)
    return {
        "triplet_anchor_bins": df,
        "q33": float(q33),
        "q67": float(q67),
        "bin_counts": bin_counts,
        "invalid_count": int(invalid_count),
        "warnings": warnings,
    }


def summarize_source_bias_by_anchor_bin(
    df_triplet_source_records: pd.DataFrame,
    triplet_anchor_df: pd.DataFrame,
) -> dict[str, object]:
    merge_cols = ["triplet_id", "anchor_cosine", "anchor_similarity_bin"]
    if {"anchor_cosine", "anchor_similarity_bin"}.issubset(df_triplet_source_records.columns):
        merged = df_triplet_source_records.copy()
    else:
        merged = df_triplet_source_records.merge(
            triplet_anchor_df[merge_cols].drop_duplicates(subset=["triplet_id"], keep="first"),
            on="triplet_id",
            how="left",
        )
    rows: list[dict[str, object]] = []
    for condition_name in OVERLAP_CONDITION_ORDER:
        for bin_name in ANCHOR_SIMILARITY_BIN_ORDER:
            subset = merged[
                (merged["condition"] == condition_name)
                & (merged["anchor_similarity_bin"] == bin_name)
            ].copy()
            values_bias_norm = subset["source_bias_norm"].to_numpy(dtype=np.float64)
            row = {
                "condition": str(condition_name),
                "anchor_similarity_bin": str(bin_name),
                "n_triplets": int(subset["triplet_id"].nunique()) if not subset.empty else 0,
                "mean_source_bias_norm": float(np.nanmean(values_bias_norm)) if np.isfinite(values_bias_norm).any() else float("nan"),
                "sem_source_bias_norm": _sem(values_bias_norm),
                "mean_source_bias_raw": float(np.nanmean(subset["source_bias_raw"].to_numpy(dtype=np.float64))) if not subset.empty and np.isfinite(subset["source_bias_raw"].to_numpy(dtype=np.float64)).any() else float("nan"),
                "sem_source_bias_raw": _sem(subset["source_bias_raw"].to_numpy(dtype=np.float64)) if not subset.empty else 0.0,
                "mean_cos_to_uSP": float(np.nanmean(subset["cos_to_uSP"].to_numpy(dtype=np.float64))) if not subset.empty and np.isfinite(subset["cos_to_uSP"].to_numpy(dtype=np.float64)).any() else float("nan"),
                "sem_cos_to_uSP": _sem(subset["cos_to_uSP"].to_numpy(dtype=np.float64)) if not subset.empty else 0.0,
                "mean_cos_to_uDP": float(np.nanmean(subset["cos_to_uDP"].to_numpy(dtype=np.float64))) if not subset.empty and np.isfinite(subset["cos_to_uDP"].to_numpy(dtype=np.float64)).any() else float("nan"),
                "sem_cos_to_uDP": _sem(subset["cos_to_uDP"].to_numpy(dtype=np.float64)) if not subset.empty else 0.0,
                "abs_mean_source_bias_norm": float(abs(np.nanmean(values_bias_norm))) if np.isfinite(values_bias_norm).any() else float("nan"),
                "mean_abs_source_bias_norm": float(np.nanmean(np.abs(values_bias_norm))) if np.isfinite(values_bias_norm).any() else float("nan"),
            }
            rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values(["condition", "anchor_similarity_bin"], kind="stable").reset_index(drop=True)
    return {
        "summary_df": summary_df,
        "merged_records": merged,
    }


def compute_anchor_bin_sensitivity_metrics(
    stratified_summary_df: pd.DataFrame,
) -> dict[str, object]:
    # This block tests whether source-bias metrics become more discriminative
    # when triplet-specific source anchors are more separable. It is explicitly
    # supplementary and does not replace the full-sample primary analysis.
    condition_metrics: dict[str, object] = {}
    primary_conditions = ("sample_remove_SPonly", "distractor_remove_DPonly")
    secondary_conditions = ("sample_remove_SDP", "distractor_remove_SDP")
    for condition_name in (*primary_conditions, *secondary_conditions):
        subset = stratified_summary_df[stratified_summary_df["condition"] == condition_name].copy()
        bin_to_value = {
            str(row.anchor_similarity_bin): float(row.abs_mean_source_bias_norm)
            for row in subset.itertuples(index=False)
        }
        ordered = [bin_to_value.get(bin_name, float("nan")) for bin_name in ANCHOR_SIMILARITY_BIN_ORDER]
        valid = np.asarray(ordered, dtype=np.float64)
        condition_metrics[str(condition_name)] = {
            "abs_mean_source_bias_norm_by_bin": {bin_name: float(bin_to_value.get(bin_name, float("nan"))) for bin_name in ANCHOR_SIMILARITY_BIN_ORDER},
            "high_minus_low_abs_mean_source_bias_norm": float(bin_to_value.get("high", float("nan")) - bin_to_value.get("low", float("nan"))) if np.isfinite(bin_to_value.get("high", float("nan"))) and np.isfinite(bin_to_value.get("low", float("nan"))) else float("nan"),
            "monotonic_non_decreasing_abs_mean_source_bias_norm": bool(np.all(np.diff(valid[np.isfinite(valid)]) >= -1e-12)) if np.isfinite(valid).sum() >= 2 else False,
            "analysis_role": "primary" if condition_name in primary_conditions else "secondary",
        }
    return {
        "primary_conditions": list(primary_conditions),
        "secondary_conditions": list(secondary_conditions),
        "condition_metrics": condition_metrics,
    }


def fit_quadratic_trend(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> dict[str, object]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 3 or np.unique(xx).size < 3:
        return {"status": "insufficient_points"}
    try:
        coef = np.polyfit(xx, yy, deg=2)
    except np.linalg.LinAlgError:
        return {"status": "fit_error"}
    a, b, c = [float(val) for val in coef.tolist()]
    vertex_x = float(-b / (2.0 * a)) if abs(a) > EPS else float("nan")
    x_fit = np.linspace(float(xx.min()), float(xx.max()), num=200)
    y_fit = np.polyval(coef, x_fit)
    return {
        "status": "ok",
        "quadratic_coef": a,
        "linear_coef": b,
        "intercept": c,
        "vertex_x": vertex_x,
        "vertex_within_range": bool(np.isfinite(vertex_x) and float(xx.min()) <= vertex_x <= float(xx.max())),
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


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


def summarize_delay_trend(df_delay_records: pd.DataFrame, metric_name: str) -> dict[str, object]:
    sub = df_delay_records.groupby("delay_ms", sort=True)[metric_name].mean().reset_index()
    x = sub["delay_ms"].to_numpy(dtype=np.float64)
    y = sub[metric_name].to_numpy(dtype=np.float64)
    quadratic = fit_quadratic_trend(x, y)
    spearman = compute_spearman_summary(x, y)
    peak_idx = int(np.argmax(y)) if y.size > 0 else -1
    peak_delay_ms = float(x[peak_idx]) if peak_idx >= 0 else None
    peak_value = float(y[peak_idx]) if peak_idx >= 0 else None
    if peak_idx >= 0 and (len(x) - peak_idx) >= 2:
        x_tail = x[peak_idx:]
        y_tail = y[peak_idx:]
        tail_summary = {
            "status": "ok",
            "n_points": int(len(x_tail)),
            "slope": float(np.polyfit(x_tail, y_tail, deg=1)[0]),
            "spearman": compute_spearman_summary(x_tail, y_tail),
        }
    else:
        tail_summary = {"status": "insufficient_points"}
    supports_rise_then_fall = bool(
        quadratic.get("status") == "ok"
        and float(quadratic["quadratic_coef"]) < 0.0
        and bool(quadratic["vertex_within_range"])
    )
    return {
        "metric": metric_name,
        "peak_delay_ms": peak_delay_ms,
        "peak_value": peak_value,
        "quadratic_fit": quadratic,
        "spearman": spearman,
        "supports_rise_then_fall": supports_rise_then_fall,
        "tail_summary": tail_summary,
    }


def _weighted_feature_columns() -> tuple[str, ...]:
    return ("X_SP", "X_DP", "X_SDP", "area_SPonly", "area_DPonly", "area_SDP", "sample_SDP_area", "distractor_SDP_area")


def build_condition_feature_table(
    df_triplets: pd.DataFrame,
    *,
    conditions: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for triplet_row in df_triplets.itertuples(index=False):
        triplet_dict = dict(triplet_row._asdict())
        x_sp = float(triplet_dict.get("X_SP", 0.0))
        x_dp = float(triplet_dict.get("X_DP", 0.0))
        x_sdp = float(triplet_dict.get("X_SDP", 0.0))
        base = {
            "abs_X_SP": x_sp,
            "abs_X_DP": x_dp,
            "abs_X_SDP": x_sdp,
            "abs_X_SDP_sample": x_sdp,
            "abs_X_SDP_distr": x_sdp,
        }
        for condition_name in conditions:
            values = dict(base)
            if condition_name == "sample_remove_SPonly":
                values["abs_X_SP"] = 0.0
            elif condition_name == "distractor_remove_DPonly":
                values["abs_X_DP"] = 0.0
            elif condition_name == "sample_remove_SDP":
                values["abs_X_SDP_sample"] = 0.0
            elif condition_name == "distractor_remove_SDP":
                values["abs_X_SDP_distr"] = 0.0
            elif condition_name == "both_remove_SDP":
                values["abs_X_SDP_sample"] = 0.0
                values["abs_X_SDP_distr"] = 0.0
            values["delta_X_SP"] = float(values["abs_X_SP"] - base["abs_X_SP"])
            values["delta_X_DP"] = float(values["abs_X_DP"] - base["abs_X_DP"])
            values["delta_X_SDP"] = float(values["abs_X_SDP"] - base["abs_X_SDP"])
            values["delta_X_SDP_sample"] = float(values["abs_X_SDP_sample"] - base["abs_X_SDP_sample"])
            values["delta_X_SDP_distr"] = float(values["abs_X_SDP_distr"] - base["abs_X_SDP_distr"])
            rows.append(
                {
                    "triplet_id": int(triplet_dict["triplet_id"]),
                    "condition": str(condition_name),
                    **{column: triplet_dict[column] for column in triplet_dict.keys() if column in _weighted_feature_columns()},
                    **values,
                }
            )
    return pd.DataFrame(rows)


def compute_overlap_bias_features(
    df_triplets: pd.DataFrame,
    *,
    conditions: Sequence[str],
    eps: float = EPS,
) -> pd.DataFrame:
    df = build_condition_feature_table(df_triplets, conditions=conditions)
    if df.empty:
        return df
    df = df.copy()
    df["abs_X_SDP_eff"] = 0.5 * (df["abs_X_SDP_sample"] + df["abs_X_SDP_distr"])
    denominator = df["abs_X_SP"] + df["abs_X_DP"] + df["abs_X_SDP_eff"] + float(eps)
    df["overlap_bias_1"] = df["abs_X_DP"] - df["abs_X_SP"]
    df["overlap_bias_2"] = df["abs_X_SDP_sample"] - df["abs_X_SDP_distr"]
    df["overlap_bias_3"] = (df["abs_X_DP"] - df["abs_X_SP"]) / denominator
    return df


def compute_overlap_bias_correlations(
    df_overlap_records: pd.DataFrame,
    *,
    bias_columns: Sequence[str] = ("overlap_bias_1", "overlap_bias_2", "overlap_bias_3"),
    target_columns: Sequence[str] = ("phi_deg", "cos_phi"),
) -> dict[str, object]:
    pooled: dict[str, object] = {}
    within_condition: dict[str, object] = {}
    for bias_col in bias_columns:
        pooled[str(bias_col)] = {}
        for target_col in target_columns:
            pooled[str(bias_col)][str(target_col)] = compute_spearman_summary(
                df_overlap_records[bias_col].to_numpy(dtype=np.float64),
                df_overlap_records[target_col].to_numpy(dtype=np.float64),
            )
    for condition_name, subset in df_overlap_records.groupby("condition", sort=False):
        within_condition[str(condition_name)] = {}
        for bias_col in bias_columns:
            within_condition[str(condition_name)][str(bias_col)] = {}
            for target_col in target_columns:
                within_condition[str(condition_name)][str(bias_col)][str(target_col)] = compute_spearman_summary(
                    subset[bias_col].to_numpy(dtype=np.float64),
                    subset[target_col].to_numpy(dtype=np.float64),
                )
    distribution_rows: list[dict[str, object]] = []
    for condition_name, subset in df_overlap_records.groupby("condition", sort=False):
        distribution_rows.append(
            {
                "condition": str(condition_name),
                "n": int(len(subset)),
                "mean_phi_deg": float(subset["phi_deg"].mean(skipna=True)),
                "sem_phi_deg": _sem(subset["phi_deg"].to_numpy(dtype=np.float64)),
                "mean_cos_phi": float(subset["cos_phi"].mean(skipna=True)),
                "sem_cos_phi": _sem(subset["cos_phi"].to_numpy(dtype=np.float64)),
                "mean_M_condition": float(subset["M_condition"].mean(skipna=True)),
            }
        )
    return {
        "pooled_correlations_exploratory": pooled,
        "within_condition_correlations_primary": within_condition,
        "condition_distribution_summary": distribution_rows,
    }


def collect_delay_sweep_outputs(
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    delay_ms_values: Sequence[float],
    spec: ExperimentSpec,
    readout_step: int,
) -> dict[str, np.ndarray]:
    dynamic_grouped: list[np.ndarray] = []
    static_grouped: list[np.ndarray] = []
    for delay_ms in delay_ms_values:
        delay2_steps = int(round((float(delay_ms) * ms) / spec.dt))
        dynamic = run_overlap_perturbed_distractor_task(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            delay1_steps=spec.delay1_steps,
            delay2_steps=delay2_steps,
            stsp_mode="dynamic",
            readout_step=readout_step,
            phase_reset=spec.phase_reset,
        )
        static = run_overlap_perturbed_distractor_task(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            delay1_steps=spec.delay1_steps,
            delay2_steps=delay2_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
            phase_reset=spec.phase_reset,
        )
        dynamic_grouped.append(np.asarray(dynamic.grouped_voltage, dtype=np.float64))
        static_grouped.append(np.asarray(static.grouped_voltage, dtype=np.float64))
    dynamic_arr = np.stack(dynamic_grouped, axis=1)
    static_arr = np.stack(static_grouped, axis=1)
    return {
        "delay_ms": np.asarray(delay_ms_values, dtype=np.float64),
        "dynamic_grouped_voltage": dynamic_arr,
        "static_grouped_voltage": static_arr,
        "delta_v": compute_delta_v(dynamic_arr, static_arr),
    }


def collect_overlap_condition_outputs(
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    batch_masks: Sequence[TripletMaskBundle],
    direction_delay_ms: float,
    spec: ExperimentSpec,
    readout_step: int,
    include_conditions: Sequence[str] = OVERLAP_CONDITION_ORDER,
) -> dict[str, np.ndarray]:
    condition_specs = _condition_spec_table()
    delay2_steps = int(round((float(direction_delay_ms) * ms) / spec.dt))
    static = run_overlap_perturbed_distractor_task(
        net=net,
        sample_spikes=sample_spikes,
        distractor_spikes=distractor_spikes,
        probe_spikes=probe_spikes,
        delay1_steps=spec.delay1_steps,
        delay2_steps=delay2_steps,
        stsp_mode="static_frozen",
        readout_step=readout_step,
        phase_reset=spec.phase_reset,
    )
    condition_grouped: list[np.ndarray] = []
    for condition_name in include_conditions:
        spec_row = condition_specs[str(condition_name)]
        sample_mask = _build_condition_mask_batch(batch_masks, spec_row.sample_mask_key)
        distractor_mask = _build_condition_mask_batch(batch_masks, spec_row.distractor_mask_key)
        rollout = run_overlap_perturbed_distractor_task(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            delay1_steps=spec.delay1_steps,
            delay2_steps=delay2_steps,
            stsp_mode=spec_row.stsp_mode,
            readout_step=readout_step,
            sample_input_mask=None if sample_mask is None else sample_mask.to(device=sample_spikes.device),
            distractor_input_mask=None if distractor_mask is None else distractor_mask.to(device=sample_spikes.device),
            phase_reset=spec.phase_reset,
        )
        condition_grouped.append(np.asarray(rollout.grouped_voltage, dtype=np.float64))
    condition_arr = np.stack(condition_grouped, axis=1)
    static_arr = np.asarray(static.grouped_voltage, dtype=np.float64)
    static_repeated = np.repeat(static_arr[:, None, :], repeats=condition_arr.shape[1], axis=1)
    return {
        "condition_name": np.asarray(include_conditions),
        "condition_grouped_voltage": condition_arr,
        "static_grouped_voltage": static_arr,
        "delta_v": compute_delta_v(condition_arr, static_repeated),
    }


def _select_case_triplets(delay_records: pd.DataFrame, overlap_records: pd.DataFrame, *, count: int) -> list[int]:
    if int(count) <= 0 or delay_records.empty or overlap_records.empty:
        return []
    delay_summary = delay_records.groupby("triplet_id", sort=False).agg(mean_cos_theta=("cos_theta", "mean"), mean_M=("M", "mean")).reset_index()
    overlap_summary = overlap_records.groupby("triplet_id", sort=False).agg(mean_phi_deg=("phi_deg", "mean"), mean_M_condition=("M_condition", "mean")).reset_index()
    merged = delay_summary.merge(overlap_summary, on="triplet_id", how="inner")
    if merged.empty:
        return []
    merged = merged.sort_values(
        ["mean_cos_theta", "mean_M", "mean_phi_deg", "mean_M_condition", "triplet_id"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    take = min(int(count), len(merged))
    positions = np.linspace(0, len(merged) - 1, num=take).astype(np.int64)
    return merged.iloc[sorted(dict.fromkeys(int(pos) for pos in positions.tolist()))]["triplet_id"].astype(int).tolist()


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
        return np.zeros_like(arr, dtype=np.float64)
    return arr / max_abs


def plot_mechanism_schematic() -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_MEDIUM)
    origin = np.zeros(2, dtype=np.float64)
    for vec, label, color in (
        (np.asarray([1.8, 0.8]), "short delay", "#72B7B2"),
        (np.asarray([3.0, 1.2]), "mid delay", "#4C78A8"),
        (np.asarray([2.2, 0.9]), "long delay", "#F58518"),
    ):
        ax.arrow(0.0, 0.0, vec[0], vec[1], color=color, width=0.02, head_width=0.12, length_includes_head=True, alpha=ALPHA_BAR)
        ax.text(vec[0] + 0.08, vec[1] + 0.04, label, color=color)
    for vec, label, color in (
        (np.asarray([2.5, 1.1]), "full", "#4C78A8"),
        (np.asarray([1.2, 2.4]), "overlap rotates", "#B279A2"),
        (np.asarray([2.0, -1.4]), "overlap rotates", "#9D755D"),
    ):
        ax.arrow(origin[0], origin[1], vec[0], vec[1], color=color, width=0.01, head_width=0.11, linestyle="--", length_includes_head=True, alpha=ALPHA_GUIDE)
        ax.text(vec[0] + 0.06, vec[1] - 0.16, label, color=color)
    ax.scatter([0.0], [0.0], s=48, color="#444444")
    ax.text(0.04, 0.08, "static / origin", fontsize=11)
    ax.axhline(0.0, color="#BBBBBB", linewidth=1.0)
    ax.axvline(0.0, color="#BBBBBB", linewidth=1.0)
    ax.set_title("Mechanistic decomposition in centered grouped-voltage space")
    ax.set_xlabel("Direction axis 1")
    ax.set_ylabel("Direction axis 2")
    ax.text(-0.1, -2.1, "Delay changes vector length; overlap rotates vector direction.", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=ALPHA_FILL)
    fig.tight_layout()
    return fig


def plot_delay_block(df_delay_records: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL, sharex=True)
    panel_specs = (
        ("A", "Effective strength A(delay)", "#4C78A8"),
        ("M", "Total magnitude M(delay)", "#F58518"),
        ("cos_theta", "Direction stability cos(theta)", "#54A24B"),
    )
    for ax, (metric_name, ylabel, color) in zip(axes, panel_specs):
        summary = df_delay_records.groupby("delay_ms", sort=True)[metric_name].mean().reset_index()
        summary["sem"] = [
            _sem(df_delay_records[df_delay_records["delay_ms"] == delay_ms][metric_name].to_numpy(dtype=np.float64))
            for delay_ms in summary["delay_ms"].to_numpy(dtype=np.float64)
        ]
        x = summary["delay_ms"].to_numpy(dtype=np.float64)
        y = summary[metric_name].to_numpy(dtype=np.float64)
        sem = summary["sem"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=color, linewidth=LINE_WIDTH_PRIMARY, marker=MARKER_CIRCLE)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=ALPHA_FILL)
        fit = fit_quadratic_trend(x, y)
        if fit.get("status") == "ok":
            ax.plot(fit["x_fit"], fit["y_fit"], color="#333333", linestyle="--", linewidth=1.4)
        ax.set_xlabel("Distractor -> probe delay (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def plot_overlap_block(df_overlap_records: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    metrics = (("phi_deg", "Rotation angle phi (deg)"), ("cos_phi", "Direction cosine to full_dynamic"))
    positions = np.arange(len(OVERLAP_CONDITION_ORDER), dtype=np.float64)
    for ax, (metric_name, ylabel) in zip(axes, metrics):
        box_data = []
        for condition_name in OVERLAP_CONDITION_ORDER:
            subset = df_overlap_records[df_overlap_records["condition"] == condition_name][metric_name].to_numpy(dtype=np.float64)
            box_data.append(subset[np.isfinite(subset)])
        ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True, boxprops={"facecolor": "#EAEAEA", "edgecolor": "#555555"})
        for pos, condition_name in zip(positions, OVERLAP_CONDITION_ORDER):
            subset = df_overlap_records[df_overlap_records["condition"] == condition_name][metric_name].to_numpy(dtype=np.float64)
            subset = subset[np.isfinite(subset)]
            if subset.size <= 0:
                continue
            jitter = np.linspace(-0.12, 0.12, num=subset.size) if subset.size > 1 else np.asarray([0.0], dtype=np.float64)
            ax.scatter(pos + jitter, subset, s=18, alpha=ALPHA_SCATTER, color=CONDITION_COLORS[condition_name])
        ax.set_xticks(positions)
        ax.set_xticklabels(OVERLAP_CONDITION_ORDER, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_triplet_anchor_similarity(df_triplet_anchor_vectors: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    values = (
        df_triplet_anchor_vectors["anchor_cosine"].to_numpy(dtype=np.float64)
        if not df_triplet_anchor_vectors.empty and "anchor_cosine" in df_triplet_anchor_vectors.columns
        else np.asarray([], dtype=np.float64)
    )
    values = values[np.isfinite(values)]
    if values.size <= 0:
        ax.text(0.5, 0.5, "No valid triplet anchor cosines", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    bins = min(20, max(5, values.size))
    ax.hist(values, bins=bins, color=CONDITION_COLORS["full_dynamic"], alpha=ALPHA_BAR, edgecolor="white")
    ax.axvline(float(np.mean(values)), color=CONDITION_COLORS["sample_remove_SPonly"], linewidth=LINE_WIDTH_PRIMARY, linestyle="--", label=f"mean={float(np.mean(values)):.2f}")
    ax.set_xlabel("cos(uSP_triplet_unit, uDP_triplet_unit)")
    ax.set_ylabel("Triplet count")
    ax.set_title("Triplet-level source-anchor similarity")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def plot_source_anchor_block(df_source_records: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL)
    positions = np.arange(len(OVERLAP_CONDITION_ORDER), dtype=np.float64)

    cos_summary = summarize_source_anchor_effects(df_source_records)["condition_summary"]
    cos_uSP = np.asarray([row["mean_cos_to_uSP"] for row in cos_summary], dtype=np.float64)
    cos_uDP = np.asarray([row["mean_cos_to_uDP"] for row in cos_summary], dtype=np.float64)
    cos_uSP_sem = np.asarray([row["sem_cos_to_uSP"] for row in cos_summary], dtype=np.float64)
    cos_uDP_sem = np.asarray([row["sem_cos_to_uDP"] for row in cos_summary], dtype=np.float64)
    width = 0.36
    axes[0].bar(positions - width / 2.0, cos_uSP, width=width, yerr=cos_uSP_sem, color=CONDITION_COLORS["full_dynamic"], alpha=ALPHA_BAR, label="uSP")
    axes[0].bar(positions + width / 2.0, cos_uDP, width=width, yerr=cos_uDP_sem, color=CONDITION_COLORS["distractor_remove_DPonly"], alpha=ALPHA_BAR, label="uDP")
    axes[0].axhline(0.0, color="#666666", linewidth=LINE_WIDTH_REFERENCE)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(OVERLAP_CONDITION_ORDER, rotation=25, ha="right")
    axes[0].set_ylabel("Mean cosine to source anchor")
    axes[0].set_title("Dual-anchor alignment by condition (triplet-specific anchors)")
    axes[0].grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(axes[0])

    source_bias = np.asarray([row["mean_source_bias_norm"] for row in cos_summary], dtype=np.float64)
    source_bias_sem = np.asarray([row["sem_source_bias_norm"] for row in cos_summary], dtype=np.float64)
    colors = [CONDITION_COLORS[condition_name] for condition_name in OVERLAP_CONDITION_ORDER]
    axes[1].bar(positions, source_bias, yerr=source_bias_sem, color=colors, alpha=0.88)
    axes[1].axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(OVERLAP_CONDITION_ORDER, rotation=25, ha="right")
    axes[1].set_ylabel("Mean normalized source bias")
    axes[1].set_title("Positive = distractor-source, negative = sample-source (triplet-specific anchors)")
    axes[1].grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_anchor_similarity_bin_counts(triplet_anchor_bin_df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_MEDIUM)
    counts = [
        int((triplet_anchor_bin_df["anchor_similarity_bin"] == bin_name).sum()) if not triplet_anchor_bin_df.empty else 0
        for bin_name in ANCHOR_SIMILARITY_BIN_ORDER
    ]
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    positions = np.arange(len(ANCHOR_SIMILARITY_BIN_ORDER), dtype=np.float64)
    ax.bar(positions, counts, color=colors, alpha=ALPHA_BAR)
    for pos, value in zip(positions, counts):
        ax.text(pos, value + 0.2, str(int(value)), ha="center", va="bottom", fontsize=10)
    invalid_count = int((triplet_anchor_bin_df["anchor_similarity_bin"] == "invalid").sum()) if not triplet_anchor_bin_df.empty else 0
    ax.set_xticks(positions)
    ax.set_xticklabels(ANCHOR_SIMILARITY_BIN_ORDER)
    ax.set_ylabel("Triplet count")
    ax.set_title(f"Anchor-similarity tertile counts (invalid={invalid_count})")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def plot_source_bias_by_anchor_bin(stratified_summary_df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_MEDIUM_TALL)
    positions = np.arange(len(ANCHOR_SIMILARITY_BIN_ORDER), dtype=np.float64)
    conditions = ("sample_remove_SPonly", "distractor_remove_DPonly", "sample_remove_SDP", "distractor_remove_SDP")
    for condition_name in conditions:
        subset = stratified_summary_df[stratified_summary_df["condition"] == condition_name].copy()
        y = np.asarray(
            [
                subset[subset["anchor_similarity_bin"] == bin_name]["mean_source_bias_norm"].iloc[0]
                if not subset[subset["anchor_similarity_bin"] == bin_name].empty
                else float("nan")
                for bin_name in ANCHOR_SIMILARITY_BIN_ORDER
            ],
            dtype=np.float64,
        )
        sem = np.asarray(
            [
                subset[subset["anchor_similarity_bin"] == bin_name]["sem_source_bias_norm"].iloc[0]
                if not subset[subset["anchor_similarity_bin"] == bin_name].empty
                else 0.0
                for bin_name in ANCHOR_SIMILARITY_BIN_ORDER
            ],
            dtype=np.float64,
        )
        ax.errorbar(
            positions,
            y,
            yerr=sem,
            marker=MARKER_CIRCLE,
            linewidth=LINE_WIDTH_PRIMARY,
            capsize=3.0,
            color=CONDITION_COLORS[condition_name],
            label=condition_name,
        )
    ax.axhline(0.0, color="#333333", linewidth=LINE_WIDTH_REFERENCE)
    ax.set_xticks(positions)
    ax.set_xticklabels(ANCHOR_SIMILARITY_BIN_ORDER)
    ax.set_ylabel("Mean source_bias_norm")
    ax.set_title("Triplet-specific anchor cosine stratification\nSupplementary sensitivity analysis, not a main-result replacement")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(ax, compact=True)
    fig.tight_layout()
    return fig


def plot_abs_source_bias_by_anchor_bin(stratified_summary_df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_MEDIUM_TALL)
    positions = np.arange(len(ANCHOR_SIMILARITY_BIN_ORDER), dtype=np.float64)
    conditions = ("sample_remove_SPonly", "distractor_remove_DPonly")
    for condition_name in conditions:
        subset = stratified_summary_df[stratified_summary_df["condition"] == condition_name].copy()
        y = np.asarray(
            [
                subset[subset["anchor_similarity_bin"] == bin_name]["abs_mean_source_bias_norm"].iloc[0]
                if not subset[subset["anchor_similarity_bin"] == bin_name].empty
                else float("nan")
                for bin_name in ANCHOR_SIMILARITY_BIN_ORDER
            ],
            dtype=np.float64,
        )
        ax.plot(
            positions,
            y,
            marker=MARKER_CIRCLE,
            linewidth=LINE_WIDTH_PRIMARY,
            color=CONDITION_COLORS[condition_name],
            label=condition_name,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(ANCHOR_SIMILARITY_BIN_ORDER)
    ax.set_ylabel("abs(mean source_bias_norm)")
    ax.set_title("Absolute source-bias magnitude by anchor-similarity bin")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def _summarize_within_condition_rho(correlation_summary: Mapping[str, object], bias_col: str, target_col: str) -> str:
    values: list[float] = []
    for condition_payload in correlation_summary.values():
        payload = condition_payload.get(bias_col, {}).get(target_col, {})
        if payload.get("status") == "ok" and payload.get("rho") is not None:
            values.append(float(payload["rho"]))
    if not values:
        return "within-condition rho unavailable"
    return f"within-condition median rho={float(np.median(values)):.2f}"


def plot_overlap_bias_correlation(df_overlap_records: pd.DataFrame, correlation_summary: Mapping[str, object]) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL_WIDE, sharey=True)
    for ax, bias_col in zip(axes, ("overlap_bias_1", "overlap_bias_2", "overlap_bias_3")):
        for condition_name in OVERLAP_CONDITION_ORDER:
            subset = df_overlap_records[df_overlap_records["condition"] == condition_name]
            if subset.empty:
                continue
            ax.scatter(
                subset[bias_col].to_numpy(dtype=np.float64),
                subset["phi_deg"].to_numpy(dtype=np.float64),
                s=24,
                alpha=ALPHA_SCATTER,
                color=CONDITION_COLORS[condition_name],
                label=condition_name,
            )
        pooled = correlation_summary["pooled_correlations_exploratory"][bias_col]["phi_deg"]
        pooled_text = "pooled rho unavailable"
        if pooled.get("status") == "ok":
            pooled_text = f"pooled rho={float(pooled['rho']):.2f}, p={float(pooled['p_value']):.3g}"
        within_text = _summarize_within_condition_rho(
            correlation_summary["within_condition_correlations_primary"],
            bias_col=bias_col,
            target_col="phi_deg",
        )
        ax.set_title(f"{bias_col}\n{pooled_text}\n{within_text}")
        ax.set_xlabel(bias_col)
        ax.grid(alpha=GRID_ALPHA)
    axes[0].set_ylabel("phi_deg")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    apply_standard_legend(axes[0], handles=list(dedup.values()), labels=list(dedup.keys()), compact=True)
    fig.tight_layout()
    return fig


def plot_representative_vector_examples(
    *,
    case_triplets: Sequence[int],
    delay_ms_values: Sequence[float],
    delay_vector_map: Mapping[int, Mapping[float, np.ndarray]],
    direction_vector_map: Mapping[int, Mapping[str, np.ndarray]],
) -> plt.Figure:
    apply_publication_style()
    n_cases = max(1, len(case_triplets))
    fig, axes = plt.subplots(n_cases, 2, figsize=case_grid_figsize(n_cases, width=12.4, row_height=4.2), squeeze=False)
    if not case_triplets:
        axes[0, 0].text(0.5, 0.5, "No representative triplets available", ha="center", va="center")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
        fig.tight_layout()
        return fig
    delay_norm = Normalize(vmin=float(min(delay_ms_values)), vmax=float(max(delay_ms_values)))
    delay_mapper = ScalarMappable(norm=delay_norm, cmap=CMAP_OVERLAP)
    for row_idx, triplet_id in enumerate(case_triplets):
        ax_delay = axes[row_idx, 0]
        ax_overlap = axes[row_idx, 1]
        delay_vectors = [np.asarray(delay_vector_map[int(triplet_id)][float(delay_ms)], dtype=np.float64) for delay_ms in delay_ms_values]
        delay_proj = _normalize_projection_for_plot(_project_vectors_to_2d(delay_vectors))
        for idx, delay_ms in enumerate(delay_ms_values):
            color = delay_mapper.to_rgba(float(delay_ms))
            ax_delay.scatter(delay_proj[idx, 0], delay_proj[idx, 1], s=38, color=color)
            ax_delay.text(delay_proj[idx, 0] + 0.03, delay_proj[idx, 1] + 0.03, f"{int(delay_ms)}", fontsize=8, clip_on=True)
        ax_delay.plot(delay_proj[:, 0], delay_proj[:, 1], color="#666666", linewidth=1.1, alpha=ALPHA_GUIDE)
        ax_delay.scatter([0.0], [0.0], color="#333333", s=36)
        ax_delay.set_title(f"Triplet {int(triplet_id)}: delay sweep")
        ax_delay.set_xlabel("PC1")
        ax_delay.set_ylabel("PC2")
        ax_delay.set_xlim(-1.25, 1.25)
        ax_delay.set_ylim(-1.25, 1.25)
        ax_delay.grid(alpha=GRID_ALPHA)
        overlap_vectors = [np.asarray(direction_vector_map[int(triplet_id)][condition_name], dtype=np.float64) for condition_name in OVERLAP_CONDITION_ORDER]
        overlap_proj = _normalize_projection_for_plot(_project_vectors_to_2d(overlap_vectors))
        for idx, condition_name in enumerate(OVERLAP_CONDITION_ORDER):
            ax_overlap.scatter(overlap_proj[idx, 0], overlap_proj[idx, 1], s=42, color=CONDITION_COLORS[condition_name])
            ax_overlap.text(overlap_proj[idx, 0] + 0.03, overlap_proj[idx, 1] + 0.03, condition_name, fontsize=8, clip_on=True)
        ax_overlap.scatter([0.0], [0.0], color="#333333", s=36)
        ax_overlap.set_title(f"Triplet {int(triplet_id)}: overlap rotation")
        ax_overlap.set_xlabel("PC1")
        ax_overlap.set_ylabel("PC2")
        ax_overlap.set_xlim(-1.25, 1.25)
        ax_overlap.set_ylim(-1.25, 1.25)
        ax_overlap.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distractor strength-direction decomposition experiment.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay1-ms", type=float, default=DEFAULT_DELAY1_MS)
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--distractor-ms", type=float, default=DEFAULT_DISTRACTOR_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    parser.add_argument("--delay-sweep-ms", type=float, nargs="+", default=list(DEFAULT_DELAY_SWEEP_MS))
    parser.add_argument("--direction-delay-ms", type=float, default=DEFAULT_DIRECTION_DELAY_MS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--use-dilated-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
    parser.add_argument("--num-control-candidates", type=int, default=DEFAULT_NUM_CONTROL_CANDIDATES)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _validate_positive("--batch-size", int(args.batch_size))
    _validate_positive("--max-probes", int(args.max_probes))
    _validate_positive("--samples-per-probe", int(args.samples_per_probe))
    _validate_positive("--max-triplets", int(args.max_triplets))
    _validate_positive("--num-sim-bins", int(args.num_sim_bins))
    _validate_positive("--save-case-count", int(args.save_case_count), allow_zero=True)
    _validate_positive("--num-control-candidates", int(args.num_control_candidates))
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--delay1-ms", float(args.delay1_ms), allow_zero=True)
    _validate_positive("--distractor-ms", float(args.distractor_ms))
    _validate_positive("--probe-ms", float(args.probe_ms))
    _validate_positive("--direction-delay-ms", float(args.direction_delay_ms), allow_zero=True)
    delay_ms_values = _sanitize_delay_sweep(args.delay_sweep_ms)

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay1_ms=float(args.delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(args.direction_delay_ms),
        probe_ms=float(args.probe_ms),
    )
    if spec.sample_steps <= 0 or spec.distractor_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/distractor/probe duration must resolve to positive steps.")

    layout = prepare_result_layout(args.output_dir)
    result_root = layout.root
    output_dir = layout.data_dir
    figures_dir = layout.figure_dir
    logs_dir = layout.log_dir

    dataset = _load_dataset(dataset_root=args.dataset_root, split=args.split)
    images, labels, flat_normalized = build_dataset_arrays(dataset)
    num_classes = int(len(np.unique(labels)))
    class_index = build_class_index(dataset, num_classes=num_classes)
    max_duration_ms = max(float(args.sample_ms), float(args.distractor_ms), float(args.probe_ms))
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=spec.dt,
        max_duration_ms=max_duration_ms,
    )
    readout_step = resolve_readout_step(
        readout_mode="decision_offset",
        trace_steps=int(spec.probe_steps),
        decision_offset=int(getattr(net.layer3, "decision_time_offset", 0)),
        explicit_step=None,
    )

    df_triplets = build_triplet_specs(
        images=images,
        labels=labels,
        flat_normalized=flat_normalized,
        class_index=class_index,
        max_probes=int(args.max_probes),
        samples_per_probe=int(args.samples_per_probe),
        num_bins=int(args.num_sim_bins),
        max_triplets=int(args.max_triplets),
        seed=int(args.seed),
    )

    mask_records: list[TripletMaskBundle] = []
    for triplet_row in df_triplets.itertuples(index=False):
        triplet_id = int(triplet_row.triplet_id)
        mask_records.append(
            build_probe_relevant_masks_for_triplet(
                sample_image=images[int(triplet_row.sample_id)],
                distractor_image=images[int(triplet_row.distractor_id)],
                probe_image=images[int(triplet_row.probe_id)],
                foreground_threshold=float(args.foreground_threshold),
                use_dilated_overlap=bool(args.use_dilated_overlap),
                dilation_radius=int(args.dilation_radius),
                seed=mix_seed(int(args.seed), triplet_id, int(triplet_row.sample_id), int(triplet_row.distractor_id), int(triplet_row.probe_id)),
                num_control_candidates=int(args.num_control_candidates),
            )
        )
    df_triplets = _augment_triplet_specs_with_mask_metadata(df_triplets, mask_records, spec=spec, tau_ms=float(DEFAULT_TAU_MS))
    condition_feature_table = compute_overlap_bias_features(df_triplets, conditions=OVERLAP_CONDITION_ORDER)
    triplet_anchor_specs = build_triplet_source_anchor_specs(df_triplets)
    triplet_anchor_outputs = compute_triplet_single_source_anchor_vectors(
        net=net,
        images=images,
        encoder=encoder,
        device=device,
        readout_step=readout_step,
        spec=spec,
        triplet_anchor_df=triplet_anchor_specs,
        batch_size=int(args.batch_size),
    )
    df_triplet_anchor_vectors = triplet_anchor_outputs["triplet_anchor_table"]
    triplet_anchor_lookup = {
        int(row.triplet_id): dict(row._asdict())
        for row in df_triplet_anchor_vectors.itertuples(index=False)
    }

    delay_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    delay_triplet_ids: list[int] = []
    direction_triplet_ids: list[int] = []
    delay_delta_arrays: list[np.ndarray] = []
    direction_delta_arrays: list[np.ndarray] = []
    u_ref_per_triplet: list[np.ndarray] = []
    u_full_per_triplet: list[np.ndarray] = []
    delay_vector_map: dict[int, dict[float, np.ndarray]] = {}
    direction_vector_map: dict[int, dict[str, np.ndarray]] = {}
    delay_reference_statuses: dict[str, int] = {}
    overlap_reference_statuses: dict[str, int] = {}
    skipped_delay_triplets: list[int] = []
    skipped_overlap_triplets: list[int] = []

    batch_starts = range(0, len(df_triplets), int(args.batch_size))
    total_batches = math.ceil(len(df_triplets) / int(args.batch_size))
    for batch_start in tqdm(batch_starts, total=total_batches, desc="Running strength-direction decomposition"):
        batch_df = df_triplets.iloc[batch_start : batch_start + int(args.batch_size)].copy().reset_index(drop=True)
        sample_spikes, distractor_spikes, probe_spikes = prepare_triplet_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            spec=spec,
            device=device,
        )
        batch_triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        batch_masks = [mask_records[triplet_id] for triplet_id in batch_triplet_ids]

        delay_outputs = collect_delay_sweep_outputs(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            delay_ms_values=delay_ms_values,
            spec=spec,
            readout_step=readout_step,
        )
        overlap_outputs = collect_overlap_condition_outputs(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            batch_masks=batch_masks,
            direction_delay_ms=float(args.direction_delay_ms),
            spec=spec,
            readout_step=readout_step,
            include_conditions=OVERLAP_CONDITION_ORDER,
        )

        for batch_idx, triplet_row in enumerate(batch_df.itertuples(index=False)):
            triplet_id = int(triplet_row.triplet_id)
            delay_delta_by_triplet = {
                float(delay_ms): np.asarray(delay_outputs["delta_v"][batch_idx, delay_idx], dtype=np.float64)
                for delay_idx, delay_ms in enumerate(delay_ms_values)
            }
            ref_info = compute_reference_direction_from_delays(delay_delta_by_triplet)
            delay_status = str(ref_info["status"])
            delay_reference_statuses[delay_status] = delay_reference_statuses.get(delay_status, 0) + 1
            if ref_info["u_ref"] is not None:
                u_ref = np.asarray(ref_info["u_ref"], dtype=np.float64)
                delay_triplet_ids.append(triplet_id)
                u_ref_per_triplet.append(u_ref.astype(np.float32, copy=False))
                delay_delta_arrays.append(np.stack([delay_delta_by_triplet[float(delay_ms)] for delay_ms in delay_ms_values], axis=0).astype(np.float32, copy=False))
                delay_vector_map[triplet_id] = {float(delay_ms): np.asarray(delay_delta_by_triplet[float(delay_ms)], dtype=np.float64) for delay_ms in delay_ms_values}
                for delay_ms in delay_ms_values:
                    metrics = compute_strength_metrics(delay_delta_by_triplet[float(delay_ms)], u_ref)
                    delay_rows.append(
                        {
                            "triplet_id": triplet_id,
                            "delay_ms": float(delay_ms),
                            "M": float(metrics["M"]),
                            "A": float(metrics["A"]),
                            "cos_theta": float(metrics["cos_theta"]),
                            "theta_deg": float(metrics["theta_deg"]),
                            "reference_status": delay_status,
                            "reference_fallback_delay_ms": ref_info.get("fallback_delay_ms"),
                        }
                    )
            else:
                skipped_delay_triplets.append(triplet_id)

            delta_v_by_condition = {
                str(condition_name): np.asarray(overlap_outputs["delta_v"][batch_idx, cond_idx], dtype=np.float64)
                for cond_idx, condition_name in enumerate(OVERLAP_CONDITION_ORDER)
            }
            rotation_info = compute_direction_rotation_metrics(delta_v_by_condition)
            overlap_status = str(rotation_info["status"])
            overlap_reference_statuses[overlap_status] = overlap_reference_statuses.get(overlap_status, 0) + 1
            if rotation_info["u_full"] is None:
                skipped_overlap_triplets.append(triplet_id)
                continue
            u_full = np.asarray(rotation_info["u_full"], dtype=np.float64)
            direction_triplet_ids.append(triplet_id)
            u_full_per_triplet.append(u_full.astype(np.float32, copy=False))
            direction_delta_arrays.append(np.stack([delta_v_by_condition[condition_name] for condition_name in OVERLAP_CONDITION_ORDER], axis=0).astype(np.float32, copy=False))
            direction_vector_map[triplet_id] = {condition_name: np.asarray(delta_v_by_condition[condition_name], dtype=np.float64) for condition_name in OVERLAP_CONDITION_ORDER}
            feature_subset = condition_feature_table[condition_feature_table["triplet_id"] == triplet_id].copy()
            probe_id = int(triplet_row.probe_id)
            anchor_meta = triplet_anchor_lookup.get(triplet_id, {})
            for condition_name in OVERLAP_CONDITION_ORDER:
                rotation_metrics = rotation_info["metrics"][condition_name]
                feature_row = feature_subset[feature_subset["condition"] == condition_name]
                if feature_row.empty:
                    continue
                feature_dict = dict(feature_row.iloc[0].to_dict())
                source_metrics = compute_triplet_source_anchor_metrics(
                    delta_v_condition=delta_v_by_condition[condition_name],
                    uSP_triplet_raw=anchor_meta.get("uSP_raw"),
                    uSP_triplet_unit=anchor_meta.get("uSP_unit"),
                    uDP_triplet_raw=anchor_meta.get("uDP_raw"),
                    uDP_triplet_unit=anchor_meta.get("uDP_unit"),
                )
                overlap_rows.append(
                    {
                        "triplet_id": triplet_id,
                        "probe_id": probe_id,
                        "sample_id": int(triplet_row.sample_id),
                        "distractor_id": int(triplet_row.distractor_id),
                        "condition": str(condition_name),
                        "cos_phi": float(rotation_metrics["cos_phi"]),
                        "phi_deg": float(rotation_metrics["phi_deg"]),
                        "M_condition": float(rotation_metrics["M_condition"]),
                        **source_metrics,
                        "uSP_status": str(anchor_meta.get("uSP_status", "skip_zero_norm")),
                        "uDP_status": str(anchor_meta.get("uDP_status", "skip_zero_norm")),
                        "uSP_norm": float(anchor_meta.get("uSP_norm", float("nan"))),
                        "uDP_norm": float(anchor_meta.get("uDP_norm", float("nan"))),
                        "abs_X_SP": float(feature_dict["abs_X_SP"]),
                        "abs_X_DP": float(feature_dict["abs_X_DP"]),
                        "abs_X_SDP_sample": float(feature_dict["abs_X_SDP_sample"]),
                        "abs_X_SDP_distr": float(feature_dict["abs_X_SDP_distr"]),
                        "delta_X_SP": float(feature_dict["delta_X_SP"]),
                        "delta_X_DP": float(feature_dict["delta_X_DP"]),
                        "delta_X_SDP_sample": float(feature_dict["delta_X_SDP_sample"]),
                        "delta_X_SDP_distr": float(feature_dict["delta_X_SDP_distr"]),
                        "overlap_bias_1": float(feature_dict["overlap_bias_1"]),
                        "overlap_bias_2": float(feature_dict["overlap_bias_2"]),
                        "overlap_bias_3": float(feature_dict["overlap_bias_3"]),
                        "reference_status": overlap_status,
                    }
                )

    df_delay_records = pd.DataFrame(delay_rows).sort_values(["triplet_id", "delay_ms"], kind="stable").reset_index(drop=True)
    df_overlap_records = pd.DataFrame(overlap_rows).sort_values(["triplet_id", "condition"], kind="stable").reset_index(drop=True)
    if df_delay_records.empty:
        raise RuntimeError("Delay block produced no valid records.")
    if df_overlap_records.empty:
        raise RuntimeError("Overlap block produced no valid records.")

    anchor_bin_outputs = assign_anchor_similarity_bins(df_triplet_anchor_vectors)
    df_triplet_anchor_bins = anchor_bin_outputs["triplet_anchor_bins"].copy()
    df_overlap_records = df_overlap_records.merge(
        df_triplet_anchor_bins[["triplet_id", "anchor_cosine", "anchor_similarity_bin"]].drop_duplicates(subset=["triplet_id"], keep="first"),
        on="triplet_id",
        how="left",
    )
    stratified_outputs = summarize_source_bias_by_anchor_bin(df_overlap_records, df_triplet_anchor_bins)
    df_source_anchor_bin_summary = stratified_outputs["summary_df"]
    anchor_bin_sensitivity = compute_anchor_bin_sensitivity_metrics(df_source_anchor_bin_summary)

    delay_strength_csv = save_tidy_csv(df_delay_records, output_dir / "delay_strength_records.csv", sort_by=["triplet_id", "delay_ms"])
    overlap_direction_csv = save_tidy_csv(df_overlap_records, output_dir / "overlap_direction_records.csv", sort_by=["triplet_id", "condition"])
    triplet_source_anchor_records_csv = save_tidy_csv(
        df_overlap_records[
            [
                "triplet_id",
                "probe_id",
                "sample_id",
                "distractor_id",
                "condition",
                "anchor_cosine",
                "anchor_similarity_bin",
                "cos_to_uSP",
                "cos_to_uDP",
                "proj_to_uSP_unit",
                "proj_to_uDP_unit",
                "proj_to_uSP_raw",
                "proj_to_uDP_raw",
                "source_bias_raw",
                "source_bias_norm",
                "cos_phi",
                "phi_deg",
                "M_condition",
                "uSP_status",
                "uDP_status",
                "uSP_norm",
                "uDP_norm",
            ]
        ].copy(),
        output_dir / "triplet_source_anchor_records.csv",
        sort_by=["triplet_id", "condition"],
    )
    triplet_anchor_vectors_csv = save_tidy_csv(
        df_triplet_anchor_vectors[
            [
                "triplet_id",
                "probe_id",
                "sample_id",
                "distractor_id",
                "uSP_status",
                "uDP_status",
                "uSP_norm",
                "uDP_norm",
                "anchor_cosine",
            ]
        ].copy(),
        output_dir / "triplet_anchor_vectors.csv",
        sort_by=["triplet_id"],
    )
    triplet_anchor_bins_csv = save_tidy_csv(
        df_triplet_anchor_bins[
            [
                "triplet_id",
                "probe_id",
                "anchor_cosine",
                "anchor_similarity_bin",
                "uSP_norm",
                "uDP_norm",
                "uSP_status",
                "uDP_status",
            ]
        ].copy(),
        output_dir / "triplet_anchor_bins.csv",
        sort_by=["triplet_id"],
    )
    source_anchor_bin_summary_csv = save_tidy_csv(
        df_source_anchor_bin_summary,
        output_dir / "source_anchor_bin_summary.csv",
    )

    delta_npz_path = output_dir / "delta_v_arrays.npz"
    np.savez_compressed(
        delta_npz_path,
        triplet_id_delay=np.asarray(delay_triplet_ids, dtype=np.int64),
        delay_ms=np.asarray(delay_ms_values, dtype=np.float32),
        delta_v_delay=np.stack(delay_delta_arrays, axis=0) if delay_delta_arrays else np.zeros((0, len(delay_ms_values), num_classes), dtype=np.float32),
        triplet_id_condition=np.asarray(direction_triplet_ids, dtype=np.int64),
        condition_name=np.asarray(OVERLAP_CONDITION_ORDER),
        delta_v_condition=np.stack(direction_delta_arrays, axis=0) if direction_delta_arrays else np.zeros((0, len(OVERLAP_CONDITION_ORDER), num_classes), dtype=np.float32),
        u_ref_per_triplet=np.stack(u_ref_per_triplet, axis=0) if u_ref_per_triplet else np.zeros((0, num_classes), dtype=np.float32),
        u_full_per_triplet=np.stack(u_full_per_triplet, axis=0) if u_full_per_triplet else np.zeros((0, num_classes), dtype=np.float32),
    )
    triplet_anchor_vectors_npz = output_dir / "triplet_anchor_vectors.npz"
    np.savez_compressed(
        triplet_anchor_vectors_npz,
        triplet_ids=df_triplet_anchor_vectors["triplet_id"].to_numpy(dtype=np.int64, copy=False) if not df_triplet_anchor_vectors.empty else np.zeros(0, dtype=np.int64),
        uSP_raw=np.stack(
            [
                np.asarray(row.uSP_raw, dtype=np.float32)
                if row.uSP_raw is not None
                else np.zeros(num_classes, dtype=np.float32)
                for row in df_triplet_anchor_vectors.itertuples(index=False)
            ],
            axis=0,
        )
        if not df_triplet_anchor_vectors.empty
        else np.zeros((0, num_classes), dtype=np.float32),
        uSP_unit=np.stack(
            [
                np.asarray(row.uSP_unit, dtype=np.float32)
                if row.uSP_unit is not None
                else np.zeros(num_classes, dtype=np.float32)
                for row in df_triplet_anchor_vectors.itertuples(index=False)
            ],
            axis=0,
        )
        if not df_triplet_anchor_vectors.empty
        else np.zeros((0, num_classes), dtype=np.float32),
        uDP_raw=np.stack(
            [
                np.asarray(row.uDP_raw, dtype=np.float32)
                if row.uDP_raw is not None
                else np.zeros(num_classes, dtype=np.float32)
                for row in df_triplet_anchor_vectors.itertuples(index=False)
            ],
            axis=0,
        )
        if not df_triplet_anchor_vectors.empty
        else np.zeros((0, num_classes), dtype=np.float32),
        uDP_unit=np.stack(
            [
                np.asarray(row.uDP_unit, dtype=np.float32)
                if row.uDP_unit is not None
                else np.zeros(num_classes, dtype=np.float32)
                for row in df_triplet_anchor_vectors.itertuples(index=False)
            ],
            axis=0,
        )
        if not df_triplet_anchor_vectors.empty
        else np.zeros((0, num_classes), dtype=np.float32),
        uSP_valid=(df_triplet_anchor_vectors["uSP_status"] == "ok").to_numpy(dtype=bool, copy=False) if not df_triplet_anchor_vectors.empty else np.zeros(0, dtype=bool),
        uDP_valid=(df_triplet_anchor_vectors["uDP_status"] == "ok").to_numpy(dtype=bool, copy=False) if not df_triplet_anchor_vectors.empty else np.zeros(0, dtype=bool),
    )

    delay_summary_by_delay = []
    for delay_ms in delay_ms_values:
        subset = df_delay_records[df_delay_records["delay_ms"] == float(delay_ms)]
        delay_summary_by_delay.append(
            {
                "delay_ms": float(delay_ms),
                "n_triplets": int(subset["triplet_id"].nunique()),
                "mean_A": float(subset["A"].mean()),
                "sem_A": _sem(subset["A"].to_numpy(dtype=np.float64)),
                "mean_M": float(subset["M"].mean()),
                "sem_M": _sem(subset["M"].to_numpy(dtype=np.float64)),
                "mean_cos_theta": float(subset["cos_theta"].mean(skipna=True)),
                "sem_cos_theta": _sem(subset["cos_theta"].to_numpy(dtype=np.float64)),
                "mean_theta_deg": float(subset["theta_deg"].mean(skipna=True)),
                "sem_theta_deg": _sem(subset["theta_deg"].to_numpy(dtype=np.float64)),
            }
        )
    overlap_correlation_summary = compute_overlap_bias_correlations(df_overlap_records)
    source_anchor_summary = summarize_source_anchor_effects(df_overlap_records)
    source_anchor_summary["triplet_anchor_diagnostic"] = {
        "n_triplets": int(len(df_triplet_anchor_vectors)),
        "n_valid_anchor_pairs": int(np.isfinite(df_triplet_anchor_vectors["anchor_cosine"].to_numpy(dtype=np.float64)).sum()) if not df_triplet_anchor_vectors.empty else 0,
        "mean_anchor_cosine": float(df_triplet_anchor_vectors["anchor_cosine"].mean(skipna=True)) if not df_triplet_anchor_vectors.empty else float("nan"),
        "sem_anchor_cosine": _sem(df_triplet_anchor_vectors["anchor_cosine"].to_numpy(dtype=np.float64)) if not df_triplet_anchor_vectors.empty else 0.0,
        "uSP_status_counts": df_triplet_anchor_vectors["uSP_status"].value_counts(dropna=False).to_dict() if not df_triplet_anchor_vectors.empty else {},
        "uDP_status_counts": df_triplet_anchor_vectors["uDP_status"].value_counts(dropna=False).to_dict() if not df_triplet_anchor_vectors.empty else {},
    }
    case_triplets = _select_case_triplets(df_delay_records, df_overlap_records, count=int(args.save_case_count))
    source_anchor_summary_json = _save_json(source_anchor_summary, output_dir / "source_anchor_summary.json")
    source_anchor_bin_summary_json_payload = {
        "q33": float(anchor_bin_outputs["q33"]),
        "q67": float(anchor_bin_outputs["q67"]),
        "bin_counts": anchor_bin_outputs["bin_counts"],
        "invalid_count": int(anchor_bin_outputs["invalid_count"]),
        "warnings": list(anchor_bin_outputs["warnings"]),
        "condition_bin_summary": df_source_anchor_bin_summary.to_dict(orient="records"),
        "sensitivity_metrics": anchor_bin_sensitivity,
        "assumptions": {
            "anchor_similarity_stratification": "Triplets were stratified into low/middle/high source-anchor similarity bins based on tertiles of triplet-specific cos(uSP, uDP). This analysis was used as a mechanistic sensitivity analysis to assess whether source-bias metrics become more discriminative when the two source anchors are more geometrically separable. It is supplementary to, rather than a replacement for, the full-sample primary analysis.",
            "binning_rule": "tertiles on valid triplet-specific anchor cosine",
        },
    }
    source_anchor_bin_summary_json = _save_json(source_anchor_bin_summary_json_payload, output_dir / "source_anchor_bin_summary.json")

    summary_metrics = {
        "delay_block": {
            "n_valid_triplets": int(df_delay_records["triplet_id"].nunique()),
            "n_skipped_triplets": int(len(skipped_delay_triplets)),
            "skipped_triplet_ids": [int(triplet_id) for triplet_id in skipped_delay_triplets],
            "reference_status_counts": delay_reference_statuses,
            "delay_aggregate": delay_summary_by_delay,
            "trend_A": summarize_delay_trend(df_delay_records, "A"),
            "trend_M": summarize_delay_trend(df_delay_records, "M"),
            "trend_cos_theta": summarize_delay_trend(df_delay_records, "cos_theta"),
            "spearman_delay_vs_A": compute_spearman_summary(
                df_delay_records["delay_ms"].to_numpy(dtype=np.float64),
                df_delay_records["A"].to_numpy(dtype=np.float64),
            ),
            "spearman_delay_vs_M": compute_spearman_summary(
                df_delay_records["delay_ms"].to_numpy(dtype=np.float64),
                df_delay_records["M"].to_numpy(dtype=np.float64),
            ),
        },
        "overlap_block": {
            "n_valid_triplets": int(df_overlap_records["triplet_id"].nunique()),
            "n_skipped_triplets": int(len(skipped_overlap_triplets)),
            "skipped_triplet_ids": [int(triplet_id) for triplet_id in skipped_overlap_triplets],
            "reference_status_counts": overlap_reference_statuses,
            **overlap_correlation_summary,
        },
        "source_anchor_block": {
            "n_triplets_with_uSP": int((df_triplet_anchor_vectors["uSP_status"] == "ok").sum()) if not df_triplet_anchor_vectors.empty else 0,
            "n_triplets_with_uDP": int((df_triplet_anchor_vectors["uDP_status"] == "ok").sum()) if not df_triplet_anchor_vectors.empty else 0,
            **source_anchor_summary,
        },
        "anchor_similarity_stratification_block": {
            "q33": float(anchor_bin_outputs["q33"]),
            "q67": float(anchor_bin_outputs["q67"]),
            "bin_counts": anchor_bin_outputs["bin_counts"],
            "invalid_count": int(anchor_bin_outputs["invalid_count"]),
            "warnings": list(anchor_bin_outputs["warnings"]),
            "sensitivity_metrics": anchor_bin_sensitivity,
        },
        "assumptions": {
            "static_as_origin": True,
            "delta_v_definition": "centered_grouped_voltage(V_condition) - centered_grouped_voltage(V_static)",
            "center_operator": "subtract mean across grouped-voltage channels only; no L2 normalization",
            "strength_reference_direction": "normalize(sum over delays of DeltaV(delay))",
            "strength_reference_fallback": "use the direction of the maximum-norm non-zero DeltaV across delays when the summed vector norm is too small",
            "overlap_reference_direction": "normalize(DeltaV(full_dynamic))",
            "source_anchor_definition": {
                "uSP": "triplet-specific empirical source anchor from single-source sample -> delay2 -> probe DMS rollouts",
                "uDP": "triplet-specific empirical source anchor from single-source distractor -> delay2 -> probe DMS rollouts",
                "fixed_anchor_delay": "delay2 only",
                "aggregation": "none; anchors are computed independently for each triplet",
                "personalization": "triplet-specific, not probe-averaged",
            },
            "anchor_similarity_stratification": "Triplets were stratified into low/middle/high source-anchor similarity bins based on tertiles of triplet-specific cos(uSP, uDP). This analysis was used as a mechanistic sensitivity analysis to assess whether source-bias metrics become more discriminative when the two source anchors are more geometrically separable. It is supplementary to, rather than a replacement for, the full-sample primary analysis.",
            "no_fixed_delay_threshold": True,
            "mechanistic_goal": {"delay_controls_strength": True, "overlap_controls_direction": True},
            "full_direction_space": int(num_classes),
            "overlap_bias_reporting": {"pooled_correlations_exploratory": True, "within_condition_correlations_primary": True},
            "time_weight_tau_ms": float(DEFAULT_TAU_MS),
            "abs_X_SDP_eff_definition": "0.5 * (abs_X_SDP_sample + abs_X_SDP_distr)",
        },
        "case_triplet_ids": [int(triplet_id) for triplet_id in case_triplets],
    }
    summary_json = _save_json(summary_metrics, output_dir / "summary_metrics.json")

    figure_paths: dict[str, str] = {}
    if not bool(args.skip_figures):
        fig1 = plot_mechanism_schematic()
        out1 = save_figure_all_formats(fig1, figures_dir / "figure_1_mechanism_schematic")
        plt.close(fig1)
        figure_paths.update({f"figure_1_{key}": value for key, value in out1.items()})

        fig2 = plot_delay_block(df_delay_records)
        out2 = save_figure_all_formats(fig2, figures_dir / "figure_2_delay_block")
        plt.close(fig2)
        figure_paths.update({f"figure_2_{key}": value for key, value in out2.items()})

        fig3 = plot_overlap_block(df_overlap_records)
        out3 = save_figure_all_formats(fig3, figures_dir / "figure_3_overlap_block")
        plt.close(fig3)
        figure_paths.update({f"figure_3_{key}": value for key, value in out3.items()})

        fig4 = plot_overlap_bias_correlation(df_overlap_records, overlap_correlation_summary)
        out4 = save_figure_all_formats(fig4, figures_dir / "figure_4_overlap_bias_correlation")
        plt.close(fig4)
        figure_paths.update({f"figure_4_{key}": value for key, value in out4.items()})

        fig5 = plot_representative_vector_examples(
            case_triplets=case_triplets,
            delay_ms_values=delay_ms_values,
            delay_vector_map=delay_vector_map,
            direction_vector_map=direction_vector_map,
        )
        out5 = save_figure_all_formats(fig5, figures_dir / "figure_5_representative_vector_examples")
        plt.close(fig5)
        figure_paths.update({f"figure_5_{key}": value for key, value in out5.items()})

        fig6 = plot_triplet_anchor_similarity(df_triplet_anchor_vectors)
        out6 = save_figure_all_formats(fig6, figures_dir / "figure_6_triplet_anchor_similarity")
        plt.close(fig6)
        figure_paths.update({f"figure_6_{key}": value for key, value in out6.items()})

        fig7 = plot_source_anchor_block(df_overlap_records)
        out7 = save_figure_all_formats(fig7, figures_dir / "figure_7_source_anchor_block")
        plt.close(fig7)
        figure_paths.update({f"figure_7_{key}": value for key, value in out7.items()})

        fig8 = plot_anchor_similarity_bin_counts(df_triplet_anchor_bins)
        out8 = save_figure_all_formats(fig8, figures_dir / "figure_8_anchor_similarity_bin_counts")
        plt.close(fig8)
        figure_paths.update({f"figure_8_{key}": value for key, value in out8.items()})

        fig9 = plot_source_bias_by_anchor_bin(df_source_anchor_bin_summary)
        out9 = save_figure_all_formats(fig9, figures_dir / "figure_9_source_bias_by_anchor_bin")
        plt.close(fig9)
        figure_paths.update({f"figure_9_{key}": value for key, value in out9.items()})

        fig10 = plot_abs_source_bias_by_anchor_bin(df_source_anchor_bin_summary)
        out10 = save_figure_all_formats(fig10, figures_dir / "figure_10_abs_source_bias_by_anchor_bin")
        plt.close(fig10)
        figure_paths.update({f"figure_10_{key}": value for key, value in out10.items()})

    run_config_path = save_run_config(
        {
            "model_path": str(Path(args.model_path).resolve()),
            "config_argument": args.config,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": str(args.split),
            "output_dir": str(result_root.resolve()),
            "device": str(device),
            "seed": int(args.seed),
            "delay1_ms": float(args.delay1_ms),
            "sample_ms": float(args.sample_ms),
            "distractor_ms": float(args.distractor_ms),
            "probe_ms": float(args.probe_ms),
            "delay_sweep_ms": [float(delay_ms) for delay_ms in delay_ms_values],
            "direction_delay_ms": float(args.direction_delay_ms),
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_triplets": int(args.max_triplets),
            "num_sim_bins": int(args.num_sim_bins),
            "foreground_threshold": float(args.foreground_threshold),
            "use_dilated_overlap": bool(args.use_dilated_overlap),
            "dilation_radius": int(args.dilation_radius),
            "save_case_count": int(args.save_case_count),
            "num_control_candidates": int(args.num_control_candidates),
            "skip_figures": bool(args.skip_figures),
            "readout_step": int(readout_step),
            "tau_ms": float(DEFAULT_TAU_MS),
            "outputs": {
                "delay_strength_records_csv": str(Path(delay_strength_csv).resolve()),
                "overlap_direction_records_csv": str(Path(overlap_direction_csv).resolve()),
                "triplet_source_anchor_records_csv": str(Path(triplet_source_anchor_records_csv).resolve()),
                "triplet_anchor_vectors_csv": str(Path(triplet_anchor_vectors_csv).resolve()),
                "triplet_anchor_bins_csv": str(Path(triplet_anchor_bins_csv).resolve()),
                "source_anchor_bin_summary_csv": str(Path(source_anchor_bin_summary_csv).resolve()),
                "delta_v_arrays_npz": str(delta_npz_path.resolve()),
                "triplet_anchor_vectors_npz": str(triplet_anchor_vectors_npz.resolve()),
                "summary_metrics_json": str(Path(summary_json).resolve()),
                "source_anchor_summary_json": str(Path(source_anchor_summary_json).resolve()),
                "source_anchor_bin_summary_json": str(Path(source_anchor_bin_summary_json).resolve()),
                **figure_paths,
            },
            "assumptions": summary_metrics["assumptions"],
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "distractor_strength_direction_decomposition_experiment",
            "triplet_count": int(df_results["triplet_id"].nunique()),
            "artifact_summary_metrics_json": str(summary_json.resolve()),
            "source_anchor_summary_json": str(source_anchor_summary_json.resolve()),
            "source_anchor_bin_summary_json": str(source_anchor_bin_summary_json.resolve()),
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=distractor_strength_direction_decomposition_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"triplets={int(df_results['triplet_id'].nunique())}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )

    print("\n=== Distractor Strength-Direction Decomposition Summary ===")
    print(f"Triplets (delay block): {int(df_delay_records['triplet_id'].nunique())}")
    print(f"Triplets (overlap block): {int(df_overlap_records['triplet_id'].nunique())}")
    print(f"Delay records CSV: {delay_strength_csv}")
    print(f"Overlap records CSV: {overlap_direction_csv}")
    print(f"Source-anchor records CSV: {triplet_source_anchor_records_csv}")
    print(f"Triplet-anchor vectors CSV: {triplet_anchor_vectors_csv}")
    print(f"Triplet-anchor bins CSV: {triplet_anchor_bins_csv}")
    print(f"Source-anchor bin summary CSV: {source_anchor_bin_summary_csv}")
    print(f"Delta-V arrays NPZ: {delta_npz_path}")
    print(f"Triplet-anchor vectors NPZ: {triplet_anchor_vectors_npz}")
    print(f"Summary JSON: {summary_json}")
    print(f"Source-anchor summary JSON: {source_anchor_summary_json}")
    print(f"Source-anchor bin summary JSON: {source_anchor_bin_summary_json}")
    print(f"Run config: {run_config_path}")


if __name__ == "__main__":
    main()
