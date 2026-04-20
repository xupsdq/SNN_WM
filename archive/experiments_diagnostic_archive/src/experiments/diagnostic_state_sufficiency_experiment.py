from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from input_function import build_mnist_skeleton_loader
from src.config.units import ms
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import (
    reset_fast_state_in_place,
    reset_stsp_to_baseline_in_place,
    run_monitored_dms_rollout,
    run_monitored_probe_only_rollout,
)
from src.experiments.common.ping_common import LAYER_KEYS
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.diagnostic_region_utils import (
    bootstrap_rate_ci,
    build_dataset_arrays,
    compute_region_specific_overlap,
    encode_images,
    estimate_diagnostic_regions,
    mix_seed,
    paired_bootstrap_diff_summary,
    rank_correlation,
    select_sample_types_for_probe,
)
from src.plotting.common.io import (
    COLOR_DYNAMIC,
    COLOR_NOISE,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    apply_publication_style,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)

DONOR_TYPES: Tuple[str, ...] = ("diagnostic", "baseline", "nondiagnostic")
GAIN_SWEEP_ALPHAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITION_BASELINE = "baseline_no_memory"
CONDITION_DIAGNOSTIC = "inject_diagnostic_donor"
CONDITION_BASELINE_DONOR = "inject_baseline_donor"
CONDITION_NONDIAGNOSTIC = "inject_nondiagnostic_donor"
CONDITION_SHAM = "inject_sham_donor"
CONDITION_GAIN = "inject_diagnostic_gain_sweep"
CORE_CONDITION_ORDER: Tuple[str, ...] = (
    CONDITION_BASELINE,
    CONDITION_DIAGNOSTIC,
    CONDITION_BASELINE_DONOR,
    CONDITION_NONDIAGNOSTIC,
    CONDITION_SHAM,
)
DEFAULT_SAVE_DIR = "results/fig6_state_sufficiency"
DEFAULT_PATCH_SIZE = 4
DEFAULT_PROBE_POOL_LIMIT = 2000
DEFAULT_PROBE_POOL_PER_CLASS = 200
DEFAULT_BASELINE_EARLY_STOP_MULTIPLIER = 2.0

CONDITION_COLORS: Dict[str, str] = {
    CONDITION_BASELINE: COLOR_NOISE,
    CONDITION_DIAGNOSTIC: "#009E73",
    CONDITION_BASELINE_DONOR: "#0072B2",
    CONDITION_NONDIAGNOSTIC: "#D55E00",
    CONDITION_SHAM: "#CC79A7",
    CONDITION_GAIN: COLOR_DYNAMIC,
}


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round((self.sample_ms * ms) / self.dt))

    @property
    def delay_steps(self) -> int:
        return int(round((self.delay_ms * ms) / self.dt))

    @property
    def probe_steps(self) -> int:
        return int(round((self.probe_ms * ms) / self.dt))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _errorbar_from_ci(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    lows_arr = np.asarray(lows, dtype=np.float64)
    highs_arr = np.asarray(highs, dtype=np.float64)
    lower = np.minimum(lows_arr, highs_arr)
    upper = np.maximum(lows_arr, highs_arr)
    return np.vstack(
        [
            np.clip(values_arr - lower, a_min=0.0, a_max=None),
            np.clip(upper - values_arr, a_min=0.0, a_max=None),
        ]
    )


def _slope_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return float("nan")
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    x_centered = x_use - float(x_use.mean())
    denom = float(np.sum(x_centered ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x_centered * (y_use - float(y_use.mean()))) / denom)


def _stack_probe_images(dataset, probe_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.stack([dataset[int(probe_id)][0] for probe_id in probe_ids], dim=0).to(device)


def _score_candidates_for_probe(
    probe_id: int,
    probe_label: int,
    image_matrix_flat: np.ndarray,
    dataset_labels: np.ndarray,
    diagnostic_mask: np.ndarray,
    nondiagnostic_mask: np.ndarray,
) -> pd.DataFrame:
    probe_vector = image_matrix_flat[int(probe_id)]
    candidate_ids = np.arange(len(dataset_labels), dtype=np.int64)
    keep_mask = candidate_ids != int(probe_id)
    filtered_ids = candidate_ids[keep_mask]
    filtered_labels = dataset_labels[keep_mask]
    filtered_matrix = image_matrix_flat[keep_mask]
    diagnostic_scores, nondiagnostic_scores = compute_region_specific_overlap(
        probe_vector=probe_vector,
        candidate_matrix=filtered_matrix,
        diagnostic_mask=diagnostic_mask,
        nondiagnostic_mask=nondiagnostic_mask,
    )
    rows: List[Dict[str, object]] = []
    for local_idx, candidate_id in enumerate(filtered_ids.tolist()):
        diag_score = float(diagnostic_scores[local_idx])
        nond_score = float(nondiagnostic_scores[local_idx])
        candidate_label = int(filtered_labels[local_idx])
        rows.append(
            {
                "probe_id": int(probe_id),
                "probe_label": int(probe_label),
                "candidate_id": int(candidate_id),
                "candidate_label": candidate_label,
                "label_relation": "same_label" if candidate_label == int(probe_label) else "different_label",
                "diagnostic_overlap_score": diag_score,
                "nondiagnostic_overlap_score": nond_score,
                "diagnostic_margin": float(diag_score - nond_score),
            }
        )
    return pd.DataFrame(rows).sort_values(["label_relation", "candidate_id"], kind="stable").reset_index(drop=True)


def _select_baseline_donor_for_probe(scored_df: pd.DataFrame, label_relation: str) -> Dict[str, object] | None:
    subset = scored_df[scored_df["label_relation"] == str(label_relation)].copy()
    if subset.empty:
        return None
    diag_median = float(subset["diagnostic_overlap_score"].median())
    nond_median = float(subset["nondiagnostic_overlap_score"].median())
    margin_median = float(subset["diagnostic_margin"].median())
    subset["median_objective"] = (
        (subset["diagnostic_overlap_score"] - diag_median).abs()
        + (subset["nondiagnostic_overlap_score"] - nond_median).abs()
        + (subset["diagnostic_margin"] - margin_median).abs()
    )
    subset["abs_margin"] = subset["diagnostic_margin"].abs()
    chosen = subset.sort_values(["median_objective", "abs_margin", "candidate_id"], kind="stable").iloc[0]
    return {
        "baseline_sample_id": int(chosen["candidate_id"]),
        "baseline_sample_label": int(chosen["candidate_label"]),
        "baseline_diagnostic_overlap_score": float(chosen["diagnostic_overlap_score"]),
        "baseline_nondiagnostic_overlap_score": float(chosen["nondiagnostic_overlap_score"]),
        "baseline_diagnostic_margin": float(chosen["diagnostic_margin"]),
        "baseline_selection_objective": float(chosen["median_objective"]),
    }


def build_donor_selection_table(
    probe_region_summary: pd.DataFrame,
    mask_lookup: Mapping[int, Mapping[str, np.ndarray | int | str]],
    image_matrix_flat: np.ndarray,
    dataset_labels: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: List[Dict[str, object]] = []
    donor_rows: List[Dict[str, object]] = []
    probe_trial_id = 0
    for row in probe_region_summary.itertuples(index=False):
        probe_id = int(row.probe_id)
        probe_label = int(row.probe_label)
        if int(row.is_region_valid) != 1:
            selection_rows.append(
                {
                    "probe_id": probe_id,
                    "probe_label": probe_label,
                    "selection_status": "excluded",
                    "selection_exclusion_reason": str(row.region_exclusion_reason),
                }
            )
            continue
        selection = dict(
            select_sample_types_for_probe(
                probe_id=probe_id,
                probe_label=probe_label,
                image_matrix_flat=image_matrix_flat,
                dataset_labels=dataset_labels,
                diagnostic_mask=np.asarray(mask_lookup[probe_id]["diagnostic_mask"], dtype=np.bool_),
                nondiagnostic_mask=np.asarray(mask_lookup[probe_id]["nondiagnostic_mask"], dtype=np.bool_),
            )
        )
        if str(selection.get("selection_status")) != "selected":
            selection_rows.append(selection)
            continue
        scored_df = _score_candidates_for_probe(
            probe_id=probe_id,
            probe_label=probe_label,
            image_matrix_flat=image_matrix_flat,
            dataset_labels=dataset_labels,
            diagnostic_mask=np.asarray(mask_lookup[probe_id]["diagnostic_mask"], dtype=np.bool_),
            nondiagnostic_mask=np.asarray(mask_lookup[probe_id]["nondiagnostic_mask"], dtype=np.bool_),
        )
        baseline_pick = _select_baseline_donor_for_probe(scored_df=scored_df, label_relation=str(selection["label_relation"]))
        if baseline_pick is None:
            selection_rows.append(
                {
                    "probe_id": probe_id,
                    "probe_label": probe_label,
                    "selection_status": "excluded",
                    "selection_exclusion_reason": "no_baseline_candidate_for_label_relation",
                }
            )
            continue
        selection_rows.append({**selection, **baseline_pick})
        donor_specs = [
            (
                "diagnostic",
                int(selection["diagnostic_sample_id"]),
                int(selection["diagnostic_sample_label"]),
                float(selection["diagnostic_overlap_score"]),
                float(selection["diagnostic_nondiagnostic_overlap_score"]),
                float(selection["diagnostic_margin"]),
            ),
            (
                "baseline",
                int(baseline_pick["baseline_sample_id"]),
                int(baseline_pick["baseline_sample_label"]),
                float(baseline_pick["baseline_diagnostic_overlap_score"]),
                float(baseline_pick["baseline_nondiagnostic_overlap_score"]),
                float(baseline_pick["baseline_diagnostic_margin"]),
            ),
            (
                "nondiagnostic",
                int(selection["nondiagnostic_sample_id"]),
                int(selection["nondiagnostic_sample_label"]),
                float(selection["nondiagnostic_overlap_score"]),
                float(selection["nondiagnostic_nondiagnostic_overlap_score"]),
                float(selection["nondiagnostic_overlap_score"]) - float(selection["nondiagnostic_nondiagnostic_overlap_score"]),
            ),
        ]
        for donor_type, sample_id, sample_label, diag_score, nond_score, margin in donor_specs:
            donor_rows.append(
                {
                    "trial_id": int(probe_trial_id),
                    "probe_id": probe_id,
                    "probe_label": probe_label,
                    "donor_uid": f"probe_{probe_id:05d}_{donor_type}",
                    "donor_type": str(donor_type),
                    "donor_sample_id": int(sample_id),
                    "donor_sample_label": int(sample_label),
                    "label_relation": str(selection["label_relation"]),
                    "diagnostic_overlap_score": float(diag_score),
                    "nondiagnostic_overlap_score": float(nond_score),
                    "diagnostic_margin": float(margin),
                    "donor_matches_original_diagnostic_rule": 1,
                    "baseline_selection_objective": float(baseline_pick["baseline_selection_objective"]),
                }
            )
        probe_trial_id += 1
    selection_df = pd.DataFrame(selection_rows).sort_values(["probe_id"], kind="stable").reset_index(drop=True)
    donor_df = pd.DataFrame(donor_rows).sort_values(["trial_id", "donor_type"], kind="stable").reset_index(drop=True)
    return selection_df, donor_df


def _run_harvest_batch_rollout(
    net,
    sample_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str = "dynamic",
) -> Mapping[str, Mapping[str, torch.Tensor]]:
    probe_spikes = torch.zeros(
        (sample_spikes.shape[0], 1, sample_spikes.shape[2], sample_spikes.shape[3], sample_spikes.shape[4]),
        dtype=sample_spikes.dtype,
        device=sample_spikes.device,
    )
    out = run_monitored_dms_rollout(
        net=net,
        sample_spikes=sample_spikes,
        probe_spikes=probe_spikes,
        delay_steps=delay_steps,
        stsp_mode=stsp_mode,
        phase_reset=True,
        intervention_plan=None,
        record_state_names=(),
    )
    return out["boundary_states"]["pre_intervention"]


def harvest_trial_ux_state(
    net,
    sample_spikes: torch.Tensor,
    delay_steps: int,
    stsp_mode: str = "dynamic",
    sample_idx: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    boundary_state = _run_harvest_batch_rollout(net=net, sample_spikes=sample_spikes, delay_steps=delay_steps, stsp_mode=stsp_mode)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        out[layer_key] = {
            "u": boundary_state[layer_key]["u"][int(sample_idx)].detach().cpu().clone(),
            "x": boundary_state[layer_key]["x"][int(sample_idx)].detach().cpu().clone(),
        }
    return out


def harvest_donor_state_bank(
    net,
    encoder,
    dataset,
    donor_metadata: pd.DataFrame,
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    validate_required_columns(
        donor_metadata,
        [
            "donor_uid",
            "donor_sample_id",
            "donor_sample_label",
            "probe_id",
            "probe_label",
            "donor_type",
            "diagnostic_overlap_score",
            "nondiagnostic_overlap_score",
            "diagnostic_margin",
            "label_relation",
        ],
    )
    state_bank: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    metadata_rows: List[Dict[str, object]] = []
    for start in tqdm(range(0, len(donor_metadata), batch_size), desc="HarvestDonorStates"):
        batch = donor_metadata.iloc[start:start + batch_size].copy().reset_index(drop=True)
        sample_images = torch.stack([dataset[int(idx)][0] for idx in batch["donor_sample_id"].tolist()], dim=0).to(device)
        sample_spikes = encode_images(encoder, sample_images, spec.sample_steps)
        boundary_state = _run_harvest_batch_rollout(net=net, sample_spikes=sample_spikes, delay_steps=spec.delay_steps)
        for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
            donor_uid = str(row.donor_uid)
            layer_map: Dict[str, Dict[str, torch.Tensor]] = {}
            for layer_key in LAYER_KEYS:
                layer_map[layer_key] = {
                    "u": boundary_state[layer_key]["u"][idx_in_batch].detach().cpu().clone(),
                    "x": boundary_state[layer_key]["x"][idx_in_batch].detach().cpu().clone(),
                }
            state_bank[donor_uid] = layer_map
            metadata_rows.append(
                {
                    "trial_id": int(row.trial_id),
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "donor_uid": donor_uid,
                    "donor_type": str(row.donor_type),
                    "donor_sample_id": int(row.donor_sample_id),
                    "donor_sample_label": int(row.donor_sample_label),
                    "label_relation": str(row.label_relation),
                    "donor_matches_probe_label": int(int(row.donor_sample_label) == int(row.probe_label)),
                    "diagnostic_overlap_score": float(row.diagnostic_overlap_score),
                    "nondiagnostic_overlap_score": float(row.nondiagnostic_overlap_score),
                    "diagnostic_margin": float(row.diagnostic_margin),
                    "donor_matches_original_diagnostic_rule": int(row.donor_matches_original_diagnostic_rule),
                    "baseline_selection_objective": float(row.baseline_selection_objective),
                    "harvest_seed": int(seed),
                }
            )
    return {
        "metadata": pd.DataFrame(metadata_rows).sort_values(["trial_id", "donor_type"], kind="stable").reset_index(drop=True),
        "state_bank": state_bank,
    }


def set_fast_state_to_baseline_in_place(net) -> None:
    reset_fast_state_in_place(net)


def set_stsp_to_baseline_in_place(net) -> None:
    reset_stsp_to_baseline_in_place(net)


def _move_state_batch_to_device(
    state_batch: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key, layer_state in state_batch.items():
        out[layer_key] = {
            "u": layer_state["u"].to(device=device),
            "x": layer_state["x"].to(device=device),
        }
    return out


def _stack_donor_state_batch(
    state_bank: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    donor_uids: Sequence[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        out[layer_key] = {
            "u": torch.stack([state_bank[str(donor_uid)][layer_key]["u"] for donor_uid in donor_uids], dim=0).contiguous(),
            "x": torch.stack([state_bank[str(donor_uid)][layer_key]["x"] for donor_uid in donor_uids], dim=0).contiguous(),
        }
    return out


def _build_baseline_state_batch_from_template(
    net,
    template_state: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        out[layer_key] = {
            "u": torch.full_like(template_state[layer_key]["u"], float(layer.stsp_U)),
            "x": torch.ones_like(template_state[layer_key]["x"]),
        }
    return out


def _build_sham_state_batch(
    donor_state: Mapping[str, Mapping[str, torch.Tensor]],
    seed: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    sham_state: Dict[str, Dict[str, torch.Tensor]] = {}
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    for layer_idx, layer_key in enumerate(LAYER_KEYS):
        sham_state[layer_key] = {}
        for state_name in ("u", "x"):
            tensor = donor_state[layer_key][state_name].detach().cpu().clone().contiguous()
            flat = tensor.view(tensor.shape[0], -1)
            shuffled = torch.empty_like(flat)
            for batch_idx in range(flat.shape[0]):
                perm_seed = mix_seed(seed, layer_idx + 1, batch_idx + 1, 11 if state_name == "u" else 17)
                rng.manual_seed(int(perm_seed))
                perm = torch.randperm(flat.shape[1], generator=rng)
                shuffled[batch_idx] = flat[batch_idx].index_select(0, perm)
            sham_state[layer_key][state_name] = shuffled.view_as(tensor).contiguous()
    return sham_state


def _build_scaled_state_batch(
    net,
    donor_state: Mapping[str, Mapping[str, torch.Tensor]],
    alpha: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    donor_cpu = {layer_key: {"u": value["u"].detach().cpu(), "x": value["x"].detach().cpu()} for layer_key, value in donor_state.items()}
    baseline_state = _build_baseline_state_batch_from_template(net=net, template_state=donor_cpu)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_key in LAYER_KEYS:
        out[layer_key] = {
            "u": (1.0 - float(alpha)) * baseline_state[layer_key]["u"] + float(alpha) * donor_cpu[layer_key]["u"],
            "x": (1.0 - float(alpha)) * baseline_state[layer_key]["x"] + float(alpha) * donor_cpu[layer_key]["x"],
        }
    return out


def inject_ux_state_in_place(net, donor_state: Mapping[str, Mapping[str, torch.Tensor]]) -> Dict[str, object]:
    shape_ok = 1
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key)
            u_src = donor_state[layer_key]["u"]
            x_src = donor_state[layer_key]["x"]
            if layer.u_pre is None or layer.x_pre is None or layer.u_pre.shape != u_src.shape or layer.x_pre.shape != x_src.shape:
                shape_ok = 0
                continue
            layer.u_pre.copy_(u_src)
            layer.x_pre.copy_(x_src)
    return {"inject_ux_applied": 1, "inject_ux_shape_ok": int(shape_ok)}


def inject_sham_ux_state_in_place(
    net,
    donor_state: Mapping[str, Mapping[str, torch.Tensor]],
    seed: int,
) -> Dict[str, object]:
    sham_state = _build_sham_state_batch(
        donor_state={layer_key: {"u": value["u"].detach().cpu(), "x": value["x"].detach().cpu()} for layer_key, value in donor_state.items()},
        seed=seed,
    )
    record = inject_ux_state_in_place(net, _move_state_batch_to_device(sham_state, device=next(net.parameters()).device))
    record["sham_applied"] = 1
    return record


def inject_scaled_ux_state_in_place(
    net,
    donor_state: Mapping[str, Mapping[str, torch.Tensor]],
    alpha: float,
) -> Dict[str, object]:
    scaled_state = _build_scaled_state_batch(
        net=net,
        donor_state={layer_key: {"u": value["u"].detach().cpu(), "x": value["x"].detach().cpu()} for layer_key, value in donor_state.items()},
        alpha=alpha,
    )
    record = inject_ux_state_in_place(net, _move_state_batch_to_device(scaled_state, device=next(net.parameters()).device))
    record["alpha"] = float(alpha)
    return record


def _batch_allclose(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> np.ndarray:
    if actual.shape != expected.shape:
        return np.zeros((actual.shape[0],), dtype=np.int64)
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=0.0)
    return close_mask.view(close_mask.shape[0], -1).all(dim=1).to(torch.int64).cpu().numpy()


def _batch_tensor_differs(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> np.ndarray:
    if actual.shape != expected.shape:
        return np.ones((actual.shape[0],), dtype=np.int64)
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=0.0)
    return (~close_mask.view(close_mask.shape[0], -1).all(dim=1)).to(torch.int64).cpu().numpy()


def validate_injection_boundary_state(
    net,
    boundary_pre: Mapping[str, Mapping[str, torch.Tensor]],
    boundary_post: Mapping[str, Mapping[str, torch.Tensor]],
    expected_state: Mapping[str, Mapping[str, torch.Tensor]],
    donor_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    sham_condition: bool = False,
    atol: float = 1e-6,
) -> Dict[str, np.ndarray]:
    batch_size = next(iter(expected_state.values()))["u"].shape[0]
    fast_pre_ok = np.ones((batch_size,), dtype=np.int64)
    fast_post_ok = np.ones((batch_size,), dtype=np.int64)
    ux_match_ok = np.ones((batch_size,), dtype=np.int64)
    sham_applied = np.zeros((batch_size,), dtype=np.int64)
    for layer_key in LAYER_KEYS:
        layer = getattr(net, layer_key)
        pre_layer = boundary_pre[layer_key]
        post_layer = boundary_post[layer_key]
        fast_pre_ok &= _batch_allclose(pre_layer["v_mem"], torch.full_like(pre_layer["v_mem"], layer.V_L), atol=atol)
        fast_pre_ok &= _batch_allclose(pre_layer["g_e"], torch.zeros_like(pre_layer["g_e"]), atol=atol)
        fast_pre_ok &= _batch_allclose(pre_layer["inh_trace"], torch.zeros_like(pre_layer["inh_trace"]), atol=atol)
        fast_pre_ok &= _batch_allclose(pre_layer["res"].to(torch.float32), torch.zeros_like(pre_layer["res"], dtype=torch.float32), atol=0.0)
        fast_post_ok &= _batch_allclose(post_layer["v_mem"], torch.full_like(post_layer["v_mem"], layer.V_L), atol=atol)
        fast_post_ok &= _batch_allclose(post_layer["g_e"], torch.zeros_like(post_layer["g_e"]), atol=atol)
        fast_post_ok &= _batch_allclose(post_layer["inh_trace"], torch.zeros_like(post_layer["inh_trace"]), atol=atol)
        fast_post_ok &= _batch_allclose(post_layer["res"].to(torch.float32), torch.zeros_like(post_layer["res"], dtype=torch.float32), atol=0.0)
        ux_match_ok &= _batch_allclose(post_layer["u"], expected_state[layer_key]["u"], atol=atol)
        ux_match_ok &= _batch_allclose(post_layer["x"], expected_state[layer_key]["x"], atol=atol)
        if sham_condition and donor_state is not None:
            sham_applied |= _batch_tensor_differs(expected_state[layer_key]["u"], donor_state[layer_key]["u"], atol=atol)
            sham_applied |= _batch_tensor_differs(expected_state[layer_key]["x"], donor_state[layer_key]["x"], atol=atol)
    return {
        "injection_faststate_ok": (fast_pre_ok & fast_post_ok).astype(np.int64, copy=False),
        "injection_ux_match_ok": ux_match_ok.astype(np.int64, copy=False),
        "sham_applied": sham_applied.astype(np.int64, copy=False),
    }


def _run_condition_batches(
    net,
    encoder,
    dataset,
    donor_metadata: pd.DataFrame,
    state_bank: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    seed: int,
    condition: str,
    mode: str,
    donor_type_filter: str | None,
    alpha: float | None,
) -> List[Dict[str, object]]:
    if donor_type_filter is None:
        subset = donor_metadata.copy()
    else:
        subset = donor_metadata[donor_metadata["donor_type"] == str(donor_type_filter)].copy()
    subset = subset.sort_values(["trial_id", "donor_type"], kind="stable").reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    if subset.empty:
        return rows
    for start in tqdm(range(0, len(subset), batch_size), desc=f"Readout[{condition}]"):
        batch = subset.iloc[start:start + batch_size].copy().reset_index(drop=True)
        probe_images = _stack_probe_images(dataset=dataset, probe_ids=batch["probe_id"].tolist(), device=device)
        probe_spikes = encode_images(encoder, probe_images, spec.probe_steps)
        donor_batch_cpu = _stack_donor_state_batch(state_bank=state_bank, donor_uids=batch["donor_uid"].tolist())
        baseline_state_cpu = _build_baseline_state_batch_from_template(net=net, template_state=donor_batch_cpu)
        donor_batch_device = _move_state_batch_to_device(donor_batch_cpu, device=device)
        expected_state_cpu: Dict[str, Dict[str, torch.Tensor]]
        intervention_seed = mix_seed(seed, start + 1, len(batch), len(rows) + 1)
        if mode == "baseline":
            expected_state_cpu = baseline_state_cpu

            def before_probe_fn(local_net, _ctx):
                set_fast_state_to_baseline_in_place(local_net)
                set_stsp_to_baseline_in_place(local_net)
                return {"baseline_no_memory_applied": 1}

        elif mode == "direct":
            expected_state_cpu = donor_batch_cpu

            def before_probe_fn(local_net, _ctx):
                set_fast_state_to_baseline_in_place(local_net)
                set_stsp_to_baseline_in_place(local_net)
                record = inject_ux_state_in_place(local_net, donor_batch_device)
                record["donor_injection_applied"] = 1
                return record

        elif mode == "sham":
            expected_state_cpu = _build_sham_state_batch(donor_state=donor_batch_cpu, seed=intervention_seed)
            expected_state_device = _move_state_batch_to_device(expected_state_cpu, device=device)

            def before_probe_fn(local_net, _ctx):
                set_fast_state_to_baseline_in_place(local_net)
                set_stsp_to_baseline_in_place(local_net)
                record = inject_ux_state_in_place(local_net, expected_state_device)
                record["sham_applied"] = 1
                return record

        elif mode == "scaled":
            if alpha is None:
                raise ValueError("scaled mode requires alpha")
            expected_state_cpu = _build_scaled_state_batch(net=net, donor_state=donor_batch_cpu, alpha=float(alpha))
            expected_state_device = _move_state_batch_to_device(expected_state_cpu, device=device)

            def before_probe_fn(local_net, _ctx):
                set_fast_state_to_baseline_in_place(local_net)
                set_stsp_to_baseline_in_place(local_net)
                record = inject_ux_state_in_place(local_net, expected_state_device)
                record["alpha"] = float(alpha)
                return record

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        out = run_monitored_probe_only_rollout(
            net=net,
            probe_spikes=probe_spikes,
            stsp_mode="dynamic",
            phase_reset=True,
            intervention_plan={"before_probe_fn": before_probe_fn},
            record_state_names=(),
        )
        pred = out["predictions"]["prediction_probe"].numpy().astype(np.int64, copy=False)
        fire_t = out["predictions"]["first_fire_t_probe"].numpy().astype(np.int64, copy=False)
        validation = validate_injection_boundary_state(
            net=net,
            boundary_pre=out["boundary_states"]["pre_intervention"],
            boundary_post=out["boundary_states"]["post_intervention"],
            expected_state=expected_state_cpu,
            donor_state=donor_batch_cpu,
            sham_condition=(mode == "sham"),
        )
        for idx_in_batch, row in enumerate(batch.itertuples(index=False)):
            predicted_label = int(pred[idx_in_batch])
            probe_label = int(row.probe_label)
            donor_label = int(row.donor_sample_label)
            rows.append(
                {
                    "trial_id": int(row.trial_id),
                    "donor_uid": str(row.donor_uid),
                    "condition": str(condition),
                    "condition_order": int(CORE_CONDITION_ORDER.index(condition) if condition in CORE_CONDITION_ORDER else len(CORE_CONDITION_ORDER)),
                    "alpha": float(alpha) if alpha is not None else float("nan"),
                    "probe_id": int(row.probe_id),
                    "probe_label": probe_label,
                    "donor_sample_id": int(row.donor_sample_id),
                    "donor_sample_label": donor_label,
                    "donor_type": str(row.donor_type),
                    "label_relation": str(row.label_relation),
                    "predicted_label": predicted_label,
                    "first_fire_t_probe": int(fire_t[idx_in_batch]),
                    "is_correct": int(predicted_label == probe_label),
                    "is_silent": int(predicted_label == -1),
                    "pred_is_probe": int(predicted_label == probe_label),
                    "pred_is_donor": int(predicted_label == donor_label),
                    "pred_is_donor_shifted": int((predicted_label == donor_label) and (donor_label != probe_label)),
                    "donor_matches_probe_label": int(donor_label == probe_label),
                    "donor_matches_original_diagnostic_rule": int(row.donor_matches_original_diagnostic_rule),
                    "diagnostic_overlap_score": float(row.diagnostic_overlap_score),
                    "nondiagnostic_overlap_score": float(row.nondiagnostic_overlap_score),
                    "diagnostic_margin": float(row.diagnostic_margin),
                    "injection_faststate_ok": int(validation["injection_faststate_ok"][idx_in_batch]),
                    "injection_ux_match_ok": int(validation["injection_ux_match_ok"][idx_in_batch]),
                    "sham_applied": int(validation["sham_applied"][idx_in_batch]) if mode == "sham" else 0,
                }
            )
    return rows


def run_recipient_readout_assay(
    net,
    encoder,
    dataset,
    donor_metadata: pd.DataFrame,
    state_bank: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    spec: ExperimentSpec,
    batch_size: int,
    device: torch.device,
    seed: int,
    run_gain_sweep: bool,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    condition_specs = [
        (CONDITION_BASELINE, "baseline", None, None),
        (CONDITION_DIAGNOSTIC, "direct", "diagnostic", None),
        (CONDITION_BASELINE_DONOR, "direct", "baseline", None),
        (CONDITION_NONDIAGNOSTIC, "direct", "nondiagnostic", None),
        (CONDITION_SHAM, "sham", "diagnostic", None),
    ]
    if run_gain_sweep:
        for alpha in GAIN_SWEEP_ALPHAS:
            condition_specs.append((CONDITION_GAIN, "scaled", "diagnostic", float(alpha)))
    for idx, (condition, mode, donor_type, alpha) in enumerate(condition_specs):
        records.extend(
            _run_condition_batches(
                net=net,
                encoder=encoder,
                dataset=dataset,
                donor_metadata=donor_metadata,
                state_bank=state_bank,
                spec=spec,
                batch_size=batch_size,
                device=device,
                seed=mix_seed(seed, idx + 1, 701),
                condition=condition,
                mode=mode,
                donor_type_filter=donor_type,
                alpha=alpha,
            )
        )
    df_trials = pd.DataFrame(records)
    if df_trials.empty:
        raise ValueError("Recipient readout assay produced no trials.")
    return df_trials.sort_values(["condition_order", "condition", "alpha", "trial_id", "donor_type"], kind="stable").reset_index(drop=True)


def _reference_donor_type_for_condition(condition: str) -> str | None:
    if condition in {CONDITION_DIAGNOSTIC, CONDITION_SHAM, CONDITION_GAIN}:
        return "diagnostic"
    if condition == CONDITION_BASELINE_DONOR:
        return "baseline"
    if condition == CONDITION_NONDIAGNOSTIC:
        return "nondiagnostic"
    return None


def _matched_baseline_subset(
    df_trials: pd.DataFrame,
    condition: str,
    alpha: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if condition == CONDITION_BASELINE:
        sub = df_trials[df_trials["condition"] == CONDITION_BASELINE].copy()
        return sub, sub
    ref_donor_type = _reference_donor_type_for_condition(condition)
    base = df_trials[(df_trials["condition"] == CONDITION_BASELINE) & (df_trials["donor_type"] == str(ref_donor_type))].copy()
    target = df_trials[df_trials["condition"] == condition].copy()
    if alpha is not None:
        target = target[np.isclose(target["alpha"].to_numpy(dtype=np.float64), float(alpha), equal_nan=False)].copy()
    return target, base


def _paired_condition_merge(
    df_trials: pd.DataFrame,
    condition: str,
    alpha: float | None = None,
) -> pd.DataFrame:
    target_subset, base_subset = _matched_baseline_subset(df_trials, condition=condition, alpha=alpha)
    if target_subset.empty or base_subset.empty:
        return pd.DataFrame()
    merge_keys = ["trial_id", "probe_id"]
    metric_columns = [
        "predicted_label",
        "is_correct",
        "pred_is_probe",
        "pred_is_donor",
        "pred_is_donor_shifted",
        "is_silent",
    ]
    target = target_subset.reset_index().rename(columns={"index": "row_index"})
    base = base_subset.copy()
    target_cols = ["row_index", "condition", "alpha", "donor_type"] + merge_keys + metric_columns
    base_cols = merge_keys + metric_columns
    return target[target_cols].merge(
        base[base_cols],
        on=merge_keys,
        how="inner",
        suffixes=("_cond", "_base"),
    )


def annotate_probe_support_trial_metrics(df_trials: pd.DataFrame) -> pd.DataFrame:
    df = df_trials.copy()
    df["probe_support_hit"] = df["pred_is_probe"].astype(np.int64, copy=False)
    df["probe_support_gain_vs_baseline"] = np.zeros(len(df), dtype=np.float64)
    df["rescued_probe_trial"] = np.zeros(len(df), dtype=np.int64)
    df["silenced_in_baseline_but_supported_now"] = np.zeros(len(df), dtype=np.int64)
    df["misled_away_from_probe"] = np.zeros(len(df), dtype=np.int64)
    df["support_minus_mislead"] = np.zeros(len(df), dtype=np.int64)
    df["not_silent"] = (1 - df["is_silent"].to_numpy(dtype=np.int64)).astype(np.int64, copy=False)

    grouped = df.groupby(["condition", "alpha"], dropna=False, sort=True)
    for (condition, alpha_value), _subset in grouped:
        if str(condition) == CONDITION_BASELINE:
            continue
        alpha = None if pd.isna(alpha_value) else float(alpha_value)
        merged = _paired_condition_merge(df, condition=str(condition), alpha=alpha)
        if merged.empty:
            continue
        gain = merged["pred_is_probe_cond"].to_numpy(dtype=np.float64) - merged["pred_is_probe_base"].to_numpy(dtype=np.float64)
        rescued = (
            (merged["is_correct_base"].to_numpy(dtype=np.int64) == 0)
            & (merged["pred_is_probe_cond"].to_numpy(dtype=np.int64) == 1)
        ).astype(np.int64, copy=False)
        silence_rescue = (
            (merged["is_silent_base"].to_numpy(dtype=np.int64) == 1)
            & (merged["pred_is_probe_cond"].to_numpy(dtype=np.int64) == 1)
        ).astype(np.int64, copy=False)
        misled = (
            (merged["pred_is_probe_base"].to_numpy(dtype=np.int64) == 1)
            & (merged["pred_is_probe_cond"].to_numpy(dtype=np.int64) == 0)
            & (merged["predicted_label_cond"].to_numpy(dtype=np.int64) != -1)
        ).astype(np.int64, copy=False)
        support_minus_mislead = rescued - misled
        row_index = merged["row_index"].to_numpy(dtype=np.int64)
        df.loc[row_index, "probe_support_gain_vs_baseline"] = gain
        df.loc[row_index, "rescued_probe_trial"] = rescued
        df.loc[row_index, "silenced_in_baseline_but_supported_now"] = silence_rescue
        df.loc[row_index, "misled_away_from_probe"] = misled
        df.loc[row_index, "support_minus_mislead"] = support_minus_mislead
    return df


def build_probe_support_transition_table(df_trials: pd.DataFrame) -> pd.DataFrame:
    merged = _paired_condition_merge(df_trials, condition=CONDITION_DIAGNOSTIC, alpha=None)
    if merged.empty:
        return pd.DataFrame(
            columns=["comparison", "transition", "rate_percent", "count", "n_pairs"]
        )
    pred_is_probe_cond = merged["pred_is_probe_cond"].to_numpy(dtype=np.int64)
    pred_is_probe_base = merged["pred_is_probe_base"].to_numpy(dtype=np.int64)
    is_correct_base = merged["is_correct_base"].to_numpy(dtype=np.int64)
    is_silent_base = merged["is_silent_base"].to_numpy(dtype=np.int64)
    rows = [
        {
            "comparison": f"{CONDITION_DIAGNOSTIC}_vs_{CONDITION_BASELINE}",
            "transition": "rescued",
            "rate_percent": 100.0 * float(((is_correct_base == 0) & (pred_is_probe_cond == 1)).mean()),
            "count": int(((is_correct_base == 0) & (pred_is_probe_cond == 1)).sum()),
            "n_pairs": int(len(merged)),
        },
        {
            "comparison": f"{CONDITION_DIAGNOSTIC}_vs_{CONDITION_BASELINE}",
            "transition": "misled",
            "rate_percent": 100.0
            * float(((pred_is_probe_base == 1) & (pred_is_probe_cond == 0) & (merged["predicted_label_cond"].to_numpy(dtype=np.int64) != -1)).mean()),
            "count": int(((pred_is_probe_base == 1) & (pred_is_probe_cond == 0) & (merged["predicted_label_cond"].to_numpy(dtype=np.int64) != -1)).sum()),
            "n_pairs": int(len(merged)),
        },
        {
            "comparison": f"{CONDITION_DIAGNOSTIC}_vs_{CONDITION_BASELINE}",
            "transition": "silence_rescued",
            "rate_percent": 100.0 * float(((is_silent_base == 1) & (pred_is_probe_cond == 1)).mean()),
            "count": int(((is_silent_base == 1) & (pred_is_probe_cond == 1)).sum()),
            "n_pairs": int(len(merged)),
        },
        {
            "comparison": f"{CONDITION_DIAGNOSTIC}_vs_{CONDITION_BASELINE}",
            "transition": "remained_correct",
            "rate_percent": 100.0 * float(((pred_is_probe_base == 1) & (pred_is_probe_cond == 1)).mean()),
            "count": int(((pred_is_probe_base == 1) & (pred_is_probe_cond == 1)).sum()),
            "n_pairs": int(len(merged)),
        },
    ]
    return pd.DataFrame(rows)


def summarize_condition_metrics(df_trials: pd.DataFrame, num_boot: int, seed: int) -> pd.DataFrame:
    # Diagnostic donors are chosen by overlap with probe-critical evidence, so the primary
    # question is whether injected latent STSP states support probe-consistent readout.
    # Donor prediction / donor shift remain in the outputs only as secondary misleading subtypes.
    rows: List[Dict[str, object]] = []
    grouped = df_trials.groupby(["condition", "alpha"], dropna=False, sort=True)
    for (condition, alpha_value), subset in grouped:
        subset = subset.copy().reset_index(drop=True)
        if subset.empty:
            continue
        alpha = float(alpha_value) if pd.notna(alpha_value) else float("nan")
        target_subset, base_subset = _matched_baseline_subset(df_trials, condition=str(condition), alpha=None if math.isnan(alpha) else alpha)
        probe_prediction_rate = 100.0 * float(subset["pred_is_probe"].mean())
        acc_probe = 100.0 * float(subset["is_correct"].mean())
        donor_prediction_rate = 100.0 * float(subset["pred_is_donor"].mean())
        donor_shift_rate = 100.0 * float(subset["pred_is_donor_shifted"].mean())
        misleading_subset = subset[subset["donor_matches_probe_label"] == 0].copy()
        misleading_rate = float("nan") if misleading_subset.empty else 100.0 * float(misleading_subset["pred_is_donor"].mean())
        rescue_rate = 100.0 * float(subset["rescued_probe_trial"].mean())
        silence_rescue_rate = 100.0 * float(subset["silenced_in_baseline_but_supported_now"].mean())
        mislead_rate = 100.0 * float(subset["misled_away_from_probe"].mean())
        support_vs_mislead_balance = rescue_rate - mislead_rate
        silent_rate = 100.0 * float(subset["is_silent"].mean())
        silence_suppression_rate = 100.0 - silent_rate
        delta_probe = 0.0
        delta_donor = 0.0
        delta_shift = 0.0
        delta_silent = 0.0
        if condition != CONDITION_BASELINE and not target_subset.empty and not base_subset.empty:
            merged = _paired_condition_merge(df_trials, condition=str(condition), alpha=None if math.isnan(alpha) else alpha)
            if not merged.empty:
                delta_probe = 100.0 * float((merged["pred_is_probe_cond"] - merged["pred_is_probe_base"]).mean())
                delta_donor = 100.0 * float((merged["pred_is_donor_cond"] - merged["pred_is_donor_base"]).mean())
                delta_shift = 100.0 * float((merged["pred_is_donor_shifted_cond"] - merged["pred_is_donor_shifted_base"]).mean())
                delta_silent = 100.0 * float((merged["is_silent_cond"] - merged["is_silent_base"]).mean())
        ci_probe_low, ci_probe_high = bootstrap_rate_ci(subset["pred_is_probe"].to_numpy(dtype=np.float64), n_boot=num_boot, seed=mix_seed(seed, 11, len(rows) + 1))
        ci_rescue_low, ci_rescue_high = bootstrap_rate_ci(subset["rescued_probe_trial"].to_numpy(dtype=np.float64), n_boot=num_boot, seed=mix_seed(seed, 12, len(rows) + 1))
        ci_silence_rescue_low, ci_silence_rescue_high = bootstrap_rate_ci(
            subset["silenced_in_baseline_but_supported_now"].to_numpy(dtype=np.float64),
            n_boot=num_boot,
            seed=mix_seed(seed, 13, len(rows) + 1),
        )
        ci_mislead_low, ci_mislead_high = bootstrap_rate_ci(subset["misled_away_from_probe"].to_numpy(dtype=np.float64), n_boot=num_boot, seed=mix_seed(seed, 14, len(rows) + 1))
        ci_silent_low, ci_silent_high = bootstrap_rate_ci(subset["is_silent"].to_numpy(dtype=np.float64), n_boot=num_boot, seed=mix_seed(seed, 15, len(rows) + 1))
        ci_donor_low, ci_donor_high = bootstrap_rate_ci(subset["pred_is_donor"].to_numpy(dtype=np.float64), n_boot=num_boot, seed=mix_seed(seed, 21, len(rows) + 1))
        ci_shift_low, ci_shift_high = bootstrap_rate_ci(subset["pred_is_donor_shifted"].to_numpy(dtype=np.float64), n_boot=num_boot, seed=mix_seed(seed, 31, len(rows) + 1))
        rows.append(
            {
                "condition": str(condition),
                "alpha": alpha,
                "n_trials": int(len(subset)),
                "acc_probe": acc_probe,
                "probe_prediction_rate": probe_prediction_rate,
                "probe_prediction_ci_low": float(ci_probe_low),
                "probe_prediction_ci_high": float(ci_probe_high),
                "delta_vs_baseline_no_memory_probe_prediction_rate": float(delta_probe),
                "rescue_rate": float(rescue_rate),
                "rescue_rate_ci_low": float(ci_rescue_low),
                "rescue_rate_ci_high": float(ci_rescue_high),
                "silence_rescue_rate": float(silence_rescue_rate),
                "silence_rescue_ci_low": float(ci_silence_rescue_low),
                "silence_rescue_ci_high": float(ci_silence_rescue_high),
                "mislead_rate": float(mislead_rate),
                "mislead_rate_ci_low": float(ci_mislead_low),
                "mislead_rate_ci_high": float(ci_mislead_high),
                "support_vs_mislead_balance": float(support_vs_mislead_balance),
                "silent_rate": float(silent_rate),
                "silent_rate_ci_low": float(ci_silent_low),
                "silent_rate_ci_high": float(ci_silent_high),
                "silence_suppression_rate": float(silence_suppression_rate),
                "silence_suppression_ci_low": float(100.0 - ci_silent_high),
                "silence_suppression_ci_high": float(100.0 - ci_silent_low),
                "donor_prediction_rate": donor_prediction_rate,
                "donor_prediction_ci_low": float(ci_donor_low),
                "donor_prediction_ci_high": float(ci_donor_high),
                "donor_shift_rate": donor_shift_rate,
                "donor_shift_ci_low": float(ci_shift_low),
                "donor_shift_ci_high": float(ci_shift_high),
                "misleading_bias_rate": float(misleading_rate),
                "delta_vs_baseline_no_memory_donor_prediction_rate": float(delta_donor),
                "delta_vs_baseline_no_memory_donor_shift_rate": float(delta_shift),
                "delta_vs_baseline_no_memory_silent_rate": float(delta_silent),
                "injection_faststate_ok_rate": 100.0 * float(subset["injection_faststate_ok"].mean()),
                "injection_ux_match_ok_rate": 100.0 * float(subset["injection_ux_match_ok"].mean()),
                "sham_applied_rate": 100.0 * float(subset["sham_applied"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["condition", "alpha"], kind="stable").reset_index(drop=True)
    if not summary.empty and (summary["condition"] == CONDITION_GAIN).any():
        gain = summary[summary["condition"] == CONDITION_GAIN].copy().sort_values("alpha", kind="stable")
        x = gain["alpha"].to_numpy(dtype=np.float64)
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_slope_probe_prediction_rate"] = _slope_from_xy(x=x, y=gain["probe_prediction_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_rank_corr_probe_prediction_rate"] = rank_correlation(x=x, y=gain["probe_prediction_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_slope_rescue_rate"] = _slope_from_xy(x=x, y=gain["rescue_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_rank_corr_rescue_rate"] = rank_correlation(x=x, y=gain["rescue_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_slope_silence_suppression_rate"] = _slope_from_xy(x=x, y=gain["silence_suppression_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_rank_corr_silence_suppression_rate"] = rank_correlation(x=x, y=gain["silence_suppression_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_slope_donor_prediction_rate"] = _slope_from_xy(x=x, y=gain["donor_prediction_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_rank_corr_donor_prediction_rate"] = rank_correlation(x=x, y=gain["donor_prediction_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_slope_donor_shift_rate"] = _slope_from_xy(x=x, y=gain["donor_shift_rate"].to_numpy(dtype=np.float64))
        summary.loc[summary["condition"] == CONDITION_GAIN, "gain_sweep_rank_corr_donor_shift_rate"] = rank_correlation(x=x, y=gain["donor_shift_rate"].to_numpy(dtype=np.float64))
    return summary


def _paired_contrast(
    df_trials: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    metric: str,
    n_boot: int,
    seed: int,
    donor_type_a: str | None = None,
    donor_type_b: str | None = None,
    alpha_a: float | None = None,
    alpha_b: float | None = None,
    comparison_label: str | None = None,
) -> Dict[str, object]:
    sub_a = df_trials[df_trials["condition"] == str(condition_a)].copy()
    sub_b = df_trials[df_trials["condition"] == str(condition_b)].copy()
    if donor_type_a is not None:
        sub_a = sub_a[sub_a["donor_type"] == str(donor_type_a)].copy()
    if donor_type_b is not None:
        sub_b = sub_b[sub_b["donor_type"] == str(donor_type_b)].copy()
    if alpha_a is not None:
        sub_a = sub_a[np.isclose(sub_a["alpha"].to_numpy(dtype=np.float64), float(alpha_a), equal_nan=False)].copy()
    if alpha_b is not None:
        sub_b = sub_b[np.isclose(sub_b["alpha"].to_numpy(dtype=np.float64), float(alpha_b), equal_nan=False)].copy()
    merged = sub_a[["trial_id", "probe_id", metric]].merge(
        sub_b[["trial_id", "probe_id", metric]],
        on=["trial_id", "probe_id"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    boot = paired_bootstrap_diff_summary(
        merged[f"{metric}_a"].to_numpy(dtype=np.float64),
        merged[f"{metric}_b"].to_numpy(dtype=np.float64),
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "comparison": str(comparison_label) if comparison_label is not None else f"{condition_a}_minus_{condition_b}",
        "metric": str(metric),
        "observed_diff_pp": float(boot["observed_diff_pp"]),
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
        "n_pairs": int(boot["n_pairs"]),
    }


def _bootstrap_gain_monotonicity(
    df_trials: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    gain = df_trials[df_trials["condition"] == CONDITION_GAIN].copy()
    pivot = gain.pivot_table(index="trial_id", columns="alpha", values=metric, aggfunc="first").sort_index(axis=1)
    alpha_values = pivot.columns.to_numpy(dtype=np.float64)
    matrix = pivot.to_numpy(dtype=np.float64)
    if matrix.shape[0] == 0:
        empty = {"comparison": "gain_sweep_monotonicity", "metric": f"{metric}_slope", "observed_diff_pp": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_pairs": 0}
        empty_corr = dict(empty)
        empty_corr["metric"] = f"{metric}_rank_corr"
        return empty, empty_corr
    slope_obs = _slope_from_xy(alpha_values, 100.0 * matrix.mean(axis=0))
    corr_obs = rank_correlation(alpha_values, 100.0 * matrix.mean(axis=0))
    rng = np.random.default_rng(seed)
    boot_slope = np.zeros(n_boot, dtype=np.float64)
    boot_corr = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        mean_curve = 100.0 * matrix[sample_idx].mean(axis=0)
        boot_slope[idx] = _slope_from_xy(alpha_values, mean_curve)
        boot_corr[idx] = rank_correlation(alpha_values, mean_curve)
    return (
        {
            "comparison": "gain_sweep_monotonicity",
            "metric": f"{metric}_slope",
            "observed_diff_pp": float(slope_obs),
            "ci_low": float(np.percentile(boot_slope, 2.5)),
            "ci_high": float(np.percentile(boot_slope, 97.5)),
            "n_pairs": int(matrix.shape[0]),
        },
        {
            "comparison": "gain_sweep_monotonicity",
            "metric": f"{metric}_rank_corr",
            "observed_diff_pp": float(corr_obs),
            "ci_low": float(np.percentile(boot_corr, 2.5)),
            "ci_high": float(np.percentile(boot_corr, 97.5)),
            "n_pairs": int(matrix.shape[0]),
        },
    )


def build_bootstrap_contrasts(df_trials: pd.DataFrame, n_boot: int, seed: int, run_gain_sweep: bool) -> pd.DataFrame:
    rows = [
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_BASELINE, "pred_is_probe", n_boot, mix_seed(seed, 101), donor_type_a="diagnostic", donor_type_b="diagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_BASELINE, "is_correct", n_boot, mix_seed(seed, 102), donor_type_a="diagnostic", donor_type_b="diagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_BASELINE_DONOR, "pred_is_probe", n_boot, mix_seed(seed, 111), donor_type_a="diagnostic", donor_type_b="baseline"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_BASELINE_DONOR, "rescued_probe_trial", n_boot, mix_seed(seed, 112), donor_type_a="diagnostic", donor_type_b="baseline"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_NONDIAGNOSTIC, "pred_is_probe", n_boot, mix_seed(seed, 121), donor_type_a="diagnostic", donor_type_b="nondiagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_NONDIAGNOSTIC, "support_minus_mislead", n_boot, mix_seed(seed, 122), donor_type_a="diagnostic", donor_type_b="nondiagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_SHAM, "pred_is_probe", n_boot, mix_seed(seed, 131), donor_type_a="diagnostic", donor_type_b="diagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_SHAM, "rescued_probe_trial", n_boot, mix_seed(seed, 132), donor_type_a="diagnostic", donor_type_b="diagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_BASELINE, "pred_is_donor", n_boot, mix_seed(seed, 191), donor_type_a="diagnostic", donor_type_b="diagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_NONDIAGNOSTIC, "pred_is_donor_shifted", n_boot, mix_seed(seed, 192), donor_type_a="diagnostic", donor_type_b="nondiagnostic"),
        _paired_contrast(df_trials, CONDITION_DIAGNOSTIC, CONDITION_SHAM, "pred_is_donor_shifted", n_boot, mix_seed(seed, 193), donor_type_a="diagnostic", donor_type_b="diagnostic"),
    ]
    if run_gain_sweep:
        gain_comparison = f"{CONDITION_GAIN}(alpha=1)_minus(alpha=0)"
        rows.append(_paired_contrast(df_trials, CONDITION_GAIN, CONDITION_GAIN, "pred_is_probe", n_boot, mix_seed(seed, 141), alpha_a=1.0, alpha_b=0.0, comparison_label=gain_comparison))
        rows.append(_paired_contrast(df_trials, CONDITION_GAIN, CONDITION_GAIN, "rescued_probe_trial", n_boot, mix_seed(seed, 142), alpha_a=1.0, alpha_b=0.0, comparison_label=gain_comparison))
        rows.append(_paired_contrast(df_trials, CONDITION_GAIN, CONDITION_GAIN, "not_silent", n_boot, mix_seed(seed, 143), alpha_a=1.0, alpha_b=0.0, comparison_label=gain_comparison))
        rows.append(_paired_contrast(df_trials, CONDITION_GAIN, CONDITION_GAIN, "pred_is_donor_shifted", n_boot, mix_seed(seed, 144), alpha_a=1.0, alpha_b=0.0, comparison_label=gain_comparison))
        rows.extend(_bootstrap_gain_monotonicity(df_trials=df_trials, metric="pred_is_probe", n_boot=n_boot, seed=mix_seed(seed, 151)))
        rows.extend(_bootstrap_gain_monotonicity(df_trials=df_trials, metric="rescued_probe_trial", n_boot=n_boot, seed=mix_seed(seed, 152)))
        rows.extend(_bootstrap_gain_monotonicity(df_trials=df_trials, metric="not_silent", n_boot=n_boot, seed=mix_seed(seed, 153)))
        rows.extend(_bootstrap_gain_monotonicity(df_trials=df_trials, metric="pred_is_donor_shifted", n_boot=n_boot, seed=mix_seed(seed, 154)))
        rows.extend(_bootstrap_gain_monotonicity(df_trials=df_trials, metric="pred_is_donor", n_boot=n_boot, seed=mix_seed(seed, 155)))
    return pd.DataFrame(rows).sort_values(["comparison", "metric"], kind="stable").reset_index(drop=True)


def build_error_destination_table(df_trials: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    condition_order = list(CORE_CONDITION_ORDER)
    if (df_trials["condition"] == CONDITION_GAIN).any():
        condition_order.append(CONDITION_GAIN)
    for condition in condition_order:
        subset = df_trials[df_trials["condition"] == condition].copy()
        if condition == CONDITION_GAIN:
            subset = subset[np.isclose(subset["alpha"].to_numpy(dtype=np.float64), 1.0, equal_nan=False)].copy()
        err = subset[subset["predicted_label"] != subset["probe_label"]].copy()
        rows.extend(
            [
                {"condition": condition, "destination": "probe", "rate_percent": 100.0 * float((err["predicted_label"] == err["probe_label"]).mean()) if len(err) > 0 else 0.0, "n_error": int(len(err))},
                {"condition": condition, "destination": "donor", "rate_percent": 100.0 * float((err["predicted_label"] == err["donor_sample_label"]).mean()) if len(err) > 0 else 0.0, "n_error": int(len(err))},
                {"condition": condition, "destination": "silent", "rate_percent": 100.0 * float((err["predicted_label"] == -1).mean()) if len(err) > 0 else 0.0, "n_error": int(len(err))},
                {
                    "condition": condition,
                    "destination": "other",
                    "rate_percent": 100.0
                    * float(((err["predicted_label"] >= 0) & (err["predicted_label"] != err["probe_label"]) & (err["predicted_label"] != err["donor_sample_label"])).mean())
                    if len(err) > 0
                    else 0.0,
                    "n_error": int(len(err)),
                },
            ]
        )
        if len(err) > 0:
            total = sum(float(row["rate_percent"]) for row in rows[-4:])
            rows[-1]["rate_percent"] = float(rows[-1]["rate_percent"]) + (100.0 - total)
    return pd.DataFrame(rows).sort_values(["condition", "destination"], kind="stable").reset_index(drop=True)


def build_interpretation_table(df_bootstrap: pd.DataFrame) -> pd.DataFrame:
    def pick(comparison: str, metric: str) -> pd.Series | None:
        subset = df_bootstrap[(df_bootstrap["comparison"] == comparison) & (df_bootstrap["metric"] == metric)]
        if subset.empty:
            return None
        return subset.iloc[0]

    def choose(comparison: str, metrics: Sequence[str]) -> pd.Series | None:
        candidates = [pick(comparison, metric) for metric in metrics]
        for candidate in candidates:
            if candidate is not None and np.isfinite(float(candidate["ci_low"])) and float(candidate["ci_low"]) > 0.0:
                return candidate
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    rows: List[Dict[str, object]] = []
    row_a = choose(f"{CONDITION_DIAGNOSTIC}_minus_{CONDITION_BASELINE}", ("pred_is_probe", "is_correct"))
    row_b = choose(f"{CONDITION_DIAGNOSTIC}_minus_{CONDITION_SHAM}", ("pred_is_probe", "rescued_probe_trial"))
    row_c = choose(f"{CONDITION_DIAGNOSTIC}_minus_{CONDITION_NONDIAGNOSTIC}", ("support_minus_mislead", "pred_is_probe"))
    gain_row = choose("gain_sweep_monotonicity", ("pred_is_probe_slope", "rescued_probe_trial_slope", "not_silent_slope"))
    rows.append({"claim": "diagnostic_probe_support_sufficiency", "supported": int(row_a is not None and float(row_a["ci_low"]) > 0.0), "comparison": f"{CONDITION_DIAGNOSTIC}_minus_{CONDITION_BASELINE}", "metric": str(row_a["metric"]) if row_a is not None else "pred_is_probe", "observed_diff_pp": float(row_a["observed_diff_pp"]) if row_a is not None else float("nan"), "ci_low": float(row_a["ci_low"]) if row_a is not None else float("nan"), "ci_high": float(row_a["ci_high"]) if row_a is not None else float("nan")})
    rows.append({"claim": "structured_supportive_state_not_arbitrary_injection", "supported": int(row_b is not None and float(row_b["ci_low"]) > 0.0), "comparison": f"{CONDITION_DIAGNOSTIC}_minus_{CONDITION_SHAM}", "metric": str(row_b["metric"]) if row_b is not None else "pred_is_probe", "observed_diff_pp": float(row_b["observed_diff_pp"]) if row_b is not None else float("nan"), "ci_low": float(row_b["ci_low"]) if row_b is not None else float("nan"), "ci_high": float(row_b["ci_high"]) if row_b is not None else float("nan")})
    rows.append({"claim": "diagnostic_feature_specific_support", "supported": int(row_c is not None and float(row_c["ci_low"]) > 0.0), "comparison": f"{CONDITION_DIAGNOSTIC}_minus_{CONDITION_NONDIAGNOSTIC}", "metric": str(row_c["metric"]) if row_c is not None else "support_minus_mislead", "observed_diff_pp": float(row_c["observed_diff_pp"]) if row_c is not None else float("nan"), "ci_low": float(row_c["ci_low"]) if row_c is not None else float("nan"), "ci_high": float(row_c["ci_high"]) if row_c is not None else float("nan")})
    rows.append({"claim": "continuous_probe_support_controllability", "supported": int(gain_row is not None and float(gain_row["ci_low"]) > 0.0), "comparison": "gain_sweep_monotonicity", "metric": str(gain_row["metric"]) if gain_row is not None else "pred_is_probe_slope", "observed_diff_pp": float(gain_row["observed_diff_pp"]) if gain_row is not None else float("nan"), "ci_low": float(gain_row["ci_low"]) if gain_row is not None else float("nan"), "ci_high": float(gain_row["ci_high"]) if gain_row is not None else float("nan")})
    return pd.DataFrame(rows)


def make_figure_sufficiency_summary(df_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    summary = df_summary[df_summary["condition"].isin(CORE_CONDITION_ORDER)].copy()
    summary = summary.groupby("condition", as_index=False).first().set_index("condition").reindex(CORE_CONDITION_ORDER).reset_index()
    metrics = [
        ("probe_prediction_rate", "Probe support"),
        ("rescue_rate", "Rescue"),
        ("support_vs_mislead_balance", "Support-mislead"),
    ]
    x = np.arange(len(summary), dtype=np.float64)
    width = 0.24
    offsets = np.linspace(-width, width, num=len(metrics))
    colors = ["#4C4C4C", CONDITION_COLORS[CONDITION_DIAGNOSTIC], "#E69F00"]
    for idx, (metric, label) in enumerate(metrics):
        ax.bar(x + offsets[idx], summary[metric].to_numpy(dtype=np.float64), width=width, label=label, color=colors[idx], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(list(CORE_CONDITION_ORDER), rotation=20, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Probe-support sufficiency summary")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def make_figure_error_destination(df_error: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    plot_df = df_error[df_error["condition"].isin(CORE_CONDITION_ORDER)].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=list(CORE_CONDITION_ORDER), ordered=True)
    pivot = plot_df.pivot_table(index="condition", columns="destination", values="rate_percent", aggfunc="first").reindex(CORE_CONDITION_ORDER)
    colors = {"probe": "#888888", "donor": CONDITION_COLORS[CONDITION_DIAGNOSTIC], "silent": "#BBBBBB", "other": "#E69F00"}
    bottom = np.zeros(len(pivot), dtype=np.float64)
    for destination in ["probe", "donor", "silent", "other"]:
        values = pivot[destination].to_numpy(dtype=np.float64) if destination in pivot.columns else np.zeros(len(pivot))
        ax.bar(pivot.index.tolist(), values, bottom=bottom, color=colors[destination], label=destination)
        bottom += values
    ax.set_ylabel("Secondary error destination (%)")
    ax.set_title("Misleading side-effect destinations after injection")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def make_figure_gain_sweep(df_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    gain = df_summary[df_summary["condition"] == CONDITION_GAIN].copy().sort_values("alpha", kind="stable")
    x = gain["alpha"].to_numpy(dtype=np.float64)
    probe_support = gain["probe_prediction_rate"].to_numpy(dtype=np.float64)
    rescue = gain["rescue_rate"].to_numpy(dtype=np.float64)
    silence_suppression = gain["silence_suppression_rate"].to_numpy(dtype=np.float64)
    probe_support_err = _errorbar_from_ci(probe_support, gain["probe_prediction_ci_low"].to_numpy(dtype=np.float64), gain["probe_prediction_ci_high"].to_numpy(dtype=np.float64))
    rescue_err = _errorbar_from_ci(rescue, gain["rescue_rate_ci_low"].to_numpy(dtype=np.float64), gain["rescue_rate_ci_high"].to_numpy(dtype=np.float64))
    silence_suppression_err = _errorbar_from_ci(
        silence_suppression,
        gain["silence_suppression_ci_low"].to_numpy(dtype=np.float64),
        gain["silence_suppression_ci_high"].to_numpy(dtype=np.float64),
    )
    ax.errorbar(x, probe_support, yerr=probe_support_err, marker="o", linewidth=2.0, color=CONDITION_COLORS[CONDITION_DIAGNOSTIC], label="probe support")
    ax.errorbar(x, rescue, yerr=rescue_err, marker="s", linewidth=2.0, color="#E69F00", label="rescue")
    ax.errorbar(x, silence_suppression, yerr=silence_suppression_err, marker="^", linewidth=2.0, color=CONDITION_COLORS[CONDITION_BASELINE_DONOR], label="silence suppression")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Probe-support gain sweep")
    ax.grid(alpha=0.2)
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def make_figure_margin_scatter(df_trials: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    subset = df_trials[df_trials["condition"] == CONDITION_DIAGNOSTIC].copy()
    x = subset["diagnostic_margin"].to_numpy(dtype=np.float64)
    y = 100.0 * subset["probe_support_hit"].to_numpy(dtype=np.float64)
    ax.scatter(x, y, s=32, alpha=0.75, color=CONDITION_COLORS[CONDITION_DIAGNOSTIC])
    slope = _slope_from_xy(x, y)
    corr = rank_correlation(x, y)
    if len(subset) >= 2 and np.isfinite(slope):
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        y_line = float(np.nanmean(y)) + float(slope) * (x_line - float(np.nanmean(x)))
        ax.plot(x_line, y_line, color="#333333", linewidth=2.0)
    ax.set_xlabel("Diagnostic margin")
    ax.set_ylabel("Probe-support hit (%)")
    ax.set_title("Diagnostic margin vs probe-support readout")
    ax.grid(alpha=0.2)
    ax.text(0.02, 0.98, f"slope={slope:.3f}\nrank r={corr:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    fig.tight_layout()
    return fig


def make_figure_support_transition(df_transition: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    plot_df = df_transition.copy()
    order = ["rescued", "misled", "silence_rescued", "remained_correct"]
    plot_df["transition"] = pd.Categorical(plot_df["transition"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("transition", kind="stable")
    colors = {
        "rescued": CONDITION_COLORS[CONDITION_DIAGNOSTIC],
        "misled": CONDITION_COLORS[CONDITION_NONDIAGNOSTIC],
        "silence_rescued": CONDITION_COLORS[CONDITION_BASELINE_DONOR],
        "remained_correct": "#666666",
    }
    ax.bar(
        plot_df["transition"].astype(str).tolist(),
        plot_df["rate_percent"].to_numpy(dtype=np.float64),
        color=[colors[str(name)] for name in plot_df["transition"].astype(str).tolist()],
        alpha=0.9,
    )
    ax.set_ylabel("Rate (%)")
    ax.set_title("Baseline-to-diagnostic support transitions")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def make_figure_sanity_checks(df_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    plot_df = df_summary[df_summary["condition"].isin(list(CORE_CONDITION_ORDER) + [CONDITION_GAIN])].copy()
    if (plot_df["condition"] == CONDITION_GAIN).any():
        plot_df = plot_df[(plot_df["condition"] != CONDITION_GAIN) | np.isclose(plot_df["alpha"].to_numpy(dtype=np.float64), 1.0, equal_nan=False)].copy()
    plot_df = plot_df.groupby("condition", as_index=False).first().set_index("condition").reindex(list(CORE_CONDITION_ORDER) + [CONDITION_GAIN]).dropna(how="all")
    matrix = np.vstack([plot_df["injection_faststate_ok_rate"].to_numpy(dtype=np.float64), plot_df["injection_ux_match_ok_rate"].to_numpy(dtype=np.float64)])
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=100.0, cmap="viridis")
    ax.set_xticks(np.arange(len(plot_df.index)))
    ax.set_xticklabels(plot_df.index.tolist(), rotation=20, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fast-state ok", "u/x match ok"])
    ax.set_title("Injection sanity checks")
    fig.colorbar(im, ax=ax, label="Pass rate (%)")
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic latent STSP probe-support sufficiency assay.")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--trial-count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=500.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--num-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-gain-sweep", type=_parse_bool, default=True)
    parser.add_argument("--stage", type=str, default="all", choices=["all", "harvest", "readout"])
    parser.add_argument("--donor-bank-path", type=str, default="")
    parser.add_argument("--cache-diagnostic-regions", action="store_true")
    return parser


def _default_donor_bank_path(save_dir: Path) -> Path:
    return save_dir / "donor_state_bank.pt"


def _print_interpretation(df_interpretation: pd.DataFrame) -> None:
    for row in df_interpretation.itertuples(index=False):
        verdict = "SUPPORTED" if int(row.supported) == 1 else "not_supported"
        print(f"[Summary] {row.claim}: {verdict} | {row.comparison} | {row.metric} | obs={float(row.observed_diff_pp):.3f}, ci=[{float(row.ci_low):.3f}, {float(row.ci_high):.3f}]")


def main() -> None:
    args = build_argparser().parse_args()
    if args.trial_count <= 0 or args.batch_size <= 0 or args.num_boot <= 0:
        raise ValueError("trial-count, batch-size, and num-boot must be positive.")
    if args.sample_ms <= 0 or args.delay_ms < 0 or args.probe_ms <= 0 or args.dt_ms <= 0:
        raise ValueError("sample/delay/probe/dt values are invalid.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    spec = ExperimentSpec(dt=float(args.dt_ms * ms), sample_ms=float(args.sample_ms), delay_ms=float(args.delay_ms), probe_ms=float(args.probe_ms))
    if spec.sample_steps <= 0 or spec.probe_steps <= 0:
        raise ValueError("sample/probe duration must resolve to positive steps.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    donor_bank_path = Path(args.donor_bank_path) if args.donor_bank_path else _default_donor_bank_path(save_dir)
    baseline_batch_size = max(int(args.batch_size), 128)

    net, encoder = load_model_and_encoder(model_path=args.model_path, device=device, dt=spec.dt, max_duration_ms=max(spec.sample_ms, spec.delay_ms, spec.probe_ms))
    _, _, test_loader = build_mnist_skeleton_loader(root=args.dataset_root, batch_size=1, input_size=28, num_workers=0)
    dataset = test_loader.dataset
    raw_images, dataset_labels, image_matrix_flat = build_dataset_arrays(dataset)

    donor_metadata: pd.DataFrame
    state_bank: Dict[str, Mapping[str, Mapping[str, torch.Tensor]]]
    if args.stage in {"all", "harvest"}:
        probe_region_summary, diagnostic_region_table, mask_lookup = estimate_diagnostic_regions(
            net=net,
            encoder=encoder,
            raw_images=raw_images,
            dataset_labels=dataset_labels,
            spec=spec,
            trial_count=args.trial_count,
            patch_size=DEFAULT_PATCH_SIZE,
            diagnostic_method="occlusion",
            delay_values_ms=[int(args.delay_ms)],
            batch_size=args.batch_size,
            baseline_batch_size=baseline_batch_size,
            device=device,
            seed=args.seed,
            save_dir=save_dir,
            cache_diagnostic_regions=args.cache_diagnostic_regions,
            probe_pool_limit=DEFAULT_PROBE_POOL_LIMIT,
            probe_pool_per_class=DEFAULT_PROBE_POOL_PER_CLASS,
            early_stop_multiplier=DEFAULT_BASELINE_EARLY_STOP_MULTIPLIER,
        )
        selection_df, donor_candidates = build_donor_selection_table(
            probe_region_summary=probe_region_summary,
            mask_lookup=mask_lookup,
            image_matrix_flat=image_matrix_flat,
            dataset_labels=dataset_labels,
        )
        harvested = harvest_donor_state_bank(net=net, encoder=encoder, dataset=dataset, donor_metadata=donor_candidates, spec=spec, batch_size=args.batch_size, device=device, seed=args.seed)
        donor_metadata = harvested["metadata"]
        state_bank = harvested["state_bank"]
        torch.save({"config": {"seed": int(args.seed), "sample_ms": float(args.sample_ms), "delay_ms": float(args.delay_ms), "probe_ms": float(args.probe_ms), "dt_ms": float(args.dt_ms)}, "metadata": donor_metadata.to_dict("records"), "state_bank": state_bank}, donor_bank_path)
        save_tidy_csv(donor_metadata, save_dir / "donor_state_metadata.csv", sort_by=["trial_id", "donor_type"])
        if not selection_df.empty:
            save_tidy_csv(selection_df, save_dir / "probe_anchor_selection.csv", sort_by=["probe_id"])
        if not diagnostic_region_table.empty:
            save_tidy_csv(diagnostic_region_table, save_dir / "diagnostic_region_table.csv", sort_by=["probe_id", "patch_id"])
        print(f"[Done] Saved donor bank: {donor_bank_path}")
        if args.stage == "harvest":
            save_run_config({"stage": str(args.stage), "model_path": str(args.model_path), "dataset_root": str(args.dataset_root), "save_dir": str(save_dir), "donor_bank_path": str(donor_bank_path), "seed": int(args.seed), "device": str(device), "trial_count": int(args.trial_count), "batch_size": int(args.batch_size), "baseline_batch_size": int(baseline_batch_size), "sample_ms": float(args.sample_ms), "delay_ms": float(args.delay_ms), "probe_ms": float(args.probe_ms), "dt_ms": float(args.dt_ms)}, save_dir)
            return
    else:
        if not donor_bank_path.exists():
            raise FileNotFoundError(f"Donor bank not found for readout stage: {donor_bank_path}")
        payload = torch.load(donor_bank_path, map_location="cpu")
        donor_metadata = pd.DataFrame(payload["metadata"]).sort_values(["trial_id", "donor_type"], kind="stable").reset_index(drop=True)
        state_bank = payload["state_bank"]

    df_trials = run_recipient_readout_assay(net=net, encoder=encoder, dataset=dataset, donor_metadata=donor_metadata, state_bank=state_bank, spec=spec, batch_size=args.batch_size, device=device, seed=args.seed, run_gain_sweep=bool(args.run_gain_sweep))
    df_trials = annotate_probe_support_trial_metrics(df_trials)
    df_summary = summarize_condition_metrics(df_trials=df_trials, num_boot=args.num_boot, seed=args.seed + 2000)
    df_bootstrap = build_bootstrap_contrasts(df_trials=df_trials, n_boot=args.num_boot, seed=args.seed + 3000, run_gain_sweep=bool(args.run_gain_sweep))
    df_error = build_error_destination_table(df_trials)
    df_transition = build_probe_support_transition_table(df_trials)
    df_interpretation = build_interpretation_table(df_bootstrap)

    trial_csv = save_tidy_csv(df_trials, save_dir / "trial_level_injection_results.csv", sort_by=["condition", "alpha", "trial_id"])
    donor_csv = save_tidy_csv(donor_metadata, save_dir / "donor_state_metadata.csv", sort_by=["trial_id", "donor_type"])
    summary_csv = save_tidy_csv(df_summary, save_dir / "condition_summary.csv", sort_by=["condition", "alpha"])
    bootstrap_csv = save_tidy_csv(df_bootstrap, save_dir / "bootstrap_contrasts.csv", sort_by=["comparison", "metric"])
    interpretation_csv = save_tidy_csv(df_interpretation, save_dir / "interpretation_summary.csv", sort_by=["claim"])

    fig1_paths = save_figure_all_formats(make_figure_sufficiency_summary(df_summary), save_dir / "figure_sufficiency_summary")
    plt.close("all")
    fig2_paths = save_figure_all_formats(make_figure_error_destination(df_error), save_dir / "figure_error_destination")
    plt.close("all")
    fig3_paths = save_figure_all_formats(make_figure_gain_sweep(df_summary), save_dir / "figure_gain_sweep")
    plt.close("all")
    fig4_paths = save_figure_all_formats(make_figure_margin_scatter(df_trials), save_dir / "figure_margin_scatter")
    plt.close("all")
    fig5_paths = save_figure_all_formats(make_figure_sanity_checks(df_summary), save_dir / "figure_sanity_checks")
    plt.close("all")
    fig6_paths = {}
    if not df_transition.empty:
        transition_csv = save_tidy_csv(df_transition, save_dir / "support_transition_summary.csv", sort_by=["comparison", "transition"])
        fig6_paths = save_figure_all_formats(make_figure_support_transition(df_transition), save_dir / "figure_support_transition")
        plt.close("all")
    else:
        transition_csv = None

    run_config = save_run_config(
        {
            "model_path": str(args.model_path),
            "dataset_root": str(args.dataset_root),
            "save_dir": str(save_dir),
            "donor_bank_path": str(donor_bank_path),
            "stage": str(args.stage),
            "seed": int(args.seed),
            "device": str(device),
            "trial_count": int(args.trial_count),
            "batch_size": int(args.batch_size),
            "baseline_batch_size": int(baseline_batch_size),
            "sample_ms": float(args.sample_ms),
            "delay_ms": float(args.delay_ms),
            "probe_ms": float(args.probe_ms),
            "dt_ms": float(args.dt_ms),
            "run_gain_sweep": bool(args.run_gain_sweep),
            "output_files": {
                "trial_level_injection_results_csv": str(trial_csv),
                "donor_state_metadata_csv": str(donor_csv),
                "condition_summary_csv": str(summary_csv),
                "bootstrap_contrasts_csv": str(bootstrap_csv),
                "interpretation_summary_csv": str(interpretation_csv),
                "figure_sufficiency_summary": fig1_paths,
                "figure_error_destination": fig2_paths,
                "figure_gain_sweep": fig3_paths,
                "figure_margin_scatter": fig4_paths,
                "figure_sanity_checks": fig5_paths,
                "support_transition_summary_csv": str(transition_csv) if transition_csv is not None else "",
                "figure_support_transition": fig6_paths,
            },
        },
        save_dir,
    )

    _print_interpretation(df_interpretation)
    print(f"[Done] Saved: {trial_csv}")
    print(f"[Done] Saved: {donor_csv}")
    print(f"[Done] Saved: {summary_csv}")
    print(f"[Done] Saved: {bootstrap_csv}")
    print(f"[Done] Saved: {interpretation_csv}")
    if transition_csv is not None:
        print(f"[Done] Saved: {transition_csv}")
    print(f"[Done] Saved: {run_config}")


if __name__ == "__main__":
    main()


# This assay tests causal sufficiency by holding the probe constant and directly injecting naturally harvested latent STSP states into a baseline recipient network.
# Because diagnostic donors are selected based on overlap with probe-critical evidence, this assay is interpreted as testing probe-support sufficiency rather than donor-label takeover.
