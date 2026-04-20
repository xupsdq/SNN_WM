from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

try:
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    HAS_STATSMODELS = True
except ModuleNotFoundError:
    smf = None
    ConvergenceWarning = Warning
    HAS_STATSMODELS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.units import ms
from src.experiments.common.dataset import build_class_index
from src.experiments.common.json_io import save_json_payload as _save_json
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout, save_log_lines, save_summary_json
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.seed import mix_seed
from src.experiments.common.voltage_readout import resolve_readout_step
from src.experiments.distractor.shared.analysis import (
    compute_delta_v,
    compute_reference_direction_from_delays,
    compute_strength_metrics,
)
from src.experiments.distractor.shared.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DELAY1_MS,
    DEFAULT_DELAY_SWEEP_MS,
    DEFAULT_DILATION_RADIUS,
    DEFAULT_DIRECTION_DELAY_MS,
    DEFAULT_DISTRACTOR_MS,
    DEFAULT_FOREGROUND_THRESHOLD,
    DEFAULT_MAX_PROBES,
    DEFAULT_MAX_TRIPLETS,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_CONTROL_CANDIDATES,
    DEFAULT_NUM_SIM_BINS,
    DEFAULT_PROBE_MS,
    DEFAULT_SAMPLE_MS,
    DEFAULT_SAMPLES_PER_PROBE,
    DEFAULT_TAU_MS,
    EPS,
    ExperimentSpec,
    sanitize_delay_sweep as _sanitize_delay_sweep,
    sem as _sem,
    validate_positive as _validate_positive,
)
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
    ALPHA_SCATTER,
    DISTRACTOR_CONTROL_CONDITION_COLORS,
    DISTRACTOR_MAIN_CONDITION_COLORS,
    FIGSIZE_SINGLE_PANEL_MEDIUM,
    FIGSIZE_SINGLE_PANEL_TALL,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    MARKER_CIRCLE,
    apply_standard_legend,
    horizontal_panel_figsize,
)

DEFAULT_OUTPUT_DIR = "results/distractor_strength_effect_independence_experiment"

MAIN_CONDITION_ORDER: tuple[str, ...] = (
    "full_dynamic",
    "sample_remove_SPonly",
    "distractor_remove_DPonly",
    "sample_remove_SDP",
    "distractor_remove_SDP",
    "both_remove_SDP",
)
CONTROL_CONDITION_ORDER: tuple[str, ...] = (
    "sample_remove_SPonly_control",
    "distractor_remove_DPonly_control",
    "sample_remove_SDP_control",
    "distractor_remove_SDP_control",
    "both_remove_SDP_control",
)
CONDITION_COLORS: dict[str, str] = {
    **DISTRACTOR_MAIN_CONDITION_COLORS,
    **DISTRACTOR_CONTROL_CONDITION_COLORS,
}


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    stsp_mode: str
    sample_mask_key: str | None
    distractor_mask_key: str | None


@dataclass
class ModelFitBundle:
    status: str
    formula: str
    delay_modeling_mode: str
    model_type: str | None
    fallback_used: bool
    fallback_reason: str | None
    result: object | None
    companion_ols: object | None
    aic: float | None
    bic: float | None
    warnings: list[str]


@dataclass
class InternalWaldResult:
    statistic: float
    pvalue: float
    df_num: int
    df_denom: int | None


class InternalModel:
    def __init__(self, exog_names: Sequence[str]) -> None:
        self.exog_names = list(exog_names)


class InternalOLSResult:
    def __init__(
        self,
        *,
        formula: str,
        exog_names: Sequence[str],
        params: np.ndarray,
        cov_params: np.ndarray,
        resid: np.ndarray,
        y: np.ndarray,
        design_builder: Callable[[pd.DataFrame], np.ndarray],
        nobs: int,
        df_resid: int,
        n_clusters: int | None,
    ) -> None:
        self.formula = str(formula)
        self.model = InternalModel(exog_names)
        self.params = pd.Series(np.asarray(params, dtype=np.float64), index=self.model.exog_names, dtype=np.float64)
        self._cov_params = np.asarray(cov_params, dtype=np.float64)
        self.resid = np.asarray(resid, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._design_builder = design_builder
        self.nobs = int(nobs)
        self.df_resid = int(df_resid)
        self.n_clusters = None if n_clusters is None else int(n_clusters)
        se = np.sqrt(np.clip(np.diag(self._cov_params), a_min=0.0, a_max=None))
        self.bse = pd.Series(se, index=self.model.exog_names, dtype=np.float64)
        z_scores = np.divide(
            self.params.to_numpy(dtype=np.float64),
            np.where(se > EPS, se, np.inf),
            out=np.zeros_like(se, dtype=np.float64),
            where=np.isfinite(se),
        )
        self.pvalues = pd.Series(2.0 * stats.norm.sf(np.abs(z_scores)), index=self.model.exog_names, dtype=np.float64)
        y_centered = self._y - float(np.mean(self._y))
        rss = float(np.sum(np.square(self.resid)))
        tss = float(np.sum(np.square(y_centered)))
        self.rsquared = 1.0 - (rss / tss) if tss > EPS else None
        sigma2 = rss / max(int(nobs), 1)
        loglike = -0.5 * float(nobs) * (math.log(2.0 * math.pi * sigma2) + 1.0) if sigma2 > EPS else float("nan")
        n_params = len(self.model.exog_names)
        self.aic = float(-2.0 * loglike + 2.0 * n_params) if np.isfinite(loglike) else None
        self.bic = float(-2.0 * loglike + math.log(max(int(nobs), 1)) * n_params) if np.isfinite(loglike) else None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        design = self._design_builder(df)
        return design @ self.params.to_numpy(dtype=np.float64)

    def wald_test(self, constraint: np.ndarray) -> InternalWaldResult:
        r_mat = np.asarray(constraint, dtype=np.float64)
        beta = self.params.to_numpy(dtype=np.float64)
        diff = r_mat @ beta
        cov = r_mat @ self._cov_params @ r_mat.T
        inv_cov = np.linalg.pinv(cov)
        stat = float(diff.T @ inv_cov @ diff)
        df_num = int(r_mat.shape[0])
        pvalue = float(stats.chi2.sf(stat, df=max(df_num, 1)))
        return InternalWaldResult(
            statistic=stat,
            pvalue=pvalue,
            df_num=df_num,
            df_denom=self.n_clusters - 1 if self.n_clusters is not None else self.df_resid,
        )


def _condition_spec_table() -> dict[str, ConditionSpec]:
    return {
        "full_dynamic": ConditionSpec("full_dynamic", "dynamic", None, None),
        "sample_remove_SPonly": ConditionSpec("sample_remove_SPonly", "dynamic", "sample_sp_only_mask", None),
        "distractor_remove_DPonly": ConditionSpec("distractor_remove_DPonly", "dynamic", None, "distractor_dp_only_mask"),
        "sample_remove_SDP": ConditionSpec("sample_remove_SDP", "dynamic", "sample_sdp_mask", None),
        "distractor_remove_SDP": ConditionSpec("distractor_remove_SDP", "dynamic", None, "distractor_sdp_mask"),
        "both_remove_SDP": ConditionSpec("both_remove_SDP", "dynamic", "sample_sdp_mask", "distractor_sdp_mask"),
        "sample_remove_SPonly_control": ConditionSpec("sample_remove_SPonly_control", "dynamic", "sample_sp_only_control_mask", None),
        "distractor_remove_DPonly_control": ConditionSpec("distractor_remove_DPonly_control", "dynamic", None, "distractor_dp_only_control_mask"),
        "sample_remove_SDP_control": ConditionSpec("sample_remove_SDP_control", "dynamic", "sample_sdp_control_mask", None),
        "distractor_remove_SDP_control": ConditionSpec("distractor_remove_SDP_control", "dynamic", None, "distractor_sdp_control_mask"),
        "both_remove_SDP_control": ConditionSpec(
            "both_remove_SDP_control",
            "dynamic",
            "sample_sdp_control_mask",
            "distractor_sdp_control_mask",
        ),
    }


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _ensure_condition_categories(df: pd.DataFrame, conditions: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["condition"] = pd.Categorical(out["condition"], categories=list(conditions), ordered=True)
    return out


def _condition_order(exclude_controls_from_strength_model: bool) -> tuple[str, ...]:
    if bool(exclude_controls_from_strength_model):
        return MAIN_CONDITION_ORDER
    return MAIN_CONDITION_ORDER + CONTROL_CONDITION_ORDER


def _design_spec_from_dataframe(df_model: pd.DataFrame) -> dict[str, object]:
    condition_levels = list(df_model["condition"].cat.categories) if hasattr(df_model["condition"], "cat") else sorted(df_model["condition"].astype(str).unique().tolist())
    delay_levels = sorted(df_model["delay_ms"].astype(float).unique().tolist())
    return {
        "condition_levels": condition_levels,
        "condition_base": condition_levels[0] if condition_levels else None,
        "delay_levels": delay_levels,
        "delay_base": delay_levels[0] if delay_levels else None,
    }


def _build_design_matrix(
    df_model: pd.DataFrame,
    *,
    formula: str,
    design_spec: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, list[str], Callable[[pd.DataFrame], np.ndarray]]:
    spec = dict(_design_spec_from_dataframe(df_model) if design_spec is None else design_spec)
    condition_levels = [str(value) for value in spec["condition_levels"]]
    delay_levels = [float(value) for value in spec["delay_levels"]]
    condition_base = str(spec["condition_base"])
    delay_base = float(spec["delay_base"])
    rhs = str(formula).split("~", maxsplit=1)[1].strip()
    terms = [term.strip() for term in rhs.split("+") if term.strip()]

    def make_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        local = frame.copy()
        local["condition"] = local["condition"].astype(str)
        local["delay_ms"] = local["delay_ms"].astype(float)
        if "delay_c" not in local.columns:
            delay_mean = float(df_model["delay_ms"].mean())
            local["delay_c"] = local["delay_ms"] - delay_mean
        if "delay_c2" not in local.columns:
            local["delay_c2"] = local["delay_c"] ** 2
        columns = [np.ones(len(local), dtype=np.float64)]
        names = ["Intercept"]
        cond_masks = {level: (local["condition"].to_numpy() == level).astype(np.float64) for level in condition_levels}
        delay_masks = {level: (local["delay_ms"].to_numpy(dtype=np.float64) == level).astype(np.float64) for level in delay_levels}
        for term in terms:
            if term == "delay_c":
                columns.append(local["delay_c"].to_numpy(dtype=np.float64))
                names.append("delay_c")
            elif term == "delay_c2":
                columns.append(local["delay_c2"].to_numpy(dtype=np.float64))
                names.append("delay_c2")
            elif term == "C(condition)":
                for level in condition_levels:
                    if level == condition_base:
                        continue
                    columns.append(cond_masks[level])
                    names.append(f"C(condition)[T.{level}]")
            elif term == "delay_c:C(condition)":
                for level in condition_levels:
                    if level == condition_base:
                        continue
                    columns.append(local["delay_c"].to_numpy(dtype=np.float64) * cond_masks[level])
                    names.append(f"delay_c:C(condition)[T.{level}]")
            elif term == "delay_c2:C(condition)":
                for level in condition_levels:
                    if level == condition_base:
                        continue
                    columns.append(local["delay_c2"].to_numpy(dtype=np.float64) * cond_masks[level])
                    names.append(f"delay_c2:C(condition)[T.{level}]")
            elif term == "C(delay_ms)":
                for level in delay_levels:
                    if level == delay_base:
                        continue
                    columns.append(delay_masks[level])
                    names.append(f"C(delay_ms)[T.{level:g}]")
            elif term == "C(delay_ms):C(condition)":
                for delay_level in delay_levels:
                    if delay_level == delay_base:
                        continue
                    for condition_level in condition_levels:
                        if condition_level == condition_base:
                            continue
                        columns.append(delay_masks[delay_level] * cond_masks[condition_level])
                        names.append(f"C(delay_ms)[T.{delay_level:g}]:C(condition)[T.{condition_level}]")
            else:
                raise ValueError(f"Unsupported formula term without statsmodels: {term}")
        return np.column_stack(columns), names

    design_matrix, exog_names = make_matrix(df_model)
    return design_matrix, exog_names, lambda frame: make_matrix(frame)[0]


def _fit_internal_ols(
    df_model: pd.DataFrame,
    *,
    formula: str,
    cluster_groups: pd.Series | None,
) -> InternalOLSResult:
    design_spec = _design_spec_from_dataframe(df_model)
    x_mat, exog_names, predictor = _build_design_matrix(df_model, formula=formula, design_spec=design_spec)
    y_vec = df_model["M"].to_numpy(dtype=np.float64)
    beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    fitted = x_mat @ beta
    resid = y_vec - fitted
    nobs = int(len(y_vec))
    p = int(x_mat.shape[1])
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    if cluster_groups is not None:
        groups = np.asarray(cluster_groups, dtype=np.int64)
        unique_groups = np.unique(groups)
        meat = np.zeros((p, p), dtype=np.float64)
        for group in unique_groups:
            mask = groups == group
            xg = x_mat[mask]
            ug = resid[mask]
            xu = xg.T @ ug
            meat += np.outer(xu, xu)
        g = max(int(unique_groups.size), 1)
        n = max(nobs, 1)
        correction = 1.0
        if g > 1 and n > p:
            correction = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
        cov = correction * (xtx_inv @ meat @ xtx_inv)
        n_clusters = g
    else:
        sigma2 = float(np.sum(np.square(resid))) / max(nobs - p, 1)
        cov = sigma2 * xtx_inv
        n_clusters = None
    return InternalOLSResult(
        formula=formula,
        exog_names=exog_names,
        params=beta,
        cov_params=cov,
        resid=resid,
        y=y_vec,
        design_builder=lambda frame: _build_design_matrix(frame, formula=formula, design_spec=design_spec)[0],
        nobs=nobs,
        df_resid=max(nobs - p, 1),
        n_clusters=n_clusters,
    )


def collect_strength_effect_outputs(
    net,
    sample_spikes: torch.Tensor,
    distractor_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    *,
    batch_masks: Sequence[TripletMaskBundle],
    delay_ms_values: Sequence[float],
    spec: ExperimentSpec,
    readout_step: int,
    include_conditions: Sequence[str],
) -> dict[str, np.ndarray]:
    condition_specs = _condition_spec_table()
    static_grouped: list[np.ndarray] = []
    condition_grouped: list[np.ndarray] = []
    for delay_ms in delay_ms_values:
        delay2_steps = int(round((float(delay_ms) * ms) / spec.dt))
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
        static_grouped.append(np.asarray(static.grouped_voltage, dtype=np.float64))
        per_condition: list[np.ndarray] = []
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
            per_condition.append(np.asarray(rollout.grouped_voltage, dtype=np.float64))
        condition_grouped.append(np.stack(per_condition, axis=1))
    static_arr = np.stack(static_grouped, axis=1)
    condition_arr = np.stack(condition_grouped, axis=1)
    static_repeated = np.repeat(static_arr[:, :, None, :], repeats=condition_arr.shape[2], axis=2)
    return {
        "delay_ms": np.asarray(delay_ms_values, dtype=np.float64),
        "condition_name": np.asarray(include_conditions),
        "static_grouped_voltage": static_arr,
        "condition_grouped_voltage": condition_arr,
        "delta_v": compute_delta_v(condition_arr, static_repeated),
    }


def build_strength_effect_table(
    *,
    df_triplets: pd.DataFrame,
    delay_ms_values: Sequence[float],
    condition_names: Sequence[str],
    delta_v: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    triplet_ids: list[int] = []
    u_ref_vectors: list[np.ndarray] = []
    full_dynamic_idx = int(list(condition_names).index("full_dynamic"))
    for triplet_idx, triplet_row in enumerate(df_triplets.itertuples(index=False)):
        triplet_id = int(triplet_row.triplet_id)
        triplet_ids.append(triplet_id)
        delta_v_by_delay = {
            float(delay_ms): np.asarray(delta_v[triplet_idx, delay_idx, full_dynamic_idx], dtype=np.float64)
            for delay_idx, delay_ms in enumerate(delay_ms_values)
        }
        ref_info = compute_reference_direction_from_delays(delta_v_by_delay)
        u_ref = None if ref_info["u_ref"] is None else np.asarray(ref_info["u_ref"], dtype=np.float64)
        if u_ref is None:
            u_ref_vectors.append(np.full(int(delta_v.shape[-1]), np.nan, dtype=np.float32))
        else:
            u_ref_vectors.append(u_ref.astype(np.float32, copy=False))
        for delay_idx, delay_ms in enumerate(delay_ms_values):
            for condition_idx, condition_name in enumerate(condition_names):
                vec = np.asarray(delta_v[triplet_idx, delay_idx, condition_idx], dtype=np.float64)
                magnitude = float(np.linalg.norm(vec))
                if u_ref is None:
                    strength_metrics = {
                        "M": magnitude,
                        "A": float("nan"),
                        "cos_theta": float("nan"),
                        "theta_deg": float("nan"),
                    }
                else:
                    strength_metrics = compute_strength_metrics(vec, u_ref)
                rows.append(
                    {
                        "triplet_id": triplet_id,
                        "sample_id": int(triplet_row.sample_id),
                        "distractor_id": int(triplet_row.distractor_id),
                        "probe_id": int(triplet_row.probe_id),
                        "sample_label": int(triplet_row.sample_label),
                        "distractor_label": int(triplet_row.distractor_label),
                        "probe_label": int(triplet_row.probe_label),
                        "delay_ms": float(delay_ms),
                        "condition": str(condition_name),
                        "M": float(strength_metrics["M"]),
                        "A": float(strength_metrics["A"]),
                        "cos_theta": float(strength_metrics["cos_theta"]),
                        "theta_deg": float(strength_metrics["theta_deg"]),
                        "reference_status": str(ref_info["status"]),
                        "reference_fallback_delay_ms": ref_info.get("fallback_delay_ms"),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["triplet_id", "delay_ms", "condition"], kind="stable").reset_index(drop=True)
    npz_payload = {
        "triplet_id_strength_effect": np.asarray(triplet_ids, dtype=np.int64),
        "delay_ms_strength_effect": np.asarray(delay_ms_values, dtype=np.float32),
        "condition_name_strength_effect": np.asarray(condition_names),
        "delta_v_strength_effect": np.asarray(delta_v, dtype=np.float32),
        "u_ref_per_triplet": np.stack(u_ref_vectors, axis=0) if u_ref_vectors else np.zeros((0, 0), dtype=np.float32),
    }
    return df, npz_payload


def _prepare_model_dataframe(df_records: pd.DataFrame, *, condition_order: Sequence[str]) -> pd.DataFrame:
    df = _ensure_condition_categories(df_records, condition_order)
    df = df[np.isfinite(df["M"].to_numpy(dtype=np.float64))].copy()
    delay_mean = float(df["delay_ms"].mean())
    df["delay_c"] = df["delay_ms"] - delay_mean
    df["delay_c2"] = df["delay_c"] ** 2
    return df


def _build_wald_summary(test_result, *, label: str, param_names: Sequence[str]) -> dict[str, object]:
    return {
        "label": label,
        "status": "ok",
        "statistic": _safe_float(getattr(test_result, "statistic", None)),
        "p_value": _safe_float(getattr(test_result, "pvalue", None)),
        "df_constraint": _safe_float(getattr(test_result, "df_num", None)) or len(param_names),
        "df_denom": _safe_float(getattr(test_result, "df_denom", None)),
        "param_names": [str(name) for name in param_names],
    }


def _wald_test_for_params(result, predicate: Callable[[str], bool], *, label: str) -> dict[str, object]:
    if result is None:
        return {"label": label, "status": "model_unavailable"}
    exog_names = list(getattr(result.model, "exog_names", []))
    param_names = [str(name) for name in exog_names if predicate(str(name))]
    if not param_names:
        return {"label": label, "status": "no_terms", "param_names": []}
    param_index = {str(name): idx for idx, name in enumerate(exog_names)}
    constraint = np.zeros((len(param_names), len(exog_names)), dtype=np.float64)
    for row_idx, name in enumerate(param_names):
        constraint[row_idx, param_index[name]] = 1.0
    try:
        test_result = result.wald_test(constraint)
    except Exception as exc:
        return {"label": label, "status": "test_error", "error": str(exc), "param_names": param_names}
    return _build_wald_summary(test_result, label=label, param_names=param_names)


def _fit_model_bundle(
    df_model: pd.DataFrame,
    *,
    formula: str,
    delay_modeling_mode: str,
    cluster_robust_fallback: bool,
    allow_mixedlm: bool = True,
) -> ModelFitBundle:
    mixed_warnings: list[str] = []
    fallback_reason: str | None = None
    if allow_mixedlm and HAS_STATSMODELS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=ConvergenceWarning)
                model = smf.mixedlm(formula, data=df_model, groups=df_model["triplet_id"])
                result = model.fit(reml=False, method="lbfgs", disp=False)
            mixed_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            if not bool(getattr(result, "converged", False)):
                raise RuntimeError("MixedLM did not converge.")
            if mixed_warnings:
                raise RuntimeError("MixedLM raised convergence warnings: " + " | ".join(mixed_warnings))
            companion_ols = smf.ols(formula, data=df_model).fit()
            return ModelFitBundle(
                status="ok",
                formula=formula,
                delay_modeling_mode=delay_modeling_mode,
                model_type="MixedLM",
                fallback_used=False,
                fallback_reason=None,
                result=result,
                companion_ols=companion_ols,
                aic=_safe_float(getattr(result, "aic", None)),
                bic=_safe_float(getattr(result, "bic", None)),
                warnings=mixed_warnings,
            )
        except Exception as exc:
            fallback_reason = str(exc)
    elif allow_mixedlm and not HAS_STATSMODELS:
        fallback_reason = "statsmodels is unavailable; using internal OLS fallback"
    if not bool(cluster_robust_fallback):
        return ModelFitBundle(
            status="fit_failed",
            formula=formula,
            delay_modeling_mode=delay_modeling_mode,
            model_type=None,
            fallback_used=False,
            fallback_reason=fallback_reason,
            result=None,
            companion_ols=None,
            aic=None,
            bic=None,
            warnings=mixed_warnings,
        )
    try:
        if HAS_STATSMODELS:
            ols_result = smf.ols(formula, data=df_model).fit(
                cov_type="cluster",
                cov_kwds={"groups": df_model["triplet_id"]},
            )
            companion_ols = smf.ols(formula, data=df_model).fit()
        else:
            ols_result = _fit_internal_ols(df_model, formula=formula, cluster_groups=df_model["triplet_id"])
            companion_ols = _fit_internal_ols(df_model, formula=formula, cluster_groups=None)
    except Exception as exc:
        message = str(exc) if fallback_reason is None else f"{fallback_reason} | OLS fallback failed: {exc}"
        return ModelFitBundle(
            status="fit_failed",
            formula=formula,
            delay_modeling_mode=delay_modeling_mode,
            model_type=None,
            fallback_used=True,
            fallback_reason=message,
            result=None,
            companion_ols=None,
            aic=None,
            bic=None,
            warnings=mixed_warnings,
        )
    return ModelFitBundle(
        status="ok",
        formula=formula,
        delay_modeling_mode=delay_modeling_mode,
        model_type="OLS_cluster_robust",
        fallback_used=True,
        fallback_reason=fallback_reason,
        result=ols_result,
        companion_ols=companion_ols,
        aic=_safe_float(getattr(companion_ols, "aic", None)),
        bic=_safe_float(getattr(companion_ols, "bic", None)),
        warnings=mixed_warnings,
    )


def _coefficient_summary(result, param_name: str) -> dict[str, object]:
    if result is None:
        return {"status": "model_unavailable", "param_name": param_name}
    params = getattr(result, "params", {})
    if param_name not in params.index:
        return {"status": "missing_param", "param_name": param_name}
    pvalues = getattr(result, "pvalues", {})
    return {
        "status": "ok",
        "param_name": param_name,
        "coefficient": _safe_float(params[param_name]),
        "p_value": _safe_float(pvalues[param_name]) if hasattr(pvalues, "index") and param_name in pvalues.index else None,
    }


def _linear_direction_from_slope(slope: float | None) -> str:
    if slope is None:
        return "undetermined"
    if slope > 0.0:
        return "positive"
    if slope < 0.0:
        return "negative"
    return "flat"


def _quadratic_peak_summary(bundle: ModelFitBundle, df_model: pd.DataFrame) -> dict[str, object]:
    coef_linear = _coefficient_summary(bundle.result, "delay_c")
    coef_quadratic = _coefficient_summary(bundle.result, "delay_c2")
    a = coef_quadratic.get("coefficient")
    b = coef_linear.get("coefficient")
    if a is None or b is None or abs(float(a)) <= EPS:
        return {
            "status": "unavailable",
            "full_dynamic_peak_delay_ms": None,
            "supports_rise_then_fall": False,
            "delay_c2_coefficient": a,
        }
    delay_mean = float(df_model["delay_ms"].mean())
    peak_delay_ms = float(delay_mean - (float(b) / (2.0 * float(a))))
    min_delay = float(df_model["delay_ms"].min())
    max_delay = float(df_model["delay_ms"].max())
    return {
        "status": "ok",
        "full_dynamic_peak_delay_ms": peak_delay_ms,
        "peak_within_observed_range": bool(min_delay <= peak_delay_ms <= max_delay),
        "supports_rise_then_fall": bool(float(a) < 0.0 and min_delay <= peak_delay_ms <= max_delay),
        "delay_c2_coefficient": float(a),
        "delay_c_coefficient": float(b),
    }


def _bundle_base_summary(bundle: ModelFitBundle, df_model: pd.DataFrame) -> dict[str, object]:
    return {
        "status": bundle.status,
        "formula": bundle.formula,
        "delay_modeling_mode": bundle.delay_modeling_mode,
        "model_type": bundle.model_type,
        "fallback_used": bool(bundle.fallback_used),
        "fallback_reason": bundle.fallback_reason,
        "aic": bundle.aic,
        "bic": bundle.bic,
        "n_rows": int(len(df_model)),
        "n_triplets": int(df_model["triplet_id"].nunique()),
        "warnings": [str(item) for item in bundle.warnings],
    }


def fit_strength_effect_model(
    df_records: pd.DataFrame,
    *,
    cluster_robust_fallback: bool = True,
    condition_order: Sequence[str] = MAIN_CONDITION_ORDER,
) -> dict[str, object]:
    df_model = _prepare_model_dataframe(df_records, condition_order=condition_order)
    formula = "M ~ delay_c + C(condition) + delay_c:C(condition)"
    bundle = _fit_model_bundle(
        df_model,
        formula=formula,
        delay_modeling_mode="linear_continuous",
        cluster_robust_fallback=cluster_robust_fallback,
        allow_mixedlm=True,
    )
    summary = _bundle_base_summary(bundle, df_model)
    if bundle.status == "ok":
        delay_coef = _coefficient_summary(bundle.result, "delay_c")
        summary.update(
            {
                "delay_main_effect": {
                    **_wald_test_for_params(bundle.result, lambda name: name == "delay_c", label="delay_main_effect"),
                    "slope": delay_coef.get("coefficient"),
                    "direction": _linear_direction_from_slope(delay_coef.get("coefficient")),
                },
                "condition_main_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("C(condition)[") and ":C(" not in name,
                    label="condition_main_effect",
                ),
                "interaction_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("delay_c:C(condition)["),
                    label="interaction_effect",
                ),
            }
        )
    else:
        summary.update(
            {
                "delay_main_effect": {"status": "model_unavailable"},
                "condition_main_effect": {"status": "model_unavailable"},
                "interaction_effect": {"status": "model_unavailable"},
            }
        )
    return {"data": df_model, "bundle": bundle, "summary": summary}


def fit_strength_effect_model_quadratic(
    df_records: pd.DataFrame,
    *,
    cluster_robust_fallback: bool = True,
    condition_order: Sequence[str] = MAIN_CONDITION_ORDER,
) -> dict[str, object]:
    df_model = _prepare_model_dataframe(df_records, condition_order=condition_order)
    formula = "M ~ delay_c + delay_c2 + C(condition) + delay_c:C(condition) + delay_c2:C(condition)"
    bundle = _fit_model_bundle(
        df_model,
        formula=formula,
        delay_modeling_mode="quadratic_continuous",
        cluster_robust_fallback=cluster_robust_fallback,
        allow_mixedlm=True,
    )
    summary = _bundle_base_summary(bundle, df_model)
    if bundle.status == "ok":
        summary.update(
            {
                "delay_main_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name in {"delay_c", "delay_c2"},
                    label="delay_main_effect",
                ),
                "quadratic_term": _wald_test_for_params(
                    bundle.result,
                    lambda name: name == "delay_c2",
                    label="quadratic_term",
                ),
                "condition_main_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("C(condition)[") and ":C(" not in name,
                    label="condition_main_effect",
                ),
                "interaction_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("delay_c:C(condition)[") or name.startswith("delay_c2:C(condition)["),
                    label="interaction_effect",
                ),
                "peak_summary": _quadratic_peak_summary(bundle, df_model),
            }
        )
    else:
        summary.update(
            {
                "delay_main_effect": {"status": "model_unavailable"},
                "quadratic_term": {"status": "model_unavailable"},
                "condition_main_effect": {"status": "model_unavailable"},
                "interaction_effect": {"status": "model_unavailable"},
                "peak_summary": {"status": "model_unavailable"},
            }
        )
    return {"data": df_model, "bundle": bundle, "summary": summary}


def _fit_strength_effect_model_factor_delay(
    df_records: pd.DataFrame,
    *,
    cluster_robust_fallback: bool = True,
    condition_order: Sequence[str] = MAIN_CONDITION_ORDER,
) -> dict[str, object]:
    df_model = _prepare_model_dataframe(df_records, condition_order=condition_order)
    formula = "M ~ C(delay_ms) + C(condition) + C(delay_ms):C(condition)"
    bundle = _fit_model_bundle(
        df_model,
        formula=formula,
        delay_modeling_mode="factor_delay",
        cluster_robust_fallback=cluster_robust_fallback,
        allow_mixedlm=False,
    )
    summary = _bundle_base_summary(bundle, df_model)
    if bundle.status == "ok":
        summary.update(
            {
                "delay_main_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("C(delay_ms)[") and ":C(condition)[" not in name,
                    label="delay_main_effect",
                ),
                "condition_main_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("C(condition)[") and ":C(" not in name,
                    label="condition_main_effect",
                ),
                "interaction_effect": _wald_test_for_params(
                    bundle.result,
                    lambda name: name.startswith("C(delay_ms)[") and ":C(condition)[" in name,
                    label="interaction_effect",
                ),
            }
        )
    else:
        summary.update(
            {
                "delay_main_effect": {"status": "model_unavailable"},
                "condition_main_effect": {"status": "model_unavailable"},
                "interaction_effect": {"status": "model_unavailable"},
            }
        )
    return {"data": df_model, "bundle": bundle, "summary": summary}


def _model_selection_summary(linear_fit: dict[str, object], quadratic_fit: dict[str, object]) -> dict[str, object]:
    linear_bundle = linear_fit["bundle"]
    quadratic_bundle = quadratic_fit["bundle"]
    comparison = {
        "linear_status": linear_bundle.status,
        "quadratic_status": quadratic_bundle.status,
        "selected_delay_model": "linear_continuous",
        "supports_quadratic_model": False,
        "delta_aic_quadratic_minus_linear": None,
        "delta_bic_quadratic_minus_linear": None,
        "quadratic_improves_fit": False,
        "quadratic_term_p_value": None,
    }
    if linear_bundle.status != "ok" and quadratic_bundle.status == "ok":
        comparison["selected_delay_model"] = "quadratic_continuous"
        comparison["supports_quadratic_model"] = True
        return comparison
    if linear_bundle.status != "ok" or quadratic_bundle.status != "ok":
        return comparison
    delta_aic = None if quadratic_bundle.aic is None or linear_bundle.aic is None else float(quadratic_bundle.aic - linear_bundle.aic)
    delta_bic = None if quadratic_bundle.bic is None or linear_bundle.bic is None else float(quadratic_bundle.bic - linear_bundle.bic)
    quadratic_term = quadratic_fit["summary"].get("quadratic_term", {})
    quadratic_term_p = quadratic_term.get("p_value")
    improves = bool(
        (delta_aic is not None and delta_aic <= -2.0)
        or (delta_bic is not None and delta_bic <= -2.0)
        or (quadratic_term_p is not None and float(quadratic_term_p) < 0.05)
    )
    comparison.update(
        {
            "delta_aic_quadratic_minus_linear": delta_aic,
            "delta_bic_quadratic_minus_linear": delta_bic,
            "quadratic_improves_fit": improves,
            "quadratic_term_p_value": quadratic_term_p,
            "supports_quadratic_model": True,
            "selected_delay_model": "quadratic_continuous" if improves else "linear_continuous",
        }
    )
    return comparison


def _partial_r2_from_formulas(df_model: pd.DataFrame, *, full_formula: str, reduced_formula: str) -> dict[str, object]:
    try:
        if HAS_STATSMODELS:
            full_fit = smf.ols(full_formula, data=df_model).fit()
            reduced_fit = smf.ols(reduced_formula, data=df_model).fit()
        else:
            full_fit = _fit_internal_ols(df_model, formula=full_formula, cluster_groups=None)
            reduced_fit = _fit_internal_ols(df_model, formula=reduced_formula, cluster_groups=None)
    except Exception as exc:
        return {"status": "fit_error", "error": str(exc)}
    sse_full = float(np.sum(np.square(full_fit.resid)))
    sse_reduced = float(np.sum(np.square(reduced_fit.resid)))
    if sse_reduced <= EPS:
        return {"status": "degenerate_reduced_model", "partial_r2": None}
    partial_r2 = max(0.0, min(1.0, (sse_reduced - sse_full) / sse_reduced))
    return {
        "status": "ok",
        "partial_r2": partial_r2,
        "full_r2": _safe_float(full_fit.rsquared),
        "reduced_r2": _safe_float(reduced_fit.rsquared),
        "delta_r2": None if full_fit.rsquared is None or reduced_fit.rsquared is None else float(full_fit.rsquared - reduced_fit.rsquared),
    }


def compute_effect_size_summary(
    df_model: pd.DataFrame,
    *,
    chosen_bundle: ModelFitBundle,
    linear_bundle: ModelFitBundle | None = None,
    quadratic_bundle: ModelFitBundle | None = None,
) -> dict[str, object]:
    mode = str(chosen_bundle.delay_modeling_mode)
    if mode == "quadratic_continuous":
        full_formula = "M ~ delay_c + delay_c2 + C(condition) + delay_c:C(condition) + delay_c2:C(condition)"
        delay_formula = "M ~ C(condition) + delay_c:C(condition) + delay_c2:C(condition)"
        condition_formula = "M ~ delay_c + delay_c2 + delay_c:C(condition) + delay_c2:C(condition)"
        interaction_formula = "M ~ delay_c + delay_c2 + C(condition)"
        nonlinear_formula = "M ~ delay_c + C(condition) + delay_c:C(condition)"
    elif mode == "factor_delay":
        full_formula = "M ~ C(delay_ms) + C(condition) + C(delay_ms):C(condition)"
        delay_formula = "M ~ C(condition) + C(delay_ms):C(condition)"
        condition_formula = "M ~ C(delay_ms) + C(delay_ms):C(condition)"
        interaction_formula = "M ~ C(delay_ms) + C(condition)"
        nonlinear_formula = None
    else:
        full_formula = "M ~ delay_c + C(condition) + delay_c:C(condition)"
        delay_formula = "M ~ C(condition) + delay_c:C(condition)"
        condition_formula = "M ~ delay_c + delay_c:C(condition)"
        interaction_formula = "M ~ delay_c + C(condition)"
        nonlinear_formula = None
    summary = {
        "source_model_for_effect_sizes": "companion_ols_fixed_effects",
        "delay_main": _partial_r2_from_formulas(df_model, full_formula=full_formula, reduced_formula=delay_formula),
        "condition_main": _partial_r2_from_formulas(df_model, full_formula=full_formula, reduced_formula=condition_formula),
        "interaction": _partial_r2_from_formulas(df_model, full_formula=full_formula, reduced_formula=interaction_formula),
    }
    if nonlinear_formula is not None:
        summary["quadratic_component"] = _partial_r2_from_formulas(
            df_model,
            full_formula=full_formula,
            reduced_formula=nonlinear_formula,
        )
    if linear_bundle is not None and quadratic_bundle is not None:
        summary["continuous_model_comparison"] = {
            "linear_aic": linear_bundle.aic,
            "linear_bic": linear_bundle.bic,
            "quadratic_aic": quadratic_bundle.aic,
            "quadratic_bic": quadratic_bundle.bic,
            "delta_aic_quadratic_minus_linear": None
            if linear_bundle.aic is None or quadratic_bundle.aic is None
            else float(quadratic_bundle.aic - linear_bundle.aic),
            "delta_bic_quadratic_minus_linear": None
            if linear_bundle.bic is None or quadratic_bundle.bic is None
            else float(quadratic_bundle.bic - linear_bundle.bic),
        }
    ranking: list[dict[str, object]] = []
    for label in ("delay_main", "condition_main", "interaction"):
        ranking.append({"effect": label, "partial_r2": summary[label].get("partial_r2")})
    ranking.sort(key=lambda item: (-1.0 if item["partial_r2"] is None else -float(item["partial_r2"]), item["effect"]))
    summary["relative_strength_ranking"] = ranking
    delay_r2 = summary["delay_main"].get("partial_r2")
    condition_r2 = summary["condition_main"].get("partial_r2")
    if delay_r2 is not None and condition_r2 is not None:
        summary["delay_vs_condition_ratio"] = None if float(condition_r2) <= EPS else float(delay_r2) / float(condition_r2)
    else:
        summary["delay_vs_condition_ratio"] = None
    return summary


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.ones(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = float((m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def _paired_test_against_full_dynamic(reference: np.ndarray, target: np.ndarray) -> dict[str, object]:
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(ref) & np.isfinite(tgt)
    ref = ref[mask]
    tgt = tgt[mask]
    if ref.size == 0:
        return {"status": "no_pairs", "n_pairs": 0}
    delta = tgt - ref
    if np.allclose(delta, 0.0):
        return {
            "status": "all_zero_differences",
            "n_pairs": int(delta.size),
            "method": "none",
            "p_value": 1.0,
            "statistic": 0.0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
        }
    try:
        stat = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided", correction=False)
        return {
            "status": "ok",
            "n_pairs": int(delta.size),
            "method": "wilcoxon",
            "p_value": float(stat.pvalue),
            "statistic": _safe_float(stat.statistic),
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
        }
    except Exception:
        ttest = stats.ttest_rel(tgt, ref, nan_policy="omit")
        return {
            "status": "ok",
            "n_pairs": int(delta.size),
            "method": "paired_t_test",
            "p_value": float(ttest.pvalue),
            "statistic": _safe_float(ttest.statistic),
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
        }


def _resolve_requested_delay(delay_ms_values: Sequence[float], requested_delay_ms: float) -> tuple[float, float]:
    available = np.asarray(delay_ms_values, dtype=np.float64)
    idx = int(np.argmin(np.abs(available - float(requested_delay_ms))))
    return float(requested_delay_ms), float(available[idx])


def build_fixed_delay_comparisons(
    df_records: pd.DataFrame,
    *,
    fixed_delay_requests: Sequence[float],
    delay_ms_values: Sequence[float],
    condition_order: Sequence[str],
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for requested_delay in fixed_delay_requests:
        requested_delay_ms, resolved_delay_ms = _resolve_requested_delay(delay_ms_values, requested_delay)
        subset = df_records[df_records["delay_ms"] == resolved_delay_ms].copy()
        if subset.empty:
            results.append(
                {
                    "requested_delay_ms": requested_delay_ms,
                    "resolved_delay_ms": resolved_delay_ms,
                    "status": "no_records",
                    "comparisons": [],
                }
            )
            continue
        pivot = subset.pivot_table(index="triplet_id", columns="condition", values="M", aggfunc="mean")
        raw_tests: list[dict[str, object]] = []
        raw_pvalues: list[float] = []
        for condition_name in condition_order:
            if condition_name == "full_dynamic" or condition_name not in pivot.columns or "full_dynamic" not in pivot.columns:
                continue
            merged = pivot[["full_dynamic", condition_name]].dropna()
            test_summary = _paired_test_against_full_dynamic(
                merged["full_dynamic"].to_numpy(dtype=np.float64),
                merged[condition_name].to_numpy(dtype=np.float64),
            )
            test_summary.update({"condition": str(condition_name)})
            raw_tests.append(test_summary)
            raw_pvalues.append(float(test_summary.get("p_value", 1.0)) if test_summary.get("p_value") is not None else 1.0)
        if raw_tests:
            adjusted = _holm_adjust(raw_pvalues)
            for payload, adj in zip(raw_tests, adjusted):
                payload["p_value_holm"] = float(adj)
        results.append(
            {
                "requested_delay_ms": requested_delay_ms,
                "resolved_delay_ms": resolved_delay_ms,
                "status": "ok",
                "n_triplets": int(subset["triplet_id"].nunique()),
                "comparisons": raw_tests,
            }
        )
    return {"fixed_delay_comparisons": results}


def _fit_triplet_condition_slope(delay_ms: np.ndarray, values: np.ndarray) -> float | None:
    mask = np.isfinite(delay_ms) & np.isfinite(values)
    xx = np.asarray(delay_ms[mask], dtype=np.float64)
    yy = np.asarray(values[mask], dtype=np.float64)
    if xx.size < 2 or np.unique(xx).size < 2:
        return None
    return float(np.polyfit(xx, yy, deg=1)[0])


def build_triplet_level_slope_summary(
    df_records: pd.DataFrame,
    *,
    condition_order: Sequence[str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for (triplet_id, condition_name), subset in df_records.groupby(["triplet_id", "condition"], sort=False):
        slope = _fit_triplet_condition_slope(
            subset["delay_ms"].to_numpy(dtype=np.float64),
            subset["M"].to_numpy(dtype=np.float64),
        )
        if slope is None:
            continue
        rows.append({"triplet_id": int(triplet_id), "condition": str(condition_name), "slope": float(slope)})
    df_slopes = pd.DataFrame(rows)
    if df_slopes.empty:
        return {"status": "no_valid_slopes", "condition_summaries": [], "condition_vs_full_dynamic": []}
    condition_summaries = []
    for condition_name in condition_order:
        subset = df_slopes[df_slopes["condition"] == condition_name]
        if subset.empty:
            continue
        condition_summaries.append(
            {
                "condition": str(condition_name),
                "n_triplets": int(subset["triplet_id"].nunique()),
                "mean_slope": float(subset["slope"].mean()),
                "sem_slope": _sem(subset["slope"].to_numpy(dtype=np.float64)),
            }
        )
    pivot = df_slopes.pivot_table(index="triplet_id", columns="condition", values="slope", aggfunc="mean")
    raw_tests: list[dict[str, object]] = []
    pvalues: list[float] = []
    for condition_name in condition_order:
        if condition_name == "full_dynamic" or condition_name not in pivot.columns or "full_dynamic" not in pivot.columns:
            continue
        merged = pivot[["full_dynamic", condition_name]].dropna()
        test_summary = _paired_test_against_full_dynamic(
            merged["full_dynamic"].to_numpy(dtype=np.float64),
            merged[condition_name].to_numpy(dtype=np.float64),
        )
        test_summary.update({"condition": str(condition_name)})
        raw_tests.append(test_summary)
        pvalues.append(float(test_summary.get("p_value", 1.0)) if test_summary.get("p_value") is not None else 1.0)
    if raw_tests:
        adjusted = _holm_adjust(pvalues)
        for payload, adj in zip(raw_tests, adjusted):
            payload["p_value_holm"] = float(adj)
    return {
        "status": "ok",
        "condition_summaries": condition_summaries,
        "condition_vs_full_dynamic": raw_tests,
    }


def plot_strength_vs_delay_by_condition(
    df_records: pd.DataFrame,
    *,
    condition_order: Sequence[str],
    fitted_bundle: ModelFitBundle | None = None,
) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    summary = (
        df_records[df_records["condition"].isin(condition_order)]
        .groupby(["condition", "delay_ms"], sort=True)["M"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary["sem"] = [
        _sem(
            df_records[(df_records["condition"] == row.condition) & (df_records["delay_ms"] == float(row.delay_ms))]["M"].to_numpy(dtype=np.float64)
        )
        for row in summary.itertuples(index=False)
    ]
    for condition_name in condition_order:
        subset = summary[summary["condition"] == condition_name]
        if subset.empty:
            continue
        x = subset["delay_ms"].to_numpy(dtype=np.float64)
        y = subset["mean"].to_numpy(dtype=np.float64)
        sem = subset["sem"].to_numpy(dtype=np.float64)
        color = CONDITION_COLORS.get(str(condition_name), "#4C78A8")
        ax.plot(x, y, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=color, label=str(condition_name))
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=ALPHA_FILL)
    if fitted_bundle is not None and fitted_bundle.status == "ok" and fitted_bundle.companion_ols is not None:
        if fitted_bundle.delay_modeling_mode in {"linear_continuous", "quadratic_continuous"}:
            x_fit = np.linspace(float(df_records["delay_ms"].min()), float(df_records["delay_ms"].max()), num=200)
        else:
            x_fit = np.sort(df_records["delay_ms"].unique().astype(np.float64))
        pred_rows: list[dict[str, object]] = []
        delay_mean = float(df_records["delay_ms"].mean())
        for condition_name in condition_order:
            for delay_ms_value in x_fit:
                pred_rows.append(
                    {
                        "condition": str(condition_name),
                        "delay_ms": float(delay_ms_value),
                        "delay_c": float(delay_ms_value - delay_mean),
                        "delay_c2": float((delay_ms_value - delay_mean) ** 2),
                    }
                )
        pred_df = pd.DataFrame(pred_rows)
        pred_df["condition"] = pd.Categorical(pred_df["condition"], categories=list(condition_order), ordered=True)
        pred_df["M_pred"] = np.asarray(fitted_bundle.companion_ols.predict(pred_df), dtype=np.float64)
        for condition_name in condition_order:
            subset = pred_df[pred_df["condition"] == condition_name]
            if subset.empty:
                continue
            ax.plot(
                subset["delay_ms"].to_numpy(dtype=np.float64),
                subset["M_pred"].to_numpy(dtype=np.float64),
                linestyle="--",
                linewidth=1.4,
                color=CONDITION_COLORS.get(str(condition_name), "#4C78A8"),
                alpha=ALPHA_SCATTER,
            )
    ax.set_xlabel("Distractor -> probe delay (ms)")
    ax.set_ylabel("M = ||DeltaV||")
    ax.set_title("Strength vs delay by condition")
    ax.grid(alpha=GRID_ALPHA)
    apply_standard_legend(ax, compact=True, ncol=2)
    fig.tight_layout()
    return fig


def plot_condition_effect_fixed_delay(
    df_records: pd.DataFrame,
    *,
    fixed_delay_requests: Sequence[float],
    delay_ms_values: Sequence[float],
    condition_order: Sequence[str],
) -> plt.Figure:
    apply_publication_style()
    resolved = [_resolve_requested_delay(delay_ms_values, delay_value) for delay_value in fixed_delay_requests]
    n_panels = max(1, len(resolved))
    fig, axes = plt.subplots(1, n_panels, figsize=horizontal_panel_figsize(n_panels, panel_width=5.0, height=5.1), squeeze=False)
    for ax, (requested_delay_ms, resolved_delay_ms) in zip(axes[0], resolved):
        subset = df_records[(df_records["delay_ms"] == resolved_delay_ms) & (df_records["condition"].isin(condition_order))]
        box_data = []
        positions = np.arange(len(condition_order), dtype=np.float64)
        for condition_name in condition_order:
            values = subset[subset["condition"] == condition_name]["M"].to_numpy(dtype=np.float64)
            box_data.append(values[np.isfinite(values)])
        ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": "#F3F3F3", "edgecolor": "#555555"},
        )
        for pos, condition_name, values in zip(positions, condition_order, box_data):
            if values.size == 0:
                continue
            jitter = np.linspace(-0.12, 0.12, num=values.size) if values.size > 1 else np.asarray([0.0], dtype=np.float64)
            ax.scatter(pos + jitter, values, s=18, alpha=ALPHA_SCATTER, color=CONDITION_COLORS.get(str(condition_name), "#4C78A8"))
        ax.set_title(f"Requested {requested_delay_ms:.0f} ms\nUsing {resolved_delay_ms:.0f} ms")
        ax.set_xticks(positions)
        ax.set_xticklabels(condition_order, rotation=25, ha="right")
        ax.set_ylabel("M = ||DeltaV||")
        ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.suptitle("Condition effect at fixed delay", y=1.02)
    fig.tight_layout()
    return fig


def plot_effect_size_comparison(effect_sizes: Mapping[str, object]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_MEDIUM)
    labels = ["delay_main", "condition_main", "interaction"]
    values = []
    for label in labels:
        payload = effect_sizes.get(label, {})
        values.append(float(payload["partial_r2"]) if payload.get("partial_r2") is not None else 0.0)
    colors = ["#4C78A8", "#E45756", "#72B7B2"]
    ax.bar(labels, values, color=colors, alpha=ALPHA_BAR)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Partial R^2")
    ax.set_title("Effect size comparison")
    ax.set_ylim(0.0, max(0.12, max(values) * 1.2 if values else 0.12))
    ax.grid(alpha=GRID_ALPHA, axis="y")
    fig.tight_layout()
    return fig


def _interpretation_notes(
    *,
    chosen_summary: Mapping[str, object],
    effect_sizes: Mapping[str, object],
) -> list[str]:
    notes = [
        "This analysis does not attempt to prove that overlap has absolutely no effect on strength.",
        "The primary target is whether delay is the dominant modulator of M = ||DeltaV||, while overlap condition contributes weakly or is not detectably independent in the current dataset.",
        "A = DeltaV dot u_ref is retained only as a supplementary direction-aware readout; the main strength model is fit on M.",
    ]
    condition_effect = chosen_summary.get("condition_main_effect", {})
    interaction_effect = chosen_summary.get("interaction_effect", {})
    delay_effect = chosen_summary.get("delay_main_effect", {})
    delay_r2 = effect_sizes.get("delay_main", {}).get("partial_r2")
    condition_r2 = effect_sizes.get("condition_main", {}).get("partial_r2")
    if delay_effect.get("p_value") is not None and float(delay_effect["p_value"]) < 0.05:
        notes.append("The delay main effect is statistically detectable under the selected model.")
    else:
        notes.append("The selected model did not provide strong evidence for an independent delay main effect; this should be interpreted cautiously.")
    if condition_effect.get("p_value") is None or float(condition_effect.get("p_value", 1.0)) >= 0.05:
        notes.append("No statistically significant independent condition effect was detected in this dataset under the selected model; this is not proof of zero effect.")
    elif delay_r2 is not None and condition_r2 is not None and float(delay_r2) > float(condition_r2):
        notes.append("Condition effects are detectable but smaller than the delay contribution, supporting a delay-dominant strength account.")
    else:
        notes.append("Condition effects are detectable and should be reported directly rather than reframed as a zero effect.")
    if interaction_effect.get("p_value") is None or float(interaction_effect.get("p_value", 1.0)) >= 0.05:
        notes.append("The delay x condition interaction was not strongly detected; this should be reported as non-detection, not as proof of no interaction.")
    else:
        notes.append("The delay x condition interaction is detectable and should be described as a secondary modulation rather than ignored.")
    return notes


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strength-independence analysis for distractor delay and overlap conditions.")
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
    parser.add_argument("--strength-fixed-delay-ms", type=float, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--use-dilated-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dilation-radius", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument("--num-control-candidates", type=int, default=DEFAULT_NUM_CONTROL_CANDIDATES)
    parser.add_argument("--use-quadratic-delay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-controls-from-strength-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-robust-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _validate_positive("--batch-size", int(args.batch_size))
    _validate_positive("--max-probes", int(args.max_probes))
    _validate_positive("--samples-per-probe", int(args.samples_per_probe))
    _validate_positive("--max-triplets", int(args.max_triplets))
    _validate_positive("--num-sim-bins", int(args.num_sim_bins))
    _validate_positive("--num-control-candidates", int(args.num_control_candidates))
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--delay1-ms", float(args.delay1_ms), allow_zero=True)
    _validate_positive("--distractor-ms", float(args.distractor_ms))
    _validate_positive("--probe-ms", float(args.probe_ms))
    _validate_positive("--direction-delay-ms", float(args.direction_delay_ms), allow_zero=True)
    delay_ms_values = _sanitize_delay_sweep(args.delay_sweep_ms)
    fixed_delay_requests = [float(args.direction_delay_ms)] if args.strength_fixed_delay_ms is None else [float(value) for value in args.strength_fixed_delay_ms]
    for value in fixed_delay_requests:
        _validate_positive("--strength-fixed-delay-ms", value, allow_zero=True)

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

    included_conditions = _condition_order(bool(args.exclude_controls_from_strength_model))
    strength_rows: list[pd.DataFrame] = []
    delta_batches: list[np.ndarray] = []
    static_batches: list[np.ndarray] = []
    triplet_id_batches: list[np.ndarray] = []
    u_ref_batches: list[np.ndarray] = []
    reference_status_counts: dict[str, int] = {}

    batch_starts = range(0, len(df_triplets), int(args.batch_size))
    total_batches = math.ceil(len(df_triplets) / int(args.batch_size))
    for batch_start in tqdm(batch_starts, total=total_batches, desc="Running strength-effect independence"):
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
        outputs = collect_strength_effect_outputs(
            net=net,
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            batch_masks=batch_masks,
            delay_ms_values=delay_ms_values,
            spec=spec,
            readout_step=readout_step,
            include_conditions=included_conditions,
        )
        batch_table, batch_npz = build_strength_effect_table(
            df_triplets=batch_df,
            delay_ms_values=delay_ms_values,
            condition_names=included_conditions,
            delta_v=outputs["delta_v"],
        )
        strength_rows.append(batch_table)
        delta_batches.append(np.asarray(outputs["delta_v"], dtype=np.float32))
        static_batches.append(np.asarray(outputs["static_grouped_voltage"], dtype=np.float32))
        triplet_id_batches.append(batch_npz["triplet_id_strength_effect"])
        u_ref_batches.append(batch_npz["u_ref_per_triplet"])
        triplet_status = batch_table[["triplet_id", "reference_status"]].drop_duplicates()
        for status_name, count in triplet_status.groupby("reference_status").size().to_dict().items():
            reference_status_counts[str(status_name)] = reference_status_counts.get(str(status_name), 0) + int(count)

    df_strength_records = pd.concat(strength_rows, axis=0, ignore_index=True)
    df_strength_records = df_strength_records.sort_values(["triplet_id", "delay_ms", "condition"], kind="stable").reset_index(drop=True)
    if df_strength_records.empty:
        raise RuntimeError("Strength effect analysis produced no valid records.")

    strength_effect_csv = save_tidy_csv(
        df_strength_records,
        output_dir / "strength_effect_records.csv",
        sort_by=["triplet_id", "delay_ms", "condition"],
    )

    delta_npz_path = output_dir / "delta_v_arrays.npz"
    np.savez_compressed(
        delta_npz_path,
        triplet_id_strength_effect=np.concatenate(triplet_id_batches, axis=0) if triplet_id_batches else np.zeros((0,), dtype=np.int64),
        delay_ms_strength_effect=np.asarray(delay_ms_values, dtype=np.float32),
        condition_name_strength_effect=np.asarray(included_conditions),
        delta_v_strength_effect=np.concatenate(delta_batches, axis=0)
        if delta_batches
        else np.zeros((0, len(delay_ms_values), len(included_conditions), num_classes), dtype=np.float32),
        static_grouped_voltage_strength_effect=np.concatenate(static_batches, axis=0)
        if static_batches
        else np.zeros((0, len(delay_ms_values), num_classes), dtype=np.float32),
        u_ref_per_triplet=np.concatenate(u_ref_batches, axis=0) if u_ref_batches else np.zeros((0, num_classes), dtype=np.float32),
    )

    linear_fit = fit_strength_effect_model(
        df_strength_records,
        cluster_robust_fallback=bool(args.cluster_robust_fallback),
        condition_order=included_conditions,
    )
    quadratic_fit = fit_strength_effect_model_quadratic(
        df_strength_records,
        cluster_robust_fallback=bool(args.cluster_robust_fallback),
        condition_order=included_conditions,
    )
    factor_fit = _fit_strength_effect_model_factor_delay(
        df_strength_records,
        cluster_robust_fallback=bool(args.cluster_robust_fallback),
        condition_order=included_conditions,
    )
    model_selection = _model_selection_summary(linear_fit, quadratic_fit)
    chosen_fit = linear_fit
    if bool(args.use_quadratic_delay) and model_selection["selected_delay_model"] == "quadratic_continuous":
        chosen_fit = quadratic_fit
    elif bool(args.use_quadratic_delay) and quadratic_fit["bundle"].status != "ok" and factor_fit["bundle"].status == "ok":
        chosen_fit = factor_fit
        model_selection["selected_delay_model"] = "factor_delay"
    elif linear_fit["bundle"].status != "ok" and factor_fit["bundle"].status == "ok":
        chosen_fit = factor_fit
        model_selection["selected_delay_model"] = "factor_delay"

    if chosen_fit["bundle"].status != "ok":
        raise RuntimeError(
            "Unable to fit strength effect model. "
            f"Linear: {linear_fit['bundle'].fallback_reason}; "
            f"Quadratic: {quadratic_fit['bundle'].fallback_reason}; "
            f"Factor: {factor_fit['bundle'].fallback_reason}"
        )

    effect_sizes = compute_effect_size_summary(
        chosen_fit["data"],
        chosen_bundle=chosen_fit["bundle"],
        linear_bundle=linear_fit["bundle"],
        quadratic_bundle=quadratic_fit["bundle"],
    )
    fixed_delay_summary = build_fixed_delay_comparisons(
        df_strength_records[df_strength_records["condition"].isin(MAIN_CONDITION_ORDER)],
        fixed_delay_requests=fixed_delay_requests,
        delay_ms_values=delay_ms_values,
        condition_order=MAIN_CONDITION_ORDER,
    )
    slope_summary = build_triplet_level_slope_summary(
        df_strength_records[df_strength_records["condition"].isin(MAIN_CONDITION_ORDER)],
        condition_order=MAIN_CONDITION_ORDER,
    )

    chosen_summary = dict(chosen_fit["summary"])
    chosen_summary["final_selected_model"] = str(model_selection["selected_delay_model"])
    chosen_summary["quadratic_comparison"] = model_selection
    chosen_summary["effect_sizes"] = effect_sizes
    chosen_summary["fixed_delay_comparisons"] = fixed_delay_summary["fixed_delay_comparisons"]
    chosen_summary["triplet_level_slope_summary"] = slope_summary
    chosen_summary["assumptions"] = {
        "primary_strength_metric": "M = ||DeltaV||",
        "delta_v_definition": "centered_grouped_voltage(V_condition) - centered_grouped_voltage(V_static)",
        "strength_reference_direction": "u_ref is derived from the full_dynamic delay sweep within each triplet",
        "supplementary_metrics_only": ["A", "cos_theta", "theta_deg"],
        "controls_included_in_strength_table": not bool(args.exclude_controls_from_strength_model),
        "cluster_robust_fallback_enabled": bool(args.cluster_robust_fallback),
        "delay_modeling_requested_quadratic": bool(args.use_quadratic_delay),
        "time_weight_tau_ms": float(DEFAULT_TAU_MS),
    }
    chosen_summary["interpretation_notes"] = _interpretation_notes(
        chosen_summary=chosen_summary,
        effect_sizes=effect_sizes,
    )
    chosen_summary["reference_status_counts"] = reference_status_counts

    strength_effect_summary_json = _save_json(chosen_summary, output_dir / "strength_effect_summary.json")
    summary_metrics = {
        "strength_effect_analysis": chosen_summary,
        "assumptions": chosen_summary["assumptions"],
    }
    summary_json = _save_json(summary_metrics, output_dir / "summary_metrics.json")

    figure_paths: dict[str, str] = {}
    if not bool(args.skip_figures):
        fig1 = plot_strength_vs_delay_by_condition(
            df_strength_records[df_strength_records["condition"].isin(MAIN_CONDITION_ORDER)],
            condition_order=MAIN_CONDITION_ORDER,
            fitted_bundle=chosen_fit["bundle"],
        )
        out1 = save_figure_all_formats(fig1, figures_dir / "figure_1_strength_vs_delay_by_condition")
        plt.close(fig1)
        figure_paths.update({f"figure_1_{key}": value for key, value in out1.items()})

        fig2 = plot_condition_effect_fixed_delay(
            df_strength_records[df_strength_records["condition"].isin(MAIN_CONDITION_ORDER)],
            fixed_delay_requests=fixed_delay_requests,
            delay_ms_values=delay_ms_values,
            condition_order=MAIN_CONDITION_ORDER,
        )
        out2 = save_figure_all_formats(fig2, figures_dir / "figure_2_condition_effect_fixed_delay")
        plt.close(fig2)
        figure_paths.update({f"figure_2_{key}": value for key, value in out2.items()})

        fig3 = plot_effect_size_comparison(effect_sizes)
        out3 = save_figure_all_formats(fig3, figures_dir / "figure_3_effect_size_comparison")
        plt.close(fig3)
        figure_paths.update({f"figure_3_{key}": value for key, value in out3.items()})

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
            "strength_fixed_delay_ms": [float(value) for value in fixed_delay_requests],
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_triplets": int(args.max_triplets),
            "num_sim_bins": int(args.num_sim_bins),
            "foreground_threshold": float(args.foreground_threshold),
            "use_dilated_overlap": bool(args.use_dilated_overlap),
            "dilation_radius": int(args.dilation_radius),
            "num_control_candidates": int(args.num_control_candidates),
            "use_quadratic_delay": bool(args.use_quadratic_delay),
            "exclude_controls_from_strength_model": bool(args.exclude_controls_from_strength_model),
            "cluster_robust_fallback": bool(args.cluster_robust_fallback),
            "skip_figures": bool(args.skip_figures),
            "readout_step": int(readout_step),
            "outputs": {
                "strength_effect_records_csv": str(Path(strength_effect_csv).resolve()),
                "strength_effect_summary_json": str(Path(strength_effect_summary_json).resolve()),
                "summary_metrics_json": str(Path(summary_json).resolve()),
                "delta_v_arrays_npz": str(delta_npz_path.resolve()),
                **figure_paths,
            },
            "assumptions": chosen_summary["assumptions"],
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "distractor_strength_effect_independence_experiment",
            "artifact_strength_effect_summary_json": str(strength_effect_summary_json.resolve()),
            "artifact_summary_metrics_json": str(summary_json.resolve()),
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=distractor_strength_effect_independence_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )

    print("\n=== Distractor Strength Effect Independence Summary ===")
    print(f"Triplets: {int(df_strength_records['triplet_id'].nunique())}")
    print(f"Rows: {int(len(df_strength_records))}")
    print(f"Selected model: {chosen_summary['final_selected_model']} ({chosen_summary['model_type']})")
    print(f"Strength records CSV: {strength_effect_csv}")
    print(f"Strength summary JSON: {strength_effect_summary_json}")
    print(f"Summary metrics JSON: {summary_json}")
    print(f"Delta-V arrays NPZ: {delta_npz_path}")
    print(f"Run config: {run_config_path}")


if __name__ == "__main__":
    main()
