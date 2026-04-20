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
from scipy.ndimage import distance_transform_edt
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
from src.experiments.distractor.shared.analysis import compute_delta_v
from src.experiments.distractor.shared.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DELAY1_MS,
    DEFAULT_DELAY_SWEEP_MS,
    DEFAULT_DISTRACTOR_MS,
    DEFAULT_FOREGROUND_THRESHOLD,
    DEFAULT_MAX_PROBES,
    DEFAULT_MAX_TRIPLETS,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_SIM_BINS,
    DEFAULT_PROBE_MS,
    DEFAULT_SAMPLE_MS,
    DEFAULT_SAMPLES_PER_PROBE,
    EPS,
    ExperimentSpec,
    sanitize_delay_sweep as _sanitize_delay_sweep,
    sem as _sem,
    validate_positive as _validate_positive,
)
from src.experiments.distractor.shared.pair_sampling import build_dataset_arrays
from src.experiments.distractor.shared.triplets import (
    TripletMaskBundle,
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
    COLOR_BOX_EDGE,
    COLOR_BOX_FILL,
    FIGSIZE_SINGLE_PANEL_WIDE,
    GEOMETRY_CONDITION_COLORS,
    GRID_ALPHA,
    LINE_WIDTH_PRIMARY,
    MARKER_CIRCLE,
    apply_standard_legend,
    horizontal_panel_figsize,
)

DEFAULT_OUTPUT_DIR = "results/distractor_overlap_gain_geometry_experiment"
DEFAULT_ALPHA_VALUES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
DEFAULT_GEOMETRIES: tuple[str, ...] = ("compact", "fragmented", "shuffled_random")
DEFAULT_MIN_OVERLAP_PIXELS = 30
DEFAULT_MIN_SELECTED_PIXELS = 10
DEFAULT_SAVE_CASE_COUNT = 1

GEOMETRY_COLORS: dict[str, str] = dict(GEOMETRY_CONDITION_COLORS)


@dataclass(frozen=True)
class GeometryConditionSpec:
    name: str
    alpha: float
    geometry: str
    analysis_role: str
    selected_mask: np.ndarray | None
    sample_remove_mask: np.ndarray | None
    distractor_remove_mask: np.ndarray | None
    selected_overlap_pixels: int
    total_overlap_pixels: int
    valid_for_rollout: bool
    invalid_reason: str | None


@dataclass
class ModelFitBundle:
    status: str
    formula: str
    alpha_modeling_mode: str
    model_type: str | None
    fallback_used: bool
    fallback_reason: str | None
    result: object | None
    companion_ols: object | None
    aic: float | None
    bic: float | None
    warnings: list[str]


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


def _condition_name(alpha: float, geometry: str) -> str:
    if str(geometry) == "native":
        return "native_full"
    alpha_token = str(float(alpha)).replace(".", "p")
    return f"alpha_{alpha_token}_{geometry}"


def sanitize_alpha_values(values: Sequence[float]) -> list[float]:
    """Return sorted unique alpha values constrained to (0, 1]."""
    if not values:
        raise ValueError("--alpha-values must contain at least one value.")
    out = sorted(dict.fromkeys(float(value) for value in values))
    for value in out:
        if value <= 0.0 or value > 1.0:
            raise ValueError("All alpha values must satisfy 0 < alpha <= 1.")
    return out


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool))


def build_compact_subset(overlap_mask: np.ndarray, n_selected: int, *, seed: int) -> np.ndarray:
    """Select a high-centrality connected subset inside the original overlap mask."""
    mask = np.asarray(overlap_mask, dtype=bool)
    coords = _coords_from_mask(mask)
    if int(n_selected) <= 0 or coords.size == 0:
        return np.zeros_like(mask, dtype=bool)
    if int(n_selected) >= int(coords.shape[0]):
        return mask.copy()
    rng = np.random.default_rng(int(seed))
    distance_map = distance_transform_edt(mask)
    boundary_distance = distance_map[mask]
    centroid = coords.mean(axis=0, keepdims=True)
    centroid_distance = np.sum((coords - centroid) ** 2, axis=1)
    jitter = rng.random(coords.shape[0])
    order = np.lexsort((jitter, centroid_distance, -boundary_distance))
    chosen = coords[order[: int(n_selected)]]
    out = np.zeros_like(mask, dtype=bool)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def build_fragmented_subset(overlap_mask: np.ndarray, n_selected: int, *, seed: int) -> np.ndarray:
    """Select a maximally dispersed subset using seeded farthest-point sampling."""
    mask = np.asarray(overlap_mask, dtype=bool)
    coords = _coords_from_mask(mask)
    if int(n_selected) <= 0 or coords.size == 0:
        return np.zeros_like(mask, dtype=bool)
    if int(n_selected) >= int(coords.shape[0]):
        return mask.copy()
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(coords.shape[0])
    tie_rank = np.empty(coords.shape[0], dtype=np.int64)
    tie_rank[perm] = np.arange(coords.shape[0], dtype=np.int64)
    selected = np.zeros(coords.shape[0], dtype=bool)
    current = int(perm[0])
    selected[current] = True
    min_sq_dist = np.full(coords.shape[0], np.inf, dtype=np.float64)
    while int(selected.sum()) < int(n_selected):
        sq_dist = np.sum((coords - coords[current]) ** 2, axis=1).astype(np.float64, copy=False)
        min_sq_dist = np.minimum(min_sq_dist, sq_dist)
        min_sq_dist[selected] = -1.0
        max_value = float(np.max(min_sq_dist))
        candidates = np.flatnonzero(np.isclose(min_sq_dist, max_value))
        current = int(candidates[np.argmin(tie_rank[candidates])])
        selected[current] = True
    chosen = coords[selected]
    out = np.zeros_like(mask, dtype=bool)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def build_shuffled_random_subset(overlap_mask: np.ndarray, n_selected: int, *, seed: int) -> np.ndarray:
    """Uniformly sample a reproducible subset inside the original overlap mask."""
    mask = np.asarray(overlap_mask, dtype=bool)
    coords = _coords_from_mask(mask)
    if int(n_selected) <= 0 or coords.size == 0:
        return np.zeros_like(mask, dtype=bool)
    if int(n_selected) >= int(coords.shape[0]):
        return mask.copy()
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(coords.shape[0], size=int(n_selected), replace=False)
    chosen = coords[np.asarray(picked, dtype=np.int64)]
    out = np.zeros_like(mask, dtype=bool)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def build_overlap_subset_mask(overlap_mask: np.ndarray, *, alpha: float, geometry: str, seed: int) -> np.ndarray:
    """Build a same-area subset inside the original overlap mask for a given geometry."""
    mask = np.asarray(overlap_mask, dtype=bool)
    total = int(mask.sum())
    n_selected = int(np.rint(float(alpha) * float(total)))
    if str(geometry) == "native":
        return mask.copy()
    if str(geometry) == "compact":
        return build_compact_subset(mask, n_selected=n_selected, seed=seed)
    if str(geometry) == "fragmented":
        return build_fragmented_subset(mask, n_selected=n_selected, seed=seed)
    if str(geometry) == "shuffled_random":
        return build_shuffled_random_subset(mask, n_selected=n_selected, seed=seed)
    raise ValueError(f"Unsupported geometry: {geometry}")


def build_geometry_condition_table(
    mask_bundle: TripletMaskBundle,
    *,
    alphas: Sequence[float],
    geometries: Sequence[str],
    seed: int,
    min_overlap_pixels: int,
    min_selected_pixels: int,
) -> list[GeometryConditionSpec]:
    """Build all per-triplet geometry conditions, marking invalid ones explicitly."""
    overlap_mask = np.asarray(mask_bundle.sdp_mask, dtype=bool)
    total_overlap_pixels = int(overlap_mask.sum())
    specs: list[GeometryConditionSpec] = []
    if total_overlap_pixels < int(min_overlap_pixels):
        for alpha in alphas:
            n_selected = int(np.rint(float(alpha) * float(total_overlap_pixels)))
            if np.isclose(float(alpha), 1.0):
                specs.append(
                    GeometryConditionSpec(
                        name=_condition_name(alpha, "native"),
                        alpha=float(alpha),
                        geometry="native",
                        analysis_role="filtered_invalid",
                        selected_mask=None,
                        sample_remove_mask=None,
                        distractor_remove_mask=None,
                        selected_overlap_pixels=n_selected,
                        total_overlap_pixels=total_overlap_pixels,
                        valid_for_rollout=False,
                        invalid_reason="overlap_too_small",
                    )
                )
                continue
            for geometry in geometries:
                specs.append(
                    GeometryConditionSpec(
                        name=_condition_name(alpha, geometry),
                        alpha=float(alpha),
                        geometry=str(geometry),
                        analysis_role="filtered_invalid",
                        selected_mask=None,
                        sample_remove_mask=None,
                        distractor_remove_mask=None,
                        selected_overlap_pixels=n_selected,
                        total_overlap_pixels=total_overlap_pixels,
                        valid_for_rollout=False,
                        invalid_reason="overlap_too_small",
                    )
                )
        return specs
    for alpha in alphas:
        n_selected = int(np.rint(float(alpha) * float(total_overlap_pixels)))
        if np.isclose(float(alpha), 1.0):
            if n_selected < int(min_selected_pixels):
                specs.append(
                    GeometryConditionSpec(
                        name=_condition_name(alpha, "native"),
                        alpha=float(alpha),
                        geometry="native",
                        analysis_role="filtered_invalid",
                        selected_mask=None,
                        sample_remove_mask=None,
                        distractor_remove_mask=None,
                        selected_overlap_pixels=n_selected,
                        total_overlap_pixels=total_overlap_pixels,
                        valid_for_rollout=False,
                        invalid_reason="selected_overlap_too_small",
                    )
                )
                continue
            specs.append(
                GeometryConditionSpec(
                    name=_condition_name(alpha, "native"),
                    alpha=float(alpha),
                    geometry="native",
                    analysis_role="native_reference",
                    selected_mask=overlap_mask.copy(),
                    sample_remove_mask=np.zeros_like(overlap_mask, dtype=bool),
                    distractor_remove_mask=np.zeros_like(overlap_mask, dtype=bool),
                    selected_overlap_pixels=n_selected,
                    total_overlap_pixels=total_overlap_pixels,
                    valid_for_rollout=True,
                    invalid_reason=None,
                )
            )
            continue
        if n_selected < int(min_selected_pixels):
            for geometry in geometries:
                specs.append(
                    GeometryConditionSpec(
                        name=_condition_name(alpha, geometry),
                        alpha=float(alpha),
                        geometry=str(geometry),
                        analysis_role="filtered_invalid",
                        selected_mask=None,
                        sample_remove_mask=None,
                        distractor_remove_mask=None,
                        selected_overlap_pixels=n_selected,
                        total_overlap_pixels=total_overlap_pixels,
                        valid_for_rollout=False,
                        invalid_reason="selected_overlap_too_small",
                    )
                )
            continue
        for geometry in geometries:
            selected_mask = build_overlap_subset_mask(
                overlap_mask,
                alpha=float(alpha),
                geometry=str(geometry),
                seed=mix_seed(int(seed), int(round(float(alpha) * 1000.0)), sum(ord(ch) for ch in str(geometry))),
            )
            remove_mask = overlap_mask & ~selected_mask
            specs.append(
                GeometryConditionSpec(
                    name=_condition_name(alpha, geometry),
                    alpha=float(alpha),
                    geometry=str(geometry),
                    analysis_role="main",
                    selected_mask=selected_mask.astype(bool, copy=False),
                    sample_remove_mask=remove_mask.astype(bool, copy=False),
                    distractor_remove_mask=remove_mask.astype(bool, copy=False),
                    selected_overlap_pixels=int(selected_mask.sum()),
                    total_overlap_pixels=total_overlap_pixels,
                    valid_for_rollout=True,
                    invalid_reason=None,
                )
            )
    return specs


def _mean_pairwise_nearest_neighbor_distance(mask: np.ndarray) -> float:
    coords = _coords_from_mask(mask).astype(np.float64, copy=False)
    if coords.shape[0] <= 1:
        return 0.0
    diff = coords[:, None, :] - coords[None, :, :]
    sq_dist = np.sum(diff**2, axis=-1)
    sq_dist[sq_dist <= 0.0] = np.inf
    return float(np.mean(np.sqrt(np.min(sq_dist, axis=1))))


def _build_invalid_record(*, triplet_row, spec: GeometryConditionSpec) -> dict[str, object]:
    return {
        "triplet_id": int(triplet_row.triplet_id),
        "sample_id": int(triplet_row.sample_id),
        "distractor_id": int(triplet_row.distractor_id),
        "probe_id": int(triplet_row.probe_id),
        "sample_label": int(triplet_row.sample_label),
        "distractor_label": int(triplet_row.distractor_label),
        "probe_label": int(triplet_row.probe_label),
        "alpha": float(spec.alpha),
        "geometry": str(spec.geometry),
        "condition_name": str(spec.name),
        "analysis_role": str(spec.analysis_role),
        "delay_ms": float("nan"),
        "selected_overlap_pixels": int(spec.selected_overlap_pixels),
        "total_overlap_pixels": int(spec.total_overlap_pixels),
        "M": float("nan"),
        "M_max": float("nan"),
        "tau_peak": float("nan"),
        "gain_per_area": float("nan"),
        "M_norm": float("nan"),
        "reference_status": str(spec.invalid_reason),
    }


def _prepare_model_dataframe(
    df_model: pd.DataFrame,
    *,
    outcome_col: str,
    geometry_order: Sequence[str],
    alpha_mode: str,
) -> tuple[pd.DataFrame, str]:
    data = df_model.copy()
    data = data[np.isfinite(data[outcome_col].to_numpy(dtype=np.float64))].copy()
    if data.empty:
        raise ValueError(f"No valid rows available for model outcome {outcome_col}.")
    data["geometry"] = pd.Categorical(data["geometry"], categories=list(geometry_order), ordered=True)
    data["alpha_c"] = data["alpha"].astype(float) - float(data["alpha"].mean())
    data["alpha_label"] = pd.Categorical(data["alpha"].map(lambda value: f"{float(value):.2f}"))
    if str(alpha_mode) == "categorical":
        formula = f"{outcome_col} ~ C(alpha_label) + C(geometry) + C(alpha_label):C(geometry)"
    else:
        formula = f"{outcome_col} ~ alpha_c + C(geometry) + alpha_c:C(geometry)"
    return data, formula


def _fit_model_bundle(
    df_model: pd.DataFrame,
    *,
    formula: str,
    alpha_modeling_mode: str,
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
                alpha_modeling_mode=alpha_modeling_mode,
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
        fallback_reason = "statsmodels is unavailable; model fitting skipped"
    if not bool(cluster_robust_fallback) or not HAS_STATSMODELS:
        return ModelFitBundle(
            status="fit_failed",
            formula=formula,
            alpha_modeling_mode=alpha_modeling_mode,
            model_type=None,
            fallback_used=bool(cluster_robust_fallback),
            fallback_reason=fallback_reason,
            result=None,
            companion_ols=None,
            aic=None,
            bic=None,
            warnings=mixed_warnings,
        )
    try:
        ols_result = smf.ols(formula, data=df_model).fit(cov_type="cluster", cov_kwds={"groups": df_model["triplet_id"]})
        companion_ols = smf.ols(formula, data=df_model).fit()
    except Exception as exc:
        message = str(exc) if fallback_reason is None else f"{fallback_reason} | OLS fallback failed: {exc}"
        return ModelFitBundle(
            status="fit_failed",
            formula=formula,
            alpha_modeling_mode=alpha_modeling_mode,
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
        alpha_modeling_mode=alpha_modeling_mode,
        model_type="OLS_cluster_robust",
        fallback_used=True,
        fallback_reason=fallback_reason,
        result=ols_result,
        companion_ols=companion_ols,
        aic=_safe_float(getattr(companion_ols, "aic", None)),
        bic=_safe_float(getattr(companion_ols, "bic", None)),
        warnings=mixed_warnings,
    )


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
        return {"label": label, "status": "test_error", "param_names": param_names, "error": str(exc)}
    return {
        "label": label,
        "status": "ok",
        "statistic": _safe_float(getattr(test_result, "statistic", None)),
        "p_value": _safe_float(getattr(test_result, "pvalue", None)),
        "df_constraint": len(param_names),
        "param_names": param_names,
    }


def _partial_r2_from_formulas(df_model: pd.DataFrame, *, full_formula: str, reduced_formula: str, outcome_col: str) -> dict[str, object]:
    del outcome_col
    if not HAS_STATSMODELS:
        return {"status": "statsmodels_unavailable", "partial_r2": None}
    try:
        full_fit = smf.ols(full_formula, data=df_model).fit()
        reduced_fit = smf.ols(reduced_formula, data=df_model).fit()
    except Exception as exc:
        return {"status": "fit_error", "partial_r2": None, "error": str(exc)}
    sse_full = float(np.sum(np.square(np.asarray(full_fit.resid, dtype=np.float64))))
    sse_reduced = float(np.sum(np.square(np.asarray(reduced_fit.resid, dtype=np.float64))))
    if sse_reduced <= float(EPS):
        return {"status": "degenerate_reduced_model", "partial_r2": None}
    partial_r2 = max(0.0, min(1.0, (sse_reduced - sse_full) / sse_reduced))
    return {
        "status": "ok",
        "partial_r2": float(partial_r2),
        "full_r2": _safe_float(getattr(full_fit, "rsquared", None)),
        "reduced_r2": _safe_float(getattr(reduced_fit, "rsquared", None)),
    }


def compute_effect_size_summary(df_model: pd.DataFrame, *, outcome_col: str, alpha_mode: str) -> dict[str, object]:
    if str(alpha_mode) == "categorical":
        full_formula = f"{outcome_col} ~ C(alpha_label) + C(geometry) + C(alpha_label):C(geometry)"
        alpha_formula = f"{outcome_col} ~ C(geometry) + C(alpha_label):C(geometry)"
        geometry_formula = f"{outcome_col} ~ C(alpha_label) + C(alpha_label):C(geometry)"
        interaction_formula = f"{outcome_col} ~ C(alpha_label) + C(geometry)"
    else:
        full_formula = f"{outcome_col} ~ alpha_c + C(geometry) + alpha_c:C(geometry)"
        alpha_formula = f"{outcome_col} ~ C(geometry) + alpha_c:C(geometry)"
        geometry_formula = f"{outcome_col} ~ alpha_c + alpha_c:C(geometry)"
        interaction_formula = f"{outcome_col} ~ alpha_c + C(geometry)"
    summary = {
        "alpha_main": _partial_r2_from_formulas(df_model, full_formula=full_formula, reduced_formula=alpha_formula, outcome_col=outcome_col),
        "geometry_main": _partial_r2_from_formulas(df_model, full_formula=full_formula, reduced_formula=geometry_formula, outcome_col=outcome_col),
        "interaction": _partial_r2_from_formulas(df_model, full_formula=full_formula, reduced_formula=interaction_formula, outcome_col=outcome_col),
    }
    ranking = [{"effect": effect_name, "partial_r2": payload.get("partial_r2")} for effect_name, payload in summary.items()]
    ranking.sort(key=lambda item: (-1.0 if item["partial_r2"] is None else -float(item["partial_r2"]), item["effect"]))
    summary["relative_strength_ranking"] = ranking
    return summary


def fit_alpha_geometry_model(
    df_records: pd.DataFrame,
    *,
    outcome_col: str,
    geometry_order: Sequence[str] = DEFAULT_GEOMETRIES,
    alpha_mode: str = "continuous",
    cluster_robust_fallback: bool = True,
) -> dict[str, object]:
    df_model, formula = _prepare_model_dataframe(df_records, outcome_col=outcome_col, geometry_order=geometry_order, alpha_mode=alpha_mode)
    bundle = _fit_model_bundle(
        df_model,
        formula=formula,
        alpha_modeling_mode=str(alpha_mode),
        cluster_robust_fallback=bool(cluster_robust_fallback),
        allow_mixedlm=True,
    )
    if str(alpha_mode) == "categorical":
        alpha_pred = lambda name: name.startswith("C(alpha_label)[") and ":C(geometry)" not in name
        interaction_pred = lambda name: name.startswith("C(alpha_label)[") and ":C(geometry)" in name
    else:
        alpha_pred = lambda name: name == "alpha_c"
        interaction_pred = lambda name: name.startswith("alpha_c:C(geometry)[")
    summary = {
        "status": bundle.status,
        "outcome": str(outcome_col),
        "formula": bundle.formula,
        "alpha_modeling_mode": str(alpha_mode),
        "model_type": bundle.model_type,
        "fallback_used": bool(bundle.fallback_used),
        "fallback_reason": bundle.fallback_reason,
        "aic": bundle.aic,
        "bic": bundle.bic,
        "n_rows": int(len(df_model)),
        "n_triplets": int(df_model["triplet_id"].nunique()),
        "alpha_main_effect": _wald_test_for_params(bundle.result, alpha_pred, label="alpha_main_effect"),
        "geometry_main_effect": _wald_test_for_params(bundle.result, lambda name: name.startswith("C(geometry)[") and ":C(" not in name, label="geometry_main_effect"),
        "interaction_effect": _wald_test_for_params(bundle.result, interaction_pred, label="interaction_effect"),
        "effect_sizes": compute_effect_size_summary(df_model, outcome_col=outcome_col, alpha_mode=str(alpha_mode)),
        "warnings": [str(item) for item in bundle.warnings],
    }
    return {"data": df_model, "bundle": bundle, "summary": summary}


def _profile_similarity_to_grand_mean(subset: pd.DataFrame, grand_profile: pd.Series, *, delay_ms_values: Sequence[float]) -> dict[str, object]:
    local = subset.groupby("delay_ms", sort=True)["M_norm"].mean()
    x_vals = []
    y_vals = []
    for delay_ms in delay_ms_values:
        if float(delay_ms) not in local.index or float(delay_ms) not in grand_profile.index:
            continue
        x_vals.append(float(local.loc[float(delay_ms)]))
        y_vals.append(float(grand_profile.loc[float(delay_ms)]))
    if len(x_vals) < 2:
        return {"status": "insufficient_points", "profile_corr_to_grand_mean": None, "profile_rmse_to_grand_mean": None}
    xx = np.asarray(x_vals, dtype=np.float64)
    yy = np.asarray(y_vals, dtype=np.float64)
    corr = float(np.corrcoef(xx, yy)[0, 1]) if np.std(xx) > EPS and np.std(yy) > EPS else float("nan")
    rmse = float(np.sqrt(np.mean((xx - yy) ** 2)))
    return {"status": "ok", "profile_corr_to_grand_mean": corr, "profile_rmse_to_grand_mean": rmse}


def summarize_normalized_profiles(df_records: pd.DataFrame, *, delay_ms_values: Sequence[float]) -> dict[str, object]:
    valid = df_records[np.isfinite(df_records["M_norm"].to_numpy(dtype=np.float64))].copy()
    if valid.empty:
        return {"status": "no_valid_profiles", "condition_profiles": [], "condition_similarity": []}
    grouped_rows: list[dict[str, object]] = []
    for (alpha, geometry, delay_ms), subset in valid.groupby(["alpha", "geometry", "delay_ms"], sort=True):
        grouped_rows.append(
            {
                "alpha": float(alpha),
                "geometry": str(geometry),
                "delay_ms": float(delay_ms),
                "n_triplets": int(subset["triplet_id"].nunique()),
                "mean_M_norm": float(subset["M_norm"].mean()),
                "sem_M_norm": _sem(subset["M_norm"].to_numpy(dtype=np.float64)),
            }
        )
    grand_profile = valid.groupby("delay_ms", sort=True)["M_norm"].mean()
    similarity_rows: list[dict[str, object]] = []
    for (alpha, geometry), subset in valid.groupby(["alpha", "geometry"], sort=True):
        payload = _profile_similarity_to_grand_mean(subset, grand_profile, delay_ms_values=delay_ms_values)
        similarity_rows.append({"alpha": float(alpha), "geometry": str(geometry), **payload})
    mean_corr = np.nanmean([row["profile_corr_to_grand_mean"] for row in similarity_rows if row.get("profile_corr_to_grand_mean") is not None])
    mean_rmse = np.nanmean([row["profile_rmse_to_grand_mean"] for row in similarity_rows if row.get("profile_rmse_to_grand_mean") is not None])
    return {
        "status": "ok",
        "condition_profiles": grouped_rows,
        "condition_similarity": similarity_rows,
        "collapse_summary": {
            "mean_profile_corr_to_grand_mean": None if not np.isfinite(mean_corr) else float(mean_corr),
            "mean_profile_rmse_to_grand_mean": None if not np.isfinite(mean_rmse) else float(mean_rmse),
        },
    }


def summarize_area_normalized_gain(df_curve_summary: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for (alpha, geometry), subset in df_curve_summary.groupby(["alpha", "geometry"], sort=True):
        rows.append(
            {
                "alpha": float(alpha),
                "geometry": str(geometry),
                "n_triplets": int(subset["triplet_id"].nunique()),
                "mean_gain_per_area": float(subset["gain_per_area"].mean()),
                "sem_gain_per_area": _sem(subset["gain_per_area"].to_numpy(dtype=np.float64)),
            }
        )
    return {"status": "ok", "group_summary": rows}


def _interpret_hypotheses(
    *,
    mmax_summary: Mapping[str, object],
    gain_per_area_summary: Mapping[str, object],
    profile_summary: Mapping[str, object],
) -> dict[str, object]:
    mmax_effects = mmax_summary.get("effect_sizes", {})
    alpha_p = mmax_summary.get("alpha_main_effect", {}).get("p_value")
    geometry_p = mmax_summary.get("geometry_main_effect", {}).get("p_value")
    interaction_p = mmax_summary.get("interaction_effect", {}).get("p_value")
    gain_geometry_p = gain_per_area_summary.get("geometry_main_effect", {}).get("p_value")
    gain_interaction_p = gain_per_area_summary.get("interaction_effect", {}).get("p_value")
    alpha_r2 = mmax_effects.get("alpha_main", {}).get("partial_r2")
    geometry_r2 = mmax_effects.get("geometry_main", {}).get("partial_r2")
    interaction_r2 = mmax_effects.get("interaction", {}).get("partial_r2")
    collapse_corr = profile_summary.get("collapse_summary", {}).get("mean_profile_corr_to_grand_mean")
    collapse_rmse = profile_summary.get("collapse_summary", {}).get("mean_profile_rmse_to_grand_mean")

    area_status = "mixed"
    if alpha_p is not None and float(alpha_p) < 0.05 and (geometry_p is None or float(geometry_p) >= 0.05) and (interaction_p is None or float(interaction_p) >= 0.05):
        area_status = "supported"
    elif alpha_r2 is not None and geometry_r2 is not None and interaction_r2 is not None and float(alpha_r2) <= max(float(geometry_r2), float(interaction_r2)):
        area_status = "not_supported"

    efficiency_status = "mixed"
    if (gain_geometry_p is None or float(gain_geometry_p) >= 0.05) and (gain_interaction_p is None or float(gain_interaction_p) >= 0.05):
        efficiency_status = "supported"
    elif (gain_geometry_p is not None and float(gain_geometry_p) < 0.05) or (gain_interaction_p is not None and float(gain_interaction_p) < 0.05):
        efficiency_status = "not_supported"

    geometry_status = "mixed"
    if (
        (geometry_p is not None and float(geometry_p) < 0.05)
        or (interaction_p is not None and float(interaction_p) < 0.05)
        or (gain_geometry_p is not None and float(gain_geometry_p) < 0.05)
        or (gain_interaction_p is not None and float(gain_interaction_p) < 0.05)
    ):
        geometry_status = "supported"
    elif area_status == "supported" and efficiency_status == "supported":
        geometry_status = "not_supported"

    delay_note = "mixed"
    if collapse_corr is not None and collapse_rmse is not None and float(collapse_corr) >= 0.9 and float(collapse_rmse) <= 0.12:
        delay_note = "profiles_nearly_collapse"
    elif collapse_corr is not None and float(collapse_corr) >= 0.75:
        delay_note = "profiles_partially_collapse"
    else:
        delay_note = "profiles_do_not_collapse"
    return {
        "area_dominant": {"status": area_status},
        "constant_efficiency_per_area": {"status": efficiency_status},
        "high_dimensional_geometry": {"status": geometry_status},
        "normalized_profile_readout": {"status": delay_note},
    }


def plot_experiment_schematic(*, overlap_mask: np.ndarray, subset_examples: Mapping[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    ordered_panels = [
        ("original_SDP", np.asarray(overlap_mask, dtype=float), "magma"),
        ("native_full", np.asarray(subset_examples.get("native_full", overlap_mask), dtype=float), "magma"),
        ("compact_0.25", np.asarray(subset_examples.get("compact_0.25", np.zeros_like(overlap_mask)), dtype=float), "viridis"),
        ("compact_0.50", np.asarray(subset_examples.get("compact_0.50", np.zeros_like(overlap_mask)), dtype=float), "viridis"),
        ("compact_0.75", np.asarray(subset_examples.get("compact_0.75", np.zeros_like(overlap_mask)), dtype=float), "viridis"),
        ("fragmented_0.50", np.asarray(subset_examples.get("fragmented_0.50", np.zeros_like(overlap_mask)), dtype=float), "viridis"),
        ("shuffled_0.50", np.asarray(subset_examples.get("shuffled_0.50", np.zeros_like(overlap_mask)), dtype=float), "viridis"),
    ]
    fig, axes = plt.subplots(1, len(ordered_panels), figsize=horizontal_panel_figsize(len(ordered_panels), panel_width=5.16, height=2.8), squeeze=False)
    for ax, (title, panel, cmap) in zip(axes[0], ordered_panels):
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Subset selection stays inside the original SDP overlap", y=1.03)
    fig.tight_layout()
    return fig


def plot_delay_sweep_curves(df_records: pd.DataFrame, *, alpha_values: Sequence[float]) -> plt.Figure:
    apply_publication_style()
    n_panels = len(alpha_values)
    fig, axes = plt.subplots(1, n_panels, figsize=horizontal_panel_figsize(n_panels), squeeze=False, sharey=True)
    summary = df_records[np.isfinite(df_records["delay_ms"].to_numpy(dtype=np.float64))].copy()
    for ax, alpha in zip(axes[0], alpha_values):
        panel = summary[summary["alpha"] == float(alpha)]
        geometries = ["native"] if np.isclose(float(alpha), 1.0) else list(DEFAULT_GEOMETRIES)
        for geometry in geometries:
            subset = panel[panel["geometry"] == str(geometry)]
            if subset.empty:
                continue
            curve = subset.groupby("delay_ms", sort=True)["M"].mean().reset_index()
            sem = [_sem(subset[subset["delay_ms"] == float(delay_ms)]["M"].to_numpy(dtype=np.float64)) for delay_ms in curve["delay_ms"].to_numpy(dtype=np.float64)]
            x = curve["delay_ms"].to_numpy(dtype=np.float64)
            y = curve["M"].to_numpy(dtype=np.float64)
            y_sem = np.asarray(sem, dtype=np.float64)
            ax.plot(x, y, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=GEOMETRY_COLORS[str(geometry)], label=str(geometry))
            ax.fill_between(x, y - y_sem, y + y_sem, color=GEOMETRY_COLORS[str(geometry)], alpha=ALPHA_FILL)
        ax.set_title(f"alpha={float(alpha):.2f}")
        ax.set_xlabel("delay2 (ms)")
        ax.grid(alpha=GRID_ALPHA)
    axes[0, 0].set_ylabel("M(delay)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        apply_standard_legend(axes[0, 0], handles=handles, labels=labels, compact=True)
    fig.tight_layout()
    return fig


def plot_peak_gain_vs_area_fraction(df_curve_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    main = df_curve_summary[df_curve_summary["analysis_role"] == "main"].copy()
    for geometry in DEFAULT_GEOMETRIES:
        subset = main[main["geometry"] == str(geometry)]
        if subset.empty:
            continue
        summary = subset.groupby("alpha", sort=True)["M_max"].mean().reset_index()
        sem = [_sem(subset[subset["alpha"] == float(alpha)]["M_max"].to_numpy(dtype=np.float64)) for alpha in summary["alpha"].to_numpy(dtype=np.float64)]
        x = summary["alpha"].to_numpy(dtype=np.float64)
        y = summary["M_max"].to_numpy(dtype=np.float64)
        y_sem = np.asarray(sem, dtype=np.float64)
        ax.plot(x, y, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=GEOMETRY_COLORS[str(geometry)], label=str(geometry))
        ax.fill_between(x, y - y_sem, y + y_sem, color=GEOMETRY_COLORS[str(geometry)], alpha=ALPHA_FILL)
    native = df_curve_summary[df_curve_summary["analysis_role"] == "native_reference"].copy()
    if not native.empty:
        ax.scatter([1.0], [float(native["M_max"].mean())], s=54, color=GEOMETRY_COLORS["native"], label="native")
    ax.set_xlabel("alpha")
    ax.set_ylabel("M_max")
    ax.set_title("Peak gain vs overlap area fraction")
    ax.grid(alpha=GRID_ALPHA)
    apply_standard_legend(ax, fontsize=9)
    fig.tight_layout()
    return fig


def plot_area_normalized_gain(df_curve_summary: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    alpha_values = sorted(df_curve_summary[df_curve_summary["analysis_role"] == "main"]["alpha"].unique().tolist())
    n_panels = max(1, len(alpha_values))
    fig, axes = plt.subplots(1, n_panels, figsize=horizontal_panel_figsize(n_panels), squeeze=False, sharey=True)
    for ax, alpha in zip(axes[0], alpha_values):
        subset = df_curve_summary[(df_curve_summary["analysis_role"] == "main") & (df_curve_summary["alpha"] == float(alpha))]
        positions = np.arange(len(DEFAULT_GEOMETRIES), dtype=np.float64)
        box_data = []
        for geometry in DEFAULT_GEOMETRIES:
            values = subset[subset["geometry"] == geometry]["gain_per_area"].to_numpy(dtype=np.float64)
            box_data.append(values[np.isfinite(values)])
        ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True, boxprops={"facecolor": COLOR_BOX_FILL, "edgecolor": COLOR_BOX_EDGE})
        for pos, geometry, values in zip(positions, DEFAULT_GEOMETRIES, box_data):
            if values.size == 0:
                continue
            jitter = np.linspace(-0.10, 0.10, num=values.size) if values.size > 1 else np.asarray([0.0], dtype=np.float64)
            ax.scatter(pos + jitter, values, s=18, alpha=ALPHA_SCATTER, color=GEOMETRY_COLORS[str(geometry)])
        ax.set_title(f"alpha={float(alpha):.2f}")
        ax.set_xticks(positions)
        ax.set_xticklabels(DEFAULT_GEOMETRIES, rotation=25, ha="right")
        ax.grid(alpha=GRID_ALPHA, axis="y")
    axes[0, 0].set_ylabel("M_max / selected_overlap_pixels")
    fig.tight_layout()
    return fig


def plot_normalized_temporal_profile(df_records: pd.DataFrame, *, alpha_values: Sequence[float]) -> plt.Figure:
    apply_publication_style()
    n_panels = len(alpha_values)
    fig, axes = plt.subplots(1, n_panels, figsize=horizontal_panel_figsize(n_panels), squeeze=False, sharey=True)
    valid = df_records[np.isfinite(df_records["M_norm"].to_numpy(dtype=np.float64))].copy()
    for ax, alpha in zip(axes[0], alpha_values):
        panel = valid[valid["alpha"] == float(alpha)]
        geometries = ["native"] if np.isclose(float(alpha), 1.0) else list(DEFAULT_GEOMETRIES)
        for geometry in geometries:
            subset = panel[panel["geometry"] == str(geometry)]
            if subset.empty:
                continue
            curve = subset.groupby("delay_ms", sort=True)["M_norm"].mean().reset_index()
            sem = [_sem(subset[subset["delay_ms"] == float(delay_ms)]["M_norm"].to_numpy(dtype=np.float64)) for delay_ms in curve["delay_ms"].to_numpy(dtype=np.float64)]
            x = curve["delay_ms"].to_numpy(dtype=np.float64)
            y = curve["M_norm"].to_numpy(dtype=np.float64)
            y_sem = np.asarray(sem, dtype=np.float64)
            ax.plot(x, y, marker=MARKER_CIRCLE, linewidth=LINE_WIDTH_PRIMARY, color=GEOMETRY_COLORS[str(geometry)], label=str(geometry))
            ax.fill_between(x, y - y_sem, y + y_sem, color=GEOMETRY_COLORS[str(geometry)], alpha=ALPHA_FILL)
        ax.set_title(f"alpha={float(alpha):.2f}")
        ax.set_xlabel("delay2 (ms)")
        ax.grid(alpha=GRID_ALPHA)
    axes[0, 0].set_ylabel("M(delay) / M_max")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        apply_standard_legend(axes[0, 0], handles=handles, labels=labels, compact=True)
    fig.tight_layout()
    return fig


def plot_effect_size_comparison(effect_size_map: Mapping[str, Mapping[str, object]]) -> plt.Figure:
    apply_publication_style()
    outcomes = ("M_max", "tau_peak")
    effects = ("alpha_main", "geometry_main", "interaction")
    x = np.arange(len(effects), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_PANEL_WIDE)
    for idx, outcome in enumerate(outcomes):
        payload = effect_size_map.get(str(outcome), {})
        values = [float(payload.get(effect_name, {}).get("partial_r2") or 0.0) for effect_name in effects]
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=str(outcome))
    ax.set_xticks(x)
    ax.set_xticklabels(["alpha", "geometry", "alpha×geometry"])
    ax.set_ylabel("Partial R^2")
    ax.set_title("Effect size comparison")
    ax.grid(alpha=GRID_ALPHA, axis="y")
    apply_standard_legend(ax)
    fig.tight_layout()
    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distractor overlap gain geometry experiment.")
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
    parser.add_argument("--alpha-values", type=float, nargs="+", default=list(DEFAULT_ALPHA_VALUES))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-triplets", type=int, default=DEFAULT_MAX_TRIPLETS)
    parser.add_argument("--num-sim-bins", type=int, default=DEFAULT_NUM_SIM_BINS)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--min-overlap-pixels", type=int, default=DEFAULT_MIN_OVERLAP_PIXELS)
    parser.add_argument("--min-selected-pixels", type=int, default=DEFAULT_MIN_SELECTED_PIXELS)
    parser.add_argument("--save-case-count", type=int, default=DEFAULT_SAVE_CASE_COUNT)
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
    _validate_positive("--sample-ms", float(args.sample_ms))
    _validate_positive("--delay1-ms", float(args.delay1_ms), allow_zero=True)
    _validate_positive("--distractor-ms", float(args.distractor_ms))
    _validate_positive("--probe-ms", float(args.probe_ms))
    _validate_positive("--min-overlap-pixels", int(args.min_overlap_pixels))
    _validate_positive("--min-selected-pixels", int(args.min_selected_pixels))
    _validate_positive("--save-case-count", int(args.save_case_count), allow_zero=True)
    delay_ms_values = _sanitize_delay_sweep(args.delay_sweep_ms)
    alpha_values = sanitize_alpha_values(args.alpha_values)

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=float(args.sample_ms),
        delay1_ms=float(args.delay1_ms),
        distractor_ms=float(args.distractor_ms),
        delay2_ms=float(max(delay_ms_values)),
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
    triplet_condition_map: dict[int, dict[str, GeometryConditionSpec]] = {}
    all_condition_names: list[str] = []
    invalid_rows: list[dict[str, object]] = []
    filtered_triplets_small_overlap: list[int] = []
    invalid_alpha_triplets: dict[str, set[int]] = {f"{float(alpha):.2f}": set() for alpha in alpha_values}
    for triplet_row in df_triplets.itertuples(index=False):
        triplet_id = int(triplet_row.triplet_id)
        mask_bundle = build_probe_relevant_masks_for_triplet(
            sample_image=images[int(triplet_row.sample_id)],
            distractor_image=images[int(triplet_row.distractor_id)],
            probe_image=images[int(triplet_row.probe_id)],
            foreground_threshold=float(args.foreground_threshold),
            use_dilated_overlap=False,
            dilation_radius=0,
            seed=mix_seed(int(args.seed), triplet_id, int(triplet_row.sample_id), int(triplet_row.distractor_id), int(triplet_row.probe_id)),
            num_control_candidates=1,
        )
        mask_records.append(mask_bundle)
        condition_specs = build_geometry_condition_table(
            mask_bundle,
            alphas=alpha_values,
            geometries=DEFAULT_GEOMETRIES,
            seed=mix_seed(int(args.seed), triplet_id),
            min_overlap_pixels=int(args.min_overlap_pixels),
            min_selected_pixels=int(args.min_selected_pixels),
        )
        triplet_condition_map[triplet_id] = {spec_row.name: spec_row for spec_row in condition_specs}
        all_condition_names.extend(spec_row.name for spec_row in condition_specs)
        if int(mask_bundle.sdp_mask.sum()) < int(args.min_overlap_pixels):
            filtered_triplets_small_overlap.append(triplet_id)
        for spec_row in condition_specs:
            if not spec_row.valid_for_rollout:
                invalid_rows.append(_build_invalid_record(triplet_row=triplet_row, spec=spec_row))
                if spec_row.invalid_reason in {"selected_overlap_too_small", "overlap_too_small"}:
                    key = f"{float(spec_row.alpha):.2f}"
                    invalid_alpha_triplets.setdefault(key, set()).add(triplet_id)
    ordered_condition_names = list(dict.fromkeys(all_condition_names))
    all_triplet_ids = df_triplets["triplet_id"].astype(int).tolist()
    valid_condition_names = [
        name
        for name in ordered_condition_names
        if any(triplet_condition_map[int(triplet_id)].get(name) and triplet_condition_map[int(triplet_id)][name].valid_for_rollout for triplet_id in all_triplet_ids)
    ]

    curve_cache: dict[tuple[int, str], dict[str, object]] = {}
    batch_starts = range(0, len(df_triplets), int(args.batch_size))
    total_batches = math.ceil(len(df_triplets) / int(args.batch_size))
    for batch_start in tqdm(batch_starts, total=total_batches, desc="Overlap gain geometry"):
        batch_df = df_triplets.iloc[batch_start : batch_start + int(args.batch_size)].copy().reset_index(drop=True)
        sample_spikes, distractor_spikes, probe_spikes = prepare_triplet_spike_batch(
            images=images,
            batch_df=batch_df,
            encoder=encoder,
            spec=spec,
            device=device,
        )
        batch_triplet_ids = batch_df["triplet_id"].astype(int).tolist()
        for condition_name in valid_condition_names:
            valid_positions: list[int] = []
            valid_specs: list[GeometryConditionSpec] = []
            for pos, triplet_id in enumerate(batch_triplet_ids):
                spec_row = triplet_condition_map[int(triplet_id)].get(condition_name)
                if spec_row is None or not spec_row.valid_for_rollout:
                    continue
                valid_positions.append(pos)
                valid_specs.append(spec_row)
            if not valid_positions:
                continue
            select_index = torch.as_tensor(valid_positions, dtype=torch.long, device=sample_spikes.device)
            sample_subset = sample_spikes.index_select(0, select_index)
            distractor_subset = distractor_spikes.index_select(0, select_index)
            probe_subset = probe_spikes.index_select(0, select_index)
            sample_mask = torch.as_tensor(np.stack([spec_row.sample_remove_mask for spec_row in valid_specs], axis=0), dtype=torch.bool, device=sample_spikes.device)
            distractor_mask = torch.as_tensor(np.stack([spec_row.distractor_remove_mask for spec_row in valid_specs], axis=0), dtype=torch.bool, device=sample_spikes.device)
            for delay_ms in delay_ms_values:
                delay2_steps = int(round((float(delay_ms) * ms) / spec.dt))
                dynamic = run_overlap_perturbed_distractor_task(
                    net=net,
                    sample_spikes=sample_subset,
                    distractor_spikes=distractor_subset,
                    probe_spikes=probe_subset,
                    delay1_steps=spec.delay1_steps,
                    delay2_steps=delay2_steps,
                    stsp_mode="dynamic",
                    readout_step=readout_step,
                    sample_input_mask=sample_mask,
                    distractor_input_mask=distractor_mask,
                    phase_reset=spec.phase_reset,
                )
                static = run_overlap_perturbed_distractor_task(
                    net=net,
                    sample_spikes=sample_subset,
                    distractor_spikes=distractor_subset,
                    probe_spikes=probe_subset,
                    delay1_steps=spec.delay1_steps,
                    delay2_steps=delay2_steps,
                    stsp_mode="static_frozen",
                    readout_step=readout_step,
                    sample_input_mask=sample_mask,
                    distractor_input_mask=distractor_mask,
                    phase_reset=spec.phase_reset,
                )
                dynamic_grouped = np.asarray(dynamic.grouped_voltage, dtype=np.float64)
                static_grouped = np.asarray(static.grouped_voltage, dtype=np.float64)
                delta_v = compute_delta_v(dynamic_grouped, static_grouped)
                for local_idx, batch_pos in enumerate(valid_positions):
                    triplet_row = batch_df.iloc[int(batch_pos)]
                    triplet_id = int(triplet_row["triplet_id"])
                    spec_row = valid_specs[local_idx]
                    cache_key = (triplet_id, str(condition_name))
                    if cache_key not in curve_cache:
                        curve_cache[cache_key] = {
                            "triplet_meta": {
                                "triplet_id": triplet_id,
                                "sample_id": int(triplet_row["sample_id"]),
                                "distractor_id": int(triplet_row["distractor_id"]),
                                "probe_id": int(triplet_row["probe_id"]),
                                "sample_label": int(triplet_row["sample_label"]),
                                "distractor_label": int(triplet_row["distractor_label"]),
                                "probe_label": int(triplet_row["probe_label"]),
                            },
                            "condition_meta": {
                                "alpha": float(spec_row.alpha),
                                "geometry": str(spec_row.geometry),
                                "condition_name": str(spec_row.name),
                                "analysis_role": str(spec_row.analysis_role),
                                "selected_overlap_pixels": int(spec_row.selected_overlap_pixels),
                                "total_overlap_pixels": int(spec_row.total_overlap_pixels),
                                "selected_mask": np.asarray(spec_row.selected_mask, dtype=bool),
                            },
                            "delay_ms": [],
                            "delta_v": [],
                            "M": [],
                            "condition_grouped_voltage": [],
                            "static_grouped_voltage": [],
                        }
                    curve_cache[cache_key]["delay_ms"].append(float(delay_ms))
                    curve_cache[cache_key]["delta_v"].append(np.asarray(delta_v[local_idx], dtype=np.float32))
                    curve_cache[cache_key]["M"].append(float(np.linalg.norm(np.asarray(delta_v[local_idx], dtype=np.float64))))
                    curve_cache[cache_key]["condition_grouped_voltage"].append(np.asarray(dynamic_grouped[local_idx], dtype=np.float32))
                    curve_cache[cache_key]["static_grouped_voltage"].append(np.asarray(static_grouped[local_idx], dtype=np.float32))

    record_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    npz_triplet_id: list[int] = []
    npz_alpha: list[float] = []
    npz_geometry: list[str] = []
    npz_analysis_role: list[str] = []
    npz_selected_masks: list[np.ndarray] = []
    npz_selected_overlap_pixels: list[int] = []
    npz_total_overlap_pixels: list[int] = []
    npz_delta_v: list[np.ndarray] = []
    npz_M_curves: list[np.ndarray] = []
    npz_condition_grouped_voltage: list[np.ndarray] = []
    npz_static_grouped_voltage: list[np.ndarray] = []
    for _, payload in sorted(curve_cache.items(), key=lambda item: (item[0][0], item[0][1])):
        order = np.argsort(np.asarray(payload["delay_ms"], dtype=np.float64))
        delay_arr = np.asarray(payload["delay_ms"], dtype=np.float64)[order]
        m_arr = np.asarray(payload["M"], dtype=np.float64)[order]
        delta_arr = np.stack([payload["delta_v"][int(idx)] for idx in order], axis=0).astype(np.float32, copy=False)
        condition_voltage_arr = np.stack([payload["condition_grouped_voltage"][int(idx)] for idx in order], axis=0).astype(np.float32, copy=False)
        static_voltage_arr = np.stack([payload["static_grouped_voltage"][int(idx)] for idx in order], axis=0).astype(np.float32, copy=False)
        peak_index = int(np.argmax(m_arr))
        m_max = float(m_arr[peak_index])
        tau_peak = float(delay_arr[peak_index])
        gain_per_area = float(m_max / (float(payload["condition_meta"]["selected_overlap_pixels"]) + float(EPS)))
        if m_max <= float(EPS):
            m_norm = np.full_like(m_arr, np.nan, dtype=np.float64)
            reference_status = "all_zero"
        else:
            m_norm = m_arr / m_max
            reference_status = "ok"
        curve_rows.append(
            {
                **payload["triplet_meta"],
                **{key: value for key, value in payload["condition_meta"].items() if key != "selected_mask"},
                "M_max": m_max,
                "tau_peak": tau_peak,
                "gain_per_area": gain_per_area,
            }
        )
        for delay_ms, m_value, m_norm_value in zip(delay_arr.tolist(), m_arr.tolist(), m_norm.tolist()):
            record_rows.append(
                {
                    **payload["triplet_meta"],
                    **{key: value for key, value in payload["condition_meta"].items() if key != "selected_mask"},
                    "delay_ms": float(delay_ms),
                    "M": float(m_value),
                    "M_max": m_max,
                    "tau_peak": tau_peak,
                    "gain_per_area": gain_per_area,
                    "M_norm": float(m_norm_value) if np.isfinite(m_norm_value) else float("nan"),
                    "reference_status": reference_status,
                }
            )
        npz_triplet_id.append(int(payload["triplet_meta"]["triplet_id"]))
        npz_alpha.append(float(payload["condition_meta"]["alpha"]))
        npz_geometry.append(str(payload["condition_meta"]["geometry"]))
        npz_analysis_role.append(str(payload["condition_meta"]["analysis_role"]))
        npz_selected_masks.append(np.asarray(payload["condition_meta"]["selected_mask"], dtype=bool))
        npz_selected_overlap_pixels.append(int(payload["condition_meta"]["selected_overlap_pixels"]))
        npz_total_overlap_pixels.append(int(payload["condition_meta"]["total_overlap_pixels"]))
        npz_delta_v.append(delta_arr)
        npz_M_curves.append(m_arr.astype(np.float32, copy=False))
        npz_condition_grouped_voltage.append(condition_voltage_arr)
        npz_static_grouped_voltage.append(static_voltage_arr)

    df_records = pd.DataFrame(record_rows + invalid_rows).sort_values(["triplet_id", "alpha", "geometry", "delay_ms"], kind="stable", na_position="last").reset_index(drop=True)
    if df_records.empty:
        raise RuntimeError("Experiment produced no output records.")
    df_curve_summary = pd.DataFrame(curve_rows).sort_values(["triplet_id", "alpha", "geometry"], kind="stable").reset_index(drop=True)
    valid_main_curve_summary = df_curve_summary[df_curve_summary["analysis_role"] == "main"].copy()
    if valid_main_curve_summary.empty:
        raise RuntimeError("No valid main geometry conditions survived filtering.")

    records_csv = save_tidy_csv(df_records, output_dir / "overlap_gain_geometry_records.csv", sort_by=["triplet_id", "alpha", "geometry", "delay_ms"])

    delta_npz_path = output_dir / "delta_v_arrays.npz"
    np.savez_compressed(
        delta_npz_path,
        triplet_id=np.asarray(npz_triplet_id, dtype=np.int64),
        alpha=np.asarray(npz_alpha, dtype=np.float32),
        geometry=np.asarray(npz_geometry),
        analysis_role=np.asarray(npz_analysis_role),
        delay_ms=np.asarray(delay_ms_values, dtype=np.float32),
        delta_v=np.stack(npz_delta_v, axis=0) if npz_delta_v else np.zeros((0, len(delay_ms_values), num_classes), dtype=np.float32),
        M_curves=np.stack(npz_M_curves, axis=0) if npz_M_curves else np.zeros((0, len(delay_ms_values)), dtype=np.float32),
        selected_masks=np.stack(npz_selected_masks, axis=0) if npz_selected_masks else np.zeros((0, 0, 0), dtype=bool),
        selected_overlap_pixels=np.asarray(npz_selected_overlap_pixels, dtype=np.int64),
        total_overlap_pixels=np.asarray(npz_total_overlap_pixels, dtype=np.int64),
        condition_grouped_voltage=np.stack(npz_condition_grouped_voltage, axis=0) if npz_condition_grouped_voltage else np.zeros((0, len(delay_ms_values), num_classes), dtype=np.float32),
        static_grouped_voltage=np.stack(npz_static_grouped_voltage, axis=0) if npz_static_grouped_voltage else np.zeros((0, len(delay_ms_values), num_classes), dtype=np.float32),
    )

    mmax_fit = fit_alpha_geometry_model(valid_main_curve_summary, outcome_col="M_max", geometry_order=DEFAULT_GEOMETRIES, alpha_mode="continuous", cluster_robust_fallback=bool(args.cluster_robust_fallback))
    mmax_fit_cat = fit_alpha_geometry_model(valid_main_curve_summary, outcome_col="M_max", geometry_order=DEFAULT_GEOMETRIES, alpha_mode="categorical", cluster_robust_fallback=bool(args.cluster_robust_fallback))
    tau_fit = fit_alpha_geometry_model(valid_main_curve_summary, outcome_col="tau_peak", geometry_order=DEFAULT_GEOMETRIES, alpha_mode="continuous", cluster_robust_fallback=bool(args.cluster_robust_fallback))
    gain_fit = fit_alpha_geometry_model(valid_main_curve_summary, outcome_col="gain_per_area", geometry_order=DEFAULT_GEOMETRIES, alpha_mode="continuous", cluster_robust_fallback=bool(args.cluster_robust_fallback))
    profile_summary = summarize_normalized_profiles(df_records[df_records["analysis_role"].isin(["main", "native_reference"])], delay_ms_values=delay_ms_values)
    area_normalized_summary = summarize_area_normalized_gain(valid_main_curve_summary)
    evidence_summary = _interpret_hypotheses(
        mmax_summary=mmax_fit["summary"],
        gain_per_area_summary=gain_fit["summary"],
        profile_summary=profile_summary,
    )

    filtered_counts = {
        "n_triplets_total": int(df_triplets["triplet_id"].nunique()),
        "n_triplets_filtered_overlap_too_small": int(len(sorted(set(filtered_triplets_small_overlap)))),
        "filtered_triplet_ids_overlap_too_small": [int(triplet_id) for triplet_id in sorted(set(filtered_triplets_small_overlap))],
        "invalid_alpha_condition_counts": {key: int(len(value)) for key, value in invalid_alpha_triplets.items()},
    }

    example_triplet_id = int(df_curve_summary.sort_values(["total_overlap_pixels", "triplet_id"], ascending=[False, True], kind="stable").iloc[0]["triplet_id"])
    example_mask_bundle = mask_records[example_triplet_id]
    example_specs = triplet_condition_map[example_triplet_id]
    subset_examples = {
        "native_full": np.asarray(example_mask_bundle.sdp_mask, dtype=bool),
        "compact_0.25": np.asarray(example_specs.get(_condition_name(0.25, "compact")).selected_mask, dtype=bool) if example_specs.get(_condition_name(0.25, "compact")) and example_specs.get(_condition_name(0.25, "compact")).selected_mask is not None else np.zeros_like(example_mask_bundle.sdp_mask, dtype=bool),
        "compact_0.50": np.asarray(example_specs.get(_condition_name(0.5, "compact")).selected_mask, dtype=bool) if example_specs.get(_condition_name(0.5, "compact")) and example_specs.get(_condition_name(0.5, "compact")).selected_mask is not None else np.zeros_like(example_mask_bundle.sdp_mask, dtype=bool),
        "compact_0.75": np.asarray(example_specs.get(_condition_name(0.75, "compact")).selected_mask, dtype=bool) if example_specs.get(_condition_name(0.75, "compact")) and example_specs.get(_condition_name(0.75, "compact")).selected_mask is not None else np.zeros_like(example_mask_bundle.sdp_mask, dtype=bool),
        "fragmented_0.50": np.asarray(example_specs.get(_condition_name(0.5, "fragmented")).selected_mask, dtype=bool) if example_specs.get(_condition_name(0.5, "fragmented")) and example_specs.get(_condition_name(0.5, "fragmented")).selected_mask is not None else np.zeros_like(example_mask_bundle.sdp_mask, dtype=bool),
        "shuffled_0.50": np.asarray(example_specs.get(_condition_name(0.5, "shuffled_random")).selected_mask, dtype=bool) if example_specs.get(_condition_name(0.5, "shuffled_random")) and example_specs.get(_condition_name(0.5, "shuffled_random")).selected_mask is not None else np.zeros_like(example_mask_bundle.sdp_mask, dtype=bool),
    }

    figure_paths: dict[str, str] = {}
    if not bool(args.skip_figures):
        fig1 = plot_experiment_schematic(overlap_mask=np.asarray(example_mask_bundle.sdp_mask, dtype=bool), subset_examples=subset_examples)
        out1 = save_figure_all_formats(fig1, figures_dir / "figure_1_experiment_schematic")
        plt.close(fig1)
        figure_paths.update({f"figure_1_{key}": value for key, value in out1.items()})
        fig2 = plot_delay_sweep_curves(df_records[df_records["analysis_role"].isin(["main", "native_reference"])], alpha_values=alpha_values)
        out2 = save_figure_all_formats(fig2, figures_dir / "figure_2_delay_sweep_curves")
        plt.close(fig2)
        figure_paths.update({f"figure_2_{key}": value for key, value in out2.items()})
        fig3 = plot_peak_gain_vs_area_fraction(df_curve_summary)
        out3 = save_figure_all_formats(fig3, figures_dir / "figure_3_peak_gain_vs_area_fraction")
        plt.close(fig3)
        figure_paths.update({f"figure_3_{key}": value for key, value in out3.items()})
        fig4 = plot_area_normalized_gain(valid_main_curve_summary)
        out4 = save_figure_all_formats(fig4, figures_dir / "figure_4_area_normalized_gain")
        plt.close(fig4)
        figure_paths.update({f"figure_4_{key}": value for key, value in out4.items()})
        fig5 = plot_normalized_temporal_profile(df_records[df_records["analysis_role"].isin(["main", "native_reference"])], alpha_values=alpha_values)
        out5 = save_figure_all_formats(fig5, figures_dir / "figure_5_normalized_temporal_profile")
        plt.close(fig5)
        figure_paths.update({f"figure_5_{key}": value for key, value in out5.items()})
        fig6 = plot_effect_size_comparison({"M_max": mmax_fit["summary"]["effect_sizes"], "tau_peak": tau_fit["summary"]["effect_sizes"]})
        out6 = save_figure_all_formats(fig6, figures_dir / "figure_6_effect_size_comparison")
        plt.close(fig6)
        figure_paths.update({f"figure_6_{key}": value for key, value in out6.items()})

    summary_metrics = {
        "model_results": {
            "M_max": mmax_fit["summary"],
            "M_max_categorical_sensitivity": mmax_fit_cat["summary"],
            "tau_peak": tau_fit["summary"],
            "gain_per_area": gain_fit["summary"],
        },
        "area_normalized_results": area_normalized_summary,
        "normalized_temporal_profile_results": profile_summary,
        "hypothesis_evidence": evidence_summary,
        "filtered_counts": filtered_counts,
        "example_triplet_id": int(example_triplet_id),
        "assumptions": {
            "task": "sample -> delay1 -> distractor -> delay2 -> probe",
            "delta_v_definition": "centered_grouped_voltage(V_condition) - centered_grouped_voltage(V_static_same_subset)",
            "M_definition": "L2 norm of delta_v",
            "M_max_definition": "max over delay sweep of M(delay)",
            "tau_peak_definition": "first delay attaining M_max",
            "target_overlap": "sample ∩ distractor ∩ probe (raw SDP mask only)",
            "native_reference_only": True,
            "alpha_one_excluded_from_main_model": True,
        },
    }
    summary_json = _save_json(summary_metrics, output_dir / "overlap_gain_geometry_summary.json")

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
            "delay_sweep_ms": [float(value) for value in delay_ms_values],
            "alpha_values": [float(value) for value in alpha_values],
            "geometry_values": list(DEFAULT_GEOMETRIES),
            "batch_size": int(args.batch_size),
            "max_probes": int(args.max_probes),
            "samples_per_probe": int(args.samples_per_probe),
            "max_triplets": int(args.max_triplets),
            "num_sim_bins": int(args.num_sim_bins),
            "foreground_threshold": float(args.foreground_threshold),
            "min_overlap_pixels": int(args.min_overlap_pixels),
            "min_selected_pixels": int(args.min_selected_pixels),
            "cluster_robust_fallback": bool(args.cluster_robust_fallback),
            "skip_figures": bool(args.skip_figures),
            "readout_step": int(readout_step),
            "outputs": {
                "records_csv": str(Path(records_csv).resolve()),
                "summary_json": str(Path(summary_json).resolve()),
                "delta_v_arrays_npz": str(delta_npz_path.resolve()),
                **figure_paths,
            },
            "assumptions": summary_metrics["assumptions"],
        },
        result_root,
    )
    summary_path = save_summary_json(
        {
            "experiment": "distractor_overlap_gain_geometry_experiment",
            "triplet_count_total": int(df_triplets["triplet_id"].nunique()),
            "triplet_count_main_model": int(valid_main_curve_summary["triplet_id"].nunique()),
            "artifact_summary_metrics_json": str(summary_json.resolve()),
            "run_config_json": str(run_config_path.resolve()),
        },
        result_root,
    )
    run_log_path = save_log_lines(
        [
            "experiment=distractor_overlap_gain_geometry_experiment",
            f"model_path={args.model_path}",
            f"dataset_root={args.dataset_root}",
            f"seed={int(args.seed)}",
            f"device={device}",
            f"triplets_total={int(df_triplets['triplet_id'].nunique())}",
            f"triplets_main_model={int(valid_main_curve_summary['triplet_id'].nunique())}",
            f"result_root={result_root.resolve()}",
            f"summary_json={summary_path.resolve()}",
        ],
        logs_dir,
    )

    print("\n=== Distractor Overlap Gain Geometry Experiment Summary ===")
    print(f"Triplets total: {int(df_triplets['triplet_id'].nunique())}")
    print(f"Triplets used in main model: {int(valid_main_curve_summary['triplet_id'].nunique())}")
    print(f"Records CSV: {records_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Delta-V NPZ: {delta_npz_path}")
    print(f"Run config: {run_config_path}")
    print(f"Summary JSON (standard): {summary_path}")
    print(f"Run log: {run_log_path}")


if __name__ == "__main__":
    main()
