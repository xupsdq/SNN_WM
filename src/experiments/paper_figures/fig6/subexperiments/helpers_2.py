from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.common.input_masks import foreground_mask
from src.experiments.paper_figures.common.bundle_io import json_safe as _json_safe
from src.experiments.paper_figures.fig6.constants import MATCHED_GROUP_COLUMNS, UPDATE_GROUPS
from src.experiments.paper_figures.fig6.subexperiments.helpers_1 import (
    _image_array,
    _probe_entry_mask,
)
from src.experiments.paper_figures.fig6.types import ExperimentContext, PeakAmplifiedReentryBank


def _foreground_mask(dataset: Any, image_id: int, threshold: float) -> np.ndarray:
    return foreground_mask(_image_array(dataset, image_id), float(threshold))

def _pairwise_image_sims(dataset: Any, image_ids: Sequence[int]) -> list[float]:
    out = []
    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            out.append(_centered_cosine(_image_array(dataset, image_ids[i]).reshape(-1), _image_array(dataset, image_ids[j]).reshape(-1)))
    return out

def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0

def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)

def _as_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")

def _nan_subtract(a: Any, b: Any) -> float:
    aa = _as_float_or_nan(a)
    bb = _as_float_or_nan(b)
    return float(aa - bb) if np.isfinite(aa) and np.isfinite(bb) else float("nan")

def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float("nan") if pd.isna(numeric) else float(numeric)

def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}

def _mean_col(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")

def _mean_bool(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> float:
    if df.empty:
        return float("nan")
    arr = np.asarray(mask, dtype=bool)
    return float(np.mean(arr)) if arr.size else float("nan")

def _sem(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.std(clean, ddof=1) / np.sqrt(clean.size))

def _dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool).reshape(-1)
    bb = np.asarray(b, dtype=bool).reshape(-1)
    denom = int(aa.sum() + bb.sum())
    return _safe_div(float(2 * np.logical_and(aa, bb).sum()), float(denom))

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool).reshape(-1)
    bb = np.asarray(b, dtype=bool).reshape(-1)
    return _safe_div(float(np.logical_and(aa, bb).sum()), float(np.logical_or(aa, bb).sum()))

def _plain_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return float("nan")
    ar = pd.Series(aa[mask]).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(bb[mask]).rank(method="average").to_numpy(dtype=float)
    return _plain_cosine(ar - ar.mean(), br - br.mean())

def _high_overlap_mask(overlap: np.ndarray, n_peak: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(overlap, dtype=float).reshape(-1)
    positive = np.flatnonzero(arr >= 2)
    fallback = False
    if positive.size < int(n_peak):
        positive = np.flatnonzero(arr > 0)
        fallback = True
    chosen_count = min(max(1, int(n_peak)), int(positive.size))
    out = np.zeros(arr.size, dtype=bool)
    if chosen_count:
        chosen = positive[np.argsort(arr[positive])[-chosen_count:]]
        out[chosen] = True
    return out, fallback

def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr)) if arr.size else 0.0
    hi = float(np.nanmax(arr)) if arr.size else 1.0
    return (arr - lo) / max(hi - lo, 1e-9)

def _resize_array(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    src = np.asarray(arr)
    if src.shape == (h, w):
        return src
    rr = np.linspace(0, src.shape[0] - 1, h).round().astype(int)
    cc = np.linspace(0, src.shape[1] - 1, w).round().astype(int)
    return src[np.ix_(rr, cc)]

def _blur3(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(arr, dtype=float), 1, mode="edge")
    out = np.zeros_like(np.asarray(arr, dtype=float))
    for dr in range(3):
        for dc in range(3):
            out += padded[dr : dr + arr.shape[0], dc : dc + arr.shape[1]]
    return out / 9.0

def _top_mask(values: np.ndarray, q: float, *, positive: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    eligible = np.isfinite(arr)
    if positive is not None:
        eligible &= np.asarray(positive, dtype=bool)
    idx = np.flatnonzero(eligible.reshape(-1))
    mask = np.zeros(arr.size, dtype=bool)
    if idx.size:
        count = max(1, int(math.ceil(float(q) * idx.size)))
        chosen = idx[np.argsort(arr.reshape(-1)[idx])[-count:]]
        mask[chosen] = True
    return mask.reshape(arr.shape)

def _matched_nonpeak_mask(peak: np.ndarray, pool: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero((~peak) & pool)
    count = int(np.sum(peak))
    if candidates.size < count:
        candidates = np.flatnonzero(~peak)
    chosen = rng.choice(candidates, size=min(count, candidates.size), replace=False) if candidates.size else np.asarray([], dtype=int)
    out = np.zeros_like(peak, dtype=bool)
    out[chosen] = True
    return out

def _matched_raw_overlap_groups(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gid = 0
    for sequence_id, part in df.groupby("sequence_id", sort=True):
        part = part.sort_values("raw_overlap").copy()
        if len(part) < 2:
            continue
        for _, bucket in part.groupby(pd.qcut(part["raw_overlap"].rank(method="first"), q=min(3, len(part)), duplicates="drop"), observed=False):
            if len(bucket) < 2:
                continue
            high = bucket.sort_values("peak_weighted_overlap", ascending=False).iloc[0]
            low = bucket.sort_values("peak_weighted_overlap", ascending=True).iloc[0]
            if int(high["probe_id"]) == int(low["probe_id"]):
                continue
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "matched_group_id": f"mg_{gid:04d}",
                    "high_peak_candidate_id": int(high["probe_id"]),
                    "low_peak_candidate_id": int(low["probe_id"]),
                    "raw_overlap_difference": float(abs(high["raw_overlap"] - low["raw_overlap"])),
                    "visual_similarity_difference": float(abs(high["visual_similarity"] - low["visual_similarity"])),
                    "input_energy_difference": float(abs(high["input_energy"] - low["input_energy"])),
                    "peak_weighted_overlap_difference": float(high["peak_weighted_overlap"] - low["peak_weighted_overlap"]),
                    "class_pair_matched": bool(high["class_pair"] == low["class_pair"]),
                    "notes": f"sequence_id={int(sequence_id)}; matched within raw-overlap bucket",
                }
            )
            gid += 1
            if gid >= int(ctx.cfg.n_matched_groups):
                return pd.DataFrame(rows, columns=MATCHED_GROUP_COLUMNS)
    return pd.DataFrame(rows, columns=MATCHED_GROUP_COLUMNS)

def _matched_lookup(groups: pd.DataFrame) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for r in groups.itertuples(index=False):
        out[int(r.high_peak_candidate_id)] = (str(r.matched_group_id), "high_peak_overlap")
        out[int(r.low_peak_candidate_id)] = (str(r.matched_group_id), "low_peak_overlap")
    return out

def _sequence_index(bank: PeakAmplifiedReentryBank, sequence_id: int) -> int:
    matches = bank.sequence_meta.index[bank.sequence_meta["sequence_id"].eq(int(sequence_id))].tolist()
    if not matches:
        raise KeyError(f"Unknown sequence_id={sequence_id}")
    return int(matches[0])

def _is_proxy_mode(ctx: ExperimentContext) -> bool:
    return bool(ctx.net is None or ctx.encoder is None or torch is None)

def _df_all_proxy(df: pd.DataFrame) -> bool:
    if df.empty or "proxy_mode" not in df.columns:
        return False
    return bool(df["proxy_mode"].astype(str).str.lower().isin({"true", "1"}).all())

def _bool_col(df: pd.DataFrame, col: str, *, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([bool(default)] * len(df), index=df.index, dtype=bool)
    return df[col].astype(str).str.lower().isin({"true", "1", "yes"})

def _df_all_true(df: pd.DataFrame, col: str) -> bool:
    if df.empty or col not in df.columns:
        return False
    return bool(df[col].astype(str).str.lower().isin({"true", "1", "yes"}).all())

def _main_proxy_mode(ctx: ExperimentContext) -> bool:
    audit_path = ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
    if not audit_path.exists():
        return False
    audit = pd.read_csv(audit_path)
    return bool(not audit.empty and _bool_col(audit, "proxy_mode").any())

def _model_formula(model_name: str, target: str) -> str:
    formulas = {
        "baseline_only": f"{target} ~ 1",
        "update_only": f"{target} ~ update_count",
        "recency_only": f"{target} ~ recent_update",
        "overlap_only": f"{target} ~ input_overlap",
        "update_plus_recency": f"{target} ~ update_count + recent_update",
        "update_times_recency": f"{target} ~ update_count * recent_update",
    }
    return formulas.get(model_name, f"{target} ~ {model_name}")

def _perturbation_target(condition: str) -> str:
    name = str(condition)
    if name.startswith("intact"):
        return "intact"
    if "random" in name:
        return "random_matched_peak"
    if name == "route_peak" or "route_peak" in name:
        return "overlap_aligned_peak"
    if "nonoverlap" in name or "nonpeak" in name or "sham" in name:
        return "control_peak"
    if "peak_overlap" in name or "overlap_aligned" in name:
        return "overlap_aligned_peak"
    return "control_peak"

def _peak_perturbation_status(ctx: ExperimentContext) -> str:
    if not ctx.cfg.run_peak_perturbation:
        return "optional_not_run"
    path = ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
    if not path.exists():
        return "run_failed"
    df = pd.read_csv(path)
    if df.empty:
        return "run_empty"
    success = _bool_col(df, "route_peak_perturbation_success").any()
    return "run_successful" if success else "run_not_scientific_use"

def _peak_perturbation_claim_upgrade_allowed(ctx: ExperimentContext) -> bool:
    path = ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if df.empty or "allowed_claim_strength" not in df.columns:
        return False
    return bool(df["allowed_claim_strength"].astype(str).eq("causal_route_peak_gain").any())

def _claim_strength(ctx: ExperimentContext) -> str:
    return "causal_route_peak_gain" if _peak_perturbation_claim_upgrade_allowed(ctx) else "predictive_peak_amplified_only"

def _save_panel_d_example(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank, probe_trials: pd.DataFrame) -> None:
    if probe_trials.empty:
        return
    target = probe_trials.sort_values("peak_weighted_overlap", ascending=False).iloc[0]
    seq_idx = _sequence_index(bank, int(target["sequence_id"]))
    probe_mask = _probe_entry_mask(ctx, int(target["probe_image_id"]), mode=str(ctx.cfg.real_probe_entry_mode), cache={})
    prior = bank.prior_updated_mask[seq_idx].reshape(28, 28)
    peak = bank.peak_mask[seq_idx].reshape(28, 28)
    nonpeak = bank.nonpeak_mask[seq_idx].reshape(28, 28)
    route = probe_mask & prior
    np.savez_compressed(
        ctx.raw_dir / "panel_d_later_probe_peak_overlap_example.npz",
        probe_mask=probe_mask.astype(np.uint8),
        prior_updated_mask=prior.astype(np.uint8),
        peak_mask=peak.astype(np.uint8),
        nonpeak_mask=nonpeak.astype(np.uint8),
        raw_overlap_mask=route.astype(np.uint8),
        peak_overlap_mask=(route & peak).astype(np.uint8),
        nonpeak_overlap_mask=(route & nonpeak).astype(np.uint8),
        support_map=bank.g_final[seq_idx].reshape(28, 28).astype(np.float32),
        selected_sequence_metadata=json.dumps(bank.sequence_meta.iloc[seq_idx].to_dict(), sort_keys=True),
        selected_probe_metadata=json.dumps(_json_safe(target.to_dict()), sort_keys=True),
    )
    ctx.output_files["panel_d_later_probe_peak_overlap_example"] = "data/raw/panel_d_later_probe_peak_overlap_example.npz"

def _save_panel_c_example(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank, probe_trials: pd.DataFrame) -> None:
    if probe_trials.empty:
        return
    target = probe_trials.sort_values("peak_weighted_overlap", ascending=False).iloc[0]
    seq_idx = int(bank.sequence_meta.index[bank.sequence_meta["sequence_id"].eq(int(target["sequence_id"]))][0])
    probe_mask = _probe_entry_mask(ctx, int(target["probe_image_id"]), mode=str(ctx.cfg.real_probe_entry_mode), cache={})
    prior = bank.prior_updated_mask[seq_idx].reshape(28, 28)
    peak = bank.peak_mask[seq_idx].reshape(28, 28)
    nonpeak = bank.nonpeak_mask[seq_idx].reshape(28, 28)
    raw = probe_mask & prior
    peak_overlap = raw & peak
    nonpeak_overlap = raw & nonpeak
    np.savez_compressed(
        ctx.raw_dir / "panel_c_overlap_peak_interface_example.npz",
        probe_mask=probe_mask.astype(np.uint8),
        prior_updated_mask=prior.astype(np.uint8),
        peak_mask=peak.astype(np.uint8),
        raw_overlap_mask=raw.astype(np.uint8),
        peak_overlap_mask=peak_overlap.astype(np.uint8),
        nonpeak_overlap_mask=nonpeak_overlap.astype(np.uint8),
        support_map=bank.g_final[seq_idx].reshape(28, 28).astype(np.float32),
        selected_sequence_metadata=json.dumps(bank.sequence_meta.iloc[seq_idx].to_dict(), sort_keys=True),
        selected_probe_metadata=json.dumps(_json_safe(target.to_dict()), sort_keys=True),
    )
    ctx.output_files["panel_c_overlap_peak_interface_example"] = "data/raw/panel_c_overlap_peak_interface_example.npz"

def _first_nonzero_step(trace: np.ndarray) -> int:
    arr = np.asarray(trace, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    active = np.where(np.sum(arr > 0, axis=1) > 0)[0]
    return int(active[0]) if active.size else -1

def _class_readout_vector_from_trace(net: Any, trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    total = arr.sum(axis=0)
    n_classes = int(getattr(net.layer3, "num_classes", 10))
    neurons_per_class = int(getattr(net.layer3, "neurons_per_class", max(1, total.size // max(1, n_classes))))
    out = np.zeros(n_classes, dtype=np.float32)
    for cls in range(n_classes):
        start = cls * neurons_per_class
        end = min(start + neurons_per_class, total.size)
        if start < end:
            out[cls] = float(np.sum(total[start:end]))
    return out

def _label_evidence(vector: np.ndarray, label: int) -> float:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    idx = int(label) % max(1, arr.size)
    return float(arr[idx]) if arr.size else float("nan")

def _fire_delta(final_fire: int, s0_fire: int) -> float:
    if int(final_fire) < 0 or int(s0_fire) < 0:
        return float("nan")
    return float(int(final_fire) - int(s0_fire))

def _early_spike_count(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = max(1, min(arr.shape[0], int(math.ceil(arr.shape[0] * 0.25))))
    return float(np.sum(arr[:n] > 0))

def _spike_timing_metrics(final_trace: np.ndarray, s0_trace: np.ndarray) -> tuple[float, float, float]:
    f = np.asarray(final_trace, dtype=float)
    s = np.asarray(s0_trace, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    if s.ndim == 1:
        s = s[:, None]
    n = min(f.shape[1], s.shape[1])
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    f = f[:, :n] > 0
    s = s[:, :n] > 0
    f_any = f.any(axis=0)
    s_any = s.any(axis=0)
    f_first = np.full(n, np.nan)
    s_first = np.full(n, np.nan)
    for idx in range(n):
        if f_any[idx]:
            f_first[idx] = float(np.where(f[:, idx])[0][0])
        if s_any[idx]:
            s_first[idx] = float(np.where(s[:, idx])[0][0])
    both = np.isfinite(f_first) & np.isfinite(s_first)
    advance = both & (f_first < s_first)
    recruit = f_any & (~s_any)
    spike_advance = float(np.nanmean(s_first[both] - f_first[both])) if both.any() else float("nan")
    return float(np.mean(advance)) if advance.size else np.nan, float(np.mean(recruit)) if recruit.size else np.nan, spike_advance

def _regression_rows(ctx: ExperimentContext, df: pd.DataFrame, *, metrics: Sequence[str], n_name: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        cols = ["raw_overlap", "peak_weighted_overlap", "visual_similarity", "input_energy", metric]
        use = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(use) >= 4:
            x_full = use[["raw_overlap", "peak_weighted_overlap", "visual_similarity", "input_energy"]].to_numpy(dtype=float)
            y = use[metric].to_numpy(dtype=float)
            full = _fit_ols(x_full, y)
            x_base = use[["raw_overlap", "visual_similarity", "input_energy"]].to_numpy(dtype=float)
            base = _fit_ols(x_base, y)
            beta = full["beta"]
            p = full["p"]
            r2 = full["r2"]
            delta = r2 - base["r2"]
        else:
            beta = [np.nan] * 5
            p = [np.nan] * 5
            r2 = np.nan
            delta = np.nan
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "metric": metric,
                "beta_raw_overlap": float(beta[1]),
                "beta_peak_weighted_overlap": float(beta[2]),
                "beta_visual_similarity": float(beta[3]),
                "beta_input_energy": float(beta[4]),
                "r2": float(r2),
                "delta_r2_peak_weighted": float(delta),
                "p_peak_weighted": float(p[2]),
                n_name: int(len(use)),
            }
        )
    return pd.DataFrame(rows)

def _fit_ols(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim == 1:
        xx = xx[:, None]
    mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    xx = xx[mask]
    yy = yy[mask]
    if len(yy) < 2:
        n_coef = xx.shape[1] + 1 if xx.ndim == 2 else 2
        return {"beta": np.full(n_coef, np.nan), "se": np.full(n_coef, np.nan), "p": np.full(n_coef, np.nan), "r2": float("nan")}
    design = np.column_stack([np.ones(len(xx)), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    pred = design @ beta
    resid = yy - pred
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - float(np.sum(resid**2)) / ss_tot
    dof = max(1, len(yy) - design.shape[1])
    sigma2 = float(np.sum(resid**2) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 1e-12)
        p = np.asarray([_normal_two_sided_p(tv) for tv in t], dtype=float)
    except Exception:
        se = np.full_like(beta, np.nan)
        p = np.full_like(beta, np.nan)
    return {"beta": beta, "se": se, "p": p, "r2": float(r2)}

def _cv_r2(x: np.ndarray, y: np.ndarray, *, n_folds: int) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim == 1:
        xx = xx[:, None]
    n = len(yy)
    if n < n_folds or n_folds < 2:
        return float("nan")
    folds = np.arange(n) % int(n_folds)
    pred = np.full(n, np.nan, dtype=float)
    for fold in range(int(n_folds)):
        train = folds != fold
        test = folds == fold
        fit = _fit_ols(xx[train], yy[train])
        beta = np.asarray(fit["beta"], dtype=float)
        design = np.column_stack([np.ones(np.sum(test)), xx[test]])
        pred[test] = design @ beta
    mask = np.isfinite(pred) & np.isfinite(yy)
    if mask.sum() < 2:
        return float("nan")
    total = float(np.sum((yy[mask] - yy[mask].mean()) ** 2))
    return 0.0 if total <= 1e-12 else float(1.0 - np.sum((yy[mask] - pred[mask]) ** 2) / total)

def _normal_two_sided_p(t_value: float) -> float:
    if not np.isfinite(t_value):
        return float("nan")
    return float(math.erfc(abs(float(t_value)) / math.sqrt(2.0)))

def _standardized_coef(coef: float, x: np.ndarray, y: np.ndarray, coef_name: str, cols: list[str]) -> float:
    if coef_name == "intercept" or coef_name not in cols:
        return float("nan")
    col = cols.index(coef_name)
    sx = float(np.nanstd(x[:, col]))
    sy = float(np.nanstd(y))
    return float(coef * sx / sy) if sy > 1e-12 else float("nan")

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))

def _group_mask(df: pd.DataFrame, recent_window: int, threshold: int, group: str) -> pd.Series:
    recent = pd.to_numeric(df["time_since_last_update"], errors="coerce") < int(recent_window)
    multi = pd.to_numeric(df["update_count"], errors="coerce") >= int(threshold)
    if group == "multi_recent":
        return multi & recent
    if group == "multi_old":
        return multi & (~recent)
    if group == "single_recent":
        return (~multi) & recent
    return (~multi) & (~recent)

def _shuffle_peak_enrichment(ctx: ExperimentContext, unit_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 991)
    rows = []
    for group in UPDATE_GROUPS:
        observed = float(unit_df[unit_df["update_history_group"].eq(group)]["is_peak"].mean())
        null_vals = []
        for _ in range(int(ctx.cfg.n_null)):
            shuffled = unit_df["is_peak"].to_numpy(dtype=float).copy()
            rng.shuffle(shuffled)
            mask = unit_df["update_history_group"].eq(group).to_numpy()
            null_vals.append(float(np.mean(shuffled[mask])) if mask.any() else np.nan)
        null = np.asarray(null_vals, dtype=float)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "null_type": "sequence_label_shuffle",
                "update_history_group": group,
                "observed_P_peak": observed,
                "null_mean_P_peak": float(np.nanmean(null)),
                "null_p95_P_peak": float(np.nanpercentile(null, 95)),
                "observed_minus_null": float(observed - np.nanmean(null)),
                "empirical_p": float((np.sum(null >= observed) + 1) / (np.isfinite(null).sum() + 1)),
                "n_null": int(ctx.cfg.n_null),
            }
        )
    return pd.DataFrame(rows)

def _matched_random_controls(ctx: ExperimentContext, unit_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 804)
    rows = []
    values = unit_df["final_support"].to_numpy(dtype=float)
    for r in unit_df.sample(n=min(len(unit_df), 2000), random_state=int(ctx.cfg.network_seed)).itertuples(index=False):
        ridx = int(rng.integers(0, len(values)))
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "unit_id": int(r.unit_id),
                "observed_group": str(r.update_history_group),
                "matched_random_group": "random_unit",
                "observed_support": float(r.final_support),
                "random_support": float(values[ridx]),
                "observed_minus_random": float(r.final_support - values[ridx]),
            }
        )
    return pd.DataFrame(rows)

def _matched_peak_comparison(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "matched_group_id",
        "high_peak_probe_id",
        "low_peak_probe_id",
        "raw_overlap_difference",
        "peak_weighted_overlap_difference",
        "visual_similarity_difference",
        "input_energy_difference",
        "metric",
        "high_peak_value",
        "low_peak_value",
        "difference",
    ]
    rows = []
    for gid, part in df[df["matched_group_id"].astype(str).str.len() > 0].groupby("matched_group_id"):
        high = part[part["peak_overlap_group"].eq("high_peak_overlap")]
        low = part[part["peak_overlap_group"].eq("low_peak_overlap")]
        if high.empty or low.empty:
            continue
        h = high.iloc[0]
        l = low.iloc[0]
        metrics = [m for m in ("reentry_strength_real", "l3_trace_delta_norm", "dynamic_like_recovery_real", "decision_deflection_score_real", "reentry_strength", "DPI_L3", "dynamic_like_recovery", "decision_deflection_score") if m in part.columns]
        for metric in metrics:
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "matched_group_id": str(gid),
                    "high_peak_probe_id": int(h["probe_id"]),
                    "low_peak_probe_id": int(l["probe_id"]),
                    "raw_overlap_difference": float(abs(h["raw_overlap"] - l["raw_overlap"])),
                    "peak_weighted_overlap_difference": float(h["peak_weighted_overlap"] - l["peak_weighted_overlap"]),
                    "visual_similarity_difference": float(abs(h["visual_similarity"] - l["visual_similarity"])),
                    "input_energy_difference": float(abs(h["input_energy"] - l["input_energy"])),
                    "metric": metric,
                    "high_peak_value": float(h[metric]),
                    "low_peak_value": float(l[metric]),
                    "difference": float(h[metric] - l[metric]),
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _visual_energy_controls(ctx: ExperimentContext, reentry: pd.DataFrame, downstream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = (
        ("reentry", reentry, "reentry_strength_real" if "reentry_strength_real" in reentry.columns else "reentry_strength"),
        ("downstream", downstream, "decision_deflection_score_real" if "decision_deflection_score_real" in downstream.columns else "decision_deflection_score"),
    )
    for source, df, metric in candidates:
        if metric not in df.columns:
            continue
        for control in ("visual_similarity", "input_energy", "raw_overlap", "peak_weighted_overlap"):
            use = df[[control, metric]].apply(pd.to_numeric, errors="coerce").dropna()
            value = float(use[control].corr(use[metric])) if len(use) > 2 else float("nan")
            rows.append({"network_seed": int(ctx.cfg.network_seed), "model_or_comparison": source, "control_variable": control, "coefficient_or_difference": "pearson_r", "metric": metric, "value": value, "notes": "Pairwise diagnostic control."})
    return pd.DataFrame(rows)

def _alternative_peak_definitions(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> pd.DataFrame:
    rows = []
    delta = bank.delta_support.reshape(-1)
    definitions = {
        "top_10_percent": _top_mask(delta, 0.10, positive=delta > 0).reshape(-1),
        "top_20_percent": _top_mask(delta, 0.20, positive=delta > 0).reshape(-1),
        "zscore_threshold": delta > (float(np.nanmean(delta)) + float(np.nanstd(delta))),
        "delta_support_threshold": delta > 0,
        "support_gini_based": delta > np.nanpercentile(delta, 80),
    }
    for name, mask in definitions.items():
        rows.append({"network_seed": int(ctx.cfg.network_seed), "peak_definition": name, "metric": "n_peak_units", "value": int(np.sum(mask)), "n_units": int(mask.size)})
        rows.append({"network_seed": int(ctx.cfg.network_seed), "peak_definition": name, "metric": "mean_delta_support_peak", "value": float(np.nanmean(delta[mask])) if np.any(mask) else np.nan, "n_units": int(np.sum(mask))})
    return pd.DataFrame(rows)

def _global_support_controls(ctx: ExperimentContext, reentry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in reentry.itertuples(index=False):
        rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": int(r.sequence_id), "probe_id": int(r.probe_id), "metric": "raw_overlap", "value": float(r.raw_overlap), "notes": "Global route-control covariate."})
        rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": int(r.sequence_id), "probe_id": int(r.probe_id), "metric": "peak_weighted_overlap", "value": float(r.peak_weighted_overlap), "notes": "Peak gain-control covariate."})
    return pd.DataFrame(rows)

def _leave_one_out_timing_controls(ctx: ExperimentContext, source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if source_df.empty:
        return pd.DataFrame(rows)
    for rel, part in source_df.groupby("relative_position_from_end" if "relative_position_from_end" in source_df.columns else "removed_position", sort=True):
        rows.append({"network_seed": int(ctx.cfg.network_seed), "timing_bin": int(rel), "mean_peak_loss_fraction": _mean_col(part, "peak_loss_fraction"), "n_items": int(len(part)), "notes": "blank_same_timing leave-one-out timing control"})
    return pd.DataFrame(rows)

def _peak_source_old_vs_recent(ctx: ExperimentContext, source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if source_df.empty:
        return pd.DataFrame(rows)
    df = source_df.copy()
    df["relative_position_from_end"] = pd.to_numeric(df["seq_len"], errors="coerce") - pd.to_numeric(df["removed_position"], errors="coerce")
    df["age_group"] = np.where(df["relative_position_from_end"] < int(ctx.cfg.recent_window), "recent", "old")
    for group, part in df.groupby("age_group", sort=True):
        rows.append({"network_seed": int(ctx.cfg.network_seed), "age_group": str(group), "mean_peak_loss_fraction": _mean_col(part, "peak_loss_fraction"), "mean_peak_vs_nonpeak_loss_ratio": _mean_col(part, "peak_vs_nonpeak_loss_ratio"), "n_items": int(len(part))})
    return pd.DataFrame(rows)

def _recent_overlap_window_robustness(ctx: ExperimentContext, overlap_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if overlap_df.empty:
        return pd.DataFrame(rows)
    for window, part in overlap_df[overlap_df["overlap_type"].astype(str).eq("recent")].groupby("overlap_window", sort=True):
        rows.append({"network_seed": int(ctx.cfg.network_seed), "overlap_window": str(window), "mean_dice": _mean_col(part, "dice_peak_overlap"), "mean_peak_coverage": _mean_col(part, "peak_coverage"), "n_sequences": int(part["sequence_id"].nunique())})
    return pd.DataFrame(rows)

def _random_window_overlap_controls(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 404)
    rows = []
    for seq_idx, meta in enumerate(bank.sequence_meta.itertuples(index=False)):
        seq_len = int(meta.seq_len)
        item_maps = bank.item_activation_history[seq_idx, :seq_len, :] > 0
        peak = bank.peak_mask[seq_idx].reshape(-1)
        for k in tuple(int(v) for v in ctx.cfg.recent_overlap_windows):
            if seq_len <= 0:
                continue
            start = int(rng.integers(0, max(1, seq_len - min(k, seq_len) + 1)))
            end = min(seq_len, start + k)
            overlap = item_maps[start:end, :].sum(axis=0)
            high, fallback = _high_overlap_mask(overlap, int(np.sum(peak)))
            rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": int(meta.sequence_id), "overlap_window": f"random_{k}", "window_start_position": int(start + 1), "window_end_position": int(end), "dice_peak_overlap": _dice(peak, high), "peak_coverage": _safe_div(float(np.sum(peak & high)), float(np.sum(peak))), "fallback_used": bool(fallback)})
    return pd.DataFrame(rows)

def _real_reentry_control_s0_static(ctx: ExperimentContext, reentry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in reentry.itertuples(index=False):
        rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": int(r.sequence_id), "probe_id": int(r.probe_id), "reference_condition": "S0", "prediction_S0": int(getattr(r, "prediction_S0", -1)), "first_fire_time_S0": int(getattr(r, "first_fire_time_S0", -1)), "proxy_mode": bool(getattr(r, "proxy_mode", False)), "notes": "S0 baseline/reset reference for real rollout comparison"})
    return pd.DataFrame(rows)

def _real_downstream_metric_definitions(ctx: ExperimentContext) -> pd.DataFrame:
    metrics = {
        "early_recruitment_gain_real": "early S_final spike/readout activity minus S0",
        "P_advance_real": "fraction of channels firing earlier in S_final than S0",
        "P_recruit_real": "fraction of channels firing in S_final and not S0",
        "spike_advance_real": "mean first-fire advance among channels active in both conditions",
        "response_pattern_displacement_real": "norm between S_final and S0 response vectors",
        "decision_deflection_score_real": "probe-label evidence in S_final minus S0",
        "partial_cue_completion_gain_real": "reserved for partial-cue branch; NaN when unavailable",
    }
    return pd.DataFrame([{"network_seed": int(ctx.cfg.network_seed), "metric": key, "definition": value, "proxy_mode_not_final": True} for key, value in metrics.items()])

def _trial_condition_audit(ctx: ExperimentContext) -> pd.DataFrame:
    modules = ["sequence_bank", "peak_source_attribution", "peak_update_history", "peak_input_overlap_origin", "later_probe_peak_overlap_trials", "real_reentry_metrics", "real_downstream_metrics", "supplement", "peak_perturbation"]
    rows = []
    for module in modules:
        done = bool(ctx.completed_modules.get(module))
        rows.append({"network_seed": int(ctx.cfg.network_seed), "n_sequences": int(ctx.n_sequences), "n_probe_candidates": int(ctx.n_probe_candidates), "n_matched_groups": int(ctx.n_matched_groups), "n_conditions": int(len(UPDATE_GROUPS)), "module": module, "n_completed": int(done), "n_failed": int(not done), "notes": "single-network smoke-ready audit"})
    return pd.DataFrame(rows)

def _perturbation_unit_sets(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> pd.DataFrame:
    rows = []
    for seq_idx, meta in enumerate(bank.sequence_meta.itertuples(index=False)):
        units = np.flatnonzero(bank.peak_mask[seq_idx])
        for unit_id in units[:50]:
            rows.append({"network_seed": int(ctx.cfg.network_seed), "sequence_id": int(meta.sequence_id), "probe_id": -1, "condition": "candidate_peak_unit", "unit_id": int(unit_id), "notes": "candidate set for overlap-aligned peak perturbation"})
    return pd.DataFrame(rows)

__all__ = ('_foreground_mask', '_pairwise_image_sims', '_centered_cosine', '_safe_div', '_as_float_or_nan', '_nan_subtract', '_num', '_bool_value', '_mean_col', '_mean_bool', '_sem', '_dice', '_jaccard', '_plain_cosine', '_spearman', '_high_overlap_mask', '_normalize', '_resize_array', '_blur3', '_top_mask', '_matched_nonpeak_mask', '_matched_raw_overlap_groups', '_matched_lookup', '_sequence_index', '_is_proxy_mode', '_df_all_proxy', '_bool_col', '_df_all_true', '_main_proxy_mode', '_model_formula', '_perturbation_target', '_peak_perturbation_status', '_peak_perturbation_claim_upgrade_allowed', '_claim_strength', '_save_panel_d_example', '_save_panel_c_example', '_first_nonzero_step', '_class_readout_vector_from_trace', '_label_evidence', '_fire_delta', '_early_spike_count', '_spike_timing_metrics', '_regression_rows', '_fit_ols', '_cv_r2', '_normal_two_sided_p', '_standardized_coef', '_sigmoid', '_group_mask', '_shuffle_peak_enrichment', '_matched_random_controls', '_matched_peak_comparison', '_visual_energy_controls', '_alternative_peak_definitions', '_global_support_controls', '_leave_one_out_timing_controls', '_peak_source_old_vs_recent', '_recent_overlap_window_robustness', '_random_window_overlap_controls', '_real_reentry_control_s0_static', '_real_downstream_metric_definitions', '_trial_condition_audit', '_perturbation_unit_sets')
