from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.experiments.deterministic_discovery import ModelRuntime, ProbeDataSource, ScanConfig, TracePolicy, run_deterministic_discovery
from src.experiments.common.diagnostic_mask_utils import project_patch_values_to_image
from src.experiments.common.voltage_readout import (
    compute_voltage_margin,
    compute_voltage_margin_fixed_competitor,
    get_group_voltage_scores,
)


@dataclass(frozen=True)
class ProbeDiagnosticRecord:
    probe_id: int
    probe_label: int
    image: torch.Tensor
    foreground_mask: np.ndarray
    diagnostic_mask: np.ndarray
    nondiagnostic_mask: np.ndarray
    importance_map: np.ndarray
    selected_area: int
    metadata: Dict[str, object]
    baseline_is_correct: int
    importance_map_signed: np.ndarray
    positive_mask: np.ndarray
    negative_mask: np.ndarray
    baseline_wrong0_label: int = -1
    baseline_wrong0_score: float = float("nan")
    baseline_margin_fixed_wrong0: float = float("nan")
    wrong_mask_semantics: str = ""
    direction_score_semantics: str = "raw_margin_drop"
    record_source: str = ""


class ProbeDiagnosticProvider(Protocol):
    def collect(self) -> Tuple[List[ProbeDiagnosticRecord], pd.DataFrame]:
        ...


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _foreground_mask_from_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().to(torch.float32).numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected probe image shape [C, H, W], got {tuple(array.shape)}")
    return np.asarray(array[0] > 0.0, dtype=bool)


def _reconstruct_importance_from_pixel_rank(pixel_rank_df: pd.DataFrame, shape: Tuple[int, int]) -> np.ndarray:
    importance = np.zeros(shape, dtype=np.float64)
    if pixel_rank_df.empty:
        return importance
    for row in pixel_rank_df.itertuples(index=False):
        importance[int(row.row), int(row.col)] = float(row.importance)
    return importance


def _reconstruct_importance_from_windows(window_df: pd.DataFrame, shape: Tuple[int, int]) -> np.ndarray:
    if window_df.empty:
        return np.zeros(shape, dtype=np.float64)
    value_col = "raw_importance"
    if "raw_importance_fixed" in window_df.columns:
        fixed_values = window_df["raw_importance_fixed"].to_numpy(dtype=np.float64)
        if np.isfinite(fixed_values).any():
            value_col = "raw_importance_fixed"
    elif "discriminative_importance_fixed" in window_df.columns:
        fixed_values = window_df["discriminative_importance_fixed"].to_numpy(dtype=np.float64)
        if np.isfinite(fixed_values).any():
            value_col = "discriminative_importance_fixed"
    elif "discriminative_importance" in window_df.columns:
        value_col = "discriminative_importance"
    return project_patch_values_to_image(
        int(shape[0]),
        int(shape[1]),
        patches=[
            type(
                "PatchLike",
                (),
                {
                    "row_start": int(row.row_start),
                    "row_end": int(row.row_end),
                    "col_start": int(row.col_start),
                    "col_end": int(row.col_end),
                },
            )()
            for row in window_df.itertuples(index=False)
        ],
        values=window_df[value_col].to_numpy(dtype=np.float64),
    )


def _load_baseline_lookup(results_dir: Path) -> dict[int, dict[str, object]]:
    lookup: dict[int, dict[str, object]] = {}
    baseline_df = _read_csv_if_exists(results_dir / "baseline_probe_reference.csv")
    wrong_fixed_df = _read_csv_if_exists(results_dir / "wrong_fixed_competitor_audit.csv")
    for df in (baseline_df, wrong_fixed_df):
        if df.empty:
            continue
        for row in df.to_dict("records"):
            probe_id = int(row["probe_id"])
            payload = lookup.setdefault(probe_id, {})
            payload.update({str(key): value for key, value in row.items()})
    return lookup


def _infer_baseline_from_scores(path: Path, probe_label: int) -> dict[str, object]:
    if not path.exists():
        return {}
    score_df = pd.read_csv(path)
    if score_df.empty or "score" not in score_df.columns:
        return {}
    scores = score_df["score"].to_numpy(dtype=np.float64)
    if scores.size <= int(probe_label):
        return {}
    margin = compute_voltage_margin(scores, true_label=int(probe_label))
    predicted_label = int(np.argmax(scores))
    payload = {
        "baseline_predicted_label": predicted_label,
        "baseline_margin": float(margin.margin),
        "baseline_true_score": float(margin.true_score),
        "baseline_best_wrong_score": float(margin.best_wrong_score),
        "baseline_best_wrong_label": int(margin.best_wrong_label),
        "baseline_is_correct": int(predicted_label == int(probe_label)),
    }
    return payload


def _maybe_int(value: object, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        if isinstance(value, float) and np.isnan(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _maybe_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(out):
        return float(default)
    return out


def _coerce_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.shape != shape:
        raise ValueError(f"Mask shape mismatch: expected {shape}, got {out.shape}")
    return out


def _build_record_from_payload(
    *,
    probe_id: int,
    probe_label: int,
    image: torch.Tensor,
    foreground_mask: np.ndarray | None,
    importance_map_signed: np.ndarray,
    diagnostic_mask: np.ndarray | None,
    positive_mask: np.ndarray | None,
    negative_mask: np.ndarray | None,
    baseline_is_correct: int,
    metadata: Mapping[str, object],
) -> ProbeDiagnosticRecord:
    raw_foreground = _foreground_mask_from_image(image) if foreground_mask is None else np.asarray(foreground_mask, dtype=bool)
    shape = tuple(int(v) for v in raw_foreground.shape)
    importance_signed = np.asarray(importance_map_signed, dtype=np.float64)
    if importance_signed.shape != shape:
        raise ValueError(
            f"Importance map shape mismatch for probe {probe_id}: {importance_signed.shape} vs {raw_foreground.shape}"
        )
    diagnostic = _coerce_mask(diagnostic_mask, shape) & raw_foreground
    positive = _coerce_mask(positive_mask, shape) & raw_foreground
    negative = _coerce_mask(negative_mask, shape) & raw_foreground
    nondiagnostic = raw_foreground & ~diagnostic
    payload = dict(metadata)
    if int(baseline_is_correct) == 1:
        payload["probe_partition"] = "correct"
    elif int(baseline_is_correct) == 0:
        payload["probe_partition"] = "wrong"
        payload.setdefault("wrong_mask_semantics", "support_harm_fixed_competitor")
        payload.setdefault("positive_mask_source_semantics", "support")
        payload.setdefault("negative_mask_source_semantics", "harm")
    else:
        payload["probe_partition"] = "unknown"
    payload["diagnostic_area"] = int(diagnostic.sum())
    payload["nondiagnostic_area"] = int(nondiagnostic.sum())
    payload["positive_area"] = int(positive.sum())
    payload["negative_area"] = int(negative.sum())
    baseline_wrong0_label = _maybe_int(payload.get("baseline_wrong0_label", payload.get("wrong0_label", -1)), default=-1)
    baseline_wrong0_score = _maybe_float(payload.get("baseline_wrong0_score", payload.get("wrong0_score", float("nan"))))
    baseline_margin_fixed_wrong0 = _maybe_float(
        payload.get("baseline_margin_fixed_wrong0", payload.get("margin_fixed_wrong0", float("nan")))
    )
    wrong_mask_semantics = str(payload.get("wrong_mask_semantics", ""))
    direction_score_semantics = str(payload.get("direction_score_semantics", "raw_margin_drop"))
    record_source = str(payload.get("record_source", payload.get("load_source", "")))
    return ProbeDiagnosticRecord(
        probe_id=int(probe_id),
        probe_label=int(probe_label),
        image=image.detach().cpu().to(torch.float32),
        foreground_mask=np.asarray(raw_foreground, dtype=bool),
        diagnostic_mask=np.asarray(diagnostic, dtype=bool),
        nondiagnostic_mask=np.asarray(nondiagnostic, dtype=bool),
        importance_map=np.asarray(importance_signed, dtype=np.float64),
        selected_area=int(diagnostic.sum()),
        metadata=payload,
        baseline_is_correct=int(baseline_is_correct),
        importance_map_signed=np.asarray(importance_signed, dtype=np.float64),
        positive_mask=np.asarray(positive, dtype=bool),
        negative_mask=np.asarray(negative, dtype=bool),
        baseline_wrong0_label=-1 if int(baseline_is_correct) == 1 else int(baseline_wrong0_label),
        baseline_wrong0_score=float("nan") if int(baseline_is_correct) == 1 else float(baseline_wrong0_score),
        baseline_margin_fixed_wrong0=(
            float("nan") if int(baseline_is_correct) == 1 else float(baseline_margin_fixed_wrong0)
        ),
        wrong_mask_semantics="" if int(baseline_is_correct) == 1 else wrong_mask_semantics,
        direction_score_semantics=direction_score_semantics,
        record_source=record_source,
    )


def _resolve_baseline_status(
    *,
    probe_id: int,
    probe_label: int,
    probe_rule_row: Mapping[str, object] | None,
    baseline_lookup: Mapping[int, Mapping[str, object]],
    per_probe_metadata: Mapping[str, object],
    group_scores_path: Path,
) -> tuple[int | None, Dict[str, object]]:
    metadata: Dict[str, object] = {}
    for source in (probe_rule_row or {}, per_probe_metadata or {}, baseline_lookup.get(int(probe_id), {})):
        for key in (
            "baseline_is_correct",
            "baseline_predicted_label",
            "baseline_margin",
            "baseline_true_score",
            "baseline_best_wrong_score",
            "baseline_best_wrong_label",
            "baseline_wrong0_label",
            "baseline_wrong0_score",
            "baseline_margin_fixed_wrong0",
            "wrong0_label",
            "wrong0_score",
            "margin_fixed_wrong0",
            "wrong_mask_semantics",
            "positive_mask_source_semantics",
            "negative_mask_source_semantics",
            "first_fire_t_probe",
            "baseline_first_fire_t_probe",
        ):
            if key in source and key not in metadata:
                metadata[key] = source[key]
    inferred = _infer_baseline_from_scores(group_scores_path, probe_label=int(probe_label))
    for key, value in inferred.items():
        metadata.setdefault(key, value)
    status = metadata.get("baseline_is_correct", None)
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return None, metadata
    return int(status), metadata


def _load_per_probe_results(
    *,
    probe_dir: Path,
    probe_id: int,
    probe_label: int,
    image: torch.Tensor,
    probe_rule_row: Mapping[str, object] | None,
    root_window_df: pd.DataFrame,
    root_pixel_df: pd.DataFrame,
    run_config: Mapping[str, object],
    baseline_lookup: Mapping[int, Mapping[str, object]],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, int | None, Dict[str, object]]:
    shape = tuple(int(v) for v in image.shape[-2:])
    metadata: Dict[str, object] = {"load_source": "results", "record_source": "results", "probe_dir": str(probe_dir)}
    if probe_rule_row is not None:
        metadata.update({str(key): value for key, value in dict(probe_rule_row).items()})
    if run_config:
        for key in ("model_path", "mask_policy", "lambda_global_selected", "patch_size", "scan_stride", "probe_partition"):
            if key in run_config:
                metadata[f"run_config_{key}"] = run_config[key]

    per_probe_metadata = _read_json_if_exists(probe_dir / "probe_metadata.json")
    if per_probe_metadata:
        metadata.update({f"probe_meta_{key}": value for key, value in per_probe_metadata.items()})

    importance_map: np.ndarray | None = None
    foreground_mask: np.ndarray | None = None
    diagnostic_mask: np.ndarray | None = None
    positive_mask: np.ndarray | None = None
    negative_mask: np.ndarray | None = None

    foreground_path = probe_dir / "foreground_mask.npy"
    if foreground_path.exists():
        foreground_mask = np.asarray(np.load(foreground_path), dtype=bool)
        metadata["foreground_source"] = str(foreground_path)

    for candidate in (probe_dir / "importance_signed.npy", probe_dir / "importance_discriminative.npy"):
        if candidate.exists():
            importance_map = np.asarray(np.load(candidate), dtype=np.float64)
            metadata["importance_source"] = str(candidate)
            break

    mask_path = probe_dir / "critical_mask_nonzero.npy"
    if mask_path.exists():
        diagnostic_mask = np.asarray(np.load(mask_path), dtype=bool)
        metadata["mask_source"] = str(mask_path)

    positive_path = probe_dir / "positive_importance_mask.npy"
    negative_path = probe_dir / "negative_importance_mask.npy"
    if positive_path.exists():
        positive_mask = np.asarray(np.load(positive_path), dtype=bool)
        metadata["positive_mask_source"] = str(positive_path)
    if negative_path.exists():
        negative_mask = np.asarray(np.load(negative_path), dtype=bool)
        metadata["negative_mask_source"] = str(negative_path)

    if importance_map is None:
        pixel_path = probe_dir / "pixel_importance_rank.csv"
        if pixel_path.exists():
            importance_map = _reconstruct_importance_from_pixel_rank(pd.read_csv(pixel_path), shape)
            metadata["importance_source"] = str(pixel_path)

    if importance_map is None:
        window_path = probe_dir / "window_summary.csv"
        if window_path.exists():
            importance_map = _reconstruct_importance_from_windows(pd.read_csv(window_path), shape)
            metadata["importance_source"] = str(window_path)

    if importance_map is None and not root_pixel_df.empty:
        subset = root_pixel_df[root_pixel_df["probe_id"] == int(probe_id)].copy()
        if not subset.empty:
            importance_map = _reconstruct_importance_from_pixel_rank(subset, shape)
            metadata["importance_source"] = "root_pixel_importance_rank.csv"

    if importance_map is None and not root_window_df.empty:
        subset = root_window_df[root_window_df["probe_id"] == int(probe_id)].copy()
        if not subset.empty:
            importance_map = _reconstruct_importance_from_windows(subset, shape)
            metadata["importance_source"] = "root_window_summary.csv"

    baseline_is_correct, baseline_metadata = _resolve_baseline_status(
        probe_id=int(probe_id),
        probe_label=int(probe_label),
        probe_rule_row=probe_rule_row,
        baseline_lookup=baseline_lookup,
        per_probe_metadata=per_probe_metadata,
        group_scores_path=probe_dir / "group_scores_full.csv",
    )
    metadata.update(baseline_metadata)
    if int(_maybe_int(metadata.get("baseline_is_correct", baseline_is_correct if baseline_is_correct is not None else -1), default=-1)) == 0:
        metadata.setdefault("wrong_mask_semantics", "support_harm_fixed_competitor")
        metadata.setdefault("positive_mask_source_semantics", "support")
        metadata.setdefault("negative_mask_source_semantics", "harm")
    metadata.setdefault("direction_score_semantics", "raw_margin_drop")
    return foreground_mask, importance_map, diagnostic_mask, positive_mask, negative_mask, baseline_is_correct, metadata


def _validate_record(record: ProbeDiagnosticRecord) -> tuple[bool, str]:
    if record.image is None:
        return False, "missing_image"
    if int(record.foreground_mask.sum()) <= 0:
        return False, "empty_foreground_mask"
    if record.importance_map_signed is None:
        return False, "missing_importance_map_signed"
    if int(np.asarray(record.importance_map_signed).size) <= 0:
        return False, "empty_importance_map_signed"
    return True, ""


class ResultsProbeDiagnosticProvider:
    def __init__(
        self,
        *,
        results_dir: str | Path,
        dataset,
        probe_ids: Sequence[int] | None = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.dataset = dataset
        self.probe_ids = None if probe_ids is None else [int(probe_id) for probe_id in probe_ids]

    def _discover_probe_ids(self) -> List[int]:
        if self.probe_ids is not None:
            return list(dict.fromkeys(self.probe_ids))
        per_probe_ids = sorted(
            int(path.name.split("_")[-1])
            for path in (self.results_dir / "probes").glob("probe_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        )
        if per_probe_ids:
            return per_probe_ids
        probe_rule_path = self.results_dir / "probe_rule_table.csv"
        if probe_rule_path.exists():
            df = pd.read_csv(probe_rule_path)
            if "probe_id" in df.columns:
                return df["probe_id"].astype(np.int64).drop_duplicates().tolist()
        raise FileNotFoundError(f"No probe diagnostics found under {self.results_dir}")

    def collect(self) -> Tuple[List[ProbeDiagnosticRecord], pd.DataFrame]:
        probe_rule_df = _read_csv_if_exists(self.results_dir / "probe_rule_table.csv")
        root_window_df = _read_csv_if_exists(self.results_dir / "window_summary.csv")
        root_pixel_df = _read_csv_if_exists(self.results_dir / "pixel_importance_rank.csv")
        run_config = _read_json_if_exists(self.results_dir / "run_config.json")
        baseline_lookup = _load_baseline_lookup(self.results_dir)
        records: List[ProbeDiagnosticRecord] = []
        inventory_rows: List[Dict[str, object]] = []

        for probe_id in self._discover_probe_ids():
            image, probe_label = self.dataset[int(probe_id)]
            probe_label = int(probe_label)
            probe_dir = self.results_dir / "probes" / f"probe_{int(probe_id):05d}"
            probe_rule_row = None
            if not probe_rule_df.empty:
                subset = probe_rule_df[probe_rule_df["probe_id"] == int(probe_id)]
                if not subset.empty:
                    probe_rule_row = subset.iloc[0].to_dict()
            try:
                foreground_mask, importance_map, diagnostic_mask, positive_mask, negative_mask, baseline_is_correct, metadata = _load_per_probe_results(
                    probe_dir=probe_dir,
                    probe_id=int(probe_id),
                    probe_label=probe_label,
                    image=image,
                    probe_rule_row=probe_rule_row,
                    root_window_df=root_window_df,
                    root_pixel_df=root_pixel_df,
                    run_config=run_config,
                    baseline_lookup=baseline_lookup,
                )
                if importance_map is None:
                    inventory_rows.append(
                        {
                            "probe_id": int(probe_id),
                            "probe_label": probe_label,
                            "status": "skipped",
                            "skip_reason": "missing_signed_importance_map",
                        }
                    )
                    continue
                record = _build_record_from_payload(
                    probe_id=int(probe_id),
                    probe_label=probe_label,
                    image=image,
                    foreground_mask=foreground_mask,
                    importance_map_signed=importance_map,
                    diagnostic_mask=diagnostic_mask,
                    positive_mask=positive_mask,
                    negative_mask=negative_mask,
                    baseline_is_correct=-1 if baseline_is_correct is None else int(baseline_is_correct),
                    metadata=metadata,
                )
                is_valid, skip_reason = _validate_record(record)
                if not is_valid:
                    inventory_rows.append(
                        {
                            "probe_id": int(probe_id),
                            "probe_label": probe_label,
                            "baseline_is_correct": int(record.baseline_is_correct),
                            "status": "skipped",
                            "skip_reason": str(skip_reason),
                            "diagnostic_area": int(record.diagnostic_mask.sum()),
                            "nondiagnostic_area": int(record.nondiagnostic_mask.sum()),
                            "positive_area": int(record.positive_mask.sum()),
                            "negative_area": int(record.negative_mask.sum()),
                        }
                    )
                    continue
                records.append(record)
                inventory_rows.append(
                        {
                            "probe_id": int(probe_id),
                            "probe_label": probe_label,
                            "baseline_is_correct": int(record.baseline_is_correct),
                            "probe_partition": str(record.metadata.get("probe_partition", "")),
                            "status": "loaded",
                            "skip_reason": "",
                            "foreground_area": int(record.foreground_mask.sum()),
                            "has_importance_map": int(record.importance_map_signed is not None),
                            "selected_area": int(record.selected_area),
                            "diagnostic_area": int(record.diagnostic_mask.sum()),
                            "nondiagnostic_area": int(record.nondiagnostic_mask.sum()),
                            "positive_area": int(record.positive_mask.sum()),
                            "negative_area": int(record.negative_mask.sum()),
                            "baseline_wrong0_label": int(record.baseline_wrong0_label),
                            "baseline_margin_fixed_wrong0": float(record.baseline_margin_fixed_wrong0),
                            "importance_min": float(np.nanmin(record.importance_map_signed)),
                            "importance_max": float(np.nanmax(record.importance_map_signed)),
                            "importance_mean": float(np.nanmean(record.importance_map_signed)),
                            "importance_std": float(np.nanstd(record.importance_map_signed)),
                            "direction_score_semantics": str(record.direction_score_semantics),
                            "record_source": str(record.record_source),
                            "wrong_mask_semantics": str(record.wrong_mask_semantics),
                            "importance_source": str(record.metadata.get("importance_source", "")),
                            "foreground_source": str(record.metadata.get("foreground_source", "")),
                            "mask_source": str(record.metadata.get("mask_source", "")),
                            "positive_mask_source": str(record.metadata.get("positive_mask_source", "")),
                            "negative_mask_source": str(record.metadata.get("negative_mask_source", "")),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                inventory_rows.append(
                    {
                        "probe_id": int(probe_id),
                        "probe_label": probe_label,
                        "status": "skipped",
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return records, pd.DataFrame(inventory_rows)


class ComputeProbeDiagnosticProvider:
    def __init__(
        self,
        *,
        dataset,
        model_runtime: ModelRuntime,
        scan_config: ScanConfig,
        cache_dir: str | Path | None = None,
        probe_ids: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.model_runtime = model_runtime
        self.scan_config = scan_config
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.probe_ids = [int(probe_id) for probe_id in probe_ids]

    def _write_cache(
        self,
        *,
        probe_id: int,
        probe_label: int,
        class_scores: np.ndarray,
        foreground_mask: np.ndarray,
        importance_map_signed: np.ndarray,
        diagnostic_mask: np.ndarray,
        positive_mask: np.ndarray,
        negative_mask: np.ndarray,
        metadata: Mapping[str, object],
        window_df: pd.DataFrame,
        pixel_df: pd.DataFrame,
    ) -> None:
        if self.cache_dir is None:
            return
        probe_dir = self.cache_dir / "probes" / f"probe_{int(probe_id):05d}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        np.save(probe_dir / "importance_discriminative.npy", np.asarray(importance_map_signed, dtype=np.float64))
        np.save(probe_dir / "importance_signed.npy", np.asarray(importance_map_signed, dtype=np.float64))
        np.save(probe_dir / "foreground_mask.npy", np.asarray(foreground_mask, dtype=np.uint8))
        np.save(probe_dir / "critical_mask_nonzero.npy", np.asarray(diagnostic_mask, dtype=np.uint8))
        np.save(probe_dir / "positive_importance_mask.npy", np.asarray(positive_mask, dtype=np.uint8))
        np.save(probe_dir / "negative_importance_mask.npy", np.asarray(negative_mask, dtype=np.uint8))
        pd.DataFrame(
            [{"class_label": int(label), "score": float(score)} for label, score in enumerate(class_scores.tolist())]
        ).to_csv(probe_dir / "group_scores_full.csv", index=False)
        (probe_dir / "probe_metadata.json").write_text(
            json.dumps({str(key): value for key, value in dict(metadata).items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not window_df.empty:
            window_df.to_csv(probe_dir / "window_summary.csv", index=False)
        if not pixel_df.empty:
            pixel_df.to_csv(probe_dir / "pixel_importance_rank.csv", index=False)

        summary_path = self.cache_dir / "probe_rule_table.csv"
        row_df = pd.DataFrame(
            [
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(probe_label),
                    "baseline_is_correct": int(metadata.get("baseline_is_correct", -1)),
                    "baseline_wrong0_label": int(_maybe_int(metadata.get("baseline_wrong0_label", -1), default=-1)),
                    "baseline_margin_fixed_wrong0": float(
                        _maybe_float(metadata.get("baseline_margin_fixed_wrong0", float("nan")))
                    ),
                    "foreground_area": int(np.asarray(foreground_mask, dtype=bool).sum()),
                    "mask_area": int(np.asarray(diagnostic_mask, dtype=bool).sum()),
                    "positive_area": int(np.asarray(positive_mask, dtype=bool).sum()),
                    "negative_area": int(np.asarray(negative_mask, dtype=bool).sum()),
                }
            ]
        )
        existing = _read_csv_if_exists(summary_path)
        combined = pd.concat([existing, row_df], axis=0, ignore_index=True) if not existing.empty else row_df
        combined = combined.drop_duplicates(subset=["probe_id"], keep="last").sort_values(["probe_id"], kind="stable")
        combined.to_csv(summary_path, index=False)

    def collect(self) -> Tuple[List[ProbeDiagnosticRecord], pd.DataFrame]:
        records: List[ProbeDiagnosticRecord] = []
        inventory_rows: List[Dict[str, object]] = []
        for probe_id in self.probe_ids:
            image, probe_label = self.dataset[int(probe_id)]
            probe_label = int(probe_label)
            try:
                bundle = get_group_voltage_scores(
                    net=self.model_runtime.net,
                    encoder=self.model_runtime.encoder,
                    probe_images=image.unsqueeze(0),
                    spec=self.model_runtime.spec,
                    device=self.model_runtime.device,
                    readout_mode=self.model_runtime.readout_mode,
                    readout_step=self.model_runtime.readout_step,
                    pooling=self.model_runtime.voltage_pooling,
                    m=self.model_runtime.top_m,
                    stsp_mode="static_frozen",
                )[0]
                margin = compute_voltage_margin(bundle, true_label=probe_label)
                baseline_is_correct = int(bundle.predicted_label == int(probe_label))
                wrong0_label = int(bundle.predicted_label) if baseline_is_correct == 0 else -1
                fixed_margin = (
                    compute_voltage_margin_fixed_competitor(
                        bundle,
                        true_label=int(probe_label),
                        competitor_label=int(wrong0_label),
                    )
                    if baseline_is_correct == 0
                    else None
                )
                result = run_deterministic_discovery(
                    scan_config=self.scan_config,
                    data_source=ProbeDataSource(
                        probe_id=int(probe_id),
                        probe_label=probe_label,
                        probe_image=image,
                    ),
                    model_runtime=self.model_runtime,
                    trace_policy=TracePolicy(mode="none"),
                    fixed_competitor_label=None if baseline_is_correct == 1 else int(wrong0_label),
                )
                foreground_mask = _foreground_mask_from_image(image)
                empty_mask = np.zeros_like(foreground_mask, dtype=bool)
                metadata = {
                    "load_source": "compute",
                    "record_source": "compute",
                    "probe_partition": "correct" if baseline_is_correct == 1 else "wrong",
                    "baseline_is_correct": int(baseline_is_correct),
                    "baseline_predicted_label": int(bundle.predicted_label),
                    "baseline_true_score": float(margin.true_score),
                    "baseline_best_wrong_score": float(margin.best_wrong_score),
                    "baseline_best_wrong_label": int(margin.best_wrong_label),
                    "baseline_margin": float(margin.margin),
                    "baseline_wrong0_label": int(wrong0_label),
                    "baseline_wrong0_score": float("nan") if fixed_margin is None else float(fixed_margin.competitor_score),
                    "baseline_margin_fixed_wrong0": float("nan") if fixed_margin is None else float(fixed_margin.margin),
                    "wrong_mask_semantics": "" if baseline_is_correct == 1 else "support_harm_fixed_competitor",
                    "positive_mask_source_semantics": "" if baseline_is_correct == 1 else "support",
                    "negative_mask_source_semantics": "" if baseline_is_correct == 1 else "harm",
                    "direction_score_semantics": "raw_margin_drop",
                    "dn_selection_stage": "causal_only",
                    "active_value_col": str(result.active_value_col),
                    "uses_fixed_competitor": bool(result.uses_fixed_competitor),
                    "lambda_global": float(self.scan_config.lambda_global),
                    "patch_size": int(self.scan_config.patch_size),
                    "scan_stride": int(self.scan_config.scan_stride),
                }
                record = _build_record_from_payload(
                    probe_id=int(probe_id),
                    probe_label=probe_label,
                    image=image,
                    foreground_mask=foreground_mask,
                    importance_map_signed=result.importance_map_signed,
                    diagnostic_mask=empty_mask,
                    positive_mask=empty_mask,
                    negative_mask=empty_mask,
                    baseline_is_correct=baseline_is_correct,
                    metadata=metadata,
                )
                is_valid, skip_reason = _validate_record(record)
                if not is_valid:
                    inventory_rows.append(
                        {
                            "probe_id": int(probe_id),
                            "probe_label": probe_label,
                            "baseline_is_correct": int(baseline_is_correct),
                            "status": "skipped",
                            "skip_reason": str(skip_reason),
                            "baseline_predicted_label": int(bundle.predicted_label),
                            "diagnostic_area": int(record.diagnostic_mask.sum()),
                            "nondiagnostic_area": int(record.nondiagnostic_mask.sum()),
                            "positive_area": int(record.positive_mask.sum()),
                            "negative_area": int(record.negative_mask.sum()),
                        }
                    )
                    continue
                self._write_cache(
                    probe_id=int(probe_id),
                    probe_label=int(probe_label),
                    class_scores=np.asarray(bundle.class_scores, dtype=np.float64),
                    foreground_mask=record.foreground_mask,
                    importance_map_signed=result.importance_map_signed,
                    diagnostic_mask=record.diagnostic_mask,
                    positive_mask=record.positive_mask,
                    negative_mask=record.negative_mask,
                    metadata=record.metadata,
                    window_df=result.window_results.assign(probe_label=probe_label),
                    pixel_df=result.pixel_ranking.assign(probe_id=int(probe_id), probe_label=probe_label),
                )
                records.append(record)
                inventory_rows.append(
                    {
                        "probe_id": int(probe_id),
                        "probe_label": probe_label,
                        "baseline_is_correct": int(record.baseline_is_correct),
                        "probe_partition": str(record.metadata.get("probe_partition", "")),
                        "status": "computed",
                        "skip_reason": "",
                        "baseline_predicted_label": int(bundle.predicted_label),
                        "baseline_margin": float(margin.margin),
                        "foreground_area": int(record.foreground_mask.sum()),
                        "has_importance_map": int(record.importance_map_signed is not None),
                        "selected_area": int(record.selected_area),
                        "diagnostic_area": int(record.diagnostic_mask.sum()),
                        "nondiagnostic_area": int(record.nondiagnostic_mask.sum()),
                        "positive_area": int(record.positive_mask.sum()),
                        "negative_area": int(record.negative_mask.sum()),
                        "baseline_wrong0_label": int(record.baseline_wrong0_label),
                        "baseline_wrong0_score": float(record.baseline_wrong0_score),
                        "baseline_margin_fixed_wrong0": float(record.baseline_margin_fixed_wrong0),
                        "importance_min": float(np.nanmin(record.importance_map_signed)),
                        "importance_max": float(np.nanmax(record.importance_map_signed)),
                        "importance_mean": float(np.nanmean(record.importance_map_signed)),
                        "importance_std": float(np.nanstd(record.importance_map_signed)),
                        "direction_score_semantics": str(record.direction_score_semantics),
                        "record_source": str(record.record_source),
                        "wrong_mask_semantics": str(record.wrong_mask_semantics),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                inventory_rows.append(
                    {
                        "probe_id": int(probe_id),
                        "probe_label": probe_label,
                        "status": "skipped",
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return records, pd.DataFrame(inventory_rows)


def load_probe_diagnostics(
    *,
    results_dir: str | Path,
    dataset,
    probe_ids: Sequence[int] | None = None,
) -> Tuple[List[ProbeDiagnosticRecord], pd.DataFrame]:
    provider = ResultsProbeDiagnosticProvider(
        results_dir=results_dir,
        dataset=dataset,
        probe_ids=probe_ids,
    )
    return provider.collect()


def build_probe_diagnostics(
    *,
    dataset,
    model_runtime: ModelRuntime,
    scan_config: ScanConfig,
    cache_dir: str | Path | None = None,
    probe_ids: Sequence[int],
) -> Tuple[List[ProbeDiagnosticRecord], pd.DataFrame]:
    provider = ComputeProbeDiagnosticProvider(
        dataset=dataset,
        model_runtime=model_runtime,
        scan_config=scan_config,
        cache_dir=cache_dir,
        probe_ids=probe_ids,
    )
    return provider.collect()
