from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .common.diagnostic_mask_utils import (
    PatchSpec,
    apply_ablation,
    count_patch_grid,
    iter_patch_grid,
    project_patch_values_to_image,
)
from .common.voltage_readout import (
    ProbeScoreBundle,
    compute_voltage_margin,
    compute_voltage_margin_fixed_competitor,
    run_voltage_inference_batch,
)


@dataclass(frozen=True)
class ScanConfig:
    model_path: str
    patch_size: int
    scan_stride: int
    batch_size: int
    micro_batch_size: int
    lambda_global: float
    debug_full_stride_override: bool = False


@dataclass(frozen=True)
class TracePolicy:
    mode: str = "none"
    margin_threshold: float = 0.0
    debug_sample_rate: float = 0.0
    include_window_ids: frozenset[str] = frozenset()
    max_traces_per_probe: int = 0


@dataclass(frozen=True)
class ModelRuntime:
    net: object
    encoder: object
    model_path: str
    spec: object
    device: torch.device
    readout_mode: str
    readout_step: int | None
    voltage_pooling: str
    top_m: int


@dataclass(frozen=True)
class ProbeDataSource:
    probe_id: int
    probe_label: int
    probe_image: torch.Tensor


@dataclass(frozen=True)
class SummaryConfig:
    probe_id: int
    probe_label: int
    lambda_global: float
    fixed_competitor_label: int | None = None


@dataclass(frozen=True)
class WindowBatch:
    specs: tuple[PatchSpec, ...]
    tensors: torch.Tensor


@dataclass(frozen=True)
class DiscoveryResult:
    window_results: pd.DataFrame
    nonzero_window_results: pd.DataFrame
    window_ranking: pd.DataFrame
    projected_discriminative_map: np.ndarray
    importance_map_signed: np.ndarray
    direction_score_map: np.ndarray
    pixel_ranking: pd.DataFrame
    base_scores: np.ndarray
    saved_traces: dict[str, dict[str, object]]
    active_value_col: str
    uses_fixed_competitor: bool
    projected_nonzero_mask: np.ndarray | None = None
    positive_importance_mask: np.ndarray | None = None
    negative_importance_mask: np.ndarray | None = None


class WindowGenerator:
    def __init__(self, scan_config: ScanConfig, image_shape: Sequence[int]) -> None:
        self.scan_config = scan_config
        self.channels, self.height, self.width = [int(item) for item in image_shape]

    def _effective_stride(self) -> int:
        return 1 if bool(self.scan_config.debug_full_stride_override) else int(self.scan_config.scan_stride)

    def window_count(self) -> int:
        return count_patch_grid(
            height=self.height,
            width=self.width,
            patch_size=int(self.scan_config.patch_size),
            stride=self._effective_stride(),
        )

    def iter_windows(self) -> Iterator[PatchSpec]:
        yield from iter_patch_grid(
            height=self.height,
            width=self.width,
            patch_size=int(self.scan_config.patch_size),
            stride=self._effective_stride(),
        )


class InferenceRunner:
    def __init__(self, model_runtime: ModelRuntime, trace_policy: TracePolicy) -> None:
        self.model_runtime = model_runtime
        self.trace_policy = trace_policy

    def score_base_probe(self, probe_image: torch.Tensor) -> ProbeScoreBundle:
        result = run_voltage_inference_batch(
            net=self.model_runtime.net,
            encoder=self.model_runtime.encoder,
            probe_images=probe_image.unsqueeze(0),
            spec=self.model_runtime.spec,
            device=self.model_runtime.device,
            readout_mode=self.model_runtime.readout_mode,
            readout_step=self.model_runtime.readout_step,
            pooling=self.model_runtime.voltage_pooling,
            m=self.model_runtime.top_m,
            stsp_mode="static_frozen",
            return_full_traces=False,
        )
        return result.bundles[0]

    def score_windows(self, masked_batch: torch.Tensor) -> list[ProbeScoreBundle]:
        result = run_voltage_inference_batch(
            net=self.model_runtime.net,
            encoder=self.model_runtime.encoder,
            probe_images=masked_batch,
            spec=self.model_runtime.spec,
            device=self.model_runtime.device,
            readout_mode=self.model_runtime.readout_mode,
            readout_step=self.model_runtime.readout_step,
            pooling=self.model_runtime.voltage_pooling,
            m=self.model_runtime.top_m,
            stsp_mode="static_frozen",
            return_full_traces=False,
        )
        return result.bundles

    def score_window_with_trace(self, masked_probe: torch.Tensor) -> dict[str, object]:
        result = run_voltage_inference_batch(
            net=self.model_runtime.net,
            encoder=self.model_runtime.encoder,
            probe_images=masked_probe.unsqueeze(0),
            spec=self.model_runtime.spec,
            device=self.model_runtime.device,
            readout_mode=self.model_runtime.readout_mode,
            readout_step=self.model_runtime.readout_step,
            pooling=self.model_runtime.voltage_pooling,
            m=self.model_runtime.top_m,
            stsp_mode="static_frozen",
            return_full_traces=True,
        )
        return {
            "bundle": result.bundles[0],
            "readout_step": int(result.readout_step),
            "state_traces": result.state_traces,
        }


class SummaryExtractor:
    def __init__(self, summary_config: SummaryConfig) -> None:
        self.summary_config = summary_config

    def extract_summary(
        self,
        batch_outputs: Sequence[ProbeScoreBundle],
        *,
        batch_specs: Sequence[PatchSpec],
        base_scores: ProbeScoreBundle,
    ) -> list[dict[str, object]]:
        base_margin = compute_voltage_margin(base_scores, true_label=int(self.summary_config.probe_label))
        fixed_competitor_label = (
            None if self.summary_config.fixed_competitor_label is None else int(self.summary_config.fixed_competitor_label)
        )
        base_margin_fixed = (
            compute_voltage_margin_fixed_competitor(
                base_scores,
                true_label=int(self.summary_config.probe_label),
                competitor_label=int(fixed_competitor_label),
            )
            if fixed_competitor_label is not None
            else None
        )
        rows: list[dict[str, object]] = []
        for spec, bundle in zip(batch_specs, batch_outputs):
            masked_margin = compute_voltage_margin(bundle, true_label=int(self.summary_config.probe_label))
            raw_importance = float(base_margin.margin - masked_margin.margin)
            global_drop = float(
                np.mean(
                    np.asarray(base_scores.class_scores, dtype=np.float64)
                    - np.asarray(bundle.class_scores, dtype=np.float64)
                )
            )
            discriminative = float(raw_importance - float(self.summary_config.lambda_global) * global_drop)
            if base_margin_fixed is not None:
                masked_margin_fixed = compute_voltage_margin_fixed_competitor(
                    bundle,
                    true_label=int(self.summary_config.probe_label),
                    competitor_label=int(fixed_competitor_label),
                )
                raw_importance_fixed = float(base_margin_fixed.margin - masked_margin_fixed.margin)
                discriminative_importance_fixed = float(
                    raw_importance_fixed - float(self.summary_config.lambda_global) * global_drop
                )
                full_fixed_competitor_score = float(base_margin_fixed.competitor_score)
                masked_fixed_competitor_score = float(masked_margin_fixed.competitor_score)
                full_margin_fixed = float(base_margin_fixed.margin)
                masked_margin_fixed_value = float(masked_margin_fixed.margin)
            else:
                raw_importance_fixed = float("nan")
                discriminative_importance_fixed = float("nan")
                full_fixed_competitor_score = float("nan")
                masked_fixed_competitor_score = float("nan")
                full_margin_fixed = float("nan")
                masked_margin_fixed_value = float("nan")
            rows.append(
                {
                    "window_id": f"{self.summary_config.probe_id}:dense:{int(spec.row_start)}:{int(spec.row_end)}:{int(spec.col_start)}:{int(spec.col_end)}",
                    "probe_id": int(self.summary_config.probe_id),
                    "scan_stage": "dense",
                    "row_start": int(spec.row_start),
                    "row_end": int(spec.row_end),
                    "col_start": int(spec.col_start),
                    "col_end": int(spec.col_end),
                    "full_true_score": float(base_margin.true_score),
                    "full_best_wrong_score": float(base_margin.best_wrong_score),
                    "masked_true_score": float(masked_margin.true_score),
                    "masked_best_wrong_score": float(masked_margin.best_wrong_score),
                    "masked_margin": float(masked_margin.margin),
                    "raw_importance": raw_importance,
                    "fixed_competitor_label": -1 if fixed_competitor_label is None else int(fixed_competitor_label),
                    "full_fixed_competitor_score": full_fixed_competitor_score,
                    "masked_fixed_competitor_score": masked_fixed_competitor_score,
                    "full_margin_fixed": full_margin_fixed,
                    "masked_margin_fixed": masked_margin_fixed_value,
                    "raw_importance_fixed": raw_importance_fixed,
                    "global_drop": global_drop,
                    "discriminative_importance": discriminative,
                    "discriminative_importance_fixed": discriminative_importance_fixed,
                    "cross_boundary_flag": int(masked_margin.margin <= 0.0 or bundle.predicted_label != base_scores.predicted_label),
                    "boundary_margin": float(abs(masked_margin.margin)),
                    "trace_saved_flag": 0,
                }
            )
        return rows


class ResultWriter:
    def __init__(self, expected_rows: int | None = None) -> None:
        self._records: list[dict[str, object] | None]
        if expected_rows is not None and expected_rows > 0:
            self._records = [None] * int(expected_rows)
        else:
            self._records = []
        self._write_index = 0

    def write(self, rows: Sequence[Mapping[str, object]]) -> None:
        if self._records and self._write_index < len(self._records):
            for row in rows:
                if self._write_index >= len(self._records):
                    self._records.append(dict(row))
                else:
                    self._records[self._write_index] = dict(row)
                self._write_index += 1
            return
        for row in rows:
            self._records.append(dict(row))
            self._write_index += 1

    def to_frame(self) -> pd.DataFrame:
        rows = [row for row in self._records if row is not None]
        return pd.DataFrame(rows)


class DeterministicDiscovery:
    def __init__(self, scan_config: ScanConfig, model_runtime: ModelRuntime, trace_policy: TracePolicy) -> None:
        self.scan_config = scan_config
        self.model_runtime = model_runtime
        self.trace_policy = trace_policy

    def run(self, data_source: ProbeDataSource) -> DiscoveryResult:
        return run_deterministic_discovery(
            scan_config=self.scan_config,
            data_source=data_source,
            model_runtime=self.model_runtime,
            trace_policy=self.trace_policy,
        )


DiscoveryRepeats = DeterministicDiscovery


def should_save_trace(summary_record: Mapping[str, object], trace_policy: TracePolicy) -> bool:
    if str(trace_policy.mode) == "none" or int(trace_policy.max_traces_per_probe) <= 0:
        return False
    window_id = str(summary_record["window_id"])
    if window_id in trace_policy.include_window_ids:
        return True
    if str(trace_policy.mode) in {"anomalous", "boundary", "filtered"}:
        if int(summary_record["cross_boundary_flag"]) == 1:
            return True
        if float(summary_record["boundary_margin"]) <= float(trace_policy.margin_threshold):
            return True
    if str(trace_policy.mode) == "debug":
        if float(trace_policy.debug_sample_rate) <= 0.0:
            return False
        stable_hash = sum(ord(ch) for ch in window_id) % 10000
        return stable_hash < int(round(float(trace_policy.debug_sample_rate) * 10000.0))
    return False


def extract_summary(
    batch_outputs: Sequence[ProbeScoreBundle],
    summary_config: SummaryConfig,
    base_scores: ProbeScoreBundle,
    batch_specs: Sequence[PatchSpec],
) -> list[dict[str, object]]:
    return SummaryExtractor(summary_config).extract_summary(
        batch_outputs,
        batch_specs=batch_specs,
        base_scores=base_scores,
    )


def _yield_window_batches(
    *,
    probe_image: torch.Tensor,
    window_specs: Iterable[PatchSpec],
    batch_size: int,
) -> Iterator[WindowBatch]:
    batch_specs: list[PatchSpec] = []
    batch_tensors: list[torch.Tensor] = []
    for spec in window_specs:
        mask = np.zeros(tuple(probe_image.shape[-2:]), dtype=bool)
        mask[int(spec.row_start):int(spec.row_end), int(spec.col_start):int(spec.col_end)] = True
        batch_specs.append(spec)
        batch_tensors.append(apply_ablation(probe_image, mask, fill_value=0.0))
        if len(batch_specs) >= int(batch_size):
            yield WindowBatch(specs=tuple(batch_specs), tensors=torch.stack(batch_tensors, dim=0))
            batch_specs = []
            batch_tensors = []
    if batch_specs:
        yield WindowBatch(specs=tuple(batch_specs), tensors=torch.stack(batch_tensors, dim=0))


def _score_micro_batches(
    inference_runner: InferenceRunner,
    batch_tensors: torch.Tensor,
    micro_batch_size: int,
) -> list[ProbeScoreBundle]:
    outputs: list[ProbeScoreBundle] = []
    for start in range(0, int(batch_tensors.shape[0]), int(micro_batch_size)):
        micro = batch_tensors[start:start + int(micro_batch_size)]
        outputs.extend(inference_runner.score_windows(micro))
    return outputs


def _rows_to_patch_specs(window_df: pd.DataFrame) -> list[PatchSpec]:
    patches: list[PatchSpec] = []
    for idx, row in enumerate(window_df.itertuples(index=False)):
        patches.append(
            PatchSpec(
                patch_id=int(idx),
                patch_row=0,
                patch_col=0,
                row_start=int(row.row_start),
                row_end=int(row.row_end),
                col_start=int(row.col_start),
                col_end=int(row.col_end),
            )
        )
    return patches


def _project_window_map(
    window_df: pd.DataFrame,
    image_height: int,
    image_width: int,
    *,
    value_col: str = "raw_importance",
) -> np.ndarray:
    if window_df.empty:
        return np.zeros((image_height, image_width), dtype=np.float64)
    patches = _rows_to_patch_specs(window_df)
    values = window_df[value_col].to_numpy(dtype=np.float64)
    return project_patch_values_to_image(image_height, image_width, patches, values)


def _build_window_ranking(window_df: pd.DataFrame, *, value_col: str = "raw_importance") -> pd.DataFrame:
    if window_df.empty:
        return window_df.copy()
    ranking = window_df.sort_values(
        by=[value_col, "row_start", "col_start"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
    return ranking


def _build_pixel_ranking(projected_map: np.ndarray) -> pd.DataFrame:
    values = np.asarray(projected_map, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            rows.append(
                {
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "importance": float(values[row_idx, col_idx]),
                }
            )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(
        by=["importance", "row", "col"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
    return ranking


def run_dense_scan(
    scan_config: ScanConfig,
    data_source: ProbeDataSource,
    model_runtime: ModelRuntime,
    trace_policy: TracePolicy,
    fixed_competitor_label: int | None = None,
) -> DiscoveryResult:
    window_generator = WindowGenerator(scan_config=scan_config, image_shape=data_source.probe_image.shape)
    inference_runner = InferenceRunner(model_runtime=model_runtime, trace_policy=trace_policy)
    base_scores = inference_runner.score_base_probe(data_source.probe_image)
    summary_config = SummaryConfig(
        probe_id=int(data_source.probe_id),
        probe_label=int(data_source.probe_label),
        lambda_global=float(scan_config.lambda_global),
        fixed_competitor_label=None if fixed_competitor_label is None else int(fixed_competitor_label),
    )
    writer = ResultWriter(expected_rows=window_generator.window_count())
    trace_cache: dict[str, dict[str, object]] = {}
    trace_count = 0

    for batch in _yield_window_batches(
        probe_image=data_source.probe_image,
        window_specs=window_generator.iter_windows(),
        batch_size=int(scan_config.batch_size),
    ):
        batch_outputs = _score_micro_batches(
            inference_runner=inference_runner,
            batch_tensors=batch.tensors,
            micro_batch_size=int(scan_config.micro_batch_size),
        )
        rows = extract_summary(
            batch_outputs=batch_outputs,
            summary_config=summary_config,
            base_scores=base_scores,
            batch_specs=batch.specs,
        )
        if str(trace_policy.mode) != "none":
            for local_idx, row in enumerate(rows):
                if trace_count >= int(trace_policy.max_traces_per_probe):
                    break
                if not should_save_trace(row, trace_policy):
                    continue
                trace_payload = inference_runner.score_window_with_trace(batch.tensors[local_idx])
                trace_cache[str(row["window_id"])] = {
                    "bundle": trace_payload["bundle"],
                    "readout_step": int(trace_payload["readout_step"]),
                    "state_traces": trace_payload["state_traces"],
                }
                row["trace_saved_flag"] = 1
                trace_count += 1
        writer.write(rows)
        del batch_outputs

    window_results = writer.to_frame()
    if window_results.empty:
        projected_map = np.zeros(tuple(data_source.probe_image.shape[-2:]), dtype=np.float64)
        return DiscoveryResult(
            window_results=window_results,
            nonzero_window_results=window_results.copy(),
            window_ranking=window_results.copy(),
            projected_discriminative_map=projected_map,
            importance_map_signed=projected_map,
            direction_score_map=projected_map,
            pixel_ranking=pd.DataFrame(columns=["rank", "row", "col", "importance"]),
            base_scores=np.asarray(base_scores.class_scores, dtype=np.float64),
            saved_traces=trace_cache,
            active_value_col="raw_importance_fixed" if fixed_competitor_label is not None else "raw_importance",
            uses_fixed_competitor=bool(fixed_competitor_label is not None),
            projected_nonzero_mask=np.zeros_like(projected_map, dtype=bool),
            positive_importance_mask=np.zeros_like(projected_map, dtype=bool),
            negative_importance_mask=np.zeros_like(projected_map, dtype=bool),
        )

    active_value_col = "raw_importance_fixed" if fixed_competitor_label is not None else "raw_importance"
    active_values = window_results[active_value_col].to_numpy(dtype=np.float64)
    nonzero_mask = np.isfinite(active_values) & (active_values != 0.0)
    nonzero_window_results = window_results.loc[nonzero_mask].reset_index(drop=True)
    window_ranking = _build_window_ranking(window_results, value_col=active_value_col)
    projected_map = _project_window_map(
        window_results,
        image_height=int(data_source.probe_image.shape[-2]),
        image_width=int(data_source.probe_image.shape[-1]),
        value_col=active_value_col,
    )
    pixel_ranking = _build_pixel_ranking(projected_map)
    projected_nonzero_mask = np.isfinite(projected_map) & (projected_map != 0.0)
    positive_importance_mask = np.isfinite(projected_map) & (projected_map > 0.0)
    negative_importance_mask = np.isfinite(projected_map) & (projected_map < 0.0)
    return DiscoveryResult(
        window_results=window_results,
        nonzero_window_results=nonzero_window_results,
        window_ranking=window_ranking,
        projected_discriminative_map=projected_map,
        importance_map_signed=np.asarray(projected_map, dtype=np.float64),
        direction_score_map=np.asarray(projected_map, dtype=np.float64),
        pixel_ranking=pixel_ranking,
        base_scores=np.asarray(base_scores.class_scores, dtype=np.float64),
        saved_traces=trace_cache,
        active_value_col=str(active_value_col),
        uses_fixed_competitor=bool(fixed_competitor_label is not None),
        projected_nonzero_mask=np.asarray(projected_nonzero_mask, dtype=bool),
        positive_importance_mask=np.asarray(positive_importance_mask, dtype=bool),
        negative_importance_mask=np.asarray(negative_importance_mask, dtype=bool),
    )


def run_deterministic_discovery(
    scan_config: ScanConfig,
    data_source: ProbeDataSource,
    model_runtime: ModelRuntime,
    trace_policy: TracePolicy,
    fixed_competitor_label: int | None = None,
) -> DiscoveryResult:
    if int(scan_config.scan_stride) <= 0:
        raise ValueError("scan_stride must be positive")
    if int(scan_config.batch_size) <= 0 or int(scan_config.micro_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if int(scan_config.patch_size) <= 0:
        raise ValueError("patch_size must be positive")
    return run_dense_scan(
        scan_config=scan_config,
        data_source=data_source,
        model_runtime=model_runtime,
        trace_policy=trace_policy,
        fixed_competitor_label=fixed_competitor_label,
    )


def run_discovery_repeats(*args, **kwargs):
    forbidden = {
        "repeats",
        "n_repeats",
        "repeat_id",
        "repeat_count",
        "importance_seeds",
        "translation_jitter_px",
        "noise_std",
        "model_specs",
        "loaded_models",
    }
    present = sorted(name for name in forbidden if name in kwargs)
    if present:
        raise ValueError(f"Deprecated repeat-era arguments are not supported: {', '.join(present)}")
    warnings.warn(
        "run_discovery_repeats is deprecated; use run_deterministic_discovery instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_deterministic_discovery(*args, **kwargs)
