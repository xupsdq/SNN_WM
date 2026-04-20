from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.common.seed import mix_seed
from src.experiments.common.voltage_readout import compute_voltage_margin, get_group_voltage_scores
from src.experiments.diagnostic.shared.discovery_engine import (
    ModelRuntime,
    ProbeDataSource,
    ScanConfig,
    TracePolicy,
    run_deterministic_discovery,
)


@dataclass(frozen=True)
class RegionEstimationSpec:
    dt: float
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round(float(self.sample_ms) / float(self.dt)))

    @property
    def probe_steps(self) -> int:
        return int(round(float(self.probe_ms) / float(self.dt)))


def _region_cache_paths(save_dir: Path) -> tuple[Path, Path, Path]:
    return (
        save_dir / "diagnostic_region_summary.csv",
        save_dir / "diagnostic_region_table.csv",
        save_dir / "diagnostic_mask_lookup.pt",
    )


def _baseline_probe_scan(
    *,
    net,
    encoder,
    raw_images: torch.Tensor,
    dataset_labels: np.ndarray,
    spec,
    device: torch.device,
    batch_size: int,
    probe_pool_limit: int,
    probe_pool_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    rng = np.random.default_rng(mix_seed(seed, 91))
    candidate_rank = 0
    for probe_label in sorted({int(label) for label in dataset_labels.tolist()}):
        ids = np.flatnonzero(dataset_labels == int(probe_label)).astype(np.int64)
        ids = rng.permutation(ids)[: int(probe_pool_per_class)]
        for probe_id in ids.tolist():
            rows.append(
                {
                    "candidate_rank": int(candidate_rank),
                    "probe_id": int(probe_id),
                    "probe_label": int(probe_label),
                }
            )
            candidate_rank += 1
            if len(rows) >= int(probe_pool_limit):
                break
        if len(rows) >= int(probe_pool_limit):
            break
    candidate_pool = pd.DataFrame(rows)
    if candidate_pool.empty:
        raise ValueError("No probe candidates available for diagnostic region estimation.")

    records: List[Dict[str, object]] = []
    for start in range(0, len(candidate_pool), int(batch_size)):
        batch = candidate_pool.iloc[start : start + int(batch_size)].copy()
        bundles = get_group_voltage_scores(
            net=net,
            encoder=encoder,
            probe_images=raw_images[batch["probe_id"].tolist()],
            spec=spec,
            device=device,
            readout_mode="decision_offset",
            readout_step=None,
            pooling="top_m_mean",
            m=1,
            stsp_mode="static_frozen",
        )
        for bundle, row in zip(bundles, batch.itertuples(index=False)):
            margin = compute_voltage_margin(bundle, true_label=int(row.probe_label))
            records.append(
                {
                    "candidate_rank": int(row.candidate_rank),
                    "probe_id": int(row.probe_id),
                    "probe_label": int(row.probe_label),
                    "predicted_label": int(bundle.predicted_label),
                    "first_fire_t_probe": int(bundle.first_fire_t_probe),
                    "is_correct": int(bundle.predicted_label == int(row.probe_label)),
                    "baseline_is_correct": int(bundle.predicted_label == int(row.probe_label)),
                    "true_score": float(margin.true_score),
                    "best_wrong_score": float(margin.best_wrong_score),
                    "best_wrong_label": int(margin.best_wrong_label),
                    "margin": float(margin.margin),
                }
            )
    return pd.DataFrame(records).sort_values(["candidate_rank"], kind="stable").reset_index(drop=True)


def _select_probe_rows(df_baseline: pd.DataFrame, trial_count: int) -> pd.DataFrame:
    if df_baseline.empty:
        raise ValueError("Baseline probe scan returned no rows.")
    grouped: Dict[int, List[Dict[str, object]]] = {}
    prioritized = df_baseline.sort_values(
        ["baseline_is_correct", "candidate_rank"],
        ascending=[False, True],
        kind="stable",
    )
    for probe_label, group in prioritized.groupby("probe_label", sort=True):
        grouped[int(probe_label)] = group.to_dict("records")
    cursors = {label: 0 for label in grouped}
    selected: List[Dict[str, object]] = []
    labels = sorted(grouped)
    while len(selected) < int(trial_count):
        made_progress = False
        for label in labels:
            if cursors[label] >= len(grouped[label]):
                continue
            selected.append(dict(grouped[label][cursors[label]]))
            cursors[label] += 1
            made_progress = True
            if len(selected) >= int(trial_count):
                break
        if not made_progress:
            break
    if not selected:
        raise ValueError("No probes selected for diagnostic region estimation.")
    return pd.DataFrame(selected).reset_index(drop=True)


def _build_region_rows(
    *,
    probe_id: int,
    probe_label: int,
    projected_map: np.ndarray,
    diagnostic_mask: np.ndarray,
    nondiagnostic_mask: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    patch_id = 0
    for region_name, mask in (
        ("diagnostic", np.asarray(diagnostic_mask, dtype=bool)),
        ("nondiagnostic", np.asarray(nondiagnostic_mask, dtype=bool)),
    ):
        coords = np.argwhere(mask)
        for row_idx, col_idx in coords.tolist():
            rows.append(
                {
                    "probe_id": int(probe_id),
                    "probe_label": int(probe_label),
                    "patch_id": int(patch_id),
                    "region_type": str(region_name),
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "importance": float(np.asarray(projected_map, dtype=np.float64)[row_idx, col_idx]),
                }
            )
            patch_id += 1
    return rows


def estimate_diagnostic_regions(
    *,
    net,
    encoder,
    raw_images: torch.Tensor,
    dataset_labels: np.ndarray,
    spec,
    trial_count: int,
    patch_size: int,
    diagnostic_method: str,
    batch_size: int,
    baseline_batch_size: int,
    device: torch.device,
    seed: int,
    save_dir: Path,
    cache_diagnostic_regions: bool,
    probe_pool_limit: int,
    probe_pool_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, np.ndarray | int | str]]]:
    if str(diagnostic_method) != "occlusion":
        raise ValueError(f"Unsupported diagnostic_method: {diagnostic_method}")

    save_dir = Path(save_dir)
    summary_path, table_path, lookup_path = _region_cache_paths(save_dir)
    if bool(cache_diagnostic_regions) and summary_path.exists() and table_path.exists() and lookup_path.exists():
        return (
            pd.read_csv(summary_path),
            pd.read_csv(table_path),
            torch.load(lookup_path),
        )

    model_runtime = ModelRuntime(
        net=net,
        encoder=encoder,
        model_path="",
        spec=spec,
        device=device,
        readout_mode="decision_offset",
        readout_step=None,
        voltage_pooling="top_m_mean",
        top_m=1,
    )
    df_baseline = _baseline_probe_scan(
        net=net,
        encoder=encoder,
        raw_images=raw_images,
        dataset_labels=dataset_labels,
        spec=spec,
        device=device,
        batch_size=int(baseline_batch_size),
        probe_pool_limit=int(probe_pool_limit),
        probe_pool_per_class=int(probe_pool_per_class),
        seed=int(seed),
    )
    selected = _select_probe_rows(df_baseline, trial_count=int(trial_count))

    probe_rows: List[Dict[str, object]] = []
    region_rows: List[Dict[str, object]] = []
    mask_lookup: Dict[int, Dict[str, np.ndarray | int | str]] = {}
    trace_policy = TracePolicy(mode="none")

    for row in selected.itertuples(index=False):
        probe_id = int(row.probe_id)
        probe_label = int(row.probe_label)
        scan_config = ScanConfig(
            model_path="",
            patch_size=int(patch_size),
            scan_stride=1,
            batch_size=int(batch_size),
            micro_batch_size=max(1, min(int(batch_size), 16)),
            lambda_global=1.0,
            debug_full_stride_override=False,
        )
        fixed_competitor = int(row.best_wrong_label) if int(row.best_wrong_label) >= 0 and int(row.best_wrong_label) != probe_label else None
        result = run_deterministic_discovery(
            scan_config=scan_config,
            data_source=ProbeDataSource(
                probe_id=probe_id,
                probe_label=probe_label,
                probe_image=raw_images[probe_id].to(device),
            ),
            model_runtime=model_runtime,
            trace_policy=trace_policy,
            fixed_competitor_label=fixed_competitor,
        )
        foreground_mask = np.asarray(raw_images[probe_id][0].detach().cpu().numpy() > 0.0, dtype=bool)
        diagnostic_mask = np.asarray(result.positive_importance_mask, dtype=bool) & foreground_mask
        nondiagnostic_mask = np.asarray(result.negative_importance_mask, dtype=bool) & foreground_mask
        is_valid = int(diagnostic_mask.any() or nondiagnostic_mask.any())
        exclusion_reason = ""
        if is_valid != 1:
            exclusion_reason = "empty_signed_importance_regions"
        region_rows.extend(
            _build_region_rows(
                probe_id=probe_id,
                probe_label=probe_label,
                projected_map=result.projected_discriminative_map,
                diagnostic_mask=diagnostic_mask,
                nondiagnostic_mask=nondiagnostic_mask,
            )
        )
        mask_lookup[probe_id] = {
            "probe_id": int(probe_id),
            "probe_label": int(probe_label),
            "diagnostic_mask": np.asarray(diagnostic_mask, dtype=np.bool_),
            "nondiagnostic_mask": np.asarray(nondiagnostic_mask, dtype=np.bool_),
            "foreground_mask": np.asarray(foreground_mask, dtype=np.bool_),
            "importance_map": np.asarray(result.projected_discriminative_map, dtype=np.float32),
            "selection_semantics": "positive_vs_negative_signed_importance",
            "baseline_predicted_label": int(row.predicted_label),
            "baseline_is_correct": int(row.baseline_is_correct),
        }
        probe_rows.append(
            {
                "probe_id": int(probe_id),
                "probe_label": int(probe_label),
                "baseline_predicted_label": int(row.predicted_label),
                "baseline_is_correct": int(row.baseline_is_correct),
                "baseline_first_fire_t_probe": int(row.first_fire_t_probe),
                "baseline_margin": float(row.margin),
                "baseline_wrong0_label": int(row.best_wrong_label),
                "is_region_valid": int(is_valid),
                "region_exclusion_reason": str(exclusion_reason),
                "diagnostic_area": int(diagnostic_mask.sum()),
                "nondiagnostic_area": int(nondiagnostic_mask.sum()),
                "foreground_area": int(foreground_mask.sum()),
            }
        )

    probe_region_summary = pd.DataFrame(probe_rows).sort_values(["probe_label", "probe_id"], kind="stable").reset_index(drop=True)
    diagnostic_region_table = pd.DataFrame(region_rows).sort_values(["probe_id", "patch_id"], kind="stable").reset_index(drop=True)
    if bool(cache_diagnostic_regions):
        save_dir.mkdir(parents=True, exist_ok=True)
        probe_region_summary.to_csv(summary_path, index=False, encoding="utf-8")
        diagnostic_region_table.to_csv(table_path, index=False, encoding="utf-8")
        torch.save(mask_lookup, lookup_path)
    return probe_region_summary, diagnostic_region_table, mask_lookup
