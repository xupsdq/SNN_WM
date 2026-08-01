from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config.units import ms


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


@dataclass(frozen=True)
class Fig6Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sequence_lengths: tuple[int, ...] = (10,)
    primary_sequence_length: int = 7
    sample_ms: int = 200
    delay_ms: int = 200
    ping_ms: int = 30
    ping_amp: float = 1.0
    probe_ms: int = 100
    batch_size: int = 8
    num_sequences: int = 100
    num_probe_candidates_per_sequence: int = 8
    peak_q: float = 0.20
    recent_window: int = 2
    multi_update_threshold: int = 2
    n_null: int = 100
    n_matched_groups: int = 100
    foreground_threshold: float = 0.1
    functional_restore_mode: str = "stsp_only"
    save_full_traces: bool = False
    save_l3_trace: bool = True
    save_spike_cache: bool = False
    run_sequence_bank: bool = False
    run_peak_source_attribution: bool = False
    run_peak_update_history: bool = False
    run_peak_input_overlap_origin: bool = False
    run_real_reentry_rollout: bool = False
    run_real_downstream_metrics: bool = False
    run_peak_enrichment: bool = False
    run_update_recency_model: bool = False
    run_peak_weighted_overlap: bool = False
    run_reentry_prediction: bool = False
    run_downstream_prediction: bool = False
    run_peak_perturbation: bool = False
    run_supplement: bool = False
    run_legacy_supplement: bool = False
    run_score_shuffle_null: bool = False
    run_overlap_threshold_sensitivity: bool = False
    run_field_ping_readout: bool = False
    run_global_ping_score_spike_prediction: bool = False
    run_ping_score_spike_prediction: bool = False
    run_real_probe_score_spike_deflection: bool = False
    run_overlap_gated_stsp_recruitment: bool = False
    run_high_stsp_overlap_ablation: bool = True
    run_score_basin_sparsification: bool = False
    run_fig6_downstream_exploratory: bool = False
    force_main_outputs: bool = True
    score_eps: float = 1e-6
    score_early_windows_ms: tuple[int, ...] = (5, 10, 15, 20)
    primary_score_early_window_ms: int = 10
    score_n_bins: int = 5
    basin_radius: int = 2
    basin_top_q: float = 0.20
    gain_ratio_clip_quantiles: tuple[float, float] = (0.01, 0.99)
    real_probe_entry_mode: str = "encoded_spike"
    score_use_log_gain: bool = False
    stsp_group_quantile: float = 0.20
    fig6e_stsp_group_quantile: float = 0.50
    overlap_threshold: float = 0.05
    global_ping_amp: float = 0.5
    global_ping_ms: int = 30
    recent_overlap_windows: tuple[int, ...] = (2, 3, 4, 5)
    leave_one_out_mode: str = "blank_same_timing"
    real_reentry_reference_conditions: tuple[str, ...] = ("S_final", "S0")
    real_rollout_required_for_main: bool = True
    save_debug_figures: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_probe_batch: bool = False
    enable_high_stsp_ablation_batch: bool = False
    enable_sequence_bank_batch: bool = False
    enable_leave_one_out_batch: bool = False
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def delay_steps(self) -> int:
        return _ms_to_steps(self.delay_ms, self.dt)

    @property
    def probe_steps(self) -> int:
        return _ms_to_steps(self.probe_ms, self.dt)

    @property
    def ping_steps(self) -> int:
        return _ms_to_steps(self.ping_ms, self.dt)

    @property
    def global_ping_steps(self) -> int:
        return _ms_to_steps(self.global_ping_ms, self.dt)

    @property
    def score_early_window_steps(self) -> tuple[int, ...]:
        return tuple(_ms_to_steps(v, self.dt) for v in self.score_early_windows_ms)


@dataclass
class ExperimentContext:
    cfg: Fig6Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    meta_dir: Path
    device: Any
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any | None
    encoder: Any | None
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    run_log: list[str]
    n_sequences: int = 0
    n_probe_candidates: int = 0
    n_matched_groups: int = 0


@dataclass
class PeakAmplifiedReentryBank:
    sequence_trials: pd.DataFrame
    sequence_meta: pd.DataFrame
    probe_trials: pd.DataFrame
    matched_groups: pd.DataFrame
    update_count: np.ndarray
    last_update_position: np.ndarray
    time_since_last_update: np.ndarray
    update_exposure_by_item: np.ndarray
    item_activation_history: np.ndarray
    g_baseline: np.ndarray
    g_final: np.ndarray
    delta_support: np.ndarray
    peak_mask: np.ndarray
    nonpeak_mask: np.ndarray
    prior_updated_mask: np.ndarray
    boundaries: dict[int, Mapping[str, Mapping[str, Any]]]
    reentry_metrics: pd.DataFrame
    downstream_metrics: pd.DataFrame


__all__ = ["ExperimentContext", "Fig6Config", "PeakAmplifiedReentryBank"]
