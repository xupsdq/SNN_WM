from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - import error is reported by runtime validation.
    torch = None  # type: ignore[assignment]

from src.config.units import ms
try:
    from src.experiments.common.dataset import build_class_index, encode_images
    from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
    from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
    from src.experiments.common.model_io import load_model_and_encoder
    from src.experiments.common.monitored_dms import snapshot_boundary_state
    from src.experiments.common.ping_common import prepare_network_state
    from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
    from src.experiments.common.runtime import resolve_device, seed_everything
    _COMMON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only in minimal Python envs.
    _COMMON_IMPORT_ERROR = exc

    def build_class_index(dataset, num_classes: int) -> dict[int, list[int]]:
        out = {label: [] for label in range(int(num_classes))}
        for idx in range(len(dataset)):
            label = int(dataset[idx][1])
            if label in out:
                out[label].append(idx)
        return out

    def encode_images(*args, **kwargs):
        raise RuntimeError(f"Shared spike encoder unavailable: {_COMMON_IMPORT_ERROR}")

    def decode_prediction_and_fire_time_from_layer3(*args, **kwargs):
        raise RuntimeError(f"Shared layer3 decoder unavailable: {_COMMON_IMPORT_ERROR}")

    def load_mnist_skeleton_dataset(*args, **kwargs):
        raise RuntimeError(f"Shared MNIST loader unavailable: {_COMMON_IMPORT_ERROR}")

    def load_model_and_encoder(*args, **kwargs):
        raise RuntimeError(f"Shared model loader unavailable: {_COMMON_IMPORT_ERROR}")

    def snapshot_boundary_state(*args, **kwargs):
        return {}

    def prepare_network_state(*args, **kwargs) -> None:
        return None

    def build_run_info(**kwargs):
        return dict(kwargs)

    def finalize_run_info(meta_dir: Path, run_info: Mapping[str, Any], *, status: str):
        payload = dict(run_info)
        payload["status"] = status
        write_run_info(meta_dir, payload)

    def write_run_info(meta_dir: Path, run_info: Mapping[str, Any]):
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "run_info.json").write_text(json.dumps(_json_safe(run_info), indent=2, sort_keys=True), encoding="utf-8")

    def resolve_device(device: str):
        return "cpu"

    def seed_everything(seed: int) -> None:
        np.random.seed(int(seed))
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig6_peak_amplified_reentry"
FIG6_DESIGN_VERSION = "peak_amplified_overlap_reentry_real_rollout"
PRIMARY_LAYER = "layer1"
STATE_VARIABLE = "g"
MAIN_PANELS = {
    "A": "leave-one-item-out source attribution of final STSP peaks",
    "B": "peak probability by update count",
    "C": "alignment of final peaks with foreground-overlap routes",
    "D": "route-peak perturbation reduces real re-entry",
    "E": "route-peak perturbation changes downstream output",
    "F": "mechanism schematic: overlap route, peak gain",
}
MAIN_CLAIM = (
    "Final STSP peaks preferentially arise at locations receiving repeated and recent updates, "
    "align with recent foreground-overlap routes, and act as gain on overlap-aligned re-entry "
    "when route-peak perturbation audit supports causal use."
)
MECHANISM_BOUNDARY = {
    "overlap": "route",
    "peaks": "gain",
    "summary": "raw overlap defines the route; peak-weighted overlap modulates gain along that route",
    "forbidden_claims": [
        "peaks replace overlap",
        "peaks are the primary route",
        "peaks causally control re-entry without perturbation evidence",
    ],
}
SUPPLEMENT_PLAN = {
    "S11": "peak-origin and overlap-interface controls",
    "S12": "peak-weighted re-entry and downstream controls",
}
MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_a_peak_source_attribution.csv",
    "data/metrics/panel_a_peak_source_attribution_summary.csv",
    "data/metrics/panel_b_peak_update_history.csv",
    "data/metrics/panel_b_peak_update_history_summary.csv",
    "data/metrics/panel_c_peak_input_overlap_similarity.csv",
    "data/metrics/panel_c_peak_input_overlap_summary.csv",
    "data/raw/panel_d_route_peak_perturbation_trial_readout.csv",
    "data/metrics/panel_d_route_peak_reentry_loss_summary.csv",
    "data/metrics/panel_d_route_peak_reentry_loss_contrast.csv",
    "data/metrics/panel_d_route_peak_perturbation_audit.csv",
    "data/raw/panel_e_route_peak_downstream_trial_readout.csv",
    "data/metrics/panel_e_route_peak_downstream_summary.csv",
    "data/metrics/panel_e_route_peak_downstream_contrast.csv",
    "data/metrics/panel_e_route_peak_output_distribution.csv",
    "data/metrics/panel_de_route_peak_perturbation_scientific_use_audit.csv",
]
SUPPLEMENTARY_OUTPUTS = [
    "data/metrics/supp_s11_peak_update_group_enrichment.csv",
    "data/metrics/supp_s11_update_recency_model_comparison.csv",
    "data/metrics/supp_s11_leave_one_out_source_details.csv",
    "data/metrics/supp_s11_recent_overlap_window_robustness.csv",
    "data/metrics/supp_s11_alternative_peak_definitions.csv",
    "data/metrics/supp_s11_visual_energy_classpair_controls.csv",
    "data/metrics/supp_s12_raw_overlap_matched_peak_overlap_contrast.csv",
    "data/metrics/supp_s12_peak_weighted_regression_controls.csv",
    "data/metrics/supp_s12_real_rollout_scientific_use_audit.csv",
    "data/metrics/supp_s12_downstream_metric_breakdown.csv",
    "data/metrics/supp_s12_global_support_spike_count_controls.csv",
]
OPTIONAL_SUPPLEMENTARY_OUTPUTS = [
    "data/metrics/supp_s12_peak_perturbation_metrics.csv",
    "data/metrics/supp_s12_peak_perturbation_summary.csv",
]
PERTURBATION_UNIT_SET_ORDER = ("route_peak", "route_nonpeak", "nonroute_peak", "random_matched")
PERTURBATION_UNIT_SET_LABELS = {
    "route_peak": "Route peak",
    "route_nonpeak": "Route non-peak",
    "nonroute_peak": "Non-route peak",
    "random_matched": "Random",
}
UPDATE_GROUPS = ("single_old", "multi_old", "single_recent", "multi_recent")
MODEL_NAMES = (
    "baseline_only",
    "update_only",
    "recency_only",
    "overlap_only",
    "update_plus_recency",
    "update_times_recency",
)
DOWNSTREAM_METRICS = (
    "early_recruitment_gain",
    "P_advance",
    "P_recruit",
    "spike_advance",
    "response_pattern_displacement",
    "decision_deflection_score",
    "partial_cue_completion_gain",
)


@dataclass(frozen=True)
class Fig6Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sequence_lengths: tuple[int, ...] = (10)
    primary_sequence_length: int = 7
    sample_ms: int = 200
    delay_ms: int = 200
    probe_ms: int = 100
    batch_size: int = 8
    num_sequences: int = 100
    num_probe_candidates_per_sequence: int = 8
    peak_q: float = 0.20
    recent_window: int = 2
    multi_update_threshold: int = 2
    n_null: int = 100
    n_matched_groups: int = 100
    foreground_threshold: float = 0.0
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
    recent_overlap_windows: tuple[int, ...] = (2, 3, 4, 5)
    leave_one_out_mode: str = "blank_same_timing"
    real_reentry_reference_conditions: tuple[str, ...] = ("S_final", "S0")
    real_rollout_required_for_main: bool = True
    save_debug_figures: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_probe_batch: bool = False
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


class ProxyDataset:
    def __init__(self, *, n: int = 1000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.images: list[Any] = []
        self.labels: list[int] = []
        yy, xx = np.mgrid[0:28, 0:28]
        for idx in range(n):
            label = idx % 10
            cx = 4 + (label % 5) * 5 + rng.normal(0, 0.6)
            cy = 6 + (label // 5) * 10 + rng.normal(0, 0.6)
            blob = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (10.0 + label))
            stripe = ((xx + yy + label * 2) % (6 + label % 3) == 0).astype(float) * 0.35
            image = np.clip(blob + stripe + rng.normal(0, 0.025, (28, 28)), 0, 1).astype(np.float32)
            self.images.append(_to_tensor(image))
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[int(idx)].reshape(1, 28, 28), int(self.labels[int(idx)])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig6Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    warnings_list: list[str] = []
    device = resolve_device(cfg.device) if torch is not None else "cpu"
    dataset = _load_dataset_or_proxy(cfg.dataset_root, cfg.split, int(cfg.network_seed), warnings_list)
    class_index = build_class_index(dataset, 10)
    net = None
    encoder = None
    if torch is None:
        warnings_list.append("PyTorch import failed; using deterministic image-driven proxy mode.")
    elif Path(cfg.model_path).exists():
        try:
            net, encoder = load_model_and_encoder(
                cfg.model_path,
                device=device,
                dt=cfg.dt,
                max_duration_ms=max(cfg.sample_ms, cfg.probe_ms, 100),
            )
        except Exception as exc:
            warnings_list.append(f"Model load failed; using deterministic image-driven proxy mode: {exc}")
    else:
        warnings_list.append(f"Model checkpoint not found at {cfg.model_path}; using deterministic image-driven proxy mode.")

    ctx = ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
        meta_dir=dirs["meta"],
        device=device,
        dataset=dataset,
        class_index=class_index,
        net=net,
        encoder=encoder,
        warnings=warnings_list,
        output_files={},
        completed_modules={},
        run_log=[f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"],
    )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(ctx.meta_dir, run_info)
    try:
        _write_config_files(ctx)
        needs_sequence_trials = any(
            (
                cfg.run_sequence_bank,
                cfg.run_peak_source_attribution,
                cfg.run_peak_update_history,
                cfg.run_peak_input_overlap_origin,
                cfg.run_real_reentry_rollout,
                cfg.run_real_downstream_metrics,
                cfg.run_peak_enrichment,
                cfg.run_update_recency_model,
                cfg.run_peak_weighted_overlap,
                cfg.run_reentry_prediction,
                cfg.run_downstream_prediction,
                cfg.run_peak_perturbation,
                cfg.run_supplement,
            )
        )
        needs_bank = needs_sequence_trials
        needs_later_probe_trials = any(
            (
                cfg.run_real_reentry_rollout,
                cfg.run_real_downstream_metrics,
                cfg.run_reentry_prediction,
                cfg.run_downstream_prediction,
                cfg.run_peak_perturbation,
                cfg.run_supplement,
            )
        )
        needs_real_rollout = any(
            (
                cfg.run_real_reentry_rollout,
                cfg.run_real_downstream_metrics,
                cfg.run_peak_weighted_overlap,
                cfg.run_reentry_prediction,
                cfg.run_downstream_prediction,
                cfg.run_supplement,
                cfg.run_peak_perturbation,
            )
        )
        sequence_trials: pd.DataFrame | None = None
        bank: PeakAmplifiedReentryBank | None = None
        if needs_sequence_trials:
            sequence_trials = build_sequence_trials(ctx)
        if needs_bank and sequence_trials is not None:
            bank = run_sequence_bank(ctx, sequence_trials)
        if bank is not None and (cfg.run_peak_source_attribution or cfg.run_supplement):
            loo_bank = run_leave_one_item_out_support_bank(ctx, bank)
            compute_peak_source_attribution(ctx, bank, loo_bank)
        if bank is not None and (cfg.run_peak_update_history or cfg.run_supplement):
            compute_peak_update_history(ctx, bank)
        if bank is not None and (cfg.run_peak_input_overlap_origin or cfg.run_supplement):
            compute_peak_input_overlap_origin(ctx, bank)
        if bank is not None and cfg.run_peak_enrichment:
            define_final_peaks_and_update_groups(ctx, bank)
        if bank is not None and cfg.run_update_recency_model:
            if not (ctx.metrics_dir / "supp_legacy_panel_a_multi_recent_peak_enrichment.csv").exists():
                define_final_peaks_and_update_groups(ctx, bank)
            fit_update_recency_support_models(ctx, bank)
        if bank is not None and cfg.run_peak_weighted_overlap:
            compute_peak_weighted_overlap_definitions(ctx, bank)
        if bank is not None and needs_later_probe_trials:
            build_later_probe_peak_overlap_trials(ctx, bank)
        if bank is not None and needs_real_rollout:
            if bank.probe_trials.empty:
                build_later_probe_peak_overlap_trials(ctx, bank)
            run_real_probe_reentry_rollouts(ctx, bank)
        if bank is not None and (cfg.run_real_reentry_rollout or cfg.run_supplement):
            compute_real_peak_weighted_reentry_metrics(ctx, bank)
        if bank is not None and cfg.run_peak_weighted_overlap and not bank.reentry_metrics.empty:
            compute_real_peak_weighted_reentry_metrics(ctx, bank)
        if bank is not None and (cfg.run_real_downstream_metrics or cfg.run_supplement):
            compute_real_peak_overlap_downstream_metrics(ctx, bank)
        if bank is not None and cfg.run_reentry_prediction:
            compute_peak_weighted_reentry_metrics(ctx, bank)
        if bank is not None and cfg.run_downstream_prediction:
            compute_peak_weighted_downstream_metrics(ctx, bank)
        if bank is not None and cfg.run_supplement:
            compute_supplement_outputs(ctx, bank)
        if bank is not None and cfg.run_peak_perturbation:
            compute_route_peak_perturbation_outputs(ctx, bank)
        elif cfg.run_peak_perturbation:
            write_route_peak_perturbation_unavailable_outputs(ctx, reason="sequence_bank_unavailable")
        write_fig6_supplement_aliases(ctx)
        if bank is not None:
            write_global_mechanism_metadata(ctx)
        if cfg.save_debug_figures:
            save_debug_figures(ctx)
        summary = _write_summary(ctx)
        _write_run_log(ctx)
        finalize_run_info(ctx.meta_dir, run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(ctx.meta_dir, run_info, status="failed")
        raise


def build_sequence_trials(ctx: ExperimentContext) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    lengths = list(cfg.sequence_lengths)
    image_ids_by_label = {label: np.asarray(ids, dtype=np.int64) for label, ids in ctx.class_index.items()}
    rows: list[dict[str, Any]] = []
    for sequence_id in _progress(range(int(cfg.num_sequences)), total=int(cfg.num_sequences), desc="fig6 sequence specs", enabled=cfg.show_progress):
        seq_len = int(lengths[sequence_id % len(lengths)])
        labels = rng.choice(np.arange(10), size=seq_len, replace=seq_len > 10)
        image_ids = [int(rng.choice(image_ids_by_label[int(label)])) for label in labels]
        sims = _pairwise_image_sims(ctx.dataset, image_ids)
        sequence_seed = int(rng.integers(0, 2**31 - 1))
        for stage_k, (image_id, label) in enumerate(zip(image_ids, labels), start=1):
            rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": int(sequence_id),
                    "seq_len": int(seq_len),
                    "stage_k": int(stage_k),
                    "item_image_id": int(image_id),
                    "item_label": int(label),
                    "ordered_item_ids": ";".join(map(str, image_ids)),
                    "ordered_item_labels": ";".join(map(str, [int(v) for v in labels])),
                    "sequence_seed": int(sequence_seed),
                    "mean_pairwise_image_similarity": float(np.mean(sims)) if sims else 0.0,
                    "max_pairwise_image_similarity": float(np.max(sims)) if sims else 0.0,
                    "min_pairwise_image_similarity": float(np.min(sims)) if sims else 0.0,
                }
            )
    out = pd.DataFrame(rows, columns=SEQUENCE_TRIAL_COLUMNS)
    _save_csv(ctx, out, ctx.trial_specs_dir / "sequence_trials.csv")
    ctx.n_sequences = int(out["sequence_id"].nunique())
    ctx.completed_modules["sequence_trials"] = True
    return out


def run_sequence_bank(ctx: ExperimentContext, sequence_trials: pd.DataFrame) -> PeakAmplifiedReentryBank:
    seq_ids = sorted(sequence_trials["sequence_id"].unique())
    n_seq = len(seq_ids)
    n_units = 28 * 28
    update_count = np.zeros((n_seq, n_units), dtype=np.float32)
    last_update_position = np.zeros((n_seq, n_units), dtype=np.int16)
    time_since_last_update = np.zeros((n_seq, n_units), dtype=np.int16)
    update_exposure_by_item = np.zeros((n_seq, max(sequence_trials["seq_len"]), n_units), dtype=np.float32)
    item_activation_history = np.zeros_like(update_exposure_by_item)
    g_baseline = np.zeros((n_seq, n_units), dtype=np.float32)
    g_final = np.zeros((n_seq, n_units), dtype=np.float32)
    delta_support = np.zeros((n_seq, n_units), dtype=np.float32)
    peak_mask = np.zeros((n_seq, n_units), dtype=bool)
    nonpeak_mask = np.zeros((n_seq, n_units), dtype=bool)
    prior_updated_mask = np.zeros((n_seq, n_units), dtype=bool)
    boundaries: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    sequence_meta_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}

    for row_idx, sequence_id in _progress(enumerate(seq_ids), total=len(seq_ids), desc="fig6 sequence bank", enabled=ctx.cfg.show_progress):
        group = sequence_trials[sequence_trials["sequence_id"].eq(sequence_id)].sort_values("stage_k")
        seq_len = int(group["seq_len"].iloc[0])
        image_ids = [int(v) for v in group["item_image_id"].tolist()]
        labels = [int(v) for v in group["item_label"].tolist()]
        masks = np.stack([_foreground_mask(ctx.dataset, image_id, ctx.cfg.foreground_threshold) for image_id in image_ids], axis=0)
        exposure = masks.reshape(seq_len, -1).astype(np.float32)
        update_exposure_by_item[row_idx, :seq_len, :] = exposure
        item_activation_history[row_idx, :seq_len, :] = exposure
        update_count[row_idx] = exposure.sum(axis=0)
        for pos in range(seq_len):
            active = exposure[pos] > 0
            last_update_position[row_idx, active] = pos + 1
        time_since_last_update[row_idx] = np.where(last_update_position[row_idx] > 0, seq_len - last_update_position[row_idx], seq_len + 1)
        prior_updated_mask[row_idx] = update_count[row_idx] > 0

        baseline_map, final_map, boundary = _sequence_support_maps(ctx, image_ids, masks, update_count[row_idx], last_update_position[row_idx], seq_len, encode_cache=encode_cache)
        if boundary:
            boundaries[int(sequence_id)] = boundary
        g_baseline[row_idx] = baseline_map.reshape(-1).astype(np.float32)
        g_final[row_idx] = final_map.reshape(-1).astype(np.float32)
        delta_support[row_idx] = g_final[row_idx] - g_baseline[row_idx]
        peaks = _top_mask(delta_support[row_idx].reshape(28, 28), ctx.cfg.peak_q, positive=delta_support[row_idx].reshape(28, 28) > 0)
        peak_mask[row_idx] = peaks.reshape(-1)
        nonpeak_mask[row_idx] = _matched_nonpeak_mask(peak_mask[row_idx], prior_updated_mask[row_idx], int(ctx.cfg.network_seed) + int(sequence_id))
        sequence_meta_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(sequence_id),
                "seq_len": seq_len,
                "ordered_item_ids": ";".join(map(str, image_ids)),
                "ordered_item_labels": ";".join(map(str, labels)),
            }
        )
        for state_condition, stage_k, arrs in (
            ("S0", 0, {"G_baseline": g_baseline[row_idx]}),
            ("S_final", seq_len, {"G_final": g_final[row_idx], "delta_support": delta_support[row_idx]}),
        ):
            for key, arr in arrs.items():
                manifest_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(sequence_id),
                        "seq_len": seq_len,
                        "state_condition": state_condition,
                        "stage_k": int(stage_k),
                        "layer": PRIMARY_LAYER,
                        "state_variable": STATE_VARIABLE if key != "delta_support" else "delta_support",
                        "shape": "28x28",
                        "storage_file": "final_support_maps.npz",
                        "storage_key": f"{key}_sequence_{int(sequence_id)}",
                        "captured_after": state_condition,
                        "sample_ms": int(ctx.cfg.sample_ms),
                        "delay_ms": int(ctx.cfg.delay_ms),
                    }
                )

    _save_csv(ctx, pd.DataFrame(manifest_rows, columns=STATE_BANK_MANIFEST_COLUMNS), ctx.raw_dir / "state_bank_manifest.csv")
    np.savez_compressed(
        ctx.raw_dir / "update_history_matrix.npz",
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    np.savez_compressed(
        ctx.raw_dir / "final_support_maps.npz",
        G_baseline=g_baseline,
        G_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask.astype(np.uint8),
        nonpeak_mask=nonpeak_mask.astype(np.uint8),
        unit_ids=np.arange(n_units, dtype=np.int32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int32),
    )
    ctx.output_files["state_bank_manifest"] = "data/raw/state_bank_manifest.csv"
    ctx.output_files["update_history_matrix"] = "data/raw/update_history_matrix.npz"
    ctx.output_files["final_support_maps"] = "data/raw/final_support_maps.npz"
    ctx.completed_modules["sequence_bank"] = True
    return PeakAmplifiedReentryBank(
        sequence_trials=sequence_trials.reset_index(drop=True),
        sequence_meta=pd.DataFrame(sequence_meta_rows),
        probe_trials=pd.DataFrame(),
        matched_groups=pd.DataFrame(),
        update_count=update_count,
        last_update_position=last_update_position,
        time_since_last_update=time_since_last_update,
        update_exposure_by_item=update_exposure_by_item,
        item_activation_history=item_activation_history,
        g_baseline=g_baseline,
        g_final=g_final,
        delta_support=delta_support,
        peak_mask=peak_mask,
        nonpeak_mask=nonpeak_mask,
        prior_updated_mask=prior_updated_mask,
        boundaries=boundaries,
        reentry_metrics=pd.DataFrame(),
        downstream_metrics=pd.DataFrame(),
    )


def run_leave_one_item_out_support_bank(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> dict[int, list[dict[str, Any]]]:
    proxy_mode = _is_proxy_mode(ctx)
    rows_by_sequence: dict[int, list[dict[str, Any]]] = {}
    raw_payload: dict[str, np.ndarray] = {}
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 leave-one-out sequences", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        image_ids = [int(v) for v in str(meta.ordered_item_ids).split(";") if str(v) != ""]
        labels = [int(v) for v in str(meta.ordered_item_labels).split(";") if str(v) != ""]
        seq_len = int(meta.seq_len)
        full = bank.g_final[seq_idx].reshape(28, 28)
        baseline = bank.g_baseline[seq_idx].reshape(28, 28)
        sequence_rows: list[dict[str, Any]] = []
        for removed_idx in _progress(range(seq_len), total=seq_len, desc="fig6 leave-one-out items", enabled=ctx.cfg.show_progress):
            minus_map = _leave_one_out_support_map(ctx, image_ids, removed_idx, encode_cache=encode_cache)
            delta_minus = minus_map - baseline
            loss_map = np.maximum(full - minus_map, 0.0).astype(np.float32)
            sequence_rows.append(
                {
                    "removed_position": int(removed_idx + 1),
                    "removed_label": int(labels[removed_idx]) if removed_idx < len(labels) else -1,
                    "removed_image_id": int(image_ids[removed_idx]) if removed_idx < len(image_ids) else -1,
                    "G_minus_i": minus_map.reshape(-1).astype(np.float32),
                    "delta_minus_i": delta_minus.reshape(-1).astype(np.float32),
                    "loss_map_i": loss_map.reshape(-1).astype(np.float32),
                    "proxy_mode": bool(proxy_mode),
                }
            )
            if ctx.cfg.save_full_traces:
                raw_payload[f"sequence_{seq_id}_removed_{removed_idx + 1}_G_minus_i"] = minus_map.astype(np.float32)
                raw_payload[f"sequence_{seq_id}_removed_{removed_idx + 1}_loss_map_i"] = loss_map.astype(np.float32)
        rows_by_sequence[seq_id] = sequence_rows
    if raw_payload:
        np.savez_compressed(ctx.raw_dir / "leave_one_item_out_support_maps.npz", **raw_payload)
        ctx.output_files["leave_one_item_out_support_maps"] = "data/raw/leave_one_item_out_support_maps.npz"
    ctx.completed_modules["peak_source_attribution_replay"] = True
    return rows_by_sequence


def compute_peak_source_attribution(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank, loo_bank: dict[int, list[dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    proxy_mode = _is_proxy_mode(ctx)
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 source attribution", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        seq_len = int(meta.seq_len)
        peak = bank.peak_mask[seq_idx].reshape(-1)
        nonpeak = bank.nonpeak_mask[seq_idx].reshape(-1)
        prior = bank.prior_updated_mask[seq_idx].reshape(-1)
        seq_rows = loo_bank.get(seq_id, [])
        peak_losses = np.asarray([float(np.sum(r["loss_map_i"][peak])) for r in seq_rows], dtype=float)
        nonpeak_losses = np.asarray([float(np.sum(r["loss_map_i"][nonpeak])) for r in seq_rows], dtype=float)
        peak_total = float(np.nansum(peak_losses))
        nonpeak_total = float(np.nansum(nonpeak_losses))
        for i, replay in enumerate(seq_rows):
            loss = np.asarray(replay["loss_map_i"], dtype=float)
            peak_loss = float(peak_losses[i])
            nonpeak_loss = float(nonpeak_losses[i])
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "removed_position": int(replay["removed_position"]),
                    "removed_label": int(replay["removed_label"]),
                    "removed_image_id": int(replay["removed_image_id"]),
                    "peak_loss": peak_loss,
                    "nonpeak_loss": nonpeak_loss,
                    "prior_updated_loss": float(np.sum(loss[prior])),
                    "peak_loss_fraction": _safe_div(peak_loss, peak_total),
                    "nonpeak_loss_fraction": _safe_div(nonpeak_loss, nonpeak_total),
                    "peak_vs_nonpeak_loss_ratio": _safe_div(peak_loss, max(nonpeak_loss, 1e-12)),
                    "support_loss_total": float(np.sum(loss)),
                    "leave_one_out_mode": str(ctx.cfg.leave_one_out_mode),
                    "proxy_mode": bool(proxy_mode),
                }
            )
    df = pd.DataFrame(rows, columns=PANEL_A_SOURCE_COLUMNS)
    _save_csv(ctx, df, ctx.metrics_dir / "panel_a_peak_source_attribution.csv")
    _save_csv(ctx, df, ctx.raw_dir / "panel_a_peak_source_attribution.csv")
    if not df.empty:
        for (network_seed, seq_len, pos), part in df.groupby(["network_seed", "seq_len", "removed_position"], sort=True):
            vals = pd.to_numeric(part["peak_loss_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
            ratios = pd.to_numeric(part["peak_vs_nonpeak_loss_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "network_seed": int(network_seed),
                    "seq_len": int(seq_len),
                    "removed_position": int(pos),
                    "relative_position_from_end": int(seq_len) - int(pos),
                    "mean_peak_loss_fraction": float(np.mean(vals)) if vals.size else np.nan,
                    "sem_peak_loss_fraction": _sem(vals) if vals.size else np.nan,
                    "mean_peak_vs_nonpeak_loss_ratio": float(np.mean(ratios)) if ratios.size else np.nan,
                    "n_sequences": int(part["sequence_id"].nunique()),
                }
            )
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=PANEL_A_SOURCE_SUMMARY_COLUMNS), ctx.metrics_dir / "panel_a_peak_source_attribution_summary.csv")
    ctx.completed_modules["peak_source_attribution"] = True
    if proxy_mode:
        ctx.warnings.append("Fig.6A leave-one-out attribution used proxy support replay; use real model replay for final scientific evidence.")


def compute_peak_update_history(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    windows = tuple(int(v) for v in ctx.cfg.recent_overlap_windows)
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 update history", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        seq_len = int(meta.seq_len)
        for unit_id in _progress(range(bank.update_count.shape[1]), total=bank.update_count.shape[1], desc="fig6 update units", enabled=ctx.cfg.show_progress):
            update_count = int(bank.update_count[seq_idx, unit_id])
            last_pos = int(bank.last_update_position[seq_idx, unit_id])
            time_since = int(bank.time_since_last_update[seq_idx, unit_id])
            row = {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": seq_id,
                "seq_len": seq_len,
                "unit_id": int(unit_id),
                "is_peak": bool(bank.peak_mask[seq_idx, unit_id]),
                "is_nonpeak_control": bool(bank.nonpeak_mask[seq_idx, unit_id]),
                "update_count": update_count,
                "last_update_position": last_pos,
                "time_since_last_update": time_since,
                "is_multi_update": bool(update_count >= int(ctx.cfg.multi_update_threshold)),
                "final_support": float(bank.g_final[seq_idx, unit_id]),
                "delta_support": float(bank.delta_support[seq_idx, unit_id]),
            }
            for w in (2, 3, 4, 5):
                row[f"recent_w{w}"] = bool(time_since < w)
                row[f"is_multi_recent_w{w}"] = bool(row["is_multi_update"] and time_since < w)
            rows.append(row)
    df = pd.DataFrame(rows, columns=PANEL_B_UPDATE_HISTORY_COLUMNS)
    _save_csv(ctx, df, ctx.metrics_dir / "panel_b_peak_update_history.csv")
    _save_csv(ctx, df, ctx.raw_dir / "panel_b_peak_update_history.csv")
    groups = {
        "peak": df[df["is_peak"].astype(bool)] if not df.empty else df,
        "nonpeak_control": df[df["is_nonpeak_control"].astype(bool)] if not df.empty else df,
        "prior_updated_nonpeak": df[(~df["is_peak"].astype(bool)) & (pd.to_numeric(df["update_count"], errors="coerce") > 0)] if not df.empty else df,
    }
    for group, part in groups.items():
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "group": group,
                "mean_update_count": _mean_col(part, "update_count"),
                "P_update_ge_2": _mean_bool(part, pd.to_numeric(part.get("update_count", pd.Series(dtype=float)), errors="coerce") >= 2),
                "P_update_ge_3": _mean_bool(part, pd.to_numeric(part.get("update_count", pd.Series(dtype=float)), errors="coerce") >= 3),
                "mean_time_since_last_update": _mean_col(part, "time_since_last_update"),
                "P_recent_w2": _mean_col(part, "recent_w2"),
                "P_recent_w3": _mean_col(part, "recent_w3"),
                "P_recent_w4": _mean_col(part, "recent_w4"),
                "P_recent_w5": _mean_col(part, "recent_w5"),
                "P_multi_recent_w2": _mean_col(part, "is_multi_recent_w2"),
                "P_multi_recent_w3": _mean_col(part, "is_multi_recent_w3"),
                "P_multi_recent_w4": _mean_col(part, "is_multi_recent_w4"),
                "P_multi_recent_w5": _mean_col(part, "is_multi_recent_w5"),
                "n_units": int(len(part)),
            }
        )
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=PANEL_B_UPDATE_HISTORY_SUMMARY_COLUMNS), ctx.metrics_dir / "panel_b_peak_update_history_summary.csv")
    ctx.completed_modules["peak_update_history"] = True


def compute_peak_input_overlap_origin(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    example_payload: dict[str, np.ndarray | str] = {}
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 overlap origin", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        seq_len = int(meta.seq_len)
        item_maps = bank.item_activation_history[seq_idx, :seq_len, :] > 0
        peak = bank.peak_mask[seq_idx].reshape(-1)
        delta = bank.delta_support[seq_idx].reshape(-1)
        maps: list[tuple[str, str, int, int, np.ndarray]] = [("all", "all", 1, seq_len, item_maps.sum(axis=0))]
        for k in tuple(int(v) for v in ctx.cfg.recent_overlap_windows):
            start = max(0, seq_len - k)
            recent_map = item_maps[start:seq_len, :].sum(axis=0)
            old_map = item_maps[:start, :].sum(axis=0) if start > 0 else np.zeros_like(recent_map)
            maps.append((f"recent_{k}", "recent", start + 1, seq_len, recent_map))
            maps.append((f"old_{k}", "old", 1, start, old_map))
        for window_name, overlap_type, start_pos, end_pos, overlap in maps:
            high, fallback = _high_overlap_mask(overlap, int(np.sum(peak)))
            inter = peak & high
            n_overlap = int(np.sum(overlap >= 2))
            n_peak = int(np.sum(peak))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "overlap_window": window_name,
                    "window_start_position": int(start_pos),
                    "window_end_position": int(end_pos),
                    "n_items_in_window": int(max(0, end_pos - start_pos + 1)),
                    "overlap_type": overlap_type,
                    "n_overlap_pixels": n_overlap,
                    "n_peak_pixels": n_peak,
                    "dice_peak_overlap": _dice(peak, high),
                    "jaccard_peak_overlap": _jaccard(peak, high),
                    "peak_coverage": _safe_div(float(np.sum(inter)), float(n_peak)),
                    "overlap_precision": _safe_div(float(np.sum(inter)), float(max(1, np.sum(high)))),
                    "cosine_delta_support_overlap_count": _plain_cosine(delta, overlap),
                    "spearman_delta_support_overlap_count": _spearman(delta, overlap),
                    "fallback_used": bool(fallback),
                }
            )
        if not example_payload:
            recent2 = item_maps[max(0, seq_len - 2) : seq_len, :].sum(axis=0).reshape(28, 28)
            recent3 = item_maps[max(0, seq_len - 3) : seq_len, :].sum(axis=0).reshape(28, 28)
            old = item_maps[: max(0, seq_len - 2), :].sum(axis=0).reshape(28, 28) if seq_len > 2 else np.zeros((28, 28), dtype=np.float32)
            example_payload = {
                "peak_mask": peak.reshape(28, 28).astype(np.uint8),
                "delta_support_map": delta.reshape(28, 28).astype(np.float32),
                "all_input_overlap_count": item_maps.sum(axis=0).reshape(28, 28).astype(np.float32),
                "recent_2_overlap_count": recent2.astype(np.float32),
                "recent_3_overlap_count": recent3.astype(np.float32),
                "old_overlap_count": old.astype(np.float32),
                "high_overlap_mask_recent_2": _high_overlap_mask(recent2.reshape(-1), int(np.sum(peak)))[0].reshape(28, 28).astype(np.uint8),
                "high_overlap_mask_recent_3": _high_overlap_mask(recent3.reshape(-1), int(np.sum(peak)))[0].reshape(28, 28).astype(np.uint8),
                "selected_sequence_metadata": json.dumps(_json_safe(meta._asdict()), sort_keys=True),
            }
    df = pd.DataFrame(rows, columns=PANEL_C_ORIGIN_COLUMNS)
    _save_csv(ctx, df, ctx.metrics_dir / "panel_c_peak_input_overlap_similarity.csv")
    summary_rows = []
    if not df.empty:
        for (network_seed, window), part in df.groupby(["network_seed", "overlap_window"], sort=False):
            dice = pd.to_numeric(part["dice_peak_overlap"], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "network_seed": int(network_seed),
                    "overlap_window": str(window),
                    "mean_dice": float(np.mean(dice)) if dice.size else np.nan,
                    "sem_dice": _sem(dice) if dice.size else np.nan,
                    "mean_peak_coverage": _mean_col(part, "peak_coverage"),
                    "mean_cosine": _mean_col(part, "cosine_delta_support_overlap_count"),
                    "n_sequences": int(part["sequence_id"].nunique()),
                }
            )
    summary = pd.DataFrame(summary_rows, columns=PANEL_C_ORIGIN_SUMMARY_COLUMNS)
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_c_peak_input_overlap_similarity_summary.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_c_peak_input_overlap_summary.csv")
    if example_payload:
        np.savez_compressed(ctx.raw_dir / "panel_c_peak_input_overlap_example.npz", **example_payload)
        ctx.output_files["panel_c_peak_input_overlap_example"] = "data/raw/panel_c_peak_input_overlap_example.npz"
    ctx.completed_modules["peak_input_overlap_origin"] = True


def build_later_probe_peak_overlap_trials(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    probe_trials = build_probe_candidate_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    for r in _progress(probe_trials.itertuples(index=False), total=len(probe_trials), desc="fig6 probe definitions", enabled=ctx.cfg.show_progress):
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "probe_image_id": int(r.probe_image_id),
                "probe_label": int(r.probe_label),
                "probe_source": str(r.probe_source),
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_fraction": float(r.peak_overlap_fraction),
                "nonpeak_overlap_fraction": float(r.nonpeak_overlap_fraction),
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "peak_support_sum": float(r.peak_support_sum),
                "nonpeak_support_sum": float(r.nonpeak_support_sum),
                "class_pair": str(r.class_pair),
                "candidate_seed": int(r.candidate_seed),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_D_TRIAL_DEFINITION_COLUMNS), ctx.metrics_dir / "panel_d_later_probe_peak_overlap_definitions.csv")
    _save_panel_d_example(ctx, bank, probe_trials)
    bank.probe_trials = probe_trials
    ctx.n_probe_candidates = int(len(probe_trials))
    ctx.completed_modules["later_probe_peak_overlap_trials"] = True


def define_final_peaks_and_update_groups(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seq_lookup = {int(r.sequence_id): idx for idx, r in enumerate(bank.sequence_meta.itertuples(index=False))}
    for meta in bank.sequence_meta.itertuples(index=False):
        seq_id = int(meta.sequence_id)
        idx = seq_lookup[seq_id]
        seq_len = int(meta.seq_len)
        for unit_id in range(bank.update_count.shape[1]):
            update_count = int(bank.update_count[idx, unit_id])
            last_pos = int(bank.last_update_position[idx, unit_id])
            recent = bool(last_pos > 0 and seq_len - last_pos < int(ctx.cfg.recent_window))
            recency_group = "recent" if recent else "old"
            multiplicity_group = "multi" if update_count >= int(ctx.cfg.multi_update_threshold) else "single"
            group = f"{multiplicity_group}_{recency_group}"
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "layer": PRIMARY_LAYER,
                    "state_variable": STATE_VARIABLE,
                    "unit_id": int(unit_id),
                    "update_count": int(update_count),
                    "last_update_position": int(last_pos),
                    "time_since_last_update": int(bank.time_since_last_update[idx, unit_id]),
                    "recency_group": recency_group,
                    "multiplicity_group": multiplicity_group,
                    "update_history_group": group,
                    "is_peak": bool(bank.peak_mask[idx, unit_id]),
                    "final_support": float(bank.g_final[idx, unit_id]),
                    "baseline_support": float(bank.g_baseline[idx, unit_id]),
                    "delta_support": float(bank.delta_support[idx, unit_id]),
                }
            )
    df = pd.DataFrame(rows, columns=PANEL_A_UNIT_COLUMNS)
    overall_peak = float(df["is_peak"].mean()) if not df.empty else float("nan")
    for group in UPDATE_GROUPS:
        part = df[df["update_history_group"].eq(group)]
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "update_history_group": group,
                "P_peak": float(part["is_peak"].mean()) if len(part) else float("nan"),
                "mean_final_support": float(part["final_support"].mean()) if len(part) else float("nan"),
                "mean_delta_support": float(part["delta_support"].mean()) if len(part) else float("nan"),
                "peak_enrichment": _safe_div(float(part["is_peak"].mean()) if len(part) else float("nan"), overall_peak),
                "n_units": int(len(part)),
            }
        )
    _save_csv(ctx, df, ctx.metrics_dir / "supp_legacy_panel_a_multi_recent_peak_enrichment.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=PANEL_A_SUMMARY_COLUMNS), ctx.metrics_dir / "supp_legacy_panel_a_peak_enrichment_summary.csv")
    ctx.completed_modules["peak_enrichment"] = True


def fit_update_recency_support_models(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    unit_df = pd.read_csv(ctx.metrics_dir / "supp_legacy_panel_a_multi_recent_peak_enrichment.csv")
    metric_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for sequence_id, part in unit_df.groupby("sequence_id", sort=True):
        y_delta = pd.to_numeric(part["delta_support"], errors="coerce").to_numpy(dtype=float)
        y_final = pd.to_numeric(part["final_support"], errors="coerce").to_numpy(dtype=float)
        features = {
            "baseline_support": pd.to_numeric(part["baseline_support"], errors="coerce").to_numpy(dtype=float),
            "update_count": pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float),
            "recency": -pd.to_numeric(part["time_since_last_update"], errors="coerce").to_numpy(dtype=float),
            "overlap": pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float) > 0,
        }
        feature_matrix = {
            "baseline_only": ["baseline_support"],
            "update_only": ["update_count"],
            "recency_only": ["recency"],
            "overlap_only": ["overlap"],
            "update_plus_recency": ["update_count", "recency"],
            "update_times_recency": ["update_count", "recency", "update_x_recency"],
        }
        features["update_x_recency"] = features["update_count"] * features["recency"]
        stats_by_model: dict[str, dict[str, float]] = {}
        for target_name, y in (("delta_support", y_delta), ("final_support", y_final)):
            for model_name, cols in feature_matrix.items():
                x = np.column_stack([np.asarray(features[col], dtype=float) for col in cols])
                fit = _fit_ols(x, y)
                cv_r2 = _cv_r2(x, y, n_folds=5)
                stats_by_model[model_name] = {"r2": fit["r2"], "cv_r2": cv_r2}
                metric_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(sequence_id),
                        "layer": PRIMARY_LAYER,
                        "state_variable": STATE_VARIABLE,
                        "target": target_name,
                        "model_name": model_name,
                        "r2": float(fit["r2"]),
                        "cv_r2": float(cv_r2),
                        "auc_if_binary": float("nan"),
                        "delta_r2_vs_overlap_only": float(fit["r2"] - stats_by_model.get("overlap_only", {}).get("r2", np.nan)),
                        "delta_r2_vs_update_only": float(fit["r2"] - stats_by_model.get("update_only", {}).get("r2", np.nan)),
                        "delta_r2_vs_recency_only": float(fit["r2"] - stats_by_model.get("recency_only", {}).get("r2", np.nan)),
                        "n_units": int(len(part)),
                    }
                )
                for coef_name, coef_value, se, p in zip(["intercept"] + cols, fit["beta"], fit["se"], fit["p"]):
                    coef_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "model_name": model_name,
                            "coefficient_name": coef_name,
                            "coefficient_value": float(coef_value),
                            "standardized_coefficient": float(_standardized_coef(coef_value, x, y, coef_name, cols)),
                            "p_value": float(p),
                            "notes": f"target={target_name}; sequence_id={int(sequence_id)}; ordinary least squares",
                        }
                    )
    metrics = pd.DataFrame(metric_rows, columns=PANEL_B_METRIC_COLUMNS)
    coefs = pd.DataFrame(coef_rows, columns=PANEL_B_COEF_COLUMNS)
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_legacy_panel_b_update_recency_model_metrics.csv")
    _save_csv(ctx, coefs, ctx.metrics_dir / "supp_legacy_panel_b_update_recency_model_coefficients.csv")
    _save_csv(ctx, coefs, ctx.metrics_dir / "supp_update_recency_model_coefficients.csv")
    ctx.completed_modules["update_recency_model"] = True


def compute_peak_weighted_overlap_definitions(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    probe_trials = build_probe_candidate_trials(ctx, bank)
    rows: list[dict[str, Any]] = []
    for r in probe_trials.itertuples(index=False):
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "probe_label": int(r.probe_label),
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_fraction": float(r.peak_overlap_fraction),
                "nonpeak_overlap_fraction": float(r.nonpeak_overlap_fraction),
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "peak_support_sum": float(r.peak_support_sum),
                "nonpeak_support_sum": float(r.nonpeak_support_sum),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_C_COLUMNS), ctx.metrics_dir / "supp_legacy_panel_c_peak_weighted_overlap_definitions.csv")
    _save_panel_c_example(ctx, bank, probe_trials)
    bank.probe_trials = probe_trials
    ctx.n_probe_candidates = int(len(probe_trials))
    ctx.completed_modules["peak_weighted_overlap"] = True


def build_probe_candidate_trials(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> pd.DataFrame:
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 6006)
    image_ids_by_label = {label: np.asarray(ids, dtype=np.int64) for label, ids in ctx.class_index.items()}
    rows: list[dict[str, Any]] = []
    for seq_index, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 probe candidates", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        sequence_labels = [int(v) for v in str(meta.ordered_item_labels).split(";") if str(v) != ""]
        sequence_ids = [int(v) for v in str(meta.ordered_item_ids).split(";") if str(v) != ""]
        prior_updated = bank.prior_updated_mask[seq_index].reshape(28, 28)
        peak = bank.peak_mask[seq_index].reshape(28, 28)
        nonpeak = bank.nonpeak_mask[seq_index].reshape(28, 28)
        support = bank.g_final[seq_index].reshape(28, 28)
        for local_probe in _progress(range(int(ctx.cfg.num_probe_candidates_per_sequence)), total=int(ctx.cfg.num_probe_candidates_per_sequence), desc="fig6 probe per sequence", enabled=ctx.cfg.show_progress):
            if local_probe % 3 == 0:
                label = int(rng.choice(sequence_labels))
                source = "sequence_label"
            else:
                label = int(rng.integers(0, 10))
                source = "candidate_pool"
            probe_image_id = int(rng.choice(image_ids_by_label[label]))
            probe = _image_array(ctx.dataset, probe_image_id)
            probe_mask = probe > float(ctx.cfg.foreground_threshold)
            route_mask = probe_mask & prior_updated
            peak_overlap_mask = route_mask & peak
            nonpeak_overlap_mask = route_mask & nonpeak
            raw_overlap = _safe_div(float(route_mask.sum()), float(max(1, probe_mask.sum())))
            peak_support_sum = float((support * peak_overlap_mask).sum())
            nonpeak_support_sum = float((support * nonpeak_overlap_mask).sum())
            peak_weighted_overlap = _safe_div(peak_support_sum, float(max(1, probe_mask.sum())))
            sim = float(np.mean([_centered_cosine(probe.reshape(-1), _image_array(ctx.dataset, sid).reshape(-1)) for sid in sequence_ids]))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "probe_id": int(seq_id * 1000 + local_probe),
                    "probe_image_id": int(probe_image_id),
                    "probe_label": int(label),
                    "probe_source": source,
                    "raw_overlap": float(raw_overlap),
                    "peak_weighted_overlap": float(peak_weighted_overlap),
                    "peak_overlap_fraction": _safe_div(float(peak_overlap_mask.sum()), float(max(1, route_mask.sum()))),
                    "nonpeak_overlap_fraction": _safe_div(float(nonpeak_overlap_mask.sum()), float(max(1, route_mask.sum()))),
                    "visual_similarity": float(sim),
                    "input_energy": float(probe.sum()),
                    "class_pair": f"{sequence_labels[-1] if sequence_labels else -1}->{label}",
                    "candidate_seed": int(rng.integers(0, 2**31 - 1)),
                    "peak_support_sum": peak_support_sum,
                    "nonpeak_support_sum": nonpeak_support_sum,
                }
            )
    df = pd.DataFrame(rows, columns=PROBE_TRIAL_COLUMNS)
    groups = _matched_raw_overlap_groups(ctx, df)
    _save_csv(ctx, df, ctx.trial_specs_dir / "probe_candidate_trials.csv")
    _save_csv(ctx, groups, ctx.trial_specs_dir / "matched_raw_overlap_groups.csv")
    bank.matched_groups = groups
    ctx.n_matched_groups = int(len(groups))
    return df


def run_probe_candidate_reentry_rollouts(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    ctx.warnings.append("Legacy run_probe_candidate_reentry_rollouts redirected to real restored-state rollout; formula proxy is not a main Fig.6 result.")
    run_real_probe_reentry_rollouts(ctx, bank)


def run_real_probe_reentry_rollouts(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    trace_payload: dict[str, np.ndarray] = {}
    vector_payload: dict[str, np.ndarray] = {}
    matched_lookup = _matched_lookup(bank.matched_groups)
    proxy_mode = _is_proxy_mode(ctx)
    encode_cache: dict[tuple[Any, ...], Any] = {}
    if proxy_mode:
        ctx.warnings.append("Fig.6D/E are proxy-mode outputs and must not be used as final scientific evidence.")
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 real probes", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        matched_group_id, peak_group = matched_lookup.get(int(r.probe_id), ("", "unmatched"))
        if proxy_mode:
            final_trace, s0_trace, final_pred, s0_pred, final_fire, s0_fire, final_vector, s0_vector = _proxy_probe_rollout_pair(ctx, r)
        else:
            boundary = bank.boundaries.get(int(r.sequence_id))
            try:
                probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
                final_trace, final_pred, final_fire, final_vector = _run_real_probe_from_condition(ctx, int(r.probe_image_id), boundary, "S_final", probe_spikes=probe_spikes)
                s0_trace, s0_pred, s0_fire, s0_vector = _run_real_probe_from_condition(ctx, int(r.probe_image_id), None, "S0", probe_spikes=probe_spikes)
            except Exception as exc:
                ctx.warnings.append(f"Real Fig.6 probe rollout failed for sequence={int(r.sequence_id)} probe={int(r.probe_id)}; proxy fallback used: {exc}")
                proxy_mode = True
                final_trace, s0_trace, final_pred, s0_pred, final_fire, s0_fire, final_vector, s0_vector = _proxy_probe_rollout_pair(ctx, r)
        l3_delta = float(np.linalg.norm(final_trace.reshape(-1) - s0_trace.reshape(-1)))
        evidence_final = _label_evidence(final_vector, int(r.probe_label))
        evidence_s0 = _label_evidence(s0_vector, int(r.probe_label))
        decision_deflection = float(evidence_final - evidence_s0)
        dynamic_recovery = _plain_cosine(final_trace.reshape(-1), s0_trace.reshape(-1))
        first_delta = _fire_delta(final_fire, s0_fire)
        early_gain = _early_spike_count(final_trace) - _early_spike_count(s0_trace)
        p_advance, p_recruit, spike_advance = _spike_timing_metrics(final_trace, s0_trace)
        displacement = float(np.linalg.norm(final_vector.reshape(-1) - s0_vector.reshape(-1)))
        key = f"sequence_{int(r.sequence_id)}_probe_{int(r.probe_id)}"
        if ctx.cfg.save_l3_trace:
            trace_payload[f"{key}_Sfinal_l3_trace"] = final_trace.astype(np.float32)
            trace_payload[f"{key}_S0_l3_trace"] = s0_trace.astype(np.float32)
        vector_payload[f"{key}_Sfinal_readout_vector"] = final_vector.astype(np.float32)
        vector_payload[f"{key}_S0_readout_vector"] = s0_vector.astype(np.float32)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "matched_group_id": matched_group_id,
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_group": peak_group,
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "prediction_Sfinal": int(final_pred),
                "prediction_S0": int(s0_pred),
                "correct_Sfinal": bool(int(final_pred) == int(r.probe_label)),
                "correct_S0": bool(int(s0_pred) == int(r.probe_label)),
                "first_fire_time_Sfinal": int(final_fire),
                "first_fire_time_S0": int(s0_fire),
                "first_fire_time_delta": first_delta,
                "l3_trace_delta_norm": l3_delta,
                "reentry_strength_real": l3_delta,
                "dynamic_like_recovery_real": dynamic_recovery,
                "decision_deflection_score_real": decision_deflection,
                "proxy_mode": bool(proxy_mode),
            }
        )
        downstream_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "matched_group_id": matched_group_id,
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "peak_overlap_group": peak_group,
                "visual_similarity": float(r.visual_similarity),
                "input_energy": float(r.input_energy),
                "early_recruitment_gain_real": float(early_gain),
                "P_advance_real": p_advance,
                "P_recruit_real": p_recruit,
                "spike_advance_real": spike_advance,
                "response_pattern_displacement_real": displacement,
                "decision_deflection_score_real": decision_deflection,
                "partial_cue_completion_gain_real": float("nan"),
                "proxy_mode": bool(proxy_mode),
            }
        )
    bank.reentry_metrics = pd.DataFrame(rows, columns=PANEL_D_REAL_METRIC_COLUMNS)
    bank.downstream_metrics = pd.DataFrame(downstream_rows, columns=PANEL_E_REAL_METRIC_COLUMNS)
    np.savez_compressed(ctx.raw_dir / "reentry_trace_arrays_l3.npz", **trace_payload)
    np.savez_compressed(ctx.raw_dir / "downstream_dynamics_vectors.npz", **vector_payload)
    ctx.output_files["reentry_trace_arrays_l3"] = "data/raw/reentry_trace_arrays_l3.npz"
    ctx.output_files["downstream_dynamics_vectors"] = "data/raw/downstream_dynamics_vectors.npz"
    ctx.completed_modules["real_reentry_rollouts"] = True


def compute_real_peak_weighted_reentry_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.reentry_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "panel_d_real_reentry_metrics.csv")
    matched = df[df["matched_group_id"].astype(str).str.len() > 0].copy()
    _save_csv(ctx, matched, ctx.metrics_dir / "panel_d_raw_overlap_matched_peak_reentry.csv")
    reg = _regression_rows(ctx, df, metrics=("reentry_strength_real", "l3_trace_delta_norm", "dynamic_like_recovery_real", "decision_deflection_score_real"), n_name="n_trials")
    reg["proxy_mode"] = bool(_df_all_proxy(df))
    _save_csv(ctx, reg, ctx.metrics_dir / "panel_d_peak_overlap_reentry_regression.csv")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_raw_vs_peak_weighted_overlap_regression.csv")
    _write_standardized_panel_d_outputs(ctx, df)
    ctx.completed_modules["real_reentry_metrics"] = True


def compute_real_peak_overlap_downstream_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.downstream_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "panel_e_real_downstream_metrics.csv")
    reg = _regression_rows(
        ctx,
        df,
        metrics=(
            "early_recruitment_gain_real",
            "P_advance_real",
            "P_recruit_real",
            "spike_advance_real",
            "response_pattern_displacement_real",
            "decision_deflection_score_real",
            "partial_cue_completion_gain_real",
        ),
        n_name="n_trials",
    )
    reg["proxy_mode"] = bool(_df_all_proxy(df))
    _save_csv(ctx, reg, ctx.metrics_dir / "panel_e_peak_overlap_downstream_regression.csv")
    _write_standardized_panel_e_outputs(ctx, df)
    ctx.completed_modules["real_downstream_metrics"] = True


def compute_peak_weighted_reentry_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.reentry_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "supp_legacy_panel_d_peak_weighted_reentry_metrics.csv")
    matched = df[df["matched_group_id"].astype(str).str.len() > 0].copy()
    _save_csv(ctx, matched, ctx.metrics_dir / "supp_legacy_panel_d_matched_raw_overlap_comparison.csv")
    metrics = tuple(
        metric
        for metric in ("reentry_strength", "DPI_L3", "dynamic_like_recovery", "decision_deflection_score")
        if metric in df.columns
    )
    if not metrics:
        metrics = ("reentry_strength_real", "l3_trace_delta_norm", "dynamic_like_recovery_real", "decision_deflection_score_real")
    reg = _regression_rows(ctx, df, metrics=metrics, n_name="n_trials")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_legacy_panel_d_peak_weighted_overlap_regression.csv")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_raw_vs_peak_weighted_overlap_regression.csv")
    ctx.completed_modules["reentry_prediction"] = True


def compute_peak_weighted_downstream_metrics(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    df = bank.downstream_metrics.copy()
    _save_csv(ctx, df, ctx.metrics_dir / "supp_legacy_panel_e_peak_weighted_downstream_metrics.csv")
    metrics = tuple(metric for metric in DOWNSTREAM_METRICS if metric in df.columns)
    if not metrics:
        metrics = (
            "early_recruitment_gain_real",
            "P_advance_real",
            "P_recruit_real",
            "spike_advance_real",
            "response_pattern_displacement_real",
            "decision_deflection_score_real",
            "partial_cue_completion_gain_real",
        )
    reg = _regression_rows(ctx, df, metrics=metrics, n_name="n_trials")
    _save_csv(ctx, reg, ctx.metrics_dir / "supp_legacy_panel_e_downstream_regression.csv")
    ctx.completed_modules["downstream_prediction"] = True


def compute_supplement_outputs(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    unit_df = pd.read_csv(ctx.metrics_dir / "panel_b_peak_update_history.csv")
    source_df = pd.read_csv(ctx.metrics_dir / "panel_a_peak_source_attribution.csv")
    overlap_df = pd.read_csv(ctx.metrics_dir / "panel_c_peak_input_overlap_similarity.csv")
    compute_supp_update_recency_support_model(ctx, unit_df)
    _save_csv(ctx, unit_df[["network_seed", "sequence_id", "unit_id", "update_count"]].assign(exposure_threshold=ctx.cfg.foreground_threshold, notes="Unit update count from thresholded item foreground exposure."), ctx.metrics_dir / "supp_update_count_definition.csv")
    _save_csv(ctx, unit_df[["network_seed", "sequence_id", "unit_id", "last_update_position", "time_since_last_update"]].assign(recent_window=int(ctx.cfg.recent_window), notes="Recency is measured backward from the final sequence position."), ctx.metrics_dir / "supp_recency_definition.csv")
    _save_csv(ctx, _leave_one_out_timing_controls(ctx, source_df), ctx.metrics_dir / "supp_leave_one_out_timing_controls.csv")
    _save_csv(ctx, _peak_source_old_vs_recent(ctx, source_df), ctx.metrics_dir / "supp_peak_source_attribution_old_vs_recent.csv")
    _save_csv(ctx, _recent_overlap_window_robustness(ctx, overlap_df), ctx.metrics_dir / "supp_recent_overlap_window_robustness.csv")
    _save_csv(ctx, _random_window_overlap_controls(ctx, bank), ctx.metrics_dir / "supp_peak_overlap_origin_random_window_controls.csv")
    _save_csv(ctx, _matched_peak_comparison(ctx, bank.reentry_metrics), ctx.metrics_dir / "supp_matched_raw_overlap_peak_comparison.csv")
    _save_csv(ctx, _visual_energy_controls(ctx, bank.reentry_metrics, bank.downstream_metrics), ctx.metrics_dir / "supp_visual_energy_classpair_controls.csv")
    _save_csv(ctx, _alternative_peak_definitions(ctx, bank), ctx.metrics_dir / "supp_alternative_peak_definitions.csv")
    _save_csv(ctx, _global_support_controls(ctx, bank.reentry_metrics), ctx.metrics_dir / "supp_global_support_spike_count_controls.csv")
    _save_csv(ctx, _real_reentry_control_s0_static(ctx, bank.reentry_metrics), ctx.metrics_dir / "supp_real_reentry_control_S0_static.csv")
    _save_csv(ctx, _real_downstream_metric_definitions(ctx), ctx.metrics_dir / "supp_real_downstream_metric_definitions.csv")
    _save_csv(ctx, _trial_condition_audit(ctx), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.completed_modules["supplement"] = True


def write_fig6_supplement_aliases(ctx: ExperimentContext) -> None:
    alt_df = _read_csv_if_exists(ctx.metrics_dir / "supp_alternative_peak_definitions.csv")
    if alt_df is not None:
        _save_csv(ctx, _s11_alternative_peak_definitions(ctx, alt_df), ctx.metrics_dir / "supp_s11_alternative_peak_definitions.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 alternative peak-definition source missing: supp_alternative_peak_definitions.csv")

    unit_df = _read_csv_if_exists(ctx.metrics_dir / "panel_b_peak_update_history.csv")
    if unit_df is not None:
        _save_csv(ctx, _s11_peak_update_group_enrichment(ctx, unit_df), ctx.metrics_dir / "supp_s11_peak_update_group_enrichment.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 update-group enrichment source missing: panel_b_peak_update_history.csv")

    model_df = _read_csv_if_exists(ctx.metrics_dir / "supp_update_recency_support_model_metrics.csv")
    if model_df is not None:
        _save_csv(ctx, _s11_update_recency_model_comparison(ctx, model_df), ctx.metrics_dir / "supp_s11_update_recency_model_comparison.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 update-recency model source missing: supp_update_recency_support_model_metrics.csv")

    source_df = _read_csv_if_exists(ctx.metrics_dir / "panel_a_peak_source_attribution.csv")
    if source_df is not None:
        _save_csv(ctx, _s11_leave_one_out_source_details(ctx, source_df), ctx.metrics_dir / "supp_s11_leave_one_out_source_details.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 leave-one-out detail source missing: panel_a_peak_source_attribution.csv")

    overlap_df = _read_csv_if_exists(ctx.metrics_dir / "panel_c_peak_input_overlap_similarity.csv")
    if overlap_df is not None:
        _save_csv(ctx, _s11_recent_overlap_window_robustness(ctx, overlap_df), ctx.metrics_dir / "supp_s11_recent_overlap_window_robustness.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 overlap-window robustness source missing: panel_c_peak_input_overlap_similarity.csv")

    d_metrics = _read_csv_if_exists(ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv")
    if d_metrics is not None:
        _save_csv(ctx, _panel_d_matched_contrast(ctx, d_metrics), ctx.metrics_dir / "supp_s12_raw_overlap_matched_peak_overlap_contrast.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S12 matched contrast source missing: panel_d_peak_weighted_reentry_metrics.csv")

    d_reg = _read_csv_if_exists(ctx.metrics_dir / "panel_d_peak_weighted_reentry_regression.csv")
    e_summary = _read_csv_if_exists(ctx.metrics_dir / "panel_e_peak_weighted_downstream_summary.csv")
    controls = _s12_peak_weighted_regression_controls(ctx, d_reg, e_summary)
    if controls is not None:
        _save_csv(ctx, controls, ctx.metrics_dir / "supp_s12_peak_weighted_regression_controls.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S12 regression-control sources missing: panel_d_peak_weighted_reentry_regression.csv and panel_e_peak_weighted_downstream_summary.csv")

    _copy_csv_if_exists(ctx.metrics_dir / "panel_e_downstream_metric_breakdown.csv", ctx.metrics_dir / "supp_s12_downstream_metric_breakdown.csv", ctx)
    e_metrics = _read_csv_if_exists(ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv")
    if d_metrics is not None:
        _save_csv(ctx, _s11_visual_energy_classpair_controls(ctx, d_metrics), ctx.metrics_dir / "supp_s11_visual_energy_classpair_controls.csv")
    elif ctx.cfg.run_supplement:
        ctx.warnings.append("S11 visual-energy/class-pair control source missing: panel_d_peak_weighted_reentry_metrics.csv")
    if e_metrics is not None:
        _save_csv(ctx, _s12_global_support_controls(ctx, e_metrics), ctx.metrics_dir / "supp_s12_global_support_spike_count_controls.csv")
    elif ctx.cfg.run_supplement and not (ctx.metrics_dir / "supp_s12_global_support_spike_count_controls.csv").exists():
        ctx.warnings.append("S12 global-support control source missing: panel_e_peak_weighted_downstream_metrics.csv")

    if ctx.cfg.run_peak_perturbation:
        _write_standardized_peak_perturbation_outputs(ctx)

    audit = _real_rollout_scientific_use_audit(ctx)
    _save_csv(ctx, audit, ctx.metrics_dir / "panel_de_real_rollout_scientific_use_audit.csv")
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_s12_real_rollout_scientific_use_audit.csv")


def _write_standardized_panel_d_outputs(ctx: ExperimentContext, df: pd.DataFrame) -> None:
    metrics = _standardize_panel_d_metrics(ctx, df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv")
    _save_csv(ctx, _panel_d_summary(ctx, metrics), ctx.metrics_dir / "panel_d_peak_weighted_reentry_summary.csv")
    _save_csv(ctx, _regression_long_table(ctx, metrics, ["reentry_strength", "dynamic_like_reentry", "decision_deflection_score"]), ctx.metrics_dir / "panel_d_peak_weighted_reentry_regression.csv")
    _save_csv(ctx, _panel_d_matched_contrast(ctx, metrics), ctx.metrics_dir / "panel_d_peak_overlap_matched_contrast.csv")


def _write_standardized_panel_e_outputs(ctx: ExperimentContext, df: pd.DataFrame) -> None:
    metrics = _standardize_panel_e_metrics(ctx, df)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv")
    _save_csv(ctx, _panel_e_summary(ctx, metrics), ctx.metrics_dir / "panel_e_peak_weighted_downstream_summary.csv")
    _save_csv(ctx, _panel_e_breakdown(ctx, metrics), ctx.metrics_dir / "panel_e_downstream_metric_breakdown.csv")


def _standardize_panel_d_metrics(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "sequence_id",
        "probe_id",
        "target_label",
        "raw_overlap",
        "peak_weighted_overlap",
        "visual_similarity",
        "input_energy",
        "global_support",
        "nonpeak_support",
        "reentry_strength",
        "dynamic_like_reentry",
        "decision_deflection_score",
        "proxy_mode",
        "real_rollout",
        "final_scientific_use",
        "n_units",
        "n_probe_trials",
        "matched_set_id",
        "peak_overlap_group",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    trial = _read_csv_if_exists(ctx.metrics_dir / "panel_d_later_probe_peak_overlap_definitions.csv")
    if trial is not None and not trial.empty:
        join_cols = [
            c
            for c in ("sequence_id", "probe_id", "probe_label", "peak_support_sum", "nonpeak_support_sum", "class_pair")
            if c in trial.columns
        ]
        out = out.merge(trial[join_cols].drop_duplicates(["sequence_id", "probe_id"]), on=["sequence_id", "probe_id"], how="left")
    out["target_label"] = pd.to_numeric(out.get("probe_label", np.nan), errors="coerce")
    out["global_support"] = pd.to_numeric(out.get("peak_support_sum", np.nan), errors="coerce") + pd.to_numeric(out.get("nonpeak_support_sum", np.nan), errors="coerce")
    out["nonpeak_support"] = pd.to_numeric(out.get("nonpeak_support_sum", np.nan), errors="coerce")
    out["reentry_strength"] = pd.to_numeric(out.get("reentry_strength_real", out.get("reentry_strength", np.nan)), errors="coerce")
    out["dynamic_like_reentry"] = pd.to_numeric(out.get("dynamic_like_recovery_real", out.get("dynamic_like_recovery", np.nan)), errors="coerce")
    out["decision_deflection_score"] = pd.to_numeric(out.get("decision_deflection_score_real", out.get("decision_deflection_score", np.nan)), errors="coerce")
    out["proxy_mode"] = _bool_col(out, "proxy_mode", default=_is_proxy_mode(ctx))
    out["real_rollout"] = ~out["proxy_mode"].astype(bool)
    out["final_scientific_use"] = out["real_rollout"].astype(bool)
    out["n_units"] = 28 * 28
    out["n_probe_trials"] = out.groupby("sequence_id")["probe_id"].transform("count") if "sequence_id" in out.columns else len(out)
    out["matched_set_id"] = out.get("matched_group_id", "")
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols]


def _standardize_panel_e_metrics(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "sequence_id",
        "probe_id",
        "raw_overlap",
        "peak_weighted_overlap",
        "visual_similarity",
        "input_energy",
        "global_support",
        "nonpeak_support",
        "total_spike_count",
        "downstream_metric",
        "metric_value",
        "proxy_mode",
        "real_rollout",
        "final_scientific_use",
        "matched_set_id",
        "peak_overlap_group",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    base = df.copy()
    trial = _read_csv_if_exists(ctx.metrics_dir / "panel_d_later_probe_peak_overlap_definitions.csv")
    if trial is not None and not trial.empty:
        join_cols = [c for c in ("sequence_id", "probe_id", "peak_support_sum", "nonpeak_support_sum") if c in trial.columns]
        base = base.merge(trial[join_cols].drop_duplicates(["sequence_id", "probe_id"]), on=["sequence_id", "probe_id"], how="left")
    base["global_support"] = pd.to_numeric(base.get("peak_support_sum", np.nan), errors="coerce") + pd.to_numeric(base.get("nonpeak_support_sum", np.nan), errors="coerce")
    base["nonpeak_support"] = pd.to_numeric(base.get("nonpeak_support_sum", np.nan), errors="coerce")
    base["total_spike_count"] = pd.to_numeric(base.get("total_spike_count", base.get("P_recruit_real", np.nan)), errors="coerce")
    base["proxy_mode"] = _bool_col(base, "proxy_mode", default=_is_proxy_mode(ctx))
    base["real_rollout"] = ~base["proxy_mode"].astype(bool)
    base["final_scientific_use"] = base["real_rollout"].astype(bool)
    base["matched_set_id"] = base.get("matched_group_id", "")
    rows = []
    for metric in DOWNSTREAM_METRICS:
        source_col = f"{metric}_real" if f"{metric}_real" in base.columns else metric
        if source_col not in base.columns:
            continue
        part = base.copy()
        part["downstream_metric"] = metric
        part["metric_value"] = pd.to_numeric(part[source_col], errors="coerce")
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols]


def _panel_d_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "metric",
        "beta_peak_weighted_overlap",
        "beta_raw_overlap",
        "beta_visual_similarity",
        "beta_input_energy",
        "beta_global_support",
        "r2",
        "cv_r2",
        "n_samples",
        "real_rollout",
        "final_scientific_use",
    ]
    return _summary_regression_rows(ctx, df, ["reentry_strength", "dynamic_like_reentry", "decision_deflection_score"], "metric", cols)


def _panel_e_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "network_seed",
        "downstream_metric",
        "beta_peak_weighted_overlap",
        "beta_raw_overlap",
        "beta_visual_similarity",
        "beta_input_energy",
        "beta_global_support",
        "beta_total_spike_count",
        "r2",
        "cv_r2",
        "n_samples",
        "real_rollout",
        "final_scientific_use",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for metric, part in df.groupby("downstream_metric", sort=True):
        rows.extend(_summary_regression_rows(ctx, part, ["metric_value"], "downstream_metric", cols, label=str(metric)).to_dict("records"))
    return pd.DataFrame(rows, columns=cols)


def _summary_regression_rows(ctx: ExperimentContext, df: pd.DataFrame, metrics: Sequence[str], label_col: str, columns: Sequence[str], label: str | None = None) -> pd.DataFrame:
    rows = []
    predictors = ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count"]
    for metric in metrics:
        available = [p for p in predictors if p in df.columns]
        cols = available + [metric]
        use = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        fit = _fit_ols(use[available].to_numpy(dtype=float), use[metric].to_numpy(dtype=float)) if len(use) >= 4 and available else None
        beta = dict(zip(["intercept"] + available, np.asarray(fit["beta"], dtype=float))) if fit is not None else {}
        r2 = float(fit["r2"]) if fit is not None else np.nan
        cv = _cv_r2(use[available].to_numpy(dtype=float), use[metric].to_numpy(dtype=float), n_folds=min(5, max(2, len(use) // 2))) if len(use) >= 6 and available else np.nan
        row = {
            "network_seed": int(ctx.cfg.network_seed),
            label_col: label if label is not None else metric,
            "beta_peak_weighted_overlap": float(beta.get("peak_weighted_overlap", np.nan)),
            "beta_raw_overlap": float(beta.get("raw_overlap", np.nan)),
            "beta_visual_similarity": float(beta.get("visual_similarity", np.nan)),
            "beta_input_energy": float(beta.get("input_energy", np.nan)),
            "beta_global_support": float(beta.get("global_support", np.nan)),
            "beta_total_spike_count": float(beta.get("total_spike_count", np.nan)),
            "r2": r2,
            "cv_r2": float(cv),
            "n_samples": int(len(use)),
            "real_rollout": bool(_df_all_true(df, "real_rollout")),
            "final_scientific_use": bool(_df_all_true(df, "final_scientific_use")),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _regression_long_table(ctx: ExperimentContext, df: pd.DataFrame, dependent_metrics: Sequence[str]) -> pd.DataFrame:
    columns = [
        "network_seed",
        "dependent_metric",
        "predictor",
        "beta",
        "std_beta",
        "se",
        "t_value",
        "p_value",
        "r2",
        "n_samples",
        "model_formula",
        "real_rollout",
        "final_scientific_use",
    ]
    predictors = ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count"]
    rows = []
    for metric in dependent_metrics:
        available = [p for p in predictors if p in df.columns]
        if metric not in df.columns or not available:
            continue
        use = df[available + [metric]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(use) >= 4:
            x = use[available].to_numpy(dtype=float)
            y = use[metric].to_numpy(dtype=float)
            fit = _fit_ols(x, y)
            beta = np.asarray(fit["beta"], dtype=float)
            se = np.asarray(fit["se"], dtype=float)
            p = np.asarray(fit["p"], dtype=float)
            r2 = float(fit["r2"])
        else:
            x = np.empty((0, len(available)))
            y = np.empty(0)
            beta = np.full(len(available) + 1, np.nan)
            se = np.full(len(available) + 1, np.nan)
            p = np.full(len(available) + 1, np.nan)
            r2 = np.nan
        for idx, pred in enumerate(["intercept"] + available):
            b = float(beta[idx])
            s = float(se[idx])
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "dependent_metric": metric,
                    "predictor": pred,
                    "beta": b,
                    "std_beta": _standardized_coef(b, x, y, pred, available) if len(y) else np.nan,
                    "se": s,
                    "t_value": float(b / s) if np.isfinite(s) and s > 1e-12 else np.nan,
                    "p_value": float(p[idx]),
                    "r2": r2,
                    "n_samples": int(len(use)),
                    "model_formula": f"{metric} ~ " + " + ".join(available),
                    "real_rollout": bool(_df_all_true(df, "real_rollout")),
                    "final_scientific_use": bool(_df_all_true(df, "final_scientific_use")),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _panel_d_matched_contrast(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "matched_set_id",
        "raw_overlap_low",
        "raw_overlap_high",
        "peak_weighted_overlap_low",
        "peak_weighted_overlap_high",
        "raw_overlap_difference",
        "peak_weighted_overlap_difference",
        "visual_similarity_difference",
        "input_energy_difference",
        "reentry_low",
        "reentry_high",
        "reentry_high_minus_low",
        "decision_deflection_low",
        "decision_deflection_high",
        "decision_deflection_high_minus_low",
        "n_pairs",
    ]
    if df.empty or "matched_set_id" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    matched = df[df["matched_set_id"].astype(str).str.len() > 0]
    for gid, part in matched.groupby("matched_set_id", sort=True):
        high = part[part.get("peak_overlap_group", "").astype(str).eq("high_peak_overlap")]
        low = part[part.get("peak_overlap_group", "").astype(str).eq("low_peak_overlap")]
        if high.empty or low.empty:
            continue
        h = high.iloc[0]
        l = low.iloc[0]
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "matched_set_id": str(gid),
                "raw_overlap_low": float(l["raw_overlap"]),
                "raw_overlap_high": float(h["raw_overlap"]),
                "peak_weighted_overlap_low": float(l["peak_weighted_overlap"]),
                "peak_weighted_overlap_high": float(h["peak_weighted_overlap"]),
                "raw_overlap_difference": float(h["raw_overlap"] - l["raw_overlap"]),
                "peak_weighted_overlap_difference": float(h["peak_weighted_overlap"] - l["peak_weighted_overlap"]),
                "visual_similarity_difference": float(h["visual_similarity"] - l["visual_similarity"]),
                "input_energy_difference": float(h["input_energy"] - l["input_energy"]),
                "reentry_low": float(l.get("reentry_strength", np.nan)),
                "reentry_high": float(h.get("reentry_strength", np.nan)),
                "reentry_high_minus_low": float(h.get("reentry_strength", np.nan) - l.get("reentry_strength", np.nan)),
                "decision_deflection_low": float(l.get("decision_deflection_score", np.nan)),
                "decision_deflection_high": float(h.get("decision_deflection_score", np.nan)),
                "decision_deflection_high_minus_low": float(h.get("decision_deflection_score", np.nan) - l.get("decision_deflection_score", np.nan)),
                "n_pairs": int(min(len(high), len(low))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _panel_e_breakdown(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "downstream_metric",
        "high_peak_overlap_mean",
        "low_peak_overlap_mean",
        "high_minus_low",
        "beta_peak_weighted_overlap",
        "beta_raw_overlap",
        "beta_visual_similarity",
        "beta_input_energy",
        "n_samples",
        "real_rollout",
        "final_scientific_use",
    ]
    rows = []
    if not df.empty:
        summary = _panel_e_summary(ctx, df)
        for metric, part in df.groupby("downstream_metric", sort=True):
            high = pd.to_numeric(part[part["peak_overlap_group"].astype(str).eq("high_peak_overlap")]["metric_value"], errors="coerce")
            low = pd.to_numeric(part[part["peak_overlap_group"].astype(str).eq("low_peak_overlap")]["metric_value"], errors="coerce")
            s = summary[summary["downstream_metric"].eq(metric)]
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "downstream_metric": str(metric),
                    "high_peak_overlap_mean": float(high.mean()) if len(high) else np.nan,
                    "low_peak_overlap_mean": float(low.mean()) if len(low) else np.nan,
                    "high_minus_low": float(high.mean() - low.mean()) if len(high) and len(low) else np.nan,
                    "beta_peak_weighted_overlap": float(s["beta_peak_weighted_overlap"].iloc[0]) if not s.empty else np.nan,
                    "beta_raw_overlap": float(s["beta_raw_overlap"].iloc[0]) if not s.empty else np.nan,
                    "beta_visual_similarity": float(s["beta_visual_similarity"].iloc[0]) if not s.empty else np.nan,
                    "beta_input_energy": float(s["beta_input_energy"].iloc[0]) if not s.empty else np.nan,
                    "n_samples": int(len(part)),
                    "real_rollout": bool(_df_all_true(part, "real_rollout")),
                    "final_scientific_use": bool(_df_all_true(part, "final_scientific_use")),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _s11_peak_update_group_enrichment(ctx: ExperimentContext, unit_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "update_group", "region_type", "P_peak", "mean_final_support", "mean_delta_support", "mean_update_count", "mean_recent_update_count", "n_units"]
    rows = []
    if unit_df.empty:
        return pd.DataFrame(columns=columns)
    df = unit_df.copy()
    df["update_group"] = "single_old"
    for group in UPDATE_GROUPS:
        df.loc[_group_mask(df, int(ctx.cfg.recent_window), int(ctx.cfg.multi_update_threshold), group), "update_group"] = group
    regions = {
        "peak": df["is_peak"].astype(bool),
        "nonpeak_control": df["is_nonpeak_control"].astype(bool),
        "prior_updated_nonpeak": (~df["is_peak"].astype(bool)) & (pd.to_numeric(df["update_count"], errors="coerce") > 0),
    }
    for group in UPDATE_GROUPS:
        group_df = df[df["update_group"].eq(group)]
        for region, mask in regions.items():
            part = group_df[mask.reindex(group_df.index, fill_value=False)]
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "update_group": group,
                    "region_type": region,
                    "P_peak": _mean_col(part, "is_peak"),
                    "mean_final_support": _mean_col(part, "final_support"),
                    "mean_delta_support": _mean_col(part, "delta_support"),
                    "mean_update_count": _mean_col(part, "update_count"),
                    "mean_recent_update_count": _mean_col(part, f"is_multi_recent_w{ctx.cfg.recent_window}"),
                    "n_units": int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _s11_update_recency_model_comparison(ctx: ExperimentContext, model_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "dependent_metric", "model_name", "r2", "cv_r2", "delta_r2_vs_baseline", "n_samples", "model_formula"]
    if model_df.empty:
        return pd.DataFrame(columns=columns)
    df = model_df.copy()
    target_col = "target" if "target" in df.columns else "dependent_metric"
    rows = []
    for target, part in df.groupby(target_col, sort=True):
        baseline = part[part["model_name"].eq("baseline_only")]
        baseline_r2 = float(baseline["r2"].iloc[0]) if not baseline.empty and "r2" in baseline else np.nan
        for r in part.itertuples(index=False):
            model_name = str(getattr(r, "model_name"))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "dependent_metric": str(target),
                    "model_name": model_name,
                    "r2": float(getattr(r, "r2", np.nan)),
                    "cv_r2": float(getattr(r, "cv_r2", np.nan)),
                    "delta_r2_vs_baseline": float(getattr(r, "r2", np.nan) - baseline_r2) if np.isfinite(baseline_r2) else np.nan,
                    "n_samples": int(getattr(r, "n_units", 0)),
                    "model_formula": _model_formula(model_name, str(target)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _s11_leave_one_out_source_details(ctx: ExperimentContext, source_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "sequence_id", "removed_position", "relative_position_from_end", "peak_loss", "nonpeak_loss", "prior_updated_loss", "peak_loss_fraction", "nonpeak_loss_fraction", "peak_vs_nonpeak_loss_ratio", "support_loss_total", "n_peak_units", "n_nonpeak_units"]
    if source_df.empty:
        return pd.DataFrame(columns=columns)
    out = source_df.copy()
    out["relative_position_from_end"] = pd.to_numeric(out["seq_len"], errors="coerce") - pd.to_numeric(out["removed_position"], errors="coerce")
    out["n_peak_units"] = np.nan
    out["n_nonpeak_units"] = np.nan
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def _s11_recent_overlap_window_robustness(ctx: ExperimentContext, overlap_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "window_type", "recent_k", "dice_peak_overlap", "jaccard_peak_overlap", "peak_coverage", "overlap_precision", "cosine_delta_support_overlap", "spearman_delta_support_overlap", "old_window_control", "n_sequences"]
    rows = []
    if not overlap_df.empty:
        df = overlap_df.copy()
        df["recent_k"] = df["overlap_window"].astype(str).str.extract(r"(\d+)").astype(float)
        for (window_type, recent_k), part in df.groupby(["overlap_type", "recent_k"], dropna=False, sort=True):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "window_type": str(window_type),
                    "recent_k": int(recent_k) if np.isfinite(recent_k) else 0,
                    "dice_peak_overlap": _mean_col(part, "dice_peak_overlap"),
                    "jaccard_peak_overlap": _mean_col(part, "jaccard_peak_overlap"),
                    "peak_coverage": _mean_col(part, "peak_coverage"),
                    "overlap_precision": _mean_col(part, "overlap_precision"),
                    "cosine_delta_support_overlap": _mean_col(part, "cosine_delta_support_overlap_count"),
                    "spearman_delta_support_overlap": _mean_col(part, "spearman_delta_support_overlap_count"),
                    "old_window_control": bool(str(window_type) == "old"),
                    "n_sequences": int(part["sequence_id"].nunique()) if "sequence_id" in part.columns else int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _s11_alternative_peak_definitions(ctx: ExperimentContext, alt_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "peak_definition", "peak_quantile", "positive_delta_only", "n_peak_units", "multi_recent_enrichment", "peak_overlap_dice", "peak_overlap_coverage", "peak_weighted_overlap_effect", "n_sequences"]
    rows = []
    if not alt_df.empty:
        for name, part in alt_df.groupby("peak_definition", sort=True):
            metrics = {str(r.metric): float(r.value) for r in part.itertuples(index=False) if hasattr(r, "metric") and hasattr(r, "value")}
            q = 0.10 if "10" in str(name) else 0.30 if "30" in str(name) else 0.20 if "20" in str(name) else np.nan
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "peak_definition": str(name),
                    "peak_quantile": float(q),
                    "positive_delta_only": bool("positive" in str(name) or "top_" in str(name)),
                    "n_peak_units": int(metrics.get("n_peak_units", np.nan)) if np.isfinite(metrics.get("n_peak_units", np.nan)) else 0,
                    "multi_recent_enrichment": float(metrics.get("multi_recent_enrichment", np.nan)),
                    "peak_overlap_dice": float(metrics.get("peak_overlap_dice", np.nan)),
                    "peak_overlap_coverage": float(metrics.get("peak_overlap_coverage", np.nan)),
                    "peak_weighted_overlap_effect": float(metrics.get("peak_weighted_overlap_effect", np.nan)),
                    "n_sequences": int(ctx.n_sequences),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _s11_visual_energy_classpair_controls(ctx: ExperimentContext, d_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "comparison", "group", "mean_input_energy", "mean_foreground_area", "mean_visual_similarity", "class_pair_entropy", "class_pair_balance_stat", "n_samples", "energy_difference", "foreground_difference", "visual_similarity_difference"]
    rows = []
    if not d_metrics.empty:
        groups = {
            "high_peak_overlap": d_metrics[d_metrics.get("peak_overlap_group", "").astype(str).eq("high_peak_overlap")],
            "low_peak_overlap": d_metrics[d_metrics.get("peak_overlap_group", "").astype(str).eq("low_peak_overlap")],
            "all": d_metrics,
        }
        high_energy = _mean_col(groups["high_peak_overlap"], "input_energy")
        low_energy = _mean_col(groups["low_peak_overlap"], "input_energy")
        high_visual = _mean_col(groups["high_peak_overlap"], "visual_similarity")
        low_visual = _mean_col(groups["low_peak_overlap"], "visual_similarity")
        for group, part in groups.items():
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "comparison": "peak_overlap_group",
                    "group": group,
                    "mean_input_energy": _mean_col(part, "input_energy"),
                    "mean_foreground_area": _mean_col(part, "input_energy"),
                    "mean_visual_similarity": _mean_col(part, "visual_similarity"),
                    "class_pair_entropy": np.nan,
                    "class_pair_balance_stat": np.nan,
                    "n_samples": int(len(part)),
                    "energy_difference": float(high_energy - low_energy) if np.isfinite(high_energy) and np.isfinite(low_energy) else np.nan,
                    "foreground_difference": float(high_energy - low_energy) if np.isfinite(high_energy) and np.isfinite(low_energy) else np.nan,
                    "visual_similarity_difference": float(high_visual - low_visual) if np.isfinite(high_visual) and np.isfinite(low_visual) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _s12_peak_weighted_regression_controls(ctx: ExperimentContext, d_reg: pd.DataFrame | None, e_summary: pd.DataFrame | None) -> pd.DataFrame | None:
    columns = ["network_seed", "dependent_metric", "predictor", "beta", "std_beta", "se", "t_value", "p_value", "r2", "n_samples", "model_formula", "controls_included", "real_rollout", "final_scientific_use"]
    rows = []
    if d_reg is not None and not d_reg.empty:
        for r in d_reg.itertuples(index=False):
            row = {col: getattr(r, col, np.nan) for col in columns if hasattr(r, col)}
            row["controls_included"] = "raw_overlap,visual_similarity,input_energy,global_support"
            rows.append(row)
    if e_summary is not None and not e_summary.empty:
        predictors = ["peak_weighted_overlap", "raw_overlap", "visual_similarity", "input_energy", "global_support", "total_spike_count"]
        for r in e_summary.itertuples(index=False):
            for pred in predictors:
                attr = f"beta_{pred}"
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "dependent_metric": str(getattr(r, "downstream_metric")),
                        "predictor": pred,
                        "beta": float(getattr(r, attr, np.nan)),
                        "std_beta": np.nan,
                        "se": np.nan,
                        "t_value": np.nan,
                        "p_value": np.nan,
                        "r2": float(getattr(r, "r2", np.nan)),
                        "n_samples": int(getattr(r, "n_samples", 0)),
                        "model_formula": "metric_value ~ " + " + ".join(predictors),
                        "controls_included": "raw_overlap,visual_similarity,input_energy,global_support,total_spike_count",
                        "real_rollout": bool(getattr(r, "real_rollout", False)),
                        "final_scientific_use": bool(getattr(r, "final_scientific_use", False)),
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows, columns=columns)


def _s12_global_support_controls(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    columns = ["network_seed", "dependent_metric", "beta_peak_weighted_overlap", "beta_global_support", "beta_total_spike_count", "beta_nonpeak_support", "r2", "n_samples", "controls_included", "real_rollout", "final_scientific_use"]
    rows = []
    if not df.empty:
        predictors = ["peak_weighted_overlap", "global_support", "total_spike_count", "nonpeak_support"]
        for metric, part in df.groupby("downstream_metric", sort=True):
            available = [p for p in predictors if p in part.columns]
            use = part[available + ["metric_value"]].apply(pd.to_numeric, errors="coerce").dropna()
            fit = _fit_ols(use[available].to_numpy(dtype=float), use["metric_value"].to_numpy(dtype=float)) if len(use) >= 4 and available else None
            beta = dict(zip(["intercept"] + available, np.asarray(fit["beta"], dtype=float))) if fit is not None else {}
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "dependent_metric": str(metric),
                    "beta_peak_weighted_overlap": float(beta.get("peak_weighted_overlap", np.nan)),
                    "beta_global_support": float(beta.get("global_support", np.nan)),
                    "beta_total_spike_count": float(beta.get("total_spike_count", np.nan)),
                    "beta_nonpeak_support": float(beta.get("nonpeak_support", np.nan)),
                    "r2": float(fit["r2"]) if fit is not None else np.nan,
                    "n_samples": int(len(use)),
                    "controls_included": ",".join(available),
                    "real_rollout": bool(_df_all_true(part, "real_rollout")),
                    "final_scientific_use": bool(_df_all_true(part, "final_scientific_use")),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _write_standardized_peak_perturbation_outputs(ctx: ExperimentContext) -> None:
    d_source = _read_csv_if_exists(ctx.raw_dir / "panel_d_route_peak_perturbation_trial_readout.csv")
    e_source = _read_csv_if_exists(ctx.raw_dir / "panel_e_route_peak_downstream_trial_readout.csv")
    metric_columns = ["network_seed", "sequence_id", "condition", "perturbation_target", "reentry_strength", "decision_deflection_score", "downstream_metric", "metric_value", "overlap_aligned_peak", "control_peak", "random_matched_peak", "n_units_perturbed"]
    summary_columns = ["network_seed", "metric", "intact_value", "overlap_peak_perturb_value", "control_peak_perturb_value", "random_peak_perturb_value", "overlap_peak_reduction", "control_peak_reduction", "overlap_minus_control_reduction", "n_sequences", "claim_upgrade_allowed"]
    if d_source is None or e_source is None:
        ctx.warnings.append("Peak perturbation requested but route-peak perturbation source outputs are missing.")
        return
    rows = []
    for _, r in d_source.iterrows():
        unit_set = str(r.get("perturbation_unit_set", ""))
        target = _perturbation_target(unit_set)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.get("sequence_id", -1)) if pd.notna(r.get("sequence_id", np.nan)) else -1,
                "condition": unit_set,
                "perturbation_target": target,
                "reentry_strength": _num(r.get("reentry_strength_perturbed")),
                "decision_deflection_score": np.nan,
                "downstream_metric": "normalized_reentry_loss",
                "metric_value": _num(r.get("normalized_reentry_loss")),
                "overlap_aligned_peak": bool(target == "overlap_aligned_peak"),
                "control_peak": bool(target == "control_peak"),
                "random_matched_peak": bool(target == "random_matched_peak"),
                "n_units_perturbed": int(r.get(f"{unit_set}_unit_count", r.get("route_peak_unit_count", 0))) if unit_set in PERTURBATION_UNIT_SET_ORDER else 0,
            }
        )
    for _, r in e_source.iterrows():
        unit_set = str(r.get("perturbation_unit_set", ""))
        target = _perturbation_target(unit_set)
        for metric in ("output_switch", "response_displacement_loss", "decision_deflection_loss"):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(r.get("sequence_id", -1)) if pd.notna(r.get("sequence_id", np.nan)) else -1,
                    "condition": unit_set,
                    "perturbation_target": target,
                    "reentry_strength": np.nan,
                    "decision_deflection_score": _num(r.get("decision_deflection_loss")),
                    "downstream_metric": metric,
                    "metric_value": float(_bool_value(r.get(metric))) if metric == "output_switch" else _num(r.get(metric)),
                    "overlap_aligned_peak": bool(target == "overlap_aligned_peak"),
                    "control_peak": bool(target == "control_peak"),
                    "random_matched_peak": bool(target == "random_matched_peak"),
                    "n_units_perturbed": 0,
                }
            )
    metrics = pd.DataFrame(rows, columns=metric_columns)
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_s12_peak_perturbation_metrics.csv")
    summary_rows = []
    for metric, part in metrics.groupby("downstream_metric", sort=True):
        intact = pd.to_numeric(part[part["perturbation_target"].eq("intact")]["metric_value"], errors="coerce").mean()
        overlap = pd.to_numeric(part[part["perturbation_target"].eq("overlap_aligned_peak")]["metric_value"], errors="coerce").mean()
        control = pd.to_numeric(part[part["perturbation_target"].eq("control_peak")]["metric_value"], errors="coerce").mean()
        random = pd.to_numeric(part[part["perturbation_target"].eq("random_matched_peak")]["metric_value"], errors="coerce").mean()
        overlap_reduction = intact - overlap
        control_reduction = np.nanmax([intact - control, intact - random])
        claim_upgrade = bool(np.isfinite(overlap_reduction) and np.isfinite(control_reduction) and overlap_reduction > control_reduction and not part.empty)
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "metric": str(metric),
                "intact_value": float(intact),
                "overlap_peak_perturb_value": float(overlap),
                "control_peak_perturb_value": float(control),
                "random_peak_perturb_value": float(random),
                "overlap_peak_reduction": float(overlap_reduction),
                "control_peak_reduction": float(control_reduction),
                "overlap_minus_control_reduction": float(overlap_reduction - control_reduction),
                "n_sequences": int(part["sequence_id"].nunique()),
                "claim_upgrade_allowed": claim_upgrade,
            }
        )
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=summary_columns), ctx.metrics_dir / "supp_s12_peak_perturbation_summary.csv")


def _real_rollout_scientific_use_audit(ctx: ExperimentContext) -> pd.DataFrame:
    columns = ["network_seed", "module", "output_file", "proxy_mode", "real_rollout", "final_scientific_use", "n_rows", "n_real_rows", "n_proxy_rows", "main_claim_allowed", "claim_strength", "missing_reason"]
    rows = []
    for module, rel in (
        ("Fig6D_peak_weighted_reentry", "data/metrics/panel_d_peak_weighted_reentry_metrics.csv"),
        ("Fig6E_peak_weighted_downstream", "data/metrics/panel_e_peak_weighted_downstream_metrics.csv"),
    ):
        path = ctx.seed_dir / rel
        missing_reason = ""
        if not path.exists():
            df = pd.DataFrame()
            missing_reason = "missing_output"
        else:
            df = pd.read_csv(path)
            if df.empty:
                missing_reason = "empty_output"
        proxy = bool(_df_all_proxy(df)) if not df.empty else True
        real_rows = int(_bool_col(df, "real_rollout").sum()) if not df.empty and "real_rollout" in df.columns else 0
        proxy_rows = int(_bool_col(df, "proxy_mode").sum()) if not df.empty and "proxy_mode" in df.columns else int(len(df))
        final = bool(_df_all_true(df, "final_scientific_use")) if not df.empty else False
        real = bool(len(df) > 0 and real_rows == len(df))
        allowed = bool(real and final and len(df) > 0)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "module": module,
                "output_file": rel,
                "proxy_mode": proxy,
                "real_rollout": real,
                "final_scientific_use": final,
                "n_rows": int(len(df)),
                "n_real_rows": real_rows,
                "n_proxy_rows": proxy_rows,
                "main_claim_allowed": allowed,
                "claim_strength": _claim_strength(ctx),
                "missing_reason": missing_reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_route_peak_perturbation_outputs(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    if bank.probe_trials.empty:
        build_later_probe_peak_overlap_trials(ctx, bank)
    if _is_proxy_mode(ctx):
        write_route_peak_perturbation_unavailable_outputs(ctx, reason="proxy_mode_no_real_state_perturbation")
        return
    rows_d: list[dict[str, Any]] = []
    rows_e: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    output_distribution_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 route-peak perturbation", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        boundary = bank.boundaries.get(int(r.sequence_id))
        if not boundary:
            rows_d.extend(_diagnostic_perturbation_rows_d(ctx, r, bank, seq_idx, "missing_sfinal_boundary"))
            rows_e.extend(_diagnostic_perturbation_rows_e(ctx, r, "missing_sfinal_boundary"))
            continue
        probe_mask = _foreground_mask(ctx.dataset, int(r.probe_image_id), ctx.cfg.foreground_threshold).reshape(-1)
        peak = bank.peak_mask[seq_idx].reshape(-1).astype(bool)
        prior = bank.prior_updated_mask[seq_idx].reshape(-1).astype(bool)
        route = probe_mask.astype(bool) & prior
        set_masks = {
            "route_peak": route & peak,
            "route_nonpeak": route & ~peak,
            "nonroute_peak": peak & ~route,
        }
        set_masks["random_matched"] = _matched_random_unit_mask(~set_masks["route_peak"], int(set_masks["route_peak"].sum()), int(ctx.cfg.network_seed) + int(r.probe_id))
        counts = {name: int(mask.sum()) for name, mask in set_masks.items()}
        insufficient = {name: bool(count <= 0) for name, count in counts.items()}
        for name, mask in set_masks.items():
            for unit_id in np.flatnonzero(mask)[:200]:
                unit_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(r.sequence_id),
                        "probe_id": int(r.probe_id),
                        "perturbation_unit_set": name,
                        "unit_id": int(unit_id),
                        "notes": "route_peak = probe foreground intersect prior-updated foreground intersect final peak mask",
                    }
                )
        try:
            probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
            intact_trace, intact_pred, intact_fire, intact_vec = _run_real_probe_from_condition(ctx, int(r.probe_image_id), boundary, "S_final", probe_spikes=probe_spikes)
            s0_trace, s0_pred, s0_fire, s0_vec = _run_real_probe_from_condition(ctx, int(r.probe_image_id), None, "S0", probe_spikes=probe_spikes)
        except Exception as exc:
            reason = f"baseline_or_intact_rollout_failed:{exc}"
            ctx.warnings.append(f"Fig.6 route-peak perturbation failed for sequence={int(r.sequence_id)} probe={int(r.probe_id)}: {exc}")
            rows_d.extend(_diagnostic_perturbation_rows_d(ctx, r, bank, seq_idx, reason))
            rows_e.extend(_diagnostic_perturbation_rows_e(ctx, r, reason))
            continue
        reentry_intact = float(np.linalg.norm(intact_trace.reshape(-1) - s0_trace.reshape(-1)))
        reentry_s0 = 0.0
        response_intact = float(np.linalg.norm(intact_vec.reshape(-1) - s0_vec.reshape(-1)))
        response_s0 = 0.0
        deflection_intact = float(_label_evidence(intact_vec, int(r.probe_label)) - _label_evidence(s0_vec, int(r.probe_label)))
        deflection_s0 = 0.0
        output_distribution_rows.append(_output_distribution_row(ctx, r, "intact", intact_vec, intact_vec, js=0.0))
        for unit_set in PERTURBATION_UNIT_SET_ORDER:
            selected = np.flatnonzero(set_masks[unit_set]).astype(np.int64)
            perturb_ok = False
            reset_record: dict[str, Any] = {}
            if insufficient[unit_set]:
                pert_trace = np.full_like(intact_trace, np.nan)
                pert_vec = np.full_like(intact_vec, np.nan)
                pert_pred = -1
                pert_fire = -1
                failure_reason = "insufficient_units"
            else:
                try:
                    pert_trace, pert_pred, pert_fire, pert_vec, reset_record = _run_real_probe_with_route_peak_reset(
                        ctx,
                        int(r.probe_image_id),
                        boundary,
                        selected,
                        probe_spikes=probe_spikes,
                    )
                    perturb_ok = True
                    failure_reason = ""
                except Exception as exc:
                    pert_trace = np.full_like(intact_trace, np.nan)
                    pert_vec = np.full_like(intact_vec, np.nan)
                    pert_pred = -1
                    pert_fire = -1
                    failure_reason = f"perturbation_failed:{exc}"
                    ctx.warnings.append(f"Fig.6 {unit_set} reset failed for sequence={int(r.sequence_id)} probe={int(r.probe_id)}: {exc}")
            reentry_pert = float(np.linalg.norm(pert_trace.reshape(-1) - s0_trace.reshape(-1))) if perturb_ok else np.nan
            reentry_loss = float(reentry_intact - reentry_pert) if perturb_ok else np.nan
            normalized_loss = float(reentry_loss / max(abs(reentry_intact), 1e-9)) if perturb_ok else np.nan
            response_pert = float(np.linalg.norm(pert_vec.reshape(-1) - s0_vec.reshape(-1))) if perturb_ok else np.nan
            response_loss = float(response_intact - response_pert) if perturb_ok else np.nan
            deflection_pert = float(_label_evidence(pert_vec, int(r.probe_label)) - _label_evidence(s0_vec, int(r.probe_label))) if perturb_ok else np.nan
            deflection_loss = float(deflection_intact - deflection_pert) if perturb_ok else np.nan
            js = _js_divergence(intact_vec, pert_vec) if perturb_ok else np.nan
            output_distribution_rows.append(_output_distribution_row(ctx, r, unit_set, intact_vec, pert_vec, js=js))
            common = {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "probe_label": int(r.probe_label),
                "seq_len": int(bank.sequence_meta.iloc[seq_idx]["seq_len"]),
                "perturbation_unit_set": unit_set,
                "perturbation_condition": f"{unit_set}_reset",
                "perturbation_mode": "reset_u_x_to_s0",
                "state_condition": "S_final_then_reset_before_probe",
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "route_unit_count": int(route.sum()),
                "peak_unit_count": int(peak.sum()),
                "route_peak_unit_count": counts["route_peak"],
                "route_nonpeak_unit_count": counts["route_nonpeak"],
                "nonroute_peak_unit_count": counts["nonroute_peak"],
                "random_unit_count": counts["random_matched"],
                "insufficient_units": bool(insufficient[unit_set]),
                "restore_ok": bool(reset_record.get("restore_ok", perturb_ok)),
                "perturbation_ok": bool(perturb_ok),
                "failure_reason": failure_reason,
            }
            rows_d.append(
                {
                    **common,
                    "reentry_strength_intact": reentry_intact,
                    "reentry_strength_perturbed": reentry_pert,
                    "reentry_strength_s0": reentry_s0,
                    "reentry_loss": reentry_loss,
                    "normalized_reentry_loss": normalized_loss,
                    "prediction_intact": int(intact_pred),
                    "prediction_perturbed": int(pert_pred),
                    "prediction_s0": int(s0_pred),
                    "first_fire_time_intact": int(intact_fire),
                    "first_fire_time_perturbed": int(pert_fire),
                    "first_fire_time_s0": int(s0_fire),
                    "denominator_choice": "max(abs(reentry_strength_intact), eps)",
                    "reset_variables": str(reset_record.get("reset_variables", "u,x")),
                    "probe_input_unchanged": True,
                }
            )
            rows_e.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(r.sequence_id),
                    "probe_id": int(r.probe_id),
                    "probe_label": int(r.probe_label),
                    "perturbation_unit_set": unit_set,
                    "perturbation_condition": f"{unit_set}_reset",
                    "response_displacement_intact": response_intact,
                    "response_displacement_perturbed": response_pert,
                    "response_displacement_s0": response_s0,
                    "response_displacement_loss": response_loss,
                    "decision_deflection_intact": deflection_intact,
                    "decision_deflection_perturbed": deflection_pert,
                    "decision_deflection_s0": deflection_s0,
                    "decision_deflection_loss": deflection_loss,
                    "prediction_intact": int(intact_pred),
                    "prediction_perturbed": int(pert_pred),
                    "prediction_s0": int(s0_pred),
                    "output_switch": bool(perturb_ok and int(intact_pred) != int(pert_pred)),
                    "output_distribution_JS": js,
                    "perturbation_ok": bool(perturb_ok),
                    "insufficient_units": bool(insufficient[unit_set]),
                    "failure_reason": failure_reason,
                }
            )
    df_d = pd.DataFrame(rows_d, columns=PANEL_D_ROUTE_PEAK_TRIAL_COLUMNS)
    df_e = pd.DataFrame(rows_e, columns=PANEL_E_ROUTE_PEAK_TRIAL_COLUMNS)
    _save_csv(ctx, df_d, ctx.raw_dir / "panel_d_route_peak_perturbation_trial_readout.csv")
    _save_csv(ctx, df_e, ctx.raw_dir / "panel_e_route_peak_downstream_trial_readout.csv")
    _save_csv(ctx, _route_peak_reentry_summary(ctx, df_d), ctx.metrics_dir / "panel_d_route_peak_reentry_loss_summary.csv")
    _save_csv(ctx, _route_peak_reentry_contrast(ctx, df_d), ctx.metrics_dir / "panel_d_route_peak_reentry_loss_contrast.csv")
    _save_csv(ctx, _route_peak_perturbation_audit(ctx, df_d, df_e, reason=""), ctx.metrics_dir / "panel_d_route_peak_perturbation_audit.csv")
    _save_csv(ctx, _route_peak_downstream_summary(ctx, df_e), ctx.metrics_dir / "panel_e_route_peak_downstream_summary.csv")
    _save_csv(ctx, _route_peak_downstream_contrast(ctx, df_e), ctx.metrics_dir / "panel_e_route_peak_downstream_contrast.csv")
    _save_csv(ctx, pd.DataFrame(output_distribution_rows, columns=PANEL_E_ROUTE_PEAK_OUTPUT_DISTRIBUTION_COLUMNS), ctx.metrics_dir / "panel_e_route_peak_output_distribution.csv")
    _save_csv(ctx, _route_peak_scientific_use_audit(ctx, df_d, df_e, reason=""), ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv")
    _save_csv(ctx, pd.DataFrame(unit_rows, columns=ROUTE_PEAK_UNIT_SET_COLUMNS), ctx.trial_specs_dir / "peak_perturbation_unit_sets.csv")
    ctx.completed_modules["peak_perturbation"] = bool(_route_peak_success(df_d, df_e))
    if not ctx.completed_modules["peak_perturbation"]:
        ctx.warnings.append("Route-peak perturbation ran but did not pass causal-use audit; Fig.6D/E claims remain predictive/diagnostic.")


def write_route_peak_perturbation_unavailable_outputs(ctx: ExperimentContext, *, reason: str) -> None:
    df_d = pd.DataFrame(columns=PANEL_D_ROUTE_PEAK_TRIAL_COLUMNS)
    df_e = pd.DataFrame(columns=PANEL_E_ROUTE_PEAK_TRIAL_COLUMNS)
    _save_csv(ctx, df_d, ctx.raw_dir / "panel_d_route_peak_perturbation_trial_readout.csv")
    _save_csv(ctx, df_e, ctx.raw_dir / "panel_e_route_peak_downstream_trial_readout.csv")
    _save_csv(ctx, _route_peak_reentry_summary(ctx, df_d), ctx.metrics_dir / "panel_d_route_peak_reentry_loss_summary.csv")
    _save_csv(ctx, _route_peak_reentry_contrast(ctx, df_d), ctx.metrics_dir / "panel_d_route_peak_reentry_loss_contrast.csv")
    _save_csv(ctx, _route_peak_perturbation_audit(ctx, df_d, df_e, reason=reason), ctx.metrics_dir / "panel_d_route_peak_perturbation_audit.csv")
    _save_csv(ctx, _route_peak_downstream_summary(ctx, df_e), ctx.metrics_dir / "panel_e_route_peak_downstream_summary.csv")
    _save_csv(ctx, _route_peak_downstream_contrast(ctx, df_e), ctx.metrics_dir / "panel_e_route_peak_downstream_contrast.csv")
    _save_csv(ctx, pd.DataFrame(columns=PANEL_E_ROUTE_PEAK_OUTPUT_DISTRIBUTION_COLUMNS), ctx.metrics_dir / "panel_e_route_peak_output_distribution.csv")
    _save_csv(ctx, _route_peak_scientific_use_audit(ctx, df_d, df_e, reason=reason), ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv")
    _save_csv(ctx, pd.DataFrame(columns=ROUTE_PEAK_UNIT_SET_COLUMNS), ctx.trial_specs_dir / "peak_perturbation_unit_sets.csv")
    ctx.completed_modules["peak_perturbation"] = False
    ctx.warnings.append(f"Route-peak perturbation unavailable ({reason}); Fig.6D/E cannot use causal claim language.")


def _run_real_probe_with_route_peak_reset(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundary: Mapping[str, Mapping[str, Any]],
    selected_units: np.ndarray,
    *,
    probe_spikes: Any,
) -> tuple[np.ndarray, int, int, np.ndarray, dict[str, Any]]:
    if ctx.net is None or torch is None:
        raise RuntimeError("route-peak perturbation requires a real network")
    _restore_boundary_state(ctx.net, boundary)
    reset_record = _reset_layer1_stsp_units_to_s0(ctx.net, selected_units)
    trace, pred, fire, vector = _run_real_probe_from_condition(ctx, probe_image_id, snapshot_boundary_state(ctx.net), "S_final_route_peak_reset", probe_spikes=probe_spikes)
    return trace, pred, fire, vector, reset_record


def _reset_layer1_stsp_units_to_s0(net: Any, selected_units: np.ndarray) -> dict[str, Any]:
    if torch is None:
        return {"restore_ok": False, "reset_variables": ""}
    unit_ids = np.asarray(selected_units, dtype=np.int64).reshape(-1)
    layer = net.layer1
    variables: list[str] = []
    with torch.no_grad():
        if unit_ids.size == 0:
            return {"restore_ok": False, "reset_variables": ""}
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            return {"restore_ok": False, "reset_variables": ""}
        h = int(layer.u_pre.shape[-2])
        w = int(layer.u_pre.shape[-1])
        rr = torch.as_tensor(unit_ids // w, device=layer.u_pre.device, dtype=torch.long)
        cc = torch.as_tensor(unit_ids % w, device=layer.u_pre.device, dtype=torch.long)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr = rr[valid]
        cc = cc[valid]
        if int(rr.numel()) == 0:
            return {"restore_ok": False, "reset_variables": ""}
        layer.u_pre[..., rr, cc] = float(layer.stsp_U)
        layer.x_pre[..., rr, cc] = 1.0
        variables.extend(["u", "x"])
        if getattr(layer, "g_e", None) is not None and layer.g_e.ndim >= 4 and int(layer.g_e.shape[-2]) == h and int(layer.g_e.shape[-1]) == w:
            layer.g_e[..., rr, cc] = 0.0
            variables.append("g_e")
    return {"restore_ok": True, "reset_variables": ",".join(variables), "n_reset_units": int(unit_ids.size)}


def _matched_random_unit_mask(pool: np.ndarray, n_units: int, seed: int) -> np.ndarray:
    arr = np.asarray(pool, dtype=bool).reshape(-1)
    out = np.zeros(arr.size, dtype=bool)
    candidates = np.flatnonzero(arr)
    if int(n_units) <= 0 or candidates.size == 0:
        return out
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(candidates, size=min(int(n_units), int(candidates.size)), replace=False)
    out[chosen] = True
    return out


def _route_peak_reentry_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit_set in PERTURBATION_UNIT_SET_ORDER:
        part = df[df.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str).eq(unit_set)] if not df.empty else df
        valid = part[_bool_col(part, "perturbation_ok") & ~_bool_col(part, "insufficient_units")] if not part.empty else part
        loss = pd.to_numeric(valid.get("reentry_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        norm = pd.to_numeric(valid.get("normalized_reentry_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "perturbation_unit_set": unit_set,
                "mean_reentry_loss": float(np.mean(loss)) if loss.size else np.nan,
                "sem_reentry_loss": _sem(loss) if loss.size else np.nan,
                "mean_normalized_reentry_loss": float(np.mean(norm)) if norm.size else np.nan,
                "sem_normalized_reentry_loss": _sem(norm) if norm.size else np.nan,
                "n_trials": int(len(part)),
                "n_valid_trials": int(len(valid)),
                "insufficient_fraction": float(_bool_col(part, "insufficient_units").mean()) if len(part) else np.nan,
                "denominator_choice": "max(abs(reentry_strength_intact), eps)",
            }
        )
    return pd.DataFrame(rows, columns=PANEL_D_ROUTE_PEAK_SUMMARY_COLUMNS)


def _route_peak_reentry_contrast(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric = "normalized_reentry_loss"
    for control in ("route_nonpeak", "nonroute_peak", "random_matched"):
        diff, n_pairs = _paired_unit_set_difference(df, "route_peak", control, metric)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "contrast": f"route_peak_minus_{control}",
                "metric": metric,
                "route_peak_minus_control": diff,
                "route_peak_minus_route_nonpeak": diff if control == "route_nonpeak" else np.nan,
                "route_peak_minus_nonroute_peak": diff if control == "nonroute_peak" else np.nan,
                "route_peak_minus_random": diff if control == "random_matched" else np.nan,
                "route_peak_effect_size": diff,
                "n_valid_pairs": int(n_pairs),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_D_ROUTE_PEAK_CONTRAST_COLUMNS)


def _route_peak_downstream_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit_set in PERTURBATION_UNIT_SET_ORDER:
        part = df[df.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str).eq(unit_set)] if not df.empty else df
        valid = part[_bool_col(part, "perturbation_ok") & ~_bool_col(part, "insufficient_units")] if not part.empty else part
        switch = _bool_col(valid, "output_switch")
        resp = pd.to_numeric(valid.get("response_displacement_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        dec = pd.to_numeric(valid.get("decision_deflection_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "perturbation_unit_set": unit_set,
                "P_output_switch": float(switch.mean()) if len(valid) else np.nan,
                "mean_response_displacement_loss": float(np.mean(resp)) if resp.size else np.nan,
                "sem_response_displacement_loss": _sem(resp) if resp.size else np.nan,
                "mean_decision_deflection_loss": float(np.mean(dec)) if dec.size else np.nan,
                "sem_decision_deflection_loss": _sem(dec) if dec.size else np.nan,
                "n_trials": int(len(part)),
                "n_valid_trials": int(len(valid)),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_E_ROUTE_PEAK_SUMMARY_COLUMNS)


def _route_peak_downstream_contrast(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("output_switch", "response_displacement_loss", "decision_deflection_loss", "output_distribution_JS"):
        for control in ("route_nonpeak", "nonroute_peak", "random_matched"):
            diff, n_pairs = _paired_unit_set_difference(df, "route_peak", control, metric)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": metric,
                    "contrast": f"route_peak_minus_{control}",
                    "route_peak_minus_route_nonpeak": diff if control == "route_nonpeak" else np.nan,
                    "route_peak_minus_nonroute_peak": diff if control == "nonroute_peak" else np.nan,
                    "route_peak_minus_random": diff if control == "random_matched" else np.nan,
                    "n_valid_pairs": int(n_pairs),
                }
            )
    return pd.DataFrame(rows, columns=PANEL_E_ROUTE_PEAK_CONTRAST_COLUMNS)


def _paired_unit_set_difference(df: pd.DataFrame, left: str, right: str, metric: str) -> tuple[float, int]:
    if df.empty or metric not in df.columns:
        return np.nan, 0
    use = df[_bool_col(df, "perturbation_ok") & ~_bool_col(df, "insufficient_units")].copy()
    if use.empty:
        return np.nan, 0
    piv = use.pivot_table(index=["sequence_id", "probe_id"], columns="perturbation_unit_set", values=metric, aggfunc="mean")
    if left not in piv.columns or right not in piv.columns:
        return np.nan, 0
    diff = pd.to_numeric(piv[left], errors="coerce") - pd.to_numeric(piv[right], errors="coerce")
    diff = diff.dropna()
    return (float(diff.mean()) if len(diff) else np.nan, int(len(diff)))


def _route_peak_perturbation_audit(ctx: ExperimentContext, df_d: pd.DataFrame, df_e: pd.DataFrame, *, reason: str) -> pd.DataFrame:
    success = _route_peak_success(df_d, df_e)
    valid = df_d[_bool_col(df_d, "perturbation_ok") & ~_bool_col(df_d, "insufficient_units")] if not df_d.empty else df_d
    return pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "route_peak_perturbation_implemented": bool(ctx.cfg.run_peak_perturbation),
                "route_peak_perturbation_success": bool(success),
                "uses_real_rollout": bool(not _is_proxy_mode(ctx)),
                "proxy_mode": bool(_is_proxy_mode(ctx)),
                "probe_input_unchanged": True,
                "s0_baseline_available": bool(not df_d.empty and pd.to_numeric(df_d.get("reentry_strength_s0", pd.Series(dtype=float)), errors="coerce").notna().any()),
                "intact_sfinal_available": bool(not df_d.empty and pd.to_numeric(df_d.get("reentry_strength_intact", pd.Series(dtype=float)), errors="coerce").notna().any()),
                "route_peak_control_available": bool(_unit_set_valid(df_d, "route_peak")),
                "route_nonpeak_control_available": bool(_unit_set_valid(df_d, "route_nonpeak")),
                "nonroute_peak_control_available": bool(_unit_set_valid(df_d, "nonroute_peak")),
                "random_control_available": bool(_unit_set_valid(df_d, "random_matched")),
                "final_scientific_use": bool(success),
                "allowed_claim_strength": "causal_route_peak_gain" if success else "predictive_peak_amplified_only",
                "n_valid_trials": int(valid[["sequence_id", "probe_id"]].drop_duplicates().shape[0]) if not valid.empty else 0,
                "failure_reason": reason if not success else "",
            }
        ]
    )


def _route_peak_scientific_use_audit(ctx: ExperimentContext, df_d: pd.DataFrame, df_e: pd.DataFrame, *, reason: str) -> pd.DataFrame:
    return _route_peak_perturbation_audit(ctx, df_d, df_e, reason=reason)


def _route_peak_success(df_d: pd.DataFrame, df_e: pd.DataFrame) -> bool:
    if df_d.empty or df_e.empty:
        return False
    return all(_unit_set_valid(df_d, unit_set) for unit_set in PERTURBATION_UNIT_SET_ORDER) and all(_unit_set_valid(df_e, unit_set) for unit_set in PERTURBATION_UNIT_SET_ORDER)


def _unit_set_valid(df: pd.DataFrame, unit_set: str) -> bool:
    if df.empty or "perturbation_unit_set" not in df.columns:
        return False
    part = df[df["perturbation_unit_set"].astype(str).eq(unit_set)]
    return bool(len(part) > 0 and (_bool_col(part, "perturbation_ok") & ~_bool_col(part, "insufficient_units")).any())


def _diagnostic_perturbation_rows_d(ctx: ExperimentContext, r: Any, bank: PeakAmplifiedReentryBank, seq_idx: int, reason: str) -> list[dict[str, Any]]:
    seq_len = int(bank.sequence_meta.iloc[seq_idx]["seq_len"]) if len(bank.sequence_meta) else -1
    return [
        {
            "network_seed": int(ctx.cfg.network_seed),
            "sequence_id": int(r.sequence_id),
            "probe_id": int(r.probe_id),
            "probe_label": int(r.probe_label),
            "seq_len": seq_len,
            "perturbation_unit_set": unit_set,
            "perturbation_condition": f"{unit_set}_reset",
            "perturbation_mode": "reset_u_x_to_s0",
            "state_condition": "diagnostic_unavailable",
            "raw_overlap": float(getattr(r, "raw_overlap", np.nan)),
            "peak_weighted_overlap": float(getattr(r, "peak_weighted_overlap", np.nan)),
            "route_unit_count": 0,
            "peak_unit_count": 0,
            "route_peak_unit_count": 0,
            "route_nonpeak_unit_count": 0,
            "nonroute_peak_unit_count": 0,
            "random_unit_count": 0,
            "insufficient_units": True,
            "reentry_strength_intact": np.nan,
            "reentry_strength_perturbed": np.nan,
            "reentry_strength_s0": np.nan,
            "reentry_loss": np.nan,
            "normalized_reentry_loss": np.nan,
            "prediction_intact": -1,
            "prediction_perturbed": -1,
            "prediction_s0": -1,
            "first_fire_time_intact": -1,
            "first_fire_time_perturbed": -1,
            "first_fire_time_s0": -1,
            "restore_ok": False,
            "perturbation_ok": False,
            "denominator_choice": "max(abs(reentry_strength_intact), eps)",
            "reset_variables": "",
            "probe_input_unchanged": True,
            "failure_reason": reason,
        }
        for unit_set in PERTURBATION_UNIT_SET_ORDER
    ]


def _diagnostic_perturbation_rows_e(ctx: ExperimentContext, r: Any, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "network_seed": int(ctx.cfg.network_seed),
            "sequence_id": int(r.sequence_id),
            "probe_id": int(r.probe_id),
            "probe_label": int(r.probe_label),
            "perturbation_unit_set": unit_set,
            "perturbation_condition": f"{unit_set}_reset",
            "response_displacement_intact": np.nan,
            "response_displacement_perturbed": np.nan,
            "response_displacement_s0": np.nan,
            "response_displacement_loss": np.nan,
            "decision_deflection_intact": np.nan,
            "decision_deflection_perturbed": np.nan,
            "decision_deflection_s0": np.nan,
            "decision_deflection_loss": np.nan,
            "prediction_intact": -1,
            "prediction_perturbed": -1,
            "prediction_s0": -1,
            "output_switch": False,
            "output_distribution_JS": np.nan,
            "perturbation_ok": False,
            "insufficient_units": True,
            "failure_reason": reason,
        }
        for unit_set in PERTURBATION_UNIT_SET_ORDER
    ]


def _output_distribution_row(ctx: ExperimentContext, r: Any, unit_set: str, intact_vec: np.ndarray, condition_vec: np.ndarray, *, js: float) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(r.sequence_id),
        "probe_id": int(r.probe_id),
        "perturbation_unit_set": unit_set,
        "output_distribution_JS": float(js),
        "intact_entropy": _entropy_from_logits(intact_vec),
        "condition_entropy": _entropy_from_logits(condition_vec),
    }


def _softmax_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=float)
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    return exp / max(float(np.sum(exp)), 1e-12)


def _entropy_from_logits(values: np.ndarray) -> float:
    p = _softmax_np(values)
    if p.size == 0:
        return np.nan
    return float(-np.sum(p * np.log2(np.maximum(p, 1e-12))))


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = _softmax_np(left)
    q = _softmax_np(right)
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return np.nan
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12))))
    kl_qm = float(np.sum(q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12))))
    return float(0.5 * (kl_pm + kl_qm))


def write_global_mechanism_metadata(ctx: ExperimentContext) -> None:
    payload = {
        "figure_chain": [
            "Fig.1 functional STSP substrate",
            "Fig.2 two-item fused state",
            "Fig.3 multi-item peak-valley landscape",
            "Fig.4 overlap-aligned re-entry route",
            "Fig.5 local support-to-competition conversion",
            "Fig.6 peak-amplified overlap route",
        ],
        "allowed_language": [
            "overlap provides route",
            "peaks provide gain",
            "overlap provides route; peaks provide gain",
            "peak-modulated overlap re-entry",
            "peak-amplified overlap-aligned re-entry",
        ],
        "forbidden_language": [
            "peaks replace overlap",
            "peaks alone gate re-entry",
            "causal control without perturbation",
        ],
        "mechanism_statement": "overlap provides route; peaks provide gain; not peaks replace overlap",
        "peak_perturbation_implemented": bool(ctx.completed_modules.get("peak_perturbation")),
        "peak_perturbation_successful": bool(_peak_perturbation_claim_upgrade_allowed(ctx)),
        "allowed_claim_strength": _claim_strength(ctx),
    }
    _write_json(payload, ctx.raw_dir / "panel_f_global_mechanism_metadata.json")
    ctx.output_files["panel_f_global_mechanism_metadata"] = "data/raw/panel_f_global_mechanism_metadata.json"


def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    debug_specs = [
        ("fig6_debug_peak_source_attribution", ctx.metrics_dir / "panel_a_peak_source_attribution_summary.csv", "relative_position_from_end", "mean_peak_loss_fraction"),
        ("fig6_debug_peak_update_history", ctx.metrics_dir / "panel_b_peak_update_history_summary.csv", "group", "mean_update_count"),
        ("fig6_debug_peak_input_overlap_origin", ctx.metrics_dir / "panel_c_peak_input_overlap_summary.csv", "overlap_window", "mean_dice"),
        ("fig6_debug_real_reentry", ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv", "peak_weighted_overlap", "reentry_strength"),
        ("fig6_debug_real_downstream", ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv", "peak_weighted_overlap", "metric_value"),
        ("fig6_debug_s11_update_group_enrichment", ctx.metrics_dir / "supp_s11_peak_update_group_enrichment.csv", "update_group", "P_peak"),
        ("fig6_debug_s11_recent_overlap_window", ctx.metrics_dir / "supp_s11_recent_overlap_window_robustness.csv", "recent_k", "dice_peak_overlap"),
        ("fig6_debug_s12_matched_peak_overlap_contrast", ctx.metrics_dir / "supp_s12_raw_overlap_matched_peak_overlap_contrast.csv", "matched_set_id", "reentry_high_minus_low"),
        ("fig6_debug_s12_downstream_metric_breakdown", ctx.metrics_dir / "supp_s12_downstream_metric_breakdown.csv", "downstream_metric", "beta_peak_weighted_overlap"),
        ("fig6_debug_real_rollout_audit", ctx.metrics_dir / "panel_de_real_rollout_scientific_use_audit.csv", "module", "final_scientific_use"),
        ("fig6_debug_s12_peak_perturbation", ctx.metrics_dir / "supp_s12_peak_perturbation_summary.csv", "metric", "overlap_minus_control_reduction"),
    ]
    for name, path, x_col, y_col in debug_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if x_col not in df.columns or y_col not in df.columns:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=160)
        x = df[x_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        if pd.api.types.is_numeric_dtype(x):
            ax.scatter(pd.to_numeric(x, errors="coerce"), y, s=10)
        else:
            order = list(dict.fromkeys(map(str, x.tolist())))
            ax.scatter([order.index(str(v)) for v in x], y, s=10)
            ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        save_figure_all_formats(fig, ctx.debug_dir / name)
        plt.close(fig)
    _save_global_debug_figure(ctx)


def _sequence_support_maps(
    ctx: ExperimentContext,
    image_ids: Sequence[int],
    masks: np.ndarray,
    count_flat: np.ndarray,
    last_flat: np.ndarray,
    seq_len: int,
    *,
    encode_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Mapping[str, Any]]]:
    proxy_baseline, proxy_final = _proxy_support_maps(ctx, masks, count_flat, last_flat, seq_len)
    if ctx.net is None or ctx.encoder is None or torch is None:
        return proxy_baseline, proxy_final, {}
    try:
        spikes = _encode_sequence_cached(ctx, image_ids, ctx.cfg.sample_steps, encode_cache)
        seq_len_t, _, channels, height, width = spikes.shape
        zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
        prepare_network_state(ctx.net, 1, channels, height, width)
        with torch.no_grad():
            for _ in range(seq_len_t):
                for _ in range(ctx.cfg.sample_steps + ctx.cfg.delay_steps):
                    _step_network_once(ctx.net, zero_input, 0)
        baseline = _support_from_net(ctx.net)
        prepare_network_state(ctx.net, 1, channels, height, width)
        current_time = 0
        with torch.no_grad():
            for idx in range(seq_len_t):
                for t in range(ctx.cfg.sample_steps):
                    current_time = _step_network_once(ctx.net, spikes[idx : idx + 1, t, ...], current_time)
                for _ in range(ctx.cfg.delay_steps):
                    current_time = _step_network_once(ctx.net, zero_input, current_time)
        final = _support_from_net(ctx.net)
        boundary = snapshot_boundary_state(ctx.net)
        return _resize_array(baseline, 28, 28).astype(np.float32), _resize_array(final, 28, 28).astype(np.float32), boundary
    except Exception as exc:
        ctx.warnings.append(f"Network sequence rollout failed; using image-driven support proxy: {exc}")
        return proxy_baseline, proxy_final, {}


def _leave_one_out_support_map(ctx: ExperimentContext, image_ids: Sequence[int], removed_idx: int, *, encode_cache: dict[tuple[Any, ...], Any] | None = None) -> np.ndarray:
    masks = np.stack([_foreground_mask(ctx.dataset, image_id, ctx.cfg.foreground_threshold) for image_id in image_ids], axis=0)
    keep = masks.copy()
    if 0 <= int(removed_idx) < len(keep):
        keep[int(removed_idx)] = False
    exposure = keep.reshape(len(image_ids), -1).astype(np.float32)
    count = exposure.sum(axis=0)
    last = np.zeros_like(count, dtype=np.int16)
    for pos in range(len(image_ids)):
        active = exposure[pos] > 0
        last[active] = pos + 1
    if ctx.net is None or ctx.encoder is None or torch is None:
        _, final = _proxy_support_maps(ctx, keep, count, last, len(image_ids))
        return final.astype(np.float32)
    try:
        spikes = _encode_sequence_cached(ctx, image_ids, ctx.cfg.sample_steps, encode_cache)
        seq_len_t, _, channels, height, width = spikes.shape
        zero_input = torch.zeros((1, channels, height, width), device=ctx.device)
        prepare_network_state(ctx.net, 1, channels, height, width)
        current_time = 0
        with torch.no_grad():
            for idx in range(seq_len_t):
                for t in range(ctx.cfg.sample_steps):
                    input_t = zero_input if idx == int(removed_idx) else spikes[idx : idx + 1, t, ...]
                    current_time = _step_network_once(ctx.net, input_t, current_time)
                for _ in range(ctx.cfg.delay_steps):
                    current_time = _step_network_once(ctx.net, zero_input, current_time)
        return _resize_array(_support_from_net(ctx.net), 28, 28).astype(np.float32)
    except Exception as exc:
        ctx.warnings.append(f"Leave-one-out real replay failed; using image-driven support proxy: {exc}")
        _, final = _proxy_support_maps(ctx, keep, count, last, len(image_ids))
        return final.astype(np.float32)


def _run_real_probe_from_condition(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundary: Mapping[str, Mapping[str, Any]] | None,
    condition: str,
    *,
    probe_spikes: Any | None = None,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    if ctx.net is None or ctx.encoder is None or torch is None:
        raise RuntimeError("real probe rollout requested without net/encoder")
    spikes = probe_spikes
    if spikes is None:
        spikes = _encode_sequence_cached(ctx, [int(probe_image_id)], ctx.cfg.probe_steps, {})
    _, steps, channels, height, width = spikes.shape
    if boundary:
        _restore_boundary_state(ctx.net, boundary)
    else:
        prepare_network_state(ctx.net, 1, channels, height, width)
    if condition == "S0" and boundary is None:
        pass
    with torch.no_grad():
        ctx.net.layer3.reset_decision_state()
        if hasattr(ctx.net.layer3, "v_mem"):
            ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
        if hasattr(ctx.net.layer3, "lateral_inh"):
            ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
    traces: list[np.ndarray] = []
    current_time = 0
    with torch.no_grad():
        for t in range(int(steps)):
            s3 = _step_network_once_with_l3(ctx.net, spikes[:, t, ...], current_time, force_l3_time=t)
            traces.append(s3.detach().to(torch.float32).view(1, -1).cpu().numpy()[0])
            current_time += 1
    pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, 1)
    trace = np.stack(traces, axis=0).astype(np.float32) if traces else np.zeros((0, 1), dtype=np.float32)
    vector = _class_readout_vector_from_trace(ctx.net, trace)
    return trace, int(pred[0].item()), int(fire[0].item()), vector.astype(np.float32)


def _run_real_probe_conditions_batch(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundaries: Sequence[Mapping[str, Mapping[str, Any]] | None],
    condition_names: Sequence[str],
) -> dict[str, tuple[np.ndarray, int, int, np.ndarray]]:
    if ctx.cfg.enable_probe_batch:
        ctx.warnings.append("Fig.6 probe batch helper is scaffolded; falling back to order-preserving per-condition rollout.")
    cache: dict[tuple[Any, ...], Any] = {}
    probe_spikes = _encode_sequence_cached(ctx, [int(probe_image_id)], ctx.cfg.probe_steps, cache)
    out: dict[str, tuple[np.ndarray, int, int, np.ndarray]] = {}
    for condition, boundary in zip(condition_names, boundaries):
        out[str(condition)] = _run_real_probe_from_condition(ctx, int(probe_image_id), boundary, str(condition), probe_spikes=probe_spikes)
    return out


def _restore_boundary_state(net, boundary: Mapping[str, Mapping[str, Any]]) -> None:
    if torch is None:
        return
    with torch.no_grad():
        for layer_key, state in boundary.items():
            layer = getattr(net, layer_key)
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key in state and hasattr(layer, attr):
                    target = getattr(layer, attr)
                    target.copy_(state[src_key].to(device=target.device, dtype=target.dtype))
            if "inh_trace" in state and hasattr(layer, "lateral_inh"):
                target = layer.lateral_inh.inh_trace
                target.copy_(state["inh_trace"].to(device=target.device, dtype=target.dtype))
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))


def _step_network_once_with_l3(net: Any, input_t: Any, current_time: int, *, force_l3_time: int | None = None, stsp_mode: str = "dynamic"):
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    s3, _ = net.layer3.forward_step(s2p, current_time if force_l3_time is None else int(force_l3_time), training=False, monitor=False, stsp_mode=stsp_mode)
    return s3


def _proxy_support_maps(ctx: ExperimentContext, masks: np.ndarray, count_flat: np.ndarray, last_flat: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    count = count_flat.reshape(28, 28)
    last = last_flat.reshape(28, 28)
    recent = np.where(last > 0, 1.0 / (1.0 + seq_len - last), 0.0)
    image_mean = masks.mean(axis=0).astype(float) if len(masks) else np.zeros((28, 28), dtype=float)
    baseline = 0.18 + 0.05 * _blur3(image_mean)
    final = baseline + 0.18 * _normalize(count) + 0.22 * _normalize(recent) + 0.30 * _normalize(count * recent)
    final += np.random.default_rng(int(ctx.cfg.network_seed) + seq_len).normal(0, 0.01, final.shape)
    return baseline.astype(np.float32), np.clip(final, 0, 1.5).astype(np.float32)


def _support_from_net(net: Any) -> np.ndarray:
    layer = net.layer1
    if getattr(layer, "u_pre", None) is not None and getattr(layer, "x_pre", None) is not None:
        support = (layer.u_pre.detach().to(torch.float32) * layer.x_pre.detach().to(torch.float32)).mean(dim=1)[0].cpu().numpy()
    else:
        support = np.zeros((28, 28), dtype=np.float32)
    return np.asarray(support, dtype=np.float32)


def _step_network_once(net: Any, input_t: Any, current_time: int, *, stsp_mode: str = "dynamic") -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1


def _write_config_files(ctx: ExperimentContext) -> None:
    cfg = ctx.cfg
    _write_json(asdict(cfg), ctx.config_dir / "run_config.json")
    _write_json(asdict(cfg), ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "figure": FIGURE_ID,
            "fig6_design_version": FIG6_DESIGN_VERSION,
            "main_panels": MAIN_PANELS,
            "main_claim": MAIN_CLAIM,
            "mechanism_boundary": MECHANISM_BOUNDARY,
            "supplement_plan": SUPPLEMENT_PLAN,
            "main_required_outputs": MAIN_REQUIRED_OUTPUTS,
            "supplementary_outputs": SUPPLEMENTARY_OUTPUTS,
            "optional_supplementary_outputs": OPTIONAL_SUPPLEMENTARY_OUTPUTS,
            "claim_boundary": "A-C explain peak origin and route alignment; D-E require successful real route-peak perturbation for causal route-peak gain language.",
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "conditions": list(UPDATE_GROUPS),
            "recent_window": cfg.recent_window,
            "multi_update_threshold": cfg.multi_update_threshold,
            "mechanism_summary": {
                "overlap": "route",
                "peaks": "gain along overlap route",
                "primary_claim": "peak-weighted overlap amplifies later overlap-aligned re-entry and downstream dynamics",
            },
            "peak_weighted_overlap_definition": {
                "raw_overlap": "shared sample-probe overlap route",
                "peak_weighted_overlap": "raw overlap weighted by final STSP peak support / peak mask",
                "interpretation": "gain along overlap route, not route replacement",
            },
            "main_panels": {
                "A": "leave-one-item-out source attribution",
                "B": "peak update history: repeated and recent updates",
                "C": "peak alignment with recent input-overlap maps",
                "D": "route-peak reset re-entry loss",
                "E": "route-peak reset downstream output switch",
                "F": "mechanism schematic",
            },
            "claim_strength_rules": {
                "without_peak_perturbation": "predictive_peak_amplified_only",
                "with_successful_route_peak_perturbation": "causal_route_peak_gain",
                "forbidden": MECHANISM_BOUNDARY["forbidden_claims"],
            },
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json({"state_variable": "g = u * x", "peak_q": cfg.peak_q, "definition": "top fraction of positive delta_support units", "peak_origin_design_version": FIG6_DESIGN_VERSION}, ctx.config_dir / "peak_definition_spec.json")
    _write_json({"models": list(MODEL_NAMES), "target": ["delta_support", "final_support"], "cv": "deterministic K-fold over units", "main_status": "supplement_only"}, ctx.config_dir / "update_recency_model_spec.json")
    _write_json(
        {
            "leave_one_out_mode": cfg.leave_one_out_mode,
            "blank_same_timing": cfg.leave_one_out_mode == "blank_same_timing",
            "support_loss_definition": "max(G_full - G_minus_i, 0)",
            "peak_loss_fraction_definition": "peak_loss_i / sum_j peak_loss_j",
            "proxy_mode_handling": "proxy outputs are pipeline validation only and not final scientific evidence",
        },
        ctx.config_dir / "peak_source_attribution_spec.json",
    )
    _write_json(
        {
            "peak_definition": "top positive delta_support units by peak_q",
            "recent_windows": list(cfg.recent_overlap_windows),
            "multi_update_threshold": int(cfg.multi_update_threshold),
            "groups_summarized": ["peak", "nonpeak_control", "prior_updated_nonpeak"],
        },
        ctx.config_dir / "peak_update_history_spec.json",
    )
    _write_json(
        {
            "recent_overlap_windows": list(cfg.recent_overlap_windows),
            "high_overlap_mask_definition": "top n_peak positive-overlap units by overlap count",
            "similarity_metrics": ["dice", "jaccard", "peak_coverage", "overlap_precision", "cosine_delta_support_overlap_count"],
        },
        ctx.config_dir / "peak_input_overlap_origin_spec.json",
    )
    _write_json({"raw_overlap": "later probe input intersect prior_updated route", "peak_weighted_overlap": "D/E trial interface: final support weighted overlap within prior_updated peak route", "interpretation": "gain along overlap route, not route replacement", "main_status": "trial_interface_not_fig6c_origin_analysis"}, ctx.config_dir / "peak_weighted_overlap_spec.json")
    _write_json(
        {
            "state_conditions": list(cfg.real_reentry_reference_conditions),
            "reference_state_definition": "S_final restores sequence boundary; S0 resets network to baseline if no explicit S0 boundary is available",
            "metrics": ["normalized_reentry_loss", "P_output_switch", "response_displacement_loss", "decision_deflection_loss"],
            "proxy_mode_not_final": True,
        },
        ctx.config_dir / "real_reentry_rollout_spec.json",
    )
    _write_json(
        {
            "implemented_by_default": False,
            "enabled_flag": "--run-peak-perturbation",
            "main_status": "required_for_causal_DE",
            "unit_sets": list(PERTURBATION_UNIT_SET_ORDER),
            "reset_variables": "u/x reset to S0 values; g_e zeroed when spatially compatible",
            "probe_input_modified": False,
            "claim_strength_without_success": "predictive_peak_amplified_only",
        },
        ctx.config_dir / "peak_perturbation_spec.json",
    )


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main = [ctx.seed_dir / rel for rel in MAIN_REQUIRED_OUTPUTS]
    required_supp = [ctx.seed_dir / rel for rel in SUPPLEMENTARY_OUTPUTS] if ctx.cfg.run_supplement else []
    required_optional = [ctx.seed_dir / rel for rel in OPTIONAL_SUPPLEMENTARY_OUTPUTS] if ctx.cfg.run_peak_perturbation else []
    proxy_mode = _main_proxy_mode(ctx)
    audit_path = ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    main_claim_allowed = bool(not audit.empty and audit["allowed_claim_strength"].astype(str).eq("causal_route_peak_gain").all())
    real_rollout_available = bool(not audit.empty and audit["uses_real_rollout"].astype(str).str.lower().isin({"true", "1", "yes"}).all())
    final_scientific_use = bool(not audit.empty and audit["final_scientific_use"].astype(str).str.lower().isin({"true", "1", "yes"}).all())
    if proxy_mode and "D/E are proxy-mode outputs and must not be used as final scientific evidence." not in ctx.warnings:
        ctx.warnings.append("D/E are proxy-mode outputs and must not be used as final scientific evidence.")
    if not final_scientific_use and "Fig.6D/E outputs are not cleared for final scientific claims." not in ctx.warnings:
        ctx.warnings.append("Fig.6D/E outputs are not cleared for final scientific claims.")
    summary = {
        "figure": FIGURE_ID,
        "fig6_design_version": FIG6_DESIGN_VERSION,
        "main_panels": MAIN_PANELS,
        "main_claim": MAIN_CLAIM,
        "supplement_plan": SUPPLEMENT_PLAN,
        "mechanism_boundary": MECHANISM_BOUNDARY,
        "old_multi_recent_enrichment_demoted_from_main": True,
        "update_recency_model_demoted_to_supplement": True,
        "formula_proxy_reentry_removed_from_main": True,
        "main_a_method": "leave_one_item_out_blank_same_timing",
        "main_b_method": "peak_conditional_update_history",
        "main_c_method": "recent_input_overlap_origin",
        "main_d_method": "route_peak_reset_reentry_loss",
        "main_e_method": "route_peak_reset_downstream_output_switch",
        "proxy_mode": bool(proxy_mode),
        "real_rollout_available": bool(real_rollout_available),
        "final_scientific_use": bool(final_scientific_use),
        "main_claim_allowed": bool(main_claim_allowed),
        "claim_strength": _claim_strength(ctx),
        "peak_perturbation_status": _peak_perturbation_status(ctx),
        "peak_perturbation_claim_upgrade_allowed": _peak_perturbation_claim_upgrade_allowed(ctx),
        "forbidden_claims": MECHANISM_BOUNDARY["forbidden_claims"],
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_sequences": int(ctx.n_sequences),
        "n_probe_candidates": int(ctx.n_probe_candidates),
        "n_matched_groups": int(ctx.n_matched_groups),
        "peak_definition": {"state_variable": "g", "peak_q": float(ctx.cfg.peak_q), "positive_delta_only": True},
        "update_recency_model": {"models": list(MODEL_NAMES), "cv": "K-fold over units", "main_status": "supplement_only"},
        "peak_weighted_overlap_definition": {"raw_overlap": "route", "peak_weighted_overlap": "gain along overlap route", "main_status": "D/E trial interface"},
        "peak_perturbation_implemented": bool(ctx.completed_modules.get("peak_perturbation")),
        "peak_perturbation_successful": bool(_peak_perturbation_claim_upgrade_allowed(ctx)),
        "fig6_route_peak_perturbation": _summary_route_peak_perturbation(ctx),
        "allowed_claim_strength": _claim_strength(ctx),
        "conditions": list(UPDATE_GROUPS),
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": bool(all(path.exists() for path in required_main) and main_claim_allowed),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
        "missing_for_optional_supplementary": [_rel(path, ctx.seed_dir) for path in required_optional if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    _write_json(_artifact_manifest(ctx), ctx.seed_dir / "artifact_manifest.json")
    ctx.output_files["summary"] = "summary.json"
    return summary


def _artifact_manifest(ctx: ExperimentContext) -> dict[str, Any]:
    files = []
    for path in ctx.seed_dir.rglob("*"):
        if path.is_file():
            files.append({"path": _rel(path, ctx.seed_dir), "size_bytes": int(path.stat().st_size)})
    return {"figure": FIGURE_ID, "files": sorted(files, key=lambda x: x["path"])}


def _summary_route_peak_perturbation(ctx: ExperimentContext) -> dict[str, Any]:
    audit_path = ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv"
    contrast_path = ctx.metrics_dir / "panel_d_route_peak_reentry_loss_contrast.csv"
    summary_path = ctx.metrics_dir / "panel_d_route_peak_reentry_loss_summary.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    contrast = pd.read_csv(contrast_path) if contrast_path.exists() else pd.DataFrame()
    d_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    success = bool(not audit.empty and _bool_col(audit, "route_peak_perturbation_success").any())
    valid_trials = int(pd.to_numeric(audit.get("n_valid_trials", pd.Series([0])), errors="coerce").fillna(0).max()) if not audit.empty else 0
    contrasts: dict[str, float | None] = {}
    if not contrast.empty:
        for row in contrast.itertuples(index=False):
            contrasts[str(getattr(row, "contrast", ""))] = float(getattr(row, "route_peak_minus_control", np.nan)) if np.isfinite(float(getattr(row, "route_peak_minus_control", np.nan))) else None
    means: dict[str, float | None] = {}
    if not d_summary.empty:
        for row in d_summary.itertuples(index=False):
            value = float(getattr(row, "mean_normalized_reentry_loss", np.nan))
            means[str(getattr(row, "perturbation_unit_set", ""))] = value if np.isfinite(value) else None
    return {
        "enabled": bool(ctx.cfg.run_peak_perturbation),
        "success": success,
        "allowed_claim_strength": "causal_route_peak_gain" if success else "predictive_peak_amplified_only",
        "n_valid_trials": valid_trials,
        "route_peak_minus_controls": contrasts,
        "mean_normalized_reentry_loss": means,
    }


def _write_run_log(ctx: ExperimentContext) -> None:
    ctx.run_log.append(f"{_now()} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def _load_dataset_or_proxy(dataset_root: str, split: str, seed: int, warnings: list[str]):
    try:
        return load_mnist_skeleton_dataset(dataset_root, split)
    except Exception as exc:
        warnings.append(f"MNIST skeleton dataset load failed; using deterministic proxy dataset: {exc}")
        return ProxyDataset(seed=seed)


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = _rel(path, ctx.seed_dir)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _copy_csv_if_exists(src: Path, dst: Path, ctx: ExperimentContext) -> bool:
    df = _read_csv_if_exists(src)
    if df is None:
        return False
    _save_csv(ctx, df, dst)
    return True


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    paths = {
        "config": seed_dir / "config",
        "trial_specs": seed_dir / "data" / "trial_specs",
        "raw": seed_dir / "data" / "raw",
        "metrics": seed_dir / "data" / "metrics",
        "debug": seed_dir / "debug_figures",
        "meta": seed_dir / "meta",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    if output_root.name.startswith("seed_"):
        return output_root
    return output_root / f"seed_{int(network_seed):03d}"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _images_for_ids(dataset: Any, image_ids: Iterable[int]):
    if torch is None:
        raise RuntimeError("PyTorch is required for encoded network rollouts.")
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)


def _encode_sequence_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, cache: dict[tuple[Any, ...], Any] | None) -> Any:
    ids = tuple(int(v) for v in image_ids)
    key = ("sequence", ids, int(steps), str(ctx.device))
    if cache is None:
        cache = {}
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]


def _to_tensor(image: np.ndarray):
    if torch is not None:
        return torch.as_tensor(image, dtype=torch.float32)
    return np.asarray(image, dtype=np.float32)


def _image_array(dataset: Any, image_id: int) -> np.ndarray:
    image = dataset[int(image_id)][0]
    if torch is not None and hasattr(image, "detach"):
        arr = image.detach().cpu().to(torch.float32).squeeze().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32).squeeze()
    return np.asarray(arr, dtype=np.float32)


def _foreground_mask(dataset: Any, image_id: int, threshold: float) -> np.ndarray:
    return _image_array(dataset, image_id) > float(threshold)


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
    d_path = ctx.metrics_dir / "panel_d_peak_weighted_reentry_metrics.csv"
    e_path = ctx.metrics_dir / "panel_e_peak_weighted_downstream_metrics.csv"
    legacy_d_path = ctx.metrics_dir / "panel_d_real_reentry_metrics.csv"
    legacy_e_path = ctx.metrics_dir / "panel_e_real_downstream_metrics.csv"
    values = []
    for path in (d_path, e_path, legacy_d_path, legacy_e_path):
        if path.exists():
            df = pd.read_csv(path)
            values.append(_df_all_proxy(df))
    return bool(values and any(values))


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
    probe = _image_array(ctx.dataset, int(target["probe_image_id"]))
    probe_mask = probe > float(ctx.cfg.foreground_threshold)
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
    probe = _image_array(ctx.dataset, int(target["probe_image_id"]))
    probe_mask = probe > float(ctx.cfg.foreground_threshold)
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


def _proxy_l3_trace(ctx: ExperimentContext, dynamic_strength: float, static_strength: float, label: int, rng: np.random.Generator) -> np.ndarray:
    steps = max(4, int(ctx.cfg.probe_steps))
    t = np.linspace(0, 1, steps)
    trace = rng.normal(0, 0.02, (steps, 10))
    trace[:, int(label) % 10] += static_strength + (dynamic_strength - static_strength) * (1 - np.exp(-5 * t))
    trace[:, (int(label) + 1) % 10] += 0.25 * static_strength
    return trace.astype(np.float32)


def _proxy_probe_rollout_pair(ctx: ExperimentContext, r: Any) -> tuple[np.ndarray, np.ndarray, int, int, int, int, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(getattr(r, "candidate_seed", ctx.cfg.network_seed)))
    dynamic_strength = float(0.55 * r.raw_overlap + 0.95 * r.peak_weighted_overlap + 0.05 * r.visual_similarity + rng.normal(0.0, 0.015))
    static_strength = float(0.35 * r.raw_overlap + 0.20 * r.peak_weighted_overlap + rng.normal(0.0, 0.015))
    final_trace = _proxy_l3_trace(ctx, dynamic_strength, static_strength, int(r.probe_label), rng)
    s0_trace = _proxy_l3_trace(ctx, static_strength, static_strength, int(r.probe_label), rng)
    final_vector = final_trace.sum(axis=0).astype(np.float32)
    s0_vector = s0_trace.sum(axis=0).astype(np.float32)
    final_pred = int(np.argmax(final_vector)) if final_vector.size else -1
    s0_pred = int(np.argmax(s0_vector)) if s0_vector.size else -1
    final_fire = _first_nonzero_step(final_trace)
    s0_fire = _first_nonzero_step(s0_trace)
    return final_trace, s0_trace, final_pred, s0_pred, final_fire, s0_fire, final_vector, s0_vector


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


def compute_supp_update_recency_support_model(ctx: ExperimentContext, unit_df: pd.DataFrame) -> None:
    metric_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    if unit_df.empty:
        _save_csv(ctx, pd.DataFrame(columns=PANEL_B_METRIC_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_metrics.csv")
        _save_csv(ctx, pd.DataFrame(columns=PANEL_B_COEF_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_coefficients.csv")
        return
    for sequence_id, part in unit_df.groupby("sequence_id", sort=True):
        y_delta = pd.to_numeric(part["delta_support"], errors="coerce").to_numpy(dtype=float)
        y_final = pd.to_numeric(part["final_support"], errors="coerce").to_numpy(dtype=float)
        features = {
            "baseline_support": np.zeros(len(part), dtype=float),
            "update_count": pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float),
            "recency": -pd.to_numeric(part["time_since_last_update"], errors="coerce").to_numpy(dtype=float),
            "overlap": (pd.to_numeric(part["update_count"], errors="coerce").to_numpy(dtype=float) > 0).astype(float),
        }
        features["update_x_recency"] = features["update_count"] * features["recency"]
        feature_matrix = {
            "baseline_only": ["baseline_support"],
            "update_only": ["update_count"],
            "recency_only": ["recency"],
            "overlap_only": ["overlap"],
            "update_plus_recency": ["update_count", "recency"],
            "update_times_recency": ["update_count", "recency", "update_x_recency"],
        }
        for target_name, y in (("delta_support", y_delta), ("final_support", y_final)):
            model_fits: dict[str, float] = {}
            for model_name, cols in feature_matrix.items():
                x = np.column_stack([np.asarray(features[col], dtype=float) for col in cols])
                fit = _fit_ols(x, y)
                cv_r2 = _cv_r2(x, y, n_folds=5)
                model_fits[model_name] = float(fit["r2"])
                metric_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(sequence_id),
                        "layer": PRIMARY_LAYER,
                        "state_variable": STATE_VARIABLE,
                        "target": target_name,
                        "model_name": model_name,
                        "r2": float(fit["r2"]),
                        "cv_r2": float(cv_r2),
                        "auc_if_binary": float("nan"),
                        "delta_r2_vs_overlap_only": float(fit["r2"] - model_fits.get("overlap_only", np.nan)),
                        "delta_r2_vs_update_only": float(fit["r2"] - model_fits.get("update_only", np.nan)),
                        "delta_r2_vs_recency_only": float(fit["r2"] - model_fits.get("recency_only", np.nan)),
                        "n_units": int(len(part)),
                    }
                )
                for coef_name, coef_value, se, p in zip(["intercept"] + cols, fit["beta"], fit["se"], fit["p"]):
                    coef_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "model_name": model_name,
                            "coefficient_name": coef_name,
                            "coefficient_value": float(coef_value),
                            "standardized_coefficient": float(_standardized_coef(coef_value, x, y, coef_name, cols)),
                            "p_value": float(p),
                            "notes": f"supplement-only; target={target_name}; sequence_id={int(sequence_id)}",
                        }
                    )
    _save_csv(ctx, pd.DataFrame(metric_rows, columns=PANEL_B_METRIC_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_metrics.csv")
    _save_csv(ctx, pd.DataFrame(coef_rows, columns=PANEL_B_COEF_COLUMNS), ctx.metrics_dir / "supp_update_recency_support_model_coefficients.csv")


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


def _save_global_debug_figure(ctx: ExperimentContext) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 1.8), dpi=160)
    ax.axis("off")
    ax.text(0.05, 0.65, "Overlap provides route", fontsize=9, weight="bold")
    ax.text(0.55, 0.65, "Peaks provide gain", fontsize=9, weight="bold")
    ax.annotate("", xy=(0.50, 0.65), xytext=(0.35, 0.65), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.05, 0.25, "Predictive unless peak perturbation succeeds", fontsize=7)
    save_figure_all_formats(fig, ctx.debug_dir / "fig6_debug_global_mechanism")
    plt.close(fig)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.6 peak-amplified overlap re-entry experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-sequence-bank", action="store_true")
    parser.add_argument("--run-peak-source-attribution", action="store_true")
    parser.add_argument("--run-peak-update-history", action="store_true")
    parser.add_argument("--run-peak-input-overlap-origin", action="store_true")
    parser.add_argument("--run-real-reentry-rollout", action="store_true")
    parser.add_argument("--run-real-downstream-metrics", action="store_true")
    parser.add_argument("--run-peak-enrichment", action="store_true")
    parser.add_argument("--run-update-recency-model", action="store_true")
    parser.add_argument("--run-peak-weighted-overlap", action="store_true")
    parser.add_argument("--run-reentry-prediction", action="store_true")
    parser.add_argument("--run-downstream-prediction", action="store_true")
    parser.add_argument("--run-peak-perturbation", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequence-lengths", default="3,5,7,10")
    parser.add_argument("--primary-sequence-length", type=int, default=7)
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--num-probe-candidates-per-sequence", type=int, default=8)
    parser.add_argument("--peak-q", type=float, default=0.20)
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--multi-update-threshold", type=int, default=2)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--n-matched-groups", type=int, default=100)
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--recent-overlap-windows", default="2,3,4,5")
    parser.add_argument("--leave-one-out-mode", default="blank_same_timing", choices=["blank_same_timing"])
    parser.add_argument("--real-rollout-required-for-main", dest="real_rollout_required_for_main", action="store_true", default=True)
    parser.add_argument("--allow-proxy-main", dest="real_rollout_required_for_main", action="store_false")
    parser.add_argument("--save-full-traces", action="store_true")
    parser.add_argument("--no-save-l3-trace", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-probe-batch", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> Fig6Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    seq_lengths = tuple(int(v) for v in str(args.sequence_lengths).split(",") if str(v).strip())
    recent_windows = tuple(int(v) for v in str(args.recent_overlap_windows).split(",") if str(v).strip())
    return Fig6Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sequence_lengths=seq_lengths,
        primary_sequence_length=int(args.primary_sequence_length),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        num_sequences=4 if smoke else int(args.num_sequences),
        num_probe_candidates_per_sequence=2 if smoke else int(args.num_probe_candidates_per_sequence),
        peak_q=float(args.peak_q),
        recent_window=int(args.recent_window),
        multi_update_threshold=int(args.multi_update_threshold),
        n_null=8 if smoke else int(args.n_null),
        n_matched_groups=4 if smoke else int(args.n_matched_groups),
        foreground_threshold=float(args.foreground_threshold),
        save_full_traces=bool(args.save_full_traces),
        save_l3_trace=not bool(args.no_save_l3_trace),
        save_spike_cache=bool(args.save_spike_cache),
        run_sequence_bank=run_all or bool(args.run_sequence_bank),
        run_peak_source_attribution=run_all or bool(args.run_peak_source_attribution),
        run_peak_update_history=run_all or bool(args.run_peak_update_history),
        run_peak_input_overlap_origin=run_all or bool(args.run_peak_input_overlap_origin),
        run_real_reentry_rollout=run_all or bool(args.run_real_reentry_rollout),
        run_real_downstream_metrics=run_all or bool(args.run_real_downstream_metrics),
        run_peak_enrichment=run_all or bool(args.run_peak_enrichment),
        run_update_recency_model=run_all or bool(args.run_update_recency_model),
        run_peak_weighted_overlap=run_all or bool(args.run_peak_weighted_overlap),
        run_reentry_prediction=run_all or bool(args.run_reentry_prediction),
        run_downstream_prediction=run_all or bool(args.run_downstream_prediction),
        run_peak_perturbation=bool(args.run_peak_perturbation),
        run_supplement=run_all or bool(args.run_supplement),
        recent_overlap_windows=recent_windows,
        leave_one_out_mode=str(args.leave_one_out_mode),
        real_rollout_required_for_main=bool(args.real_rollout_required_for_main),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_probe_batch=bool(args.enable_probe_batch),
        smoke=smoke,
    )


SEQUENCE_TRIAL_COLUMNS = ["network_seed", "sequence_id", "seq_len", "stage_k", "item_image_id", "item_label", "ordered_item_ids", "ordered_item_labels", "sequence_seed", "mean_pairwise_image_similarity", "max_pairwise_image_similarity", "min_pairwise_image_similarity"]
PROBE_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_image_id", "probe_label", "probe_source", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "class_pair", "candidate_seed", "peak_support_sum", "nonpeak_support_sum"]
MATCHED_GROUP_COLUMNS = ["network_seed", "matched_group_id", "high_peak_candidate_id", "low_peak_candidate_id", "raw_overlap_difference", "visual_similarity_difference", "input_energy_difference", "peak_weighted_overlap_difference", "class_pair_matched", "notes"]
STATE_BANK_MANIFEST_COLUMNS = ["network_seed", "sequence_id", "seq_len", "state_condition", "stage_k", "layer", "state_variable", "shape", "storage_file", "storage_key", "captured_after", "sample_ms", "delay_ms"]
PANEL_A_UNIT_COLUMNS = ["network_seed", "sequence_id", "seq_len", "layer", "state_variable", "unit_id", "update_count", "last_update_position", "time_since_last_update", "recency_group", "multiplicity_group", "update_history_group", "is_peak", "final_support", "baseline_support", "delta_support"]
PANEL_A_SUMMARY_COLUMNS = ["network_seed", "update_history_group", "P_peak", "mean_final_support", "mean_delta_support", "peak_enrichment", "n_units"]
PANEL_B_METRIC_COLUMNS = ["network_seed", "sequence_id", "layer", "state_variable", "target", "model_name", "r2", "cv_r2", "auc_if_binary", "delta_r2_vs_overlap_only", "delta_r2_vs_update_only", "delta_r2_vs_recency_only", "n_units"]
PANEL_B_COEF_COLUMNS = ["network_seed", "model_name", "coefficient_name", "coefficient_value", "standardized_coefficient", "p_value", "notes"]
PANEL_C_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "peak_support_sum", "nonpeak_support_sum"]
PANEL_D_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "reentry_strength", "DPI_L3", "dynamic_like_recovery", "decision_deflection_score"]
PANEL_E_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "early_recruitment_gain", "P_advance", "P_recruit", "spike_advance", "response_pattern_displacement", "decision_deflection_score", "partial_cue_completion_gain"]
PANEL_A_SOURCE_COLUMNS = ["network_seed", "sequence_id", "seq_len", "removed_position", "removed_label", "removed_image_id", "peak_loss", "nonpeak_loss", "prior_updated_loss", "peak_loss_fraction", "nonpeak_loss_fraction", "peak_vs_nonpeak_loss_ratio", "support_loss_total", "leave_one_out_mode", "proxy_mode"]
PANEL_A_SOURCE_SUMMARY_COLUMNS = ["network_seed", "seq_len", "removed_position", "relative_position_from_end", "mean_peak_loss_fraction", "sem_peak_loss_fraction", "mean_peak_vs_nonpeak_loss_ratio", "n_sequences"]
PANEL_B_UPDATE_HISTORY_COLUMNS = ["network_seed", "sequence_id", "seq_len", "unit_id", "is_peak", "is_nonpeak_control", "update_count", "last_update_position", "time_since_last_update", "recent_w2", "recent_w3", "recent_w4", "recent_w5", "is_multi_update", "is_multi_recent_w2", "is_multi_recent_w3", "is_multi_recent_w4", "is_multi_recent_w5", "final_support", "delta_support"]
PANEL_B_UPDATE_HISTORY_SUMMARY_COLUMNS = ["network_seed", "group", "mean_update_count", "P_update_ge_2", "P_update_ge_3", "mean_time_since_last_update", "P_recent_w2", "P_recent_w3", "P_recent_w4", "P_recent_w5", "P_multi_recent_w2", "P_multi_recent_w3", "P_multi_recent_w4", "P_multi_recent_w5", "n_units"]
PANEL_C_ORIGIN_COLUMNS = ["network_seed", "sequence_id", "seq_len", "overlap_window", "window_start_position", "window_end_position", "n_items_in_window", "overlap_type", "n_overlap_pixels", "n_peak_pixels", "dice_peak_overlap", "jaccard_peak_overlap", "peak_coverage", "overlap_precision", "cosine_delta_support_overlap_count", "spearman_delta_support_overlap_count", "fallback_used"]
PANEL_C_ORIGIN_SUMMARY_COLUMNS = ["network_seed", "overlap_window", "mean_dice", "sem_dice", "mean_peak_coverage", "mean_cosine", "n_sequences"]
PANEL_D_TRIAL_DEFINITION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_image_id", "probe_label", "probe_source", "raw_overlap", "peak_weighted_overlap", "peak_overlap_fraction", "nonpeak_overlap_fraction", "visual_similarity", "input_energy", "peak_support_sum", "nonpeak_support_sum", "class_pair", "candidate_seed"]
PANEL_D_REAL_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "prediction_Sfinal", "prediction_S0", "correct_Sfinal", "correct_S0", "first_fire_time_Sfinal", "first_fire_time_S0", "first_fire_time_delta", "l3_trace_delta_norm", "reentry_strength_real", "dynamic_like_recovery_real", "decision_deflection_score_real", "proxy_mode"]
PANEL_E_REAL_METRIC_COLUMNS = ["network_seed", "sequence_id", "probe_id", "matched_group_id", "raw_overlap", "peak_weighted_overlap", "peak_overlap_group", "visual_similarity", "input_energy", "early_recruitment_gain_real", "P_advance_real", "P_recruit_real", "spike_advance_real", "response_pattern_displacement_real", "decision_deflection_score_real", "partial_cue_completion_gain_real", "proxy_mode"]
PERTURBATION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "condition", "n_perturbed_units", "raw_overlap", "peak_weighted_overlap", "reentry_strength", "DPI_L3", "early_recruitment_gain", "decision_deflection_score", "completion_gain"]
PANEL_D_ROUTE_PEAK_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "seq_len", "perturbation_unit_set", "perturbation_condition", "perturbation_mode", "state_condition", "raw_overlap", "peak_weighted_overlap", "route_unit_count", "peak_unit_count", "route_peak_unit_count", "route_nonpeak_unit_count", "nonroute_peak_unit_count", "random_unit_count", "insufficient_units", "reentry_strength_intact", "reentry_strength_perturbed", "reentry_strength_s0", "reentry_loss", "normalized_reentry_loss", "prediction_intact", "prediction_perturbed", "prediction_s0", "first_fire_time_intact", "first_fire_time_perturbed", "first_fire_time_s0", "restore_ok", "perturbation_ok", "denominator_choice", "reset_variables", "probe_input_unchanged", "failure_reason"]
PANEL_D_ROUTE_PEAK_SUMMARY_COLUMNS = ["network_seed", "perturbation_unit_set", "mean_reentry_loss", "sem_reentry_loss", "mean_normalized_reentry_loss", "sem_normalized_reentry_loss", "n_trials", "n_valid_trials", "insufficient_fraction", "denominator_choice"]
PANEL_D_ROUTE_PEAK_CONTRAST_COLUMNS = ["network_seed", "contrast", "metric", "route_peak_minus_control", "route_peak_minus_route_nonpeak", "route_peak_minus_nonroute_peak", "route_peak_minus_random", "route_peak_effect_size", "n_valid_pairs"]
PANEL_E_ROUTE_PEAK_TRIAL_COLUMNS = ["network_seed", "sequence_id", "probe_id", "probe_label", "perturbation_unit_set", "perturbation_condition", "response_displacement_intact", "response_displacement_perturbed", "response_displacement_s0", "response_displacement_loss", "decision_deflection_intact", "decision_deflection_perturbed", "decision_deflection_s0", "decision_deflection_loss", "prediction_intact", "prediction_perturbed", "prediction_s0", "output_switch", "output_distribution_JS", "perturbation_ok", "insufficient_units", "failure_reason"]
PANEL_E_ROUTE_PEAK_SUMMARY_COLUMNS = ["network_seed", "perturbation_unit_set", "P_output_switch", "mean_response_displacement_loss", "sem_response_displacement_loss", "mean_decision_deflection_loss", "sem_decision_deflection_loss", "n_trials", "n_valid_trials"]
PANEL_E_ROUTE_PEAK_CONTRAST_COLUMNS = ["network_seed", "metric", "contrast", "route_peak_minus_route_nonpeak", "route_peak_minus_nonroute_peak", "route_peak_minus_random", "n_valid_pairs"]
PANEL_E_ROUTE_PEAK_OUTPUT_DISTRIBUTION_COLUMNS = ["network_seed", "sequence_id", "probe_id", "perturbation_unit_set", "output_distribution_JS", "intact_entropy", "condition_entropy"]
ROUTE_PEAK_UNIT_SET_COLUMNS = ["network_seed", "sequence_id", "probe_id", "perturbation_unit_set", "unit_id", "notes"]


if __name__ == "__main__":
    raise SystemExit(main())
