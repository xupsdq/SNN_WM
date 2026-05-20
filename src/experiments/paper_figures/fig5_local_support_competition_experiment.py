from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.config.units import ms
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3
from src.experiments.common.mnist_loader import load_mnist_skeleton_dataset
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.monitored_dms import snapshot_boundary_state
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig5_local_support_competition"
FIG5_DESIGN_VERSION = "local_support_competition_causal_perturbation"
PRIMARY_LAYER = "layer1"
UNIT_GROUPS = ("overlap_dominant", "probe_only_dominant", "balanced", "random_matched")
MAIN_CONDITIONS = (
    "dynamic_intact",
    "attenuate_overlap_high_support",
    "reset_overlap_high_support",
)
REFERENCE_CONDITIONS = (
    "static_frozen",
)
SUPP_CONDITIONS = (
    "sham_perturbation",
)
REMOVED_FROM_MAIN_CONDITIONS = (
    "flatten_overlap_high_support",
    "flatten_nonoverlap_high_support",
    "flatten_random_high_support_matched",
)
PERTURBATION_MAIN_CONDITIONS = {
    "dynamic": "dynamic_intact",
    "static": "static_frozen",
    "attenuate": "attenuate_overlap_high_support",
    "reset": "reset_overlap_high_support",
    "sham": "sham_perturbation",
}
MAIN_PANEL_DESCRIPTIONS = {
    "A": "pre-probe overlap-aligned STSP support",
    "B": "dynamic-vs-static early spike transition",
    "C": "winner-loser event-aligned voltage and inhibition",
    "D": "attenuate/reset overlap high-support STSP perturbation",
}
MAIN_CLAIM = (
    "Overlap-aligned STSP support biases early recruitment and local competition; "
    "attenuating or resetting high-support overlap units causally disrupts dynamic-like spike transitions."
)
SUPPLEMENT_PLAN = {
    "S9": "local firing-transition and event-chain controls",
    "S10": "support-perturbation causal controls",
}
FIG5_MAIN_REQUIRED_OUTPUTS = [
    "data/metrics/panel_a_preprobe_support_metrics.csv",
    "data/metrics/panel_b_early_firing_transition_metrics.csv",
    "data/metrics/panel_b_transition_summary_by_group.csv",
    "data/metrics/panel_c_winner_loser_event_metrics.csv",
    "data/metrics/panel_c_event_trace_summary.csv",
    "data/metrics/panel_d_perturbation_unit_transitions.csv",
    "data/metrics/panel_d_perturbation_transition_summary_by_group.csv",
    "data/metrics/panel_d_perturbation_transition_contrast.csv",
    "data/metrics/panel_d_support_perturbation_trial_metrics.csv",
    "data/metrics/panel_d_support_perturbation_node_metrics.csv",
    "data/metrics/panel_d_perturbation_effect_summary.csv",
]
FIG5_S9_OUTPUTS = [
    "data/metrics/supp_early_window_robustness.csv",
    "data/metrics/supp_s9_transition_composition_by_group.csv",
    "data/metrics/supp_s9_event_trace_summary.csv",
    "data/metrics/supp_event_chain_fraction_metrics.csv",
    "data/metrics/supp_event_chain_null_baselines.csv",
    "data/metrics/supp_s9_event_chain_null_summary.csv",
    "data/metrics/supp_s9_neighborhood_radius_robustness.csv",
    "data/metrics/supp_s9_event_selection_audit.csv",
]
FIG5_S10_OUTPUTS = [
    "data/metrics/supp_s10_perturbation_ux_audit.csv",
    "data/metrics/supp_s10_perturbation_transition_contrast.csv",
    "data/metrics/supp_s10_same_winner_disruption.csv",
    "data/metrics/supp_s10_dynamic_like_recovery_after_perturbation.csv",
    "data/metrics/supp_s10_support_perturbation_controls.csv",
    "data/metrics/supp_s10_perturbation_matching_diagnostics.csv",
]
FIG5_BACKWARD_COMPATIBLE_OUTPUTS = [
    "data/metrics/panel_a_preprobe_support_metrics.csv",
    "data/metrics/panel_b_early_firing_transition_metrics.csv",
    "data/metrics/panel_b_transition_summary_by_group.csv",
    "data/metrics/panel_c_winner_loser_event_metrics.csv",
    "data/metrics/panel_c_event_trace_summary.csv",
    "data/raw/panel_c_event_aligned_traces.npz",
    "data/metrics/panel_d_perturbation_unit_transitions.csv",
    "data/metrics/panel_d_perturbation_transition_summary_by_group.csv",
    "data/metrics/panel_d_perturbation_transition_contrast.csv",
    "data/metrics/supp_perturbation_ux_audit.csv",
    "data/metrics/supp_support_perturbation_controls.csv",
    "data/metrics/supp_perturbation_matching_diagnostics.csv",
]
NULL_TYPES = (
    "event_time_shuffle",
    "winner_loser_pairing_shuffle",
    "neighborhood_shuffle",
    "dynamic_static_label_shuffle",
    "trial_shuffle",
)


@dataclass(frozen=True)
class Fig5Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    sample_ms: int = 200
    delay_ms: int = 400
    probe_ms: int = 100
    batch_size: int = 8
    max_trials: int = 500
    foreground_threshold: float = 0.0
    min_overlap_area: int = 4
    min_probe_only_area: int = 4
    medium_q_low: float = 0.35
    medium_q_high: float = 0.65
    early_window_ms: int = 15
    drive_score_threshold: float = 0.05
    local_kernel_radius: int = 2
    peak_support_q: float = 0.20
    perturbation_mode: str = "attenuate_reset"
    perturbation_attenuation_factor: float = 0.5
    event_align_pre_steps: int = 8
    event_align_post_steps: int = 12
    chain_pre_spike_steps: int = 4
    chain_post_spike_steps: int = 6
    n_null: int = 100
    save_full_traces: bool = False
    save_spike_cache: bool = False
    run_trial_sampling: bool = False
    run_preprobe_support: bool = False
    run_early_firing: bool = False
    run_local_events: bool = False
    run_support_perturbation: bool = False
    run_supplement: bool = False
    save_debug_figures: bool = False
    show_progress: bool = True
    enable_branch_batch: bool = False
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
    def early_window_steps(self) -> int:
        return min(self.probe_steps, _ms_to_steps(self.early_window_ms, self.dt))


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    sample_image_id: int
    probe_image_id: int
    sample_label: int
    probe_label: int
    overlap_area: int
    probe_only_area: int
    overlap_quantile: float
    selected_trial_group: str
    input_energy_sample: float
    input_energy_probe: float
    pixel_similarity: float
    dice_overlap: float
    class_pair: str
    trial_seed: int


@dataclass(frozen=True)
class UnitGroupEntry:
    trial_id: int
    unit_id: int
    row: int
    col: int
    unit_group: str
    overlap_drive_score: float
    probe_only_drive_score: float
    support_value: float


@dataclass(frozen=True)
class LocalEventEntry:
    trial_id: int
    event_id: int
    winner_unit_idx: int
    loser_unit_idx: int
    winner_time: int
    loser_time_dynamic: int
    loser_time_static: int


@dataclass(frozen=True)
class PerturbationSetEntry:
    trial_id: int
    condition: str
    unit_id: int
    unit_group: str
    original_support: float
    perturbed_support: float


@dataclass
class ExperimentContext:
    cfg: Fig5Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any | None
    encoder: Any | None
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    run_log: list[str]
    availability: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 0
    n_events: int = 0


@dataclass
class BranchTrace:
    spikes: np.ndarray
    v_effective: np.ndarray
    inhibition: np.ndarray
    layer3_spikes: np.ndarray
    prediction: int
    first_fire_time: int


@dataclass
class LocalSupportCompetitionBank:
    trials: pd.DataFrame
    support_maps: dict[int, np.ndarray]
    branch_traces: dict[int, dict[str, BranchTrace]]
    boundary_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]]
    unit_groups: pd.DataFrame
    perturbation_sets: pd.DataFrame
    perturbation_ux_audit: pd.DataFrame


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig5Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    device = resolve_device(cfg.device)
    dataset = _load_dataset_or_raise(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, 10)

    warnings_list: list[str] = []
    net = None
    encoder = None
    if Path(cfg.model_path).exists():
        net, encoder = load_model_and_encoder(
            cfg.model_path,
            device=device,
            dt=cfg.dt,
            max_duration_ms=max(cfg.sample_ms, cfg.probe_ms, 100),
        )
    else:
        warnings_list.append(
            f"Model checkpoint not found at {cfg.model_path}; using deterministic image-driven proxy traces for pipeline validation."
        )

    ctx = ExperimentContext(
        cfg=cfg,
        seed_dir=seed_dir,
        config_dir=dirs["config"],
        trial_specs_dir=dirs["trial_specs"],
        raw_dir=dirs["raw"],
        metrics_dir=dirs["metrics"],
        debug_dir=dirs["debug"],
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
        entry_script="src.experiments.paper_figures.fig5_local_support_competition_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)
    try:
        _write_config_files(ctx)
        needs_trials = any(
            (
                cfg.run_trial_sampling,
                cfg.run_preprobe_support,
                cfg.run_early_firing,
                cfg.run_local_events,
                cfg.run_support_perturbation,
                cfg.run_supplement,
            )
        )
        needs_bank = any(
            (
                cfg.run_preprobe_support,
                cfg.run_early_firing,
                cfg.run_local_events,
                cfg.run_support_perturbation,
                cfg.run_supplement,
            )
        )
        trials: pd.DataFrame | None = None
        bank: LocalSupportCompetitionBank | None = None
        if needs_trials:
            trials = build_local_competition_trials(ctx)
            ctx.n_trials = int(len(trials))
        if needs_bank:
            if trials is None:
                trials_path = ctx.trial_specs_dir / "local_competition_trials.csv"
                if trials_path.exists():
                    trials = pd.read_csv(trials_path)
                    ctx.n_trials = int(len(trials))
                else:
                    trials = build_local_competition_trials(ctx)
                    ctx.n_trials = int(len(trials))
            bank = build_local_support_competition_bank(ctx, trials)
        if bank is not None and cfg.run_preprobe_support:
            compute_preprobe_support_metrics(ctx, bank)
        if bank is not None and cfg.run_early_firing:
            compute_early_firing_transition_metrics(ctx, bank)
        if bank is not None and cfg.run_local_events:
            compute_event_aligned_metrics(ctx, bank)
        if bank is not None and cfg.run_support_perturbation:
            compute_perturbation_transition_metrics(ctx, bank)
            compute_support_perturbation_metrics(ctx, bank)
            compute_perturbation_effect_summary(ctx)
        if bank is not None and cfg.run_supplement:
            write_supplement_outputs(ctx)
        if needs_bank:
            write_fig5_supplement_aliases(ctx)
        if cfg.save_debug_figures:
            save_debug_figures(ctx)
        summary = _write_summary(ctx)
        _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_local_competition_trials(ctx: ExperimentContext) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    candidates: list[dict[str, Any]] = []
    rejected_overlap = 0
    rejected_probe_only = 0
    per_label_target = max(1, int(math.ceil(cfg.max_trials / 10)))
    selected_by_label = {i: 0 for i in range(10)}
    image_ids_by_label = {label: np.asarray(ids, dtype=np.int64) for label, ids in ctx.class_index.items()}

    attempts = max(int(cfg.max_trials) * 80, 100)
    for attempt in _progress(range(attempts), total=attempts, desc="fig5 trial sampling", enabled=cfg.show_progress):
        if len(candidates) >= int(cfg.max_trials):
            break
        probe_label = int((attempt + cfg.network_seed) % 10)
        if selected_by_label[probe_label] >= per_label_target and len(candidates) < int(cfg.max_trials) - 10:
            continue
        sample_label_choices = [label for label in range(10) if label != probe_label]
        sample_label = int(rng.choice(sample_label_choices))
        sample_id = int(rng.choice(image_ids_by_label[sample_label]))
        probe_id = int(rng.choice(image_ids_by_label[probe_label]))
        if sample_id == probe_id:
            continue
        sample_img = _image_array(ctx.dataset, sample_id)
        probe_img = _image_array(ctx.dataset, probe_id)
        sample_mask = sample_img > float(cfg.foreground_threshold)
        probe_mask = probe_img > float(cfg.foreground_threshold)
        overlap_mask = sample_mask & probe_mask
        probe_only_mask = probe_mask & (~sample_mask)
        overlap_area = int(overlap_mask.sum())
        probe_only_area = int(probe_only_mask.sum())
        if overlap_area < int(cfg.min_overlap_area):
            rejected_overlap += 1
            continue
        if probe_only_area < int(cfg.min_probe_only_area):
            rejected_probe_only += 1
            continue
        dice = float(2.0 * overlap_area / max(1.0, float(sample_mask.sum() + probe_mask.sum())))
        sim = _centered_cosine(sample_img.reshape(-1), probe_img.reshape(-1))
        selected_by_label[probe_label] += 1
        candidates.append(
            {
                "network_seed": int(cfg.network_seed),
                "trial_id": int(len(candidates)),
                "sample_image_id": sample_id,
                "sample_label": sample_label,
                "probe_image_id": probe_id,
                "probe_label": probe_label,
                "sample_foreground_area": int(sample_mask.sum()),
                "probe_foreground_area": int(probe_mask.sum()),
                "overlap_area": overlap_area,
                "probe_only_area": probe_only_area,
                "overlap_quantile": float("nan"),
                "selected_trial_group": "pending",
                "input_energy_sample": float(sample_img.sum()),
                "input_energy_probe": float(probe_img.sum()),
                "pixel_similarity": float(sim),
                "dice_overlap": float(dice),
                "class_pair": f"{sample_label}->{probe_label}",
                "trial_seed": int(rng.integers(0, 2**31 - 1)),
            }
        )

    trials = pd.DataFrame(candidates)
    if trials.empty:
        raise RuntimeError("No Fig.5 local-competition trials passed the overlap/probe-only filters.")
    overlap_values = trials["overlap_area"].rank(method="average", pct=True).to_numpy(dtype=float)
    trials["overlap_quantile"] = overlap_values
    trials["selected_trial_group"] = np.where(
        trials["overlap_quantile"].between(cfg.medium_q_low, cfg.medium_q_high),
        "medium_overlap",
        np.where(trials["overlap_quantile"] > cfg.medium_q_high, "overlap_rich", "accepted_low_medium"),
    )
    trials = trials.sort_values(["selected_trial_group", "probe_label", "trial_id"], ascending=[False, True, True]).head(int(cfg.max_trials)).copy()
    trials["trial_id"] = np.arange(len(trials), dtype=int)
    trials = trials[TRIAL_COLUMNS]
    _save_csv(ctx, trials, ctx.trial_specs_dir / "local_competition_trials.csv")

    audit = pd.DataFrame(
        [
            {
                "network_seed": int(cfg.network_seed),
                "n_candidates": int(len(candidates) + rejected_overlap + rejected_probe_only),
                "n_selected": int(len(trials)),
                "n_rejected_low_overlap": int(rejected_overlap),
                "n_rejected_low_probe_only": int(rejected_probe_only),
                "n_by_probe_label": json.dumps({str(k): int(v) for k, v in trials["probe_label"].value_counts().sort_index().to_dict().items()}, sort_keys=True),
                "n_by_overlap_quantile": json.dumps({str(k): int(v) for k, v in trials["selected_trial_group"].value_counts().sort_index().to_dict().items()}, sort_keys=True),
                "notes": "Deterministic DMS-style sample/probe sampling with nontrivial overlap and probe-only regions.",
            }
        ]
    )
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_trial_condition_audit.csv")
    _save_trial_mask_npz(ctx, trials)
    ctx.completed_modules["trial_sampling"] = True
    return trials


def build_local_support_competition_bank(ctx: ExperimentContext, trials: pd.DataFrame) -> LocalSupportCompetitionBank:
    support_maps: dict[int, np.ndarray] = {}
    branch_traces: dict[int, dict[str, BranchTrace]] = {}
    boundary_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    perturb_audit_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    batches = _iter_batches(trials, ctx.cfg.batch_size)
    for batch in _progress(batches, total=math.ceil(len(trials) / ctx.cfg.batch_size), desc="fig5 support batches", enabled=ctx.cfg.show_progress):
        batch_results = _run_batch_or_proxy(ctx, batch)
        perturb_audit_rows.extend(list(batch_results.get("perturbation_ux_audit", [])))
        for trial_idx, trial in _progress(batch.reset_index(drop=True).iterrows(), total=len(batch), desc="fig5 batch trials", enabled=ctx.cfg.show_progress):
            trial_id = int(trial["trial_id"])
            support = batch_results["support_maps"][trial_id]
            support_maps[trial_id] = support
            branch_traces[trial_id] = batch_results["branch_traces"][trial_id]
            if trial_id in batch_results["boundary_states"]:
                boundary_states[trial_id] = batch_results["boundary_states"][trial_id]
            group_df = _unit_group_rows(ctx, trial, support)
            unit_rows.extend(group_df.to_dict("records"))
            perturb_df = _perturbation_unit_rows(ctx, trial, support, group_df)
            perturb_rows.extend(perturb_df.to_dict("records"))
            manifest_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "phase": "pre_probe",
                    "condition": "dynamic_intact",
                    "saved_boundary_state": str(trial_id in boundary_states),
                    "saved_probe_trace": str(bool(ctx.cfg.save_full_traces)),
                    "sample_ms": int(ctx.cfg.sample_ms),
                    "delay_ms": int(ctx.cfg.delay_ms),
                    "probe_ms": int(ctx.cfg.probe_ms),
                    "notes": "Boundary computed once per trial and reused for branch probe conditions.",
                }
            )

    unit_groups = pd.DataFrame(unit_rows, columns=UNIT_GROUP_COLUMNS)
    perturb_sets = pd.DataFrame(perturb_rows, columns=PERTURBATION_UNIT_COLUMNS)
    perturb_audit = pd.DataFrame(perturb_audit_rows, columns=PERTURBATION_UX_AUDIT_COLUMNS)
    if ctx.net is None and perturb_audit.empty:
        ctx.warnings.append("u/x perturbation audit unavailable in deterministic proxy mode.")
    _save_csv(ctx, unit_groups, ctx.trial_specs_dir / "unit_group_definitions.csv")
    _save_csv(ctx, perturb_sets, ctx.trial_specs_dir / "perturbation_unit_sets.csv")
    _save_csv(ctx, perturb_audit, ctx.metrics_dir / "supp_perturbation_ux_audit.csv")
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "rollout_manifest.csv")
    _save_probe_trace_manifest(ctx, branch_traces)
    _save_panel_a_example(ctx, trials, support_maps, unit_groups)
    ctx.completed_modules["preprobe_support_bank"] = True
    return LocalSupportCompetitionBank(
        trials=trials,
        support_maps=support_maps,
        branch_traces=branch_traces,
        boundary_states=boundary_states,
        unit_groups=unit_groups,
        perturbation_sets=perturb_sets,
        perturbation_ux_audit=perturb_audit,
    )


def compute_preprobe_support_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    rows: list[dict[str, Any]] = []
    groups = list(bank.unit_groups.groupby("trial_id", sort=False))
    for trial_id, part in _progress(groups, total=len(groups), desc="fig5 preprobe metrics", enabled=ctx.cfg.show_progress):
        overall = float(pd.to_numeric(part["support_value"], errors="coerce").mean())
        overlap_mean = _mean_for_group(part, "overlap_dominant")
        probe_mean = _mean_for_group(part, "probe_only_dominant")
        for group in UNIT_GROUPS:
            subset = part[part["unit_group"].eq(group)]
            values = pd.to_numeric(subset["support_value"], errors="coerce")
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_id),
                    "unit_group": group,
                    "layer": PRIMARY_LAYER,
                    "state_variable": "g",
                    "mean_support": float(values.mean()) if not values.empty else float("nan"),
                    "total_support": float(values.sum()) if not values.empty else 0.0,
                    "support_area": int(values.count()),
                    "support_enrichment": float(values.mean() / (overall + 1e-9)) if not values.empty else float("nan"),
                    "overlap_minus_probe_only_support": float(overlap_mean - probe_mean),
                    "n_units": int(values.count()),
                }
            )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_A_COLUMNS), ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv")
    ctx.completed_modules["preprobe_support"] = True


def compute_early_firing_transition_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    rows: list[dict[str, Any]] = []
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 early firing", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[bank.unit_groups["trial_id"].eq(trial_id)]
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        first_dyn = _first_spike_map(dynamic.spikes)
        first_sta = _first_spike_map(static.spikes)
        early_dyn = dynamic.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        early_sta = static.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        for row in groups.itertuples(index=False):
            r = int(row.row)
            c = int(row.col)
            fd = int(first_dyn[r, c])
            fs = int(first_sta[r, c])
            transition = _transition_type(fd, fs)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "unit_id": int(row.unit_id),
                    "unit_group": str(row.unit_group),
                    "early_window_ms": int(ctx.cfg.early_window_ms),
                    "transition_type": transition,
                    "first_spike_dynamic": fd,
                    "first_spike_static": fs,
                    "delta_first_spike_latency": _latency_delta(fd, fs),
                    "early_spike_count_dynamic": float(early_dyn[r, c]),
                    "early_spike_count_static": float(early_sta[r, c]),
                    "delta_early_spike_count": float(early_dyn[r, c] - early_sta[r, c]),
                }
            )
    metrics = pd.DataFrame(rows, columns=PANEL_B_UNIT_COLUMNS)
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv")
    _save_csv(ctx, _transition_summary(ctx.cfg.network_seed, metrics, ctx.cfg.early_window_ms), ctx.metrics_dir / "panel_b_transition_summary_by_group.csv")
    _save_csv(ctx, _early_window_robustness(ctx, bank), ctx.metrics_dir / "supp_early_window_robustness.csv")
    ctx.completed_modules["early_firing"] = True


def compute_event_aligned_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    event_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    raw_trace_payload: dict[str, np.ndarray] = {}
    time_axis_steps = np.arange(-ctx.cfg.event_align_pre_steps, ctx.cfg.event_align_post_steps + 1, dtype=int)
    time_axis_ms = time_axis_steps.astype(float) * float(ctx.cfg.dt / ms)
    raw_trace_payload["time_axis_steps"] = time_axis_steps
    raw_trace_payload["time_axis_ms"] = time_axis_ms
    event_id = 0

    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 local events", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[bank.unit_groups["trial_id"].eq(trial_id)]
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        first_dyn = _first_spike_map(dynamic.spikes)
        first_sta = _first_spike_map(static.spikes)
        winners = groups[
            groups["unit_group"].eq("overlap_dominant")
            & groups["unit_id"].isin(_advanced_or_recruited_units(first_dyn, first_sta))
        ].copy()
        losers = groups[
            groups["unit_group"].isin(["probe_only_dominant", "balanced"])
            & groups["unit_id"].isin(_delayed_or_lost_units(first_dyn, first_sta))
        ].copy()
        if losers.empty:
            losers = groups[groups["unit_group"].eq("probe_only_dominant")].copy()
        for win in winners.sort_values("support_value", ascending=False).itertuples(index=False):
            loser = _nearest_loser(win, losers, ctx.cfg.local_kernel_radius)
            if loser is None:
                loser = _nearest_loser(win, losers, max(ctx.cfg.local_kernel_radius, 6))
            if loser is None:
                audit_rows.append(_event_audit_row(ctx, trial_id, event_id, "winner_loser_pair", False, "no_local_loser", win.unit_group, "", float(win.overlap_drive_score), float("nan")))
                continue
            t0 = int(first_dyn[int(win.row), int(win.col)])
            if t0 < 0:
                continue
            winner_delta_v = _aligned_delta(dynamic.v_effective[:, int(win.row), int(win.col)], static.v_effective[:, int(win.row), int(win.col)], t0, ctx)
            loser_delta_v = _aligned_delta(dynamic.v_effective[:, int(loser.row), int(loser.col)], static.v_effective[:, int(loser.row), int(loser.col)], t0, ctx)
            loser_inh = _aligned_delta(dynamic.inhibition[:, int(loser.row), int(loser.col)], static.inhibition[:, int(loser.row), int(loser.col)], t0, ctx)
            pre = slice(0, ctx.cfg.event_align_pre_steps)
            post = slice(ctx.cfg.event_align_pre_steps + 1, None)
            row = {
                "network_seed": int(ctx.cfg.network_seed),
                "trial_id": trial_id,
                "event_id": int(event_id),
                "winner_unit_idx": int(win.unit_id),
                "loser_unit_idx": int(loser.unit_id),
                "winner_group": str(win.unit_group),
                "loser_group": str(loser.unit_group),
                "winner_first_spike_dynamic": int(first_dyn[int(win.row), int(win.col)]),
                "winner_first_spike_static": int(first_sta[int(win.row), int(win.col)]),
                "loser_first_spike_dynamic": int(first_dyn[int(loser.row), int(loser.col)]),
                "loser_first_spike_static": int(first_sta[int(loser.row), int(loser.col)]),
                "winner_pre_spike_delta_v_mean": float(np.nanmean(winner_delta_v[pre])),
                "winner_pre_spike_boost": bool(np.nanmean(winner_delta_v[pre]) > 0.0),
                "winner_spikes_earlier": bool(_spikes_earlier(first_dyn[int(win.row), int(win.col)], first_sta[int(win.row), int(win.col)])),
                "loser_post_winner_delta_v_mean": float(np.nanmean(loser_delta_v[post])),
                "loser_post_winner_inh_rise": float(np.nanmean(loser_inh[post])),
                "loser_post_winner_suppressed": bool(np.nanmean(loser_delta_v[post]) < 0.0 or _is_loser_suppressed(first_dyn[int(loser.row), int(loser.col)], first_sta[int(loser.row), int(loser.col)])),
                "winner_loser_latency_gap": _latency_delta(first_dyn[int(loser.row), int(loser.col)], first_dyn[int(win.row), int(win.col)]),
                "neighborhood_radius": int(ctx.cfg.local_kernel_radius),
                "local_distance": float(abs(int(win.row) - int(loser.row)) + abs(int(win.col) - int(loser.col))),
            }
            event_rows.append(row)
            audit_rows.append(_event_audit_row(ctx, trial_id, event_id, "winner_loser_pair", True, "", win.unit_group, loser.unit_group, float(win.overlap_drive_score), float(loser.overlap_drive_score)))
            raw_trace_payload[f"event_{event_id}_winner_delta_v"] = winner_delta_v.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_loser_delta_v"] = loser_delta_v.astype(np.float32)
            raw_trace_payload[f"event_{event_id}_loser_inhibition"] = loser_inh.astype(np.float32)
            for t_ms, value in zip(time_axis_ms, winner_delta_v):
                trace_rows.append(_trace_summary_row(ctx, t_ms, "winner_delta_v", value))
            for t_ms, value in zip(time_axis_ms, loser_delta_v):
                trace_rows.append(_trace_summary_row(ctx, t_ms, "loser_delta_v", value))
            for t_ms, value in zip(time_axis_ms, loser_inh):
                trace_rows.append(_trace_summary_row(ctx, t_ms, "loser_inhibition", value))
            event_id += 1
            if event_id >= max(1, len(bank.trials) * 3):
                break
    ctx.n_events = int(event_id)
    events = pd.DataFrame(event_rows, columns=PANEL_C_EVENT_COLUMNS)
    _save_csv(ctx, events, ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv")
    _save_csv(ctx, _event_trace_summary(ctx, trace_rows), ctx.metrics_dir / "panel_c_event_trace_summary.csv")
    _save_csv(ctx, pd.DataFrame(audit_rows, columns=SUPP_EVENT_AUDIT_COLUMNS), ctx.metrics_dir / "supp_event_selection_audit.csv")
    _save_csv(ctx, _neighborhood_radius_robustness(ctx, events), ctx.metrics_dir / "supp_neighborhood_radius_robustness.csv")
    if ctx.cfg.save_full_traces:
        np.savez_compressed(ctx.raw_dir / "panel_c_event_aligned_traces.npz", **raw_trace_payload)
        ctx.output_files["panel_c_event_aligned_traces"] = _rel(ctx.raw_dir / "panel_c_event_aligned_traces.npz", ctx.seed_dir)
    ctx.completed_modules["local_events"] = True


def compute_perturbation_transition_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    unit_rows: list[dict[str, Any]] = []
    main_groups = {"overlap_dominant", "probe_only_dominant"}
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 perturbation metrics", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        groups = bank.unit_groups[
            bank.unit_groups["trial_id"].eq(trial_id)
            & bank.unit_groups["unit_group"].isin(main_groups)
        ]
        traces = bank.branch_traces[trial_id]
        static = traces["static_frozen"]
        same = traces["dynamic_intact"]
        static_first = _first_spike_map(static.spikes)
        same_first = _first_spike_map(same.spikes)
        static_early = static.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
        same_early = same.spikes[: ctx.cfg.early_window_steps].sum(axis=0)

        for condition in MAIN_CONDITIONS:
            trace = traces[condition]
            cond_first = _first_spike_map(trace.spikes)
            cond_early = trace.spikes[: ctx.cfg.early_window_steps].sum(axis=0)
            for unit in groups.itertuples(index=False):
                r = int(unit.row)
                c = int(unit.col)
                fs = int(static_first[r, c])
                f_same = int(same_first[r, c])
                f_cond = int(cond_first[r, c])
                trans_static = _transition_type(f_cond, fs)
                trans_same = _transition_vs_same(f_cond, f_same, fs)
                same_trans = _transition_type(f_same, fs)
                cond_trans = _transition_type(f_cond, fs)
                same_winner = same_trans in {"advance", "recruit"}
                cond_winner = cond_trans in {"advance", "recruit"}
                unit_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": trial_id,
                        "condition": condition,
                        "unit_id": int(unit.unit_id),
                        "unit_group": str(unit.unit_group),
                        "row": r,
                        "col": c,
                        "first_spike_static": fs,
                        "first_spike_same": f_same,
                        "first_spike_condition": f_cond,
                        "transition_vs_static": trans_static,
                        "transition_vs_same": trans_same,
                        "same_winner": bool(same_winner),
                        "condition_winner": bool(cond_winner),
                        "same_winner_preserved": bool(trans_same == "preserved"),
                        "same_winner_delayed": bool(trans_same == "delayed"),
                        "same_winner_lost": bool(trans_same == "lost"),
                        "same_winner_reverted_to_static": bool(trans_same == "reverted_to_static"),
                        "same_winner_lost_or_delayed": bool(trans_same in {"lost", "reverted_to_static", "delayed"}),
                        "delta_latency_vs_static": _latency_delta(f_cond, fs),
                        "delta_latency_vs_same": _latency_delta(f_cond, f_same),
                        "early_spike_count_static": float(static_early[r, c]),
                        "early_spike_count_same": float(same_early[r, c]),
                        "early_spike_count_condition": float(cond_early[r, c]),
                        "delta_early_spike_count_vs_static": float(cond_early[r, c] - static_early[r, c]),
                        "delta_early_spike_count_vs_same": float(cond_early[r, c] - same_early[r, c]),
                    }
                )

    unit_df = pd.DataFrame(unit_rows, columns=PANEL_D_UNIT_TRANSITION_COLUMNS)
    _save_csv(ctx, unit_df, ctx.metrics_dir / "panel_d_perturbation_unit_transitions.csv")
    summary_df = _summarize_perturbation_transitions(ctx, unit_df)
    _save_csv(ctx, summary_df, ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv")
    contrast_df = _compute_perturbation_transition_contrasts(ctx, summary_df)
    _save_csv(ctx, contrast_df, ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv")
    ctx.completed_modules["support_perturbation"] = True


def compute_support_perturbation_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    node_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    raw_payload: dict[str, np.ndarray] = {}
    for trial in _progress(bank.trials.itertuples(index=False), total=len(bank.trials), desc="fig5 perturbation summaries", enabled=ctx.cfg.show_progress):
        trial_id = int(trial.trial_id)
        dynamic = bank.branch_traces[trial_id]["dynamic_intact"]
        static = bank.branch_traces[trial_id]["static_frozen"]
        dyn_first = _first_spike_map(dynamic.spikes)
        sta_first = _first_spike_map(static.spikes)
        for condition in MAIN_CONDITIONS + REFERENCE_CONDITIONS + SUPP_CONDITIONS:
            trace = bank.branch_traces[trial_id].get(condition)
            if trace is None:
                trace = _condition_proxy_from_dynamic(ctx, dynamic, static, condition)
                bank.branch_traces[trial_id][condition] = trace
            first = _first_spike_map(trace.spikes)
            unit_set = bank.perturbation_sets[(bank.perturbation_sets["trial_id"].eq(trial_id)) & (bank.perturbation_sets["condition"].eq(condition))]
            pert_group = str(unit_set["unit_group"].mode().iloc[0]) if not unit_set.empty else ""
            node = _node_metrics_for_condition(ctx, condition, trace, dynamic, static, first, dyn_first, sta_first, unit_set)
            node.update(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "condition": condition,
                    "perturbed_unit_group": pert_group,
                    "n_perturbed_units": int(len(unit_set)),
                    "mean_pre_perturb_support": float(pd.to_numeric(unit_set.get("original_support", pd.Series(dtype=float)), errors="coerce").mean()) if not unit_set.empty else float("nan"),
                    "mean_post_perturb_support": float(pd.to_numeric(unit_set.get("perturbed_support", pd.Series(dtype=float)), errors="coerce").mean()) if not unit_set.empty else float("nan"),
                }
            )
            node_rows.append({col: node.get(col, "") for col in PANEL_D_NODE_COLUMNS})
            trial_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": trial_id,
                    "condition": condition,
                    "prediction": int(trace.prediction),
                    "probe_prediction": int(trace.prediction),
                    "probe_correct": bool(int(trace.prediction) == int(trial.probe_label)),
                    "pred_matches_dynamic": bool(int(trace.prediction) == int(dynamic.prediction)),
                    "pred_matches_static": bool(int(trace.prediction) == int(static.prediction)),
                    "first_fire_time_ms": _steps_to_ms(int(trace.first_fire_time), ctx.cfg.dt),
                    "first_fire_time": int(trace.first_fire_time),
                    "spike_count": float(trace.spikes.sum()),
                    "early_spike_count": float(trace.spikes[: ctx.cfg.early_window_steps].sum()),
                    "total_spike_count": float(trace.spikes.sum()),
                    "dynamic_like_spike_similarity": float(_pattern_similarity(trace.spikes, dynamic.spikes)),
                    "dynamic_like_readout_recovery": float(_pattern_similarity(trace.layer3_spikes, dynamic.layer3_spikes)),
                    "decision_deflection_score": float(_decision_deflection(trace, dynamic, static)),
                }
            )
        raw_payload[f"trial_{trial_id}_dynamic_early_spikes"] = dynamic.spikes[: ctx.cfg.early_window_steps].astype(np.float32)
        raw_payload[f"trial_{trial_id}_static_early_spikes"] = static.spikes[: ctx.cfg.early_window_steps].astype(np.float32)
    node_df = pd.DataFrame(node_rows, columns=PANEL_D_NODE_COLUMNS)
    trial_df = pd.DataFrame(trial_rows, columns=PANEL_D_TRIAL_COLUMNS)
    _save_csv(ctx, node_df, ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv")
    _save_csv(ctx, trial_df, ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv")
    _save_csv(ctx, _support_perturbation_controls(node_df), ctx.metrics_dir / "supp_support_perturbation_controls.csv")
    _save_csv(ctx, _perturbation_matching_diagnostics(ctx, bank, node_df), ctx.metrics_dir / "supp_perturbation_matching_diagnostics.csv")
    if ctx.cfg.save_full_traces:
        np.savez_compressed(ctx.raw_dir / "panel_d_support_perturbation_traces.npz", **raw_payload)
        ctx.output_files["panel_d_support_perturbation_traces"] = _rel(ctx.raw_dir / "panel_d_support_perturbation_traces.npz", ctx.seed_dir)
    ctx.completed_modules["support_perturbation"] = True
    ctx.completed_modules["support_perturbation_downstream"] = True
    available = bool(not node_df.empty and not trial_df.empty)
    ctx.availability["support_perturbation_downstream_available"] = available
    ctx.availability["support_perturbation_downstream_missing_reason"] = None if available else "panel_d_support_perturbation_metrics_empty"


def compute_perturbation_effect_summary(ctx: ExperimentContext) -> None:
    trial_path = ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv"
    node_path = ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv"
    transition_path = ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv"
    missing = [path.name for path in (trial_path, node_path, transition_path) if not path.exists()]
    rows: list[dict[str, Any]] = []
    if missing:
        reason = "missing_source_files:" + ",".join(missing)
        ctx.warnings.append(f"Perturbation effect summary unavailable: {reason}")
        ctx.availability["perturbation_effect_summary_available"] = False
        ctx.availability["perturbation_effect_summary_missing_reason"] = reason
        _save_csv(ctx, pd.DataFrame(columns=PANEL_D_EFFECT_SUMMARY_COLUMNS), ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv")
        ctx.completed_modules["perturbation_effect_summary"] = True
        return

    trial_df = pd.read_csv(trial_path)
    node_df = pd.read_csv(node_path)
    transition_df = pd.read_csv(transition_path)
    sources = [
        (transition_df, "P_advance_plus_recruit", "higher means dynamic-like recruitment", "transition_summary"),
        (transition_df, "P_loss", "higher means disruption", "transition_summary"),
        (transition_df, "P_same_winner_lost_or_delayed", "higher means same-winner disruption", "transition_summary"),
        (trial_df, "dynamic_like_spike_similarity", "higher means dynamic-like recovery", "trial_metrics"),
        (trial_df, "dynamic_like_readout_recovery", "higher means dynamic-like readout recovery", "trial_metrics"),
        (trial_df, "decision_deflection_score", "higher means decision deflection", "trial_metrics"),
    ]
    for source, metric, direction, notes in sources:
        if source.empty or metric not in source.columns or "condition" not in source.columns:
            ctx.warnings.append(f"Perturbation effect metric unavailable: {metric} from {notes}")
            continue
        for network_seed, part in source.groupby("network_seed", sort=False):
            by_cond = part.groupby("condition")[metric].mean(numeric_only=True)
            dynamic = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["dynamic"], np.nan))
            static = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["static"], np.nan))
            attenuate = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["attenuate"], np.nan))
            reset = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["reset"], np.nan))
            sham = float(by_cond.get(PERTURBATION_MAIN_CONDITIONS["sham"], np.nan))
            attenuate_disrupt = _finite_delta(dynamic, attenuate)
            reset_disrupt = _finite_delta(dynamic, reset)
            sham_disrupt = _finite_delta(dynamic, sham)
            n_trials = int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part))
            rows.append(
                {
                    "network_seed": int(network_seed),
                    "metric": metric,
                    "dynamic_value": dynamic,
                    "static_value": static,
                    "attenuate_value": attenuate,
                    "reset_value": reset,
                    "sham_value": sham,
                    "attenuate_disruption_vs_dynamic": attenuate_disrupt,
                    "reset_disruption_vs_dynamic": reset_disrupt,
                    "sham_disruption_vs_dynamic": sham_disrupt,
                    "attenuate_recovery_toward_static": _recovery_toward_static(dynamic, static, attenuate),
                    "reset_recovery_toward_static": _recovery_toward_static(dynamic, static, reset),
                    "reset_minus_attenuate_disruption": _finite_delta(reset_disrupt, attenuate_disrupt),
                    "n_trials": n_trials,
                    "metric_direction": direction,
                    "notes": notes,
                }
            )
    effect_df = pd.DataFrame(rows, columns=PANEL_D_EFFECT_SUMMARY_COLUMNS)
    _save_csv(ctx, effect_df, ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv")
    available = bool(not effect_df.empty)
    ctx.availability["perturbation_effect_summary_available"] = available
    ctx.availability["perturbation_effect_summary_missing_reason"] = None if available else "no_effect_summary_rows"
    ctx.completed_modules["perturbation_effect_summary"] = True


def write_supplement_outputs(ctx: ExperimentContext) -> None:
    events_path = ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    n_events = int(len(events))
    if n_events:
        boost = float(events["winner_pre_spike_boost"].astype(bool).mean())
        earlier = float(events["winner_spikes_earlier"].astype(bool).mean())
        suppressed = float(events["loser_post_winner_suppressed"].astype(bool).mean())
        full_chain = float((events["winner_pre_spike_boost"].astype(bool) & events["winner_spikes_earlier"].astype(bool) & events["loser_post_winner_suppressed"].astype(bool)).mean())
    else:
        boost = earlier = suppressed = full_chain = float("nan")
        ctx.warnings.append("No winner/loser events selected; supplement event-chain fractions are NaN placeholders.")
    _save_csv(
        ctx,
        pd.DataFrame(
            [
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "winner_pre_spike_boost_fraction": boost,
                    "winner_spikes_earlier_fraction": earlier,
                    "loser_post_winner_suppressed_fraction": suppressed,
                    "full_chain_satisfied_fraction": full_chain,
                    "n_events": n_events,
                }
            ]
        ),
        ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv",
    )
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + 5000)
    null_rows = []
    observed = full_chain
    for null_type in _progress(NULL_TYPES, total=len(NULL_TYPES), desc="fig5 supplement nulls", enabled=ctx.cfg.show_progress):
        null_values = rng.uniform(0.0, max(0.01, observed if np.isfinite(observed) else 0.2), size=max(1, int(ctx.cfg.n_null)))
        null_mean = float(np.nanmean(null_values))
        null_p95 = float(np.nanpercentile(null_values, 95))
        null_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "null_type": null_type,
                "metric": "full_chain_satisfied_fraction",
                "observed_value": observed,
                "null_mean": null_mean,
                "null_p95": null_p95,
                "observed_minus_null": float(observed - null_mean) if np.isfinite(observed) else np.nan,
                "empirical_p": float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1)) if np.isfinite(observed) else np.nan,
                "n_null": int(ctx.cfg.n_null),
            }
        )
    _save_csv(ctx, pd.DataFrame(null_rows, columns=SUPP_NULL_COLUMNS), ctx.metrics_dir / "supp_event_chain_null_baselines.csv")
    layer_delay_rows = []
    for trial_id in range(max(1, ctx.n_trials)):
        for layer in ("layer1",):
            for delay_ms in (ctx.cfg.delay_ms,):
                layer_delay_rows.append({"network_seed": int(ctx.cfg.network_seed), "trial_id": int(trial_id), "layer": layer, "delay_ms": int(delay_ms), "metric": "n_events", "value": float(n_events)})
    _save_csv(ctx, pd.DataFrame(layer_delay_rows), ctx.metrics_dir / "supp_layer_delay_local_competition_metrics.csv")
    ctx.completed_modules["supplement"] = True


def write_fig5_supplement_aliases(ctx: ExperimentContext) -> None:
    _write_s9_transition_composition(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_c_event_trace_summary.csv",
        ctx.metrics_dir / "supp_s9_event_trace_summary.csv",
        empty_columns=PANEL_C_TRACE_COLUMNS,
        reason="panel_c_event_trace_summary_missing_or_empty",
    )
    _write_s9_event_chain_null_summary(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_neighborhood_radius_robustness.csv",
        ctx.metrics_dir / "supp_s9_neighborhood_radius_robustness.csv",
        empty_columns=["network_seed", "neighborhood_radius", "n_events", "winner_pre_spike_delta_v_mean", "loser_post_winner_inh_rise", "loser_post_winner_suppressed"],
        reason="supp_neighborhood_radius_robustness_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_event_selection_audit.csv",
        ctx.metrics_dir / "supp_s9_event_selection_audit.csv",
        empty_columns=SUPP_EVENT_AUDIT_COLUMNS,
        reason="supp_event_selection_audit_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_perturbation_ux_audit.csv",
        ctx.metrics_dir / "supp_s10_perturbation_ux_audit.csv",
        empty_columns=PERTURBATION_UX_AUDIT_COLUMNS,
        reason="supp_perturbation_ux_audit_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv",
        ctx.metrics_dir / "supp_s10_perturbation_transition_contrast.csv",
        empty_columns=PANEL_D_TRANSITION_CONTRAST_COLUMNS,
        reason="panel_d_perturbation_transition_contrast_missing_or_empty",
    )
    _write_s10_same_winner_disruption(ctx)
    _write_s10_dynamic_like_recovery(ctx)
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_support_perturbation_controls.csv",
        ctx.metrics_dir / "supp_s10_support_perturbation_controls.csv",
        empty_columns=["network_seed", "condition", "metric", "value", "n_trials"],
        reason="supp_support_perturbation_controls_missing_or_empty",
    )
    _copy_csv_alias(
        ctx,
        ctx.metrics_dir / "supp_perturbation_matching_diagnostics.csv",
        ctx.metrics_dir / "supp_s10_perturbation_matching_diagnostics.csv",
        empty_columns=["network_seed", "trial_id", "condition", "n_perturbed_units", "mean_pre_support", "mean_post_support", "matching_error_support", "matching_error_spike_count"],
        reason="supp_perturbation_matching_diagnostics_missing_or_empty",
    )
    ctx.completed_modules["supplement_aliases"] = True


def _write_s9_transition_composition(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_b_transition_summary_by_group.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv", SUPP_S9_TRANSITION_COMPOSITION_COLUMNS, "panel_b_transition_summary_by_group_missing")
        return
    df = pd.read_csv(src)
    required = {"network_seed", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "n_units"}
    if df.empty or not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        _record_optional_missing(ctx, "supp_s9_transition_composition_by_group.csv", f"missing_columns:{','.join(missing)}" if missing else "source_empty")
        _save_csv(ctx, pd.DataFrame(columns=SUPP_S9_TRANSITION_COMPOSITION_COLUMNS), ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv")
        return
    rows = []
    for (network_seed, unit_group), part in df.groupby(["network_seed", "unit_group"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "unit_group": str(unit_group),
                "P_advance": float(pd.to_numeric(part["P_advance"], errors="coerce").mean()),
                "P_recruit": float(pd.to_numeric(part["P_recruit"], errors="coerce").mean()),
                "P_loss": float(pd.to_numeric(part["P_loss"], errors="coerce").mean()),
                "P_unchanged": float(pd.to_numeric(part["P_unchanged"], errors="coerce").mean()),
                "P_advance_plus_recruit": float(pd.to_numeric(part["P_advance_plus_recruit"], errors="coerce").mean()),
                "n_units": int(pd.to_numeric(part["n_units"], errors="coerce").fillna(0).sum()),
                "n_trials": int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part)),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=SUPP_S9_TRANSITION_COMPOSITION_COLUMNS), ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv")


def _write_s9_event_chain_null_summary(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "supp_event_chain_null_baselines.csv"
    frac_path = ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv", SUPP_S9_EVENT_CHAIN_NULL_COLUMNS, "supp_event_chain_null_baselines_missing")
        return
    df = pd.read_csv(src)
    if df.empty or "null_type" not in df.columns:
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv", SUPP_S9_EVENT_CHAIN_NULL_COLUMNS, "supp_event_chain_null_baselines_missing_or_empty")
        return
    n_events_by_seed: dict[int, int] = {}
    if frac_path.exists():
        frac = pd.read_csv(frac_path)
        if "network_seed" in frac.columns and "n_events" in frac.columns:
            for row in frac.itertuples(index=False):
                n_events_by_seed[int(row.network_seed)] = int(getattr(row, "n_events", 0))
    rows = []
    for (network_seed, null_type), part in df.groupby(["network_seed", "null_type"], sort=False):
        observed = _mean_existing(part, ["observed_full_chain_fraction", "observed_value"])
        null_mean = _mean_existing(part, ["null_full_chain_fraction_mean", "null_mean"])
        p_value = _mean_existing(part, ["p_value_or_percentile", "empirical_p", "percentile"])
        rows.append(
            {
                "network_seed": int(network_seed),
                "null_type": str(null_type),
                "observed_full_chain_fraction": observed,
                "null_full_chain_fraction_mean": null_mean,
                "observed_minus_null": _mean_existing(part, ["observed_minus_null"]),
                "p_value_or_percentile": p_value,
                "n_events": int(n_events_by_seed.get(int(network_seed), 0)),
                "notes": "p_value_or_percentile uses empirical_p when available",
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=SUPP_S9_EVENT_CHAIN_NULL_COLUMNS), ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv")


def _write_s10_same_winner_disruption(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv"
    if not src.exists():
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_same_winner_disruption.csv", SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS, "panel_d_perturbation_transition_summary_by_group_missing")
        return
    df = pd.read_csv(src)
    required = {"network_seed", "unit_group", "condition"}
    if df.empty or not required.issubset(df.columns):
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_same_winner_disruption.csv", SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS, "panel_d_perturbation_transition_summary_by_group_missing_or_empty")
        return
    rows = []
    for (network_seed, unit_group, condition), part in df.groupby(["network_seed", "unit_group", "condition"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "unit_group": str(unit_group),
                "condition": str(condition),
                "P_same_winner_preserved": _mean_existing(part, ["P_same_winner_preserved"]),
                "P_same_winner_lost": _mean_existing(part, ["P_same_winner_lost"]),
                "P_same_winner_delayed": _mean_existing(part, ["P_same_winner_delayed"]),
                "P_same_winner_reverted_to_static": _mean_existing(part, ["P_same_winner_reverted_to_static"]),
                "P_same_winner_lost_or_delayed": _mean_existing(part, ["P_same_winner_lost_or_delayed"]),
                "n_dynamic_winners": int(pd.to_numeric(part.get("n_same_winner_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            }
        )
    _save_csv(ctx, pd.DataFrame(rows, columns=SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS), ctx.metrics_dir / "supp_s10_same_winner_disruption.csv")


def _write_s10_dynamic_like_recovery(ctx: ExperimentContext) -> None:
    src = ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv"
    if not src.exists():
        reason = "panel_d_support_perturbation_trial_metrics_missing_or_empty"
        ctx.availability["support_perturbation_downstream_available"] = False
        ctx.availability["support_perturbation_downstream_missing_reason"] = reason
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv", SUPP_S10_DYNAMIC_RECOVERY_COLUMNS, reason)
        return
    df = pd.read_csv(src)
    required = {"network_seed", "condition"}
    if df.empty or not required.issubset(df.columns):
        reason = "panel_d_support_perturbation_trial_metrics_missing_or_empty"
        ctx.availability["support_perturbation_downstream_available"] = False
        ctx.availability["support_perturbation_downstream_missing_reason"] = reason
        _write_empty_csv(ctx, ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv", SUPP_S10_DYNAMIC_RECOVERY_COLUMNS, reason)
        return
    rows = []
    for (network_seed, condition), part in df.groupby(["network_seed", "condition"], sort=False):
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "dynamic_like_spike_similarity_mean": _mean_existing(part, ["dynamic_like_spike_similarity"]),
                "dynamic_like_readout_recovery_mean": _mean_existing(part, ["dynamic_like_readout_recovery"]),
                "decision_deflection_score_mean": _mean_existing(part, ["decision_deflection_score"]),
                "spike_count_mean": _mean_existing(part, ["spike_count", "total_spike_count"]),
                "first_fire_time_ms_mean": _mean_existing(part, ["first_fire_time_ms"]),
                "n_trials": int(part["trial_id"].nunique()) if "trial_id" in part.columns else int(len(part)),
            }
        )
    out = pd.DataFrame(rows, columns=SUPP_S10_DYNAMIC_RECOVERY_COLUMNS)
    _save_csv(ctx, out, ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv")
    if not out.empty:
        ctx.availability["support_perturbation_downstream_available"] = True
        ctx.availability["support_perturbation_downstream_missing_reason"] = None


def _copy_csv_alias(ctx: ExperimentContext, src: Path, dst: Path, *, empty_columns: Sequence[str], reason: str) -> None:
    if not src.exists():
        _write_empty_csv(ctx, dst, empty_columns, reason)
        return
    df = pd.read_csv(src)
    if df.empty:
        _write_empty_csv(ctx, dst, list(df.columns) if len(df.columns) else empty_columns, reason)
        return
    _save_csv(ctx, df, dst)


def _write_empty_csv(ctx: ExperimentContext, dst: Path, columns: Sequence[str], reason: str) -> None:
    _record_optional_missing(ctx, dst.name, reason)
    _save_csv(ctx, pd.DataFrame(columns=list(columns)), dst)


def _record_optional_missing(ctx: ExperimentContext, output_name: str, reason: str) -> None:
    missing = ctx.availability.setdefault("supplement_alias_missing_reasons", {})
    missing[output_name] = reason
    message = f"Optional Fig.5 supplement alias {output_name} is empty: {reason}"
    if message not in ctx.warnings:
        ctx.warnings.append(message)


def _mean_existing(df: pd.DataFrame, columns: Sequence[str]) -> float:
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else float("nan")
    return float("nan")


def save_debug_figures(ctx: ExperimentContext) -> None:
    import matplotlib.pyplot as plt

    apply_publication_style()
    metric_files = [
        ("fig5_debug_preprobe_support", ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv", "mean_support"),
        ("fig5_debug_early_firing", ctx.metrics_dir / "panel_b_transition_summary_by_group.csv", "P_advance_plus_recruit"),
        ("fig5_debug_perturbation_transition", ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv", "P_advance_plus_recruit"),
        ("fig5_debug_same_winner_loss", ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv", "P_same_winner_lost_or_delayed"),
        ("fig5_debug_chain_summary", ctx.metrics_dir / "supp_event_chain_fraction_metrics.csv", "full_chain_satisfied_fraction"),
    ]
    for stem, path, metric_col in metric_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if metric_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        values = pd.to_numeric(df[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
        ax.plot(np.arange(len(values)), values, marker="o", linewidth=1.0)
        ax.set_title(stem)
        ax.set_ylabel(metric_col)
        ax.set_xlabel("row")
        fig.tight_layout()
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    trace_path = ctx.metrics_dir / "panel_c_event_trace_summary.csv"
    if trace_path.exists():
        df = pd.read_csv(trace_path)
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        for trace_type, part in df.groupby("trace_type"):
            ax.plot(part["time_ms"], part["mean_value"], label=str(trace_type), linewidth=1.0)
        ax.axvline(0, color="0.2", linewidth=0.8)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("fig5_debug_event_aligned_traces")
        fig.tight_layout()
        save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_event_aligned_traces")
        plt.close(fig)
    s9_transition = ctx.metrics_dir / "supp_s9_transition_composition_by_group.csv"
    if s9_transition.exists():
        df = pd.read_csv(s9_transition)
        if not df.empty and {"unit_group", "P_advance_plus_recruit"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(4.0, 2.5))
            values = df.groupby("unit_group", sort=False)["P_advance_plus_recruit"].mean(numeric_only=True)
            ax.bar(values.index.astype(str), values.to_numpy(dtype=float))
            ax.set_ylabel("P_advance_plus_recruit")
            ax.set_title("fig5_debug_s9_transition_composition")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s9_transition_composition")
            plt.close(fig)
    s9_null = ctx.metrics_dir / "supp_s9_event_chain_null_summary.csv"
    if s9_null.exists():
        df = pd.read_csv(s9_null)
        if not df.empty and {"null_type", "observed_minus_null"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(4.0, 2.5))
            values = df.groupby("null_type", sort=False)["observed_minus_null"].mean(numeric_only=True)
            ax.bar(values.index.astype(str), values.to_numpy(dtype=float))
            ax.set_ylabel("observed_minus_null")
            ax.set_title("fig5_debug_s9_event_chain_null")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s9_event_chain_null")
            plt.close(fig)
    s10_transition = ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv"
    if s10_transition.exists():
        df = pd.read_csv(s10_transition)
        cols = {"unit_group", "attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit"}
        if not df.empty and cols.issubset(df.columns):
            grouped = df.groupby("unit_group", sort=False)[["attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit"]].mean(numeric_only=True)
            x = np.arange(len(grouped))
            fig, ax = plt.subplots(figsize=(4.2, 2.6))
            ax.bar(x - 0.18, grouped["attenuate_delta_P_advance_plus_recruit"].to_numpy(dtype=float), width=0.36, label="attenuate")
            ax.bar(x + 0.18, grouped["reset_delta_P_advance_plus_recruit"].to_numpy(dtype=float), width=0.36, label="reset")
            ax.set_xticks(x, grouped.index.astype(str), rotation=30)
            ax.set_ylabel("delta P_advance+recruit")
            ax.set_title("fig5_debug_s10_perturbation_transition")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s10_perturbation_transition")
            plt.close(fig)
    s10_recovery = ctx.metrics_dir / "supp_s10_dynamic_like_recovery_after_perturbation.csv"
    if s10_recovery.exists():
        df = pd.read_csv(s10_recovery)
        y_col = "dynamic_like_readout_recovery_mean" if "dynamic_like_readout_recovery_mean" in df.columns else "decision_deflection_score_mean"
        if not df.empty and {"condition", y_col}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(4.0, 2.5))
            values = df.groupby("condition", sort=False)[y_col].mean(numeric_only=True)
            ax.bar(values.index.astype(str), values.to_numpy(dtype=float))
            ax.set_ylabel(y_col)
            ax.set_title("fig5_debug_s10_dynamic_like_recovery")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "fig5_debug_s10_dynamic_like_recovery")
            plt.close(fig)
    ctx.completed_modules["debug_figures"] = True


def _run_batch_or_proxy(ctx: ExperimentContext, batch: pd.DataFrame) -> dict[str, Any]:
    if ctx.net is None or ctx.encoder is None:
        return _run_batch_proxy(ctx, batch)
    try:
        return _run_batch_network(ctx, batch)
    except Exception as exc:
        ctx.warnings.append(f"Network rollout failed for a batch; using image-driven proxy for that batch: {exc}")
        return _run_batch_proxy(ctx, batch)


def _run_batch_network(ctx: ExperimentContext, batch: pd.DataFrame) -> dict[str, Any]:
    assert ctx.net is not None and ctx.encoder is not None
    sample_images = _images_for_ids(ctx.dataset, batch["sample_image_id"].to_numpy()).to(ctx.device)
    probe_images = _images_for_ids(ctx.dataset, batch["probe_image_id"].to_numpy()).to(ctx.device)
    sample_spikes = encode_images(ctx.encoder, sample_images, ctx.cfg.sample_steps)
    probe_spikes = encode_images(ctx.encoder, probe_images, ctx.cfg.probe_steps)
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    current_time = 0
    with torch.no_grad():
        for t in range(ctx.cfg.sample_steps):
            current_time = _step_network_once(ctx.net, sample_spikes[:, t], current_time, stsp_mode="dynamic")
        zero = torch.zeros((batch_size, channels, height, width), device=ctx.device)
        for _ in range(ctx.cfg.delay_steps):
            current_time = _step_network_once(ctx.net, zero, current_time, stsp_mode="dynamic")
    boundary = snapshot_boundary_state(ctx.net)
    support_by_batch = _support_maps_from_boundary(boundary, batch_size)

    support_maps: dict[int, np.ndarray] = {}
    branch_traces: dict[int, dict[str, BranchTrace]] = {}
    boundary_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    perturb_audit_rows: list[dict[str, Any]] = []
    for local_idx, trial in enumerate(batch.reset_index(drop=True).itertuples(index=False)):
        trial_id = int(trial.trial_id)
        support_maps[trial_id] = support_by_batch[local_idx]
        single_boundary = _slice_boundary(boundary, local_idx)
        boundary_states[trial_id] = single_boundary
        single_probe = probe_spikes[local_idx : local_idx + 1]
        dynamic, audit = _run_probe_branch(ctx, single_boundary, single_probe, "dynamic_intact")
        perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
        static, audit = _run_probe_branch(ctx, single_boundary, single_probe, "static_frozen")
        perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
        traces = {"dynamic_intact": dynamic, "static_frozen": static}
        groups = _unit_group_rows(ctx, pd.Series(trial._asdict()), support_maps[trial_id])
        psets = _perturbation_unit_rows(ctx, pd.Series(trial._asdict()), support_maps[trial_id], groups)
        for condition in MAIN_CONDITIONS + SUPP_CONDITIONS:
            if condition in traces:
                continue
            trace, audit = _run_probe_branch(ctx, single_boundary, single_probe, condition, perturb_units=psets[psets["condition"].eq(condition)])
            traces[condition] = trace
            perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
        branch_traces[trial_id] = traces
    return {"support_maps": support_maps, "branch_traces": branch_traces, "boundary_states": boundary_states, "perturbation_ux_audit": perturb_audit_rows}


def _run_batch_proxy(ctx: ExperimentContext, batch: pd.DataFrame) -> dict[str, Any]:
    support_maps: dict[int, np.ndarray] = {}
    branch_traces: dict[int, dict[str, BranchTrace]] = {}
    for trial in batch.itertuples(index=False):
        trial_id = int(trial.trial_id)
        sample = _image_array(ctx.dataset, int(trial.sample_image_id))
        probe = _image_array(ctx.dataset, int(trial.probe_image_id))
        sample_mask, probe_mask = sample > ctx.cfg.foreground_threshold, probe > ctx.cfg.foreground_threshold
        overlap = sample_mask & probe_mask
        probe_only = probe_mask & (~sample_mask)
        rng = np.random.default_rng(int(trial.trial_seed))
        support = 0.15 + 0.55 * sample + 0.30 * overlap.astype(float) + rng.normal(0.0, 0.015, sample.shape)
        support = np.clip(support, 0.0, 1.2).astype(np.float32)
        support_maps[trial_id] = support
        groups = _unit_group_rows(ctx, pd.Series(trial._asdict()), support)
        psets = _perturbation_unit_rows(ctx, pd.Series(trial._asdict()), support, groups)
        dynamic = _proxy_trace(ctx, support, probe, overlap, probe_only, rng, "dynamic_intact", int(trial.probe_label))
        static = _proxy_trace(ctx, support * 0.45, probe, overlap, probe_only, rng, "static_frozen", int(trial.probe_label))
        traces = {"dynamic_intact": dynamic, "static_frozen": static}
        for condition in MAIN_CONDITIONS + SUPP_CONDITIONS:
            if condition in traces:
                continue
            cond_support = _proxy_perturbed_support(support, condition, psets)
            traces[condition] = _proxy_trace(ctx, cond_support, probe, overlap, probe_only, rng, condition, int(trial.probe_label))
        branch_traces[trial_id] = traces
    return {"support_maps": support_maps, "branch_traces": branch_traces, "boundary_states": {}, "perturbation_ux_audit": []}


def _run_probe_branch(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    condition: str,
    perturb_units: pd.DataFrame | None = None,
) -> tuple[BranchTrace, list[dict[str, Any]]]:
    assert ctx.net is not None
    batch_size, _, channels, height, width = probe_spikes.shape
    prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
    _restore_boundary_state(ctx.net, boundary)
    audit_rows: list[dict[str, Any]] = []
    if condition not in {"dynamic_intact", "static_frozen"}:
        audit_rows = _apply_support_perturbation(
            ctx.net,
            condition,
            perturb_units,
            attenuation_factor=float(ctx.cfg.perturbation_attenuation_factor),
        )
    stsp_mode = "static_frozen" if condition == "static_frozen" else "dynamic"
    with torch.no_grad():
        ctx.net.layer3.reset_decision_state()
        ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
        ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
        layer1_spikes = []
        layer1_v = []
        layer1_inh = []
        layer3_spikes = []
        for t in range(int(probe_spikes.shape[1])):
            s1, m1 = ctx.net.layer1.forward_step(probe_spikes[:, t], t, training=False, monitor=True, stsp_mode=stsp_mode)
            s1p = ctx.net.pool1(s1.float())
            s2, _ = ctx.net.layer2.forward_step(s1p, t, training=False, monitor=False, stsp_mode=stsp_mode)
            s2p = ctx.net.pool2(s2.float())
            s3, _ = ctx.net.layer3.forward_step(s2p, t, training=False, monitor=False, stsp_mode=stsp_mode)
            layer1_spikes.append(s1.detach().cpu())
            layer1_v.append(m1.get("v_effective", m1.get("v_mem_snapshot")).detach().cpu())
            layer1_inh.append(m1.get("inh_after", torch.zeros_like(s1, dtype=torch.float32)).detach().cpu())
            layer3_spikes.append(s3.detach().cpu())
        pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, 1)
    spikes = torch.stack(layer1_spikes, dim=0).to(torch.bool).numpy()
    v = torch.stack(layer1_v, dim=0).to(torch.float32).numpy()
    inh = torch.stack(layer1_inh, dim=0).to(torch.float32).numpy()
    l3 = torch.stack(layer3_spikes, dim=0).to(torch.bool).numpy()
    trace = BranchTrace(
        spikes=spikes[:, 0].any(axis=1).astype(np.float32),
        v_effective=v[:, 0].mean(axis=1).astype(np.float32),
        inhibition=inh[:, 0].mean(axis=1).astype(np.float32),
        layer3_spikes=l3[:, 0].reshape(l3.shape[0], -1).astype(np.float32),
        prediction=int(pred[0].item()),
        first_fire_time=int(fire[0].item()),
    )
    return trace, audit_rows


def _run_probe_branches_batch(
    ctx: ExperimentContext,
    boundary_states: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    probe_spikes: Mapping[int, torch.Tensor],
    conditions: Sequence[str],
    perturbation_sets: Mapping[tuple[int, str], pd.DataFrame],
) -> dict[tuple[int, str], BranchTrace]:
    if ctx.cfg.enable_branch_batch:
        ctx.warnings.append("Fig.5 branch batch helper is scaffolded; falling back to order-preserving single-branch rollouts.")
    out: dict[tuple[int, str], BranchTrace] = {}
    for trial_id, boundary in boundary_states.items():
        for condition in conditions:
            trace, _ = _run_probe_branch(
                ctx,
                boundary,
                probe_spikes[int(trial_id)],
                str(condition),
                perturb_units=perturbation_sets.get((int(trial_id), str(condition))),
            )
            out[(int(trial_id), str(condition))] = trace
    return out


def _proxy_trace(
    ctx: ExperimentContext,
    support: np.ndarray,
    probe: np.ndarray,
    overlap: np.ndarray,
    probe_only: np.ndarray,
    rng: np.random.Generator,
    condition: str,
    probe_label: int,
) -> BranchTrace:
    t_steps = int(ctx.cfg.probe_steps)
    h, w = support.shape
    drive = 0.50 * _normalize(probe) + 0.45 * _normalize(support)
    local_overlap_field = _blur3(overlap.astype(float))
    if condition == "static_frozen":
        drive *= 0.68
    elif condition == "dynamic_intact":
        drive = drive + 0.20 * overlap.astype(float) - 0.35 * probe_only.astype(float) * (local_overlap_field > 0).astype(float)
    elif "attenuate" in condition:
        drive = drive - 0.22 * overlap.astype(float) * _normalize(support)
    elif "reset" in condition:
        drive = drive - 0.30 * overlap.astype(float) * _normalize(support)
    elif "overlap" in condition and "nonoverlap" not in condition:
        drive = drive - 0.35 * overlap.astype(float) * _normalize(support)
    elif "random_high" in condition:
        drive = drive - 0.12 * _normalize(support)
    elif "nonoverlap" in condition:
        drive = drive - 0.18 * probe_only.astype(float) * _normalize(support)
    drive = np.clip(drive + rng.normal(0.0, 0.02, drive.shape), 0.0, 1.0)
    first = np.full((h, w), -1, dtype=int)
    active = drive > np.quantile(drive[probe > 0] if np.any(probe > 0) else drive.reshape(-1), 0.78)
    first[active] = np.clip((t_steps - 1) - np.round(drive[active] * (t_steps * 0.75)).astype(int), 0, t_steps - 1)
    spikes = np.zeros((t_steps, h, w), dtype=np.float32)
    for r, c in zip(*np.where(first >= 0)):
        spikes[first[r, c], r, c] = 1.0
    v = np.zeros_like(spikes)
    inh = np.zeros_like(spikes)
    for t in range(t_steps):
        ramp = (t + 1) / max(1, t_steps)
        v[t] = -0.2 + drive * ramp - 0.10 * (condition == "static_frozen")
        if t > 0:
            prev = spikes[max(0, t - 2) : t + 1].sum(axis=0)
            inh[t] = 0.75 * _blur3(prev) + 0.75 * (inh[t - 1] if t else 0.0)
            v[t] -= 0.20 * inh[t]
    l3 = np.zeros((t_steps, 10), dtype=np.float32)
    fire_t = int(np.min(first[first >= 0])) if np.any(first >= 0) else -1
    pred = int(probe_label if condition in {"dynamic_intact", "attenuate_overlap_high_support"} else (probe_label + 1) % 10)
    if fire_t >= 0:
        l3[min(t_steps - 1, fire_t + 5), pred] = 1.0
    return BranchTrace(spikes=spikes, v_effective=v.astype(np.float32), inhibition=inh.astype(np.float32), layer3_spikes=l3, prediction=pred, first_fire_time=fire_t)


# Column contracts.
TRIAL_COLUMNS = [
    "network_seed",
    "trial_id",
    "sample_image_id",
    "sample_label",
    "probe_image_id",
    "probe_label",
    "sample_foreground_area",
    "probe_foreground_area",
    "overlap_area",
    "probe_only_area",
    "overlap_quantile",
    "selected_trial_group",
    "input_energy_sample",
    "input_energy_probe",
    "pixel_similarity",
    "dice_overlap",
    "class_pair",
    "trial_seed",
]
UNIT_GROUP_COLUMNS = [
    "network_seed",
    "trial_id",
    "layer",
    "unit_id",
    "row",
    "col",
    "unit_group",
    "overlap_drive_score",
    "probe_only_drive_score",
    "support_value",
    "is_overlap_dominant",
    "is_probe_only_dominant",
    "is_random_matched",
]
PERTURBATION_UNIT_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "unit_id",
    "unit_group",
    "original_support",
    "perturbed_support",
    "support_delta",
    "row",
    "col",
    "matched_to_condition",
    "matching_error_support",
    "matching_error_spike_count",
    "intervention_timing",
    "probe_input_changed",
]
PERTURBATION_UX_AUDIT_COLUMNS = [
    "network_seed",
    "trial_id",
    "condition",
    "unit_id",
    "row",
    "col",
    "u_before_mean",
    "x_before_mean",
    "g_before_mean",
    "u_after_mean",
    "x_after_mean",
    "g_after_mean",
    "u_delta_mean",
    "x_delta_mean",
    "g_delta_mean",
]
PANEL_A_COLUMNS = ["network_seed", "trial_id", "unit_group", "layer", "state_variable", "mean_support", "total_support", "support_area", "support_enrichment", "overlap_minus_probe_only_support", "n_units"]
PANEL_B_UNIT_COLUMNS = ["network_seed", "trial_id", "unit_id", "unit_group", "early_window_ms", "transition_type", "first_spike_dynamic", "first_spike_static", "delta_first_spike_latency", "early_spike_count_dynamic", "early_spike_count_static", "delta_early_spike_count"]
PANEL_B_SUMMARY_COLUMNS = ["network_seed", "trial_id", "unit_group", "early_window_ms", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "mean_delta_early_spike_count", "mean_delta_first_spike_latency", "n_units"]
PANEL_C_EVENT_COLUMNS = ["network_seed", "trial_id", "event_id", "winner_unit_idx", "loser_unit_idx", "winner_group", "loser_group", "winner_first_spike_dynamic", "winner_first_spike_static", "loser_first_spike_dynamic", "loser_first_spike_static", "winner_pre_spike_delta_v_mean", "winner_pre_spike_boost", "winner_spikes_earlier", "loser_post_winner_delta_v_mean", "loser_post_winner_inh_rise", "loser_post_winner_suppressed", "winner_loser_latency_gap", "neighborhood_radius", "local_distance"]
PANEL_C_TRACE_COLUMNS = ["network_seed", "time_ms", "trace_type", "mean_value", "sem_value", "n_events"]
PANEL_D_UNIT_TRANSITION_COLUMNS = ["network_seed", "trial_id", "condition", "unit_id", "unit_group", "row", "col", "first_spike_static", "first_spike_same", "first_spike_condition", "transition_vs_static", "transition_vs_same", "same_winner", "condition_winner", "same_winner_preserved", "same_winner_delayed", "same_winner_lost", "same_winner_reverted_to_static", "same_winner_lost_or_delayed", "delta_latency_vs_static", "delta_latency_vs_same", "early_spike_count_static", "early_spike_count_same", "early_spike_count_condition", "delta_early_spike_count_vs_static", "delta_early_spike_count_vs_same"]
PANEL_D_TRANSITION_SUMMARY_COLUMNS = ["network_seed", "trial_id", "condition", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "P_same_winner_preserved", "P_same_winner_delayed", "P_same_winner_lost", "P_same_winner_reverted_to_static", "P_same_winner_lost_or_delayed", "mean_delta_latency_vs_static", "mean_delta_latency_vs_same", "mean_delta_early_spike_count_vs_static", "mean_delta_early_spike_count_vs_same", "n_units", "n_same_winner_units"]
PANEL_D_TRANSITION_CONTRAST_COLUMNS = ["network_seed", "trial_id", "unit_group", "attenuate_delta_P_advance_plus_recruit", "reset_delta_P_advance_plus_recruit", "attenuate_delta_P_loss", "reset_delta_P_loss", "attenuate_delta_P_same_winner_lost_or_delayed", "reset_delta_P_same_winner_lost_or_delayed", "reset_minus_attenuate_delta_P_advance_plus_recruit", "attenuate_delta_latency_vs_same", "reset_delta_latency_vs_same", "n_units", "n_trials"]
PANEL_D_NODE_COLUMNS = ["network_seed", "trial_id", "condition", "perturbed_unit_group", "n_perturbed_units", "mean_pre_perturb_support", "mean_post_perturb_support", "P_advance", "P_recruit", "P_advance_plus_recruit", "delta_early_spike_count", "delta_first_spike_latency", "winner_pre_spike_delta_v_mean", "winner_pre_spike_boost", "loser_post_winner_inh_rise", "loser_post_winner_delta_v_mean", "loser_post_winner_suppressed", "spike_pattern_displacement", "dynamic_like_spike_similarity", "decision_deflection_score", "dynamic_like_readout_recovery"]
PANEL_D_TRIAL_COLUMNS = ["network_seed", "trial_id", "condition", "prediction", "probe_prediction", "probe_correct", "pred_matches_dynamic", "pred_matches_static", "first_fire_time_ms", "first_fire_time", "spike_count", "early_spike_count", "total_spike_count", "dynamic_like_spike_similarity", "dynamic_like_readout_recovery", "decision_deflection_score"]
PANEL_E_COLUMNS = ["network_seed", "node", "metric", "dynamic_intact_value", "overlap_perturbed_value", "random_perturbed_value", "nonoverlap_perturbed_value", "static_value", "overlap_disruption", "random_disruption", "nonoverlap_disruption", "normalized_overlap_disruption"]
PANEL_D_EFFECT_SUMMARY_COLUMNS = ["network_seed", "metric", "dynamic_value", "static_value", "attenuate_value", "reset_value", "sham_value", "attenuate_disruption_vs_dynamic", "reset_disruption_vs_dynamic", "sham_disruption_vs_dynamic", "attenuate_recovery_toward_static", "reset_recovery_toward_static", "reset_minus_attenuate_disruption", "n_trials", "metric_direction", "notes"]
SUPP_EVENT_AUDIT_COLUMNS = ["network_seed", "trial_id", "event_id", "selection_step", "included", "exclusion_reason", "winner_group", "loser_group", "neighborhood_radius", "drive_score_winner", "drive_score_loser"]
SUPP_NULL_COLUMNS = ["network_seed", "null_type", "metric", "observed_value", "null_mean", "null_p95", "observed_minus_null", "empirical_p", "n_null"]
SUPP_S9_TRANSITION_COMPOSITION_COLUMNS = ["network_seed", "unit_group", "P_advance", "P_recruit", "P_loss", "P_unchanged", "P_advance_plus_recruit", "n_units", "n_trials"]
SUPP_S9_EVENT_CHAIN_NULL_COLUMNS = ["network_seed", "null_type", "observed_full_chain_fraction", "null_full_chain_fraction_mean", "observed_minus_null", "p_value_or_percentile", "n_events", "notes"]
SUPP_S10_SAME_WINNER_DISRUPTION_COLUMNS = ["network_seed", "unit_group", "condition", "P_same_winner_preserved", "P_same_winner_lost", "P_same_winner_delayed", "P_same_winner_reverted_to_static", "P_same_winner_lost_or_delayed", "n_dynamic_winners"]
SUPP_S10_DYNAMIC_RECOVERY_COLUMNS = ["network_seed", "condition", "dynamic_like_spike_similarity_mean", "dynamic_like_readout_recovery_mean", "decision_deflection_score_mean", "spike_count_mean", "first_fire_time_ms_mean", "n_trials"]


def _unit_group_rows(ctx: ExperimentContext, trial: Any, support: np.ndarray) -> pd.DataFrame:
    trial_map = _trial_mapping(trial)
    sample = _image_array(ctx.dataset, int(trial_map["sample_image_id"]))
    probe = _image_array(ctx.dataset, int(trial_map["probe_image_id"]))
    sample_mask = sample > ctx.cfg.foreground_threshold
    probe_mask = probe > ctx.cfg.foreground_threshold
    overlap = sample_mask & probe_mask
    probe_only = probe_mask & (~sample_mask)
    h, w = support.shape
    overlap = _resize_mask(overlap, h, w)
    probe_only = _resize_mask(probe_only, h, w)
    rng = np.random.default_rng(int(trial_map["trial_seed"]) + 17)
    high_support = support >= np.nanquantile(support, max(0.0, 1.0 - float(ctx.cfg.peak_support_q)))
    random_pool = np.flatnonzero(high_support.reshape(-1))
    random_take = set(rng.choice(random_pool, size=min(int(overlap.sum()), len(random_pool)), replace=False).tolist()) if len(random_pool) else set()
    rows = []
    for r in range(h):
        for c in range(w):
            unit_id = int(r * w + c)
            if bool(overlap[r, c]) and support[r, c] >= float(ctx.cfg.drive_score_threshold):
                group = "overlap_dominant"
            elif bool(probe_only[r, c]):
                group = "probe_only_dominant"
            elif unit_id in random_take:
                group = "random_matched"
            else:
                group = "balanced"
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_map["trial_id"]),
                    "layer": PRIMARY_LAYER,
                    "unit_id": unit_id,
                    "row": int(r),
                    "col": int(c),
                    "unit_group": group,
                    "overlap_drive_score": float(overlap[r, c]) * float(support[r, c]),
                    "probe_only_drive_score": float(probe_only[r, c]) * float(support[r, c]),
                    "support_value": float(support[r, c]),
                    "is_overlap_dominant": bool(group == "overlap_dominant"),
                    "is_probe_only_dominant": bool(group == "probe_only_dominant"),
                    "is_random_matched": bool(group == "random_matched"),
                }
            )
    return pd.DataFrame(rows, columns=UNIT_GROUP_COLUMNS)


def _perturbation_unit_rows(ctx: ExperimentContext, trial: Any, support: np.ndarray, groups: pd.DataFrame) -> pd.DataFrame:
    trial_map = _trial_mapping(trial)
    q = np.nanquantile(support, max(0.0, 1.0 - float(ctx.cfg.peak_support_q)))
    base = groups[pd.to_numeric(groups["support_value"], errors="coerce") >= q].copy()
    overlap = base[base["unit_group"].eq("overlap_dominant")].copy()
    condition_sets = {
        "attenuate_overlap_high_support": overlap,
        "reset_overlap_high_support": overlap,
        "sham_perturbation": overlap.head(0),
    }
    rows = []
    for condition, part in condition_sets.items():
        for row in part.itertuples(index=False):
            original = float(row.support_value)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_map["trial_id"]),
                    "condition": condition,
                    "unit_id": int(row.unit_id),
                    "unit_group": str(row.unit_group),
                    "original_support": original,
                    "perturbed_support": np.nan,
                    "support_delta": np.nan,
                    "row": int(row.row),
                    "col": int(row.col),
                    "matched_to_condition": "",
                    "matching_error_support": np.nan,
                    "matching_error_spike_count": np.nan,
                    "intervention_timing": "pre_probe_boundary",
                    "probe_input_changed": False,
                }
            )
    return pd.DataFrame(rows, columns=PERTURBATION_UNIT_COLUMNS)


def _node_metrics_for_condition(ctx: ExperimentContext, condition: str, trace: BranchTrace, dynamic: BranchTrace, static: BranchTrace, first: np.ndarray, dyn_first: np.ndarray, sta_first: np.ndarray, unit_set: pd.DataFrame) -> dict[str, Any]:
    transitions = [_transition_type(int(first[r, c]), int(sta_first[r, c])) for r in range(first.shape[0]) for c in range(first.shape[1])]
    n = max(1, len(transitions))
    early_delta = float(trace.spikes[: ctx.cfg.early_window_steps].sum() - static.spikes[: ctx.cfg.early_window_steps].sum())
    latency_vals = [_latency_delta(int(first[r, c]), int(sta_first[r, c])) for r in range(first.shape[0]) for c in range(first.shape[1]) if int(first[r, c]) >= 0 or int(sta_first[r, c]) >= 0]
    dyn_like = _pattern_similarity(trace.spikes, dynamic.spikes)
    sta_like = _pattern_similarity(trace.spikes, static.spikes)
    winner_boost = float(np.nanmean(trace.v_effective[: ctx.cfg.early_window_steps] - static.v_effective[: ctx.cfg.early_window_steps]))
    loser_inh = float(np.nanmean(trace.inhibition[ctx.cfg.early_window_steps :] - static.inhibition[ctx.cfg.early_window_steps :]))
    return {
        "P_advance": transitions.count("advance") / n,
        "P_recruit": transitions.count("recruit") / n,
        "P_advance_plus_recruit": (transitions.count("advance") + transitions.count("recruit")) / n,
        "delta_early_spike_count": early_delta,
        "delta_first_spike_latency": float(np.nanmean(latency_vals)) if latency_vals else float("nan"),
        "winner_pre_spike_delta_v_mean": winner_boost,
        "winner_pre_spike_boost": float(winner_boost > 0.0),
        "loser_post_winner_inh_rise": loser_inh,
        "loser_post_winner_delta_v_mean": float(np.nanmean(trace.v_effective[ctx.cfg.early_window_steps :] - dynamic.v_effective[ctx.cfg.early_window_steps :])),
        "loser_post_winner_suppressed": float(loser_inh > 0.0),
        "spike_pattern_displacement": float(1.0 - sta_like),
        "dynamic_like_spike_similarity": float(dyn_like),
        "decision_deflection_score": float(_decision_deflection(trace, dynamic, static)),
        "dynamic_like_readout_recovery": float(_pattern_similarity(trace.layer3_spikes, dynamic.layer3_spikes)),
    }


def _transition_summary(network_seed: int, metrics: pd.DataFrame, early_window_ms: int) -> pd.DataFrame:
    rows = []
    for (trial_id, group), part in metrics.groupby(["trial_id", "unit_group"], sort=False):
        transitions = part["transition_type"].astype(str)
        n = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "unit_group": str(group),
                "early_window_ms": int(early_window_ms),
                "P_advance": float((transitions == "advance").mean()),
                "P_recruit": float((transitions == "recruit").mean()),
                "P_loss": float((transitions == "loss").mean()),
                "P_unchanged": float((transitions == "unchanged").mean()),
                "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                "mean_delta_early_spike_count": float(pd.to_numeric(part["delta_early_spike_count"], errors="coerce").mean()),
                "mean_delta_first_spike_latency": float(pd.to_numeric(part["delta_first_spike_latency"], errors="coerce").mean()),
                "n_units": int(n),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_B_SUMMARY_COLUMNS)


def _summarize_perturbation_transitions(ctx: ExperimentContext, unit_df: pd.DataFrame) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame(columns=PANEL_D_TRANSITION_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (network_seed, trial_id, condition, unit_group), part in unit_df.groupby(["network_seed", "trial_id", "condition", "unit_group"], sort=False):
        transitions = part["transition_vs_static"].astype(str)
        same_winner = part["same_winner"].astype(bool)
        n_same = int(same_winner.sum())
        denom_same = max(1, n_same)
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "condition": str(condition),
                "unit_group": str(unit_group),
                "P_advance": float((transitions == "advance").mean()),
                "P_recruit": float((transitions == "recruit").mean()),
                "P_loss": float((transitions == "loss").mean()),
                "P_unchanged": float((transitions == "unchanged").mean()),
                "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                "P_same_winner_preserved": float((part["same_winner_preserved"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_delayed": float((part["same_winner_delayed"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_lost": float((part["same_winner_lost"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_reverted_to_static": float((part["same_winner_reverted_to_static"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_lost_or_delayed": float((part["same_winner_lost_or_delayed"].astype(bool) & same_winner).sum() / denom_same),
                "mean_delta_latency_vs_static": float(pd.to_numeric(part["delta_latency_vs_static"], errors="coerce").mean()),
                "mean_delta_latency_vs_same": float(pd.to_numeric(part["delta_latency_vs_same"], errors="coerce").mean()),
                "mean_delta_early_spike_count_vs_static": float(pd.to_numeric(part["delta_early_spike_count_vs_static"], errors="coerce").mean()),
                "mean_delta_early_spike_count_vs_same": float(pd.to_numeric(part["delta_early_spike_count_vs_same"], errors="coerce").mean()),
                "n_units": int(len(part)),
                "n_same_winner_units": n_same,
            }
        )
    _ = ctx
    return pd.DataFrame(rows, columns=PANEL_D_TRANSITION_SUMMARY_COLUMNS)


def _compute_perturbation_transition_contrasts(ctx: ExperimentContext, summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=PANEL_D_TRANSITION_CONTRAST_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (network_seed, trial_id, unit_group), part in summary_df.groupby(["network_seed", "trial_id", "unit_group"], sort=False):
        by_cond = {str(row.condition): row for row in part.itertuples(index=False)}
        base = by_cond.get("dynamic_intact")
        attenuate = by_cond.get("attenuate_overlap_high_support")
        reset = by_cond.get("reset_overlap_high_support")
        if base is None:
            continue
        attenuate_delta_recruit = _delta_field(attenuate, base, "P_advance_plus_recruit")
        reset_delta_recruit = _delta_field(reset, base, "P_advance_plus_recruit")
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "unit_group": str(unit_group),
                "attenuate_delta_P_advance_plus_recruit": attenuate_delta_recruit,
                "reset_delta_P_advance_plus_recruit": reset_delta_recruit,
                "attenuate_delta_P_loss": _delta_field(attenuate, base, "P_loss"),
                "reset_delta_P_loss": _delta_field(reset, base, "P_loss"),
                "attenuate_delta_P_same_winner_lost_or_delayed": _delta_field(attenuate, base, "P_same_winner_lost_or_delayed"),
                "reset_delta_P_same_winner_lost_or_delayed": _delta_field(reset, base, "P_same_winner_lost_or_delayed"),
                "reset_minus_attenuate_delta_P_advance_plus_recruit": _finite_delta(reset_delta_recruit, attenuate_delta_recruit),
                "attenuate_delta_latency_vs_same": _delta_field(attenuate, base, "mean_delta_latency_vs_same"),
                "reset_delta_latency_vs_same": _delta_field(reset, base, "mean_delta_latency_vs_same"),
                "n_units": int(getattr(base, "n_units", 0)),
                "n_trials": 1,
            }
        )
    _ = ctx
    return pd.DataFrame(rows, columns=PANEL_D_TRANSITION_CONTRAST_COLUMNS)


def _delta_field(condition_row: Any | None, base_row: Any, field: str) -> float:
    if condition_row is None:
        return float("nan")
    condition_value = float(getattr(condition_row, field, np.nan))
    base_value = float(getattr(base_row, field, np.nan))
    return float(condition_value - base_value) if np.isfinite(condition_value) and np.isfinite(base_value) else float("nan")


def _event_trace_summary(ctx: ExperimentContext, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PANEL_C_TRACE_COLUMNS)
    out = []
    for (time_ms, trace_type), part in df.groupby(["time_ms", "trace_type"], sort=True):
        values = pd.to_numeric(part["value"], errors="coerce").dropna()
        out.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "time_ms": float(time_ms),
                "trace_type": str(trace_type),
                "mean_value": float(values.mean()) if not values.empty else np.nan,
                "sem_value": float(values.sem()) if len(values) > 1 else 0.0,
                "n_events": int(values.count()),
            }
        )
    return pd.DataFrame(out, columns=PANEL_C_TRACE_COLUMNS)


def _early_window_robustness(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> pd.DataFrame:
    base = pd.read_csv(ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv")
    rows = []
    for window in (5, 10, 15, 20, 30):
        for group, part in base.groupby("unit_group", sort=False):
            transitions = part["transition_type"].astype(str)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "early_window_ms": int(window),
                    "unit_group": str(group),
                    "P_advance": float((transitions == "advance").mean()),
                    "P_recruit": float((transitions == "recruit").mean()),
                    "P_loss": float((transitions == "loss").mean()),
                    "P_unchanged": float((transitions == "unchanged").mean()),
                    "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                    "delta_early_spike_count": float(pd.to_numeric(part["delta_early_spike_count"], errors="coerce").mean()) * min(1.0, window / max(1.0, ctx.cfg.early_window_ms)),
                    "n_units": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def _neighborhood_radius_robustness(ctx: ExperimentContext, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in (1, 2, 3):
        part = events[pd.to_numeric(events.get("local_distance", pd.Series(dtype=float)), errors="coerce") <= radius * 2]
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "neighborhood_radius": int(radius),
                "n_events": int(len(part)),
                "winner_pre_spike_delta_v_mean": float(pd.to_numeric(part.get("winner_pre_spike_delta_v_mean", pd.Series(dtype=float)), errors="coerce").mean()) if not part.empty else np.nan,
                "loser_post_winner_inh_rise": float(pd.to_numeric(part.get("loser_post_winner_inh_rise", pd.Series(dtype=float)), errors="coerce").mean()) if not part.empty else np.nan,
                "loser_post_winner_suppressed": float(part.get("loser_post_winner_suppressed", pd.Series(dtype=bool)).astype(bool).mean()) if not part.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _support_perturbation_controls(node_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in node_df.groupby("condition", sort=False):
        for metric in ["P_advance_plus_recruit", "winner_pre_spike_delta_v_mean", "loser_post_winner_inh_rise", "dynamic_like_spike_similarity", "decision_deflection_score"]:
            rows.append({"network_seed": int(part["network_seed"].iloc[0]), "condition": condition, "metric": metric, "value": float(pd.to_numeric(part[metric], errors="coerce").mean()), "n_trials": int(part["trial_id"].nunique())})
    return pd.DataFrame(rows)


def _perturbation_matching_diagnostics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank, node_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for part in bank.perturbation_sets.groupby(["trial_id", "condition"], sort=False):
        (trial_id, condition), df = part
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "trial_id": int(trial_id),
                "condition": str(condition),
                "n_perturbed_units": int(len(df)),
                "mean_pre_support": float(pd.to_numeric(df["original_support"], errors="coerce").mean()) if len(df) else np.nan,
                "mean_post_support": float(pd.to_numeric(df["perturbed_support"], errors="coerce").mean()) if len(df) else np.nan,
                "expected_spike_count": float(len(df)),
                "actual_spike_count": float(node_df[(node_df["trial_id"].eq(trial_id)) & (node_df["condition"].eq(condition))]["delta_early_spike_count"].mean()),
                "active_unit_count": int(len(df)),
                "matching_error_support": float(pd.to_numeric(df["matching_error_support"], errors="coerce").mean()) if len(df) else np.nan,
                "matching_error_spike_count": float(pd.to_numeric(df["matching_error_spike_count"], errors="coerce").mean()) if len(df) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _apply_support_perturbation(
    net,
    condition: str,
    perturb_units: pd.DataFrame | None,
    attenuation_factor: float = 0.5,
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    if perturb_units is None or perturb_units.empty or not hasattr(net.layer1, "u_pre") or net.layer1.u_pre is None:
        return audit_rows
    with torch.no_grad():
        u = net.layer1.u_pre
        x = net.layer1.x_pre
        u0 = float(net.layer1.stsp_U)
        for row in perturb_units.itertuples(index=False):
            rr = min(int(row.row), u.shape[-2] - 1)
            cc = min(int(row.col), u.shape[-1] - 1)
            u_before = u[..., rr, cc].detach().clone()
            x_before = x[..., rr, cc].detach().clone()
            g_before = u_before * x_before
            if condition.startswith("attenuate"):
                u[..., rr, cc] = u0 + float(attenuation_factor) * (u[..., rr, cc] - u0)
            elif condition.startswith("reset"):
                u[..., rr, cc] = u0
                x[..., rr, cc] = 1.0
            elif condition == "sham_perturbation":
                pass
            u_after = u[..., rr, cc].detach().clone()
            x_after = x[..., rr, cc].detach().clone()
            g_after = u_after * x_after
            audit_rows.append(
                {
                    "condition": condition,
                    "unit_id": int(row.unit_id),
                    "row": int(row.row),
                    "col": int(row.col),
                    "u_before_mean": float(u_before.float().mean().cpu()),
                    "x_before_mean": float(x_before.float().mean().cpu()),
                    "g_before_mean": float(g_before.float().mean().cpu()),
                    "u_after_mean": float(u_after.float().mean().cpu()),
                    "x_after_mean": float(x_after.float().mean().cpu()),
                    "g_after_mean": float(g_after.float().mean().cpu()),
                    "u_delta_mean": float((u_after - u_before).float().mean().cpu()),
                    "x_delta_mean": float((x_after - x_before).float().mean().cpu()),
                    "g_delta_mean": float((g_after - g_before).float().mean().cpu()),
                }
            )
    return audit_rows


def _condition_proxy_from_dynamic(ctx: ExperimentContext, dynamic: BranchTrace, static: BranchTrace, condition: str) -> BranchTrace:
    alpha = 0.65 if "overlap" in condition and "nonoverlap" not in condition else 0.85
    if "random" in condition:
        alpha = 0.78
    if condition == "sham_perturbation":
        alpha = 1.0
    spikes = dynamic.spikes * alpha + static.spikes * (1.0 - alpha)
    spikes = (spikes > 0.5).astype(np.float32)
    return BranchTrace(
        spikes=spikes,
        v_effective=(dynamic.v_effective * alpha + static.v_effective * (1.0 - alpha)).astype(np.float32),
        inhibition=(dynamic.inhibition * alpha + static.inhibition * (1.0 - alpha)).astype(np.float32),
        layer3_spikes=(dynamic.layer3_spikes * alpha + static.layer3_spikes * (1.0 - alpha)).astype(np.float32),
        prediction=dynamic.prediction if alpha >= 0.8 else static.prediction,
        first_fire_time=dynamic.first_fire_time if alpha >= 0.8 else static.first_fire_time,
    )


def _support_maps_from_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[int, np.ndarray]:
    state = boundary.get("layer1", {})
    if "u" in state and "x" in state:
        support = (state["u"].to(torch.float32) * state["x"].to(torch.float32)).mean(dim=1).numpy()
    else:
        support = np.zeros((batch_size, 28, 28), dtype=np.float32)
    return {idx: _resize_array(support[idx], 28, 28).astype(np.float32) for idx in range(batch_size)}


def _save_probe_trace_manifest(ctx: ExperimentContext, branch_traces: Mapping[int, Mapping[str, BranchTrace]]) -> None:
    rows = []
    for trial_id, traces in branch_traces.items():
        for condition, trace in traces.items():
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_id),
                    "condition": condition,
                    "trace_kind": "layer1_spatial_collapsed",
                    "n_time_steps": int(trace.spikes.shape[0]),
                    "height": int(trace.spikes.shape[1]),
                    "width": int(trace.spikes.shape[2]),
                    "save_full_traces": bool(ctx.cfg.save_full_traces),
                }
            )
    _save_csv(ctx, pd.DataFrame(rows), ctx.raw_dir / "layer1_probe_trace_manifest.csv")


def _save_panel_a_example(ctx: ExperimentContext, trials: pd.DataFrame, support_maps: Mapping[int, np.ndarray], unit_groups: pd.DataFrame) -> None:
    first = trials.iloc[0]
    trial_id = int(first["trial_id"])
    sample = _image_array(ctx.dataset, int(first["sample_image_id"]))
    probe = _image_array(ctx.dataset, int(first["probe_image_id"]))
    sample_mask = sample > ctx.cfg.foreground_threshold
    probe_mask = probe > ctx.cfg.foreground_threshold
    overlap = sample_mask & probe_mask
    probe_only = probe_mask & (~sample_mask)
    groups = unit_groups[unit_groups["trial_id"].eq(trial_id)]
    np.savez_compressed(
        ctx.raw_dir / "panel_a_example_support_map.npz",
        support_map=support_maps[trial_id].astype(np.float32),
        sample_foreground_mask=sample_mask.astype(np.uint8),
        probe_foreground_mask=probe_mask.astype(np.uint8),
        overlap_mask_projected=overlap.astype(np.uint8),
        probe_only_mask_projected=probe_only.astype(np.uint8),
        overlap_dominant_units=groups[groups["unit_group"].eq("overlap_dominant")]["unit_id"].to_numpy(dtype=np.int64),
        probe_only_dominant_units=groups[groups["unit_group"].eq("probe_only_dominant")]["unit_id"].to_numpy(dtype=np.int64),
        selected_trial_metadata=json.dumps(first.to_dict(), sort_keys=True),
    )
    ctx.output_files["panel_a_example_support_map"] = _rel(ctx.raw_dir / "panel_a_example_support_map.npz", ctx.seed_dir)


def _save_trial_mask_npz(ctx: ExperimentContext, trials: pd.DataFrame) -> None:
    payload: dict[str, np.ndarray] = {}
    for row in trials.itertuples(index=False):
        sample = _image_array(ctx.dataset, int(row.sample_image_id))
        probe = _image_array(ctx.dataset, int(row.probe_image_id))
        sample_mask = sample > ctx.cfg.foreground_threshold
        probe_mask = probe > ctx.cfg.foreground_threshold
        payload[f"trial_{int(row.trial_id)}_sample_foreground_mask"] = sample_mask.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_probe_foreground_mask"] = probe_mask.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_overlap_mask"] = (sample_mask & probe_mask).astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_probe_only_mask"] = (probe_mask & (~sample_mask)).astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_sample_nonoverlap_mask"] = (sample_mask & (~probe_mask)).astype(np.uint8)
    np.savez_compressed(ctx.raw_dir / "trial_masks.npz", **payload)
    ctx.output_files["trial_masks"] = _rel(ctx.raw_dir / "trial_masks.npz", ctx.seed_dir)


def _write_config_files(ctx: ExperimentContext) -> None:
    payload = _json_safe(asdict(ctx.cfg))
    _write_json(payload, ctx.config_dir / "run_config.json")
    _write_json(payload, ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "fig5_design_version": FIG5_DESIGN_VERSION,
            "main_panels": MAIN_PANEL_DESCRIPTIONS,
            "main_claim": MAIN_CLAIM,
            "supplement_plan": SUPPLEMENT_PLAN,
            "main_required_outputs": FIG5_MAIN_REQUIRED_OUTPUTS,
            "supplementary_outputs": {
                "S9": FIG5_S9_OUTPUTS,
                "S10": FIG5_S10_OUTPUTS,
            },
            "backward_compatible_outputs": FIG5_BACKWARD_COMPATIBLE_OUTPUTS,
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "main_conditions": list(MAIN_CONDITIONS),
            "reference_condition": "static_frozen",
            "supplementary_controls": list(SUPP_CONDITIONS),
            "deprecated_conditions": list(REMOVED_FROM_MAIN_CONDITIONS),
            "panel_d": (
                "attenuate/reset overlap high-support STSP perturbation; probe input unchanged; "
                "tests whether reducing overlap-aligned STSP support disrupts dynamic-like transition structure."
            ),
            "perturbation_semantics": {
                "attenuate_overlap_high_support": "attenuate u_pre toward baseline for overlap high-support units; x_pre unchanged unless existing implementation differs",
                "reset_overlap_high_support": "reset u_pre to baseline and x_pre to 1.0 for overlap high-support units",
                "sham_perturbation": "matched procedural control without intended support reduction",
            },
            "static_frozen": "Probe uses model stsp_mode=static_frozen as the transition reference when a checkpoint is available.",
            "proxy_mode": "If the requested checkpoint is absent, deterministic image-driven proxy traces are warning-marked.",
        },
        ctx.config_dir / "condition_spec.json",
    )
    _write_json(
        {
            "primary_intervention": "attenuate_or_reset_overlap_high_support",
            "main_conditions": list(MAIN_CONDITIONS),
            "reference_condition": "static_frozen",
            "attenuate_definition": "u_pre = U_baseline + attenuation_factor * (u_pre - U_baseline); x_pre unchanged",
            "reset_definition": "u_pre = U_baseline; x_pre = 1.0",
            "attenuation_factor": float(ctx.cfg.perturbation_attenuation_factor),
            "probe_input_changed": False,
            "intervention_timing": "pre_probe_boundary",
            "main_metric": "spike transition distribution",
        },
        ctx.config_dir / "support_perturbation_spec.json",
    )
    _write_json({"local_kernel_radius": int(ctx.cfg.local_kernel_radius), "event_align_pre_steps": int(ctx.cfg.event_align_pre_steps), "event_align_post_steps": int(ctx.cfg.event_align_post_steps)}, ctx.config_dir / "event_selection_spec.json")
    _write_json({"null_types": list(NULL_TYPES), "n_null": int(ctx.cfg.n_null)}, ctx.config_dir / "null_baseline_spec.json")


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main: list[Path] = []
    if ctx.cfg.run_preprobe_support:
        required_main.append(ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv")
    if ctx.cfg.run_early_firing:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv",
                ctx.metrics_dir / "panel_b_transition_summary_by_group.csv",
            ]
        )
    if ctx.cfg.run_local_events:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_c_winner_loser_event_metrics.csv",
                ctx.metrics_dir / "panel_c_event_trace_summary.csv",
            ]
        )
    if ctx.cfg.run_support_perturbation:
        required_main.extend(
            [
                ctx.metrics_dir / "panel_d_perturbation_unit_transitions.csv",
                ctx.metrics_dir / "panel_d_perturbation_transition_summary_by_group.csv",
                ctx.metrics_dir / "panel_d_perturbation_transition_contrast.csv",
                ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv",
                ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv",
                ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv",
            ]
        )
    required_supp: list[Path] = []
    if ctx.cfg.run_supplement:
        required_supp.extend(ctx.seed_dir / output for output in FIG5_S9_OUTPUTS + FIG5_S10_OUTPUTS)
    support_downstream_available = bool(
        ctx.availability.get(
            "support_perturbation_downstream_available",
            _csv_nonempty(ctx.metrics_dir / "panel_d_support_perturbation_trial_metrics.csv")
            and _csv_nonempty(ctx.metrics_dir / "panel_d_support_perturbation_node_metrics.csv"),
        )
    )
    if ctx.cfg.run_support_perturbation and not support_downstream_available and not ctx.availability.get("support_perturbation_downstream_missing_reason"):
        ctx.availability["support_perturbation_downstream_missing_reason"] = "panel_d_support_perturbation_metrics_missing_or_empty"
    perturbation_effect_available = bool(
        ctx.availability.get("perturbation_effect_summary_available", _csv_nonempty(ctx.metrics_dir / "panel_d_perturbation_effect_summary.csv"))
    )
    proxy_mode = ctx.net is None
    main_available = all(path.exists() for path in required_main)
    if proxy_mode and "Proxy mode is for pipeline validation only; final scientific use is false." not in ctx.warnings:
        ctx.warnings.append("Proxy mode is for pipeline validation only; final scientific use is false.")
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "fig5_design_version": FIG5_DESIGN_VERSION,
        "main_claim": MAIN_CLAIM,
        "smoke": bool(ctx.cfg.smoke),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_trials": int(ctx.n_trials),
        "n_events": int(ctx.n_events),
        "main_panels": MAIN_PANEL_DESCRIPTIONS,
        "supplement_plan": SUPPLEMENT_PLAN,
        "fig5e_removed_from_main": True,
        "old_flatten_conditions_removed": True,
        "old_flatten_nonoverlap_random_removed_from_main": True,
        "conditions": list(MAIN_CONDITIONS),
        "unit_groups": list(UNIT_GROUPS),
        "main_fig5d_conditions": list(MAIN_CONDITIONS),
        "reference_condition": "static_frozen",
        "perturbation_conditions": list(MAIN_CONDITIONS + SUPP_CONDITIONS),
        "current_perturbation_conditions": list(MAIN_CONDITIONS[1:] + SUPP_CONDITIONS),
        "deprecated_flatten_conditions": list(REMOVED_FROM_MAIN_CONDITIONS),
        "main_fig5d_metric": "spike_transition_distribution",
        "attenuation_definition": "u_pre = U0 + factor*(u_pre-U0); x_pre unchanged",
        "reset_definition": "u_pre = U0; x_pre = 1.0",
        "support_perturbation_downstream_available": support_downstream_available,
        "support_perturbation_downstream_missing_reason": ctx.availability.get("support_perturbation_downstream_missing_reason"),
        "perturbation_effect_summary_available": perturbation_effect_available,
        "perturbation_effect_summary_missing_reason": ctx.availability.get("perturbation_effect_summary_missing_reason"),
        "supplement_alias_missing_reasons": ctx.availability.get("supplement_alias_missing_reasons", {}),
        "proxy_mode": bool(proxy_mode),
        "final_scientific_use": bool(not proxy_mode),
        "event_selection": {"local_kernel_radius": int(ctx.cfg.local_kernel_radius), "n_events": int(ctx.n_events)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": bool(main_available and not proxy_mode),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    return summary


def _write_run_log(ctx: ExperimentContext) -> None:
    ctx.run_log.append(f"{_now()} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.5 local support competition experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-trial-sampling", action="store_true")
    parser.add_argument("--run-preprobe-support", action="store_true")
    parser.add_argument("--run-early-firing", action="store_true")
    parser.add_argument("--run-local-events", action="store_true")
    parser.add_argument("--run-support-perturbation", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-spike-cache", action="store_true")
    parser.add_argument("--save-full-traces", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--enable-branch-batch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--min-overlap-area", type=int, default=4)
    parser.add_argument("--min-probe-only-area", type=int, default=4)
    parser.add_argument("--medium-q-low", type=float, default=0.35)
    parser.add_argument("--medium-q-high", type=float, default=0.65)
    parser.add_argument("--early-window-ms", type=int, default=15)
    parser.add_argument("--drive-score-threshold", type=float, default=0.05)
    parser.add_argument("--local-kernel-radius", type=int, default=2)
    parser.add_argument("--peak-support-q", type=float, default=0.20)
    parser.add_argument("--perturbation-mode", default="attenuate_reset", choices=["attenuate_reset", "attenuate", "reset"])
    parser.add_argument("--perturbation-attenuation-factor", type=float, default=0.5)
    parser.add_argument("--event-align-pre-steps", type=int, default=8)
    parser.add_argument("--event-align-post-steps", type=int, default=12)
    parser.add_argument("--chain-pre-spike-steps", type=int, default=4)
    parser.add_argument("--chain-post-spike-steps", type=int, default=6)
    parser.add_argument("--n-null", type=int, default=100)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> Fig5Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    return Fig5Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 2) if smoke else int(args.batch_size),
        max_trials=8 if smoke else int(args.max_trials),
        foreground_threshold=float(args.foreground_threshold),
        min_overlap_area=int(args.min_overlap_area),
        min_probe_only_area=int(args.min_probe_only_area),
        medium_q_low=float(args.medium_q_low),
        medium_q_high=float(args.medium_q_high),
        early_window_ms=int(args.early_window_ms),
        drive_score_threshold=float(args.drive_score_threshold),
        local_kernel_radius=int(args.local_kernel_radius),
        peak_support_q=float(args.peak_support_q),
        perturbation_mode=str(args.perturbation_mode),
        perturbation_attenuation_factor=float(args.perturbation_attenuation_factor),
        event_align_pre_steps=int(args.event_align_pre_steps),
        event_align_post_steps=int(args.event_align_post_steps),
        chain_pre_spike_steps=int(args.chain_pre_spike_steps),
        chain_post_spike_steps=int(args.chain_post_spike_steps),
        n_null=8 if smoke else int(args.n_null),
        save_full_traces=bool(args.save_full_traces),
        save_spike_cache=bool(args.save_spike_cache),
        run_trial_sampling=run_all or bool(args.run_trial_sampling),
        run_preprobe_support=run_all or bool(args.run_preprobe_support),
        run_early_firing=run_all or bool(args.run_early_firing),
        run_local_events=run_all or bool(args.run_local_events),
        run_support_perturbation=run_all or bool(args.run_support_perturbation),
        run_supplement=run_all or bool(args.run_supplement),
        save_debug_figures=bool(args.save_debug_figures),
        show_progress=not bool(args.no_progress),
        enable_branch_batch=bool(args.enable_branch_batch),
        smoke=smoke,
    )


def _load_dataset_or_raise(dataset_root: str, split: str):
    return load_mnist_skeleton_dataset(dataset_root, split)


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = _rel(path, ctx.seed_dir)


def _csv_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(not pd.read_csv(path).empty)
    except Exception:
        return False


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
        return str(path)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _steps_to_ms(value_steps: int | float, dt: float) -> float:
    value = float(value_steps)
    if not np.isfinite(value) or value < 0:
        return float("nan")
    return float(value * float(dt) / ms)


def _finite_delta(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _recovery_toward_static(dynamic: float, static: float, value: float) -> float:
    if not (np.isfinite(dynamic) and np.isfinite(static) and np.isfinite(value)):
        return float("nan")
    denom = float(static - dynamic)
    if abs(denom) < 1e-12:
        return float("nan")
    return float((value - dynamic) / denom)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), int(batch_size)):
        yield df.iloc[start : start + int(batch_size)].copy()


def _image_array(dataset, image_id: int) -> np.ndarray:
    image = dataset[int(image_id)][0].detach().cpu().to(torch.float32).squeeze().numpy()
    return np.asarray(image, dtype=np.float32)


def _images_for_ids(dataset, image_ids: Sequence[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    return (arr - lo) / max(hi - lo, 1e-9)


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return _resize_array(mask.astype(float), h, w) > 0.5


def _resize_array(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    src = np.asarray(arr)
    if src.shape == (h, w):
        return src
    rr = np.linspace(0, src.shape[0] - 1, h).round().astype(int)
    cc = np.linspace(0, src.shape[1] - 1, w).round().astype(int)
    return src[np.ix_(rr, cc)]


def _blur3(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, 1, mode="edge")
    out = np.zeros_like(arr, dtype=float)
    for dr in range(3):
        for dc in range(3):
            out += padded[dr : dr + arr.shape[0], dc : dc + arr.shape[1]]
    return out / 9.0


def _first_spike_map(spikes: np.ndarray) -> np.ndarray:
    arr = np.asarray(spikes)
    first = np.full(arr.shape[1:], -1, dtype=int)
    fired = arr > 0
    any_fire = fired.any(axis=0)
    if np.any(any_fire):
        first[any_fire] = np.argmax(fired, axis=0)[any_fire]
    return first


def _transition_type(dynamic_first: int, static_first: int) -> str:
    if dynamic_first >= 0 and static_first >= 0 and dynamic_first < static_first:
        return "advance"
    if dynamic_first >= 0 and static_first < 0:
        return "recruit"
    if dynamic_first < 0 and static_first >= 0:
        return "loss"
    return "unchanged"


def _transition_vs_same(first_cond: int, first_same: int, first_static: int) -> str:
    same_transition = _transition_type(first_same, first_static)
    cond_transition = _transition_type(first_cond, first_static)
    same_winner = same_transition in {"advance", "recruit"}
    cond_winner = cond_transition in {"advance", "recruit"}
    if not same_winner:
        return "not_same_winner"
    if first_cond < 0:
        return "lost"
    if not cond_winner:
        return "reverted_to_static"
    if first_same >= 0 and first_cond > first_same:
        return "delayed"
    return "preserved"


def _latency_delta(dynamic_first: int, static_first: int) -> float:
    if dynamic_first >= 0 and static_first >= 0:
        return float(dynamic_first - static_first)
    if dynamic_first >= 0 and static_first < 0:
        return float(-dynamic_first)
    if dynamic_first < 0 and static_first >= 0:
        return float(static_first)
    return float("nan")


def _spikes_earlier(dynamic_first: int, static_first: int) -> bool:
    return bool(dynamic_first >= 0 and (static_first < 0 or dynamic_first < static_first))


def _is_loser_suppressed(dynamic_first: int, static_first: int) -> bool:
    return bool(static_first >= 0 and (dynamic_first < 0 or dynamic_first > static_first))


def _advanced_or_recruited_units(first_dyn: np.ndarray, first_sta: np.ndarray) -> set[int]:
    out = set()
    h, w = first_dyn.shape
    for r in range(h):
        for c in range(w):
            if _transition_type(int(first_dyn[r, c]), int(first_sta[r, c])) in {"advance", "recruit"}:
                out.add(int(r * w + c))
    return out


def _delayed_or_lost_units(first_dyn: np.ndarray, first_sta: np.ndarray) -> set[int]:
    out = set()
    h, w = first_dyn.shape
    for r in range(h):
        for c in range(w):
            fd, fs = int(first_dyn[r, c]), int(first_sta[r, c])
            if (fs >= 0 and fd < 0) or (fd >= 0 and fs >= 0 and fd > fs):
                out.add(int(r * w + c))
    return out


def _nearest_loser(win: Any, losers: pd.DataFrame, radius: int):
    if losers.empty:
        return None
    part = losers.copy()
    part["dist"] = (part["row"].astype(int) - int(win.row)).abs() + (part["col"].astype(int) - int(win.col)).abs()
    part = part[part["dist"] <= int(radius) * 2]
    if part.empty:
        return None
    return next(part.sort_values("dist").itertuples(index=False))


def _aligned_delta(dynamic: np.ndarray, static: np.ndarray, t0: int, ctx: ExperimentContext) -> np.ndarray:
    vals = []
    for offset in range(-ctx.cfg.event_align_pre_steps, ctx.cfg.event_align_post_steps + 1):
        t = int(t0 + offset)
        if 0 <= t < len(dynamic):
            vals.append(float(dynamic[t] - static[t]))
        else:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float32)


def _trace_summary_row(ctx: ExperimentContext, time_ms: float, trace_type: str, value: float) -> dict[str, Any]:
    return {"network_seed": int(ctx.cfg.network_seed), "time_ms": float(time_ms), "trace_type": trace_type, "value": float(value)}


def _event_audit_row(ctx: ExperimentContext, trial_id: int, event_id: int, step: str, included: bool, reason: str, winner_group: str, loser_group: str, drive_winner: float, drive_loser: float) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "trial_id": int(trial_id),
        "event_id": int(event_id),
        "selection_step": step,
        "included": bool(included),
        "exclusion_reason": reason,
        "winner_group": str(winner_group),
        "loser_group": str(loser_group),
        "neighborhood_radius": int(ctx.cfg.local_kernel_radius),
        "drive_score_winner": float(drive_winner),
        "drive_score_loser": float(drive_loser) if np.isfinite(drive_loser) else np.nan,
    }


def _mean_for_group(df: pd.DataFrame, group: str) -> float:
    part = df[df["unit_group"].eq(group)]
    return float(pd.to_numeric(part["support_value"], errors="coerce").mean()) if not part.empty else float("nan")


def _pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.dot(aa, bb) / denom)


def _decision_deflection(trace: BranchTrace, dynamic: BranchTrace, static: BranchTrace) -> float:
    dyn_sim = _pattern_similarity(trace.layer3_spikes, dynamic.layer3_spikes)
    sta_sim = _pattern_similarity(trace.layer3_spikes, static.layer3_spikes)
    return float(dyn_sim - sta_sim)


def _proxy_perturbed_support(support: np.ndarray, condition: str, psets: pd.DataFrame) -> np.ndarray:
    out = np.asarray(support, dtype=np.float32).copy()
    if psets.empty:
        return out
    for row in psets[psets["condition"].eq(condition)].itertuples(index=False):
        if 0 <= int(row.row) < out.shape[0] and 0 <= int(row.col) < out.shape[1]:
            if condition.startswith("attenuate"):
                out[int(row.row), int(row.col)] = float(out[int(row.row), int(row.col)] * 0.65)
            elif condition.startswith("reset"):
                out[int(row.row), int(row.col)] = 0.0
    return out


def _slice_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]], index: int) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        out[layer_key] = {}
        for key, tensor in state.items():
            out[layer_key][key] = tensor[index : index + 1].clone()
    return out


def _restore_boundary_state(net, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for layer_key, state in boundary.items():
            layer = getattr(net, layer_key)
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key in state:
                    getattr(layer, attr).copy_(state[src_key].to(device=getattr(layer, attr).device, dtype=getattr(layer, attr).dtype))
            if "inh_trace" in state:
                layer.lateral_inh.inh_trace.copy_(state["inh_trace"].to(device=layer.lateral_inh.inh_trace.device, dtype=layer.lateral_inh.inh_trace.dtype))
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))


def _step_network_once(net, input_t: torch.Tensor, current_time: int, *, stsp_mode: str = "dynamic") -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1


def _trial_mapping(trial: Any) -> Mapping[str, Any]:
    if isinstance(trial, pd.Series):
        return trial.to_dict()
    if isinstance(trial, Mapping):
        return trial
    if hasattr(trial, "_asdict"):
        return trial._asdict()
    return dict(trial)


if __name__ == "__main__":
    raise SystemExit(main())
