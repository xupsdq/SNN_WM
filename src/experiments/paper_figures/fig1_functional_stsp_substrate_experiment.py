from __future__ import annotations

import argparse
import json
import math
import sys
import warnings as py_warnings
from dataclasses import asdict
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
from src.experiments.common.monitored_dms import build_layer_input_shapes, snapshot_boundary_state
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state, snapshot_ux_state
from src.experiments.common.run_info import build_run_info, finalize_run_info, write_run_info
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.paper_figures.common.bundle_io import (
    prepare_seed_dirs,
    relative_to_root,
    resolve_seed_dir,
    save_csv_with_registry,
    write_artifact_manifest,
    write_json_file,
    write_run_log,
)
from src.experiments.paper_figures.common.progress import ProgressTracker, planned_phases
from src.experiments.paper_figures.fig1.types import ExperimentContext, Fig1Config, ProbePrep
from src.plotting.common.io import apply_publication_style, save_figure_all_formats

try:
    from src.experiments.ping_memory.shared.shuffle_metrics import (
        compute_bias_table as compat_compute_bias_table,
        compute_collapse_summary as compat_compute_collapse_summary,
        compute_condition_metrics as compat_compute_condition_metrics,
    )
except Exception:  # pragma: no cover - compatibility outputs are best effort
    compat_compute_bias_table = None
    compat_compute_collapse_summary = None
    compat_compute_condition_metrics = None

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC
except Exception:  # pragma: no cover - handled at runtime with a clear error
    ConvergenceWarning = Warning
    LinearSVC = None
    f1_score = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, *, total=None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


FIGURE_ID = "fig1_functional_stsp_substrate"
NUM_CLASSES = 10
MAIN_CONDITIONS = ("dynamic_intact", "ux_trial_shuffle", "static_frozen")
SUPP_CONDITIONS = ("dynamic_intact", "spike_state_shuffle", "membrane_state_shuffle", "ux_trial_shuffle", "static_frozen")
DMS_DELAY_SWEEP_CONDITIONS = ("dynamic_intact", "static_frozen")
SHUFFLE_CONDITIONS = ("spike_state_shuffle", "membrane_state_shuffle", "ux_trial_shuffle")
SUBSTRATE_BY_CONDITION = {
    "dynamic_intact": "dynamic",
    "ux_trial_shuffle": "ux",
    "spike_state_shuffle": "spike",
    "membrane_state_shuffle": "membrane",
    "static_frozen": "static",
}
CONDITION_TO_SUBSTRATE = {
    "spike_state_shuffle": "spike",
    "membrane_state_shuffle": "membrane",
    "ux_trial_shuffle": "ux",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    run(cfg)
    return 0


def run(cfg: Fig1Config) -> dict[str, Any]:
    seed_everything(int(cfg.network_seed))
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")

    seed_dir = _resolve_seed_dir(Path(cfg.output_root), int(cfg.network_seed))
    dirs = _prepare_dirs(seed_dir)
    warnings_list: list[str] = []
    run_log = [f"{_now()} start {FIGURE_ID} seed={cfg.network_seed} smoke={cfg.smoke}"]
    device = resolve_device(cfg.device)
    dataset = load_mnist_skeleton_dataset(cfg.dataset_root, cfg.split)
    class_index = build_class_index(dataset, NUM_CLASSES)
    net, encoder = load_model_and_encoder(cfg.model_path, device=device, dt=cfg.dt, max_duration_ms=max(cfg.sample_ms, cfg.dms_sample_ms, cfg.probe_ms, 100))
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
        n_trials={},
        donor_constraint_summary={},
        run_log=run_log,
    )
    if cfg.shuffle_compat_mode:
        expected = {"dms_delay_ms": 500, "dms_num_trials": 500, "dms_batch_size": 32}
        for field_name, expected_value in expected.items():
            actual = getattr(cfg, field_name)
            if int(actual) != int(expected_value):
                ctx.warnings.append(
                    f"shuffle_compat_mode is enabled, but {field_name}={actual} "
                    f"(legacy reference default is {expected_value}); using explicit/current value."
                )
    run_info = build_run_info(
        experiment_name=FIGURE_ID,
        output_dir=seed_dir,
        entry_script="src.experiments.paper_figures.fig1_functional_stsp_substrate_experiment",
        seed=cfg.network_seed,
        dataset=f"MNIST:{cfg.split}",
        command=" ".join(sys.argv),
        model_path=cfg.model_path,
        status="running",
    )
    write_run_info(seed_dir / "meta", run_info)

    try:
        progress = ProgressTracker(
            ctx,
            planned_phases(
                (
                    ("config", True),
                    ("trial_specs", True),
                    ("baseline", cfg.run_baseline),
                    ("delay_decode", cfg.run_delay_decode),
                    ("dms_delay_sweep", cfg.run_dms_delay_sweep),
                    ("dms_shuffle", cfg.run_dms_shuffle),
                    ("firing_rate_control", cfg.run_firing_rate_control),
                    ("debug_figures", cfg.save_debug_figures),
                    ("summary", True),
                )
            ),
            fig_id="fig1",
        )
        with progress.phase("config"):
            _write_config_files(ctx)
        with progress.phase("trial_specs"):
            specs = build_trial_specs(ctx)
        if cfg.run_baseline:
            with progress.phase("baseline"):
                run_baseline_eval(ctx, specs["baseline"])
        if cfg.run_delay_decode:
            with progress.phase("delay_decode"):
                run_delay_stsp_decode(ctx, specs["delay_train"], specs["delay_test"])
        if cfg.run_dms_delay_sweep:
            with progress.phase("dms_delay_sweep"):
                run_dms_functional_delay_sweep(ctx, specs["dms"])
        if cfg.run_dms_shuffle:
            with progress.phase("dms_shuffle"):
                dms_outputs = run_dms_substrate_shuffle(ctx, specs["dms"])
            if cfg.run_firing_rate_control:
                with progress.phase("firing_rate_control"):
                    run_phase_firing_rate_control(ctx, dms_outputs.get("phase_rate_rows", []))
        elif cfg.run_firing_rate_control:
            with progress.phase("firing_rate_control"):
                _write_empty_phase_rates(ctx)
                ctx.completed_modules["firing_rate_control"] = True
        if cfg.save_debug_figures:
            with progress.phase("debug_figures"):
                save_debug_figures(ctx)
        with progress.phase("summary"):
            summary = _write_summary(ctx)
            _write_run_log(ctx)
        finalize_run_info(seed_dir / "meta", run_info, status="success")
        return summary
    except Exception:
        finalize_run_info(seed_dir / "meta", run_info, status="failed")
        raise


def build_trial_specs(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    baseline = _balanced_image_trials(
        ctx.class_index,
        per_class=cfg.baseline_eval_per_class,
        rng=rng,
        network_seed=cfg.network_seed,
        split=cfg.split,
        id_prefix="baseline",
    )
    baseline = baseline[["network_seed", "trial_id", "image_id", "label", "split"]]

    train, test, overlap = _balanced_disjoint_delay_trials(
        ctx.class_index,
        train_per_class=cfg.delay_decode_train_per_class,
        test_per_class=cfg.delay_decode_test_per_class,
        rng=rng,
        network_seed=cfg.network_seed,
    )
    if overlap:
        ctx.warnings.append(f"Delay train/test image overlap was unavoidable for {overlap} image IDs.")

    dms, audit_rows = _build_dms_trials(
        ctx.class_index,
        n_trials=cfg.dms_num_trials,
        rng=rng,
        network_seed=cfg.network_seed,
    )

    _save_csv(ctx, baseline, ctx.trial_specs_dir / "baseline_eval_trials.csv")
    _save_csv(ctx, train, ctx.trial_specs_dir / "delay_decode_train_trials.csv")
    _save_csv(ctx, test, ctx.trial_specs_dir / "delay_decode_test_trials.csv")
    _save_csv(ctx, dms, ctx.trial_specs_dir / "dms_shuffle_trials.csv")
    audit = pd.DataFrame(audit_rows)
    _save_csv(ctx, audit, ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.n_trials.update({"baseline": len(baseline), "delay_train": len(train), "delay_test": len(test), "dms": len(dms)})
    ctx.completed_modules["trial_specs"] = True
    return {"baseline": baseline, "delay_train": train, "delay_test": test, "dms": dms}


def run_baseline_eval(ctx: ExperimentContext, trials: pd.DataFrame):
    from src.experiments.paper_figures.fig1.subexperiments.baseline import run_baseline_eval as _impl

    return _impl(ctx, trials)


def run_delay_stsp_decode(ctx: ExperimentContext, train_trials: pd.DataFrame, test_trials: pd.DataFrame):
    from src.experiments.paper_figures.fig1.subexperiments.delay_decode import run_delay_stsp_decode as _impl

    return _impl(ctx, train_trials, test_trials)


def run_dms_functional_delay_sweep(ctx: ExperimentContext, dms_trials: pd.DataFrame):
    from src.experiments.paper_figures.fig1.subexperiments.dms_delay_sweep import run_dms_functional_delay_sweep as _impl

    return _impl(ctx, dms_trials)


def run_dms_substrate_shuffle(ctx: ExperimentContext, dms_trials: pd.DataFrame):
    from src.experiments.paper_figures.fig1.subexperiments.dms_shuffle import run_dms_substrate_shuffle as _impl

    return _impl(ctx, dms_trials)


def run_phase_firing_rate_control(ctx: ExperimentContext, rows: Sequence[Mapping[str, Any]]):
    from src.experiments.paper_figures.fig1.subexperiments.firing_rate_control import run_phase_firing_rate_control as _impl

    return _impl(ctx, rows)


def save_debug_figures(ctx: ExperimentContext) -> None:
    import matplotlib.pyplot as plt

    apply_publication_style()
    paths = [
        (ctx.metrics_dir / "panel_b_baseline_metrics_by_network.csv", "overall_recall", "baseline_recall"),
        (ctx.metrics_dir / "panel_d_condition_metrics.csv", "error_rate", "dms_error_rate"),
        (ctx.metrics_dir / "panel_e_attribution_metrics.csv", "donor_sample_attribution", "donor_attribution"),
        (ctx.metrics_dir / "supp_phase_firing_rates.csv", "spike_rate_hz", "phase_firing_rate"),
    ]
    for path, value_col, stem in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or value_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        group_col = "condition" if "condition" in df.columns else ("phase" if "phase" in df.columns else "network_seed")
        plot_df = df.groupby(group_col, as_index=False)[value_col].mean()
        ax.bar(np.arange(len(plot_df)), plot_df[value_col].to_numpy(dtype=float))
        ax.set_xticks(np.arange(len(plot_df)), [str(v) for v in plot_df[group_col]], rotation=30, ha="right")
        ax.set_ylabel(value_col)
        fig.tight_layout()
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    curve_path = ctx.metrics_dir / "supp_delay_decode_curve.csv"
    if curve_path.exists():
        df = pd.read_csv(curve_path)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            for layer, part in df.groupby("layer", sort=True):
                ax.plot(part["delay_ms"], part["acc"], marker="o", label=str(layer))
            ax.set_xlabel("delay_ms")
            ax.set_ylabel("acc")
            ax.legend(frameon=False)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "delay_decoding_curve")
            plt.close(fig)
    sweep_path = ctx.metrics_dir / "supp_dms_delay_sweep_metrics.csv"
    if sweep_path.exists():
        df = pd.read_csv(sweep_path)
        if not df.empty and {"delay_ms", "condition", "acc_probe"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(5, 3))
            for condition, part in df.groupby("condition", sort=True):
                ax.plot(part["delay_ms"], part["acc_probe"], marker="o", label=str(condition))
            ax.set_xlabel("delay_ms")
            ax.set_ylabel("acc_probe")
            ax.legend(frameon=False)
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "dms_delay_sweep_accuracy")
            plt.close(fig)
    contrast_path = ctx.metrics_dir / "supp_dms_delay_sweep_contrast.csv"
    if contrast_path.exists():
        df = pd.read_csv(contrast_path)
        if not df.empty and {"delay_ms", "stsp_interference"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(df["delay_ms"], df["stsp_interference"], marker="o")
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.set_xlabel("delay_ms")
            ax.set_ylabel("stsp_interference")
            fig.tight_layout()
            save_figure_all_formats(fig, ctx.debug_dir / "dms_delay_sweep_interference")
            plt.close(fig)
    ctx.completed_modules["debug_figures"] = True


def _run_sample_then_snapshot_delays(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _run_sample_then_snapshot_delays as _impl

    return _impl(*args, **kwargs)


def _append_feature_store(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _append_feature_store as _impl

    return _impl(*args, **kwargs)


def _finalize_feature_store(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _finalize_feature_store as _impl

    return _impl(*args, **kwargs)


def _run_sample_multi_delay_boundary_capture(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _run_sample_multi_delay_boundary_capture as _impl

    return _impl(*args, **kwargs)


def _run_sample_delay_capture(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _run_sample_delay_capture as _impl

    return _impl(*args, **kwargs)


def _run_probe_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _run_probe_from_boundary as _impl

    return _impl(*args, **kwargs)


def _run_probe_conditions_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _run_probe_conditions_from_boundary as _impl

    return _impl(*args, **kwargs)


def _restore_boundary_state(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _restore_boundary_state as _impl

    return _impl(*args, **kwargs)


def _make_shuffled_substrate_state_from_boundary(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _make_shuffled_substrate_state_from_boundary as _impl

    return _impl(*args, **kwargs)


def _restore_substrate_only(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _restore_substrate_only as _impl

    return _impl(*args, **kwargs)


def _reset_all_layer_states_from_shapes(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _reset_all_layer_states_from_shapes as _impl

    return _impl(*args, **kwargs)


def _apply_legacy_layer3_probe_phase_reset(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _apply_legacy_layer3_probe_phase_reset as _impl

    return _impl(*args, **kwargs)


def _prepare_condition_for_probe(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _prepare_condition_for_probe as _impl

    return _impl(*args, **kwargs)


def _intervention_for_probe_prep(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _intervention_for_probe_prep as _impl

    return _impl(*args, **kwargs)


def _condition_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _condition_metrics as _impl

    return _impl(*args, **kwargs)


def _delay_sweep_condition_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _delay_sweep_condition_metrics as _impl

    return _impl(*args, **kwargs)


def _delay_sweep_contrast(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _delay_sweep_contrast as _impl

    return _impl(*args, **kwargs)


def _sort_trial_readout(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _sort_trial_readout as _impl

    return _impl(*args, **kwargs)


def _sort_dms_delay_sweep_trial_readout(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _sort_dms_delay_sweep_trial_readout as _impl

    return _impl(*args, **kwargs)


def _validate_dms_delay_sweep_pairing(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _validate_dms_delay_sweep_pairing as _impl

    return _impl(*args, **kwargs)


def _validate_fig1_shuffle_pairing(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _validate_fig1_shuffle_pairing as _impl

    return _impl(*args, **kwargs)


def _bad_trial_ids(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _bad_trial_ids as _impl

    return _impl(*args, **kwargs)


def _compat_trial_readout(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _compat_trial_readout as _impl

    return _impl(*args, **kwargs)


def _write_compatibility_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _write_compatibility_metrics as _impl

    return _impl(*args, **kwargs)


def _donor_constraint_audit(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _donor_constraint_audit as _impl

    return _impl(*args, **kwargs)


def _attribution_metrics(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _attribution_metrics as _impl

    return _impl(*args, **kwargs)


def _balanced_image_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _balanced_image_trials as _impl

    return _impl(*args, **kwargs)


def _balanced_disjoint_delay_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _balanced_disjoint_delay_trials as _impl

    return _impl(*args, **kwargs)


def _build_dms_trials(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _build_dms_trials as _impl

    return _impl(*args, **kwargs)


def _derangement(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _derangement as _impl

    return _impl(*args, **kwargs)


def _build_constrained_trial_shuffle_plan(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _build_constrained_trial_shuffle_plan as _impl

    return _impl(*args, **kwargs)


def _build_constrained_permutation_np(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _build_constrained_permutation_np as _impl

    return _impl(*args, **kwargs)


def _sample_indices(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _sample_indices as _impl

    return _impl(*args, **kwargs)


def _images_for_ids(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _images_for_ids as _impl

    return _impl(*args, **kwargs)


def _encode_cached(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _encode_cached as _impl

    return _impl(*args, **kwargs)


def _iter_batches(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _iter_batches as _impl

    return _impl(*args, **kwargs)


def _donor_indices_for_batch(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _donor_indices_for_batch as _impl

    return _impl(*args, **kwargs)


def _init_phase_counts(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _init_phase_counts as _impl

    return _impl(*args, **kwargs)


def _intervention_manifest_row(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _intervention_manifest_row as _impl

    return _impl(*args, **kwargs)


def _write_empty_phase_rates(*args, **kwargs):
    from src.experiments.paper_figures.fig1.subexperiments.helpers import _write_empty_phase_rates as _impl

    return _impl(*args, **kwargs)


def _write_config_files(ctx: ExperimentContext) -> None:
    payload = _json_safe(asdict(ctx.cfg))
    payload["strict_all_three_distinct_donor"] = True
    payload["shuffle_semantics"] = {
        "shuffle_compat_mode": bool(ctx.cfg.shuffle_compat_mode),
        "pure_substrate_only": bool(ctx.cfg.pure_substrate_only),
        "donor_plan": "constrained_all_three_label_distinct",
        "strict_all_three_distinct_donor": True,
        "substrate_definition": "legacy_shuffle",
        "dms_delay_ms": int(ctx.cfg.dms_delay_ms),
        "dms_num_trials": int(ctx.cfg.dms_num_trials),
        "dms_batch_size": int(ctx.cfg.dms_batch_size),
    }
    payload["delay_sweep_semantics"] = {
        "dms_delay_sweep_ms": [int(v) for v in ctx.cfg.dms_delay_sweep_ms],
        "delay_sweep_conditions": list(DMS_DELAY_SWEEP_CONDITIONS),
        "delay_sweep_design": (
            "sample and probe have different labels; sample-induced STSP state is captured after each delay "
            "and the same probe is evaluated under dynamic_intact and static_frozen conditions."
        ),
        "primary_metric": "static-minus-dynamic probe accuracy contrast vs delay",
        "contrast": "acc_static - acc_dynamic",
    }
    _write_json(payload, ctx.config_dir / "run_config.json")
    _write_json(payload, ctx.seed_dir / "run_config.json")
    _write_json(
        {
            "main_panels": ["A", "B", "C", "D", "E"],
            "required_main_conditions": list(MAIN_CONDITIONS),
            "supplementary_outputs": [
                "supp_class_recall_by_digit.csv",
                "supp_confusion_matrix_long.csv",
                "supp_delay_decode_curve.csv",
                "supp_dms_delay_sweep_metrics.csv",
                "supp_dms_delay_sweep_contrast.csv",
                "supp_dms_delay_sweep_trial_readout.csv",
                "supp_substrate_shuffle_metrics.csv",
                "supp_phase_firing_rates.csv",
                "supp_trial_condition_audit.csv",
                "supp_dms_shuffle_donor_constraint_audit.csv",
            ],
        },
        ctx.config_dir / "figure_requirements.json",
    )
    _write_json(
        {
            "conditions": {
                "dynamic_intact": "Boundary state restored; dynamic STSP during probe.",
                "ux_trial_shuffle": "Legacy-compatible pure-substrate trial shuffle of u_pre/x_pre. Full network state is reset before probe and only shuffled u/x substrate is restored.",
                "spike_state_shuffle": "Legacy-compatible pure-substrate trial shuffle of g_e/res/lateral_inh.inh_trace.",
                "membrane_state_shuffle": "Legacy-compatible pure-substrate trial shuffle of v_mem only.",
                "static_frozen": "Boundary state restored; probe uses stsp_mode=static_frozen.",
            },
            "donor_plan": "constrained_all_three_label_distinct",
            "strict_all_three_distinct_donor": True,
            "substrate_definition": "legacy_shuffle",
            "pure_substrate_only": bool(ctx.cfg.pure_substrate_only),
            "boundary_once_design": "Each DMS batch runs sample+delay once, captures the boundary, then runs one probe per condition.",
            "static_frozen_approximation": "The low-level API exposes static_frozen rather than exact freeze-current-u/x-gain; this approximation is recorded in outputs.",
        },
        ctx.config_dir / "condition_spec.json",
    )


def _write_summary(ctx: ExperimentContext) -> dict[str, Any]:
    required_main = [
        ctx.metrics_dir / "panel_b_baseline_metrics_by_network.csv",
        ctx.metrics_dir / "panel_c_delay_decode_metrics.csv",
        ctx.metrics_dir / "panel_d_condition_metrics.csv",
        ctx.metrics_dir / "panel_e_attribution_metrics.csv",
    ]
    required_supp = [
        ctx.metrics_dir / "supp_class_recall_by_digit.csv",
        ctx.metrics_dir / "supp_confusion_matrix_long.csv",
        ctx.metrics_dir / "supp_delay_decode_curve.csv",
        ctx.metrics_dir / "supp_substrate_shuffle_metrics.csv",
        ctx.metrics_dir / "supp_phase_firing_rates.csv",
        ctx.metrics_dir / "supp_trial_condition_audit.csv",
        ctx.metrics_dir / "supp_dms_shuffle_donor_constraint_audit.csv",
    ]
    if ctx.cfg.run_dms_delay_sweep:
        required_supp.extend(
            [
                ctx.metrics_dir / "supp_dms_delay_sweep_metrics.csv",
                ctx.metrics_dir / "supp_dms_delay_sweep_contrast.csv",
                ctx.raw_dir / "supp_dms_delay_sweep_trial_readout.csv",
            ]
        )
    compat_paths = [
        ctx.metrics_dir / "compat_metrics_condition_summary.csv",
        ctx.metrics_dir / "compat_metrics_error_bias.csv",
        ctx.metrics_dir / "compat_metrics_collapse_summary.csv",
        ctx.metrics_dir / "compat_metrics_bootstrap_tests.csv",
    ]
    compatibility_metrics_available = all(path.exists() for path in compat_paths)
    summary = {
        "figure": FIGURE_ID,
        "network_seed": int(ctx.cfg.network_seed),
        "run_mode": "single_network",
        "smoke": bool(ctx.cfg.smoke),
        "shuffle_compat_mode": bool(ctx.cfg.shuffle_compat_mode),
        "pure_substrate_only": bool(ctx.cfg.pure_substrate_only),
        "donor_plan": "constrained_all_three_label_distinct",
        "strict_all_three_distinct_donor": True,
        "substrate_definition": "legacy_shuffle",
        "dms_delay_ms": int(ctx.cfg.dms_delay_ms),
        "dms_delay_sweep_ms": [int(v) for v in ctx.cfg.dms_delay_sweep_ms],
        "dms_num_trials": int(ctx.cfg.dms_num_trials),
        "dms_batch_size": int(ctx.cfg.dms_batch_size),
        "delay_sweep_completed": bool(ctx.completed_modules.get("dms_delay_sweep", False)),
        "validation_passed": bool(ctx.completed_modules.get("dms_shuffle", False)) if ctx.cfg.run_dms_shuffle else None,
        "compatibility_metrics_available": bool(compatibility_metrics_available),
        "completed_modules": ctx.completed_modules,
        "output_files": ctx.output_files,
        "n_trials": ctx.n_trials,
        "conditions": {"main": list(MAIN_CONDITIONS), "supplementary": list(SUPP_CONDITIONS)},
        "warnings": ctx.warnings,
        "main_claim_supported_fields_available": all(path.exists() for path in required_main),
        "missing_for_main_figure": [_rel(path, ctx.seed_dir) for path in required_main if not path.exists()],
        "missing_for_supplementary": [_rel(path, ctx.seed_dir) for path in required_supp if not path.exists()],
    }
    donor_constraint_summary = ctx.donor_constraint_summary or {
        "donor_constraint_audit_available": False,
        "strict_all_three_distinct_donor": True,
        "n_donor_sample_conflict": 0,
        "n_donor_probe_conflict": 0,
        "n_sample_probe_conflict": 0,
        "n_all_three_distinct_fail": 0,
        "n_self_swap": 0,
        "used_relaxed_rule": 0,
        "donor_constraint_status": "not_run" if not ctx.cfg.run_dms_shuffle else "failed",
    }
    summary.update(donor_constraint_summary)
    _write_json(summary, ctx.seed_dir / "summary.json")
    ctx.output_files["summary"] = "summary.json"
    write_artifact_manifest(ctx, experiment_id=FIGURE_ID, title="Fig.1 functional STSP substrate")
    return summary


def _write_run_log(ctx: ExperimentContext) -> None:
    write_run_log(ctx, now_text=_now())


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    save_csv_with_registry(ctx, df, path)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    write_json_file(payload, path, json_safe_fn=_json_safe)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prepare_dirs(seed_dir: Path) -> dict[str, Path]:
    return prepare_seed_dirs(seed_dir, include_root_layout=True)


def _resolve_seed_dir(output_root: Path, network_seed: int) -> Path:
    return resolve_seed_dir(output_root, network_seed)


def _rel(path: Path, root: Path) -> str:
    return relative_to_root(path, root)


def _ms_to_steps(value_ms: int | float, dt: float) -> int:
    return max(1, int(round((float(value_ms) * ms) / float(dt))))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_from_args(args: argparse.Namespace) -> Fig1Config:
    smoke = bool(args.smoke)
    run_all = bool(args.run_all)
    delay_points_ms = tuple(int(v) for v in str(args.delay_points_ms).split(",") if str(v).strip())
    dms_delay_sweep_ms = tuple(int(v) for v in str(args.dms_delay_sweep_ms).split(",") if str(v).strip())
    if smoke:
        delay_points_ms = delay_points_ms[:2]
        dms_delay_sweep_ms = dms_delay_sweep_ms[:2]
    return Fig1Config(
        model_path=str(args.model_path),
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        network_seed=int(args.network_seed),
        device=str(args.device),
        split=str(args.split),
        baseline_eval_per_class=2 if smoke else int(args.baseline_eval_per_class),
        delay_decode_train_per_class=2 if smoke else int(args.delay_decode_train_per_class),
        delay_decode_test_per_class=2 if smoke else int(args.delay_decode_test_per_class),
        delay_points_ms=delay_points_ms,
        sample_ms=int(args.sample_ms),
        delay_ms=int(args.delay_ms),
        dms_sample_ms=int(args.dms_sample_ms),
        dms_delay_ms=int(args.dms_delay_ms),
        dms_delay_sweep_ms=dms_delay_sweep_ms,
        probe_ms=int(args.probe_ms),
        batch_size=min(int(args.batch_size), 16) if smoke else int(args.batch_size),
        dms_batch_size=min(int(args.dms_batch_size), 4) if smoke else int(args.dms_batch_size),
        dms_num_trials=min(int(args.dms_num_trials), 20) if smoke else int(args.dms_num_trials),
        delay_decode_backend=str(args.delay_decode_backend),
        delay_decode_torch_ridge_lambda=float(args.delay_decode_torch_ridge_lambda),
        run_baseline=run_all or bool(args.run_baseline),
        run_delay_decode=run_all or bool(args.run_delay_decode),
        run_dms_delay_sweep=run_all or bool(args.run_dms_delay_sweep),
        run_dms_shuffle=run_all or bool(args.run_dms_shuffle),
        run_firing_rate_control=run_all or bool(args.run_firing_rate_control),
        save_debug_figures=bool(args.save_debug_figures),
        save_feature_cache=bool(args.save_feature_cache),
        show_progress=not bool(args.no_progress),
        use_encode_cache=not bool(args.no_encode_cache),
        enable_condition_batch=bool(args.enable_condition_batch),
        enable_gpu_metrics=bool(args.enable_gpu_metrics),
        shuffle_compat_mode=bool(args.shuffle_compat_mode),
        pure_substrate_only=bool(args.pure_substrate_only),
        shuffle_num_boot=int(args.shuffle_num_boot),
        shuffle_rng_offset=int(args.shuffle_rng_offset),
        smoke=smoke,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Fig.1 functional STSP substrate experiment.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-delay-decode", action="store_true")
    parser.add_argument("--run-dms-delay-sweep", action="store_true")
    parser.add_argument("--run-dms-shuffle", action="store_true")
    parser.add_argument("--run-firing-rate-control", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--save-feature-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-encode-cache", action="store_true")
    parser.add_argument("--enable-condition-batch", action="store_true")
    parser.add_argument("--enable-gpu-metrics", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shuffle-compat-mode", action="store_true")
    parser.add_argument("--pure-substrate-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-num-boot", type=int, default=1000)
    parser.add_argument("--shuffle-rng-offset", type=int, default=17)
    parser.add_argument("--baseline-eval-per-class", type=int, default=100)
    parser.add_argument("--delay-decode-train-per-class", type=int, default=50)
    parser.add_argument("--delay-decode-test-per-class", type=int, default=50)
    parser.add_argument("--delay-decode-backend", default="torch_linear_probe", choices=["torch_linear_probe", "sklearn_linear_svc"])
    parser.add_argument("--delay-decode-torch-ridge-lambda", type=float, default=1.0)
    parser.add_argument("--delay-points-ms", default="100,200,400,800,1200")
    parser.add_argument("--sample-ms", type=int, default=200)
    parser.add_argument("--delay-ms", type=int, default=1200)
    parser.add_argument("--dms-sample-ms", type=int, default=200)
    parser.add_argument("--dms-delay-ms", type=int, default=400)
    parser.add_argument("--dms-delay-sweep-ms", default="100,200,400,800,1200")
    parser.add_argument("--probe-ms", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dms-batch-size", type=int, default=16)
    parser.add_argument("--dms-num-trials", type=int, default=100)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
