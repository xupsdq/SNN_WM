from __future__ import annotations

import argparse
import json
import math
import sys
import warnings as py_warnings
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Fig1Config:
    model_path: str
    dataset_root: str
    output_root: str
    network_seed: int
    device: str = "auto"
    split: str = "test"
    dt: float = 0.001
    baseline_eval_per_class: int = 100
    delay_decode_train_per_class: int = 50
    delay_decode_test_per_class: int = 50
    delay_points_ms: tuple[int, ...] = (100, 200, 400, 800, 1200)
    sample_ms: int = 200
    delay_ms: int = 1200
    dms_sample_ms: int = 200
    dms_delay_ms: int = 400
    dms_delay_sweep_ms: tuple[int, ...] = (100, 200, 400, 800, 1200)
    probe_ms: int = 100
    batch_size: int = 64
    dms_batch_size: int = 16
    dms_num_trials: int = 100
    run_baseline: bool = False
    run_delay_decode: bool = False
    run_dms_delay_sweep: bool = False
    run_dms_shuffle: bool = False
    run_firing_rate_control: bool = False
    save_debug_figures: bool = False
    save_feature_cache: bool = False
    show_progress: bool = True
    use_encode_cache: bool = True
    enable_condition_batch: bool = False
    shuffle_compat_mode: bool = False
    pure_substrate_only: bool = True
    shuffle_num_boot: int = 1000
    shuffle_rng_offset: int = 17
    smoke: bool = False

    @property
    def sample_steps(self) -> int:
        return _ms_to_steps(self.sample_ms, self.dt)

    @property
    def dms_sample_steps(self) -> int:
        return _ms_to_steps(self.dms_sample_ms, self.dt)

    @property
    def dms_delay_steps(self) -> int:
        return _ms_to_steps(self.dms_delay_ms, self.dt)

    @property
    def dms_delay_sweep_steps(self) -> tuple[int, ...]:
        return tuple(_ms_to_steps(v, self.dt) for v in self.dms_delay_sweep_ms)

    @property
    def probe_steps(self) -> int:
        return _ms_to_steps(self.probe_ms, self.dt)


@dataclass
class ExperimentContext:
    cfg: Fig1Config
    seed_dir: Path
    config_dir: Path
    trial_specs_dir: Path
    raw_dir: Path
    metrics_dir: Path
    debug_dir: Path
    device: torch.device
    dataset: Any
    class_index: dict[int, list[int]]
    net: Any
    encoder: Any
    warnings: list[str]
    output_files: dict[str, str]
    completed_modules: dict[str, bool]
    n_trials: dict[str, int]
    donor_constraint_summary: dict[str, Any]
    run_log: list[str]


@dataclass(frozen=True)
class ProbePrep:
    stsp_mode: str
    pure_substrate_only: int
    target_substrate: str
    reset_applied: int
    restore_ok: int
    legacy_phase_reset_applied: int


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
        _write_config_files(ctx)
        specs = build_trial_specs(ctx)
        if cfg.run_baseline:
            run_baseline_eval(ctx, specs["baseline"])
        if cfg.run_delay_decode:
            run_delay_stsp_decode(ctx, specs["delay_train"], specs["delay_test"])
        if cfg.run_dms_delay_sweep:
            run_dms_functional_delay_sweep(ctx, specs["dms"])
        if cfg.run_dms_shuffle:
            dms_outputs = run_dms_substrate_shuffle(ctx, specs["dms"])
            if cfg.run_firing_rate_control:
                run_phase_firing_rate_control(ctx, dms_outputs.get("phase_rate_rows", []))
        elif cfg.run_firing_rate_control:
            _write_empty_phase_rates(ctx)
            ctx.completed_modules["firing_rate_control"] = True
        if cfg.save_debug_figures:
            save_debug_figures(ctx)
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


def run_baseline_eval(ctx: ExperimentContext, trials: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    batches = _iter_batches(trials, ctx.cfg.batch_size)
    for batch in _progress(
        batches,
        total=math.ceil(len(trials) / ctx.cfg.batch_size),
        desc="fig1 baseline batches",
        enabled=ctx.cfg.show_progress,
    ):
        spikes = _encode_cached(ctx, batch["image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
        with torch.no_grad():
            _ = ctx.net(spikes, layer_idx=3, monitor=False)
        pred, fire_t = decode_prediction_and_fire_time_from_layer3(ctx.net, len(batch))
        pred_np = pred.numpy().astype(int, copy=False)
        fire_np = fire_t.numpy().astype(int, copy=False)
        for i, rec in enumerate(batch.to_dict("records")):
            label = int(rec["label"])
            prediction = int(pred_np[i])
            silent = prediction < 0
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(rec["trial_id"]),
                    "image_id": int(rec["image_id"]),
                    "label": label,
                    "prediction": prediction,
                    "correct": int(prediction == label),
                    "first_fire_time_ms": -1 if silent else int(fire_np[i]),
                    "silent": int(silent),
                }
            )
    pred_df = pd.DataFrame(rows)
    _save_csv(ctx, pred_df, ctx.raw_dir / "panel_b_baseline_trial_predictions.csv")

    n = max(1, len(pred_df))
    n_correct = int(pred_df["correct"].sum()) if not pred_df.empty else 0
    n_silent = int(pred_df["silent"].sum()) if not pred_df.empty else 0
    metrics = pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "overall_recall": float(n_correct / n),
                "error_rate": float(1.0 - n_correct / n),
                "n_trials": int(len(pred_df)),
                "n_correct": n_correct,
                "n_silent": n_silent,
                "silent_rate": float(n_silent / n),
            }
        ]
    )
    _save_csv(ctx, metrics, ctx.metrics_dir / "panel_b_baseline_metrics_by_network.csv")

    recall_rows = []
    for digit in range(NUM_CLASSES):
        sub = pred_df[pred_df["label"] == digit]
        denom = max(1, len(sub))
        recall_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "digit": digit,
                "class_recall": float(sub["correct"].sum() / denom),
                "n_trials": int(len(sub)),
                "n_correct": int(sub["correct"].sum()),
            }
        )
    _save_csv(ctx, pd.DataFrame(recall_rows), ctx.metrics_dir / "supp_class_recall_by_digit.csv")

    conf_rows = []
    for true_label in range(NUM_CLASSES):
        for pred_label in range(-1, NUM_CLASSES):
            count = int(((pred_df["label"] == true_label) & (pred_df["prediction"] == pred_label)).sum())
            conf_rows.append({"network_seed": int(ctx.cfg.network_seed), "true_label": true_label, "pred_label": pred_label, "count": count})
    _save_csv(ctx, pd.DataFrame(conf_rows), ctx.metrics_dir / "supp_confusion_matrix_long.csv")
    ctx.completed_modules["baseline"] = True


def run_delay_stsp_decode(ctx: ExperimentContext, train_trials: pd.DataFrame, test_trials: pd.DataFrame) -> None:
    if LinearSVC is None or f1_score is None:
        raise RuntimeError("scikit-learn is required for delay STSP decoding.")
    all_trials = pd.concat(
        [
            train_trials.drop(columns=["set"], errors="ignore").assign(set="train"),
            test_trials.drop(columns=["set"], errors="ignore").assign(set="test"),
        ],
        ignore_index=True,
    )
    feature_store_lists: dict[tuple[str, int, str], dict[str, list[np.ndarray]]] = {}
    max_delay = max(int(v) for v in ctx.cfg.delay_points_ms)
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    batches = _iter_batches(all_trials, ctx.cfg.batch_size)
    for batch in _progress(
        batches,
        total=math.ceil(len(all_trials) / ctx.cfg.batch_size),
        desc="fig1 delay batches",
        enabled=ctx.cfg.show_progress,
    ):
        spikes = _encode_cached(ctx, batch["image_id"].to_numpy(), ctx.cfg.sample_steps, cache=encode_cache)
        with torch.no_grad():
            _run_sample_then_snapshot_delays(ctx.net, spikes, ctx.cfg.sample_steps, ctx.device, ctx.cfg.delay_points_ms, ctx.cfg.dt, max_delay, batch, feature_store_lists)
    feature_store = _finalize_feature_store(feature_store_lists)

    pred_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for layer in _progress(LAYER_KEYS, total=len(LAYER_KEYS), desc="fig1 decode layers", enabled=ctx.cfg.show_progress):
        for delay_ms in _progress(ctx.cfg.delay_points_ms, total=len(ctx.cfg.delay_points_ms), desc=f"fig1 {layer} delays", enabled=ctx.cfg.show_progress):
            x_train, y_train, ids_train = feature_store[(layer, int(delay_ms), "train")]
            x_test, y_test, ids_test = feature_store[(layer, int(delay_ms), "test")]
            clf = LinearSVC(max_iter=20000, dual=True)
            with py_warnings.catch_warnings(record=True) as caught:
                py_warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(x_train, y_train)
            for item in caught:
                if issubclass(item.category, ConvergenceWarning):
                    ctx.warnings.append(f"LinearSVC convergence warning for {layer} delay_ms={delay_ms}: {item.message}")
            pred = clf.predict(x_test)
            acc = float(np.mean(pred == y_test)) if len(y_test) else float("nan")
            macro_f1 = float(f1_score(y_test, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))
            for trial_id, true_label, pred_label in zip(ids_test, y_test, pred):
                pred_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "layer": layer,
                        "delay_ms": int(delay_ms),
                        "trial_id": int(trial_id),
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "correct": int(pred_label == true_label),
                    }
                )
            feature_path = ""
            if ctx.cfg.save_feature_cache:
                cache_dir = ctx.raw_dir / "feature_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"{layer}_delay_{int(delay_ms)}ms_features.npz"
                np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
                feature_path = _rel(cache_path, ctx.seed_dir)
            for set_name, x_mat, ids in (("train", x_train, ids_train), ("test", x_test, ids_test)):
                manifest_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "layer": layer,
                        "delay_ms": int(delay_ms),
                        "set": set_name,
                        "n_trials": int(x_mat.shape[0]),
                        "n_features": int(x_mat.shape[1]) if x_mat.ndim == 2 else 0,
                        "feature_type": "ux_concat",
                        "feature_cache_saved": int(bool(feature_path)),
                        "feature_cache_path": feature_path,
                    }
                )
            metric_rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "layer": layer,
                    "delay_ms": int(delay_ms),
                    "feature_type": "ux_concat",
                    "classifier": "LinearSVC",
                    "acc": acc,
                    "macro_f1": macro_f1,
                    "chance": 1.0 / NUM_CLASSES,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    metric_df = pd.DataFrame(metric_rows)
    _save_csv(ctx, pred_df, ctx.raw_dir / "panel_c_delay_decode_predictions.csv")
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "panel_c_delay_stsp_features_manifest.csv")
    _save_csv(ctx, metric_df, ctx.metrics_dir / "panel_c_delay_decode_metrics.csv")
    _save_csv(ctx, metric_df.copy(), ctx.metrics_dir / "supp_delay_decode_curve.csv")
    ctx.completed_modules["delay_decode"] = True


def run_dms_functional_delay_sweep(ctx: ExperimentContext, dms_trials: pd.DataFrame) -> None:
    """Supplementary Fig. S2C functional delay sweep for paired DMS probe readout."""
    trial_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    delay_points = tuple(int(v) for v in ctx.cfg.dms_delay_sweep_ms)
    batches = _iter_batches(dms_trials, ctx.cfg.dms_batch_size)
    total_batches = math.ceil(len(dms_trials) / max(1, int(ctx.cfg.dms_batch_size)))
    for batch in _progress(batches, total=total_batches, desc="fig1 dms delay sweep batches", enabled=ctx.cfg.show_progress):
        sample_spikes = _encode_cached(ctx, batch["sample_image_id"].to_numpy(), ctx.cfg.dms_sample_steps, cache=encode_cache)
        probe_spikes = _encode_cached(ctx, batch["probe_image_id"].to_numpy(), ctx.cfg.probe_steps, cache=encode_cache)
        snapshots_by_delay, layer_input_shapes = _run_sample_multi_delay_boundary_capture(
            ctx,
            sample_spikes,
            batch,
            delay_points,
        )
        identity = np.arange(len(batch), dtype=np.int64)
        for delay_ms in delay_points:
            delay_ms = int(delay_ms)
            condition_results = _run_probe_conditions_from_boundary(
                ctx,
                snapshots_by_delay[delay_ms],
                probe_spikes,
                DMS_DELAY_SWEEP_CONDITIONS,
                identity,
                layer_input_shapes,
                start_time_steps=ctx.cfg.dms_sample_steps + _ms_to_steps(delay_ms, ctx.cfg.dt),
            )
            for condition, _intervention, prep, prediction, fire_t in condition_results:
                for i, rec in enumerate(batch.to_dict("records")):
                    pred = int(prediction[i])
                    sample_label = int(rec["sample_label"])
                    probe_label = int(rec["probe_label"])
                    silent = pred < 0
                    fire_t_probe = int(fire_t[i])
                    trial_rows.append(
                        {
                            "network_seed": int(ctx.cfg.network_seed),
                            "trial_id": int(rec["trial_id"]),
                            "delay_ms": delay_ms,
                            "condition": condition,
                            "stsp_mode": prep.stsp_mode,
                            "sample_image_id": int(rec["sample_image_id"]),
                            "sample_label": sample_label,
                            "probe_image_id": int(rec["probe_image_id"]),
                            "probe_label": probe_label,
                            "prediction": pred,
                            "prediction_probe": pred,
                            "correct_probe": int(pred == probe_label),
                            "is_correct_probe": int(pred == probe_label),
                            "pred_is_sample": int(pred == sample_label),
                            "pred_is_original_sample": int(pred == sample_label),
                            "pred_is_probe": int(pred == probe_label),
                            "pred_is_other": int((not silent) and pred not in {sample_label, probe_label}),
                            "first_fire_time_ms": -1 if silent else int(round(fire_t_probe * ctx.cfg.dt / ms)),
                            "first_fire_t_probe": fire_t_probe,
                            "silent": int(silent),
                            "is_silent_probe": int(silent),
                            "sample_probe_same_label": int(sample_label == probe_label),
                            "pure_boundary_restored": 1,
                            "restore_ok": prep.restore_ok,
                            "legacy_phase_reset_applied": prep.legacy_phase_reset_applied,
                        }
                    )

    trial_df = _sort_dms_delay_sweep_trial_readout(pd.DataFrame(trial_rows))
    _validate_dms_delay_sweep_pairing(trial_df, delay_points)
    metrics_df = _delay_sweep_condition_metrics(ctx.cfg.network_seed, trial_df)
    contrast_df = _delay_sweep_contrast(ctx.cfg.network_seed, metrics_df)
    _save_csv(ctx, trial_df, ctx.raw_dir / "supp_dms_delay_sweep_trial_readout.csv")
    _save_csv(ctx, metrics_df, ctx.metrics_dir / "supp_dms_delay_sweep_metrics.csv")
    _save_csv(ctx, contrast_df, ctx.metrics_dir / "supp_dms_delay_sweep_contrast.csv")
    ctx.completed_modules["dms_delay_sweep"] = True


def run_dms_substrate_shuffle(ctx: ExperimentContext, dms_trials: pd.DataFrame) -> dict[str, Any]:
    trial_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    phase_rate_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    rng = np.random.default_rng(int(ctx.cfg.network_seed) + int(ctx.cfg.shuffle_rng_offset))
    batches = _iter_batches(dms_trials, ctx.cfg.dms_batch_size)
    total_batches = math.ceil(len(dms_trials) / max(1, int(ctx.cfg.dms_batch_size)))
    for batch in _progress(batches, total=total_batches, desc="fig1 dms batches", enabled=ctx.cfg.show_progress):
        sample_spikes = _encode_cached(ctx, batch["sample_image_id"].to_numpy(), ctx.cfg.dms_sample_steps, cache=encode_cache)
        probe_spikes = _encode_cached(ctx, batch["probe_image_id"].to_numpy(), ctx.cfg.probe_steps, cache=encode_cache)
        boundary, dynamic_rates, layer_input_shapes = _run_sample_delay_capture(ctx, sample_spikes, batch)
        phase_rate_rows.extend(dynamic_rates)

        sample_labels = batch["sample_label"].to_numpy(dtype=np.int64)
        probe_labels = batch["probe_label"].to_numpy(dtype=np.int64)
        trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
        donor_indices, plan_info = _build_constrained_trial_shuffle_plan(sample_labels, probe_labels, rng)
        donor_trial_ids = trial_ids[donor_indices]
        donor_sample_labels = sample_labels[donor_indices]

        condition_results = _run_probe_conditions_from_boundary(
            ctx,
            boundary,
            probe_spikes,
            SUPP_CONDITIONS,
            donor_indices,
            layer_input_shapes,
        )
        for condition, intervention, prep, prediction, fire_t in condition_results:
            intervention_rows.append(_intervention_manifest_row(ctx.cfg.network_seed, condition, intervention))
            for i, rec in enumerate(batch.to_dict("records")):
                pred = int(prediction[i])
                sample_label = int(rec["sample_label"])
                probe_label = int(rec["probe_label"])
                donor_label = int(donor_sample_labels[i])
                donor_distinct = int(donor_label != sample_label)
                donor_sample_conflict = int(donor_label == sample_label)
                donor_probe_conflict = int(donor_label == probe_label)
                sample_probe_conflict = int(sample_label == probe_label)
                all_three_label_distinct = int(
                    donor_label != sample_label and donor_label != probe_label and sample_label != probe_label
                )
                silent = pred < 0
                fire_t_probe = int(fire_t[i])
                trial_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "trial_id": int(rec["trial_id"]),
                        "condition": condition,
                        "stsp_mode": prep.stsp_mode,
                        "sample_label": sample_label,
                        "probe_label": probe_label,
                        "donor_batch_index": int(donor_indices[i]),
                        "donor_trial_id": int(donor_trial_ids[i]),
                        "donor_sample_label": donor_label,
                        "donor_is_distinct": donor_distinct,
                        "is_self_swap": int(int(donor_indices[i]) == i),
                        "donor_sample_conflict": donor_sample_conflict,
                        "donor_probe_conflict": donor_probe_conflict,
                        "sample_probe_conflict": sample_probe_conflict,
                        "all_three_label_distinct": all_three_label_distinct,
                        "prediction": pred,
                        "prediction_probe": pred,
                        "correct_probe": int(pred == probe_label),
                        "is_correct_probe": int(pred == probe_label),
                        "pred_is_original_sample": int(pred == sample_label),
                        "pred_is_donor_sample": int(pred == donor_label),
                        "pred_is_donor_shifted_memory": int((pred == donor_label) and (donor_label != sample_label)),
                        "pred_is_probe": int(pred == probe_label),
                        "pred_is_other": int((not silent) and pred not in {sample_label, donor_label, probe_label}),
                        "first_fire_time_ms": -1 if silent else int(round(fire_t_probe * ctx.cfg.dt / ms)),
                        "first_fire_t_probe": fire_t_probe,
                        "silent": int(silent),
                        "is_silent_probe": int(silent),
                        "pure_substrate_only": prep.pure_substrate_only,
                        "target_substrate": prep.target_substrate,
                        "restore_ok": prep.restore_ok,
                        "reset_applied": prep.reset_applied,
                        "legacy_phase_reset_applied": prep.legacy_phase_reset_applied,
                        "used_relaxed_rule": int(plan_info["used_relaxed_rule"]),
                        "strict_all_three_distinct": int(plan_info["strict_all_three_distinct"]),
                    }
                )

    trial_df = _sort_trial_readout(pd.DataFrame(trial_rows))
    _validate_fig1_shuffle_pairing(trial_df, pure_substrate_only=ctx.cfg.pure_substrate_only)
    _save_csv(ctx, trial_df, ctx.raw_dir / "panel_d_dms_condition_trial_readout.csv")
    _save_csv(ctx, _compat_trial_readout(trial_df), ctx.raw_dir / "trial_readout_compat.csv")
    audit_df, audit_summary = _donor_constraint_audit(ctx.cfg.network_seed, trial_df)
    ctx.donor_constraint_summary = audit_summary
    _save_csv(ctx, audit_df, ctx.metrics_dir / "supp_dms_shuffle_donor_constraint_audit.csv")
    metrics_df = _condition_metrics(ctx.cfg.network_seed, trial_df)
    _save_csv(ctx, metrics_df[metrics_df["condition"].isin(MAIN_CONDITIONS)].copy(), ctx.metrics_dir / "panel_d_condition_metrics.csv")
    _save_csv(ctx, _attribution_metrics(ctx.cfg.network_seed, metrics_df), ctx.metrics_dir / "panel_e_attribution_metrics.csv")
    supp = metrics_df.copy()
    supp["substrate"] = supp["condition"].map(SUBSTRATE_BY_CONDITION).fillna("")
    supp = supp[
        [
            "network_seed",
            "condition",
            "substrate",
            "acc_probe",
            "error_rate",
            "sample_attribution_rate",
            "donor_attribution_rate",
            "raw_donor_label_match_rate",
            "probe_attribution_rate",
            "other_attribution_rate",
            "silent_rate",
            "n_trials",
        ]
    ]
    _save_csv(ctx, supp, ctx.metrics_dir / "supp_substrate_shuffle_metrics.csv")
    _write_compatibility_metrics(ctx, trial_df)
    intervention_df = pd.DataFrame(intervention_rows).drop_duplicates(["network_seed", "condition", "substrate"], keep="last")
    _save_csv(ctx, intervention_df, ctx.raw_dir / "supp_state_intervention_manifest.csv")
    ctx.completed_modules["dms_shuffle"] = True
    return {"phase_rate_rows": phase_rate_rows}


def run_phase_firing_rate_control(ctx: ExperimentContext, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(columns=["network_seed", "trial_id", "layer", "phase", "time_window_ms", "spike_count", "spike_rate_hz"])
    _save_csv(ctx, df, ctx.metrics_dir / "supp_phase_firing_rates.csv")
    ctx.completed_modules["firing_rate_control"] = True


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


def _run_sample_then_snapshot_delays(net, spikes: torch.Tensor, sample_steps: int, device: torch.device, delay_points_ms: Sequence[int], dt: float, max_delay_ms: int, batch: pd.DataFrame, store: dict) -> None:
    batch_size, _, channels, height, width = spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=device)
    current_time = 0
    snapshots_by_delay: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    def step(input_t: torch.Tensor) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode="dynamic")
        s1p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        s2p = net.pool2(s2.float())
        net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        current_time += 1

    for t in range(sample_steps):
        step(spikes[:, t, ...])
    delay_to_steps = {int(delay): _ms_to_steps(delay, dt) for delay in delay_points_ms}
    for delay_step in range(1, _ms_to_steps(max_delay_ms, dt) + 1):
        step(zero_input)
        for delay_ms, target_step in delay_to_steps.items():
            if delay_step == target_step:
                snapshots_by_delay[int(delay_ms)] = snapshot_ux_state(net, batch_size=batch_size)

    for delay_ms, snapshot in snapshots_by_delay.items():
        for set_name in ("train", "test"):
            mask = batch["set"].astype(str).eq(set_name).to_numpy()
            if not np.any(mask):
                continue
            for layer in LAYER_KEYS:
                ux = np.concatenate([snapshot[layer]["u"][mask], snapshot[layer]["x"][mask]], axis=1)
                labels = batch.loc[mask, "label"].to_numpy(dtype=np.int64)
                trial_ids = batch.loc[mask, "trial_id"].to_numpy(dtype=np.int64)
                key = (layer, int(delay_ms), set_name)
                _append_feature_store(store, key, ux, labels, trial_ids)


def _append_feature_store(store: dict[tuple[str, int, str], dict[str, list[np.ndarray]]], key: tuple[str, int, str], x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> None:
    if key not in store:
        store[key] = {"x": [], "y": [], "ids": []}
    store[key]["x"].append(np.asarray(x))
    store[key]["y"].append(np.asarray(y))
    store[key]["ids"].append(np.asarray(ids))


def _finalize_feature_store(store: dict[tuple[str, int, str], dict[str, list[np.ndarray]]]) -> dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, payload in store.items():
        out[key] = (
            np.vstack(payload["x"]).astype(np.float32, copy=False),
            np.concatenate(payload["y"]).astype(np.int64, copy=False),
            np.concatenate(payload["ids"]).astype(np.int64, copy=False),
        )
    return out


def _run_sample_multi_delay_boundary_capture(
    ctx: ExperimentContext,
    sample_spikes: torch.Tensor,
    batch: pd.DataFrame,
    delay_points_ms: Sequence[int],
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], dict[str, tuple[int, ...]]]:
    # S2C captures full post-delay boundary state for functional probe readout,
    # not only the u/x features used by the STSP decoder.
    net = ctx.net
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    current_time = 0

    def step(input_t: torch.Tensor) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode="dynamic")
        s1p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        s2p = net.pool2(s2.float())
        net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        current_time += 1

    for t in range(ctx.cfg.dms_sample_steps):
        step(sample_spikes[:, t, ...])

    delay_to_steps = {int(delay): _ms_to_steps(delay, ctx.cfg.dt) for delay in delay_points_ms}
    max_delay_steps = max(delay_to_steps.values()) if delay_to_steps else 0
    snapshots_by_delay: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for delay_step in range(1, max_delay_steps + 1):
        step(zero_input)
        for delay_ms, target_step in delay_to_steps.items():
            if delay_step == target_step:
                snapshots_by_delay[int(delay_ms)] = snapshot_boundary_state(net)

    missing = sorted(set(delay_to_steps).difference(snapshots_by_delay))
    if missing:
        first_trial = int(batch.iloc[0]["trial_id"]) if len(batch) else -1
        raise RuntimeError(f"Missing DMS delay sweep boundary snapshots for delay_ms={missing}, batch_first_trial={first_trial}.")
    return snapshots_by_delay, layer_input_shapes


def _run_sample_delay_capture(ctx: ExperimentContext, sample_spikes: torch.Tensor, batch: pd.DataFrame) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    net = ctx.net
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    layer_input_shapes = build_layer_input_shapes(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=ctx.device)
    current_time = 0
    phase_counts = _init_phase_counts(batch)

    def record(layer_key: str, phase: str, spikes_t: torch.Tensor) -> None:
        counts = spikes_t.detach().to(torch.float32).flatten(start_dim=1).sum(dim=1).cpu().numpy()
        for idx, count in enumerate(counts):
            phase_counts[(int(batch.iloc[idx]["trial_id"]), layer_key, phase)] += float(count)

    def step(input_t: torch.Tensor, phase: str) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode="dynamic")
        record("layer1", phase, s1)
        s1p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        record("layer2", phase, s2)
        s2p = net.pool2(s2.float())
        s3, _ = net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        record("layer3", phase, s3)
        current_time += 1

    for t in range(ctx.cfg.dms_sample_steps):
        step(sample_spikes[:, t, ...], "stimulus")
    half_delay = max(1, ctx.cfg.dms_delay_steps // 2)
    for t in range(ctx.cfg.dms_delay_steps):
        step(zero_input, "early_delay" if t < half_delay else "late_delay")
    boundary = snapshot_boundary_state(net)
    rows = []
    windows = {"stimulus": ctx.cfg.dms_sample_steps, "early_delay": half_delay, "late_delay": ctx.cfg.dms_delay_steps - half_delay, "probe": ctx.cfg.probe_steps}
    for (trial_id, layer, phase), count in phase_counts.items():
        window = max(1, windows[phase])
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "trial_id": int(trial_id),
                "layer": layer,
                "phase": phase,
                "time_window_ms": int(round(window * ctx.cfg.dt / ms)),
                "spike_count": float(count),
                "spike_rate_hz": float(count / (window * ctx.cfg.dt)),
            }
        )
    return boundary, rows, layer_input_shapes


def _run_probe_from_boundary(
    ctx: ExperimentContext,
    probe_spikes: torch.Tensor,
    *,
    stsp_mode: str,
    start_time_steps: int,
    force_layer3_probe_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    net = ctx.net
    batch_size = probe_spikes.shape[0]
    with torch.no_grad():
        for t in range(probe_spikes.shape[1]):
            current_time = int(start_time_steps) + int(t)
            input_t = probe_spikes[:, t, ...]
            s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s1p = net.pool1(s1.float())
            s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
            s2p = net.pool2(s2.float())
            layer3_time = int(t) if force_layer3_probe_time else current_time
            net.layer3.forward_step(s2p, layer3_time, training=False, monitor=False, stsp_mode=stsp_mode)
    pred, fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size)
    return pred.numpy().astype(int, copy=False), fire_t.numpy().astype(int, copy=False)


def _run_probe_conditions_from_boundary(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    conditions: Sequence[str],
    donor_indices: np.ndarray,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
    *,
    start_time_steps: int | None = None,
) -> list[tuple[str, dict[str, Any], ProbePrep, np.ndarray, np.ndarray]]:
    if ctx.cfg.enable_condition_batch:
        ctx.warnings.append("Fig.1 condition batch helper is scaffolded; falling back to order-preserving per-condition rollout.")
    results: list[tuple[str, dict[str, Any], ProbePrep, np.ndarray, np.ndarray]] = []
    for condition in _progress(conditions, total=len(conditions), desc="fig1 dms conditions", enabled=ctx.cfg.show_progress):
        prep = _prepare_condition_for_probe(ctx, boundary, condition, donor_indices, layer_input_shapes)
        prediction, fire_t = _run_probe_from_boundary(
            ctx,
            probe_spikes,
            stsp_mode=prep.stsp_mode,
            start_time_steps=ctx.cfg.dms_sample_steps + ctx.cfg.dms_delay_steps if start_time_steps is None else int(start_time_steps),
            force_layer3_probe_time=True,
        )
        results.append((condition, _intervention_for_probe_prep(condition, prep), prep, prediction, fire_t))
    return results


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


def _make_shuffled_substrate_state_from_boundary(
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    substrate: str,
    donor_idx: np.ndarray,
) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    donor_idx = np.asarray(donor_idx, dtype=np.int64)
    for layer_key, layer_state in boundary.items():
        captured: dict[str, torch.Tensor] = {}
        if substrate == "ux":
            key_pairs = (("u", "u_pre"), ("x", "x_pre"))
        elif substrate == "membrane":
            key_pairs = (("v_mem", "v_mem"),)
        elif substrate == "spike":
            key_pairs = (("g_e", "g_e"), ("res", "res"), ("inh_trace", "lateral_inh.inh_trace"))
        else:
            raise ValueError(f"Unsupported shuffle substrate: {substrate}")
        for boundary_key, restore_key in key_pairs:
            if boundary_key not in layer_state:
                continue
            tensor = layer_state[boundary_key]
            if tensor.shape[0] != len(donor_idx):
                raise ValueError(
                    f"{layer_key}.{boundary_key} batch mismatch: state batch={tensor.shape[0]} donor_idx={len(donor_idx)}"
                )
            idx = torch.as_tensor(donor_idx, dtype=torch.long, device=tensor.device)
            captured[restore_key] = tensor.index_select(0, idx).contiguous()
        if captured:
            state[layer_key] = captured
    if not state:
        raise ValueError(f"No state fields were available for shuffle substrate: {substrate}")
    return state


def _restore_substrate_only(net, substrate_state: Mapping[str, Mapping[str, torch.Tensor]]) -> int:
    restore_ok = 1
    with torch.no_grad():
        for layer_key, state_items in substrate_state.items():
            layer = getattr(net, layer_key, None)
            if layer is None:
                restore_ok = 0
                continue
            for state_name, saved in state_items.items():
                if state_name == "lateral_inh.inh_trace":
                    live = getattr(getattr(layer, "lateral_inh", None), "inh_trace", None)
                else:
                    live = getattr(layer, state_name, None)
                if live is None or tuple(live.shape) != tuple(saved.shape):
                    restore_ok = 0
                    continue
                live.copy_(saved.to(device=live.device, dtype=live.dtype))
                live_cpu = live.detach().cpu()
                saved_cpu = saved.detach().cpu().to(dtype=live.dtype)
                if torch.is_floating_point(live_cpu):
                    restored_equal = torch.allclose(live_cpu, saved_cpu, atol=0.0, rtol=0.0, equal_nan=True)
                else:
                    restored_equal = torch.equal(live_cpu, saved_cpu)
                if not restored_equal:
                    restore_ok = 0
    return int(restore_ok)


def _reset_all_layer_states_from_shapes(net, layer_input_shapes: Mapping[str, tuple[int, ...]]) -> None:
    with torch.no_grad():
        for layer_key in LAYER_KEYS:
            layer = getattr(net, layer_key, None)
            if layer is not None:
                layer.reset_state(layer_input_shapes[layer_key])


def _apply_legacy_layer3_probe_phase_reset(net) -> None:
    with torch.no_grad():
        net.layer3.reset_decision_state()
        net.layer3.v_mem.fill_(net.layer3.V_L)
        net.layer3.lateral_inh.reset_state(net.layer3.output_shape)


def _prepare_condition_for_probe(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    condition: str,
    donor_idx: np.ndarray,
    layer_input_shapes: Mapping[str, tuple[int, ...]],
) -> ProbePrep:
    if condition in CONDITION_TO_SUBSTRATE:
        substrate = CONDITION_TO_SUBSTRATE[condition]
        substrate_state = _make_shuffled_substrate_state_from_boundary(boundary, substrate, donor_idx)
        reset_applied = 0
        if ctx.cfg.pure_substrate_only:
            # Boundary-once + pure-substrate reconstruction: sample+delay runs once
            # per batch, then each shuffle probe starts from a clean network with
            # only the donor-shuffled target substrate restored.
            _reset_all_layer_states_from_shapes(ctx.net, layer_input_shapes)
            reset_applied = 1
        else:
            _restore_boundary_state(ctx.net, boundary)
        restore_ok = _restore_substrate_only(ctx.net, substrate_state)
        with torch.no_grad():
            ctx.net.layer3.reset_decision_state()
        return ProbePrep(
            stsp_mode="dynamic",
            pure_substrate_only=int(bool(ctx.cfg.pure_substrate_only)),
            target_substrate=substrate,
            reset_applied=reset_applied,
            restore_ok=restore_ok,
            legacy_phase_reset_applied=0,
        )

    _restore_boundary_state(ctx.net, boundary)
    _apply_legacy_layer3_probe_phase_reset(ctx.net)
    if condition == "dynamic_intact":
        return ProbePrep("dynamic", 0, "none", 0, 1, 1)
    if condition == "static_frozen":
        return ProbePrep("static_frozen", 0, "none", 0, 1, 1)
    raise ValueError(f"Unsupported DMS shuffle condition: {condition}")


def _intervention_for_probe_prep(condition: str, prep: ProbePrep) -> dict[str, Any]:
    if condition == "dynamic_intact":
        return {
            "replaced_variables": [],
            "frozen_variables": [],
            "donor_mapping": "none",
            "notes": "Boundary state restored; dynamic STSP during probe.",
        }
    if condition == "static_frozen":
        return {
            "replaced_variables": [],
            "frozen_variables": [f"{layer_key}.stsp_mode" for layer_key in LAYER_KEYS],
            "donor_mapping": "none",
            "notes": "Boundary state restored; probe uses stsp_mode=static_frozen.",
        }
    replaced = {
        "ux": ["u_pre", "x_pre"],
        "membrane": ["v_mem"],
        "spike": ["g_e", "res", "lateral_inh.inh_trace"],
    }[prep.target_substrate]
    return {
        "replaced_variables": [f"{layer_key}.{name}" for layer_key in LAYER_KEYS for name in replaced],
        "frozen_variables": [],
        "donor_mapping": "constrained_all_three_label_distinct",
        "notes": (
            f"Legacy-compatible pure-substrate trial shuffle for {prep.target_substrate}; "
            f"reset_applied={prep.reset_applied}, restore_ok={prep.restore_ok}."
        ),
    }


def _condition_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in trial_df.groupby("condition", sort=False):
        denom = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "acc_probe": float(part["correct_probe"].sum() / denom),
                "error_rate": float(1.0 - part["correct_probe"].sum() / denom),
                "sample_attribution_rate": float(part["pred_is_original_sample"].sum() / denom),
                "donor_attribution_rate": float(part["pred_is_donor_shifted_memory"].sum() / denom),
                "raw_donor_label_match_rate": float(part["pred_is_donor_sample"].sum() / denom),
                "probe_attribution_rate": float(part["pred_is_probe"].sum() / denom),
                "other_attribution_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    order = {name: idx for idx, name in enumerate(SUPP_CONDITIONS)}
    df = pd.DataFrame(rows)
    df["_order"] = df["condition"].map(order).fillna(99)
    return df.sort_values("_order", kind="stable").drop(columns=["_order"]).reset_index(drop=True)


def _delay_sweep_condition_metrics(network_seed: int, trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (delay_ms, condition), part in trial_df.groupby(["delay_ms", "condition"], sort=False):
        denom = max(1, len(part))
        n_correct = int(part["correct_probe"].sum())
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay_ms": int(delay_ms),
                "condition": str(condition),
                "acc_probe": float(n_correct / denom),
                "error_rate": float(1.0 - n_correct / denom),
                "sample_attribution_rate": float(part["pred_is_original_sample"].sum() / denom),
                "probe_attribution_rate": float(part["pred_is_probe"].sum() / denom),
                "other_attribution_rate": float(part["pred_is_other"].sum() / denom),
                "silent_rate": float(part["silent"].sum() / denom),
                "n_trials": int(len(part)),
            }
        )
    order = {name: idx for idx, name in enumerate(DMS_DELAY_SWEEP_CONDITIONS)}
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "network_seed",
                "delay_ms",
                "condition",
                "acc_probe",
                "error_rate",
                "sample_attribution_rate",
                "probe_attribution_rate",
                "other_attribution_rate",
                "silent_rate",
                "n_trials",
            ]
        )
    df["_order"] = df["condition"].map(order).fillna(99)
    return df.sort_values(["delay_ms", "_order"], kind="stable").drop(columns=["_order"]).reset_index(drop=True)


def _delay_sweep_contrast(network_seed: int, metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for delay_ms, part in metrics_df.groupby("delay_ms", sort=True):
        by_condition = {str(row["condition"]): row for row in part.to_dict("records")}
        dynamic = by_condition.get("dynamic_intact")
        static = by_condition.get("static_frozen")
        if dynamic is None or static is None:
            raise ValueError(f"Missing dynamic/static delay sweep metrics for delay_ms={int(delay_ms)}.")
        acc_dynamic = float(dynamic["acc_probe"])
        acc_static = float(static["acc_probe"])
        sample_bias_dynamic = float(dynamic["sample_attribution_rate"])
        sample_bias_static = float(static["sample_attribution_rate"])
        rows.append(
            {
                "network_seed": int(network_seed),
                "delay_ms": int(delay_ms),
                "acc_dynamic": acc_dynamic,
                "acc_static": acc_static,
                "stsp_interference": float(acc_static - acc_dynamic),
                "stsp_modulation_signed": float(acc_dynamic - acc_static),
                "sample_bias_dynamic": sample_bias_dynamic,
                "sample_bias_static": sample_bias_static,
                "sample_bias_excess_dynamic_minus_static": float(sample_bias_dynamic - sample_bias_static),
                "silent_dynamic": float(dynamic["silent_rate"]),
                "silent_static": float(static["silent_rate"]),
                "n_trials_dynamic": int(dynamic["n_trials"]),
                "n_trials_static": int(static["n_trials"]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "network_seed",
            "delay_ms",
            "acc_dynamic",
            "acc_static",
            "stsp_interference",
            "stsp_modulation_signed",
            "sample_bias_dynamic",
            "sample_bias_static",
            "sample_bias_excess_dynamic_minus_static",
            "silent_dynamic",
            "silent_static",
            "n_trials_dynamic",
            "n_trials_static",
        ],
    )


def _sort_trial_readout(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return trial_df
    order = {condition: idx for idx, condition in enumerate(SUPP_CONDITIONS)}
    out = trial_df.copy()
    out["_condition_order"] = out["condition"].map(order).fillna(99).astype(int)
    out = out.sort_values(["trial_id", "_condition_order"], kind="stable").drop(columns=["_condition_order"])
    return out.reset_index(drop=True)


def _sort_dms_delay_sweep_trial_readout(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "trial_id",
        "delay_ms",
        "condition",
        "stsp_mode",
        "sample_image_id",
        "sample_label",
        "probe_image_id",
        "probe_label",
        "prediction",
        "prediction_probe",
        "correct_probe",
        "is_correct_probe",
        "pred_is_sample",
        "pred_is_original_sample",
        "pred_is_probe",
        "pred_is_other",
        "first_fire_time_ms",
        "first_fire_t_probe",
        "silent",
        "is_silent_probe",
        "sample_probe_same_label",
        "pure_boundary_restored",
        "restore_ok",
        "legacy_phase_reset_applied",
    ]
    if trial_df.empty:
        return pd.DataFrame(columns=columns)
    order = {condition: idx for idx, condition in enumerate(DMS_DELAY_SWEEP_CONDITIONS)}
    out = trial_df.copy()
    out["_condition_order"] = out["condition"].map(order).fillna(99).astype(int)
    out = out.sort_values(["trial_id", "delay_ms", "_condition_order"], kind="stable").drop(columns=["_condition_order"])
    return out[[col for col in columns if col in out.columns]].reset_index(drop=True)


def _validate_dms_delay_sweep_pairing(trial_df: pd.DataFrame, delay_points_ms: Sequence[int]) -> None:
    required_columns = {
        "trial_id",
        "delay_ms",
        "condition",
        "sample_label",
        "probe_label",
        "sample_image_id",
        "probe_image_id",
        "prediction",
    }
    missing = sorted(required_columns.difference(trial_df.columns))
    if missing:
        raise ValueError(f"DMS delay sweep trial readout is missing required columns: {missing}")
    expected_delays = tuple(int(v) for v in delay_points_ms)
    expected_conditions = set(DMS_DELAY_SWEEP_CONDITIONS)
    actual_conditions = set(trial_df["condition"].astype(str).unique())
    if actual_conditions != expected_conditions:
        raise ValueError(f"DMS delay sweep conditions mismatch: expected={sorted(expected_conditions)}, actual={sorted(actual_conditions)}")
    actual_delays = set(int(v) for v in trial_df["delay_ms"].unique())
    if actual_delays != set(expected_delays):
        raise ValueError(f"DMS delay sweep delay_ms mismatch: expected={sorted(set(expected_delays))}, actual={sorted(actual_delays)}")

    counts = trial_df.groupby(["trial_id", "delay_ms"])["condition"].nunique(dropna=False)
    if not (counts == len(DMS_DELAY_SWEEP_CONDITIONS)).all():
        bad = counts[counts != len(DMS_DELAY_SWEEP_CONDITIONS)].index.tolist()
        raise ValueError(f"Each trial_id x delay_ms must include both delay sweep conditions. Bad pairs: {bad[:10]}")

    invariant_cols = ["sample_label", "probe_label", "sample_image_id", "probe_image_id"]
    for col in invariant_cols:
        uniq = trial_df.groupby("trial_id")[col].nunique(dropna=False)
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not paired-identical across delays/conditions for ids: {bad_ids[:10]}")
    if bool((trial_df["sample_label"].astype(int) == trial_df["probe_label"].astype(int)).any()):
        raise ValueError("DMS delay sweep requires sample_label != probe_label for every row.")

    pred = pd.to_numeric(trial_df["prediction"], errors="coerce")
    if pred.isna().any() or bool(((pred < -1) | (pred >= NUM_CLASSES)).any()):
        raise ValueError("DMS delay sweep prediction contains non-integer or out-of-range labels.")


def _validate_fig1_shuffle_pairing(trial_df: pd.DataFrame, pure_substrate_only: bool) -> None:
    required_columns = {
        "trial_id",
        "condition",
        "sample_label",
        "probe_label",
        "donor_batch_index",
        "donor_trial_id",
        "donor_sample_label",
        "is_self_swap",
        "used_relaxed_rule",
        "donor_sample_conflict",
        "donor_probe_conflict",
        "sample_probe_conflict",
        "all_three_label_distinct",
        "donor_is_distinct",
        "reset_applied",
        "restore_ok",
        "prediction",
    }
    missing = sorted(required_columns.difference(trial_df.columns))
    if missing:
        raise ValueError(f"Fig.1 shuffle trial readout is missing required columns: {missing}")
    expected = len(SUPP_CONDITIONS)
    count_per_trial = trial_df.groupby("trial_id").size()
    if not (count_per_trial == expected).all():
        bad_ids = count_per_trial[count_per_trial != expected].index.tolist()
        raise ValueError(f"Each trial_id must appear exactly {expected} times. Bad ids: {bad_ids[:10]}")
    expected_conditions = list(SUPP_CONDITIONS)
    for trial_id, part in trial_df.groupby("trial_id", sort=False):
        conditions = part["condition"].astype(str).tolist()
        if conditions != expected_conditions:
            raise ValueError(f"Condition order mismatch for trial_id={int(trial_id)}: {conditions}")
    for col in [
        "sample_label",
        "probe_label",
        "donor_trial_id",
        "donor_sample_label",
        "donor_batch_index",
        "is_self_swap",
        "used_relaxed_rule",
        "donor_sample_conflict",
        "donor_probe_conflict",
        "sample_probe_conflict",
        "all_three_label_distinct",
        "donor_is_distinct",
    ]:
        uniq = trial_df.groupby("trial_id")[col].nunique(dropna=False)
        if not (uniq == 1).all():
            bad_ids = uniq[uniq != 1].index.tolist()
            raise ValueError(f"{col} is not paired-identical across conditions for ids: {bad_ids[:10]}")
    if bool((trial_df["sample_probe_conflict"] != 0).any()):
        bad_ids = _bad_trial_ids(trial_df, trial_df["sample_probe_conflict"] != 0)
        raise ValueError(f"Found sample_probe_conflict in Fig.1 shuffle trial readout. Bad trial_id: {bad_ids}")
    shuffle_rows = trial_df[trial_df["condition"].isin(SHUFFLE_CONDITIONS)]
    if len(shuffle_rows):
        checks = [
            ("donor_sample_conflict", "Found donor_sample_conflict in strict donor mapping."),
            ("donor_probe_conflict", "Found donor_probe_conflict in strict donor mapping."),
            ("all_three_label_distinct", "Found all_three_label_distinct failure in strict donor mapping.", 1),
            ("donor_is_distinct", "Found donor_is_distinct != 1 in strict donor mapping.", 1),
            ("is_self_swap", "Found self swap in strict donor mapping."),
        ]
        for item in checks:
            col = item[0]
            message = item[1]
            expected_value = item[2] if len(item) > 2 else 0
            mask = shuffle_rows[col] != expected_value
            if bool(mask.any()):
                bad_ids = _bad_trial_ids(shuffle_rows, mask)
                raise ValueError(f"{message} Bad trial_id: {bad_ids}")
    if len(shuffle_rows) and bool(pure_substrate_only):
        if bool((shuffle_rows["reset_applied"] != 1).any()):
            raise ValueError("Pure substrate mode expected reset_applied=1 for all shuffle rows.")
        if bool((shuffle_rows["restore_ok"] != 1).any()):
            raise ValueError("Pure substrate mode has restore_ok=0 rows.")
    pred = pd.to_numeric(trial_df["prediction"], errors="coerce")
    if pred.isna().any() or bool(((pred < -1) | (pred >= NUM_CLASSES)).any()):
        raise ValueError("prediction contains non-integer or out-of-range labels.")


def _bad_trial_ids(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    return [int(v) for v in df.loc[mask, "trial_id"].drop_duplicates().head(10).tolist()]


def _compat_trial_readout(trial_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network_seed",
        "trial_id",
        "condition",
        "stsp_mode",
        "sample_label",
        "probe_label",
        "donor_batch_index",
        "donor_trial_id",
        "donor_sample_label",
        "donor_is_distinct",
        "is_self_swap",
        "donor_sample_conflict",
        "donor_probe_conflict",
        "sample_probe_conflict",
        "all_three_label_distinct",
        "prediction_probe",
        "first_fire_t_probe",
        "is_correct_probe",
        "is_silent_probe",
        "pred_is_original_sample",
        "pred_is_donor_sample",
        "pred_is_donor_shifted_memory",
        "pure_substrate_only",
        "target_substrate",
        "restore_ok",
        "reset_applied",
        "legacy_phase_reset_applied",
        "used_relaxed_rule",
        "strict_all_three_distinct",
    ]
    return trial_df[[col for col in columns if col in trial_df.columns]].copy()


def _write_compatibility_metrics(ctx: ExperimentContext, trial_df: pd.DataFrame) -> None:
    if compat_compute_condition_metrics is None or compat_compute_bias_table is None or compat_compute_collapse_summary is None:
        ctx.warnings.append("Compatibility shuffle metrics were not generated because shared shuffle_metrics helpers are unavailable.")
        return
    try:
        compat_trials = _compat_trial_readout(trial_df)
        metrics_condition = compat_compute_condition_metrics(
            compat_trials,
            condition_order=SUPP_CONDITIONS,
            shuffle_condition="ux_trial_shuffle",
            static_condition="static_frozen",
        )
        metrics_bias = compat_compute_bias_table(compat_trials, NUM_CLASSES, condition_order=SUPP_CONDITIONS)
        collapse_summary, bootstrap_tests = compat_compute_collapse_summary(
            compat_trials,
            metrics_condition,
            metrics_bias,
            n_boot=int(ctx.cfg.shuffle_num_boot),
            seed=int(ctx.cfg.network_seed) + 100,
            dynamic_condition="dynamic_intact",
            shuffle_condition="ux_trial_shuffle",
            static_condition="static_frozen",
        )
    except Exception as exc:
        ctx.warnings.append(f"Compatibility shuffle metrics were not generated: {exc}")
        return
    _save_csv(ctx, metrics_condition, ctx.metrics_dir / "compat_metrics_condition_summary.csv")
    _save_csv(ctx, metrics_bias, ctx.metrics_dir / "compat_metrics_error_bias.csv")
    _save_csv(ctx, collapse_summary, ctx.metrics_dir / "compat_metrics_collapse_summary.csv")
    _save_csv(ctx, bootstrap_tests, ctx.metrics_dir / "compat_metrics_bootstrap_tests.csv")


def _donor_constraint_audit(network_seed: int, trial_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, part in trial_df.groupby("condition", sort=False):
        n_all_three_distinct_fail = int((part["all_three_label_distinct"] != 1).sum())
        rows.append(
            {
                "network_seed": int(network_seed),
                "audit_type": "dms_shuffle_donor_constraint",
                "condition": str(condition),
                "n_trials": int(len(part)),
                "n_donor_sample_conflict": int(part["donor_sample_conflict"].sum()),
                "n_donor_probe_conflict": int(part["donor_probe_conflict"].sum()),
                "n_sample_probe_conflict": int(part["sample_probe_conflict"].sum()),
                "n_all_three_distinct_fail": n_all_three_distinct_fail,
                "n_self_swap": int(part["is_self_swap"].sum()),
                "strict_all_three_distinct": 1,
                "used_relaxed_rule": int(part["used_relaxed_rule"].max()) if len(part) else 0,
                "notes": "Strict all-three-distinct donor mapping audit by condition.",
            }
        )

    if trial_df.empty:
        unique_trials = trial_df
    else:
        unique_trials = trial_df.drop_duplicates("trial_id", keep="first")
    summary = {
        "donor_constraint_audit_available": bool(len(rows)),
        "strict_all_three_distinct_donor": True,
        "n_donor_sample_conflict": int(unique_trials["donor_sample_conflict"].sum()) if len(unique_trials) else 0,
        "n_donor_probe_conflict": int(unique_trials["donor_probe_conflict"].sum()) if len(unique_trials) else 0,
        "n_sample_probe_conflict": int(unique_trials["sample_probe_conflict"].sum()) if len(unique_trials) else 0,
        "n_all_three_distinct_fail": int((unique_trials["all_three_label_distinct"] != 1).sum()) if len(unique_trials) else 0,
        "n_self_swap": int(unique_trials["is_self_swap"].sum()) if len(unique_trials) else 0,
        "used_relaxed_rule": int(unique_trials["used_relaxed_rule"].max()) if len(unique_trials) else 0,
    }
    fail_keys = [
        "n_donor_sample_conflict",
        "n_donor_probe_conflict",
        "n_sample_probe_conflict",
        "n_all_three_distinct_fail",
        "n_self_swap",
        "used_relaxed_rule",
    ]
    summary["donor_constraint_status"] = "failed" if any(int(summary[key]) > 0 for key in fail_keys) else "passed"
    return pd.DataFrame(rows), summary


def _attribution_metrics(network_seed: int, metrics_df: pd.DataFrame) -> pd.DataFrame:
    dynamic = metrics_df[metrics_df["condition"] == "dynamic_intact"]
    dyn_original = float(dynamic["sample_attribution_rate"].iloc[0]) if not dynamic.empty else float("nan")
    dyn_donor = float(dynamic["donor_attribution_rate"].iloc[0]) if not dynamic.empty else float("nan")
    rows = []
    for condition in ("dynamic_intact", "ux_trial_shuffle"):
        row = metrics_df[metrics_df["condition"] == condition]
        if row.empty:
            continue
        original = float(row["sample_attribution_rate"].iloc[0])
        donor = float(row["donor_attribution_rate"].iloc[0])
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": condition,
                "original_sample_attribution": original,
                "donor_sample_attribution": donor,
                "donor_shift_gain_vs_dynamic": float(donor - dyn_donor),
                "original_drop_vs_dynamic": float(dyn_original - original),
            }
        )
    return pd.DataFrame(rows)


def _balanced_image_trials(class_index: Mapping[int, Sequence[int]], per_class: int, rng: np.random.Generator, network_seed: int, split: str, id_prefix: str) -> pd.DataFrame:
    rows = []
    trial_id = 0
    for cls in range(NUM_CLASSES):
        indices = _sample_indices(class_index[cls], int(per_class), rng, replace=len(class_index[cls]) < int(per_class))
        for image_id in indices:
            rows.append({"network_seed": int(network_seed), "set": id_prefix, "trial_id": trial_id, "image_id": int(image_id), "label": cls, "class": cls, "split": split})
            trial_id += 1
    rng.shuffle(rows)
    for new_id, row in enumerate(rows):
        row["trial_id"] = int(new_id)
    return pd.DataFrame(rows)


def _balanced_disjoint_delay_trials(class_index: Mapping[int, Sequence[int]], train_per_class: int, test_per_class: int, rng: np.random.Generator, network_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_rows, test_rows = [], []
    trial_train, trial_test = 0, 0
    overlap = 0
    for cls in range(NUM_CLASSES):
        indices = np.asarray(class_index[cls], dtype=np.int64)
        perm = rng.permutation(indices)
        need = int(train_per_class) + int(test_per_class)
        if len(perm) >= need:
            train_idx = perm[: int(train_per_class)]
            test_idx = perm[int(train_per_class) : need]
        else:
            train_idx = _sample_indices(indices, int(train_per_class), rng, replace=len(indices) < int(train_per_class))
            remaining = np.asarray([idx for idx in indices if idx not in set(train_idx)], dtype=np.int64)
            source = remaining if len(remaining) >= int(test_per_class) else indices
            test_idx = _sample_indices(source, int(test_per_class), rng, replace=len(source) < int(test_per_class))
            overlap += len(set(map(int, train_idx)).intersection(set(map(int, test_idx))))
        for image_id in train_idx:
            train_rows.append({"network_seed": int(network_seed), "set": "train", "trial_id": trial_train, "image_id": int(image_id), "label": cls, "class": cls})
            trial_train += 1
        for image_id in test_idx:
            test_rows.append({"network_seed": int(network_seed), "set": "test", "trial_id": trial_test, "image_id": int(image_id), "label": cls, "class": cls})
            trial_test += 1
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    for idx, row in enumerate(train_rows):
        row["trial_id"] = int(idx)
    for idx, row in enumerate(test_rows):
        row["trial_id"] = int(idx)
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows), int(overlap)


def _build_dms_trials(class_index: Mapping[int, Sequence[int]], n_trials: int, rng: np.random.Generator, network_seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sample_labels = np.asarray([i % NUM_CLASSES for i in range(int(n_trials))], dtype=np.int64)
    rng.shuffle(sample_labels)
    rows = []
    for trial_id, sample_label in enumerate(sample_labels):
        probe_choices = [c for c in range(NUM_CLASSES) if c != int(sample_label)]
        probe_label = int(rng.choice(probe_choices))
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "sample_image_id": int(rng.choice(class_index[int(sample_label)])),
                "sample_label": int(sample_label),
                "probe_image_id": int(rng.choice(class_index[probe_label])),
                "probe_label": probe_label,
            }
        )
    df = pd.DataFrame(rows)
    audit_rows = [
        {
            "network_seed": int(network_seed),
            "audit_type": "donor_plan",
            "label": "all",
            "count": int(len(df)),
            "fixed_point_count": 0,
            "fixed_point_rate": 0.0,
            "notes": "Donor mapping is constructed per DMS batch with strict all-three-distinct semantics.",
        }
    ]
    for col in ("sample_label", "probe_label"):
        for label, count in df[col].value_counts().sort_index().items():
            audit_rows.append(
                {
                    "network_seed": int(network_seed),
                    "audit_type": f"class_count_{col}",
                    "label": int(label),
                    "count": int(count),
                    "fixed_point_count": 0,
                    "fixed_point_rate": 0.0,
                    "notes": "",
                }
            )
    return df, audit_rows


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    perm = np.roll(np.arange(n), 1)
    return perm


def _build_constrained_trial_shuffle_plan(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    sample_labels = np.asarray(sample_labels, dtype=np.int64)
    probe_labels = np.asarray(probe_labels, dtype=np.int64)
    if len(sample_labels) != len(probe_labels):
        raise ValueError("sample_labels and probe_labels must have the same length.")
    n = len(sample_labels)
    identity = np.arange(n, dtype=np.int64)
    if n <= 1:
        donor_idx = None
    else:
        donor_idx = _build_constrained_permutation_np(
            sample_labels,
            probe_labels,
            rng,
            require_no_self=True,
            require_all_three_label_distinct=True,
        )
    if donor_idx is None:
        sample_counts = {int(k): int(v) for k, v in zip(*np.unique(sample_labels, return_counts=True))}
        probe_counts = {int(k): int(v) for k, v in zip(*np.unique(probe_labels, return_counts=True))}
        raise RuntimeError(
            "Failed to build strict all-three-distinct DMS donor mapping: batch composition cannot support "
            "all-three-distinct donor mapping. Increase dms_batch_size, use a balanced DMS batch construction, "
            "or explicitly disable strict mode only for debugging. "
            f"batch_size={n}, sample_label_counts={sample_counts}, probe_label_counts={probe_counts}"
        )

    donor_sample = sample_labels[donor_idx]
    n_donor_sample_conflict = int(np.sum(donor_sample == sample_labels))
    n_donor_probe_conflict = int(np.sum(donor_sample == probe_labels))
    n_self_swap = int(np.sum(donor_idx == identity))
    if n_donor_sample_conflict or n_donor_probe_conflict or n_self_swap:
        raise RuntimeError(
            "Invalid strict shuffle plan: "
            f"n_donor_sample_conflict={n_donor_sample_conflict}, "
            f"n_donor_probe_conflict={n_donor_probe_conflict}, n_self_swap={n_self_swap}."
        )
    if np.any(donor_sample == probe_labels):
        raise RuntimeError("Invalid shuffle plan: donor_sample_label equals receiver probe_label.")
    return donor_idx.astype(np.int64, copy=False), {
        "n_self_swap": n_self_swap,
        "used_relaxed_rule": 0,
        "strict_all_three_distinct": 1,
        "n_donor_sample_conflict": n_donor_sample_conflict,
        "n_donor_probe_conflict": n_donor_probe_conflict,
    }


def _build_constrained_permutation_np(
    sample_labels: np.ndarray,
    probe_labels: np.ndarray,
    rng: np.random.Generator,
    *,
    require_no_self: bool,
    require_all_three_label_distinct: bool = True,
) -> np.ndarray | None:
    n = len(sample_labels)
    candidates: list[list[int]] = []
    for recv_i in range(n):
        receiver_sample_label = sample_labels[recv_i]
        receiver_probe_label = probe_labels[recv_i]
        cand = [
            donor_i
            for donor_i in range(n)
            if (not require_no_self or donor_i != recv_i)
            and sample_labels[donor_i] != receiver_probe_label
            and (not require_all_three_label_distinct or sample_labels[donor_i] != receiver_sample_label)
        ]
        if not cand:
            return None
        rng.shuffle(cand)
        candidates.append(cand)

    order = sorted(range(n), key=lambda idx: len(candidates[idx]))
    donor_for_recv = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=np.bool_)

    def dfs(depth: int) -> bool:
        if depth == n:
            return True
        recv_i = order[depth]
        cand = candidates[recv_i][:]
        rng.shuffle(cand)
        for donor_i in cand:
            if used[donor_i]:
                continue
            used[donor_i] = True
            donor_for_recv[recv_i] = donor_i
            if dfs(depth + 1):
                return True
            donor_for_recv[recv_i] = -1
            used[donor_i] = False
        return False

    return donor_for_recv if dfs(0) else None


def _sample_indices(indices: Sequence[int], count: int, rng: np.random.Generator, replace: bool) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if len(arr) == 0:
        raise ValueError("Cannot sample from an empty class index.")
    return rng.choice(arr, size=int(count), replace=bool(replace))


def _images_for_ids(dataset, image_ids: Iterable[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)


def _encode_cached(ctx: ExperimentContext, image_ids: Iterable[int], steps: int, *, cache: dict[tuple[Any, ...], torch.Tensor]) -> torch.Tensor:
    ids = tuple(int(v) for v in image_ids)
    key = (ids, int(steps), str(ctx.device))
    if (not ctx.cfg.use_encode_cache) or key not in cache:
        images = _images_for_ids(ctx.dataset, ids).to(ctx.device)
        spikes = encode_images(ctx.encoder, images, int(steps))
        if not ctx.cfg.use_encode_cache:
            return spikes
        cache[key] = spikes
    return cache[key]


def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), int(batch_size)):
        yield df.iloc[start : start + int(batch_size)].reset_index(drop=True)


def _donor_indices_for_batch(batch: pd.DataFrame) -> np.ndarray:
    trial_ids = batch["trial_id"].to_numpy(dtype=np.int64)
    index_by_trial = {int(trial_id): idx for idx, trial_id in enumerate(trial_ids)}
    donor = []
    for donor_id in batch["donor_trial_id"].to_numpy(dtype=np.int64):
        donor.append(index_by_trial.get(int(donor_id), 0))
    return np.asarray(donor, dtype=np.int64)


def _init_phase_counts(batch: pd.DataFrame) -> dict[tuple[int, str, str], float]:
    out: dict[tuple[int, str, str], float] = {}
    for trial_id in batch["trial_id"].astype(int).tolist():
        for layer in LAYER_KEYS:
            for phase in ("stimulus", "early_delay", "late_delay", "probe"):
                out[(int(trial_id), layer, phase)] = 0.0
    return out


def _intervention_manifest_row(network_seed: int, condition: str, intervention: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "network_seed": int(network_seed),
        "condition": condition,
        "substrate": SUBSTRATE_BY_CONDITION.get(condition, ""),
        "replaced_variables": ";".join(intervention.get("replaced_variables", [])),
        "frozen_variables": ";".join(intervention.get("frozen_variables", [])),
        "donor_mapping": str(intervention.get("donor_mapping", "")),
        "notes": str(intervention.get("notes", "")),
    }


def _write_empty_phase_rates(ctx: ExperimentContext) -> None:
    _save_csv(ctx, pd.DataFrame(columns=["network_seed", "trial_id", "layer", "phase", "time_window_ms", "spike_count", "spike_rate_hz"]), ctx.metrics_dir / "supp_phase_firing_rates.csv")


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
        "primary_metric": "probe accuracy vs delay",
        "contrast": "acc_static - acc_dynamic",
    }
    _write_json(payload, ctx.config_dir / "run_config.json")
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
    return summary


def _write_run_log(ctx: ExperimentContext) -> None:
    ctx.run_log.append(f"{_now()} completed modules={sorted(k for k, v in ctx.completed_modules.items() if v)}")
    path = ctx.seed_dir / "run_log.txt"
    path.write_text("\n".join(ctx.run_log) + "\n", encoding="utf-8")
    ctx.output_files["run_log"] = "run_log.txt"


def _save_csv(ctx: ExperimentContext, df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    ctx.output_files[path.stem] = _rel(path, ctx.seed_dir)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shuffle-compat-mode", action="store_true")
    parser.add_argument("--pure-substrate-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-num-boot", type=int, default=1000)
    parser.add_argument("--shuffle-rng-offset", type=int, default=17)
    parser.add_argument("--baseline-eval-per-class", type=int, default=100)
    parser.add_argument("--delay-decode-train-per-class", type=int, default=50)
    parser.add_argument("--delay-decode-test-per-class", type=int, default=50)
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
