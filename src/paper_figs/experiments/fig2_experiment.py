from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_figs.common.io import (
    prepare_layout,
    save_csv,
    save_json,
    save_npz,
    write_artifact_manifest,
)
from src.paper_figs.common.model_env import (
    DT,
    build_class_index,
    encode_images,
    load_mnist_skeleton_dataset,
    load_paper_model_and_encoder,
)
from src.paper_figs.common.runtime import (
    build_common_parser,
    format_smoke_command,
    resolve_device_strict,
    run_python_module,
    seed_everything,
    setup_logger,
)
from src.paper_figs.common.sampling import sample_mismatch_pair_specs

FIGURE_ID = "fig2"
MODULE_NAME = "src.paper_figs.experiments.fig2_experiment"
DEFAULT_OUTPUT_DIR = str(Path("results") / "paper_figs" / FIGURE_ID)
DEFAULT_SAMPLE_MS = 200.0
DEFAULT_PROBE_MS = 100.0
DEFAULT_DELAY_POINTS_MS = [100, 200, 300, 400, 500, 600, 1000, 1500, 2000]
DEFAULT_DECODE_DELAY_POINTS_MS = [100, 200, 400, 800, 1200]


@dataclass(frozen=True)
class DmsSpec:
    sample_ms: float
    probe_ms: float

    @property
    def sample_steps(self) -> int:
        return int(round(self.sample_ms))

    @property
    def probe_steps(self) -> int:
        return int(round(self.probe_ms))


def build_argparser():
    parser = build_common_parser(
        description="Fig2 paper experiment: delay dependence, silent raster, engram decode, and substrate shuffle.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--sample-ms", type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument("--probe-ms", type=float, default=DEFAULT_PROBE_MS)
    return parser


def build_fig2_config(smoke: bool) -> dict[str, object]:
    if smoke:
        return {
            "delay_points_ms": [100, 300, 600],
            "num_pairs": 24,
            "decode_delay_points_ms": [100, 400],
            "decode_train_per_class": 4,
            "decode_test_per_class": 2,
            "decode_batch_size": 8,
            "shuffle_trials": 24,
            "shuffle_batch_size": 8,
            "shuffle_num_boot": 64,
            "representative_sample_label": 1,
            "representative_probe_label": 3,
            "rate_smooth_window": 11,
        }
    return {
        "delay_points_ms": list(DEFAULT_DELAY_POINTS_MS),
        "num_pairs": 300,
        "decode_delay_points_ms": list(DEFAULT_DECODE_DELAY_POINTS_MS),
        "decode_train_per_class": 100,
        "decode_test_per_class": 100,
        "decode_batch_size": 1,
        "shuffle_trials": 500,
        "shuffle_batch_size": 32,
        "shuffle_num_boot": 5000,
        "representative_sample_label": 1,
        "representative_probe_label": 3,
        "rate_smooth_window": 15,
    }


def _load_images_for_specs(dataset, df_specs: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sample_images = torch.stack([dataset[int(idx)][0].detach().cpu().to(torch.float32) for idx in df_specs["sample_index"]], dim=0).to(device)
    probe_images = torch.stack([dataset[int(idx)][0].detach().cpu().to(torch.float32) for idx in df_specs["probe_index"]], dim=0).to(device)
    return sample_images, probe_images


def build_dms_pair_bank(dataset, class_index: Dict[int, List[int]], *, num_pairs: int, seed: int) -> pd.DataFrame:
    return sample_mismatch_pair_specs(class_index, num_pairs=num_pairs, num_classes=len(class_index), seed=seed)


def compute_memory_effect_vs_delay(
    net,
    sample_spikes: torch.Tensor,
    probe_spikes: torch.Tensor,
    sample_labels: torch.Tensor,
    probe_labels: torch.Tensor,
    delay_points_ms: Iterable[int],
) -> pd.DataFrame:
    rows = []
    total_samples = int(probe_labels.numel())
    for delay_ms in delay_points_ms:
        delay_steps = int(round(float(delay_ms)))
        with torch.no_grad():
            result_static = net.forward_classify_session(
                sample_spikes,
                probe_spikes,
                delay_duration_steps=delay_steps,
                stsp_mode="static_frozen",
            )
            result_dynamic = net.forward_classify_session(
                sample_spikes,
                probe_spikes,
                delay_duration_steps=delay_steps,
                stsp_mode="dynamic",
            )
        pred_static = result_static["prediction"]
        pred_dynamic = result_dynamic["prediction"]

        num_correct_static = int((pred_static == probe_labels).sum().item())
        num_correct_dynamic = int((pred_dynamic == probe_labels).sum().item())
        num_hallu_static = int((pred_static == sample_labels).sum().item())
        num_hallu_dynamic = int((pred_dynamic == sample_labels).sum().item())

        acc_static = 100.0 * num_correct_static / max(1, total_samples)
        acc_dynamic = 100.0 * num_correct_dynamic / max(1, total_samples)
        ami = 100.0 * (num_hallu_dynamic - num_hallu_static) / max(1, total_samples)
        random_error_dyn = (100.0 - acc_dynamic) - ami

        rows.append(
            {
                "delay_ms": int(delay_ms),
                "acc_static": float(acc_static),
                "acc_dynamic": float(acc_dynamic),
                "acc_drop": float(acc_static - acc_dynamic),
                "ami": float(ami),
                "random_error_dyn": float(random_error_dyn),
            }
        )
    return pd.DataFrame(rows)


def _exp_decay_with_offset(t, amp, tau, offset):
    return amp * np.exp(-t / tau) + offset


def _exp_decay_zero_offset(t, amp, tau):
    return amp * np.exp(-t / tau)


def _fit_curve(delays: np.ndarray, values: np.ndarray, metric_name: str) -> dict[str, object]:
    if metric_name == "acc_drop":
        try:
            params, _ = curve_fit(
                _exp_decay_with_offset,
                delays,
                values,
                p0=[float(np.max(values)), 1500.0, float(np.min(values))],
                maxfev=5000,
            )
            return {
                "metric_name": metric_name,
                "tau_ms": float(params[1]),
                "offset": float(params[2]),
                "fit_success": True,
            }
        except Exception:
            return {"metric_name": metric_name, "tau_ms": None, "offset": None, "fit_success": False}
    try:
        params, _ = curve_fit(
            _exp_decay_zero_offset,
            delays,
            values,
            p0=[float(np.max(values)), 1500.0],
            maxfev=5000,
        )
        return {"metric_name": metric_name, "tau_ms": float(params[1]), "offset": 0.0, "fit_success": True}
    except Exception:
        return {"metric_name": metric_name, "tau_ms": None, "offset": None, "fit_success": False}


def fit_memory_effect_decay(df_metrics: pd.DataFrame) -> pd.DataFrame:
    delays = df_metrics["delay_ms"].to_numpy(dtype=float)
    acc_drop = df_metrics["acc_drop"].to_numpy(dtype=float)
    ami = np.clip(df_metrics["ami"].to_numpy(dtype=float), a_min=1e-6, a_max=None)
    return pd.DataFrame([_fit_curve(delays, acc_drop, "acc_drop"), _fit_curve(delays, ami, "ami")])


def _choose_representative_pair(dataset, class_index: Dict[int, List[int]], sample_label: int, probe_label: int) -> dict[str, int]:
    return {
        "trial_id": 0,
        "sample_label": int(sample_label),
        "probe_label": int(probe_label),
        "sample_index": int(class_index[int(sample_label)][0]),
        "probe_index": int(class_index[int(probe_label)][0]),
    }


def flatten_single_trial_spikes(spikes: torch.Tensor) -> np.ndarray:
    t_steps, batch_size, channels, height, width = spikes.shape
    if batch_size != 1:
        raise ValueError("Expected a single-trial spike tensor with batch size 1.")
    return spikes[:, 0, ...].reshape(t_steps, channels * height * width).to(torch.bool).cpu().numpy()


def phase_for_step(step_index: int, phase_slices: Dict[str, List[int]]) -> str:
    for phase_name in ("sample", "delay", "probe"):
        start, end = phase_slices[phase_name]
        if start <= step_index < end:
            return phase_name
    return "unknown"


def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(float, copy=True)
    pad_left = int(window) // 2
    pad_right = int(window) - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def compute_population_rate_timeseries(
    spikes: torch.Tensor,
    phase_slices: Dict[str, List[int]],
    dt_ms: float,
    smooth_window: int,
    trial_id: int,
) -> pd.DataFrame:
    t_steps, batch_size, channels, height, width = spikes.shape
    n_neurons = int(batch_size * channels * height * width)
    spike_count = spikes.reshape(t_steps, -1).sum(dim=1).detach().cpu().numpy().astype(float)
    rate_raw = spike_count / max(1, n_neurons)
    rate_smooth = moving_average_1d(rate_raw, smooth_window)
    return pd.DataFrame(
        {
            "trial_id": int(trial_id),
            "t_step": np.arange(t_steps, dtype=int),
            "time_ms": np.arange(t_steps, dtype=float) * float(dt_ms),
            "phase": [phase_for_step(step_idx, phase_slices) for step_idx in range(t_steps)],
            "population_rate_raw": rate_raw.astype(float),
            "population_rate_smoothed": rate_smooth.astype(float),
        }
    )


def compute_phase_rate_summary(spikes: torch.Tensor, phase_slices: Dict[str, List[int]], layer_name: str) -> pd.DataFrame:
    t_steps, batch_size, channels, height, width = spikes.shape
    n_neurons = int(batch_size * channels * height * width)
    rows = []
    for phase_name in ("sample", "delay", "probe"):
        start, end = phase_slices[phase_name]
        segment = spikes[start:end]
        duration = int(end - start)
        spike_count = int(segment.sum().item())
        rate = float(spike_count / max(1, n_neurons * max(1, duration)))
        rows.append(
            {
                "layer": layer_name,
                "phase": phase_name,
                "spike_count": spike_count,
                "rate_spikes_per_neuron_step": rate,
            }
        )
    df = pd.DataFrame(rows)
    rates = {str(row["phase"]): float(row["rate_spikes_per_neuron_step"]) for _, row in df.iterrows()}
    eps = 1e-12
    df["ratio_delay_over_sample"] = float(rates["delay"] / max(rates["sample"], eps))
    df["ratio_delay_over_probe"] = float(rates["delay"] / max(rates["probe"], eps))
    return df


def extract_raster_points(layer_tensors: Dict[str, torch.Tensor], phase_slices: Dict[str, List[int]], trial_id: int) -> pd.DataFrame:
    rows = []
    for layer_name, spikes in layer_tensors.items():
        flat = flatten_single_trial_spikes(spikes)
        t_idx, neuron_idx = np.where(flat)
        for step, neuron in zip(t_idx.tolist(), neuron_idx.tolist()):
            rows.append(
                {
                    "trial_id": int(trial_id),
                    "layer": layer_name,
                    "t_step": int(step),
                    "time_ms": float(step),
                    "neuron_index": int(neuron),
                    "phase": phase_for_step(int(step), phase_slices),
                }
            )
    return pd.DataFrame(rows)


def capture_representative_silent_trial(
    net,
    encoder,
    dataset,
    class_index: Dict[int, List[int]],
    *,
    sample_label: int,
    probe_label: int,
    delay_ms: int,
    sample_ms: float,
    probe_ms: float,
    rate_smooth_window: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    trial = _choose_representative_pair(dataset, class_index, sample_label, probe_label)
    sample_image = dataset[int(trial["sample_index"])][0].detach().cpu().to(torch.float32).unsqueeze(0).to(device)
    probe_image = dataset[int(trial["probe_index"])][0].detach().cpu().to(torch.float32).unsqueeze(0).to(device)
    sample_spikes = encode_images(encoder, sample_image, int(round(sample_ms)))
    probe_spikes = encode_images(encoder, probe_image, int(round(probe_ms)))

    with torch.no_grad():
        trace = net.forward_dms_spike_trace_session(
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=int(round(delay_ms)),
            stsp_mode="dynamic",
            phase_reset=True,
        )

    phase_slices = {str(key): [int(value[0]), int(value[1])] for key, value in trace["phase_slices"].items()}
    layer_tensors = {
        "layer1": trace["layer1_spikes"],
        "layer2": trace["layer2_spikes"],
        "layer3": trace["layer3_spikes"],
    }

    raster_df = extract_raster_points(layer_tensors, phase_slices, trial_id=int(trial["trial_id"]))
    rate_df = compute_population_rate_timeseries(
        layer_tensors["layer1"],
        phase_slices=phase_slices,
        dt_ms=float(DT / DT),
        smooth_window=rate_smooth_window,
        trial_id=int(trial["trial_id"]),
    )
    phase_df = pd.concat(
        [
            compute_phase_rate_summary(layer_tensors[layer_name], phase_slices=phase_slices, layer_name=layer_name)
            for layer_name in ("layer1", "layer2", "layer3")
        ],
        ignore_index=True,
    )
    trial_arrays = {
        "trial_id": np.asarray([int(trial["trial_id"])], dtype=np.int64),
        "sample_index": np.asarray([int(trial["sample_index"])], dtype=np.int64),
        "probe_index": np.asarray([int(trial["probe_index"])], dtype=np.int64),
        "sample_label": np.asarray([int(trial["sample_label"])], dtype=np.int64),
        "probe_label": np.asarray([int(trial["probe_label"])], dtype=np.int64),
        "layer1_spikes": trace["layer1_spikes"].numpy().astype(np.uint8, copy=False),
        "layer2_spikes": trace["layer2_spikes"].numpy().astype(np.uint8, copy=False),
        "layer3_spikes": trace["layer3_spikes"].numpy().astype(np.uint8, copy=False),
        "phase_slices_sample": np.asarray(phase_slices["sample"], dtype=np.int64),
        "phase_slices_delay": np.asarray(phase_slices["delay"], dtype=np.int64),
        "phase_slices_probe": np.asarray(phase_slices["probe"], dtype=np.int64),
        "prediction_probe": trace["predictions"]["prediction_probe"].numpy().astype(np.int64, copy=False),
        "first_fire_t_probe": trace["predictions"]["first_fire_t_probe"].numpy().astype(np.int64, copy=False),
    }
    return raster_df, rate_df, phase_df, trial_arrays


def run_delay_state_decode(args, layout, *, delay_points_ms: list[int], train_per_class: int, test_per_class: int, batch_size: int, logger) -> pd.DataFrame:
    stage_dir = layout.staging_path("engram_decode")
    run_python_module(
        "src.experiments.engram_decode",
        [
            "--model-path",
            args.model_path,
            "--dataset-root",
            args.dataset_root,
            "--save-dir",
            str(stage_dir),
            "--device",
            str(args.device),
            "--sample-duration-ms",
            str(float(args.sample_ms)),
            "--delay-points-ms",
            ",".join(str(item) for item in delay_points_ms),
            "--train-per-class",
            str(int(train_per_class)),
            "--test-per-class",
            str(int(test_per_class)),
            "--batch-size",
            str(int(batch_size)),
            "--no-save-diagnostic-plots",
            "--skip-figures",
        ],
        logger=logger,
        cwd=Path.cwd(),
    )
    df = pd.read_csv(stage_dir / "data" / "engram_decode_metrics.csv")
    return df[["layer", "delay_ms", "acc", "acc_ci_low", "acc_ci_high", "macro_f1", "perm_p"]].copy()


def run_delay_end_substrate_shuffle(args, layout, *, trials: int, batch_size: int, num_boot: int, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_dir = layout.staging_path("ux_shuffle_memory_collapse")
    run_python_module(
        "src.experiments.ux_shuffle_memory_collapse",
        [
            "--model-path",
            args.model_path,
            "--save-dir",
            str(stage_dir),
            "--trials",
            str(int(trials)),
            "--batch-size",
            str(int(batch_size)),
            "--seed",
            str(int(args.seed)),
            "--sample-ms",
            str(float(args.sample_ms)),
            "--delay-ms",
            "500.0",
            "--probe-ms",
            str(float(args.probe_ms)),
            "--num-boot",
            str(int(num_boot)),
            "--skip-figures",
        ],
        logger=logger,
        cwd=Path.cwd(),
    )
    data_dir = stage_dir / "data"
    condition_df = pd.read_csv(data_dir / "metrics_condition_summary.csv")
    collapse_df = pd.read_csv(data_dir / "metrics_collapse_summary.csv")
    bootstrap_df = pd.read_csv(data_dir / "metrics_bootstrap_tests.csv")
    collapse_cols = [
        "substrate",
        "collapse_toward_static_improvement_pp",
        "ami_drop_A_minus_B_pp",
        "sample_pred_rate_drop_A_minus_B_pp",
    ]
    return (
        condition_df[["condition", "acc_probe", "abs_rate_pred_original_sample", "abs_rate_pred_change_under_bmap"]].copy(),
        collapse_df[collapse_cols].copy(),
        bootstrap_df.copy(),
    )


def build_fig2_summary(
    df_memory: pd.DataFrame,
    df_fit: pd.DataFrame,
    df_phase: pd.DataFrame,
    df_decode: pd.DataFrame,
    df_shuffle: pd.DataFrame,
) -> dict[str, object]:
    max_drop_row = df_memory.loc[df_memory["acc_drop"].idxmax()]
    acc_drop_fit = df_fit.loc[df_fit["metric_name"] == "acc_drop"].iloc[0].to_dict()
    decode_best = {}
    for layer_name, df_layer in df_decode.groupby("layer"):
        row = df_layer.loc[df_layer["acc"].idxmax()]
        decode_best[str(layer_name)] = {
            "best_delay_ms": int(row["delay_ms"]),
            "best_acc": float(row["acc"]),
        }
    layer1_phase = df_phase.loc[df_phase["layer"] == "layer1"].reset_index(drop=True)
    collapse_row = df_shuffle.loc[df_shuffle["collapse_toward_static_improvement_pp"].idxmax()]
    return {
        "figure": FIGURE_ID,
        "panel_b": {
            "max_acc_drop_delay_ms": int(max_drop_row["delay_ms"]),
            "max_acc_drop_pp": float(max_drop_row["acc_drop"]),
            "accuracy_drop_tau_ms": acc_drop_fit.get("tau_ms"),
            "accuracy_drop_fit_success": bool(acc_drop_fit.get("fit_success")),
        },
        "panel_c": {
            "layer1_delay_over_sample": float(layer1_phase["ratio_delay_over_sample"].iloc[0]),
            "layer1_delay_over_probe": float(layer1_phase["ratio_delay_over_probe"].iloc[0]),
        },
        "panel_d": decode_best,
        "panel_e": {
            "best_collapse_substrate": str(collapse_row["substrate"]),
            "best_collapse_gain_pp": float(collapse_row["collapse_toward_static_improvement_pp"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    config = build_fig2_config(bool(args.smoke))
    seed_everything(int(args.seed))
    device = resolve_device_strict(args.device)
    layout = prepare_layout(args.output_dir)
    logger = setup_logger(layout.log_file(), f"paper_{FIGURE_ID}")

    smoke_command = format_smoke_command(MODULE_NAME, layout.root)
    logger.info("[Init] figure=%s", FIGURE_ID)
    logger.info("[Init] output_dir=%s", layout.root)
    logger.info("[Init] device=%s", device)
    logger.info("[Init] smoke=%s", bool(args.smoke))

    dataset = load_mnist_skeleton_dataset(args.dataset_root, split="test")
    class_index = build_class_index(dataset, num_classes=10)
    spec = DmsSpec(sample_ms=float(args.sample_ms), probe_ms=float(args.probe_ms))
    net, encoder = load_paper_model_and_encoder(
        model_path=args.model_path,
        device=device,
        max_duration_ms=max(float(args.sample_ms), float(args.probe_ms)),
    )

    df_specs = build_dms_pair_bank(dataset, class_index, num_pairs=int(config["num_pairs"]), seed=int(args.seed))
    sample_images, probe_images = _load_images_for_specs(dataset, df_specs, device)
    sample_spikes = encode_images(encoder, sample_images, spec.sample_steps)
    probe_spikes = encode_images(encoder, probe_images, spec.probe_steps)
    sample_labels = torch.as_tensor(df_specs["sample_label"].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
    probe_labels = torch.as_tensor(df_specs["probe_label"].to_numpy(dtype=np.int64), dtype=torch.long, device=device)

    df_memory = compute_memory_effect_vs_delay(
        net,
        sample_spikes=sample_spikes,
        probe_spikes=probe_spikes,
        sample_labels=sample_labels,
        probe_labels=probe_labels,
        delay_points_ms=list(config["delay_points_ms"]),
    )
    df_fit = fit_memory_effect_decay(df_memory)

    raster_df, rate_df, phase_df, trial_arrays = capture_representative_silent_trial(
        net,
        encoder,
        dataset,
        class_index,
        sample_label=int(config["representative_sample_label"]),
        probe_label=int(config["representative_probe_label"]),
        delay_ms=400,
        sample_ms=float(args.sample_ms),
        probe_ms=60.0,
        rate_smooth_window=int(config["rate_smooth_window"]),
        device=device,
    )

    df_decode = run_delay_state_decode(
        args,
        layout,
        delay_points_ms=list(config["decode_delay_points_ms"]),
        train_per_class=int(config["decode_train_per_class"]),
        test_per_class=int(config["decode_test_per_class"]),
        batch_size=int(config["decode_batch_size"]),
        logger=logger,
    )
    df_shuffle_condition, df_shuffle_collapse, df_shuffle_bootstrap = run_delay_end_substrate_shuffle(
        args,
        layout,
        trials=int(config["shuffle_trials"]),
        batch_size=int(config["shuffle_batch_size"]),
        num_boot=int(config["shuffle_num_boot"]),
        logger=logger,
    )

    artifact_paths = {
        "run_config_json": str(
            save_json(
                {
                    "figure": FIGURE_ID,
                    "module_name": MODULE_NAME,
                    "model_path": str(Path(args.model_path).resolve()),
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "device_requested": str(args.device),
                    "device_resolved": str(device),
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "smoke_command": smoke_command,
                    "panel_b_delay_points_ms": list(config["delay_points_ms"]),
                    "panel_b_num_pairs": int(config["num_pairs"]),
                    "panel_d_delay_points_ms": list(config["decode_delay_points_ms"]),
                    "panel_d_train_per_class": int(config["decode_train_per_class"]),
                    "panel_d_test_per_class": int(config["decode_test_per_class"]),
                    "panel_d_batch_size": int(config["decode_batch_size"]),
                    "panel_e_trials": int(config["shuffle_trials"]),
                    "panel_e_batch_size": int(config["shuffle_batch_size"]),
                    "panel_e_num_boot": int(config["shuffle_num_boot"]),
                },
                layout.root_file("run_config.json"),
            )
        ),
        "panel_b_memory_effect_csv": str(save_csv(df_memory, layout.data_file("panel_b_memory_effect_vs_delay.csv"), sort_by=["delay_ms"])),
        "panel_b_fit_summary_csv": str(save_csv(df_fit, layout.data_file("panel_b_fit_summary.csv"), sort_by=["metric_name"])),
        "panel_c_raster_points_csv": str(save_csv(raster_df, layout.data_file("panel_c_raster_points.csv"), sort_by=["layer", "t_step", "neuron_index"])),
        "panel_c_population_rate_csv": str(save_csv(rate_df, layout.data_file("panel_c_population_rate.csv"), sort_by=["t_step"])),
        "panel_c_phase_rate_summary_csv": str(save_csv(phase_df, layout.data_file("panel_c_phase_rate_summary.csv"), sort_by=["layer", "phase"])),
        "panel_c_representative_trial_npz": str(save_npz(layout.array_file("panel_c_representative_trial.npz"), **trial_arrays)),
        "panel_d_decode_metrics_csv": str(save_csv(df_decode, layout.data_file("panel_d_decode_metrics.csv"), sort_by=["layer", "delay_ms"])),
        "panel_e_condition_summary_csv": str(save_csv(df_shuffle_condition, layout.data_file("panel_e_condition_summary.csv"), sort_by=["condition"])),
        "panel_e_collapse_summary_csv": str(save_csv(df_shuffle_collapse, layout.data_file("panel_e_collapse_summary.csv"), sort_by=["substrate"])),
        "panel_e_bootstrap_tests_csv": str(save_csv(df_shuffle_bootstrap, layout.data_file("panel_e_bootstrap_tests.csv"))),
    }

    summary = build_fig2_summary(df_memory, df_fit, phase_df, df_decode, df_shuffle_collapse)
    artifact_paths["summary_json"] = str(save_json({**summary, "saved_artifacts": artifact_paths}, layout.root_file("summary.json")))
    artifact_paths["artifact_manifest_json"] = str(write_artifact_manifest(layout, artifact_paths))

    logger.info("[Done] Fig2 artifacts saved.")


if __name__ == "__main__":
    main()
