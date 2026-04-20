import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import ms


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay_ms: float
    probe_ms: float
    phase_reset: bool

    @property
    def sample_steps(self) -> int:
        return int((self.sample_ms * ms) / self.dt)

    @property
    def delay_steps(self) -> int:
        return int((self.delay_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)

    @property
    def total_steps(self) -> int:
        return self.sample_steps + self.delay_steps + self.probe_steps


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def initialize_output_dirs(experiment_name: str):
    root_dir = os.path.join("results", experiment_name)
    output_dirs = {
        "root": root_dir,
        "figure": os.path.join(root_dir, "figure"),
        "log": os.path.join(root_dir, "log"),
        "data": os.path.join(root_dir, "data"),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)
    return output_dirs


def setup_logger(log_dir: str, experiment_name: str) -> logging.Logger:
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(data, path: str, logger: logging.Logger = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)
    if logger is not None:
        logger.info("[Save] JSON saved to %s", path)


def save_csv(df: pd.DataFrame, path: str, logger: logging.Logger = None) -> None:
    df.to_csv(path, index=False)
    if logger is not None:
        logger.info("[Save] CSV saved to %s", path)


def save_figure_multi_format(fig, figure_dir: str, basename: str, logger: logging.Logger) -> None:
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(figure_dir, f"{basename}.{ext}")
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(path, **save_kwargs)
        logger.info("[Save] Figure saved to %s", path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float, logger: logging.Logger = None) -> None:
    if logger is not None:
        logger.info("[Init] Executing gain compensation: scale=%.4fx", scaling_factor)

    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    spec: ExperimentSpec,
    logger: logging.Logger,
) -> Tuple[SDNN_Network, DoGSpikeEncoder, float]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    net = SDNN_Network(device=str(device)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    stsp_gain_compensation_factor = 1.0 / float(net.layer3.stsp_U)
    compensate_stsp_gain(net, scaling_factor=stsp_gain_compensation_factor, logger=logger)
    net.eval()

    max_duration_ms = max(spec.sample_ms, spec.probe_ms)
    encoder = DoGSpikeEncoder(dt=spec.dt, max_duration=max_duration_ms * ms, device=str(device))
    return net, encoder, stsp_gain_compensation_factor


def build_class_index(dataset, num_classes: int) -> Dict[int, List[int]]:
    class_index: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_index[int(label)].append(idx)

    for cls in range(num_classes):
        if len(class_index[cls]) == 0:
            raise ValueError(f"Class {cls} has no samples in dataset")
    return class_index


def sample_single_image(dataset, class_index: Dict[int, List[int]], target_class: int, rng: random.Random):
    idx = rng.choice(class_index[target_class])
    img, lbl = dataset[idx]
    return img, int(lbl), int(idx)


def encode_single_image(
    encoder: DoGSpikeEncoder,
    image_tensor: torch.Tensor,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    batch_img = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        spikes = encoder.forward(batch_img)
    return spikes[:, :steps, ...].contiguous()


def flatten_single_trial_spikes(spikes: torch.Tensor) -> np.ndarray:
    if spikes.dim() != 5:
        raise ValueError(f"Expected spikes to be 5D, got {tuple(spikes.shape)}")

    t_steps, batch_size, channels, height, width = spikes.shape
    if batch_size != 1:
        raise ValueError("This experiment expects single-trial traces with batch size = 1")

    flat = spikes[:, 0, ...].reshape(t_steps, channels * height * width)
    return flat.to(torch.bool).cpu().numpy()


def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float, copy=True)

    window = int(max(1, window))
    if values.size == 0:
        return values.astype(float, copy=True)

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def phase_for_step(step_idx: int, phase_slices: Dict[str, List[int]]) -> str:
    for phase in ["sample", "delay", "probe"]:
        start, end = phase_slices[phase]
        if start <= step_idx < end:
            return phase
    return "unknown"


def compute_population_rate_timeseries(
    spikes: torch.Tensor,
    phase_slices: Dict[str, List[int]],
    dt_ms: float,
    smooth_window: int,
) -> pd.DataFrame:
    if spikes.dim() != 5:
        raise ValueError(f"Expected spikes to be 5D, got {tuple(spikes.shape)}")

    t_steps, batch_size, channels, height, width = spikes.shape
    n_neurons = int(batch_size * channels * height * width)
    spike_count = spikes.reshape(t_steps, -1).sum(dim=1).detach().cpu().numpy().astype(float)
    population_rate_raw = spike_count / max(1, n_neurons)
    population_rate_smoothed = moving_average_1d(population_rate_raw, smooth_window)

    if len(population_rate_smoothed) != t_steps:
        raise ValueError("Smoothed firing-rate length mismatch")

    phases = [phase_for_step(step_idx, phase_slices) for step_idx in range(t_steps)]
    time_steps = np.arange(t_steps, dtype=int)
    time_ms = time_steps.astype(float) * float(dt_ms)

    return pd.DataFrame(
        {
            "time_step": time_steps,
            "time_ms": time_ms,
            "spike_count": spike_count.astype(int),
            "population_rate_raw": population_rate_raw.astype(float),
            "population_rate_smoothed": population_rate_smoothed.astype(float),
            "phase": phases,
        }
    )


def compute_phase_rate_table(
    spikes: torch.Tensor,
    phase_slices: Dict[str, List[int]],
    layer_name: str,
) -> pd.DataFrame:
    t_steps, batch_size, channels, height, width = spikes.shape
    n_neurons = int(batch_size * channels * height * width)

    rows = []
    for phase in ["sample", "delay", "probe"]:
        start, end = phase_slices[phase]
        if not (0 <= start <= end <= t_steps):
            raise ValueError(f"Invalid phase slice for {phase}: [{start}, {end}) with T={t_steps}")

        duration = int(end - start)
        seg = spikes[start:end]
        spike_count = int(seg.sum().item())
        denom = max(1, n_neurons * max(1, duration))
        rate = float(spike_count / denom)

        rows.append(
            {
                "layer": layer_name,
                "phase": phase,
                "start_step": int(start),
                "end_step": int(end),
                "duration_steps": duration,
                "n_neurons": n_neurons,
                "spike_count": spike_count,
                "rate_spikes_per_neuron_step": rate,
            }
        )

    df = pd.DataFrame(rows)
    rates = {row["phase"]: row["rate_spikes_per_neuron_step"] for _, row in df.iterrows()}
    eps = 1e-12
    ratio_delay_sample = float(rates["delay"] / max(rates["sample"], eps))
    ratio_delay_probe = float(rates["delay"] / max(rates["probe"], eps))
    df["ratio_delay_over_sample"] = ratio_delay_sample
    df["ratio_delay_over_probe"] = ratio_delay_probe
    return df


def log_activity_silent_status(df_rates: pd.DataFrame, threshold: float, logger: logging.Logger) -> None:
    by_layer = df_rates.groupby("layer", as_index=False).first()
    for _, row in by_layer.iterrows():
        layer = row["layer"]
        ratio_delay_sample = float(row["ratio_delay_over_sample"])
        ratio_delay_probe = float(row["ratio_delay_over_probe"])

        if ratio_delay_sample > threshold:
            logger.info(
                "[Warn] %s delay/sample ratio=%.4f > threshold=%.4f",
                layer,
                ratio_delay_sample,
                threshold,
            )
        if ratio_delay_probe > threshold:
            logger.info(
                "[Warn] %s delay/probe ratio=%.4f > threshold=%.4f",
                layer,
                ratio_delay_probe,
                threshold,
            )


def add_phase_annotations(ax, phase_slices: Dict[str, List[int]], dt_ms: float) -> None:
    phase_colors = {
        "sample": "#d62728",
        "delay": "#7f7f7f",
        "probe": "#1f77b4",
    }

    ymax = ax.get_ylim()[1]
    for phase_name in ["sample", "delay", "probe"]:
        start, end = phase_slices[phase_name]
        start_ms = start * dt_ms
        end_ms = end * dt_ms
        color = phase_colors[phase_name]
        alpha = 0.10 if phase_name == "delay" else 0.07
        ax.axvspan(start_ms, end_ms, color=color, alpha=alpha, linewidth=0, zorder=0)
        ax.axvline(start_ms, color=color, linewidth=1.2, alpha=0.75, zorder=1)
        center_ms = 0.5 * (start_ms + end_ms)
        ax.text(
            center_ms,
            ymax,
            phase_name.capitalize(),
            color=color,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.axvline(phase_slices["probe"][1] * dt_ms, color="black", linewidth=1.0, alpha=0.5, zorder=1)


def plot_layer1_raster_rate(
    layer1_flat_spikes: np.ndarray,
    rate_df: pd.DataFrame,
    phase_slices: Dict[str, List[int]],
    dt_ms: float,
    figure_dir: str,
    logger: logging.Logger,
) -> None:
    t_steps, n_neurons = layer1_flat_spikes.shape
    time_ms = rate_df["time_ms"].to_numpy(dtype=float)
    spike_t_idx, neuron_idx = np.where(layer1_flat_spikes)
    spike_t_ms = spike_t_idx.astype(float) * dt_ms

    fig, ax_raster = plt.subplots(figsize=(15, 7))
    ax_rate = ax_raster.twinx()

    if len(spike_t_idx) > 0:
        ax_raster.scatter(
            spike_t_ms,
            neuron_idx,
            s=8.0,
            c="black",
            marker=".",
            linewidths=0,
            alpha=0.95,
            zorder=3,
        )

    ax_raster.set_xlim(0.0, float(t_steps * dt_ms))
    ax_raster.set_ylim(-1, n_neurons)
    ax_raster.set_xlabel(f"Time (ms, dt={dt_ms:.3f} ms)")
    ax_raster.set_ylabel("Neuron Index")
    ax_raster.set_title("Layer1 Spike Raster with Population Firing Rate", fontsize=14, fontweight="bold")

    line_rate, = ax_rate.plot(
        time_ms,
        rate_df["population_rate_smoothed"].to_numpy(dtype=float),
        color="#d62728",
        linewidth=2.4,
        label="Layer1 population rate (smoothed)",
        zorder=4,
    )
    line_rate_raw, = ax_rate.plot(
        time_ms,
        rate_df["population_rate_raw"].to_numpy(dtype=float),
        color="#ff9896",
        linewidth=1.0,
        alpha=0.45,
        label="Layer1 population rate (raw)",
        zorder=2,
    )
    ax_rate.set_ylabel("Population Firing Rate (spikes / neuron / step)")
    ax_rate.set_xlim(ax_raster.get_xlim())
    ax_rate.set_ylim(bottom=0.0)

    add_phase_annotations(ax_raster, phase_slices, dt_ms)
    ax_rate.patch.set_alpha(0.0)

    handles = [line_rate, line_rate_raw]
    labels = [h.get_label() for h in handles]
    ax_rate.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    save_figure_multi_format(fig, figure_dir, "layer1_raster_rate", logger)
    plt.close(fig)


def extract_phase_peak_rates(rate_df: pd.DataFrame) -> Dict[str, float]:
    peaks = {}
    for phase in ["sample", "delay", "probe"]:
        phase_values = rate_df.loc[rate_df["phase"] == phase, "population_rate_smoothed"].to_numpy(dtype=float)
        peaks[phase] = float(phase_values.max()) if phase_values.size > 0 else 0.0
    return peaks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-trial DMS activity-silent raster experiment focused on layer1")
    parser.add_argument("--model-path", type=str, default=os.path.join("results", "sdnn_deep_final", "net_final.pth"))
    parser.add_argument("--experiment-name", type=str, default="activity_silent_single_trial_dms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)

    parser.add_argument("--sample-label", type=int, default=1)
    parser.add_argument("--probe-label", type=int, default=3)

    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=400.0)
    parser.add_argument("--probe-ms", type=float, default=60.0)

    parser.add_argument("--stsp-mode", type=str, default="dynamic", choices=["dynamic", "static_frozen"])
    parser.add_argument("--no-phase-reset", action="store_true")
    parser.add_argument("--silent-warn-threshold", type=float, default=0.2)
    parser.add_argument("--rate-smooth-window", type=int, default=15)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    label_pair = [args.sample_label, args.probe_label]
    if min(label_pair) < 0 or max(label_pair) >= args.num_classes:
        raise ValueError("Label index out of range")
    if args.rate_smooth_window <= 0:
        raise ValueError("rate-smooth-window must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=args.sample_ms,
        delay_ms=args.delay_ms,
        probe_ms=args.probe_ms,
        phase_reset=(not args.no_phase_reset),
    )
    for name, steps in [
        ("sample", spec.sample_steps),
        ("delay", spec.delay_steps),
        ("probe", spec.probe_steps),
    ]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    output_dirs = initialize_output_dirs(args.experiment_name)
    logger = setup_logger(output_dirs["log"], args.experiment_name)

    logger.info("[Init] Run started at %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("[Init] Save dir: %s", output_dirs["root"])
    logger.info("[Init] Device: %s", device)
    logger.info("[Init] Model path: %s", args.model_path)
    logger.info("[Init] Labels: sample=%d, probe=%d", args.sample_label, args.probe_label)
    logger.info(
        "[Init] Timing: sample=%.1f ms (%d steps), delay=%.1f ms (%d steps), probe=%.1f ms (%d steps), total=%d steps",
        spec.sample_ms,
        spec.sample_steps,
        spec.delay_ms,
        spec.delay_steps,
        spec.probe_ms,
        spec.probe_steps,
        spec.total_steps,
    )

    net, encoder, stsp_gain_compensation_factor = load_model_and_encoder(args.model_path, device, spec, logger)

    run_config = {
        "experiment_name": args.experiment_name,
        "model_path": args.model_path,
        "seed": int(args.seed),
        "device": str(device),
        "num_classes": int(args.num_classes),
        "sample_label": int(args.sample_label),
        "probe_label": int(args.probe_label),
        "sample_ms": float(spec.sample_ms),
        "delay_ms": float(spec.delay_ms),
        "probe_ms": float(spec.probe_ms),
        "dt_ms": float(spec.dt / ms),
        "stsp_mode": args.stsp_mode,
        "phase_reset": bool(spec.phase_reset),
        "silent_warn_threshold": float(args.silent_warn_threshold),
        "rate_smooth_window": int(args.rate_smooth_window),
        "stsp_gain_compensation_factor": float(stsp_gain_compensation_factor),
    }
    save_json(run_config, os.path.join(output_dirs["root"], "run_config.json"), logger)

    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)
    rng = random.Random(args.seed)

    img_s, lbl_s, idx_s = sample_single_image(dataset, class_index, args.sample_label, rng)
    img_p, lbl_p, idx_p = sample_single_image(dataset, class_index, args.probe_label, rng)

    sample_spikes = encode_single_image(encoder, img_s, spec.sample_steps, device)
    probe_spikes = encode_single_image(encoder, img_p, spec.probe_steps, device)

    with torch.no_grad():
        trace = net.forward_dms_spike_trace_session(
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=spec.delay_steps,
            stsp_mode=args.stsp_mode,
            phase_reset=spec.phase_reset,
        )

    phase_slices = trace["phase_slices"]
    for phase in ["sample", "delay", "probe"]:
        if phase not in phase_slices:
            raise ValueError(f"Missing phase slice: {phase}")

    logger.info("[Trace] Phase slices: %s", {k: [int(v[0]), int(v[1])] for k, v in phase_slices.items()})

    layer_tensors = {
        "layer1": trace["layer1_spikes"],
        "layer2": trace["layer2_spikes"],
        "layer3": trace["layer3_spikes"],
    }

    layer_shapes = {}
    layer_rate_tables = []
    for layer_name, spikes in layer_tensors.items():
        t_steps, batch_size, channels, height, width = spikes.shape
        layer_shapes[layer_name] = [int(t_steps), int(batch_size), int(channels), int(height), int(width)]
        if t_steps != spec.total_steps:
            raise ValueError(f"{layer_name} total time mismatch: got {t_steps}, expected {spec.total_steps}")

        df_layer = compute_phase_rate_table(spikes=spikes, phase_slices=phase_slices, layer_name=layer_name)
        layer_rate_tables.append(df_layer)

    full_rate_df = pd.concat(layer_rate_tables, axis=0, ignore_index=True)
    layer1_rate_by_phase_df = full_rate_df.loc[full_rate_df["layer"] == "layer1"].reset_index(drop=True)

    layer1_timeseries_df = compute_population_rate_timeseries(
        spikes=layer_tensors["layer1"],
        phase_slices=phase_slices,
        dt_ms=float(spec.dt / ms),
        smooth_window=int(args.rate_smooth_window),
    )

    layer1_flat_spikes = flatten_single_trial_spikes(layer_tensors["layer1"])
    plot_layer1_raster_rate(
        layer1_flat_spikes=layer1_flat_spikes,
        rate_df=layer1_timeseries_df,
        phase_slices=phase_slices,
        dt_ms=float(spec.dt / ms),
        figure_dir=output_dirs["figure"],
        logger=logger,
    )

    layer1_ts_path = os.path.join(output_dirs["data"], "layer1_firing_rate_timeseries.csv")
    layer1_phase_path = os.path.join(output_dirs["data"], "layer1_firing_rate_by_phase.csv")
    full_phase_path = os.path.join(output_dirs["data"], "full_firing_rate_by_phase.csv")

    save_csv(layer1_timeseries_df, layer1_ts_path, logger)
    save_csv(layer1_rate_by_phase_df, layer1_phase_path, logger)
    save_csv(full_rate_df, full_phase_path, logger)

    log_activity_silent_status(full_rate_df, threshold=float(args.silent_warn_threshold), logger=logger)

    pred = trace["predictions"]
    prediction_probe = int(pred["prediction_probe"][0].item())
    first_fire_t_probe = int(pred["first_fire_t_probe"][0].item())

    metadata = {
        "seed": int(args.seed),
        "device": str(device),
        "model_path": args.model_path,
        "experiment_name": args.experiment_name,
        "stsp_mode": args.stsp_mode,
        "phase_reset": bool(spec.phase_reset),
        "trial_type": "match" if int(lbl_s) == int(lbl_p) else "mismatch",
        "labels": {
            "sample": int(lbl_s),
            "probe": int(lbl_p),
        },
        "dataset_indices": {
            "sample_index": int(idx_s),
            "probe_index": int(idx_p),
        },
        "timing_ms": {
            "sample": float(spec.sample_ms),
            "delay": float(spec.delay_ms),
            "probe": float(spec.probe_ms),
        },
        "timing_steps": {
            "sample": int(spec.sample_steps),
            "delay": int(spec.delay_steps),
            "probe": int(spec.probe_steps),
            "total": int(spec.total_steps),
        },
        "phase_slices": {k: [int(v[0]), int(v[1])] for k, v in phase_slices.items()},
        "predictions": {
            "prediction_probe": prediction_probe,
            "first_fire_t_probe": first_fire_t_probe,
        },
        "layer_shapes": layer_shapes,
        "outputs": {
            "layer1_raster_rate_png": os.path.join(output_dirs["figure"], "layer1_raster_rate.png"),
            "layer1_raster_rate_pdf": os.path.join(output_dirs["figure"], "layer1_raster_rate.pdf"),
            "layer1_raster_rate_svg": os.path.join(output_dirs["figure"], "layer1_raster_rate.svg"),
            "layer1_firing_rate_timeseries_csv": layer1_ts_path,
            "layer1_firing_rate_by_phase_csv": layer1_phase_path,
            "full_firing_rate_by_phase_csv": full_phase_path,
        },
    }
    metadata_path = os.path.join(output_dirs["data"], "trial_metadata.json")
    save_json(metadata, metadata_path, logger)

    layer1_rates = {
        row["phase"]: float(row["rate_spikes_per_neuron_step"])
        for _, row in layer1_rate_by_phase_df.iterrows()
    }
    layer1_delay_over_sample = float(layer1_rate_by_phase_df["ratio_delay_over_sample"].iloc[0])
    layer1_delay_over_probe = float(layer1_rate_by_phase_df["ratio_delay_over_probe"].iloc[0])
    peak_rates = extract_phase_peak_rates(layer1_timeseries_df)

    layer1_is_silent_vs_sample = bool(layer1_delay_over_sample <= float(args.silent_warn_threshold))
    layer1_is_silent_vs_probe = bool(layer1_delay_over_probe <= float(args.silent_warn_threshold))

    if layer1_is_silent_vs_sample and layer1_is_silent_vs_probe:
        summary_text = (
            "Layer1 firing activity is strongly reduced during the delay period compared with sample and probe, "
            "supporting an activity-silent interpretation."
        )
    else:
        summary_text = (
            "Layer1 firing activity is reduced during the delay period, but the reduction does not meet both "
            "activity-silent thresholds."
        )

    summary = {
        "experiment_name": args.experiment_name,
        "trial_type": metadata["trial_type"],
        "sample_label": int(lbl_s),
        "probe_label": int(lbl_p),
        "stsp_mode": args.stsp_mode,
        "phase_reset": bool(spec.phase_reset),
        "layer1_sample_rate": layer1_rates["sample"],
        "layer1_delay_rate": layer1_rates["delay"],
        "layer1_probe_rate": layer1_rates["probe"],
        "layer1_ratio_delay_over_sample": layer1_delay_over_sample,
        "layer1_ratio_delay_over_probe": layer1_delay_over_probe,
        "layer1_peak_rate_sample_phase": peak_rates["sample"],
        "layer1_peak_rate_delay_phase": peak_rates["delay"],
        "layer1_peak_rate_probe_phase": peak_rates["probe"],
        "silent_warn_threshold": float(args.silent_warn_threshold),
        "layer1_is_silent_vs_sample": layer1_is_silent_vs_sample,
        "layer1_is_silent_vs_probe": layer1_is_silent_vs_probe,
        "summary_text": summary_text,
    }
    save_json(summary, os.path.join(output_dirs["root"], "summary.json"), logger)

    logger.info("[Summary] layer1 sample rate=%.6f", layer1_rates["sample"])
    logger.info("[Summary] layer1 delay rate=%.6f", layer1_rates["delay"])
    logger.info("[Summary] layer1 probe rate=%.6f", layer1_rates["probe"])
    logger.info("[Summary] layer1 delay/sample ratio=%.6f", layer1_delay_over_sample)
    logger.info("[Summary] layer1 delay/probe ratio=%.6f", layer1_delay_over_probe)
    logger.info("[Summary] prediction_probe=%d first_fire_t_probe=%d", prediction_probe, first_fire_t_probe)
    logger.info("[Done] Figure directory: %s", output_dirs["figure"])
    logger.info("[Done] Data directory: %s", output_dirs["data"])
    logger.info("[Done] Log file: %s", os.path.join(output_dirs["log"], "run.log"))


if __name__ == "__main__":
    main()
