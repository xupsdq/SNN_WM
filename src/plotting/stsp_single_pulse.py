from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.units import ms
from src.config.paths import DEFAULT_PATH_CONFIG
from src.core.network import SDNN_Network
from src.experiments.common.dataset import build_class_index, encode_images
from src.experiments.common.model_io import load_model_and_encoder
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader
from src.plotting.common.io import PUBLICATION_ANNOTATION_FONT_SIZE, save_figure_all_formats
from src.plotting.common.style import DEFAULT_SUBPLOT_ADJUST, DYNAMIC_COLOR, STATIC_COLOR, apply_paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sample-driven STSP state trajectories for layer-3 presynaptic inputs."
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_PATH_CONFIG.model_path))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay-ms", type=float, default=800.0)
    parser.add_argument("--sample-label", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--top-k-units", type=int, default=25)
    parser.add_argument("--save-dir", type=str, default="results/stsp_single_pulse")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_layer(net: SDNN_Network, layer_index: int):
    return getattr(net, f"layer{layer_index}")


def _recover_tau_ms(decay: float, dt_seconds: float) -> float:
    if decay <= 0.0:
        return float("inf")
    if math.isclose(decay, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return float("inf")
    return (-dt_seconds / math.log(decay)) / ms


def load_layer_stsp_params(net: SDNN_Network, layer_index: int) -> Dict[str, float]:
    layer = _get_layer(net, layer_index)
    dt_seconds = float(layer.dt)
    return {
        "U": float(layer.stsp_U),
        "dt_ms": dt_seconds / ms,
        "tau_d_ms": _recover_tau_ms(float(layer.stsp_decay_x), dt_seconds),
        "tau_f_ms": _recover_tau_ms(float(layer.stsp_decay_u), dt_seconds),
    }


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not Path(args.dataset_root).exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if int(args.layer) != 3:
        raise ValueError("This script only supports --layer 3 because the plotted presynaptic input is pool2 -> layer3.")
    if float(args.sample_ms) <= 0.0:
        raise ValueError("--sample-ms must be positive.")
    if float(args.delay_ms) < 0.0:
        raise ValueError("--delay-ms must be non-negative.")
    if int(args.num_trials) <= 0:
        raise ValueError("--num-trials must be positive.")
    if int(args.top_k_units) <= 0:
        raise ValueError("--top-k-units must be positive.")
    if args.sample_index is not None and int(args.num_trials) != 1:
        raise ValueError("--sample-index is only supported when --num-trials=1.")


def ms_to_steps(duration_ms: float, dt_ms: float, *, allow_zero: bool = False) -> int:
    steps = int(round(float(duration_ms) / float(dt_ms)))
    if allow_zero and steps == 0:
        return 0
    if steps <= 0:
        raise ValueError(f"Duration {duration_ms} ms maps to {steps} steps with dt={dt_ms} ms.")
    return steps


def load_runtime(args: argparse.Namespace) -> Tuple[SDNN_Network, object, object, torch.device, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=1.0 * ms,
        max_duration_ms=float(args.sample_ms),
    )
    _, _, test_loader = build_mnist_skeleton_loader(root=args.dataset_root, batch_size=1)
    dataset = test_loader.dataset
    params = load_layer_stsp_params(net, int(args.layer))
    return net, encoder, dataset, device, params


def select_sample_batch(
    dataset,
    class_index: Dict[int, List[int]],
    sample_label: int,
    sample_index: int | None,
    num_trials: int,
    rng: random.Random,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample_index out of range: {sample_index}")
        image, label = dataset[int(sample_index)]
        images = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        return images, [int(sample_index)], [int(label)]

    if int(sample_label) not in class_index:
        raise ValueError(f"sample_label not present in dataset index: {sample_label}")

    candidates = list(class_index[int(sample_label)])
    if len(candidates) == 0:
        raise ValueError(f"No samples available for sample_label={sample_label}")

    if num_trials <= len(candidates):
        selected_indices = rng.sample(candidates, k=num_trials)
    else:
        selected_indices = [rng.choice(candidates) for _ in range(num_trials)]

    images: List[torch.Tensor] = []
    labels: List[int] = []
    for idx in selected_indices:
        image, label = dataset[int(idx)]
        images.append(image)
        labels.append(int(label))
    batch = torch.stack(images, dim=0).to(device=device, dtype=torch.float32)
    return batch, [int(idx) for idx in selected_indices], labels


def run_sample_delay_rollout(
    net: SDNN_Network,
    sample_spikes: torch.Tensor,
    delay_steps: int,
) -> Dict[str, object]:
    batch_size, _, channels, height, width = sample_spikes.shape
    dummy_test = torch.zeros(
        (batch_size, 0, channels, height, width),
        dtype=sample_spikes.dtype,
        device=sample_spikes.device,
    )

    with torch.no_grad():
        res = net.forward_dms_session(
            sample_spikes,
            dummy_test,
            delay_duration_steps=int(delay_steps),
            stsp_mode="dynamic",
        )

    total_steps = int(sample_spikes.shape[1]) + int(delay_steps)
    u = require_trace_tensor("u", res["u"])[:total_steps]
    x = require_trace_tensor("x", res["x"])[:total_steps]
    spikes = require_trace_tensor("spikes", res["spikes"])[:total_steps]

    gain_tensor = res.get("gain")
    if gain_tensor is not None:
        gain = require_trace_tensor("gain", gain_tensor)[:total_steps]
        gain_source = "stsp_gain"
    else:
        gain = u * x
        gain_source = "u*x fallback"

    return {
        "u": u,
        "x": x,
        "gain": gain,
        "spikes": spikes,
        "gain_source": gain_source,
    }


def require_trace_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 5:
        raise ValueError(f"{name} must be 5D [T, B, C, H, W], got shape {tuple(tensor.shape)}")
    return tensor.detach().to(torch.float32).cpu()


def flatten_trace(tensor: torch.Tensor) -> torch.Tensor:
    tensor = require_trace_tensor("trace", tensor)
    time_steps, batch_size = tensor.shape[:2]
    return tensor.reshape(time_steps, batch_size, -1)


def select_active_presynaptic_subset(
    spikes: torch.Tensor,
    sample_steps: int,
    top_k_units: int,
) -> torch.Tensor:
    flat_spikes = flatten_trace(spikes)
    sample_counts = flat_spikes[:sample_steps].sum(dim=(0, 1))
    active_indices = torch.nonzero(sample_counts > 0, as_tuple=False).squeeze(1)
    if active_indices.numel() == 0:
        raise ValueError("No active presynaptic units were found during the sample phase.")

    active_counts = sample_counts[active_indices]
    order = torch.argsort(active_counts, descending=True)
    selected = active_indices[order]
    if selected.numel() > int(top_k_units):
        selected = selected[: int(top_k_units)]
    return selected.to(torch.long)


def compute_subset_state_trajectories(
    u: torch.Tensor,
    x: torch.Tensor,
    gain: torch.Tensor,
    subset_indices: torch.Tensor,
) -> Dict[str, np.ndarray]:
    flat_u = flatten_trace(u)
    flat_x = flatten_trace(x)
    flat_gain = flatten_trace(gain)

    subset_u = flat_u[:, :, subset_indices]
    subset_x = flat_x[:, :, subset_indices]
    subset_gain = flat_gain[:, :, subset_indices]

    return {
        "u": subset_u.mean(dim=2).mean(dim=1).numpy(),
        "x": subset_x.mean(dim=2).mean(dim=1).numpy(),
        "gain": subset_gain.mean(dim=2).mean(dim=1).numpy(),
    }


def build_presynaptic_raster_events(
    spikes: torch.Tensor,
    subset_indices: torch.Tensor,
    dt_ms: float,
    exemplar_trial: int = 0,
) -> List[np.ndarray]:
    flat_spikes = flatten_trace(spikes).to(torch.bool)
    if exemplar_trial < 0 or exemplar_trial >= flat_spikes.shape[1]:
        raise IndexError(f"exemplar_trial out of range: {exemplar_trial}")

    exemplar = flat_spikes[:, exemplar_trial, subset_indices].numpy()
    events: List[np.ndarray] = []
    for unit_idx in range(exemplar.shape[1]):
        event_steps = np.flatnonzero(exemplar[:, unit_idx])
        events.append(event_steps.astype(np.float64) * float(dt_ms))
    return events


def build_figure(
    raster_events: List[np.ndarray],
    trajectories: Dict[str, np.ndarray],
    params: Dict[str, float],
    sample_duration_ms: float,
    total_duration_ms: float,
    gain_source: str,
) -> plt.Figure:
    apply_paper_style()

    fig, ax_raster = plt.subplots(1, 1, figsize=(10.6, 4.2))
    fig.subplots_adjust(**DEFAULT_SUBPLOT_ADJUST)
    ax_state = ax_raster.twinx()

    ax_raster.axvspan(0.0, sample_duration_ms, color="0.92", linewidth=0, zorder=0)
    ax_raster.axvline(sample_duration_ms, color="0.30", linestyle="--", linewidth=1.0, alpha=0.85)
    ax_state.axvline(sample_duration_ms, color="0.30", linestyle="--", linewidth=1.0, alpha=0.85)
    ax_raster.set_xlim(0.0, total_duration_ms)

    line_offsets = np.arange(len(raster_events), dtype=np.float64)
    if len(raster_events) > 0:
        for unit_idx, event_times in enumerate(raster_events):
            if event_times.size == 0:
                continue
            y = np.full(event_times.shape, line_offsets[unit_idx], dtype=np.float64)
            ax_raster.scatter(event_times, y, s=18.0, c="black", marker="o", linewidths=0, zorder=3)
    if len(raster_events) <= 1:
        ax_raster.set_ylim(-0.75, 0.75)
    else:
        ax_raster.set_ylim(-0.75, len(raster_events) - 0.25)
    if len(raster_events) <= 12:
        ax_raster.set_yticks(line_offsets)
    else:
        tick_idx = np.unique(np.linspace(0, len(raster_events) - 1, num=6, dtype=int))
        ax_raster.set_yticks(tick_idx.astype(np.float64))
    ax_raster.set_ylabel("Selected presynaptic units")
    ax_raster.set_xlabel("Time (ms)")

    time_ms = np.arange(trajectories["u"].shape[0], dtype=np.float64) * float(params["dt_ms"])
    gain_label = "effective gain" if gain_source == "stsp_gain" else "effective gain (u*x fallback)"
    line_u, = ax_state.plot(time_ms, trajectories["u"], color=DYNAMIC_COLOR, alpha=0.35, linewidth=1.5, label="u")
    line_x, = ax_state.plot(time_ms, trajectories["x"], color=STATIC_COLOR, alpha=0.35, linewidth=1.5, label="x")
    line_gain, = ax_state.plot(time_ms, trajectories["gain"], color="#C62828", linewidth=3.0, label=gain_label)
    ax_state.set_ylabel("State / gain")
    ax_state.yaxis.set_label_position("right")
    ax_state.yaxis.tick_right()
    ax_state.spines["right"].set_visible(True)
    ax_state.spines["right"].set_linewidth(0.8)
    ymax = max(
        1.02,
        float(np.max(trajectories["u"])) * 1.08,
        float(np.max(trajectories["x"])) * 1.08,
        float(np.max(trajectories["gain"])) * 1.08,
    )
    ax_state.set_ylim(0.0, ymax)
    ax_state.legend(handles=[line_u, line_x, line_gain], loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0.0)
    return fig


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(int(args.seed))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    net, encoder, dataset, device, params = load_runtime(args)
    dt_ms = float(params["dt_ms"])
    sample_steps = ms_to_steps(float(args.sample_ms), dt_ms)
    delay_steps = ms_to_steps(float(args.delay_ms), dt_ms, allow_zero=True)

    class_index = build_class_index(dataset, num_classes=10)
    rng = random.Random(int(args.seed))
    sample_images, sample_indices, sample_labels = select_sample_batch(
        dataset=dataset,
        class_index=class_index,
        sample_label=int(args.sample_label),
        sample_index=args.sample_index,
        num_trials=int(args.num_trials),
        rng=rng,
        device=device,
    )

    sample_spikes = encode_images(encoder, sample_images, sample_steps)
    rollout = run_sample_delay_rollout(net=net, sample_spikes=sample_spikes, delay_steps=delay_steps)
    subset_indices = select_active_presynaptic_subset(
        spikes=rollout["spikes"],
        sample_steps=sample_steps,
        top_k_units=int(args.top_k_units),
    )
    trajectories = compute_subset_state_trajectories(
        u=rollout["u"],
        x=rollout["x"],
        gain=rollout["gain"],
        subset_indices=subset_indices,
    )
    raster_events = build_presynaptic_raster_events(
        spikes=rollout["spikes"],
        subset_indices=subset_indices,
        dt_ms=dt_ms,
        exemplar_trial=0,
    )

    actual_sample_ms = sample_steps * dt_ms
    total_duration_ms = (sample_steps + delay_steps) * dt_ms
    fig = build_figure(
        raster_events=raster_events,
        trajectories=trajectories,
        params=params,
        sample_duration_ms=actual_sample_ms,
        total_duration_ms=total_duration_ms,
        gain_source=str(rollout["gain_source"]),
    )
    figure_paths = save_figure_all_formats(fig, save_dir / "figure_main")
    plt.close(fig)

    print("[Done] Saved figure files:")
    for ext in ("png", "pdf", "svg"):
        print(f"  {ext}: {figure_paths[ext]}")
    print(f"[Info] Device: {device}")
    print(f"[Info] Sample indices: {sample_indices}")
    print(f"[Info] Sample labels: {sample_labels}")
    print(f"[Info] Selected presynaptic units: {int(subset_indices.numel())}")
    print(f"[Info] Gain source: {rollout['gain_source']}")


if __name__ == "__main__":
    main()
