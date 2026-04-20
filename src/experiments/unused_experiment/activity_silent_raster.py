import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import ms


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    sample_ms: float
    delay1_ms: float
    distractor_ms: float
    delay2_ms: float
    probe_ms: float
    phase_reset: bool

    @property
    def sample_steps(self) -> int:
        return int((self.sample_ms * ms) / self.dt)

    @property
    def delay1_steps(self) -> int:
        return int((self.delay1_ms * ms) / self.dt)

    @property
    def distractor_steps(self) -> int:
        return int((self.distractor_ms * ms) / self.dt)

    @property
    def delay2_steps(self) -> int:
        return int((self.delay2_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)

    @property
    def total_steps(self) -> int:
        return self.sample_steps + self.delay1_steps + self.distractor_steps + self.delay2_steps + self.probe_steps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor


def load_model_and_encoder(model_path: str, device: torch.device, spec: ExperimentSpec) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    net = SDNN_Network(device=str(device)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    max_duration_ms = max(spec.sample_ms, spec.distractor_ms, spec.probe_ms)
    encoder = DoGSpikeEncoder(dt=spec.dt, max_duration=max_duration_ms * ms, device=str(device))
    return net, encoder


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


def encode_single_image(encoder: DoGSpikeEncoder, image_tensor: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    batch_img = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        spikes = encoder.forward(batch_img)
    return spikes[:, :steps, ...].contiguous()


def flatten_single_trial_spikes(spikes: torch.Tensor) -> np.ndarray:
    # spikes: [T, B, C, H, W]
    if spikes.dim() != 5:
        raise ValueError(f"Expected spikes to be 5D, got {tuple(spikes.shape)}")

    T, B, C, H, W = spikes.shape
    if B != 1:
        raise ValueError("This experiment expects single-trial traces with batch size = 1")

    flat = spikes[:, 0, ...].reshape(T, C * H * W)
    return flat.to(torch.bool).cpu().numpy()


def plot_raster_layer(
    flat_spikes: np.ndarray,
    phase_slices: Dict[str, List[int]],
    title: str,
    save_path: str,
    dt_ms: float,
) -> None:
    # flat_spikes: [T, N]
    T, N = flat_spikes.shape
    t_idx, neuron_idx = np.where(flat_spikes)

    phase_colors = {
        "sample": "#d62728",
        "delay1": "#7f7f7f",
        "distractor": "#9467bd",
        "delay2": "#7f7f7f",
        "probe": "#1f77b4",
    }

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    for phase_name, (start, end) in phase_slices.items():
        color = phase_colors.get(phase_name, "#cccccc")
        alpha = 0.08 if "delay" in phase_name else 0.06
        ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0)
        ax.axvline(start, color=color, linewidth=1.0, alpha=0.6)
    ax.axvline(T, color="black", linewidth=1.0, alpha=0.5)

    if len(t_idx) > 0:
        ax.scatter(t_idx, neuron_idx, s=0.7, c="black", marker=".", linewidths=0, alpha=0.9)

    ax.set_xlim(0, T)
    ax.set_ylim(0, N)
    ax.set_xlabel(f"Time Step (dt={dt_ms:.3f} ms)")
    ax.set_ylabel("Neuron Index (channel x position)")
    ax.set_title(f"{title} | Neurons={N}, Time={T} steps")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_phase_rate_table(spikes: torch.Tensor, phase_slices: Dict[str, List[int]], layer_name: str) -> pd.DataFrame:
    # spikes: [T, B, C, H, W]
    T, B, C, H, W = spikes.shape
    n_neurons = int(B * C * H * W)

    rows = []
    for phase in ["sample", "delay1", "distractor", "delay2", "probe"]:
        start, end = phase_slices[phase]
        if not (0 <= start <= end <= T):
            raise ValueError(f"Invalid phase slice for {phase}: [{start}, {end}) with T={T}")
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
    rates = {r["phase"]: r["rate_spikes_per_neuron_step"] for _, r in df.iterrows()}
    eps = 1e-12
    ratio_delay1_sample = rates["delay1"] / max(rates["sample"], eps)
    ratio_delay2_sample = rates["delay2"] / max(rates["sample"], eps)
    ratio_delay1_distractor = rates["delay1"] / max(rates["distractor"], eps)

    df["ratio_delay1_over_sample"] = ratio_delay1_sample
    df["ratio_delay2_over_sample"] = ratio_delay2_sample
    df["ratio_delay1_over_distractor"] = ratio_delay1_distractor
    return df


def check_activity_silent_warnings(df_rates: pd.DataFrame, threshold: float) -> None:
    by_layer = df_rates.groupby("layer", as_index=False).first()
    for _, row in by_layer.iterrows():
        layer = row["layer"]
        r1 = float(row["ratio_delay1_over_sample"])
        r2 = float(row["ratio_delay2_over_sample"])
        r3 = float(row["ratio_delay1_over_distractor"])

        if r1 > threshold:
            print(f"[Warn] {layer}: delay1/sample ratio={r1:.4f} > threshold={threshold:.4f}")
        if r2 > threshold:
            print(f"[Warn] {layer}: delay2/sample ratio={r2:.4f} > threshold={threshold:.4f}")
        if r3 > threshold:
            print(f"[Warn] {layer}: delay1/distractor ratio={r3:.4f} > threshold={threshold:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-trial activity-silent raster experiment (3 layers)")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/activity_silent_single_trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)

    parser.add_argument("--sample-label", type=int, default=1)
    parser.add_argument("--distractor-label", type=int, default=8)
    parser.add_argument("--probe-label", type=int, default=3)

    parser.add_argument("--sample-ms", type=float, default=200.0)
    parser.add_argument("--delay1-ms", type=float, default=400.0)
    parser.add_argument("--distractor-ms", type=float, default=200.0)
    parser.add_argument("--delay2-ms", type=float, default=400.0)
    parser.add_argument("--probe-ms", type=float, default=100.0)

    parser.add_argument("--stsp-mode", type=str, default="dynamic", choices=["dynamic", "static_frozen"])
    parser.add_argument("--no-phase-reset", action="store_true")
    parser.add_argument("--silent-warn-threshold", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    label_triplet = [args.sample_label, args.distractor_label, args.probe_label]
    if len(set(label_triplet)) != 3:
        raise ValueError("sample-label, distractor-label, probe-label must be pairwise different")
    if min(label_triplet) < 0 or max(label_triplet) >= args.num_classes:
        raise ValueError("Label index out of range")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ExperimentSpec(
        dt=1.0 * ms,
        sample_ms=args.sample_ms,
        delay1_ms=args.delay1_ms,
        distractor_ms=args.distractor_ms,
        delay2_ms=args.delay2_ms,
        probe_ms=args.probe_ms,
        phase_reset=(not args.no_phase_reset),
    )
    for name, steps in [
        ("sample", spec.sample_steps),
        ("delay1", spec.delay1_steps),
        ("distractor", spec.distractor_steps),
        ("delay2", spec.delay2_steps),
        ("probe", spec.probe_steps),
    ]:
        if steps <= 0:
            raise ValueError(f"{name} steps must be positive")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {args.save_dir}")
    print(
        f"[Init] Labels: sample={args.sample_label}, distractor={args.distractor_label}, probe={args.probe_label}"
    )
    print(
        f"[Init] Timing steps: sample={spec.sample_steps}, delay1={spec.delay1_steps}, "
        f"distractor={spec.distractor_steps}, delay2={spec.delay2_steps}, probe={spec.probe_steps}, "
        f"total={spec.total_steps}"
    )

    net, encoder = load_model_and_encoder(args.model_path, device, spec)

    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)
    rng = random.Random(args.seed)

    img_s, lbl_s, idx_s = sample_single_image(dataset, class_index, args.sample_label, rng)
    img_d, lbl_d, idx_d = sample_single_image(dataset, class_index, args.distractor_label, rng)
    img_p, lbl_p, idx_p = sample_single_image(dataset, class_index, args.probe_label, rng)

    sample_spikes = encode_single_image(encoder, img_s, spec.sample_steps, device)
    distractor_spikes = encode_single_image(encoder, img_d, spec.distractor_steps, device)
    probe_spikes = encode_single_image(encoder, img_p, spec.probe_steps, device)

    with torch.no_grad():
        trace = net.forward_dual_task_spike_trace_session(
            sample_spikes=sample_spikes,
            distractor_spikes=distractor_spikes,
            probe_spikes=probe_spikes,
            delay1_steps=spec.delay1_steps,
            delay2_steps=spec.delay2_steps,
            stsp_mode=args.stsp_mode,
            phase_reset=spec.phase_reset,
        )

    phase_slices = trace["phase_slices"]
    layer_tensors = {
        "layer1": trace["layer1_spikes"],
        "layer2": trace["layer2_spikes"],
        "layer3": trace["layer3_spikes"],
    }

    phase_expected = ["sample", "delay1", "distractor", "delay2", "probe"]
    for phase in phase_expected:
        if phase not in phase_slices:
            raise ValueError(f"Missing phase slice: {phase}")

    layer_rate_tables = []
    layer_shapes = {}
    raster_paths = {}
    for layer_name, spikes in layer_tensors.items():
        T, B, C, H, W = spikes.shape
        layer_shapes[layer_name] = [int(T), int(B), int(C), int(H), int(W)]
        if T != spec.total_steps:
            raise ValueError(f"{layer_name} total time mismatch: got {T}, expected {spec.total_steps}")

        flat_np = flatten_single_trial_spikes(spikes)
        raster_path = os.path.join(args.save_dir, f"raster_{layer_name}.png")
        plot_raster_layer(
            flat_spikes=flat_np,
            phase_slices=phase_slices,
            title=f"{layer_name.upper()} Spike Raster",
            save_path=raster_path,
            dt_ms=float(spec.dt / ms),
        )
        raster_paths[layer_name] = raster_path

        df_layer = compute_phase_rate_table(spikes=spikes, phase_slices=phase_slices, layer_name=layer_name)
        layer_rate_tables.append(df_layer)

    df_rates = pd.concat(layer_rate_tables, axis=0, ignore_index=True)
    rates_csv = os.path.join(args.save_dir, "firing_rate_by_phase.csv")
    df_rates.to_csv(rates_csv, index=False)

    check_activity_silent_warnings(df_rates, threshold=float(args.silent_warn_threshold))

    pred = trace["predictions"]
    metadata = {
        "seed": int(args.seed),
        "device": str(device),
        "model_path": args.model_path,
        "stsp_mode": args.stsp_mode,
        "phase_reset": bool(spec.phase_reset),
        "labels": {
            "sample": int(lbl_s),
            "distractor": int(lbl_d),
            "probe": int(lbl_p),
        },
        "dataset_indices": {
            "sample_index": int(idx_s),
            "distractor_index": int(idx_d),
            "probe_index": int(idx_p),
        },
        "timing_ms": {
            "sample": float(spec.sample_ms),
            "delay1": float(spec.delay1_ms),
            "distractor": float(spec.distractor_ms),
            "delay2": float(spec.delay2_ms),
            "probe": float(spec.probe_ms),
        },
        "timing_steps": {
            "sample": int(spec.sample_steps),
            "delay1": int(spec.delay1_steps),
            "distractor": int(spec.distractor_steps),
            "delay2": int(spec.delay2_steps),
            "probe": int(spec.probe_steps),
            "total": int(spec.total_steps),
        },
        "phase_slices": {k: [int(v[0]), int(v[1])] for k, v in phase_slices.items()},
        "predictions": {
            "prediction_distractor": int(pred["prediction_distractor"][0].item()),
            "prediction_probe": int(pred["prediction_probe"][0].item()),
            "first_fire_t_distractor": int(pred["first_fire_t_distractor"][0].item()),
            "first_fire_t_probe": int(pred["first_fire_t_probe"][0].item()),
        },
        "layer_shapes": layer_shapes,
        "outputs": {
            "raster_layer1": raster_paths["layer1"],
            "raster_layer2": raster_paths["layer2"],
            "raster_layer3": raster_paths["layer3"],
            "firing_rate_csv": rates_csv,
        },
    }
    metadata_path = os.path.join(args.save_dir, "trial_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n=== Activity-Silent Single-Trial Summary ===")
    print(f"Saved: {raster_paths['layer1']}")
    print(f"Saved: {raster_paths['layer2']}")
    print(f"Saved: {raster_paths['layer3']}")
    print(f"Saved: {rates_csv}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
