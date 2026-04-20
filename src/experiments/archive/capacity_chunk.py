"""Supplementary/exploratory script.

This file is no longer part of the main-text figure pipeline.
Use the plot_fig*.py scripts plus figure_utils_common.py for the main figure path.
"""
import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import ms


@dataclass(frozen=True)
class ExperimentSpec:
    dt: float
    chunk_ms: float
    probe_ms: float
    m0_repeats: int
    gap_ms: float
    interference_burst_ms: float
    fixed_m0_to_probe_ms: float
    n_values: Tuple[int, ...]

    @property
    def chunk_steps(self) -> int:
        return int((self.chunk_ms * ms) / self.dt)

    @property
    def probe_steps(self) -> int:
        return int((self.probe_ms * ms) / self.dt)

    @property
    def gap_steps(self) -> int:
        return int((self.gap_ms * ms) / self.dt)

    @property
    def interference_burst_steps(self) -> int:
        return int((self.interference_burst_ms * ms) / self.dt)

    @property
    def unit_steps(self) -> int:
        return self.gap_steps + self.interference_burst_steps

    @property
    def fixed_m0_to_probe_steps(self) -> int:
        return int((self.fixed_m0_to_probe_ms * ms) / self.dt)

    def post_delay_steps(self, n_interfere: int) -> int:
        return self.fixed_m0_to_probe_steps - n_interfere * self.unit_steps


def parse_n_values(value: str) -> Tuple[int, ...]:
    n_values = [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    if len(n_values) == 0:
        raise ValueError("n_values cannot be empty")
    if any(n < 0 for n in n_values):
        raise ValueError("n_values must be non-negative")
    return tuple(sorted(set(n_values)))


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

    encoder = DoGSpikeEncoder(
        dt=spec.dt,
        max_duration=spec.chunk_ms * ms,
        device=str(device),
    )
    return net, encoder


def build_class_index(dataset, num_classes: int) -> Dict[int, List[int]]:
    class_index: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_index[int(label)].append(idx)

    for cls in range(num_classes):
        if len(class_index[cls]) == 0:
            raise ValueError(f"Class {cls} has no samples in dataset")
    return class_index


def sample_trials(
    dataset,
    class_index: Dict[int, List[int]],
    num_trials: int,
    num_classes: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs_m0: List[torch.Tensor] = []
    lbls_m0: List[int] = []
    imgs_m1: List[torch.Tensor] = []
    lbls_m1: List[int] = []

    for _ in range(num_trials):
        lbl_m0 = rng.randint(0, num_classes - 1)
        idx_m0 = rng.choice(class_index[lbl_m0])
        img_m0, _ = dataset[idx_m0]

        candidates = [c for c in range(num_classes) if c != lbl_m0]
        lbl_m1 = rng.choice(candidates)
        idx_m1 = rng.choice(class_index[lbl_m1])
        img_m1, _ = dataset[idx_m1]

        imgs_m0.append(img_m0)
        lbls_m0.append(lbl_m0)
        imgs_m1.append(img_m1)
        lbls_m1.append(lbl_m1)

    return (
        torch.stack(imgs_m0, dim=0),
        torch.tensor(lbls_m0, dtype=torch.long),
        torch.stack(imgs_m1, dim=0),
        torch.tensor(lbls_m1, dtype=torch.long),
    )


def encode_trial_batch(
    encoder: DoGSpikeEncoder,
    imgs_m0: torch.Tensor,
    imgs_m1: torch.Tensor,
    probe_steps: int,
    interference_burst_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs_m0 = imgs_m0.to(device)
    imgs_m1 = imgs_m1.to(device)

    with torch.no_grad():
        m0_full = encoder.forward(imgs_m0)
        m1_full = encoder.forward(imgs_m1)

    gamma_steps = int(encoder.gamma_cycle_steps)
    active_gamma_indices = [int(i) for i in encoder.active_gamma_indices]

    if len(active_gamma_indices) != 3:
        raise ValueError("This experiment expects exactly 3 active gamma indices.")

    def build_block(source: torch.Tensor, blank_gamma_per_pulse: int) -> torch.Tensor:
        bsz, _, c, h, w = source.shape
        zero_gamma = torch.zeros((bsz, gamma_steps, c, h, w), device=source.device, dtype=source.dtype)
        segs: List[torch.Tensor] = []
        for idx in active_gamma_indices:
            s = idx * gamma_steps
            e = s + gamma_steps
            if e > source.shape[1]:
                raise ValueError("Encoded chunk is shorter than required active gamma windows.")
            segs.append(source[:, s:e, ...])
            for _ in range(blank_gamma_per_pulse):
                segs.append(zero_gamma)
        return torch.cat(segs, dim=1).contiguous()

    # M0 write pattern: (1 input gamma + 2 blank gamma) x 3 = 180 ms
    m0_write = build_block(m0_full, blank_gamma_per_pulse=2)
    # Interference pattern: (1 input gamma + 1 blank gamma) x 3 = 120 ms
    m0_interference = build_block(m0_full, blank_gamma_per_pulse=1)
    m1_interference = build_block(m1_full, blank_gamma_per_pulse=1)

    if m0_interference.shape[1] != interference_burst_steps:
        raise ValueError(
            f"Interference burst length mismatch: got {m0_interference.shape[1]} steps, "
            f"expected {interference_burst_steps} steps."
        )

    m0_probe = m0_full[:, :probe_steps, ...].contiguous()

    return m0_write, m0_interference, m1_interference, m0_probe


def build_sample_timeline(
    condition: str,
    n_interfere: int,
    m0_write: torch.Tensor,
    m0_interference: torch.Tensor,
    m1_interference: torch.Tensor,
    spec: ExperimentSpec,
) -> Tuple[torch.Tensor, int]:
    if condition not in {"blank", "same", "different"}:
        raise ValueError(f"Unknown condition: {condition}")

    bsz, _, c, h, w = m0_interference.shape
    zero_interference = torch.zeros_like(m0_interference)
    gap = torch.zeros((bsz, spec.gap_steps, c, h, w), device=m0_interference.device)

    segments: List[torch.Tensor] = []

    # M0 write-in stage
    for _ in range(spec.m0_repeats):
        segments.append(m0_write)

    # Interference stage
    for _ in range(n_interfere):
        segments.append(gap)
        if condition == "blank":
            segments.append(zero_interference)
        elif condition == "same":
            segments.append(m0_interference)
        else:
            segments.append(m1_interference)

    sample_spikes = torch.cat(segments, dim=1)
    post_delay_steps = spec.post_delay_steps(n_interfere)

    if post_delay_steps < 0:
        raise ValueError(f"post_delay_steps < 0 for N={n_interfere}. Check N range and timing.")

    # Timeline invariance check: M0 end -> probe start must be fixed
    m0_to_probe_steps = n_interfere * spec.unit_steps + post_delay_steps
    assert m0_to_probe_steps == spec.fixed_m0_to_probe_steps, (
        f"Timeline mismatch at N={n_interfere}: got {m0_to_probe_steps}, "
        f"expected {spec.fixed_m0_to_probe_steps}"
    )

    return sample_spikes, post_delay_steps


def run_condition_n(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset,
    class_index: Dict[int, List[int]],
    condition: str,
    n_interfere: int,
    trials_per_n: int,
    batch_size: int,
    num_classes: int,
    spec: ExperimentSpec,
    rng: random.Random,
    device: torch.device,
) -> List[Dict[str, int]]:
    records: List[Dict[str, int]] = []

    starts = range(0, trials_per_n, batch_size)
    for start in tqdm(starts, desc=f"{condition}-N{n_interfere}", leave=False):
        curr_bs = min(batch_size, trials_per_n - start)

        imgs_m0, lbl_m0, imgs_m1, lbl_m1 = sample_trials(
            dataset=dataset,
            class_index=class_index,
            num_trials=curr_bs,
            num_classes=num_classes,
            rng=rng,
        )

        m0_write, m0_interference, m1_interference, m0_probe = encode_trial_batch(
            encoder=encoder,
            imgs_m0=imgs_m0,
            imgs_m1=imgs_m1,
            probe_steps=spec.probe_steps,
            interference_burst_steps=spec.interference_burst_steps,
            device=device,
        )

        sample_spikes, post_delay_steps = build_sample_timeline(
            condition=condition,
            n_interfere=n_interfere,
            m0_write=m0_write,
            m0_interference=m0_interference,
            m1_interference=m1_interference,
            spec=spec,
        )

        assert sample_spikes.shape[0] == m0_probe.shape[0], "Batch size mismatch between sample and probe"

        with torch.no_grad():
            out = net.forward_classify_session(
                sample_spikes=sample_spikes,
                test_spikes=m0_probe,
                delay_duration_steps=post_delay_steps,
                stsp_mode="dynamic",
            )

        pred = out["prediction"].detach().cpu()
        firing_times = net.layer3.firing_times.detach().cpu()
        min_times, _ = torch.min(firing_times, dim=1)
        lbl_m0_cpu = lbl_m0.detach().cpu()
        lbl_m1_cpu = lbl_m1.detach().cpu()

        valid_pred = ((pred >= -1) & (pred < num_classes)).all().item()
        assert valid_pred, "Prediction contains out-of-range label"

        for i in range(curr_bs):
            p = int(pred[i].item())
            y = int(lbl_m0_cpu[i].item())
            y_diff = int(lbl_m1_cpu[i].item())
            t_fire = float(min_times[i].item())
            first_fire_t = -1 if np.isinf(t_fire) else int(t_fire)
            records.append(
                {
                    "condition": condition,
                    "N": int(n_interfere),
                    "trial_index": int(start + i),
                    "sample_label": y,
                    "different_label": y_diff,
                    "prediction": p,
                    "is_correct": int(p == y),
                    "is_silent": int(p == -1),
                    "first_fire_t": first_fire_t,
                }
            )

    if len(records) > 0:
        arr = np.array([r["first_fire_t"] for r in records], dtype=int)
        p_silent = 100.0 * float((arr == -1).mean())
        p_t20 = 100.0 * float((arr == 20).mean())
        p_t80 = 100.0 * float((arr == 80).mean())
        p_other = 100.0 - p_silent - p_t20 - p_t80
        print(
            f"[Diag] {condition}-N{n_interfere}: "
            f"silent={p_silent:.1f}% | t20={p_t20:.1f}% | t80={p_t80:.1f}% | other={p_other:.1f}%"
        )

    return records


def compute_accuracy_table(df_trials: pd.DataFrame, n_values: Sequence[int]) -> pd.DataFrame:
    agg = (
        df_trials.groupby(["N", "condition"], as_index=False)
        .agg(
            total_trials=("is_correct", "size"),
            correct=("is_correct", "sum"),
            silent=("is_silent", "sum"),
        )
        .sort_values(["N", "condition"])
    )
    agg["accuracy"] = 100.0 * agg["correct"] / agg["total_trials"]
    agg["silent_rate"] = 100.0 * agg["silent"] / agg["total_trials"]

    pivot_acc = agg.pivot(index="N", columns="condition", values="accuracy").reset_index()
    pivot_acc = pivot_acc.rename(
        columns={
            "blank": "acc_blank",
            "same": "acc_same",
            "different": "acc_different",
        }
    )

    for key in ["acc_blank", "acc_same", "acc_different"]:
        if key not in pivot_acc.columns:
            pivot_acc[key] = np.nan

    pivot_acc = pivot_acc[["N", "acc_blank", "acc_same", "acc_different"]]
    pivot_acc = pivot_acc.sort_values("N").reset_index(drop=True)

    pivot_acc["repeat_gain"] = pivot_acc["acc_same"] - pivot_acc["acc_blank"]
    pivot_acc["interference_damage"] = pivot_acc["acc_same"] - pivot_acc["acc_different"]
    pivot_acc["total_damage"] = pivot_acc["acc_blank"] - pivot_acc["acc_different"]

    # Keep requested N order if provided
    n_order = pd.Index(list(n_values), dtype=int, name="N")
    pivot_acc = pivot_acc.set_index("N").reindex(n_order).reset_index()

    return pivot_acc


def save_confusions(df_trials: pd.DataFrame, save_dir: str, num_classes: int) -> None:
    true_labels = list(range(num_classes))
    pred_labels = [-1] + list(range(num_classes))

    for (condition, n_value), group in df_trials.groupby(["condition", "N"]):
        cm = pd.crosstab(group["sample_label"], group["prediction"], dropna=False)
        cm = cm.reindex(index=true_labels, columns=pred_labels, fill_value=0)
        out_path = os.path.join(save_dir, f"confusion_{condition}_N{int(n_value)}.csv")
        cm.to_csv(out_path)


def plot_accuracy_vs_n(metrics: pd.DataFrame, save_dir: str) -> None:
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(9, 6))

    x = metrics["N"].to_numpy()
    plt.plot(x, metrics["acc_blank"], marker="o", linewidth=2.2, label="blank", color="#7f7f7f")
    plt.plot(x, metrics["acc_same"], marker="o", linewidth=2.2, label="same", color="#1f77b4")
    plt.plot(x, metrics["acc_different"], marker="o", linewidth=2.2, label="different", color="#d62728")

    plt.xlabel("N (interference units)")
    plt.ylabel("Probe Accuracy (%)")
    plt.title("Capacity Curves under Chunk Interference")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, "accuracy_vs_N.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_effect_decomposition(metrics: pd.DataFrame, save_dir: str) -> None:
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(9, 6))

    x = metrics["N"].to_numpy()
    plt.plot(x, metrics["repeat_gain"], marker="s", linewidth=2.2, label="repeat_gain = same - blank", color="#2ca02c")
    plt.plot(
        x,
        metrics["interference_damage"],
        marker="s",
        linewidth=2.2,
        label="interference_damage = same - different",
        color="#ff7f0e",
    )
    plt.plot(x, metrics["total_damage"], marker="s", linewidth=2.2, label="total_damage = blank - different", color="#9467bd")

    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("N (interference units)")
    plt.ylabel("Effect Size (percentage points)")
    plt.title("Effect Decomposition")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, "effect_decomposition_vs_N.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def report_capacity_knee(metrics: pd.DataFrame, chance: float = 10.0) -> Tuple[int, float]:
    threshold = max(chance + 5.0, 15.0)
    mask = metrics["acc_different"] <= threshold
    if mask.any():
        n_star = int(metrics.loc[mask, "N"].iloc[0])
    else:
        n_star = -1
    return n_star, threshold


def validate_spec(spec: ExperimentSpec) -> None:
    if spec.probe_steps > spec.chunk_steps:
        raise ValueError("probe_steps cannot exceed chunk_steps")
    if spec.interference_burst_steps <= 0:
        raise ValueError("interference_burst_steps must be positive")

    for n_val in spec.n_values:
        post = spec.post_delay_steps(n_val)
        if post < 0:
            raise ValueError(f"Negative post delay at N={n_val}: {post}")
        m0_to_probe = n_val * spec.unit_steps + post
        assert m0_to_probe == spec.fixed_m0_to_probe_steps


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    n_values = parse_n_values(args.n_values)
    spec = ExperimentSpec(
        dt=1.0 * ms,
        chunk_ms=200.0,
        probe_ms=100.0,
        m0_repeats=1,
        gap_ms=80.0,
        interference_burst_ms=120.0,
        fixed_m0_to_probe_ms=1500.0,
        n_values=n_values,
    )
    validate_spec(spec)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Init] Device: {device}")
    print(f"[Init] Save dir: {args.save_dir}")
    print(
        "[Init] Timing: "
        f"chunk={spec.chunk_steps} steps, probe={spec.probe_steps} steps, "
        f"interference_burst={spec.interference_burst_steps} steps, gap={spec.gap_steps} steps, "
        f"M0_repeats={spec.m0_repeats}, unit={spec.unit_steps} steps"
    )
    print(
        "[Init] Delay anchor: "
        f"fixed M0->probe = {spec.fixed_m0_to_probe_steps} steps ({spec.fixed_m0_to_probe_ms:.0f} ms)"
    )

    net, encoder = load_model_and_encoder(args.model_path, device, spec)

    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    class_index = build_class_index(dataset, num_classes=args.num_classes)

    all_records: List[Dict[str, int]] = []
    rng = random.Random(args.seed)

    conditions = ["blank", "same", "different"]
    for n_interfere in spec.n_values:
        print(f"\n[Run] N={n_interfere}, post_delay={spec.post_delay_steps(n_interfere)} steps")
        for condition in conditions:
            records = run_condition_n(
                net=net,
                encoder=encoder,
                dataset=dataset,
                class_index=class_index,
                condition=condition,
                n_interfere=n_interfere,
                trials_per_n=args.trials_per_n,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
                spec=spec,
                rng=rng,
                device=device,
            )
            all_records.extend(records)

    df_trials = pd.DataFrame(all_records)

    # Required output 1: trial-level predictions
    trial_csv = os.path.join(args.save_dir, "trial_level_predictions.csv")
    df_trials.to_csv(trial_csv, index=False)

    # Required output 2: metrics table
    metrics = compute_accuracy_table(df_trials, n_values=spec.n_values)
    metrics_csv = os.path.join(args.save_dir, "metrics_capacity.csv")
    metrics.to_csv(metrics_csv, index=False)

    # Required output 3: confusion files
    save_confusions(df_trials, args.save_dir, num_classes=args.num_classes)

    # Required output 4: plots
    plot_accuracy_vs_n(metrics, args.save_dir)
    plot_effect_decomposition(metrics, args.save_dir)

    n_star, threshold = report_capacity_knee(metrics, chance=10.0)

    print("\n=== Capacity Experiment Summary ===")
    print("Time baseline stability: inspect acc_blank vs N (expected smoother trend).")
    print("Repeat effect: repeat_gain = acc_same - acc_blank.")
    print("Interference effect: interference_damage = acc_same - acc_different.")

    if n_star >= 0:
        print(f"Capacity knee N*: {n_star} (first N with acc_different <= {threshold:.1f}%).")
    else:
        print(f"Capacity knee N*: not reached within tested N (threshold {threshold:.1f}%).")

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {trial_csv}")
    print(f"Saved: {os.path.join(args.save_dir, 'accuracy_vs_N.png')}")
    print(f"Saved: {os.path.join(args.save_dir, 'effect_decomposition_vs_N.png')}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Working memory capacity experiment with chunk interference")
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--save-dir", type=str, default="results/capacity_chunk_experiment")
    parser.add_argument("--n-values", type=str, default="0,1,2,4,5,6,7")
    parser.add_argument("--trials-per-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
