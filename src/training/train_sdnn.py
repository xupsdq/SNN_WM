from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config.paths import DEFAULT_PATH_CONFIG
from src.core.monitoring import ClassSensitivityMonitor
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import ms
from src.training.plotting import (
    plot_class_sensitivity,
    plot_learned_kernels,
    plot_weight_distribution_evolution,
)


DEFAULT_SAVE_DIR = "results/sdnn_train_single"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = DEFAULT_SAVE_DIR


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    dataset_root: str = str(DEFAULT_PATH_CONFIG.dataset_root)
    device: str = "auto"
    seed: int = 42
    batch_size: int = 32
    input_size: int = 28
    l1_epochs: int = 2
    l2_epochs: int = 10
    l3_epochs: int = 500
    max_batches: int | None = None
    eval_max_batches: int | None = None
    l3_eval_every: int | None = 100
    skip_final_eval: bool = False
    enable_stsp: bool = False
    smoke: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(raw_device)


def _limited_batches(loader, max_batches: int | None = None):
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def set_network_stsp_enabled(net, enabled: bool) -> None:
    for layer in (net.layer1, net.layer2, net.layer3):
        layer.enable_stsp = bool(enabled)


def evaluate_network(net, encoder, test_loader, device: torch.device | None = None, max_batches: int | None = None) -> float:
    device = device or DEVICE
    print("\n=== Starting Evaluation on Test Set ===")
    net.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    correct = 0
    total = 0

    original_noise = net.layer3.current_noise_std.item()
    net.layer3.current_noise_std.fill_(0.0)

    try:
        with torch.no_grad():
            pbar = tqdm(_limited_batches(test_loader, max_batches), desc="Testing")
            for _, (images, labels) in pbar:
                images = images.to(device)
                labels = labels.to(device)

                spike_train = encoder.forward(images)
                net(spike_train, layer_idx=3, labels=None, monitor=False)
                firing_times = net.layer3.firing_times

                _, min_indices = torch.min(firing_times, dim=1)
                predicted_class = min_indices // net.layer3.neurons_per_class

                batch_correct = (predicted_class == labels).sum().item()
                correct += batch_correct
                total += labels.size(0)

                all_preds.extend(predicted_class.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
                pbar.set_postfix({"Current Acc": f"{correct / total:.4f}"})
    finally:
        net.layer3.current_noise_std.fill_(original_noise)

    final_acc = correct / total if total else 0.0
    print(f"\n>>> Test Set Accuracy: {final_acc * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Cls {i}" for i in range(10)],
        yticklabels=[f"Cls {i}" for i in range(10)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Acc: {final_acc:.2%})")
    save_path = os.path.join(SAVE_DIR, "test_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return float(final_acc)


def train_layer1(net, encoder, train_loader, epochs: int = 2, max_batches: int | None = None) -> None:
    print(f"\n=== [Phase 1] Training Layer 1 for {epochs} Epochs ===")
    l1_save_dir = os.path.join(SAVE_DIR, "layer1_logs")
    os.makedirs(l1_save_dir, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        pbar = tqdm(train_loader, desc=f"L1 Epoch {epoch + 1}")
        for batch_idx, (images, _) in _limited_batches(pbar, max_batches):
            images = images.to(DEVICE)
            with torch.no_grad():
                spike_train = encoder.forward(images)

            do_monitor = batch_idx % 1000 == 0
            net(spike_train, layer_idx=1, monitor=do_monitor)

            if do_monitor:
                save_prefix = os.path.join(l1_save_dir, f"forensics_e{epoch}_b{batch_idx}")
                plot_learned_kernels(net.get_kernels(layer_idx=1), save_prefix + "_kernels.png", f"e{epoch}_b{batch_idx}")
                plot_weight_distribution_evolution(net, l1_save_dir, epoch, layer_idx=1)

        torch.save(net.state_dict(), os.path.join(SAVE_DIR, f"net_e{epoch}_L1.pth"))

    final_theta = net.layer1.theta.detach().cpu().numpy()
    print(f"[Verify] L1 Theta Saved - Min: {final_theta.min() * 1000:.2f}mV, Max: {final_theta.max() * 1000:.2f}mV")
    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "net_after_L1.pth"))


def train_layer2(net, encoder, train_loader, epochs: int = 10, max_batches: int | None = None) -> None:
    print(f"\n=== [Phase 2] Training Layer 2 for {epochs} Epochs ===")
    l2_save_dir = os.path.join(SAVE_DIR, "layer2_logs")
    os.makedirs(l2_save_dir, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        pbar = tqdm(train_loader, desc=f"L2 Epoch {epoch + 1}")
        for batch_idx, (images, _) in _limited_batches(pbar, max_batches):
            images = images.to(DEVICE)
            with torch.no_grad():
                spike_train = encoder.forward(images)

            do_monitor = batch_idx % 1000 == 0
            net(spike_train, layer_idx=2, monitor=do_monitor)
            if do_monitor:
                plot_weight_distribution_evolution(net, l2_save_dir, epoch, layer_idx=2)

        torch.save(net.state_dict(), os.path.join(SAVE_DIR, f"net_e{epoch}_L2.pth"))

    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "net_after_L2.pth"))


def train_layer3(
    net,
    encoder,
    train_loader,
    test_loader,
    epochs: int = 10,
    max_batches: int | None = None,
    eval_max_batches: int | None = None,
    eval_every: int | None = 100,
) -> None:
    print(f"\n=== [Phase 3] Training Layer 3 (R-STDP with Adaptive Rates) for {epochs} Epochs ===")
    cs_monitor = ClassSensitivityMonitor(num_classes=10, neurons_per_class=net.layer3.neurons_per_class, device=DEVICE)
    l3_save_dir = os.path.join(SAVE_DIR, "layer3_logs")
    os.makedirs(l3_save_dir, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        pbar = tqdm(train_loader, desc=f"L3 Epoch {epoch + 1}")
        for batch_idx, (images, labels) in _limited_batches(pbar, max_batches):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                spike_train = encoder.forward(images)

            do_monitor = batch_idx % 1000 == 0
            net(spike_train, layer_idx=3, labels=labels, monitor=do_monitor)

            _, min_indices = torch.min(net.layer3.firing_times, dim=1)
            predicted_class = min_indices // net.layer3.neurons_per_class
            batch_acc = (predicted_class == labels).sum().item() / labels.size(0)
            curr_r, curr_p = net.layer3.update_adaptive_rates(batch_accuracy=batch_acc)
            pbar.set_postfix(
                {
                    "RunAcc": f"{net.layer3.running_avg_acc.item() * 100:.1f}%",
                    "Rw": f"{curr_r:.2f}",
                    "Pn": f"{curr_p:.2f}",
                    "batch_acc": f"{batch_acc:.2f}",
                }
            )

            cs_monitor.update(labels, net.layer3.firing_times)
            if do_monitor:
                save_path_sens = os.path.join(l3_save_dir, f"sensitivity_e{epoch + 100}_b{batch_idx}.png")
                norm_cm, n_counts = cs_monitor.get_metrics()
                plot_class_sensitivity(norm_cm, n_counts, save_path_sens, f"Epoch {epoch + 100} Batch {batch_idx}")
                cs_monitor.reset()

        torch.save(net.state_dict(), os.path.join(SAVE_DIR, f"net_e{epoch + 100}_L3.pth"))
        if eval_every is not None and eval_every > 0 and epoch % eval_every == 0:
            evaluate_network(net, encoder, test_loader, device=DEVICE, max_batches=eval_max_batches)

    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "net_final.pth"))


def compensate_stsp_gain(net, scaling_factor: float = 5.0) -> None:
    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor
            if hasattr(net.layer3, "target_norm"):
                net.layer3.target_norm *= scaling_factor


def train_single_network(config: TrainingConfig) -> dict[str, Any]:
    global DEVICE, SAVE_DIR

    start_time = time.time()
    seed_everything(config.seed)
    DEVICE = _resolve_device(config.device)
    SAVE_DIR = str(Path(config.output_dir))
    output_dir = Path(SAVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = asdict(config)
    run_config["resolved_device"] = str(DEVICE)
    _write_json(output_dir / "run_config.json", run_config)

    print(f"[Init] Using device: {DEVICE}")
    print(f"[Init] Seed: {config.seed}")
    print(f"[Init] Output dir: {output_dir}")

    dt = 1.0 * ms
    train_loader, _, test_loader = build_mnist_skeleton_loader(
        root=config.dataset_root,
        batch_size=config.batch_size,
        input_size=config.input_size,
    )

    encoder = DoGSpikeEncoder(dt=dt, theta_freq=5.0, gamma_freq=50.0, max_duration=200 * ms, device=DEVICE)
    net = SDNN_Network(device=DEVICE).to(DEVICE)
    set_network_stsp_enabled(net, config.enable_stsp)
    print(f"[Init] STSP enabled during training: {config.enable_stsp}")

    train_layer1(net, encoder, train_loader, epochs=config.l1_epochs, max_batches=config.max_batches)
    train_layer2(net, encoder, train_loader, epochs=config.l2_epochs, max_batches=config.max_batches)
    train_layer3(
        net,
        encoder,
        train_loader,
        test_loader,
        epochs=config.l3_epochs,
        max_batches=config.max_batches,
        eval_max_batches=config.eval_max_batches,
        eval_every=config.l3_eval_every,
    )

    final_accuracy = None
    if not config.skip_final_eval:
        final_accuracy = evaluate_network(net, encoder, test_loader, device=DEVICE, max_batches=config.eval_max_batches)

    elapsed_seconds = time.time() - start_time
    summary = {
        "status": "success",
        "seed": int(config.seed),
        "device": str(DEVICE),
        "dataset_root": config.dataset_root,
        "output_dir": str(output_dir),
        "batch_size": int(config.batch_size),
        "l1_epochs": int(config.l1_epochs),
        "l2_epochs": int(config.l2_epochs),
        "l3_epochs": int(config.l3_epochs),
        "max_batches": config.max_batches,
        "eval_max_batches": config.eval_max_batches,
        "final_accuracy": final_accuracy,
        "elapsed_seconds": elapsed_seconds,
        "checkpoints": ["net_after_L1.pth", "net_after_L2.pth", "net_final.pth"],
    }
    _write_json(output_dir / "summary.json", summary)
    if config.smoke:
        _write_json(output_dir / "smoke_summary.json", summary)
    print(f"[Done] Training finished in {elapsed_seconds:.1f}s")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one SDNN network from scratch.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_PATH_CONFIG.dataset_root))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--l1-epochs", type=int, default=2)
    parser.add_argument("--l2-epochs", type=int, default=10)
    parser.add_argument("--l3-epochs", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--l3-eval-every", type=int, default=100, help="Use 0 to disable periodic L3 evaluation.")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--enable-stsp", action="store_true", help="Enable dynamic STSP during training. Default is disabled.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    parser.add_argument("--smoke-eval-batches", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    l3_eval_every = None if args.l3_eval_every <= 0 else args.l3_eval_every
    config = TrainingConfig(
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        input_size=args.input_size,
        l1_epochs=1 if args.smoke else args.l1_epochs,
        l2_epochs=1 if args.smoke else args.l2_epochs,
        l3_epochs=1 if args.smoke else args.l3_epochs,
        max_batches=args.smoke_batches if args.smoke else args.max_batches,
        eval_max_batches=args.smoke_eval_batches if args.smoke else args.eval_max_batches,
        l3_eval_every=1 if args.smoke else l3_eval_every,
        skip_final_eval=args.skip_final_eval,
        enable_stsp=args.enable_stsp,
        smoke=args.smoke,
    )
    train_single_network(config)


if __name__ == "__main__":
    main()
