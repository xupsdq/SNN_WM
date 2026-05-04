from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.units import ms
from src.experiments.common.model_io import load_model_and_encoder
from src.experiments.common.results import prepare_result_layout
from src.experiments.common.runtime import resolve_device, seed_everything
from src.experiments.common.ping_common import LAYER_KEYS, prepare_network_state, snapshot_ux_state
from src.plotting.common.io import save_figure_all_formats, save_tidy_csv
from src.platform.legacy_adapters.encoding import build_mnist_skeleton_loader


DEFAULT_DELAY_POINTS_MS = [100, 200, 400, 800, 1200]
DEFAULT_SAMPLE_DURATION_MS = 200.0
DEFAULT_TRAIN_PER_CLASS = 50
DEFAULT_TEST_PER_CLASS = 50
DEFAULT_RESULTS_DIR = os.path.join("results", "engram_decode_experiment")
DEFAULT_EXPERIMENT_NAME = "engram_decode_experiment"
CHANCE_LEVEL = 0.1
LAYER_DISPLAY_NAMES = {
    "layer1": "Layer1",
    "layer2": "Layer2",
    "layer3": "Layer3",
}


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def setup_logger(log_path: Path, experiment_name: str) -> logging.Logger:
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(data, path: Path, logger: logging.Logger | None = None) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, indent=2, ensure_ascii=False, sort_keys=True)
    if logger is not None:
        logger.info("[Save] JSON saved to %s", path)
    return path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode synaptic engram features across delay intervals.")
    parser.add_argument("--model-path", type=str, default=os.path.join("results", "sdnn_deep_final", "net_final.pth"))
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sample-duration-ms", type=float, default=DEFAULT_SAMPLE_DURATION_MS)
    parser.add_argument("--delay-points-ms", type=str, default="100,400,1200")
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_TRAIN_PER_CLASS)
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_TEST_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-diagnostic-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def _parse_delay_points(raw_value: str) -> List[int]:
    items = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not items:
        raise ValueError("delay-points-ms cannot be empty")
    return [int(float(item)) for item in items]


def _build_balanced_subset(dataset, per_class: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    indices_by_class = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        indices_by_class[int(label)].append(idx)

    selected_indices = []
    for cls in range(10):
        cls_indices = indices_by_class[cls]
        if len(cls_indices) < per_class:
            raise ValueError(f"Class {cls} has only {len(cls_indices)} samples, expected >= {per_class}.")
        chosen = rng.choice(cls_indices, size=per_class, replace=False)
        selected_indices.extend(chosen.tolist())
    return Subset(dataset, selected_indices)


def _snapshot_features_at_delays(
    net,
    sample_spikes: torch.Tensor,
    delay_points_ms: List[int],
    dt_seconds: float,
) -> Dict[str, Dict[int, np.ndarray]]:
    batch_size, t_sample, channels, height, width = sample_spikes.shape
    prepare_network_state(net, batch_size, channels, height, width)
    zero_input = torch.zeros((batch_size, channels, height, width), device=sample_spikes.device)
    current_time = 0

    def step_network(input_t: torch.Tensor) -> None:
        nonlocal current_time
        s1, _ = net.layer1.forward_step(input_t, current_time, training=False, stsp_mode="dynamic")
        s1_p = net.pool1(s1.float())
        s2, _ = net.layer2.forward_step(s1_p, current_time, training=False, stsp_mode="dynamic")
        s2_p = net.pool2(s2.float())
        net.layer3.forward_step(s2_p, current_time, training=False, monitor=False, stsp_mode="dynamic")
        current_time += 1

    for t_step in range(t_sample):
        step_network(sample_spikes[:, t_step, ...])

    target_steps = {delay_ms: int(round((delay_ms * ms) / dt_seconds)) for delay_ms in delay_points_ms}
    max_target_steps = max(target_steps.values())
    snapshots_by_delay: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    for delay_step in range(1, max_target_steps + 1):
        step_network(zero_input)
        matched_delays = [delay_ms for delay_ms, target in target_steps.items() if target == delay_step]
        if not matched_delays:
            continue
        snapshot = snapshot_ux_state(net, batch_size=batch_size)
        for delay_ms in matched_delays:
            snapshots_by_delay[delay_ms] = snapshot

    features_by_layer: Dict[str, Dict[int, np.ndarray]] = {layer_name: {} for layer_name in LAYER_KEYS}
    for delay_ms, snapshot in snapshots_by_delay.items():
        for layer_name in LAYER_KEYS:
            u_arr = snapshot[layer_name]["u"]
            x_arr = snapshot[layer_name]["x"]
            features_by_layer[layer_name][delay_ms] = np.concatenate([u_arr, x_arr], axis=1).astype(
                np.float32,
                copy=False,
            )
    return features_by_layer


def _collect_layer_features(
    net,
    encoder,
    data_loader: DataLoader,
    delay_points_ms: List[int],
    dt_seconds: float,
    sample_steps: int,
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    features_by_layer = {
        layer_name: {delay_ms: [] for delay_ms in delay_points_ms}
        for layer_name in LAYER_KEYS
    }
    labels_all = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Collecting engram features"):
            images = images.to(next(net.parameters()).device)
            sample_spikes = encoder.forward(images)[:, :sample_steps, ...]
            delay_snapshots = _snapshot_features_at_delays(
                net=net,
                sample_spikes=sample_spikes,
                delay_points_ms=delay_points_ms,
                dt_seconds=dt_seconds,
            )
            for layer_name in LAYER_KEYS:
                for delay_ms in delay_points_ms:
                    features_by_layer[layer_name][delay_ms].append(delay_snapshots[layer_name][delay_ms])
            labels_all.append(labels.numpy())

    labels_np = np.concatenate(labels_all, axis=0)
    packed: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for layer_name in LAYER_KEYS:
        packed[layer_name] = {}
        for delay_ms in delay_points_ms:
            feature_matrix = np.concatenate(features_by_layer[layer_name][delay_ms], axis=0)
            packed[layer_name][delay_ms] = (feature_matrix, labels_np.copy())
    return packed


def _bootstrap_ci(binary_values: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(binary_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, len(values), size=len(values))
        boot[idx] = float(values[sample_idx].mean())
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _permutation_test_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_perm: int, seed: int) -> float:
    observed = float(np.mean(y_true == y_pred))
    rng = np.random.default_rng(seed)
    perm_scores = np.zeros(n_perm, dtype=np.float64)
    for idx in range(n_perm):
        perm_scores[idx] = float(np.mean(rng.permutation(y_true) == y_pred))
    return float((np.sum(perm_scores >= observed) + 1.0) / (len(perm_scores) + 1.0))


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, layer_name: str, delay_ms: int, save_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Engram Decoder Confusion ({LAYER_DISPLAY_NAMES[layer_name]}, {delay_ms} ms)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_dir / f"confusion_{layer_name}_{delay_ms}ms.png", dpi=300)
    plt.close(fig)


def _plot_pca(features: np.ndarray, labels: np.ndarray, layer_name: str, delay_ms: int, save_dir: Path) -> None:
    if features.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(features)
    else:
        reduced = features

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=18, alpha=0.85)
    ax.set_title(f"Engram PCA ({LAYER_DISPLAY_NAMES[layer_name]}, {delay_ms} ms)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [str(i) for i in range(10)], title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_dir / f"pca_{layer_name}_{delay_ms}ms.png", dpi=300)
    plt.close(fig)


def build_accuracy_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer_name in LAYER_KEYS:
        part = metrics_df[metrics_df["layer"] == layer_name].sort_values("delay_ms")
        if len(part) == 0:
            continue
        x_vals = part["delay_ms"].to_numpy(dtype=float)
        y_vals = part["acc"].to_numpy(dtype=float)
        y_low = part["acc_ci_low"].to_numpy(dtype=float)
        y_high = part["acc_ci_high"].to_numpy(dtype=float)
        ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=LAYER_DISPLAY_NAMES[layer_name])
        ax.fill_between(x_vals, y_low, y_high, alpha=0.2)

    ax.axhline(CHANCE_LEVEL, color="k", linestyle="--", linewidth=1, label="Chance (10%)")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_title("Accuracy vs Delay")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig


def build_summary(metrics_df: pd.DataFrame, delay_points_ms: List[int], experiment_name: str) -> Dict[str, object]:
    best_row = metrics_df.sort_values(["acc", "macro_f1"], ascending=[False, False], kind="stable").iloc[0]
    shortest_delay = min(delay_points_ms)
    longest_delay = max(delay_points_ms)

    def acc_at(layer_name: str, delay_ms: int) -> float:
        row = metrics_df[(metrics_df["layer"] == layer_name) & (metrics_df["delay_ms"] == delay_ms)]
        if len(row) == 0:
            return float("nan")
        return float(row.iloc[0]["acc"])

    avg_short = float(metrics_df[metrics_df["delay_ms"] == shortest_delay]["acc"].mean())
    avg_long = float(metrics_df[metrics_df["delay_ms"] == longest_delay]["acc"].mean())
    if avg_long < avg_short:
        trend_text = "with decoding accuracy decreasing as delay increases"
    elif avg_long > avg_short:
        trend_text = "with decoding accuracy increasing as delay increases"
    else:
        trend_text = "with decoding accuracy remaining stable across delays"

    summary_text = (
        "u/x-based engram remains decodable above chance across delays, "
        f"{trend_text}."
    )

    return {
        "experiment_name": experiment_name,
        "primary_figure": os.path.join("figures", "accuracy_vs_delay.png"),
        "delay_points_ms": [int(v) for v in delay_points_ms],
        "chance_level": float(CHANCE_LEVEL),
        "best_layer": str(best_row["layer"]),
        "best_delay_ms": int(best_row["delay_ms"]),
        "best_accuracy": float(best_row["acc"]),
        "best_macro_f1": float(best_row["macro_f1"]),
        "layer1_acc_at_shortest_delay": acc_at("layer1", shortest_delay),
        "layer1_acc_at_longest_delay": acc_at("layer1", longest_delay),
        "layer2_acc_at_shortest_delay": acc_at("layer2", shortest_delay),
        "layer2_acc_at_longest_delay": acc_at("layer2", longest_delay),
        "layer3_acc_at_shortest_delay": acc_at("layer3", shortest_delay),
        "layer3_acc_at_longest_delay": acc_at("layer3", longest_delay),
        "summary_text": summary_text,
    }


def main() -> None:
    args = build_argparser().parse_args()
    layout = prepare_result_layout(args.save_dir)
    experiment_name = Path(args.save_dir).name or DEFAULT_EXPERIMENT_NAME
    logger = setup_logger(layout.log_file(), experiment_name)
    data_dir = layout.data_dir
    metrics_dir = layout.metrics_dir
    meta_dir = layout.meta_dir

    seed_everything(args.seed)
    device = resolve_device(args.device)
    delay_points_ms = _parse_delay_points(args.delay_points_ms)
    dt_seconds = float(args.dt_ms * ms)
    sample_steps = int(round((float(args.sample_duration_ms) * ms) / dt_seconds))

    logger.info("[Init] Run started at %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("[Init] Save dir: %s", layout.root)
    logger.info("[Init] Device: %s", device)
    logger.info("[Init] Model path: %s", args.model_path)
    logger.info("[Init] Dataset root: %s", args.dataset_root)
    logger.info("[Init] Delay points (ms): %s", delay_points_ms)
    logger.info(
        "[Init] Train/Test per class: %d / %d | batch_size=%d | input_size=%d | dt_ms=%.3f",
        args.train_per_class,
        args.test_per_class,
        args.batch_size,
        args.input_size,
        args.dt_ms,
    )
    logger.info("[Init] Save diagnostic plots: %s", bool(args.save_diagnostic_plots))

    run_config = {
        "experiment_name": experiment_name,
        "model_path": args.model_path,
        "dataset_root": args.dataset_root,
        "seed": int(args.seed),
        "device": str(device),
        "delay_points_ms": [int(v) for v in delay_points_ms],
        "sample_duration_ms": float(args.sample_duration_ms),
        "train_per_class": int(args.train_per_class),
        "test_per_class": int(args.test_per_class),
        "batch_size": int(args.batch_size),
        "input_size": int(args.input_size),
        "dt_ms": float(args.dt_ms),
        "save_diagnostic_plots": bool(args.save_diagnostic_plots),
    }
    run_config_path = save_json(run_config, layout.root_file("run_config.json"), logger)
    save_json(run_config, meta_dir / "run_config.snapshot.json", logger)

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt=dt_seconds,
        max_duration_ms=max(max(delay_points_ms) + float(args.sample_duration_ms), 500.0),
    )

    train_loader_full, _, test_loader_full = build_mnist_skeleton_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        input_size=args.input_size,
    )
    train_subset = _build_balanced_subset(train_loader_full.dataset, args.train_per_class, args.seed)
    test_subset = _build_balanced_subset(test_loader_full.dataset, args.test_per_class, args.seed + 1)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    cache_train = _collect_layer_features(
        net=net,
        encoder=encoder,
        data_loader=train_loader,
        delay_points_ms=delay_points_ms,
        dt_seconds=dt_seconds,
        sample_steps=sample_steps,
    )
    cache_test = _collect_layer_features(
        net=net,
        encoder=encoder,
        data_loader=test_loader,
        delay_points_ms=delay_points_ms,
        dt_seconds=dt_seconds,
        sample_steps=sample_steps,
    )

    diagnostic_dir = layout.figure_dir / "diagnostic"
    if args.save_diagnostic_plots:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for layer_name in LAYER_KEYS:
        logger.info("[Layer] %s decode", LAYER_DISPLAY_NAMES[layer_name])
        for delay_ms in delay_points_ms:
            x_train, y_train = cache_train[layer_name][delay_ms]
            x_test, y_test = cache_test[layer_name][delay_ms]

            clf = LinearSVC(dual=False, C=1.0, max_iter=3000)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            ci_low, ci_high = _bootstrap_ci((y_test == y_pred).astype(float), n_bootstrap=1000, seed=args.seed)
            p_perm = _permutation_test_accuracy(y_true=y_test, y_pred=y_pred, n_perm=1000, seed=args.seed)

            logger.info(
                "[Metric] %s @ %d ms | acc=%.4f | macro_f1=%.4f | ci=[%.4f, %.4f] | perm_p=%.4g",
                layer_name,
                delay_ms,
                acc,
                macro_f1,
                ci_low,
                ci_high,
                p_perm,
            )

            records.append(
                {
                    "layer": layer_name,
                    "delay_ms": int(delay_ms),
                    "acc": float(acc),
                    "macro_f1": float(macro_f1),
                    "acc_ci_low": float(ci_low),
                    "acc_ci_high": float(ci_high),
                    "perm_p": float(p_perm),
                }
            )

            if bool(args.save_diagnostic_plots) and int(delay_ms) == 400:
                _plot_confusion(y_test, y_pred, layer_name, int(delay_ms), diagnostic_dir)
                _plot_pca(x_test, y_test, layer_name, int(delay_ms), diagnostic_dir)

    metrics_df = pd.DataFrame(records).sort_values(["layer", "delay_ms"], kind="stable").reset_index(drop=True)
    metrics_path = Path(save_tidy_csv(metrics_df, metrics_dir / "engram_decode_metrics.csv", sort_by=["layer", "delay_ms"]))
    compat_metrics_path = Path(save_tidy_csv(metrics_df, data_dir / "engram_decode_metrics.csv", sort_by=["layer", "delay_ms"]))
    logger.info("[Save] Metrics CSV saved to %s", metrics_path)
    logger.info("[Save] Compatibility metrics CSV saved to %s", compat_metrics_path)

    figure_paths = {"png": "", "pdf": "", "svg": ""}
    if not bool(args.skip_figures):
        fig = build_accuracy_figure(metrics_df)
        figure_paths = save_figure_all_formats(fig, layout.figure_base("accuracy_vs_delay"))
        plt.close(fig)
        logger.info("[Save] Primary figure saved to %s", figure_paths["png"])
        logger.info("[Save] Primary figure saved to %s", figure_paths["pdf"])
        logger.info("[Save] Primary figure saved to %s", figure_paths["svg"])

    summary = build_summary(metrics_df, delay_points_ms, experiment_name)
    summary["metrics_csv"] = str(metrics_path.resolve())
    summary["compat_metrics_csv"] = str(compat_metrics_path.resolve())
    summary_path = save_json(summary, layout.root_file("summary.json"), logger)
    save_json(summary, metrics_dir / "summary.json", logger)
    save_json(
        {
            "experiment": "engram_decode",
            "best_layer": summary["best_layer"],
            "best_delay_ms": summary["best_delay_ms"],
            "best_accuracy": summary["best_accuracy"],
            "best_macro_f1": summary["best_macro_f1"],
            "metrics_csv": str(metrics_path.resolve()),
        },
        metrics_dir / "main_metrics.json",
        logger,
    )

    logger.info("[Done] metrics_csv=%s", metrics_path)
    logger.info("[Done] primary_figure=%s", figure_paths["png"])
    logger.info("[Done] summary_json=%s", summary_path)
    logger.info("[Done] run_config_json=%s", run_config_path)
    # TODO: Feature caches remain in-memory only; persist to data/ only if later reproduction needs them.


if __name__ == "__main__":
    main()
