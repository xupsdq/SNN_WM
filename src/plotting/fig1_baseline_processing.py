import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from src.platform.legacy_adapters.network import SDNN_Network
from figure_utils_common import (
    COLOR_NOISE,
    COLOR_STATIC,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    PUBLICATION_TWO_COLUMN_FIGSIZE,
    save_figure_all_formats,
    save_run_config,
    save_tidy_csv,
    validate_required_columns,
)
from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from paper_plot_style import DEFAULT_SUBPLOT_ADJUST, PANEL_LABEL_FONT_SIZE, apply_paper_style
from src.experiments.common.decoding import decode_prediction_and_fire_time_from_layer3 as shared_decode_prediction_and_fire_time_from_layer3
from src.experiments.common.model_io import compensate_stsp_gain as shared_compensate_stsp_gain
from src.experiments.common.model_io import load_model_and_encoder as shared_load_model_and_encoder
from src.experiments.common.runtime import seed_everything as shared_seed_everything
from src.platform.legacy_adapters.units import ms


TRIAL_COLUMNS: List[str] = [
    "trial_id",
    "image_index",
    "true_label",
    "pred_label",
    "is_correct",
    "is_silent",
    "first_fire_t",
]
CONFUSION_COLUMNS: List[str] = [
    "true_label",
    "pred_label",
    "count",
    "fraction_row_normalized",
]
RECALL_COLUMNS: List[str] = [
    "label",
    "recall",
    "n_trials",
    "overall_accuracy",
]
FORBIDDEN_FIGURE_TERMS: Tuple[str, ...] = (
    "dms",
    "delay",
    "sample",
    "probe",
    "memory",
    "donor",
    "stsp-on-vs-off",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 1 baseline classification plotter for single-image sensory processing."
    )
    parser.add_argument("--model-path", type=str, default="results/sdnn_deep_final/net_final.pth")
    parser.add_argument("--dataset-root", type=str, default="./MNIST")
    parser.add_argument("--save-dir", type=str, default="results/fig1_baseline_processing")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--max-duration-ms", type=float, default=200.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
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
            if hasattr(net.layer3, "target_norm"):
                net.layer3.target_norm *= scaling_factor


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    dt_ms: float,
    max_duration_ms: float,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path_obj}")

    net = SDNN_Network(device=str(device)).to(device)
    net.load_state_dict(torch.load(model_path_obj, map_location=device))
    compensate_stsp_gain(net, scaling_factor=1.0 / net.layer3.stsp_U)
    net.eval()

    encoder = DoGSpikeEncoder(
        dt=dt_ms * ms,
        max_duration=max_duration_ms * ms,
        device=str(device),
    )
    return net, encoder


def decode_prediction_and_fire_time_from_layer3(
    net: SDNN_Network,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_times = net.layer3.firing_times
    if flat_times.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: firing_times={flat_times.shape[0]}, expected={batch_size}")

    has_fired = (flat_times != float("inf")).any(dim=1)
    min_times, min_indices = torch.min(flat_times, dim=1)
    pred = (min_indices // net.layer3.neurons_per_class).long()
    pred[~has_fired] = -1

    fire_t = min_times.clone()
    fire_t[~has_fired] = -1
    return pred.detach().cpu(), fire_t.to(torch.long).detach().cpu()


def run_baseline_classification(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset_root: str,
    batch_size: int,
    input_size: int,
    device: torch.device,
) -> pd.DataFrame:
    _, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=0,
    )
    dataset = test_loader.dataset
    if len(dataset) == 0:
        raise ValueError("Test dataset is empty")

    original_noise = float(net.layer3.current_noise_std.item())
    rows: List[Dict[str, int]] = []
    image_offset = 0

    try:
        net.layer3.current_noise_std.fill_(0.0)
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Baseline classification", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                spike_train = encoder.forward(images)
                _ = net(spike_train, layer_idx=3, labels=None, monitor=False)
                pred, fire_t = decode_prediction_and_fire_time_from_layer3(net, batch_size=len(labels))

                labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
                pred_np = pred.numpy().astype(np.int64, copy=False)
                fire_t_np = fire_t.numpy().astype(np.int64, copy=False)

                for batch_idx in range(len(labels_np)):
                    true_label = int(labels_np[batch_idx])
                    pred_label = int(pred_np[batch_idx])
                    first_fire_t = int(fire_t_np[batch_idx])
                    is_silent = int(pred_label == -1)
                    is_correct = int((pred_label == true_label) and (is_silent == 0))
                    rows.append(
                        {
                            "trial_id": int(image_offset + batch_idx),
                            "image_index": int(image_offset + batch_idx),
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "is_correct": is_correct,
                            "is_silent": is_silent,
                            "first_fire_t": first_fire_t,
                        }
                    )
                image_offset += len(labels_np)
    finally:
        net.layer3.current_noise_std.fill_(original_noise)

    df_trials = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    validate_required_columns(df_trials, TRIAL_COLUMNS)
    if len(df_trials) != len(dataset):
        raise ValueError(f"Trial table length mismatch: got {len(df_trials)}, expected {len(dataset)}")
    return df_trials


def build_confusion_matrix(df_trials: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df_trials, ["true_label", "pred_label"])
    true_labels = sorted(int(x) for x in pd.unique(df_trials["true_label"]))
    pred_labels = list(true_labels)
    if (df_trials["pred_label"] == -1).any():
        pred_labels.append(-1)

    rows: List[Dict[str, float]] = []
    for true_label in true_labels:
        sub = df_trials[df_trials["true_label"] == true_label]
        total = int(len(sub))
        if total <= 0:
            raise ValueError(f"Class {true_label} has zero trials")
        counts = sub["pred_label"].value_counts().to_dict()
        for pred_label in pred_labels:
            count = int(counts.get(pred_label, 0))
            rows.append(
                {
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "count": count,
                    "fraction_row_normalized": float(count / total),
                }
            )

    df_confusion = pd.DataFrame(rows, columns=CONFUSION_COLUMNS)
    validate_required_columns(df_confusion, CONFUSION_COLUMNS)
    return df_confusion


def summarize_class_recall(df_trials: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df_trials, ["true_label", "is_correct"])
    overall_accuracy = float(df_trials["is_correct"].mean())

    rows: List[Dict[str, float]] = []
    for label in sorted(int(x) for x in pd.unique(df_trials["true_label"])):
        sub = df_trials[df_trials["true_label"] == label]
        rows.append(
            {
                "label": int(label),
                "recall": float(sub["is_correct"].mean()),
                "n_trials": int(len(sub)),
                "overall_accuracy": overall_accuracy,
            }
        )

    df_recall = pd.DataFrame(rows, columns=RECALL_COLUMNS)
    validate_required_columns(df_recall, RECALL_COLUMNS)
    return df_recall

def build_metrics_summary(df_trials: pd.DataFrame, df_recall: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df_trials, ["is_correct", "is_silent"])
    validate_required_columns(df_recall, RECALL_COLUMNS)

    rows: List[Dict[str, float]] = [
        {
            "section": "overall",
            "group": "all_trials",
            "metric": "accuracy",
            "value": float(df_trials["is_correct"].mean()),
        },
        {
            "section": "overall",
            "group": "all_trials",
            "metric": "silent_rate",
            "value": float(df_trials["is_silent"].mean()),
        },
        {
            "section": "overall",
            "group": "all_trials",
            "metric": "n_trials",
            "value": float(len(df_trials)),
        },
    ]
    for _, row in df_recall.iterrows():
        rows.append(
            {
                "section": "class_recall",
                "group": f"class_{int(row['label'])}",
                "metric": "recall",
                "value": float(row["recall"]),
            }
        )
    return pd.DataFrame(rows)


def _pred_tick_labels(pred_labels: Sequence[int]) -> List[str]:
    return ["silent" if int(label) == -1 else str(int(label)) for label in pred_labels]


def plot_confusion_matrix_main(
    ax: plt.Axes,
    df_confusion: pd.DataFrame,
    num_classes: int,
) -> None:
    validate_required_columns(df_confusion, CONFUSION_COLUMNS)

    pred_labels = list(range(num_classes))
    if (df_confusion["pred_label"] == -1).any():
        pred_labels.append(-1)

    matrix = (
        df_confusion.pivot(index="true_label", columns="pred_label", values="fraction_row_normalized")
        .reindex(index=range(num_classes), columns=pred_labels, fill_value=0.0)
        .to_numpy(dtype=np.float64, copy=False)
    )

    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(pred_labels)))
    ax.set_xticklabels(_pred_tick_labels(pred_labels))
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    ax.tick_params(axis="x", rotation=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            if value <= 0.025:
                continue
            ax.text(
                j,
                i,
                f"{value * 100.0:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value >= 0.6 else "#222222",
            )

    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row fraction")
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    cbar.ax.tick_params(length=3, pad=3)


def plot_class_recall_summary(
    ax: plt.Axes,
    df_recall: pd.DataFrame,
    n_trials: int,
) -> None:
    validate_required_columns(df_recall, RECALL_COLUMNS)
    df_sorted = df_recall.sort_values("label", kind="stable").reset_index(drop=True)

    labels = df_sorted["label"].astype(int).tolist()
    recall = df_sorted["recall"].to_numpy(dtype=np.float64, copy=False)
    overall_accuracy = float(df_sorted["overall_accuracy"].iloc[0])

    bars = ax.bar(labels, recall, color=COLOR_STATIC, edgecolor="#222222", linewidth=0.8, width=0.72)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(labels)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    for bar, value in zip(bars, recall):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.03, value + 0.02),
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
            color="#222222",
        )
    ax.text(
        1.01,
        1.035,
        f"overall acc. {overall_accuracy * 100.0:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PUBLICATION_ANNOTATION_FONT_SIZE,
        color="#222222",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "#444444", "boxstyle": "round,pad=0.22", "alpha": 0.92},
    )

def create_figure_main(
    df_confusion: pd.DataFrame,
    df_recall: pd.DataFrame,
    n_trials: int,
    num_classes: int,
) -> plt.Figure:
    apply_paper_style()
    fig = plt.figure(figsize=PUBLICATION_TWO_COLUMN_FIGSIZE)
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.10, 1.00])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    plot_confusion_matrix_main(ax_left, df_confusion=df_confusion, num_classes=num_classes)
    plot_class_recall_summary(ax_right, df_recall=df_recall, n_trials=n_trials)
    for label, axis in [("A", ax_left), ("B", ax_right)]:
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    fig.subplots_adjust(**DEFAULT_SUBPLOT_ADJUST)
    return fig


def compute_reference_style_accuracy(
    net: SDNN_Network,
    encoder: DoGSpikeEncoder,
    dataset_root: str,
    batch_size: int,
    input_size: int,
    device: torch.device,
) -> float:
    _, _, test_loader = build_mnist_skeleton_loader(
        root=dataset_root,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=0,
    )

    original_noise = float(net.layer3.current_noise_std.item())
    correct = 0
    total = 0

    try:
        net.layer3.current_noise_std.fill_(0.0)
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Reference accuracy", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                spike_train = encoder.forward(images)
                _ = net(spike_train, layer_idx=3, labels=None, monitor=False)

                firing_times = net.layer3.firing_times
                _, min_indices = torch.min(firing_times, dim=1)
                predicted_class = (min_indices // net.layer3.neurons_per_class).long()

                correct += int((predicted_class == labels).sum().item())
                total += int(labels.size(0))
    finally:
        net.layer3.current_noise_std.fill_(original_noise)

    if total <= 0:
        raise ValueError("Reference evaluation saw zero samples")
    return float(correct / total)


def validate_trial_level(df_trials: pd.DataFrame) -> None:
    if list(df_trials.columns) != TRIAL_COLUMNS:
        raise ValueError(f"Unexpected trial_level.csv columns: {list(df_trials.columns)}")
    if not np.array_equal(df_trials["trial_id"].to_numpy(), np.arange(len(df_trials), dtype=np.int64)):
        raise ValueError("trial_id is not a contiguous sequence starting at 0")
    if not np.array_equal(df_trials["image_index"].to_numpy(), np.arange(len(df_trials), dtype=np.int64)):
        raise ValueError("image_index is not aligned with dataset order")

    silent_mask = df_trials["is_silent"] == 1
    if not (df_trials.loc[silent_mask, "first_fire_t"] == -1).all():
        raise ValueError("Silent trials must have first_fire_t == -1")

    expected_correct = (
        (df_trials["pred_label"] == df_trials["true_label"]) & (df_trials["is_silent"] == 0)
    ).astype(np.int64)
    if not np.array_equal(df_trials["is_correct"].to_numpy(dtype=np.int64), expected_correct.to_numpy(dtype=np.int64)):
        raise ValueError("is_correct does not match prediction/label equality")


def validate_confusion_matrix(df_trials: pd.DataFrame, df_confusion: pd.DataFrame) -> None:
    if list(df_confusion.columns) != CONFUSION_COLUMNS:
        raise ValueError(f"Unexpected metrics_confusion_matrix.csv columns: {list(df_confusion.columns)}")

    totals = df_trials.groupby("true_label").size().to_dict()
    for true_label, sub in df_confusion.groupby("true_label", sort=False):
        row_count_sum = int(sub["count"].sum())
        expected_total = int(totals[int(true_label)])
        if row_count_sum != expected_total:
            raise ValueError(f"Confusion count sum mismatch for true_label={true_label}: {row_count_sum} != {expected_total}")
        row_fraction_sum = float(sub["fraction_row_normalized"].sum())
        if not np.isclose(row_fraction_sum, 1.0, atol=1e-9):
            raise ValueError(
                f"Confusion row fractions must sum to 1.0 for true_label={true_label}; got {row_fraction_sum:.12f}"
            )


def validate_class_recall(df_trials: pd.DataFrame, df_recall: pd.DataFrame) -> None:
    if list(df_recall.columns) != RECALL_COLUMNS:
        raise ValueError(f"Unexpected metrics_class_recall.csv columns: {list(df_recall.columns)}")

    if int(df_recall["n_trials"].sum()) != len(df_trials):
        raise ValueError("metrics_class_recall.csv n_trials sum does not match trial count")

    overall_accuracy = float(df_trials["is_correct"].mean())
    if not np.allclose(df_recall["overall_accuracy"].to_numpy(dtype=np.float64), overall_accuracy, atol=1e-12):
        raise ValueError("overall_accuracy column is not constant or does not match trial-level accuracy")


def validate_figure_text(fig: plt.Figure) -> None:
    texts: List[str] = []
    for ax in fig.axes:
        texts.append(ax.get_title())
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(text.get_text() for text in legend.get_texts())
        texts.extend(text.get_text() for text in ax.texts)
    texts.append(fig._suptitle.get_text() if fig._suptitle is not None else "")

    joined = " ".join(texts).lower()
    for term in FORBIDDEN_FIGURE_TERMS:
        if term in joined:
            raise ValueError(f"Forbidden figure term detected: {term}")


def validate_output_directory(save_dir: Path) -> List[str]:
    expected = {
        "trial_level.csv",
        "metrics_summary.csv",
        "metrics_confusion_matrix.csv",
        "metrics_class_recall.csv",
        "figure_main.png",
        "figure_main.pdf",
        "figure_main.svg",
        "run_config.json",
    }
    actual = {path.name for path in save_dir.iterdir() if path.is_file()}
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Missing expected output files: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected extra files in output directory: {unexpected}")
    return sorted(actual)


seed_everything = shared_seed_everything
decode_prediction_and_fire_time_from_layer3 = shared_decode_prediction_and_fire_time_from_layer3


def compensate_stsp_gain(net: SDNN_Network, scaling_factor: float) -> None:
    shared_compensate_stsp_gain(net, scaling_factor=scaling_factor, include_target_norm=True)


def load_model_and_encoder(
    model_path: str,
    device: torch.device,
    dt_ms: float,
    max_duration_ms: float,
) -> Tuple[SDNN_Network, DoGSpikeEncoder]:
    return shared_load_model_and_encoder(
        model_path=model_path,
        device=device,
        dt=dt_ms * ms,
        max_duration_ms=max_duration_ms,
        include_target_norm=True,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_classes <= 0:
        raise ValueError("num-classes must be positive")

    seed_everything(int(args.seed))
    apply_paper_style()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, encoder = load_model_and_encoder(
        model_path=args.model_path,
        device=device,
        dt_ms=float(args.dt_ms),
        max_duration_ms=float(args.max_duration_ms),
    )

    df_trials = run_baseline_classification(
        net=net,
        encoder=encoder,
        dataset_root=args.dataset_root,
        batch_size=int(args.batch_size),
        input_size=int(args.input_size),
        device=device,
    )
    df_confusion = build_confusion_matrix(df_trials)
    df_recall = summarize_class_recall(df_trials)
    df_metrics_summary = build_metrics_summary(df_trials, df_recall)

    trial_csv = save_tidy_csv(df_trials[TRIAL_COLUMNS], save_dir / "trial_level.csv", sort_by="trial_id")
    metrics_summary_csv = save_tidy_csv(
        df_metrics_summary,
        save_dir / "metrics_summary.csv",
        sort_by=["section", "group", "metric"],
    )
    confusion_csv = save_tidy_csv(
        df_confusion[CONFUSION_COLUMNS],
        save_dir / "metrics_confusion_matrix.csv",
        sort_by=["true_label", "pred_label"],
    )
    recall_csv = save_tidy_csv(
        df_recall[RECALL_COLUMNS],
        save_dir / "metrics_class_recall.csv",
        sort_by="label",
    )

    fig = create_figure_main(
        df_confusion=df_confusion,
        df_recall=df_recall,
        n_trials=len(df_trials),
        num_classes=int(args.num_classes),
    )
    figure_paths = save_figure_all_formats(fig, save_dir / "figure_main")
    validate_figure_text(fig)
    plt.close(fig)

    reference_accuracy = compute_reference_style_accuracy(
        net=net,
        encoder=encoder,
        dataset_root=args.dataset_root,
        batch_size=int(args.batch_size),
        input_size=int(args.input_size),
        device=device,
    )
    overall_accuracy = float(df_trials["is_correct"].mean())
    if not np.isclose(reference_accuracy, overall_accuracy, atol=1e-12):
        raise ValueError(
            f"Reference accuracy mismatch: baseline={overall_accuracy:.12f}, reference={reference_accuracy:.12f}"
        )

    validate_trial_level(df_trials)
    validate_confusion_matrix(df_trials, df_confusion)
    validate_class_recall(df_trials, df_recall)

    run_config = {
        "figure_id": "Figure 1",
        "figure_script": "plot_fig1_baseline_processing.py",
        "figure_purpose": "Baseline single-image sensory classification for the main text figure.",
        "model_path": args.model_path,
        "dataset_root": args.dataset_root,
        "save_dir": str(save_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "input_size": int(args.input_size),
        "num_classes": int(args.num_classes),
        "dt_ms": float(args.dt_ms),
        "max_duration_ms": float(args.max_duration_ms),
        "seed": int(args.seed),
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "reference_accuracy_main_style": reference_accuracy,
            "n_trials": int(len(df_trials)),
            "n_silent_trials": int(df_trials["is_silent"].sum()),
        },
        "outputs": {
            "trial_level_csv": trial_csv,
            "metrics_summary_csv": metrics_summary_csv,
            "metrics_confusion_matrix_csv": confusion_csv,
            "metrics_class_recall_csv": recall_csv,
            "figure_main": figure_paths,
        },
        "validation": {
            "forbidden_terms_checked": list(FORBIDDEN_FIGURE_TERMS),
        },
    }
    save_run_config(run_config, save_dir)
    output_files = validate_output_directory(save_dir)
    run_config["validation"]["output_files"] = output_files
    save_run_config(run_config, save_dir)

    print(f"[Done] Saved baseline Figure 1 outputs to {save_dir}")
    print(f"[Done] Overall accuracy: {overall_accuracy:.4%}")
    print(f"[Done] Silent trials: {int(df_trials['is_silent'].sum())}")


if __name__ == "__main__":
    main()
