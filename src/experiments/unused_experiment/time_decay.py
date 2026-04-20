import json
import logging
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.optimize import curve_fit

from src.platform.legacy_adapters.encoding import DoGSpikeEncoder, build_mnist_skeleton_loader
from src.platform.legacy_adapters.network import SDNN_Network
from src.platform.legacy_adapters.units import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("results", "sdnn_deep_final", "net_final.pth")
EXPERIMENT_NAME = "time_decay_experiment_ami"
RESULTS_ROOT = "results"

NUM_PROBE_PAIRS = 300
NUM_CLASSES = 10
DT = 1.0 * ms

DELAY_POINTS_MS = [100, 200, 300, 400, 500, 600, 1000, 1500, 2000]

SAMPLE_DURATION_MS = 200 * ms
TEST_DURATION_MS = 100 * ms

SAMPLE_STEPS = int(SAMPLE_DURATION_MS / DT)
TEST_STEPS = int(TEST_DURATION_MS / DT)


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
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def initialize_output_dirs(experiment_name):
    root_dir = os.path.join(RESULTS_ROOT, experiment_name)
    output_dirs = {
        "root": root_dir,
        "figure": os.path.join(root_dir, "figure"),
        "log": os.path.join(root_dir, "log"),
        "data": os.path.join(root_dir, "data"),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)
    return output_dirs


def setup_logger(log_dir, experiment_name):
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


def save_json(data, path, logger=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)
    if logger is not None:
        logger.info("[Save] JSON saved to %s", path)


def save_figure_multi_format(fig, figure_dir, basename, logger):
    saved_paths = []
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(figure_dir, f"{basename}.{ext}")
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
        logger.info("[Save] Figure saved to %s", path)
    return saved_paths


def compensate_stsp_gain(net, scaling_factor=5.0, logger=None):
    message = f"[Info] Executing Gain Compensation: Scale = {scaling_factor:.4f}x"
    if logger is not None:
        logger.info(message)
    else:
        print(message)

    with torch.no_grad():
        if hasattr(net, "layer1"):
            net.layer1.kernels.data *= scaling_factor
        if hasattr(net, "layer2"):
            net.layer2.kernels.data *= scaling_factor
        if hasattr(net, "layer3"):
            net.layer3.kernels.data *= scaling_factor


def generate_probe_batch_tensor(dataset, num_pairs, logger):
    logger.info("[Data] Generating tensor batch for %d probe pairs...", num_pairs)

    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    list_img_sample = []
    list_img_test = []
    list_lbl_sample = []
    list_lbl_test = []

    for _ in range(num_pairs):
        cls_sample = random.randint(0, NUM_CLASSES - 1)
        possible_test = list(range(NUM_CLASSES))
        possible_test.remove(cls_sample)
        cls_test = random.choice(possible_test)

        idx_sample = random.choice(class_indices[cls_sample])
        idx_test = random.choice(class_indices[cls_test])

        img_s, lbl_s = dataset[idx_sample]
        img_t, lbl_t = dataset[idx_test]

        list_img_sample.append(img_s)
        list_img_test.append(img_t)
        list_lbl_sample.append(lbl_s)
        list_lbl_test.append(lbl_t)

    batch_img_sample = torch.stack(list_img_sample).to(DEVICE)
    batch_img_test = torch.stack(list_img_test).to(DEVICE)
    batch_lbl_sample = torch.tensor(list_lbl_sample).to(DEVICE)
    batch_lbl_test = torch.tensor(list_lbl_test).to(DEVICE)

    return batch_img_sample, batch_img_test, batch_lbl_sample, batch_lbl_test


def pre_encode_spikes(encoder, batch_img, steps, logger, tag):
    logger.info("[Data] Pre-encoding %s spikes (Batch=%d, Steps=%d)...", tag, batch_img.shape[0], steps)
    with torch.no_grad():
        full_spikes = encoder.forward(batch_img)
        return full_spikes[:, :steps, ...].contiguous()


def run_decay_session_ami(net, spikes_sample, spikes_test, lbl_sample, lbl_test, delay_points, logger):
    metrics = {
        "delay_ms": [],
        "acc_static": [],
        "acc_dynamic": [],
        "err_total_static": [],
        "err_total_dynamic": [],
        "ami": [],
        "random_error_dyn": [],
    }

    total_samples = lbl_sample.shape[0]
    logger.info("[Exp] Starting AMI analysis session on %d samples.", total_samples)

    start_time = time.time()

    for delay_ms in delay_points:
        delay_steps = int((delay_ms * ms) / DT)

        with torch.no_grad():
            res_s = net.forward_classify_session(
                spikes_sample, spikes_test, delay_steps, stsp_mode="static_frozen"
            )
        pred_s = res_s["prediction"]

        with torch.no_grad():
            res_d = net.forward_classify_session(
                spikes_sample, spikes_test, delay_steps, stsp_mode="dynamic"
            )
        pred_d = res_d["prediction"]

        num_correct_s = (pred_s == lbl_test).sum().item()
        num_correct_d = (pred_d == lbl_test).sum().item()

        num_hallu_s = (pred_s == lbl_sample).sum().item()
        num_hallu_d = (pred_d == lbl_sample).sum().item()

        acc_s = num_correct_s / total_samples * 100.0
        acc_d = num_correct_d / total_samples * 100.0

        rate_err_s = 100.0 - acc_s
        rate_err_d = 100.0 - acc_d

        abs_hallu_rate_s = num_hallu_s / total_samples * 100.0
        abs_hallu_rate_d = num_hallu_d / total_samples * 100.0

        ami = abs_hallu_rate_d - abs_hallu_rate_s
        random_error_dyn = rate_err_d - ami
        acc_drop = acc_s - acc_d

        metrics["delay_ms"].append(delay_ms)
        metrics["acc_static"].append(acc_s)
        metrics["acc_dynamic"].append(acc_d)
        metrics["err_total_static"].append(rate_err_s)
        metrics["err_total_dynamic"].append(rate_err_d)
        metrics["ami"].append(ami)
        metrics["random_error_dyn"].append(random_error_dyn)

        logger.info(
            "[%dms] Acc_Drop: %.1f%% | AMI: %.1f%% | Rnd: %.1f%% | Acc_Static: %.1f%% | Acc_Dynamic: %.1f%%",
            delay_ms,
            acc_drop,
            ami,
            random_error_dyn,
            acc_s,
            acc_d,
        )

    total_time = time.time() - start_time
    logger.info("[Exp] All done. Total time: %.2fs", total_time)
    return metrics


def exponential_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def exponential_decay_zero(t, A, tau):
    return A * np.exp(-t / tau)


def fit_accuracy_drop(delays, acc_drop, logger):
    fit_result = {
        "success": False,
        "model": "A*exp(-t/tau)+C",
        "params": {},
        "tau_ms": None,
        "offset": None,
    }

    try:
        p0 = [float(np.max(acc_drop)), 1500.0, 0.0]
        popt, _ = curve_fit(exponential_decay, delays, acc_drop, p0=p0, maxfev=5000)
        fit_result.update(
            {
                "success": True,
                "params": {
                    "A": float(popt[0]),
                    "tau_ms": float(popt[1]),
                    "C": float(popt[2]),
                },
                "tau_ms": float(popt[1]),
                "offset": float(popt[2]),
            }
        )
        logger.info("[Fit] Accuracy drop fit success: tau=%.2f ms, offset=%.4f", popt[1], popt[2])
    except Exception as exc:
        fit_result["error"] = str(exc)
        logger.info("[Fit] Accuracy drop fit failed: %s", exc)

    return fit_result


def fit_ami_decay(delays, ami, logger):
    fit_result = {
        "success": False,
        "model": "A*exp(-t/tau)",
        "params": {},
        "tau_ms": None,
    }

    try:
        p0 = [float(np.max(ami)), 1500.0]
        popt, _ = curve_fit(exponential_decay_zero, delays, ami, p0=p0, maxfev=5000)
        fit_result.update(
            {
                "success": True,
                "params": {
                    "A": float(popt[0]),
                    "tau_ms": float(popt[1]),
                },
                "tau_ms": float(popt[1]),
            }
        )
        logger.info("[Fit] AMI decay fit success: tau=%.2f ms", popt[1])
    except Exception as exc:
        fit_result["error"] = str(exc)
        logger.info("[Fit] AMI decay fit failed: %s", exc)

    return fit_result


def plot_accuracy_drop(metrics, figure_dir, fit_result, logger):
    delays = np.asarray(metrics["delay_ms"], dtype=float)
    acc_static = np.asarray(metrics["acc_static"], dtype=float)
    acc_dynamic = np.asarray(metrics["acc_dynamic"], dtype=float)
    acc_drop = acc_static - acc_dynamic

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(delays, acc_drop, "s-", color="#ff7f0e", linewidth=2.5, markersize=8, label="Accuracy Drop")

    if fit_result["success"]:
        t_smooth = np.linspace(delays.min(), delays.max(), 200)
        params = fit_result["params"]
        y_fit = exponential_decay(
            t_smooth,
            params["A"],
            params["tau_ms"],
            params["C"],
        )
        ax.plot(
            t_smooth,
            y_fit,
            "k:",
            alpha=0.6,
            label=f"Fit ($\\tau \\approx {fit_result['tau_ms']:.0f}$ ms)",
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="-")

    max_drop_idx = int(np.argmax(acc_drop))
    ax.text(
        delays[max_drop_idx],
        acc_drop[max_drop_idx] + 0.5,
        f"Max Drop: {acc_drop[max_drop_idx]:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

    ax.set_title("Net Impact of STSP on Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Delay Duration (ms)", fontsize=12)
    ax.set_ylabel("Accuracy Drop (Static - Dynamic) (pp)", fontsize=12)
    ax.legend()
    fig.tight_layout()

    save_figure_multi_format(fig, figure_dir, "accuracy_drop", logger)
    plt.close(fig)


def plot_ami_decay(metrics, figure_dir, fit_result, logger):
    delays = np.asarray(metrics["delay_ms"], dtype=float)
    ami = np.asarray(metrics["ami"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(delays, ami, "o-", color="#d62728", linewidth=2, markersize=8, label="AMI (Memory Strength)")

    if fit_result["success"]:
        t_smooth = np.linspace(delays.min(), delays.max(), 200)
        params = fit_result["params"]
        y_fit = exponential_decay_zero(
            t_smooth,
            params["A"],
            params["tau_ms"],
        )
        ax.plot(
            t_smooth,
            y_fit,
            "k--",
            alpha=0.6,
            label=f"Fit ($\\tau \\approx {fit_result['tau_ms']:.0f}$ ms)",
        )

    ax.set_title("Decay of Absolute Memory Induction (AMI)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Delay Duration (ms)", fontsize=12)
    ax.set_ylabel("AMI (% of Total Samples)", fontsize=12)
    ax.set_ylim(bottom=-1.0)
    ax.legend()
    fig.tight_layout()

    save_figure_multi_format(fig, figure_dir, "ami_decay", logger)
    plt.close(fig)


def plot_error_composition(metrics, figure_dir, logger):
    delays = np.asarray(metrics["delay_ms"], dtype=float)
    ami = np.asarray(metrics["ami"], dtype=float)
    random_error_dyn = np.asarray(metrics["random_error_dyn"], dtype=float)
    err_total_static = np.asarray(metrics["err_total_static"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    width = float(delays[-1] * 0.08)

    ax.bar(
        delays,
        random_error_dyn,
        width=width,
        color="#bdbdbd",
        edgecolor="black",
        alpha=0.8,
        label="Random Noise Error",
    )
    ax.bar(
        delays,
        ami,
        width=width,
        bottom=random_error_dyn,
        color="#d62728",
        edgecolor="black",
        alpha=0.9,
        label="Memory Induced Error (AMI)",
    )
    ax.plot(delays, err_total_static, "b--", linewidth=2, label="Baseline Error (Static)")

    ax.set_title("Decomposition of Classification Errors", fontsize=14, fontweight="bold")
    ax.set_xlabel("Delay Duration (ms)", fontsize=12)
    ax.set_ylabel("Error Rate (% of Total Samples)", fontsize=12)
    ax.legend(loc="upper right")
    fig.tight_layout()

    save_figure_multi_format(fig, figure_dir, "error_composition", logger)
    plt.close(fig)


def save_metrics_by_delay(metrics, data_dir, logger):
    df = pd.DataFrame(metrics)
    df["acc_drop"] = df["acc_static"] - df["acc_dynamic"]
    df = df[
        [
            "delay_ms",
            "acc_static",
            "acc_dynamic",
            "acc_drop",
            "err_total_static",
            "err_total_dynamic",
            "ami",
            "random_error_dyn",
        ]
    ]

    csv_path = os.path.join(data_dir, "metrics_by_delay.csv")
    df.to_csv(csv_path, index=False)
    logger.info("[Save] Metrics CSV saved to %s", csv_path)
    return df


def build_summary(experiment_name, num_probe_pairs, metrics, fit_results):
    delays = np.asarray(metrics["delay_ms"], dtype=float)
    ami = np.asarray(metrics["ami"], dtype=float)
    acc_static = np.asarray(metrics["acc_static"], dtype=float)
    acc_dynamic = np.asarray(metrics["acc_dynamic"], dtype=float)
    acc_drop = acc_static - acc_dynamic

    max_acc_drop_idx = int(np.argmax(acc_drop))
    min_acc_drop_idx = int(np.argmin(acc_drop))
    max_ami_idx = int(np.argmax(ami))

    acc_drop_change_short_to_long = float(acc_drop[-1] - acc_drop[0])
    strongest_region = "short delays" if max_acc_drop_idx <= len(delays) // 2 else "long delays"

    if acc_drop_change_short_to_long < 0:
        acc_drop_trend = "decreases with delay"
    elif acc_drop_change_short_to_long > 0:
        acc_drop_trend = "increases with delay"
    else:
        acc_drop_trend = "remains stable across delays"

    summary_text = (
        f"Dynamic-memory effect {acc_drop_trend}, and the strongest net accuracy drop appears at "
        f"{strongest_region}."
    )

    return {
        "experiment_name": experiment_name,
        "num_probe_pairs": int(num_probe_pairs),
        "num_delays": int(len(delays)),
        "max_acc_drop": float(acc_drop[max_acc_drop_idx]),
        "max_acc_drop_delay_ms": float(delays[max_acc_drop_idx]),
        "min_acc_drop": float(acc_drop[min_acc_drop_idx]),
        "min_acc_drop_delay_ms": float(delays[min_acc_drop_idx]),
        "acc_drop_at_shortest_delay": float(acc_drop[0]),
        "acc_drop_at_longest_delay": float(acc_drop[-1]),
        "acc_drop_change_short_to_long": acc_drop_change_short_to_long,
        "accuracy_drop_fit_success": bool(fit_results["accuracy_drop_fit"]["success"]),
        "accuracy_drop_tau_ms": fit_results["accuracy_drop_fit"].get("tau_ms"),
        "max_ami": float(ami[max_ami_idx]),
        "max_ami_delay_ms": float(delays[max_ami_idx]),
        "ami_at_longest_delay": float(ami[-1]),
        "ami_fit_success": bool(fit_results["ami_decay_fit"]["success"]),
        "ami_tau_ms": fit_results["ami_decay_fit"].get("tau_ms"),
        "summary_text": summary_text,
    }


def main():
    sns.set(style="whitegrid", font_scale=1.1)

    output_dirs = initialize_output_dirs(EXPERIMENT_NAME)
    logger = setup_logger(output_dirs["log"], EXPERIMENT_NAME)

    logger.info("[Init] Run started at %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("[Init] Device: %s", DEVICE)
    logger.info("[Init] Model path: %s", MODEL_PATH)
    logger.info("[Init] Delay points (ms): %s", DELAY_POINTS_MS)
    logger.info("[Init] Num probe pairs: %d", NUM_PROBE_PAIRS)

    net = SDNN_Network(device=DEVICE).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        logger.info("[Error] Model not found: %s", MODEL_PATH)
        return

    stsp_gain_compensation_factor = 1.0 / float(net.layer3.stsp_U)
    compensate_stsp_gain(net, scaling_factor=stsp_gain_compensation_factor, logger=logger)
    net.eval()

    run_config = {
        "experiment_name": EXPERIMENT_NAME,
        "model_path": MODEL_PATH,
        "device": str(DEVICE),
        "num_probe_pairs": NUM_PROBE_PAIRS,
        "num_classes": NUM_CLASSES,
        "dt_ms": float(DT / ms),
        "delay_points_ms": DELAY_POINTS_MS,
        "sample_duration_ms": float(SAMPLE_DURATION_MS / ms),
        "test_duration_ms": float(TEST_DURATION_MS / ms),
        "sample_steps": SAMPLE_STEPS,
        "test_steps": TEST_STEPS,
        "stsp_gain_compensation_factor": float(stsp_gain_compensation_factor),
    }
    save_json(run_config, os.path.join(output_dirs["root"], "run_config.json"), logger)

    _, _, test_loader = build_mnist_skeleton_loader(batch_size=1)
    dataset = test_loader.dataset
    encoder = DoGSpikeEncoder(dt=DT, max_duration=SAMPLE_DURATION_MS, device=DEVICE)

    batch_img_sample, batch_img_test, batch_lbl_sample, batch_lbl_test = generate_probe_batch_tensor(
        dataset, NUM_PROBE_PAIRS, logger
    )

    spikes_sample = pre_encode_spikes(encoder, batch_img_sample, SAMPLE_STEPS, logger, "sample")
    spikes_test = pre_encode_spikes(encoder, batch_img_test, TEST_STEPS, logger, "test")

    metrics = run_decay_session_ami(
        net,
        spikes_sample,
        spikes_test,
        batch_lbl_sample,
        batch_lbl_test,
        DELAY_POINTS_MS,
        logger,
    )

    delays = np.asarray(metrics["delay_ms"], dtype=float)
    ami = np.asarray(metrics["ami"], dtype=float)
    acc_drop = np.asarray(metrics["acc_static"], dtype=float) - np.asarray(metrics["acc_dynamic"], dtype=float)

    fit_results = {
        "accuracy_drop_fit": fit_accuracy_drop(delays, acc_drop, logger),
        "ami_decay_fit": fit_ami_decay(delays, ami, logger),
    }
    save_json(fit_results, os.path.join(output_dirs["data"], "fit_results.json"), logger)

    save_metrics_by_delay(metrics, output_dirs["data"], logger)

    plot_accuracy_drop(metrics, output_dirs["figure"], fit_results["accuracy_drop_fit"], logger)
    plot_ami_decay(metrics, output_dirs["figure"], fit_results["ami_decay_fit"], logger)
    plot_error_composition(metrics, output_dirs["figure"], logger)

    summary = build_summary(EXPERIMENT_NAME, NUM_PROBE_PAIRS, metrics, fit_results)
    save_json(summary, os.path.join(output_dirs["root"], "summary.json"), logger)

    logger.info("[Done] Root output directory: %s", output_dirs["root"])
    logger.info("[Done] Figure directory: %s", output_dirs["figure"])
    logger.info("[Done] Data directory: %s", output_dirs["data"])
    logger.info("[Done] Log file: %s", os.path.join(output_dirs["log"], "run.log"))


if __name__ == "__main__":
    main()
