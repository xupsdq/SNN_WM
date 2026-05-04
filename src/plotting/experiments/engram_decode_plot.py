from __future__ import annotations

import matplotlib.pyplot as plt

from src.plotting.common.colors import get_plot_color
from src.plotting.experiments._common import main_for, read_bundle_csv
from src.plotting.experiments._plot_builders import color_for


def plot_bundle(input_dir):
    metrics = read_bundle_csv(input_dir, "engram_decode_metrics.csv", ["layer", "delay_ms", "acc"])
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for layer_name, part in metrics.groupby("layer", sort=True):
        part = part.sort_values("delay_ms")
        x = part["delay_ms"].to_numpy(dtype=float)
        y = part["acc"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=str(layer_name), color=color_for(layer_name))
        if {"acc_ci_low", "acc_ci_high"}.issubset(part.columns):
            ax.fill_between(
                x,
                part["acc_ci_low"].to_numpy(dtype=float),
                part["acc_ci_high"].to_numpy(dtype=float),
                color=color_for(layer_name),
                alpha=0.16,
            )
    ax.axhline(0.10, color=get_plot_color("other_residual"), linestyle="--", linewidth=1.2, label="Chance (10%)")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_title("Accuracy vs Delay")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return {"accuracy_vs_delay": fig}


if __name__ == "__main__":
    raise SystemExit(main_for("engram_decode", plot_bundle, title="Engram Decode"))
