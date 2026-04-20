import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text

from figure_utils_common import (
    COLOR_DYNAMIC,
    COLOR_STATIC,
    PUBLICATION_ANNOTATION_FONT_SIZE,
    apply_publication_style,
    save_figure_all_formats,
)


PHASE_SAMPLE = "#F0B3AC"
PHASE_DELAY = "#E5E5E5"
PHASE_PROBE = "#BFD9F2"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#444444"
LIGHT_EDGE = "#B9B9B9"
PANEL_BG = "#FCFCFC"


def _promote_text_sizes(fig: plt.Figure) -> None:
    for artist in fig.findobj(match=lambda obj: isinstance(obj, Text)):
        artist.set_fontsize(max(float(artist.get_fontsize()), PUBLICATION_ANNOTATION_FONT_SIZE))


def add_panel_header(ax: Axes, letter: str, title: str, subtitle: str) -> None:
    ax.text(
        0.015,
        0.97,
        letter,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )
    ax.text(
        0.065,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )
    ax.text(
        0.065,
        0.90,
        subtitle,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#555555",
        transform=ax.transAxes,
    )


def draw_phase_bar(ax: Axes, x: float, y: float, width: float, height: float) -> dict[str, float]:
    sample_w = width * 0.30
    delay_w = width * 0.42
    probe_w = width - sample_w - delay_w

    x_sample = x
    x_delay = x + sample_w
    x_probe = x + sample_w + delay_w

    for x0, w0, color in (
        (x_sample, sample_w, PHASE_SAMPLE),
        (x_delay, delay_w, PHASE_DELAY),
        (x_probe, probe_w, PHASE_PROBE),
    ):
        ax.add_patch(
            Rectangle(
                (x0, y),
                w0,
                height,
                facecolor=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
            )
        )

    ax.plot([x_delay, x_delay], [y, y + height], color=EDGE_COLOR, linewidth=1.1)
    ax.plot([x_probe, x_probe], [y, y + height], color=EDGE_COLOR, linewidth=1.1)

    return {
        "sample_start": x_sample,
        "sample_center": x_sample + sample_w * 0.5,
        "delay_start": x_delay,
        "delay_center": x_delay + delay_w * 0.5,
        "probe_start": x_probe,
        "probe_center": x_probe + probe_w * 0.5,
        "end": x + width,
        "y": y,
        "height": height,
    }


def draw_flow_arrow(ax: Axes, x0: float, x1: float, y: float, color: str = EDGE_COLOR, lw: float = 1.8) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_digit_icon(ax: Axes, center_x: float, center_y: float, digit: str) -> None:
    size = 0.075
    ax.add_patch(
        FancyBboxPatch(
            (center_x - size / 2, center_y - size / 2),
            size,
            size,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="white",
            edgecolor=EDGE_COLOR,
            linewidth=0.9,
        )
    )
    ax.text(
        center_x,
        center_y,
        digit,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def draw_no_input_icon(ax: Axes, center_x: float, center_y: float) -> None:
    radius = 0.035
    ax.add_patch(Circle((center_x, center_y), radius, facecolor="white", edgecolor=LIGHT_EDGE, linewidth=1.0))
    ax.plot(
        [center_x - radius * 0.75, center_x + radius * 0.75],
        [center_y + radius * 0.75, center_y - radius * 0.75],
        color=LIGHT_EDGE,
        linewidth=1.2,
    )


def draw_decision_icon(ax: Axes, center_x: float, center_y: float) -> None:
    width = 0.125
    height = 0.075
    ax.add_patch(
        FancyBboxPatch(
            (center_x - width / 2, center_y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="white",
            edgecolor=EDGE_COLOR,
            linewidth=0.9,
        )
    )
    ax.text(center_x, center_y + 0.012, "match?", ha="center", va="center", fontsize=8.3, color=TEXT_COLOR)
    ax.text(center_x, center_y - 0.014, "yes / no", ha="center", va="center", fontsize=7.2, color="#666666")


def draw_task_timeline(ax: Axes) -> None:
    add_panel_header(
        ax,
        "A",
        "Task Timeline",
        "Sample, blank delay, and probe are separated in time to test working memory.",
    )
    phase = draw_phase_bar(ax, x=0.08, y=0.48, width=0.84, height=0.16)

    label_y = 0.73
    ax.text(phase["sample_center"], label_y, "Sample", ha="center", va="bottom", fontsize=10, fontweight="semibold")
    ax.text(
        phase["sample_center"],
        label_y - 0.06,
        "(input image)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    ax.text(phase["delay_center"], label_y, "Delay", ha="center", va="bottom", fontsize=10, fontweight="semibold")
    ax.text(
        phase["delay_center"],
        label_y - 0.06,
        "(memory period)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    ax.text(phase["probe_center"], label_y, "Probe", ha="center", va="bottom", fontsize=10, fontweight="semibold")
    ax.text(
        phase["probe_center"],
        label_y - 0.06,
        "(decision)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    icon_y = 0.32
    draw_digit_icon(ax, phase["sample_center"], icon_y + 0.03, digit="3")
    ax.text(phase["sample_center"], icon_y - 0.05, "MNIST stimulus", ha="center", va="top", fontsize=8.5)

    draw_no_input_icon(ax, phase["delay_center"], icon_y + 0.03)
    ax.text(phase["delay_center"], icon_y - 0.05, "no external input", ha="center", va="top", fontsize=8.5)

    draw_decision_icon(ax, phase["probe_center"], icon_y + 0.03)
    ax.text(phase["probe_center"], icon_y - 0.05, "network must respond", ha="center", va="top", fontsize=8.5)

    arrow_y = 0.16
    draw_flow_arrow(ax, phase["sample_center"] - 0.06, phase["delay_center"] - 0.08, arrow_y)
    draw_flow_arrow(ax, phase["delay_center"] + 0.08, phase["probe_center"] - 0.07, arrow_y)
    ax.text(phase["sample_center"], arrow_y + 0.035, "encode", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.text(phase["delay_center"], arrow_y + 0.035, "maintain", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.text(phase["probe_center"], arrow_y + 0.035, "report", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.text(
        0.50,
        0.05,
        "Information must bridge the blank delay from sample to probe.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT_COLOR,
    )


def draw_spike_train(
    ax: Axes,
    x_positions: list[float],
    y_base: float,
    spike_height: float,
    color: str,
    as_dots: bool = True,
) -> None:
    for x in x_positions:
        if as_dots:
            ax.add_patch(Circle((x, y_base + spike_height), 0.008, facecolor=color, edgecolor=color))
        ax.plot([x, x], [y_base, y_base + spike_height], color=color, linewidth=1.3)


def mechanism_box(ax: Axes, x: float, y: float, width: float, height: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=PANEL_BG,
            edgecolor=LIGHT_EDGE,
            linewidth=0.9,
        )
    )
    ax.text(x + 0.03, y + height - 0.04, title, ha="left", va="top", fontsize=10, fontweight="semibold")


def draw_memory_mechanism(ax: Axes, x: float, y: float, width: float, height: float, persistent: bool) -> None:
    title = "Persistent firing hypothesis" if persistent else "Synaptic STSP hypothesis"
    mechanism_box(ax, x, y, width, height, title)

    phase = draw_phase_bar(ax, x=x + 0.04, y=y + height * 0.62, width=width - 0.08, height=0.08)
    phase_label_y = y + height * 0.73
    ax.text(phase["sample_center"], phase_label_y, "sample", ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.text(phase["delay_center"], phase_label_y, "delay", ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.text(phase["probe_center"], phase_label_y, "probe", ha="center", va="bottom", fontsize=7.5, color="#555555")

    y_base = y + height * 0.33
    ax.text(x + 0.03, y_base, "neuron", ha="left", va="center", fontsize=8, color="#666666")
    ax.plot([x + 0.12, x + width - 0.05], [y_base, y_base], color=EDGE_COLOR, linewidth=1.0)

    sample_spikes = [
        phase["sample_start"] + 0.03,
        phase["sample_start"] + 0.07,
        phase["sample_start"] + 0.11,
    ]
    delay_spikes = [
        phase["delay_start"] + 0.05,
        phase["delay_start"] + 0.11,
        phase["delay_start"] + 0.17,
    ]
    probe_spikes = [phase["probe_start"] + 0.05, phase["probe_start"] + 0.10]

    draw_spike_train(ax, sample_spikes, y_base, 0.12, COLOR_DYNAMIC)
    if persistent:
        draw_spike_train(ax, delay_spikes, y_base, 0.12, EDGE_COLOR)
        draw_spike_train(ax, probe_spikes, y_base, 0.12, "#2C7FB8")
        ax.text(
            phase["delay_center"],
            y + 0.12,
            "spikes persist through the delay",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#555555",
        )
    else:
        draw_spike_train(ax, probe_spikes, y_base, 0.12, "#2C7FB8")
        for x_mark in delay_spikes:
            ax.add_patch(Circle((x_mark, y_base + 0.12), 0.008, facecolor="white", edgecolor=LIGHT_EDGE, linewidth=1.0))
        state_w = 0.16
        state_h = 0.09
        state_x = phase["delay_center"] - state_w / 2
        state_y = y + 0.11
        ax.add_patch(
            FancyBboxPatch(
                (state_x, state_y),
                state_w,
                state_h,
                boxstyle="round,pad=0.01,rounding_size=0.014",
                facecolor="#FFF7F6",
                edgecolor=COLOR_DYNAMIC,
                linewidth=1.0,
            )
        )
        ax.text(phase["delay_center"], state_y + state_h * 0.62, "synaptic state", ha="center", va="center", fontsize=7.6)
        ax.text(
            phase["delay_center"],
            state_y + state_h * 0.28,
            "u / x",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold",
            color=COLOR_DYNAMIC,
        )
        draw_flow_arrow(ax, sample_spikes[-1] + 0.01, phase["delay_center"] - 0.09, y_base + 0.06, color=COLOR_DYNAMIC, lw=1.4)
        draw_flow_arrow(ax, phase["delay_center"] + 0.09, probe_spikes[0] - 0.02, y_base + 0.06, color=COLOR_DYNAMIC, lw=1.4)
        ax.text(
            phase["delay_center"],
            y + 0.12,
            "delay spiking can be silent while u/x retains memory",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#555555",
        )


def draw_memory_requirement(ax: Axes) -> None:
    add_panel_header(
        ax,
        "B",
        "Memory Requirement",
        "The key question is whether delay memory requires persistent firing or can remain hidden in synapses.",
    )
    draw_memory_mechanism(ax, x=0.05, y=0.14, width=0.42, height=0.66, persistent=True)
    draw_memory_mechanism(ax, x=0.53, y=0.14, width=0.42, height=0.66, persistent=False)
    ax.text(
        0.50,
        0.05,
        "Both mechanisms can support the probe decision, but only the STSP account predicts silent delay activity.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=TEXT_COLOR,
    )


def draw_network_cartoon(ax: Axes, x: float, y: float, width: float, height: float, accent: str, dynamic: bool) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=PANEL_BG,
            edgecolor=accent,
            linewidth=1.2,
        )
    )

    phase = draw_phase_bar(ax, x=x + 0.04, y=y + height - 0.17, width=width - 0.08, height=0.08)
    draw_flow_arrow(ax, phase["sample_center"] - 0.04, phase["delay_center"] - 0.06, y + height - 0.22, color=EDGE_COLOR, lw=1.5)
    draw_flow_arrow(ax, phase["delay_center"] + 0.06, phase["probe_center"] - 0.05, y + height - 0.22, color=EDGE_COLOR, lw=1.5)

    input_x = x + 0.08
    hidden_x = x + width * 0.50
    output_x = x + width - 0.10
    layer_y = y + height * 0.39

    draw_digit_icon(ax, input_x, layer_y, digit="8")
    ax.text(input_x, y + 0.12, "sample", ha="center", va="center", fontsize=7.8, color="#666666")

    for offset in (-0.04, 0.0, 0.04):
        ax.add_patch(Circle((hidden_x, layer_y + offset), 0.018, facecolor="white", edgecolor=EDGE_COLOR, linewidth=0.9))

    draw_decision_icon(ax, output_x, layer_y)
    ax.text(output_x, y + 0.12, "probe readout", ha="center", va="center", fontsize=7.8, color="#666666")

    draw_flow_arrow(ax, input_x + 0.05, hidden_x - 0.05, layer_y, color=accent, lw=1.8)
    draw_flow_arrow(ax, hidden_x + 0.05, output_x - 0.07, layer_y, color=accent, lw=1.8)

    syn_w = 0.12
    syn_h = 0.08
    syn_x = hidden_x - syn_w / 2
    syn_y = y + 0.15
    syn_label = "u/x evolves" if dynamic else "W fixed"
    syn_fill = "#FFF4F3" if dynamic else "#F3F3F3"
    ax.add_patch(
        FancyBboxPatch(
            (syn_x, syn_y),
            syn_w,
            syn_h,
            boxstyle="round,pad=0.01,rounding_size=0.014",
            facecolor=syn_fill,
            edgecolor=accent,
            linewidth=1.0,
        )
    )
    ax.text(hidden_x, syn_y + syn_h / 2, syn_label, ha="center", va="center", fontsize=8.2, color=TEXT_COLOR)
    ax.add_patch(
        FancyArrowPatch(
            (hidden_x, layer_y - 0.07),
            (hidden_x, syn_y + syn_h),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=accent,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_model_comparison(ax: Axes) -> None:
    add_panel_header(
        ax,
        "C",
        "Model Comparison",
        "Experiments compare dynamic synapses against a static control under the same sample-delay-probe task.",
    )

    left_x = 0.05
    right_x = 0.55
    box_y = 0.18
    box_w = 0.38
    box_h = 0.60

    ax.text(
        left_x + box_w / 2,
        0.80,
        "Dynamic synapses (STSP enabled)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color=COLOR_DYNAMIC,
    )
    ax.text(
        right_x + box_w / 2,
        0.80,
        "Static synapses (STSP disabled)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color=COLOR_STATIC,
    )

    draw_network_cartoon(ax, left_x, box_y, box_w, box_h, accent=COLOR_DYNAMIC, dynamic=True)
    draw_network_cartoon(ax, right_x, box_y, box_w, box_h, accent=COLOR_STATIC, dynamic=False)

    ax.add_patch(
        FancyArrowPatch(
            (0.46, 0.48),
            (0.54, 0.48),
            arrowstyle="<->",
            mutation_scale=13,
            linewidth=1.5,
            color=EDGE_COLOR,
        )
    )
    ax.text(0.50, 0.535, "Dynamic vs Static", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")
    ax.text(
        0.50,
        0.06,
        "Does STSP support behaviorally relevant memory without persistent firing?",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="semibold",
        color=TEXT_COLOR,
    )


def build_figure() -> plt.Figure:
    apply_publication_style()
    fig = plt.figure(figsize=(10.0, 12.0), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.08, 1.08], hspace=0.12)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    draw_task_timeline(axes[0])
    draw_memory_requirement(axes[1])
    draw_model_comparison(axes[2])
    _promote_text_sizes(fig)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a schematic figure for the sample-delay-probe task.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save figure_task_paradigm_schematic.{png,pdf,svg}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure()
    out_base = args.output_dir / "figure_task_paradigm_schematic"
    saved = save_figure_all_formats(fig, out_base)
    plt.close(fig)
    for ext, path in saved.items():
        print(f"{ext.upper()}: {path}")


if __name__ == "__main__":
    main()
