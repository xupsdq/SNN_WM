from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from .context import BuildContext
from .contracts import FigureContract
from .plotting import (
    bivariate_quantile_trajectory,
    clean_axis,
    data_axis,
    estimation_plot,
    heatmap,
    make_figure,
    mean_ci,
    network_line,
    network_means,
    paired_dumbbell,
    scatter_relationship,
    schematic_chain,
    stacked_composition,
)
from .style import (
    CORAL,
    CYAN,
    GRAY,
    GRAY_DARK,
    GRAY_LIGHT,
    GRAY_PALE,
    INK,
    NAVY,
    ORANGE,
    PURPLE,
    TEAL,
    WHITE,
    count_cmap,
    sequential_cmap,
    signed_cmap,
    strength_cmap,
)


def build_s1(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    baseline = context.store.read("fig1.baseline")
    confusion = context.store.read("fig1.confusion").copy()
    recall = context.store.read("fig1.class_recall")
    confusion_sum = (
        confusion.groupby(["true_label", "pred_label"], as_index=False)["count"]
        .sum()
    )
    confusion_sum["fraction"] = confusion_sum["count"] / confusion_sum.groupby(
        "true_label"
    )["count"].transform("sum")
    baseline_long = baseline.melt(
        id_vars="network_seed",
        value_vars=("overall_recall", "error_rate", "silent_rate"),
        var_name="endpoint",
        value_name="value",
    )
    true_order = tuple(
        str(int(value))
        for value in sorted(confusion_sum["true_label"].unique())
    )
    pred_values = tuple(sorted(confusion_sum["pred_label"].unique()))
    pred_labels = tuple(
        "∅" if int(value) < 0 else str(int(value))
        for value in pred_values
    )
    confusion_matrix = confusion_sum.assign(
        row_label=lambda data: data["true_label"].astype(int).astype(str),
        column_label=lambda data: data["pred_label"].map(
            dict(zip(pred_values, pred_labels))
        ),
        value=lambda data: data["fraction"],
        section="confusion",
    )
    recall_matrix = (
        recall.groupby(["network_seed", "digit"], as_index=False)[
            "class_recall"
        ]
        .mean()
        .assign(
            row_label=lambda data: data["digit"].astype(int).astype(str),
            column_label="Recall",
            value=lambda data: data["class_recall"],
            section="class_recall",
        )
    )
    baseline_matrix = baseline_long.assign(
        row_label="All",
        column_label=lambda data: data["endpoint"].map(
            {
                "overall_recall": "Recall",
                "error_rate": "Error",
                "silent_rate": "Silent",
            }
        ),
        section="behavior",
    )
    capture = pd.concat(
        [
            confusion_matrix,
            recall_matrix,
            baseline_matrix,
        ],
        ignore_index=True,
        sort=False,
    )
    axis = data_axis(slots["a"], left=0.07, right=0.07, bottom=0.18, top=0.20)
    heatmap(
        axis,
        capture,
        row="row_label",
        column="column_label",
        value="value",
        row_order=(*true_order, "All"),
        column_order=(*pred_labels, "Recall", "Error", "Silent"),
        cmap=sequential_cmap(),
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Rate",
    )
    axis.axvline(len(pred_labels) - 0.5, color=WHITE, linewidth=1.4)
    axis.axhline(len(true_order) - 0.5, color=WHITE, linewidth=1.4)
    axis.set_xlabel("Response / summary")
    axis.set_ylabel("True class")
    context.capture_panel(
        "s1",
        "a",
        capture,
        metrics=("fraction", "class_recall", "value"),
        groups=("section",),
    )

    phase = context.store.read("p0.phase_firing").copy()
    phase["log_rate"] = np.log10(phase["mean_spike_rate_hz"].astype(float) + 1.0)
    phase_order = ("stimulus", "early_delay", "late_delay", "probe")
    phase_summary = (
        phase.groupby(["network_seed", "layer", "phase"], as_index=False)[
            "log_rate"
        ]
        .mean()
    )
    axis = data_axis(slots["b"], left=0.14, right=0.03, bottom=0.19, top=0.24)
    heatmap(
        axis,
        phase_summary,
        row="layer",
        column="phase",
        value="log_rate",
        row_order=("layer1", "layer2", "layer3"),
        column_order=phase_order,
        cmap=sequential_cmap(),
        colorbar_label="log10(rate + 1)",
        annotate=True,
    )
    axis.set_xticklabels(("stim.", "early", "late", "probe"))
    axis.set_xlabel("Phase")
    axis.set_ylabel("")
    context.capture_panel(
        "s1",
        "b",
        phase_summary,
        metrics=("log_rate",),
        groups=("layer", "phase"),
    )

    decode = context.store.read("fig1.delay_curve")
    decode_summary = (
        decode.groupby(["network_seed", "layer", "delay_ms"], as_index=False)[
            ["acc", "macro_f1", "chance"]
        ]
        .mean()
    )
    decode_long = decode_summary.melt(
        id_vars=("network_seed", "layer", "delay_ms", "chance"),
        value_vars=("acc", "macro_f1"),
        var_name="metric",
        value_name="value",
    )
    decode_long["row_label"] = decode_long.apply(
        lambda row: (
            f"{str(row['layer']).replace('layer', 'L')} "
            f"{'Acc' if row['metric'] == 'acc' else 'F1'}"
        ),
        axis=1,
    )
    decode_rows = tuple(
        f"L{layer} {metric}"
        for layer in (1, 2, 3)
        for metric in ("Acc", "F1")
    )
    axis = data_axis(slots["c"], left=0.21, right=0.03, bottom=0.19, top=0.24)
    heatmap(
        axis,
        decode_long,
        row="row_label",
        column="delay_ms",
        value="value",
        row_order=decode_rows,
        cmap=sequential_cmap(),
        vmin=float(decode_summary["chance"].median()),
        vmax=1.0,
        colorbar_label="Score",
    )
    axis.set_xlabel("Delay (ms)")
    axis.set_ylabel("")
    context.capture_panel(
        "s1",
        "c",
        decode_long,
        metrics=("value",),
        groups=("layer", "metric", "delay_ms"),
    )

    contrast = context.store.read("fig1.delay_contrast")
    contrast_long = contrast.melt(
        id_vars=("network_seed", "delay_ms"),
        value_vars=(
            "stsp_interference",
            "sample_bias_excess_dynamic_minus_static",
        ),
        var_name="endpoint",
        value_name="value",
    )
    axis = data_axis(slots["d"], left=0.24, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        contrast_long,
        x="delay_ms",
        value="value",
        group="endpoint",
        group_order=(
            "stsp_interference",
            "sample_bias_excess_dynamic_minus_static",
        ),
        labels=("Accuracy", "Bias"),
        colors=(NAVY, CORAL),
        linestyles=("-", "--"),
        xlabel="Delay (ms)",
        ylabel="Dynamic - static",
        show_networks=False,
        null=0.0,
    )
    context.capture_panel(
        "s1",
        "d",
        plotted,
        metrics=("value",),
        groups=("endpoint", "delay_ms"),
    )

    substrate = context.store.read("fig1.substrate").copy()
    substrate["category"] = substrate["substrate"].astype(str)
    order = tuple(
        value
        for value in ("dynamic", "spike", "membrane", "ux", "static")
        if value in set(substrate["category"])
    )
    substrate_labels = {
        "dynamic": "Dynamic",
        "spike": "Spike",
        "membrane": "Membrane",
        "ux": "u/x",
        "static": "Static",
    }
    axis = data_axis(slots["e"], left=0.32, right=0.04, bottom=0.18, top=0.27)
    plotted = estimation_plot(
        axis,
        substrate,
        category="category",
        value="acc_probe",
        order=order,
        labels=tuple(substrate_labels[item] for item in order),
        colors=tuple(
            NAVY if item == "dynamic" else CORAL
            if item in {"spike", "membrane", "ux"}
            else GRAY
            for item in order
        ),
        null=None,
        xlabel="Probe accuracy",
    )
    context.capture_panel(
        "s1",
        "e",
        plotted,
        metrics=("acc_probe",),
        groups=("category",),
    )

    attribution = (
        substrate.groupby(["network_seed", "substrate"], as_index=False)[
            ["sample_attribution_rate", "donor_attribution_rate"]
        ]
        .mean()
    )
    axis = data_axis(slots["f"], left=0.22, right=0.04, bottom=0.24, top=0.27)
    for index, substrate_name in enumerate(order):
        part = attribution.loc[attribution["substrate"].eq(substrate_name)]
        for _, row in part.iterrows():
            axis.plot(
                [index - 0.12, index + 0.12],
                [
                    row["sample_attribution_rate"],
                    row["donor_attribution_rate"],
                ],
                color=GRAY_LIGHT,
                linewidth=0.45,
                zorder=1,
            )
        for offset, column, color, face in (
            (-0.12, "sample_attribution_rate", NAVY, WHITE),
            (0.12, "donor_attribution_rate", CORAL, CORAL),
        ):
            values = part[column].to_numpy(float)
            mean, low, high = mean_ci(values)
            axis.scatter(
                np.full(len(values), index + offset),
                values,
                s=7,
                facecolor=face,
                edgecolor=color,
                linewidth=0.5,
                alpha=0.5,
                zorder=2,
            )
            axis.errorbar(
                [index + offset],
                [mean],
                yerr=[[mean - low], [high - mean]],
                color=INK,
                marker="o",
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.6,
                linewidth=0.9,
                capsize=2,
                zorder=4,
            )
    axis.set_xticks(np.arange(len(order)))
    axis.set_xticklabels(
        tuple(
            {
                "dynamic": "Dyn.",
                "spike": "Spike",
                "membrane": "Mem.",
                "ux": "u/x",
                "static": "Static",
            }[value]
            for value in order
        ),
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Attribution")
    clean_axis(axis, grid_axis="y")
    axis.plot([], [], color=NAVY, marker="o", markerfacecolor=WHITE, label="Orig.")
    axis.plot([], [], color=CORAL, marker="o", label="Donor")
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
    )
    context.capture_panel(
        "s1",
        "f",
        attribution,
        metrics=("sample_attribution_rate", "donor_attribution_rate"),
        groups=("substrate",),
    )
    return fig


def build_s2(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    decomp = context.store.read("fixed.decomp_cell")
    valid = decomp.loc[decomp["valid"].eq(1)]
    axis = data_axis(slots["a"], left=0.20, right=0.04, bottom=0.22, top=0.28)
    for metric, label, color in (
        ("same_B_common_update_cosine", "Cos", NAVY),
        (
            "processing_residual_gamma_energy_fraction",
            r"$\Gamma$",
            PURPLE,
        ),
    ):
        for prefix_k, linestyle in ((1, "-"), (5, "--")):
            values = valid.loc[valid["prefix_k"].eq(prefix_k), metric].dropna()
            axis.hist(
                values,
                bins=45,
                density=True,
                histtype="step",
                linewidth=1.0,
                color=color,
                linestyle=linestyle,
                label=f"{label} K{prefix_k}",
            )
    axis.set_xlabel("Metric value")
    axis.set_ylabel("Density")
    clean_axis(axis)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
    )
    context.capture_panel(
        "s2",
        "a",
        valid[
            [
                "network_seed",
                "prefix_k",
                "same_B_common_update_cosine",
                "processing_residual_gamma_energy_fraction",
            ]
        ],
        metrics=(
            "same_B_common_update_cosine",
            "processing_residual_gamma_energy_fraction",
        ),
        groups=("prefix_k",),
    )

    common = (
        valid.groupby(
            ["network_seed", "prefix_k", "history_family_id"],
            as_index=False,
        )["same_B_common_update_cosine"]
        .mean()
    )
    seeds = tuple(sorted(common["network_seed"].unique()))
    common["row_label"] = common.apply(
        lambda row: f"K{int(row['prefix_k'])} · {int(row['network_seed'])}",
        axis=1,
    )
    row_order = tuple(
        f"K{prefix_k} · {int(seed)}"
        for prefix_k in (1, 5)
        for seed in seeds
    )
    axis = data_axis(slots["b"], left=0.24, right=0.03, bottom=0.22, top=0.28)
    heatmap(
        axis,
        common,
        row="row_label",
        column="history_family_id",
        value="same_B_common_update_cosine",
        row_order=row_order,
        cmap=sequential_cmap(),
        vmin=0.5,
        vmax=1.0,
        colorbar_label="",
    )
    axis.set_yticks(
        ((len(seeds) - 1) / 2.0, len(seeds) + (len(seeds) - 1) / 2.0)
    )
    axis.set_yticklabels(("K1", "K5"))
    axis.axhline(len(seeds) - 0.5, color=WHITE, linewidth=1.2)
    axis.set_xlabel("History family")
    axis.set_ylabel("")
    context.capture_panel(
        "s2",
        "b",
        common,
        metrics=("same_B_common_update_cosine",),
        groups=("prefix_k", "history_family_id"),
    )

    swap = context.store.read("fixed.swap_cell")
    swap_primary = swap.loc[
        swap["valid"].eq(1)
        & swap["swap_scope"].eq("layer1_only")
        & swap["endpoint"].eq("layer2_update")
    ]
    gamma_network = (
        valid.groupby(["network_seed", "prefix_k"], as_index=False)[
            "processing_residual_gamma_energy_fraction"
        ]
        .mean()
    )
    align_network = (
        swap_primary.groupby(["network_seed", "prefix_k"], as_index=False)[
            "effect_alignment_cosine"
        ]
        .mean()
    )
    relation = gamma_network.merge(
        align_network,
        on=["network_seed", "prefix_k"],
        how="inner",
        validate="one_to_one",
    )
    axis = data_axis(slots["c"], left=0.23, right=0.04, bottom=0.22, top=0.28)
    plotted = scatter_relationship(
        axis,
        relation,
        x="processing_residual_gamma_energy_fraction",
        y="effect_alignment_cosine",
        group="prefix_k",
        group_colors={1: CYAN, 5: TEAL},
        xlabel="Gamma energy fraction",
        ylabel="Alignment",
        zero_lines=True,
    )
    context.capture_panel(
        "s2",
        "c",
        plotted,
        metrics=(
            "processing_residual_gamma_energy_fraction",
            "effect_alignment_cosine",
        ),
        groups=("prefix_k",),
    )

    components = valid.melt(
        id_vars=("network_seed", "prefix_k"),
        value_vars=(
            "total_contrast_norm",
            "local_replay_contrast_norm",
            "processing_residual_gamma_norm",
        ),
        var_name="component",
        value_name="value",
    )
    component_order = (
        "total_contrast_norm",
        "local_replay_contrast_norm",
        "processing_residual_gamma_norm",
    )
    axis = data_axis(slots["d"], left=0.37, right=0.04, bottom=0.20, top=0.28)
    plotted = estimation_plot(
        axis,
        components,
        category="component",
        value="value",
        order=component_order,
        labels=("T norm", "L norm", "Gamma norm"),
        colors=(NAVY, CYAN, PURPLE),
        null=0.0,
        xlabel="Vector norm",
    )
    context.capture_panel(
        "s2",
        "d",
        plotted,
        metrics=("value",),
        groups=("component",),
    )

    event = context.store.read("fixed.event_cell")
    event = event.loc[event["valid"].eq(1)]
    axis = data_axis(slots["e"], left=0.25, right=0.04, bottom=0.22, top=0.28)
    _density_by_prefix(
        axis,
        event,
        "event_gamma_enrichment",
        xlabel="Gamma enrichment",
    )
    summary = context.store.read("fixed.event_summary")
    if not np.allclose(summary["valid_coverage"].astype(float), 1.0):
        context.add_qc(
            "s2",
            "event_coverage",
            "warning",
            "Event coverage was non-unit and remains available in panel data.",
        )
    context.capture_panel(
        "s2",
        "e",
        event,
        metrics=("event_gamma_enrichment",),
        groups=("prefix_k",),
    )

    scalars = context.store.read("fixed.scalars")
    l2 = scalars.loc[
        scalars["endpoint"].eq("layer1_only_layer2_update_donor_transfer")
    ][["network_seed", "prefix_k", "value"]].rename(columns={"value": "l2"})
    l3 = scalars.loc[
        scalars["endpoint"].eq(
            "layer1_only_early_class_score_donor_transfer"
        )
    ][["network_seed", "prefix_k", "value"]].rename(columns={"value": "l3"})
    transfer = l2.merge(
        l3,
        on=["network_seed", "prefix_k"],
        how="inner",
        validate="one_to_one",
    )
    axis = data_axis(slots["f"], left=0.23, right=0.04, bottom=0.22, top=0.28)
    plotted = scatter_relationship(
        axis,
        transfer,
        x="l2",
        y="l3",
        group="prefix_k",
        group_colors={1: CYAN, 5: TEAL},
        xlabel="L2 donor transfer",
        ylabel="Early L3 transfer",
        zero_lines=True,
    )
    context.capture_panel(
        "s2",
        "f",
        plotted,
        metrics=("l2", "l3"),
        groups=("prefix_k",),
    )
    return fig


def build_s3(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    entry = context.store.read("overlap.entry").copy()
    axis = data_axis(slots["a"], left=0.16, right=0.04, bottom=0.19, top=0.24)
    sample = entry.iloc[
        np.linspace(0, len(entry) - 1, min(len(entry), 5000)).astype(int)
    ]
    image = axis.scatter(
        sample["pixel_similarity"],
        sample["dice_overlap"],
        c=sample["acc_drop"],
        cmap=signed_cmap(),
        s=7,
        alpha=0.35,
        edgecolor="none",
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label("Δ accuracy")
    axis.set_xlabel("Pixel similarity")
    axis.set_ylabel("Input/STSP overlap")
    clean_axis(axis)
    context.capture_panel(
        "s3",
        "a",
        sample,
        metrics=("pixel_similarity", "dice_overlap", "acc_drop"),
    )

    matched = context.store.read("overlap.matched")
    natural = (
        matched.groupby(["network_seed", "overlap_group"], as_index=False)[
            "acc_drop"
        ]
        .mean()
        .rename(columns={"overlap_group": "contrast", "acc_drop": "value"})
    )
    perturb = context.store.read("overlap.perturb_contrast").melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
        var_name="contrast",
        value_name="value",
    )
    combined = pd.concat(
        [
            natural.assign(source="matched observation"),
            perturb.assign(source="intervention"),
        ],
        ignore_index=True,
    )
    order = tuple(combined["contrast"].drop_duplicates().tolist())
    axis = data_axis(slots["b"], left=0.34, right=0.04, bottom=0.19, top=0.24)
    plotted = estimation_plot(
        axis,
        combined,
        category="contrast",
        value="value",
        order=order,
        labels=tuple(
            {
                "high overlap": "High overlap",
                "low overlap": "Low overlap",
                "dynamic_minus_overlap_reset": "Dynamic",
                "nonoverlap_reset_minus_overlap_reset": "Non-overlap",
                "random_reset_minus_overlap_reset": "Random",
            }.get(str(item), _short_label(item, width=14))
            for item in order
        ),
        colors=tuple(
            TEAL if "overlap" in str(item).lower() else GRAY
            for item in order
        ),
        null=0.0,
        xlabel="Accuracy effect",
    )
    context.capture_panel(
        "s3",
        "b",
        plotted,
        metrics=("value",),
        groups=("contrast",),
    )

    perturb_summary = context.store.read("overlap.perturb_summary")
    decision = context.store.read("overlap.decision")
    condition_labels = {
        "full_static": "Static",
        "full_dynamic_intact": "Dynamic",
        "full_dynamic": "Dynamic",
        "l1_overlap_reset": "Overlap",
        "sample_keep_overlap_only_dynamic": "Overlap",
        "l1_nonoverlap_reset": "Non-overlap",
        "sample_keep_nonoverlap_only_dynamic": "Non-overlap",
        "l1_random_matched_reset": "Random",
        "sample_random_matched_dynamic": "Random",
    }
    condition_order = ("Static", "Dynamic", "Overlap", "Non-overlap", "Random")
    c_data = pd.concat(
        [
            perturb_summary[
                ["network_seed", "condition", "mean_accuracy_drop_vs_static"]
            ]
            .rename(columns={"mean_accuracy_drop_vs_static": "value"})
            .assign(endpoint="Accuracy"),
            decision[
                ["network_seed", "condition", "mean_dynamic_like_recovery"]
            ]
            .rename(columns={"mean_dynamic_like_recovery": "value"})
            .assign(endpoint="DPI"),
            decision[
                ["network_seed", "condition", "mean_decision_deflection_score"]
            ]
            .rename(columns={"mean_decision_deflection_score": "value"})
            .assign(endpoint="Decision"),
        ],
        ignore_index=True,
    )
    c_data["condition_label"] = c_data["condition"].map(condition_labels)
    c_data = c_data.dropna(subset=["condition_label"])
    c_data["condition_index"] = c_data["condition_label"].map(
        {value: index for index, value in enumerate(condition_order)}
    )
    axis = data_axis(slots["c"], left=0.22, right=0.04, bottom=0.25, top=0.28)
    plotted = network_line(
        axis,
        c_data,
        x="condition_index",
        value="value",
        group="endpoint",
        group_order=("Accuracy", "DPI", "Decision"),
        labels=("Acc.", "DPI", "Dec."),
        colors=(NAVY, TEAL, CORAL),
        linestyles=("-", "--", "-."),
        ylabel="Endpoint value",
        show_networks=False,
        null=0.0,
    )
    condition_short = {
        "Static": "Static",
        "Dynamic": "Dyn.",
        "Overlap": "Ovlp.",
        "Non-overlap": "Non-ovlp.",
        "Random": "Rand.",
    }
    axis.set_xticks(np.arange(len(condition_order)))
    axis.set_xticklabels(
        tuple(condition_short.get(value, value) for value in condition_order),
        rotation=35,
        ha="right",
    )
    axis.set_yscale("symlog", linthresh=0.01)
    context.capture_panel(
        "s3",
        "c",
        plotted,
        metrics=("value",),
        groups=("endpoint", "condition_label"),
    )

    class_pair = context.store.read("overlap.class_pair").copy()
    parsed = class_pair["class_pair"].astype(str).apply(_parse_pair)
    class_pair["class_a"] = [item[0] for item in parsed]
    class_pair["class_b"] = [item[1] for item in parsed]
    valid_pairs = class_pair.dropna(subset=["class_a", "class_b"])
    axis = data_axis(slots["d"], left=0.23, right=0.02, bottom=0.22, top=0.28)
    if valid_pairs.empty:
        fallback = (
            class_pair.groupby(["network_seed", "class_pair"], as_index=False)[
                "mean_DPI_L3"
            ]
            .mean()
        )
        fallback["metric"] = "DPI"
        heatmap(
            axis,
            fallback,
            row="class_pair",
            column="metric",
            value="mean_DPI_L3",
            cmap=signed_cmap(),
            center=0.0,
            colorbar_label="",
        )
        capture = fallback
    else:
        capture = valid_pairs
        heatmap(
            axis,
            valid_pairs,
            row="class_a",
            column="class_b",
            value="mean_DPI_L3",
            cmap=signed_cmap(),
            center=0.0,
            colorbar_label="",
        )
    axis.set_xlabel("Probe class")
    axis.set_ylabel("Sample class")
    context.capture_panel(
        "s3",
        "d",
        capture,
        metrics=("mean_DPI_L3",),
        groups=("class_pair",),
    )

    support = context.store.read("competition.support")
    transitions = context.store.read("competition.transitions")
    group_order = (
        "overlap_dominant",
        "probe_only_dominant",
        "balanced",
        "random_matched",
    )
    support_network = (
        support.groupby(["network_seed", "unit_group"], as_index=False)[
            "mean_support"
        ]
        .mean()
    )
    transition_network = (
        transitions.groupby(["network_seed", "unit_group"], as_index=False)[
            ["P_advance", "P_recruit", "P_loss"]
        ]
        .mean()
        .melt(
            id_vars=("network_seed", "unit_group"),
            var_name="transition",
            value_name="probability",
        )
    )
    support_network = support_network.rename(
        columns={"mean_support": "value"}
    ).assign(metric="Support")
    transition_network = transition_network.rename(
        columns={"probability": "value"}
    )
    transition_network["metric"] = transition_network["transition"].map(
        {
            "P_advance": "Advance",
            "P_recruit": "Recruit",
            "P_loss": "Loss",
        }
    )
    matrix_data = pd.concat(
        [
            support_network[["network_seed", "unit_group", "metric", "value"]],
            transition_network[
                ["network_seed", "unit_group", "metric", "value"]
            ],
        ],
        ignore_index=True,
    )
    axis = data_axis(slots["e"], left=0.24, right=0.09, bottom=0.24, top=0.28)
    heatmap(
        axis,
        matrix_data,
        row="metric",
        column="unit_group",
        value="value",
        row_order=("Support", "Advance", "Recruit", "Loss"),
        column_order=group_order,
        cmap=sequential_cmap(),
        vmin=0.0,
        vmax=1.0,
        colorbar_label="",
    )
    short_groups = ("Ovlp.", "Probe", "Bal.", "Rand.")
    axis.set_xticklabels(short_groups, rotation=35, ha="right")
    axis.set_xlabel("Site group")
    axis.set_ylabel("")
    context.capture_panel(
        "s3",
        "e",
        matrix_data,
        metrics=("value",),
        groups=("metric", "unit_group"),
    )

    event = context.store.read("p0.event_chain")
    window = context.store.read("competition.window")
    radius = context.store.read("competition.radius")
    null_order = tuple(event["null_type"].drop_duplicates().tolist())
    event_short = {
        "event_time_shuffle": "Tm",
        "winner_loser_pairing_shuffle": "Pr",
        "neighborhood_shuffle": "Nb",
        "dynamic_static_label_shuffle": "Lb",
        "trial_shuffle": "Tr",
        "conservative_max_across_five_nulls": "Mx",
    }
    event_line = event.loc[event["null_type"].isin(null_order)].copy()
    event_profile = network_means(
        event_line,
        ("null_type",),
        "observed_minus_null",
    ).rename(columns={"observed_minus_null": "value"})
    event_profile["setting"] = event_profile["null_type"].map(event_short)
    event_profile["robustness"] = "Null"
    event_positions = {
        value: index for index, value in enumerate(event_short.values())
    }
    event_profile["x_position"] = event_profile["setting"].map(
        event_positions
    )
    window_network = (
        window.groupby(["network_seed", "early_window_ms"], as_index=False)[
            "P_advance_plus_recruit"
        ]
        .mean()
        .rename(columns={"P_advance_plus_recruit": "value"})
    )
    window_order = tuple(sorted(window_network["early_window_ms"].unique()))
    window_start = len(event_positions) + 1
    window_positions = {
        value: window_start + index for index, value in enumerate(window_order)
    }
    window_network["setting"] = window_network["early_window_ms"].map(
        {value: str(int(value)) for value in window_order}
    )
    window_network["x_position"] = window_network["early_window_ms"].map(
        window_positions
    )
    window_network["robustness"] = "Window"
    radius_profile = radius[
        ["network_seed", "neighborhood_radius", "winner_pre_spike_delta_v_mean"]
    ].rename(columns={"winner_pre_spike_delta_v_mean": "value"})
    radius_order = tuple(sorted(radius_profile["neighborhood_radius"].unique()))
    radius_start = window_start + len(window_order) + 1
    radius_positions = {
        value: radius_start + index for index, value in enumerate(radius_order)
    }
    radius_profile["setting"] = radius_profile["neighborhood_radius"].map(
        {value: f"R{int(value)}" for value in radius_order}
    )
    radius_profile["x_position"] = radius_profile["neighborhood_radius"].map(
        radius_positions
    )
    radius_profile["robustness"] = "Radius"
    capture = pd.concat(
        [
            event_profile,
            window_network,
            radius_profile,
        ],
        ignore_index=True,
        sort=False,
    )
    reference = capture.groupby(
        ["network_seed", "robustness"],
        observed=True,
    )["value"].transform("mean")
    capture["relative_deviation"] = (
        capture["value"] - reference
    ) / np.maximum(np.abs(reference), 1e-9)
    axis = data_axis(slots["f"], left=0.18, right=0.04, bottom=0.28, top=0.24)
    plotted = network_line(
        axis,
        capture,
        x="x_position",
        value="relative_deviation",
        group="robustness",
        group_order=("Null", "Window", "Radius"),
        labels=("Null", "Window", "Radius"),
        colors=(CORAL, NAVY, TEAL),
        linestyles=("-", "--", "-."),
        ylabel="Relative deviation",
        show_networks=False,
        null=0.0,
    )
    setting_table = (
        capture[["x_position", "setting"]]
        .drop_duplicates()
        .sort_values("x_position")
    )
    axis.set_xticks(setting_table["x_position"])
    setting_labels = tuple(
        "" if str(value) in {"10", "20"} else str(value)
        for value in setting_table["setting"]
    )
    axis.set_xticklabels(setting_labels, rotation=45, ha="right")
    context.capture_panel(
        "s3",
        "f",
        plotted,
        metrics=("relative_deviation",),
        groups=("robustness",),
    )

    perturb = context.store.read("competition.perturb_contrast")
    same_winner = context.store.read("competition.same_winner")
    writeback = context.store.read("competition.writeback")
    perturb_long = perturb.melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_transition_mass",
            "attenuate_transition_mass",
            "reset_transition_mass",
        ),
        var_name="condition",
        value_name="value",
    )
    transition_labels = {
        "dynamic_transition_mass": "Trans. dyn",
        "attenuate_transition_mass": "Trans. atten",
        "reset_transition_mass": "Trans. reset",
    }
    perturb_long["row"] = perturb_long["condition"].map(transition_labels)
    perturb_long["endpoint"] = "transition"
    winner_network = (
        same_winner.groupby(["network_seed", "condition"], as_index=False)[
            "P_same_winner_lost_or_delayed"
        ]
        .mean()
    )
    winner_order = tuple(winner_network["condition"].drop_duplicates().tolist())
    winner_labels = {
        "dynamic_intact": "Dynamic",
        "attenuate_overlap_high_support": "Attenuate",
        "reset_overlap_high_support": "Reset",
    }
    winner_network = winner_network.rename(
        columns={"P_same_winner_lost_or_delayed": "value"}
    )
    winner_network["row"] = winner_network["condition"].map(
        {key: f"Winner {value.lower()}" for key, value in winner_labels.items()}
    )
    winner_network["endpoint"] = "winner"
    writeback_primary = writeback.loc[
        writeback["metric"].eq(
            writeback["metric"].drop_duplicates().iloc[0]
        )
    ].dropna(subset=["value"])
    writeback_order = tuple(
        writeback_primary["condition"].drop_duplicates().tolist()
    )
    writeback_labels = {
        "attenuate_l1_stsp": "Attenuate",
        "dynamic_intact": "Dynamic",
        "reset_l1_stsp": "Reset",
        "sham_perturbation": "Sham",
    }
    writeback_primary = writeback_primary.copy()
    writeback_primary["row"] = writeback_primary["condition"].map(
        {
            key: f"Write {value.lower()}"
            for key, value in writeback_labels.items()
        }
    )
    writeback_primary["endpoint"] = "writeback"
    capture = pd.concat(
        [
            perturb_long,
            winner_network,
            writeback_primary,
        ],
        ignore_index=True,
        sort=False,
    )
    row_order = (
        tuple(transition_labels.values())
        + tuple(f"Winner {winner_labels[item].lower()}" for item in winner_order)
        + tuple(
            f"Write {writeback_labels.get(item, str(item)).lower()}"
            for item in writeback_order
        )
    )
    axis = data_axis(slots["g"], left=0.36, right=0.04, bottom=0.20, top=0.24)
    plotted = estimation_plot(
        axis,
        capture,
        category="row",
        value="value",
        order=row_order,
        labels=row_order,
        colors=(
            *(NAVY for _ in transition_labels),
            *(CORAL for _ in winner_order),
            *(TEAL for _ in writeback_order),
        ),
        null=0.0,
        xlabel="Rate / effect",
    )
    axis.set_xscale("symlog", linthresh=0.02)
    context.capture_panel(
        "s3",
        "g",
        plotted,
        metrics=("value",),
        groups=("endpoint", "condition"),
    )
    return fig


def build_s4(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    cell = context.store.read("bridge.cell")
    n_families = int(cell["history_family_id"].nunique())
    n_mappings = int(
        cell[["b_anchor_id", "c_anchor_id"]].drop_duplicates().shape[0]
    )
    nodes = schematic_chain(
        data_axis(slots["a"], left=0.03, right=0.03, bottom=0.10, top=0.24),
        ("A/C history", "Exact B", "Post-B L1", "Same C"),
        colors=(WHITE, CORAL, TEAL, WHITE),
        subtitle=(
            f"{n_families} frozen history families; {n_mappings} B-to-C "
            "mappings; K1 and K5."
        ),
    )
    context.capture_panel("s4", "a", nodes)

    boundary = context.store.read("bridge.boundary")
    joint = boundary.loc[
        boundary["endpoint"].eq(
            "joint_ux_input_driven_boundary_displacement"
        )
    ]
    axis = data_axis(slots["b"], left=0.28, right=0.04, bottom=0.19, top=0.24)
    plotted = estimation_plot(
        axis,
        joint,
        category="prefix_k",
        value="value",
        order=(1, 5),
        labels=("K1", "K5"),
        colors=(NAVY, TEAL),
        null=0.0,
        xlabel="Post-B - passive boundary",
        connect_pairs=True,
    )
    context.capture_panel(
        "s4",
        "b",
        plotted,
        metrics=("value",),
        groups=("prefix_k",),
    )

    network = context.store.read("bridge.network")
    for panel_id, value, label, color in (
        (
            "c",
            "layer1_to_layer2_update_donor_transfer",
            "L2 donor transfer",
            NAVY,
        ),
        (
            "d",
            "layer1_to_early_class_score_donor_transfer",
            "Early L3 transfer",
            TEAL,
        ),
    ):
        axis = data_axis(
            slots[panel_id],
            left=0.20,
            right=0.04,
            bottom=0.22,
            top=0.24,
        )
        valid_column = (
            "layer2_update_transfer_valid"
            if panel_id == "c"
            else "early_class_score_transfer_valid"
        )
        cell_valid = cell.loc[
            pd.to_numeric(cell[valid_column], errors="coerce").eq(1)
        ]
        pooled_values = pd.to_numeric(
            cell_valid[value],
            errors="coerce",
        ).dropna()
        for prefix_k, prefix_color, linestyle in (
            (1, CYAN, "-"),
            (5, TEAL, "--"),
        ):
            values = np.sort(
                pd.to_numeric(
                    cell_valid.loc[
                        cell_valid["prefix_k"].eq(prefix_k),
                        value,
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(float)
            )
            cumulative = np.arange(1, len(values) + 1) / len(values)
            axis.plot(
                values,
                cumulative,
                color=prefix_color,
                linestyle=linestyle,
                linewidth=1.0,
                label=f"K{prefix_k}",
            )
        for index, (prefix_k, prefix_color) in enumerate(
            ((1, CYAN), (5, TEAL))
        ):
            values = network.loc[
                network["prefix_k"].eq(prefix_k),
                value,
            ].to_numpy(float)
            mean, low, high = mean_ci(values)
            axis.errorbar(
                [mean],
                [0.07 + 0.07 * index],
                xerr=[[mean - low], [high - mean]],
                color=INK,
                marker="o",
                markerfacecolor=prefix_color,
                markeredgecolor=INK,
                markeredgewidth=0.5,
                linewidth=0.9,
                capsize=2,
                zorder=4,
            )
        if pooled_values.min() < 0.0 < pooled_values.max():
            axis.set_xscale("symlog", linthresh=1.0)
            axis.axvline(0.0, color=GRAY_LIGHT, linewidth=0.7, zorder=0)
        axis.set_xlabel(label)
        axis.set_ylabel("Cell CDF")
        axis.set_ylim(0.0, 1.02)
        clean_axis(axis, grid_axis="y")
        axis.legend(
            handles=[
                *axis.get_legend_handles_labels()[0],
                Line2D(
                    [],
                    [],
                    color=INK,
                    marker="o",
                    markerfacecolor=WHITE,
                    markeredgecolor=INK,
                    linewidth=0.9,
                    label="Net.",
                ),
            ],
            frameon=False,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0.0,
        )
        context.capture_panel(
            "s4",
            panel_id,
            cell_valid[
                ["network_seed", "prefix_k", "history_family_id", value]
            ],
            metrics=(value,),
            groups=("prefix_k",),
        )

    mapping = (
        cell.groupby(
            ["network_seed", "prefix_k", "B_label", "C_label"],
            as_index=False,
        )["layer1_to_layer2_update_donor_transfer"]
        .mean()
    )
    b_labels = tuple(sorted(mapping["B_label"].unique()))
    mapping["row_label"] = mapping.apply(
        lambda row: f"K{int(row['prefix_k'])} · B{int(row['B_label'])}",
        axis=1,
    )
    row_order = tuple(
        f"K{prefix_k} · B{int(b_label)}"
        for prefix_k in (1, 5)
        for b_label in b_labels
    )
    axis = data_axis(slots["e"], left=0.27, right=0.03, bottom=0.20, top=0.24)
    heatmap(
        axis,
        mapping,
        row="row_label",
        column="C_label",
        value="layer1_to_layer2_update_donor_transfer",
        row_order=row_order,
        cmap=sequential_cmap(),
        vmin=0.0,
        vmax=1.0,
        colorbar_label="",
    )
    tick_indices = tuple(range(0, len(row_order), 2))
    axis.set_yticks(tick_indices)
    axis.set_yticklabels([row_order[index] for index in tick_indices])
    axis.axhline(len(b_labels) - 0.5, color=WHITE, linewidth=1.2)
    axis.set_xlabel("C class")
    axis.set_ylabel("")
    context.capture_panel(
        "s4",
        "e",
        mapping,
        metrics=("layer1_to_layer2_update_donor_transfer",),
        groups=("prefix_k", "B_label", "C_label"),
    )

    inference = context.store.read("bridge.inference").copy()
    inference_labels = {
        "joint_ux_input_driven_boundary_displacement": "Boundary",
        "layer1_only_early_class_score_donor_transfer": "Early L3",
        "layer1_only_layer2_update_donor_transfer": "L2 transfer",
        "processing_residual_gamma_energy_fraction": "Residual Γ",
    }
    inference["row"] = inference.apply(
        lambda row: (
            f"{inference_labels[row['endpoint']]} K{int(row['prefix_k'])}"
        ),
        axis=1,
    )
    order = tuple(
        f"{label} K{prefix_k}"
        for label in inference_labels.values()
        for prefix_k in (1, 5)
    )
    axis = data_axis(slots["f"], left=0.30, right=0.04, bottom=0.18, top=0.24)
    indexed = inference.set_index("row").reindex(order)
    for index, (_, row) in enumerate(indexed.iterrows()):
        color = TEAL if int(row["prefix_k"]) == 5 else NAVY
        axis.plot(
            [row["ci95_low"], row["ci95_high"]],
            [index, index],
            color=INK,
            linewidth=1.0,
        )
        axis.scatter(
            [row["mean"]],
            [index],
            s=20,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
        )
    axis.axvline(0.0, color=INK, linestyle=":", linewidth=0.7)
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels(order)
    axis.invert_yaxis()
    axis.set_xlabel("Effect [95% CI]")
    clean_axis(axis, grid_axis="x")
    context.capture_panel(
        "s4",
        "f",
        inference,
        metrics=("mean",),
        groups=("endpoint", "prefix_k"),
    )
    return fig


def build_s5(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    update = context.store.read("prog.update").copy()
    update["state"] = (
        update["layer"].astype(str)
        + " | "
        + update["state_variable"].astype(str)
    )
    update_summary = network_means(
        update,
        ("state", "stage_k"),
        "observed_minus_natural_decay",
    )
    axis = data_axis(slots["a"], left=0.30, right=0.02, bottom=0.22, top=0.29)
    heatmap(
        axis,
        update_summary,
        row="state",
        column="stage_k",
        value="observed_minus_natural_decay",
        cmap=signed_cmap(),
        center=0.0,
        colorbar_label="",
    )
    axis.set_xlabel("Stage")
    axis.set_ylabel("")
    context.capture_panel(
        "s5",
        "a",
        update_summary,
        metrics=("observed_minus_natural_decay",),
        groups=("state", "stage_k"),
    )

    stage = context.store.read("p0.progressive_stage").copy()
    joint_name = _preferred_value(
        stage["state_variable"],
        ("ux_joint_mean", "ux_concat", "ux", "u"),
    )
    joint = stage.loc[stage["state_variable"].eq(joint_name)]
    trajectory = joint.melt(
        id_vars=("network_seed", "stage_k"),
        value_vars=("state_displacement", "natural_decay_displacement"),
        var_name="trajectory",
        value_name="value",
    )
    axis = data_axis(slots["b"], left=0.25, right=0.04, bottom=0.22, top=0.29)
    plotted = network_line(
        axis,
        trajectory,
        x="stage_k",
        value="value",
        group="trajectory",
        group_order=(
            "state_displacement",
            "natural_decay_displacement",
        ),
        labels=("Obs.", "Pass."),
        colors=(NAVY, GRAY),
        linestyles=("-", "--"),
        xlabel="Stage",
        ylabel="L2 displacement",
        show_networks=False,
    )
    context.capture_panel(
        "s5",
        "b",
        plotted,
        metrics=("value",),
        groups=("trajectory", "stage_k"),
    )

    progressive_network = context.store.read("p0.progressive_network")
    joint_network = progressive_network.loc[
        progressive_network["state_variable"].eq(joint_name)
    ]
    axis = data_axis(slots["c"], left=0.25, right=0.04, bottom=0.22, top=0.29)
    plotted = paired_dumbbell(
        axis,
        joint_network,
        left_value="late_mean_k7_k10",
        right_value="early_mean_k2_k5",
        left_label="Late",
        right_label="Early",
        ylabel="Increment beyond passive",
        left_color=GRAY,
        right_color=CORAL,
        null=0.0,
    )
    context.capture_panel(
        "s5",
        "c",
        plotted,
        metrics=("late_mean_k7_k10", "early_mean_k2_k5"),
    )

    weights = context.store.read("prog.weights")
    weight_summary = network_means(
        weights,
        ("item_position", "stage_k"),
        "item_weight",
    )
    axis = data_axis(slots["d"], left=0.24, right=0.02, bottom=0.22, top=0.29)
    heatmap(
        axis,
        weight_summary,
        row="item_position",
        column="stage_k",
        value="item_weight",
        cmap=strength_cmap(),
        colorbar_label="",
    )
    axis.set_xlabel("Stage")
    axis.set_ylabel("Item position")
    context.capture_panel(
        "s5",
        "d",
        weight_summary,
        metrics=("item_weight",),
        groups=("item_position", "stage_k"),
    )

    order = context.store.read("prog.order")
    true_order = order.loc[order["condition"].eq("true_order")]
    order_summary = network_means(
        true_order,
        ("seq_len", "delay_ms"),
        "order_specificity_index",
    )
    axis = data_axis(slots["e"], left=0.24, right=0.02, bottom=0.22, top=0.29)
    heatmap(
        axis,
        order_summary,
        row="seq_len",
        column="delay_ms",
        value="order_specificity_index",
        cmap=signed_cmap(),
        center=0.0,
        colorbar_label="",
        annotate=True,
    )
    axis.set_xlabel("Delay (ms)")
    axis.set_ylabel("Sequence length")
    context.capture_panel(
        "s5",
        "e",
        order_summary,
        metrics=("order_specificity_index",),
        groups=("seq_len", "delay_ms"),
    )

    axis = data_axis(slots["f"], left=0.29, right=0.04, bottom=0.20, top=0.29)
    stage_order = tuple(sorted(joint["stage_k"].unique()))
    plotted = estimation_plot(
        axis,
        joint,
        category="stage_k",
        value="observed_minus_natural_decay",
        order=stage_order,
        labels=tuple(f"K{int(value)}" for value in stage_order),
        colors=tuple(
            CORAL if int(value) <= 5 else TEAL for value in stage_order
        ),
        null=0.0,
        xlabel="Observed - passive",
    )
    context.capture_panel(
        "s5",
        "f",
        plotted,
        metrics=("observed_minus_natural_decay",),
        groups=("stage_k",),
    )
    return fig


def build_s6(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    region = context.store.read("multi.region")
    axis = data_axis(slots["a"], left=0.25, right=0.02, bottom=0.22, top=0.29)
    plotted = stacked_composition(
        axis,
        region,
        category="entry_condition",
        components=("old_mass", "middle_mass", "recent_mass", "other_mass"),
        labels=("Old", "Middle", "Recent", "Other"),
        colors=(CYAN, NAVY, TEAL, GRAY_LIGHT),
        category_order=("peak", "valley", "random"),
        category_labels=("Peak", "Valley", "Random"),
        xlabel="Readout mass",
    )
    context.capture_panel(
        "s6",
        "a",
        plotted,
        metrics=("old_mass", "middle_mass", "recent_mass", "other_mass"),
        groups=("entry_condition",),
    )

    ping = context.store.read("multi.global_ping").copy()
    ping["quantile"] = ping["score_quantile_bin"].map(
        {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
    )
    axis = data_axis(slots["b"], left=0.24, right=0.04, bottom=0.22, top=0.29)
    quantile_colors = sequential_cmap()(np.linspace(0.35, 0.95, 5))
    plotted = bivariate_quantile_trajectory(
        axis,
        ping,
        quantile="score_quantile_bin",
        x_value="spike_probability",
        y_value="mean_early_spike_count",
        order=("Q1", "Q2", "Q3", "Q4", "Q5"),
        colors=quantile_colors,
        xlabel="Spike probability",
        ylabel="Early count",
    )
    context.capture_panel(
        "s6",
        "b",
        plotted,
        metrics=("spike_probability", "mean_early_spike_count"),
        groups=("score_quantile_bin",),
    )

    window = context.store.read("multi.window_probe")
    window_network = network_means(
        window,
        ("early_window_ms",),
        "value",
    )
    axis = data_axis(slots["c"], left=0.24, right=0.04, bottom=0.22, top=0.29)
    plotted = network_line(
        axis,
        window_network,
        x="early_window_ms",
        value="value",
        colors=(NAVY,),
        xlabel="Early window (ms)",
        ylabel="Q5 - Q1 deflection",
        show_networks=True,
        null=0.0,
    )
    context.capture_panel(
        "s6",
        "c",
        plotted,
        metrics=("value",),
        groups=("early_window_ms",),
    )

    threshold = context.store.read("multi.threshold")
    threshold_summary = network_means(
        threshold,
        ("stsp_group_quantile", "overlap_threshold"),
        "value",
    )
    axis = data_axis(slots["d"], left=0.25, right=0.02, bottom=0.22, top=0.29)
    heatmap(
        axis,
        threshold_summary,
        row="stsp_group_quantile",
        column="overlap_threshold",
        value="value",
        cmap=signed_cmap(),
        center=0.0,
        colorbar_label="",
        annotate=True,
    )
    axis.set_xlabel("Overlap threshold")
    axis.set_ylabel("STSP quantile")
    context.capture_panel(
        "s6",
        "d",
        threshold_summary,
        metrics=("value",),
        groups=("stsp_group_quantile", "overlap_threshold"),
    )

    ablation = context.store.read("multi.ablation_pair")
    axis = data_axis(slots["e"], left=0.25, right=0.04, bottom=0.22, top=0.29)
    plotted = paired_dumbbell(
        axis,
        ablation,
        left_value="matched_removal",
        right_value="high_stsp_overlap",
        left_label="Matched",
        right_label="High",
        ylabel="Loss of firing",
        left_color=GRAY,
        right_color=TEAL,
        null=0.0,
    )
    context.capture_panel(
        "s6",
        "e",
        plotted,
        metrics=("matched_removal", "high_stsp_overlap"),
    )

    shuffle = context.store.read("multi.shuffle").copy()
    shuffle["observed_minus_null"] = (
        shuffle["observed_value"].astype(float)
        - shuffle["null_value"].astype(float)
    )
    shuffle_network = network_means(
        shuffle,
        ("endpoint",),
        "observed_minus_null",
    )
    endpoint = (
        "overlap_interaction"
        if "overlap_interaction" in set(shuffle_network["endpoint"])
        else shuffle_network["endpoint"].iloc[0]
    )
    shuffle_primary = shuffle_network.loc[
        shuffle_network["endpoint"].eq(endpoint),
        ["network_seed", "observed_minus_null"],
    ]
    availability = context.store.read("multi.availability")
    availability_network = (
        network_means(
            availability,
            ("overlap_group",),
            "nonzero_fraction",
        )
        .pivot(
            index="network_seed",
            columns="overlap_group",
            values="nonzero_fraction",
        )
        .reset_index()
    )
    availability_network["coverage_gain"] = (
        availability_network["overlap"]
        - availability_network["no_overlap"]
    )
    threshold_network = (
        threshold.groupby("network_seed", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "threshold_effect"})
    )
    control_capture = (
        shuffle_primary.merge(
            threshold_network,
            on="network_seed",
            validate="one_to_one",
        )
        .merge(
            availability_network[["network_seed", "coverage_gain"]],
            on="network_seed",
            validate="one_to_one",
        )
    )
    rank_order = (
        control_capture.sort_values("coverage_gain")["network_seed"].tolist()
    )
    rank_lookup = {
        seed: index + 1 for index, seed in enumerate(rank_order)
    }
    control_profile = control_capture.melt(
        id_vars="network_seed",
        value_vars=(
            "observed_minus_null",
            "threshold_effect",
            "coverage_gain",
        ),
        var_name="control",
        value_name="raw_value",
    )
    control_profile["relative_value"] = control_profile.groupby(
        "control",
        observed=True,
    )["raw_value"].transform(lambda values: values / values.mean())
    control_profile["coverage_rank"] = control_profile["network_seed"].map(
        rank_lookup
    )
    axis = data_axis(slots["f"], left=0.23, right=0.04, bottom=0.22, top=0.29)
    plotted = network_line(
        axis,
        control_profile,
        x="coverage_rank",
        value="relative_value",
        group="control",
        group_order=(
            "observed_minus_null",
            "threshold_effect",
            "coverage_gain",
        ),
        labels=("Shuf.", "Thresh.", "Cover."),
        colors=(NAVY, CORAL, TEAL),
        linestyles=("-", "--", "-."),
        xlabel="Coverage rank",
        ylabel="Relative value",
        show_networks=False,
        null=1.0,
    )
    axis.set_xticks(())
    context.capture_panel(
        "s6",
        "f",
        plotted,
        metrics=("raw_value", "relative_value"),
        groups=("control",),
    )
    return fig


def build_s7(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    dual = context.store.read("pair.dual")
    facets = (
        ("layer1", "u"),
        ("layer1", "x"),
        ("layer2", "ux_concat"),
        ("layer3", "g"),
    )
    facet_frames: list[pd.DataFrame] = []
    for layer, state_variable in facets:
        part = dual.loc[
            dual["layer"].eq(layer)
            & dual["state_variable"].eq(state_variable)
        ]
        if len(part) > 1200:
            part = part.iloc[
                np.linspace(0, len(part) - 1, 1200).astype(int)
            ]
        facet_frames.append(
            part.assign(
                layer=layer,
                state_variable=state_variable,
                facet=(
                    f"{layer.replace('layer', 'L')} "
                    f"{state_variable.replace('ux_concat', 'u/x')}"
                ),
            )
        )
    facet_data = pd.concat(facet_frames, ignore_index=True)
    facet_order = tuple(facet_data["facet"].drop_duplicates())
    axis = data_axis(slots["a"], left=0.16, right=0.04, bottom=0.20, top=0.34)
    plotted = scatter_relationship(
        axis,
        facet_data,
        x="sim_to_A",
        y="sim_to_B",
        group="facet",
        group_colors=dict(
            zip(facet_order, (CYAN, NAVY, TEAL, PURPLE))
        ),
        xlabel="Similarity to A",
        ylabel="Similarity to B",
        max_points=5000,
        identity=True,
    )
    context.capture_panel(
        "s7",
        "a",
        plotted,
        metrics=("sim_to_A", "sim_to_B"),
        groups=("layer", "state_variable"),
    )

    comparison = context.store.read("pair.model_comparison")
    primary = comparison.loc[
        comparison["layer"].eq("layer2")
        & comparison["state_variable"].eq("ux_concat")
    ]
    model_order = tuple(primary["model_name"].drop_duplicates())
    axis = data_axis(slots["b"], left=0.31, right=0.04, bottom=0.19, top=0.24)
    plotted = estimation_plot(
        axis,
        primary,
        category="model_name",
        value="cv_r2",
        order=model_order,
        labels=tuple(_short_label(value) for value in model_order),
        colors=tuple(
            TEAL if "unconstrained" in str(value) else NAVY
            if "sum" in str(value) or "mean" in str(value)
            else GRAY
            for value in model_order
        ),
        null=0.0,
        xlabel=r"Cross-validated $R^2$",
    )
    context.capture_panel(
        "s7",
        "b",
        plotted,
        metrics=("cv_r2",),
        groups=("model_name",),
    )

    specificity = context.store.read("pair.specificity")
    specificity = specificity.loc[
        specificity["layer"].eq("layer2")
        & specificity["state_variable"].eq("ux_concat")
    ]
    residual = context.store.read("pair.residual")
    residual = residual.loc[
        residual["layer"].eq("layer2")
        & residual["state_variable"].eq("ux_concat")
    ]
    pair_network = context.store.read("p0.pair_network")
    endpoints = pd.concat(
        [
            network_means(specificity, (), "true_minus_shuffled")
            .rename(columns={"true_minus_shuffled": "value"})
            .assign(endpoint="Pair specificity"),
            network_means(residual, (), "residual_pair_specificity")
            .rename(columns={"residual_pair_specificity": "value"})
            .assign(endpoint="Residual specificity"),
            pair_network[["network_seed", "min_component_similarity"]]
            .rename(columns={"min_component_similarity": "value"})
            .assign(endpoint="Min constituent similarity"),
        ],
        ignore_index=True,
    )
    endpoint_order = (
        "Pair specificity",
        "Residual specificity",
        "Min constituent similarity",
    )
    axis = data_axis(slots["c"], left=0.42, right=0.04, bottom=0.20, top=0.28)
    plotted = estimation_plot(
        axis,
        endpoints,
        category="endpoint",
        value="value",
        order=endpoint_order,
        labels=("Pair", "Residual", "Min constituent"),
        colors=(NAVY, PURPLE, TEAL),
        null=0.0,
        xlabel="Network effect",
    )
    context.capture_panel(
        "s7",
        "c",
        plotted,
        metrics=("value",),
        groups=("endpoint",),
    )

    nulls = context.store.read("pair.null")
    endpoint_name = (
        "delta_r2_interaction_beyond_bounded_saturation"
    )
    null_primary = nulls.loc[nulls["endpoint"].eq(endpoint_name)]
    axis = data_axis(slots["d"], left=0.24, right=0.04, bottom=0.22, top=0.28)
    for null_model, part in null_primary.groupby("null_model", observed=True):
        axis.hist(
            part["delta_r2"].dropna(),
            bins=35,
            density=True,
            histtype="step",
            linewidth=0.9,
            alpha=0.8,
            label="Null",
        )
    observed = float(null_primary["observed_reference_delta_r2"].median())
    axis.axvline(observed, color=CORAL, linewidth=1.2, label="Obs.")
    axis.set_xlabel(r"$\Delta R^2$")
    axis.set_ylabel("Null density")
    clean_axis(axis)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
    )
    interaction = context.store.read("pair.interaction").copy()
    interaction["endpoint"] = "Observed"
    capture = pd.concat(
        [
            null_primary.assign(source="null"),
            interaction.rename(
                columns={
                    "delta_r2_interaction_beyond_bounded_saturation": "delta_r2"
                }
            ).assign(source="observed"),
        ],
        ignore_index=True,
        sort=False,
    )
    context.capture_panel(
        "s7",
        "d",
        capture,
        metrics=("delta_r2", "observed_reference_delta_r2"),
        groups=("source", "null_model"),
    )

    delay_layer = context.store.read("pair.delay_layer")
    delay_primary = delay_layer.loc[
        delay_layer["layer"].eq("layer2")
        & delay_layer["metric"].isin(
            ("dual_retention", "pair_specificity", "residual_pair_specificity")
        )
    ]
    metric_order = (
        "dual_retention",
        "pair_specificity",
        "residual_pair_specificity",
    )
    axis = data_axis(slots["e"], left=0.22, right=0.04, bottom=0.20, top=0.24)
    plotted = network_line(
        axis,
        delay_primary,
        x="delay2_ms",
        value="value",
        group="metric",
        group_order=metric_order,
        labels=("Dual", "Pair", "Residual"),
        colors=(NAVY, TEAL, PURPLE),
        linestyles=("-", "--", "-."),
        xlabel="Delay (ms)",
        ylabel="State metric",
        show_networks=False,
        null=0.0,
    )
    context.capture_panel(
        "s7",
        "e",
        plotted,
        metrics=("value",),
        groups=("metric", "delay2_ms"),
    )

    delay = context.store.read("pair.delay_contrast")
    delay_long = delay.melt(
        id_vars=("network_seed", "delay2_ms", "keep_prob"),
        value_vars=(
            "completion_gain_SAB_minus_SB",
            "completion_gain_SAB_minus_S0",
        ),
        var_name="contrast",
        value_name="value",
    )
    axis = data_axis(slots["f"], left=0.22, right=0.04, bottom=0.20, top=0.24)
    plotted = network_line(
        axis,
        delay_long,
        x="delay2_ms",
        value="value",
        group="contrast",
        group_order=(
            "completion_gain_SAB_minus_SB",
            "completion_gain_SAB_minus_S0",
        ),
        labels=("SAB - SB", "SAB - S0"),
        colors=(CORAL, NAVY),
        linestyles=("-", "--"),
        xlabel="Delay (ms)",
        ylabel="Completion gain",
        show_networks=False,
        null=0.0,
    )
    context.capture_panel(
        "s7",
        "f",
        plotted,
        metrics=("value",),
        groups=("contrast", "delay2_ms"),
    )

    ping = context.store.read("pair.ping_sweep").copy()
    ping["setting"] = np.where(
        ping["sweep_type"].eq("amplitude"),
        "A=" + ping["ping_amp"].astype(str),
        "T=" + ping["ping_ms"].astype(str),
    )
    setting_order = tuple(ping["setting"].drop_duplicates())
    ping_primary = ping.loc[ping["state_condition"].isin(("S0", "S_AB"))]
    ping_plot = network_means(
        ping_primary,
        ("state_condition", "setting"),
        "pair_member_readout_rate",
    )
    ping_positions = {
        value: index for index, value in enumerate(setting_order)
    }
    ping_plot["x_position"] = ping_plot["setting"].map(ping_positions)
    ping_plot = ping_plot.rename(
        columns={"pair_member_readout_rate": "value"}
    )
    ping_plot["series"] = (
        "Ping " + ping_plot["state_condition"].astype(str).str.replace("_", "")
    )

    cue = context.store.read("pair.partial_cue")
    cue_primary = cue.loc[cue["state_condition"].eq("S_AB")]
    targets = tuple(cue_primary["target_item"].drop_duplicates())
    keep_order = tuple(sorted(cue_primary["keep_prob"].astype(float).unique()))
    cue_plot = network_means(
        cue_primary,
        ("target_item", "keep_prob"),
        "target_recovery_gain_vs_S0",
    )
    cue_start = len(setting_order) + 1
    cue_positions = {
        value: cue_start + index for index, value in enumerate(keep_order)
    }
    cue_plot["x_position"] = cue_plot["keep_prob"].map(cue_positions)
    cue_plot = cue_plot.rename(
        columns={"target_recovery_gain_vs_S0": "value"}
    )
    cue_plot["series"] = "Cue " + cue_plot["target_item"].astype(str)
    access_data = pd.concat(
        [
            ping_plot[["network_seed", "x_position", "series", "value"]],
            cue_plot[["network_seed", "x_position", "series", "value"]],
        ],
        ignore_index=True,
    )
    series_order = ("Ping S0", "Ping SAB", *tuple(f"Cue {item}" for item in targets))
    axis = data_axis(slots["g"], left=0.16, right=0.04, bottom=0.27, top=0.24)
    plotted = network_line(
        axis,
        access_data,
        x="x_position",
        value="value",
        group="series",
        group_order=series_order,
        labels=series_order,
        colors=(GRAY, TEAL, NAVY, CORAL)[: len(series_order)],
        linestyles=("--", "-", "-", "-.")[: len(series_order)],
        xlabel="Ping amp.  |  ms  |  cue",
        ylabel="Readout / gain",
        show_networks=False,
        null=0.0,
    )
    keep_ticks = tuple(
        keep_order[index]
        for index in sorted(
            {0, len(keep_order) // 3, 2 * len(keep_order) // 3, len(keep_order) - 1}
        )
    )
    axis.set_xticks(
        tuple(ping_positions[value] for value in setting_order)
        + tuple(cue_positions[value] for value in keep_ticks)
    )
    axis.set_xticklabels(
        tuple(
            f"{float(str(value).split('=')[-1]):g}"
            for value in setting_order
        )
        + tuple(f"{value:g}" for value in keep_ticks),
        rotation=45,
        ha="right",
    )
    amplitude_count = sum(
        str(value).startswith("A=") for value in setting_order
    )
    axis.axvline(
        amplitude_count - 0.5,
        color=GRAY_LIGHT,
        linewidth=0.7,
        zorder=0,
    )
    axis.axvline(
        len(setting_order),
        color=GRAY_LIGHT,
        linewidth=0.7,
        zorder=0,
    )
    context.capture_panel(
        "s7",
        "g",
        plotted,
        metrics=("value",),
        groups=("series",),
    )
    return fig


def build_s8(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    multi_network = context.store.read("p0.multi_network")
    axis = data_axis(slots["a"], left=0.23, right=0.04, bottom=0.22, top=0.29)
    plotted = network_line(
        axis,
        multi_network,
        x="seq_len",
        value="n_eff",
        colors=(NAVY,),
        xlabel="Sequence length K",
        ylabel="Effective constituents",
        show_networks=True,
    )
    x_values = np.linspace(
        float(multi_network["seq_len"].min()),
        float(multi_network["seq_len"].max()),
        100,
    )
    axis.plot(
        x_values,
        x_values,
        color=GRAY,
        linestyle=":",
        linewidth=0.8,
        label="Linear K",
    )
    summary = network_means(multi_network, ("seq_len",), "n_eff")
    fit_frame = (
        summary.groupby("seq_len", as_index=False)["n_eff"].mean()
        .sort_values("seq_len")
    )
    try:
        params, _ = curve_fit(
            lambda x, ceiling, rate: ceiling * (1.0 - np.exp(-rate * x)),
            fit_frame["seq_len"].to_numpy(float),
            fit_frame["n_eff"].to_numpy(float),
            p0=(float(fit_frame["n_eff"].max()) * 1.2, 0.2),
            bounds=((0.0, 0.0), (100.0, 10.0)),
            maxfev=10_000,
        )
        axis.plot(
            x_values,
            params[0] * (1.0 - np.exp(-params[1] * x_values)),
            color=CORAL,
            linestyle="--",
            linewidth=1.1,
            label="Saturating fit",
        )
    except (RuntimeError, ValueError):
        context.add_qc(
            "s8",
            "saturating_fit",
            "warning",
            "Saturating reference fit was not estimable; data and linear reference remain.",
        )
    axis.legend(frameon=False, fontsize=6.8)
    context.capture_panel(
        "s8",
        "a",
        plotted,
        metrics=("n_eff",),
        groups=("seq_len",),
    )

    peak = context.store.read("prog.peak_summary")
    peak_long = peak.melt(
        id_vars=("network_seed", "seq_len"),
        value_vars=(
            "mean_peak_valley_delta",
            "mean_support_gini",
            "fraction_structured_sequences",
        ),
        var_name="endpoint",
        value_name="value",
    )
    axis = data_axis(slots["b"], left=0.23, right=0.04, bottom=0.22, top=0.29)
    plotted = network_line(
        axis,
        peak_long,
        x="seq_len",
        value="value",
        group="endpoint",
        group_order=(
            "mean_peak_valley_delta",
            "mean_support_gini",
            "fraction_structured_sequences",
        ),
        labels=(r"$\Delta$", "Gini", "Struct."),
        colors=(NAVY, PURPLE, TEAL),
        linestyles=("-", "--", "-."),
        xlabel="Sequence length K",
        ylabel="Endpoint value",
        show_networks=False,
        null=0.0,
    )
    context.capture_panel(
        "s8",
        "b",
        plotted,
        metrics=("value",),
        groups=("endpoint", "seq_len"),
    )
    axis.set_ylim(0.0, 1.05)

    access = context.store.read("prog.access")
    access = access.loc[access["state_condition"].eq("S_final")]
    access_long = access.melt(
        id_vars=("network_seed", "seq_len", "delay_ms"),
        value_vars=("latest_item_mass", "earlier_item_mass"),
        var_name="item_group",
        value_name="value",
    )
    axis = data_axis(slots["c"], left=0.23, right=0.04, bottom=0.20, top=0.24)
    plotted = network_line(
        axis,
        access_long,
        x="delay_ms",
        value="value",
        group="item_group",
        group_order=("latest_item_mass", "earlier_item_mass"),
        labels=("Latest", "Earlier"),
        colors=(CORAL, NAVY),
        linestyles=("-", "--"),
        xlabel="Delay (ms)",
        ylabel="Mass",
        show_networks=False,
    )
    context.capture_panel(
        "s8",
        "c",
        plotted,
        metrics=("value",),
        groups=("item_group", "delay_ms"),
    )

    boundary = context.store.read("prog.boundary")
    boundary_summary = network_means(
        boundary,
        ("seq_len", "delay_ms"),
        "accessible_item_count",
    ).merge(
        network_means(
            boundary,
            ("seq_len", "delay_ms"),
            "rescued_fraction",
        ),
        on=("network_seed", "seq_len", "delay_ms"),
        validate="one_to_one",
    )
    seq_order = tuple(sorted(boundary_summary["seq_len"].unique()))
    delay_order = tuple(sorted(boundary_summary["delay_ms"].unique()))
    axis = data_axis(slots["d"], left=0.20, right=0.03, bottom=0.20, top=0.24)
    heatmap(
        axis,
        boundary_summary,
        row="seq_len",
        column="delay_ms",
        value="rescued_fraction",
        row_order=seq_order,
        column_order=delay_order,
        cmap=sequential_cmap(),
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Rescued",
    )
    count_matrix = boundary_summary.pivot_table(
        index="seq_len",
        columns="delay_ms",
        values="accessible_item_count",
        aggfunc="mean",
        observed=True,
    ).reindex(index=seq_order, columns=delay_order)
    rescued_matrix = boundary_summary.pivot_table(
        index="seq_len",
        columns="delay_ms",
        values="rescued_fraction",
        aggfunc="mean",
        observed=True,
    ).reindex(index=seq_order, columns=delay_order)
    for row_index in range(len(seq_order)):
        for column_index in range(len(delay_order)):
            count = float(count_matrix.iloc[row_index, column_index])
            rescued = float(rescued_matrix.iloc[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{count:.1f}",
                ha="center",
                va="center",
                color=WHITE if rescued > 0.55 else INK,
            )
    axis.set_xlabel("Delay (ms) · cell = count")
    axis.set_ylabel("K")
    context.capture_panel(
        "s8",
        "d",
        boundary_summary,
        metrics=("accessible_item_count", "rescued_fraction"),
        groups=("seq_len", "delay_ms"),
    )

    cue = context.store.read("prog.cue")
    cue_order = tuple(cue["cue_type"].drop_duplicates())
    axis = data_axis(slots["e"], left=0.31, right=0.04, bottom=0.20, top=0.24)
    plotted = estimation_plot(
        axis,
        cue,
        category="cue_type",
        value="target_memory_gain",
        order=cue_order,
        labels=tuple(_short_label(value) for value in cue_order),
        colors=tuple(
            TEAL if value == "matched" else CORAL
            if value == "mismatched"
            else GRAY
            for value in cue_order
        ),
        null=0.0,
        xlabel="Target memory gain",
    )
    context.capture_panel(
        "s8",
        "e",
        plotted,
        metrics=("target_memory_gain",),
        groups=("cue_type",),
    )

    order = context.store.read("prog.order")
    true_order = order.loc[order["condition"].eq("true_order")]
    order_summary = network_means(
        true_order,
        ("seq_len", "delay_ms"),
        "order_specificity_index",
    )
    axis = data_axis(slots["f"], left=0.20, right=0.02, bottom=0.20, top=0.24)
    heatmap(
        axis,
        order_summary,
        row="seq_len",
        column="delay_ms",
        value="order_specificity_index",
        cmap=signed_cmap(),
        center=0.0,
        colorbar_label="",
        annotate=True,
    )
    axis.set_xlabel("Delay (ms)")
    axis.set_ylabel("K")
    context.capture_panel(
        "s8",
        "f",
        order_summary,
        metrics=("order_specificity_index",),
        groups=("seq_len", "delay_ms"),
    )

    coupling = context.store.read("prog.coupling")
    coupling_network = network_means(
        coupling,
        ("delay_ms",),
        "morphology_support_beta",
    ).merge(
        network_means(
            coupling,
            ("delay_ms",),
            "functional_gain_norm",
        ),
        on=["network_seed", "delay_ms"],
        validate="one_to_one",
    )
    delay_order = tuple(sorted(coupling_network["delay_ms"].unique()))
    delay_colors = {
        delay: color
        for delay, color in zip(
            delay_order,
            (CYAN, NAVY, TEAL, CORAL),
        )
    }
    axis = data_axis(slots["g"], left=0.18, right=0.04, bottom=0.20, top=0.24)
    plotted = scatter_relationship(
        axis,
        coupling_network,
        x="morphology_support_beta",
        y="functional_gain_norm",
        group="delay_ms",
        group_colors=delay_colors,
        xlabel="Morphology support beta",
        ylabel="Functional gain (norm.)",
        max_points=5000,
        zero_lines=True,
    )
    context.capture_panel(
        "s8",
        "g",
        plotted,
        metrics=("morphology_support_beta", "functional_gain_norm"),
        groups=("delay_ms",),
    )
    return fig


def _short_label(value: object, *, width: int = 22) -> str:
    text = str(value).replace("_", " ").replace(";", "\n")
    if len(text) <= width:
        return text
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines[:2])


def _density_by_prefix(
    ax,
    frame: pd.DataFrame,
    value: str,
    *,
    xlabel: str,
) -> None:
    colors = {1: CYAN, 5: TEAL}
    valid = frame.dropna(subset=[value])
    for prefix_k, part in valid.groupby("prefix_k", observed=True):
        ax.hist(
            part[value].astype(float),
            bins=35,
            density=True,
            histtype="step",
            linewidth=1.0,
            color=colors.get(prefix_k, NAVY),
            label=f"K{int(prefix_k)}",
        )
    ax.axvline(0.0, color=INK, linestyle=":", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    clean_axis(ax)
    ax.legend(frameon=False)


def _parse_pair(value: object) -> tuple[float, float]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(numbers) < 2:
        return (np.nan, np.nan)
    return (float(numbers[0]), float(numbers[1]))


def _preferred_value(series: pd.Series, candidates: Sequence[str]) -> str:
    values = set(series.astype(str).unique())
    for candidate in candidates:
        if candidate in values:
            return candidate
    if not values:
        raise ValueError("No state-variable values are available")
    return sorted(values)[0]


SUPPLEMENTARY_BUILDERS: Mapping[
    str,
    Callable[[BuildContext, FigureContract], Figure],
] = {
    "s1": build_s1,
    "s2": build_s2,
    "s3": build_s3,
    "s4": build_s4,
    "s5": build_s5,
    "s6": build_s6,
    "s7": build_s7,
    "s8": build_s8,
}
