from __future__ import annotations

from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    sequential_cmap,
    signed_cmap,
)


def build_fig1(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    nodes = schematic_chain(
        data_axis(slots["a"], left=0.02, right=0.02, bottom=0.08, top=0.18),
        ("Input", "L1 u/x", "Silent delay", "L2 u/x", "Later input"),
        colors=(WHITE, CYAN, GRAY_PALE, NAVY, WHITE),
        subtitle="The boundary state is inherited without sustained firing.",
    )
    context.capture_panel("fig1", "a", nodes)

    phase = context.store.read("p0.phase_firing").copy()
    phase_order = ("stimulus", "early_delay", "late_delay", "probe")
    phase["phase_index"] = phase["phase"].map(
        {value: index for index, value in enumerate(phase_order)}
    )
    phase["log_rate"] = np.log10(
        pd.to_numeric(phase["mean_spike_rate_hz"], errors="coerce") + 1.0
    )
    axis = data_axis(slots["b"], left=0.17, right=0.04, bottom=0.22, top=0.24)
    network_line(
        axis,
        phase,
        x="phase_index",
        value="log_rate",
        group="layer",
        group_order=("layer1", "layer2", "layer3"),
        labels=("L1", "L2", "L3"),
        colors=(CYAN, NAVY, TEAL),
        xlabel="Phase",
        ylabel=r"$\log_{10}(\mathrm{rate}+1)$",
        show_networks=False,
    )
    axis.set_xticks(range(4))
    axis.set_xticklabels(("stim.", "early", "late", "probe"))
    axis.set_xlim(-0.2, 3.2)
    context.capture_panel(
        "fig1",
        "b",
        phase[
            [
                "network_seed",
                "layer",
                "phase",
                "phase_index",
                "mean_spike_rate_hz",
                "log_rate",
            ]
        ],
        metrics=("mean_spike_rate_hz", "log_rate"),
        groups=("layer", "phase"),
    )

    decode = context.store.read("fig1.delay_decode").copy()
    decoder_summary = (
        decode.groupby(["network_seed", "layer", "delay_ms"], as_index=False)[
            ["acc", "macro_f1", "chance"]
        ]
        .mean()
    )
    axis = data_axis(slots["c"], left=0.12, right=0.02, bottom=0.19, top=0.24)
    plotted = heatmap(
        axis,
        decoder_summary,
        row="layer",
        column="delay_ms",
        value="acc",
        row_order=("layer1", "layer2", "layer3"),
        cmap=sequential_cmap(),
        vmin=float(decoder_summary["chance"].median()),
        vmax=1.0,
        colorbar_label="Accuracy",
        annotate=True,
    )
    axis.set_xlabel("Delay (ms)")
    axis.set_ylabel("")
    context.capture_panel(
        "fig1",
        "c",
        decoder_summary,
        metrics=("acc", "macro_f1"),
        groups=("layer", "delay_ms"),
    )

    delay = context.store.read("fig1.delay_contrast")
    axis = data_axis(slots["d"], left=0.19, right=0.04, bottom=0.23, top=0.27)
    plotted = network_line(
        axis,
        delay,
        x="delay_ms",
        value="stsp_interference",
        xlabel="Delay (ms)",
        ylabel="Dynamic - static",
        colors=(NAVY,),
        null=0.0,
    )
    context.capture_panel(
        "fig1",
        "d",
        plotted,
        metrics=("stsp_interference",),
        groups=("delay_ms",),
    )

    condition = context.store.read("fig1.condition")
    condition_order = (
        "dynamic_intact",
        "ux_trial_shuffle",
        "static_frozen",
    )
    axis = data_axis(slots["e"], left=0.34, right=0.04, bottom=0.18, top=0.27)
    plotted = estimation_plot(
        axis,
        condition.loc[condition["condition"].isin(condition_order)],
        category="condition",
        value="acc_probe",
        order=condition_order,
        labels=("Dynamic", "u/x shuffle", "Static"),
        colors=(NAVY, CORAL, GRAY),
        null=None,
        xlabel="Probe accuracy",
        connect_pairs=True,
    )
    context.capture_panel(
        "fig1",
        "e",
        plotted,
        metrics=("acc_probe",),
        groups=("condition",),
    )

    attribution = context.store.read("fig1.attribution").copy()
    long = attribution.melt(
        id_vars=("network_seed", "condition"),
        value_vars=(
            "original_sample_attribution",
            "donor_sample_attribution",
        ),
        var_name="attribution",
        value_name="value",
    )
    long["condition_index"] = long["condition"].map(
        {"dynamic_intact": 0, "ux_trial_shuffle": 1}
    )
    axis = data_axis(slots["f"], left=0.20, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        long,
        x="condition_index",
        value="value",
        group="attribution",
        group_order=(
            "original_sample_attribution",
            "donor_sample_attribution",
        ),
        labels=("Original", "Donor"),
        colors=(NAVY, CORAL),
        linestyles=("-", "--"),
        ylabel="Attribution rate",
        show_networks=True,
    )
    axis.set_xticks((0, 1))
    axis.set_xticklabels(("Intact", "Shuffle"))
    context.capture_panel(
        "fig1",
        "f",
        plotted,
        metrics=("value",),
        groups=("condition_index", "attribution"),
    )
    return fig


def build_fig2(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    nodes = schematic_chain(
        data_axis(slots["a"], left=0.02, right=0.02, bottom=0.08, top=0.18),
        ("History A/C", "Matched boundary", "Exact B", "Free / replay", "L2 update"),
        colors=(WHITE, GRAY_PALE, CORAL, CYAN, NAVY),
        subtitle=r"Identical B: total contrast $T=L+\Gamma$; K = 1 and K = 5.",
    )
    context.capture_panel("fig2", "a", nodes)

    scalars = context.store.read("fixed.scalars")
    endpoint_labels = {
        "same_B_common_update_cosine": "Cosine",
        "processing_residual_gamma_energy_fraction": r"Residual $\Gamma$",
    }
    b_data = scalars.loc[scalars["endpoint"].isin(endpoint_labels)].copy()
    b_data["row"] = b_data.apply(
        lambda row: (
            f"{endpoint_labels[row['endpoint']]} K{int(row['prefix_k'])}"
        ),
        axis=1,
    )
    b_order = tuple(
        f"{label} K{prefix_k}"
        for label in endpoint_labels.values()
        for prefix_k in (1, 5)
    )
    axis = data_axis(slots["b"], left=0.34, right=0.08, bottom=0.20, top=0.24)
    plotted = estimation_plot(
        axis,
        b_data,
        category="row",
        value="value",
        order=b_order,
        labels=b_order,
        colors=(NAVY, NAVY, PURPLE, PURPLE),
        null=None,
        xlabel="Metric value",
    )
    axis.axhline(1.5, color=GRAY_LIGHT, linewidth=0.7, zorder=0)
    context.capture_panel(
        "fig2",
        "b",
        plotted,
        metrics=("value",),
        groups=("row",),
    )

    decomp = context.store.read("fixed.decomp_summary").copy()
    long = decomp.melt(
        id_vars=("network_seed", "prefix_k"),
        value_vars=(
            "mean_total_contrast_fraction",
            "mean_local_replay_fraction",
            "mean_processing_residual_gamma_energy_fraction",
        ),
        var_name="component",
        value_name="value",
    )
    labels = {
        "mean_total_contrast_fraction": "T fraction",
        "mean_local_replay_fraction": "L fraction",
        "mean_processing_residual_gamma_energy_fraction": r"$\Gamma$",
    }
    long["row"] = long.apply(
        lambda row: f"K{int(row['prefix_k'])} {labels[row['component']]}",
        axis=1,
    )
    order = tuple(
        f"K{k} {labels[component]}"
        for k in (1, 5)
        for component in labels
    )
    color_lookup = {
        "T fraction": NAVY,
        "L fraction": CYAN,
        r"$\Gamma$": PURPLE,
    }
    colors = tuple(color_lookup[item.split(" ", 1)[1]] for item in order)
    axis = data_axis(slots["c"], left=0.23, right=0.05, bottom=0.18, top=0.24)
    plotted = estimation_plot(
        axis,
        long,
        category="row",
        value="value",
        order=order,
        labels=order,
        colors=colors,
        null=0.0,
        xlabel="Fraction / energy fraction",
    )
    context.capture_panel(
        "fig2",
        "c",
        long,
        metrics=("value",),
        groups=("prefix_k", "component"),
    )

    event = context.store.read("fixed.event_cell")
    axis = data_axis(slots["d"], left=0.20, right=0.04, bottom=0.22, top=0.27)
    plotted = scatter_relationship(
        axis,
        event.loc[event["valid"].eq(1)],
        x="changed_event_coordinate_fraction",
        y="event_gamma_enrichment",
        group="prefix_k",
        group_colors={1: CYAN, 5: TEAL},
        xlabel="Changed-event fraction",
        ylabel="Gamma enrichment",
        max_points=4000,
    )
    context.capture_panel(
        "fig2",
        "d",
        plotted,
        metrics=("event_gamma_enrichment",),
        groups=("prefix_k",),
    )

    l2 = scalars.loc[
        scalars["endpoint"].eq("layer1_only_layer2_update_donor_transfer")
    ]
    axis = data_axis(slots["e"], left=0.27, right=0.04, bottom=0.20, top=0.27)
    plotted = estimation_plot(
        axis,
        l2,
        category="prefix_k",
        value="value",
        order=(1, 5),
        labels=("K1", "K5"),
        colors=(TEAL, TEAL),
        null=0.0,
        xlabel="Donor-transfer index",
        connect_pairs=True,
    )
    context.capture_panel(
        "fig2",
        "e",
        plotted,
        metrics=("value",),
        groups=("prefix_k",),
    )

    trajectory = context.store.read("fixed.trajectory").copy()
    trajectory = trajectory.loc[
        (
            trajectory["track"].eq("natural")
            & trajectory["branch"].eq("free")
        )
        | (
            trajectory["track"].eq("stsp_isolated")
            & trajectory["branch"].eq("replay")
        )
    ]
    trajectory["series"] = trajectory.apply(
        lambda row: f"K{int(row['prefix_k'])} {row['branch']}",
        axis=1,
    )
    series_order = ("K1 free", "K1 replay", "K5 free", "K5 replay")
    axis = data_axis(slots["f"], left=0.21, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        trajectory,
        x="elapsed_steps",
        value="target_margin",
        group="series",
        group_order=series_order,
        labels=("K1 F", "K1 R", "K5 F", "K5 R"),
        colors=(NAVY, NAVY, TEAL, TEAL),
        linestyles=("-", "--", "-", "--"),
        xlabel="Steps after B",
        ylabel="Target margin",
        show_networks=False,
        null=0.0,
    )
    margin_values = pd.to_numeric(
        trajectory["target_margin"],
        errors="coerce",
    ).dropna()
    positive_margin = margin_values.loc[margin_values.gt(0.0)]
    linear_threshold = max(float(positive_margin.min()) * 0.75, 1e-7)
    axis.set_yscale("symlog", linthresh=linear_threshold, linscale=0.6)
    axis.set_ylim(0.0, float(positive_margin.max()) * 1.18)
    axis.set_yticks((0.0, 1e-5, 1e-3, 1e-1, 1e1))
    context.capture_panel(
        "fig2",
        "f",
        plotted,
        metrics=("target_margin",),
        groups=("series", "elapsed_steps"),
    )
    return fig


def build_fig3(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    nodes = schematic_chain(
        data_axis(slots["a"], left=0.02, right=0.02, bottom=0.08, top=0.18),
        (
            "Retained support",
            "Input overlap",
            "L1 recruit / loss",
            "L2 write-back",
            "L3 response",
        ),
        colors=(CYAN, TEAL, CORAL, NAVY, TEAL),
        subtitle="Spatial overlap gates conversion of silent support into firing and write-back.",
    )
    context.capture_panel("fig3", "a", nodes)

    perturb = context.store.read("overlap.perturb_contrast").copy()
    long = perturb.melt(
        id_vars="network_seed",
        value_vars=(
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
        var_name="contrast",
        value_name="value",
    )
    order = (
        "dynamic_minus_overlap_reset",
        "nonoverlap_reset_minus_overlap_reset",
        "random_reset_minus_overlap_reset",
    )
    axis = data_axis(slots["b"], left=0.31, right=0.04, bottom=0.19, top=0.24)
    plotted = estimation_plot(
        axis,
        long,
        category="contrast",
        value="value",
        order=order,
        labels=("Dynamic", "Non-overlap", "Random"),
        colors=(TEAL, GRAY, GRAY_DARK),
        null=0.0,
        xlabel="Accuracy vs overlap reset",
    )
    context.capture_panel(
        "fig3",
        "b",
        plotted,
        metrics=("value",),
        groups=("contrast",),
    )

    support = context.store.read("competition.support")
    support = support.loc[
        support["layer"].astype(str).str.contains("1")
        | support["layer"].eq("layer1")
    ]
    group_order = (
        "overlap_dominant",
        "probe_only_dominant",
        "balanced",
        "random_matched",
    )
    axis = data_axis(slots["c"], left=0.28, right=0.04, bottom=0.19, top=0.24)
    plotted = estimation_plot(
        axis,
        support.loc[support["unit_group"].isin(group_order)],
        category="unit_group",
        value="mean_support",
        order=group_order,
        labels=("Overlap", "Probe-only", "Balanced", "Random"),
        colors=(TEAL, CYAN, NAVY, GRAY),
        null=0.0,
        xlabel="Pre-input support",
    )
    context.capture_panel(
        "fig3",
        "c",
        plotted,
        metrics=("mean_support",),
        groups=("unit_group",),
    )

    transitions = context.store.read("competition.transitions")
    transition_groups = transitions.loc[
        transitions["unit_group"].isin(group_order)
    ]
    axis = data_axis(slots["d"], left=0.26, right=0.03, bottom=0.20, top=0.29)
    plotted = stacked_composition(
        axis,
        transition_groups,
        category="unit_group",
        components=("P_advance", "P_recruit", "P_loss", "P_unchanged"),
        labels=("Advance", "Recruit", "Loss", "Same"),
        colors=(NAVY, CORAL, GRAY_DARK, GRAY_LIGHT),
        category_order=group_order,
        category_labels=("Overlap", "Probe-only", "Balanced", "Random"),
        xlabel="Transition composition",
    )
    context.capture_panel(
        "fig3",
        "d",
        plotted,
        metrics=("P_advance", "P_recruit", "P_loss", "P_unchanged"),
        groups=("unit_group",),
    )

    writeback = context.store.read("p0.writeback")
    same_trial = context.store.read("p0.same_trial_path")
    endpoint_frames = (
        (
            "L2 DID",
            writeback,
            "conditional_difference_in_differences",
            NAVY,
        ),
        (
            r"L1–L2 $\beta$",
            same_trial,
            "standardized_l1_to_l2_beta",
            TEAL,
        ),
        (r"$\Delta R^2$", same_trial, "incremental_r2", PURPLE),
    )
    e_data = pd.concat(
        [
            frame[["network_seed", value]]
            .rename(columns={value: "value"})
            .assign(endpoint=label)
            for label, frame, value, _ in endpoint_frames
        ],
        ignore_index=True,
    )
    e_order = tuple(item[0] for item in endpoint_frames)
    axis = data_axis(slots["e"], left=0.36, right=0.04, bottom=0.20, top=0.27)
    plotted = estimation_plot(
        axis,
        e_data,
        category="endpoint",
        value="value",
        order=e_order,
        labels=e_order,
        colors=tuple(item[3] for item in endpoint_frames),
        null=0.0,
        xlabel="Effect",
    )
    axis.set_xlim(-0.03, 0.68)
    axis.set_xticks((0.0, 0.25, 0.50))
    context.capture_panel(
        "fig3",
        "e",
        plotted,
        metrics=("value",),
        groups=("endpoint",),
    )

    l3 = context.store.read("overlap.l3_time").copy()
    conditions = (
        "full_dynamic",
        "sample_keep_overlap_only_dynamic",
        "sample_keep_nonoverlap_only_dynamic",
    )
    l3 = l3.loc[l3["condition"].isin(conditions)]
    axis = data_axis(slots["f"], left=0.21, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        l3,
        x="time_ms",
        value="DPI_L3_t",
        group="condition",
        group_order=conditions,
        labels=("Dynamic", "Overlap", "Non-ovlp."),
        colors=(NAVY, TEAL, GRAY),
        linestyles=("-", "-", "--"),
        xlabel="Time (ms)",
        ylabel="L3 displacement",
        show_networks=False,
        null=0.0,
    )
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        handlelength=1.6,
        borderaxespad=0.0,
        labelspacing=0.18,
    )
    context.capture_panel(
        "fig3",
        "f",
        plotted,
        metrics=("DPI_L3_t",),
        groups=("condition", "time_ms"),
    )
    return fig


def build_fig4(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    nodes = schematic_chain(
        data_axis(slots["a"], left=0.03, right=0.03, bottom=0.10, top=0.24),
        ("History", "B", "Post-B state", "Identical C"),
        colors=(WHITE, CORAL, TEAL, WHITE),
        subtitle="A passive branch and an L1-only donor swap share the same C.",
    )
    context.capture_panel("fig4", "a", nodes)

    boundary = context.store.read("bridge.boundary")
    joint = boundary.loc[
        boundary["endpoint"].eq(
            "joint_ux_input_driven_boundary_displacement"
        )
    ]
    axis = data_axis(slots["b"], left=0.27, right=0.04, bottom=0.19, top=0.24)
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
        "fig4",
        "b",
        plotted,
        metrics=("value",),
        groups=("prefix_k",),
    )

    network = context.store.read("bridge.network")
    for panel_id, value, xlabel, color in (
        (
            "c",
            "layer1_to_layer2_update_donor_transfer",
            "L2 donor-transfer index",
            NAVY,
        ),
        (
            "d",
            "layer1_to_early_class_score_donor_transfer",
            "Early L3 donor-transfer index",
            TEAL,
        ),
    ):
        axis = data_axis(
            slots[panel_id],
            left=0.27,
            right=0.04,
            bottom=0.19,
            top=0.24,
        )
        plotted = estimation_plot(
            axis,
            network,
            category="prefix_k",
            value=value,
            order=(1, 5),
            labels=("K1", "K5"),
            colors=(color, color),
            null=0.0,
            xlabel=xlabel,
            connect_pairs=True,
        )
        if panel_id == "c":
            axis.set_xlim(-0.03, 0.90)
        context.capture_panel(
            "fig4",
            panel_id,
            plotted,
            metrics=(value,),
            groups=("prefix_k",),
        )

    inference = context.store.read("bridge.inference").copy()
    endpoint_labels = {
        "joint_ux_input_driven_boundary_displacement": "Boundary",
        "processing_residual_gamma_energy_fraction": "Residual Γ",
        "layer1_only_layer2_update_donor_transfer": "L2",
        "layer1_only_early_class_score_donor_transfer": "L3 early",
    }
    inference["endpoint_label"] = inference["endpoint"].map(endpoint_labels)
    inference["row"] = inference.apply(
        lambda row: f"{row['endpoint_label']} K{int(row['prefix_k'])}",
        axis=1,
    )
    order = tuple(
        f"{label} K{k}"
        for label in endpoint_labels.values()
        for k in (1, 5)
    )
    axis = data_axis(slots["e"], left=0.25, right=0.04, bottom=0.18, top=0.24)
    y = np.arange(len(order))
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
            s=22,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
            zorder=3,
        )
    axis.axvline(0.0, color=INK, linestyle=":", linewidth=0.75)
    axis.set_yticks(y)
    axis.set_yticklabels(order)
    axis.invert_yaxis()
    axis.set_xlabel("Effect [95% CI]")
    clean_axis(axis, grid_axis="x")
    context.capture_panel(
        "fig4",
        "e",
        inference,
        metrics=("mean",),
        groups=("endpoint", "prefix_k"),
    )

    nodes = schematic_chain(
        data_axis(slots["f"], left=0.04, right=0.04, bottom=0.10, top=0.24),
        (r"$S_t$", "Process B", r"$S_{t+1}$", "Process C"),
        colors=(NAVY, CORAL, TEAL, WHITE),
        loop=True,
        subtitle="The written state becomes the next input's initial condition.",
    )
    context.capture_panel("fig4", "f", nodes)
    return fig


def build_fig5(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    nodes = schematic_chain(
        data_axis(slots["a"], left=0.02, right=0.02, bottom=0.08, top=0.18),
        (
            r"$S_{k-1}$",
            "Observed input",
            r"$S_k$",
            "Passive match",
            r"$S_k^{passive}$",
        ),
        colors=(NAVY, CORAL, TEAL, GRAY_PALE, GRAY_LIGHT),
        subtitle="The same observed-versus-passive contrast is evaluated at every prefix.",
    )
    context.capture_panel("fig5", "a", nodes)

    stage = context.store.read("p0.progressive_stage")
    axis = data_axis(slots["b"], left=0.21, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        stage,
        x="stage_k",
        value="observed_minus_natural_decay",
        group="state_variable",
        group_order=("u", "x", "ux_joint_mean"),
        labels=("u", "x", "u/x"),
        colors=(CYAN, TEAL, NAVY),
        xlabel="Stage K",
        ylabel="Observed - passive",
        show_networks=False,
        null=0.0,
    )
    context.capture_panel(
        "fig5",
        "b",
        plotted,
        metrics=("observed_minus_natural_decay",),
        groups=("state_variable", "stage_k"),
    )

    scalars = context.store.read("fixed.scalars")
    endpoint_specs = (
        (
            "processing_residual_gamma_energy_fraction",
            "Γ",
            PURPLE,
        ),
        ("full_trace_event_gamma_enrichment", "Event", CORAL),
        ("layer1_only_layer2_update_donor_transfer", "L1→L2", TEAL),
    )
    c_data = scalars.loc[
        scalars["endpoint"].isin(tuple(item[0] for item in endpoint_specs))
    ].copy()
    endpoint_label_lookup = {
        endpoint: label for endpoint, label, _ in endpoint_specs
    }
    c_data["series"] = c_data["endpoint"].map(endpoint_label_lookup)
    axis = data_axis(slots["c"], left=0.22, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        c_data,
        x="prefix_k",
        value="value",
        group="series",
        group_order=tuple(item[1] for item in endpoint_specs),
        labels=tuple(item[1] for item in endpoint_specs),
        colors=tuple(item[2] for item in endpoint_specs),
        xlabel="Prefix K",
        ylabel="Endpoint value",
        show_networks=False,
        show_legend=False,
    )
    axis.set_xticks((1, 5))
    axis.set_xticklabels(("K1", "K5"))
    axis.set_yscale("log")
    axis.set_xlim(0.8, 5.25)
    label_colors = {
        label: color for _, label, color in endpoint_specs
    }
    for label, value in (
        c_data.loc[c_data["prefix_k"].eq(5)]
        .groupby("series", observed=True)["value"]
        .mean()
        .items()
    ):
        vertical_offset = -4 if label == "Event" else 4
        axis.annotate(
            label,
            xy=(5.0, value),
            xytext=(-5, vertical_offset),
            textcoords="offset points",
            color=label_colors[label],
            ha="right",
            va="top" if label == "Event" else "bottom",
            clip_on=True,
        )
    context.capture_panel(
        "fig5",
        "c",
        plotted,
        metrics=("value",),
        groups=("series", "prefix_k"),
    )

    state_labels = {"u": "u", "x": "x", "ux_joint_mean": "u/x"}
    observed = (
        stage[
            ["network_seed", "stage_k", "state_variable", "state_displacement"]
        ]
        .rename(columns={"state_displacement": "value"})
        .assign(series=lambda data: data["state_variable"].map(state_labels))
    )
    passive = (
        stage.groupby(["network_seed", "stage_k"], as_index=False)[
            "natural_decay_displacement"
        ]
        .mean()
        .rename(columns={"natural_decay_displacement": "value"})
        .assign(series="Passive")
    )
    d_data = pd.concat(
        [
            observed[["network_seed", "stage_k", "series", "value"]],
            passive[["network_seed", "stage_k", "series", "value"]],
        ],
        ignore_index=True,
    )
    axis = data_axis(slots["d"], left=0.21, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        d_data,
        x="stage_k",
        value="value",
        group="series",
        group_order=("u", "x", "u/x", "Passive"),
        labels=("u", "x", "u/x", "Passive"),
        colors=(CYAN, TEAL, NAVY, GRAY),
        linestyles=("-", "-", "-", "--"),
        xlabel="Stage K",
        ylabel="State displacement",
        show_networks=False,
    )
    context.capture_panel(
        "fig5",
        "d",
        plotted,
        metrics=("value",),
        groups=("series", "stage_k"),
    )

    network = context.store.read("p0.progressive_network")
    axis = data_axis(slots["e"], left=0.17, right=0.04, bottom=0.20, top=0.24)
    states = ("u", "x", "ux_joint_mean")
    colors = (CYAN, TEAL, NAVY)
    state_x = np.arange(len(states), dtype=float)
    for index, (state, color) in enumerate(zip(states, colors)):
        part = network.loc[network["state_variable"].eq(state)]
        for _, row in part.iterrows():
            axis.plot(
                [index - 0.13, index + 0.13],
                [row["early_mean_k2_k5"], row["late_mean_k7_k10"]],
                color=GRAY_LIGHT,
                linewidth=0.5,
            )
        for offset, column, face in (
            (-0.13, "early_mean_k2_k5", WHITE),
            (0.13, "late_mean_k7_k10", color),
        ):
            values = part[column].to_numpy(float)
            axis.scatter(
                np.full(len(values), index + offset),
                values,
                s=9,
                facecolor=face,
                edgecolor=color,
                linewidth=0.55,
                alpha=0.65,
            )
            mean, low, high = mean_ci(values)
            axis.errorbar(
                [index + offset],
                [mean],
                yerr=[[mean - low], [high - mean]],
                color=INK,
                marker="o",
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.7,
                linewidth=0.9,
                capsize=2,
                zorder=4,
            )
    axis.set_xticks(state_x)
    axis.set_xticklabels(("u", "x", "u/x joint"))
    axis.set_ylabel("Observed - passive")
    axis.text(
        0.02,
        0.98,
        "early ○   late ●",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=GRAY_DARK,
    )
    clean_axis(axis, grid_axis="y")
    context.capture_panel(
        "fig5",
        "e",
        network,
        metrics=(
            "early_mean_k2_k5",
            "late_mean_k7_k10",
            "early_minus_late",
        ),
        groups=("state_variable",),
    )

    joint = stage.loc[stage["state_variable"].eq("ux_joint_mean")]
    matrix = (
        joint.groupby(["network_seed", "stage_k"], as_index=False)[
            "observed_minus_natural_decay"
        ]
        .mean()
    )
    axis = data_axis(slots["f"], left=0.14, right=0.10, bottom=0.20, top=0.24)
    plotted = heatmap(
        axis,
        matrix,
        row="network_seed",
        column="stage_k",
        value="observed_minus_natural_decay",
        cmap=signed_cmap(),
        center=0.0,
        colorbar_label="",
    )
    seeds = sorted(matrix["network_seed"].unique())
    positions = np.arange(0, len(seeds), 4)
    axis.set_yticks(positions)
    axis.set_yticklabels([str(seeds[index]) for index in positions])
    axis.set_xlabel("Stage K")
    axis.set_ylabel("Network")
    context.capture_panel(
        "fig5",
        "f",
        matrix,
        metrics=("observed_minus_natural_decay",),
        groups=("stage_k",),
    )
    return fig


def build_fig6(
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
        "fig6",
        "a",
        plotted,
        metrics=("old_mass", "middle_mass", "recent_mass", "other_mass"),
        groups=("entry_condition",),
    )

    ping = context.store.read("multi.global_ping").copy()
    ping["quantile"] = ping["score_quantile_bin"].map(
        {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
    )
    axis = data_axis(slots["b"], left=0.24, right=0.04, bottom=0.22, top=0.28)
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
        "fig6",
        "b",
        plotted,
        metrics=("spike_probability", "mean_early_spike_count"),
        groups=("score_quantile_bin",),
    )

    probe = context.store.read("multi.real_probe").copy()
    probe["quantile"] = probe["score_quantile_bin"].map(
        {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
    )
    windows = tuple(sorted(probe["early_window_ms"].astype(int).unique()))
    palette = (NAVY, TEAL, CORAL, GRAY_DARK)
    linestyles = ("-", "--", "-.", ":")
    axis = data_axis(slots["c"], left=0.22, right=0.04, bottom=0.22, top=0.28)
    plotted = network_line(
        axis,
        probe,
        x="quantile",
        value="delta_spike_probability",
        group="early_window_ms",
        group_order=windows,
        labels=tuple(f"{window} ms" for window in windows),
        colors=palette[: len(windows)],
        linestyles=linestyles[: len(windows)],
        xlabel="Support score",
        ylabel="Dynamic - baseline",
        show_networks=False,
        null=0.0,
    )
    axis.set_xticks(range(1, 6))
    axis.set_xticklabels(("Q1", "Q2", "Q3", "Q4", "Q5"))
    context.capture_panel(
        "fig6",
        "c",
        plotted,
        metrics=("delta_spike_probability",),
        groups=("early_window_ms", "quantile"),
    )

    interaction = context.store.read("multi.interaction").copy()
    primary_window = int(
        min(interaction["early_window_ms"].astype(int).unique())
    )
    primary_threshold = float(
        np.sort(interaction["overlap_threshold"].astype(float).unique())[0]
    )
    interaction = interaction.loc[
        interaction["early_window_ms"].astype(int).eq(primary_window)
        & np.isclose(
            interaction["overlap_threshold"].astype(float),
            primary_threshold,
        )
    ]
    paired = (
        interaction.groupby("network_seed", as_index=False)[
            ["low_overlap_delta", "high_overlap_delta"]
        ]
        .mean()
        .dropna()
    )
    axis = data_axis(slots["d"], left=0.22, right=0.04, bottom=0.22, top=0.28)
    plotted = paired_dumbbell(
        axis,
        paired,
        left_value="low_overlap_delta",
        right_value="high_overlap_delta",
        left_label="Low\noverlap",
        right_label="High\noverlap",
        ylabel="Firing deflection",
        left_color=GRAY,
        right_color=TEAL,
        null=0.0,
    )
    context.capture_panel(
        "fig6",
        "d",
        plotted,
        metrics=("low_overlap_delta", "high_overlap_delta"),
        groups=(),
    )

    ablation = context.store.read("multi.ablation")
    primary = ablation.loc[
        ablation["early_window_ms"].astype(int).eq(
            int(np.median(ablation["early_window_ms"].astype(int).unique()))
        )
    ]
    wide = (
        primary.groupby(
            ["network_seed", "loss_condition"],
            as_index=False,
        )["loss_delta_spike_probability"]
        .mean()
        .pivot(
            index="network_seed",
            columns="loss_condition",
            values="loss_delta_spike_probability",
        )
        .reset_index()
    )
    axis = data_axis(slots["e"], left=0.21, right=0.04, bottom=0.22, top=0.28)
    plotted = paired_dumbbell(
        axis,
        wide,
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
        "fig6",
        "e",
        plotted,
        metrics=("matched_removal", "high_stsp_overlap"),
    )

    shuffle = context.store.read("multi.shuffle").copy()
    shuffle["observed_minus_null"] = (
        shuffle["observed_value"].astype(float)
        - shuffle["null_value"].astype(float)
    )
    shuffle_network = (
        shuffle.groupby(["network_seed", "endpoint"], as_index=False)[
            "observed_minus_null"
        ]
        .mean()
    )
    endpoint = (
        "overlap_interaction"
        if "overlap_interaction" in set(shuffle_network["endpoint"])
        else shuffle_network["endpoint"].iloc[0]
    )
    shuffle_primary = (
        shuffle_network.loc[shuffle_network["endpoint"].eq(endpoint)]
        [["network_seed", "observed_minus_null"]]
    )
    threshold = context.store.read("multi.threshold")
    threshold_network = (
        threshold.groupby("network_seed", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "threshold_effect"})
    )
    availability = context.store.read("multi.availability")
    availability_network = (
        availability.groupby(["network_seed", "overlap_group"], as_index=False)[
            "nonzero_fraction"
        ]
        .mean()
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
    axis = data_axis(slots["f"], left=0.23, right=0.04, bottom=0.22, top=0.28)
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
        "fig6",
        "f",
        plotted,
        metrics=("raw_value", "relative_value"),
        groups=("control",),
    )
    return fig


def build_fig7(
    context: BuildContext,
    contract: FigureContract,
) -> Figure:
    fig, slots = make_figure(contract)
    dual = context.store.read("pair.dual")
    primary = dual.loc[
        dual["layer"].eq("layer2")
        & dual["state_variable"].astype(str).isin(("ux", "u_x", "ux_joint"))
    ]
    if primary.empty:
        primary = dual.loc[dual["layer"].eq("layer2")]
    axis = data_axis(slots["a"], left=0.16, right=0.04, bottom=0.19, top=0.24)
    plotted = scatter_relationship(
        axis,
        primary,
        x="sim_to_A",
        y="sim_to_B",
        color=PURPLE,
        xlabel="Similarity to A",
        ylabel="Similarity to B",
        max_points=5000,
        identity=True,
    )
    context.capture_panel(
        "fig7",
        "a",
        plotted,
        metrics=("sim_to_A", "sim_to_B"),
    )

    pair_network = context.store.read("p0.pair_network")
    metric_specs = (
        ("unconstrained_cv_r2", "Additive CV R2", NAVY),
        ("linear_mixture_gain", "Mixture gain", TEAL),
        ("residual_pair_specificity", "Residual specificity", PURPLE),
    )
    b_data = pd.concat(
        [
            pair_network[["network_seed", metric]]
            .rename(columns={metric: "value"})
            .assign(endpoint=label)
            for metric, label, _ in metric_specs
        ],
        ignore_index=True,
    )
    axis = data_axis(slots["b"], left=0.36, right=0.04, bottom=0.20, top=0.24)
    plotted = estimation_plot(
        axis,
        b_data,
        category="endpoint",
        value="value",
        order=tuple(item[1] for item in metric_specs),
        labels=(r"CV $R^2$", "Mixture gain", "Residual"),
        colors=tuple(item[2] for item in metric_specs),
        null=0.0,
        xlabel="Effect",
    )
    context.capture_panel(
        "fig7",
        "b",
        plotted,
        metrics=("value",),
        groups=("endpoint",),
    )

    neutral = context.store.read("pair.neutral_ping")
    neutral_order = tuple(neutral["state_condition"].drop_duplicates().tolist())
    neutral_plot = network_means(
        neutral,
        ("state_condition",),
        "pair_access_gain_SAB_vs_S0",
    )
    neutral_positions = {
        value: index for index, value in enumerate(neutral_order)
    }
    neutral_plot["x_position"] = neutral_plot["state_condition"].map(
        neutral_positions
    )
    neutral_plot = neutral_plot.rename(
        columns={"pair_access_gain_SAB_vs_S0": "value"}
    ).assign(series="Ping")

    partial = context.store.read("pair.partial_cue").copy()
    if "S_AB" in set(partial["state_condition"]):
        partial = partial.loc[partial["state_condition"].eq("S_AB")]
    keep_values = tuple(sorted(partial["keep_prob"].astype(float).unique()))
    partial_plot = network_means(
        partial,
        ("target_item", "keep_prob"),
        "target_recovery_gain_vs_S0",
    )
    cue_start = len(neutral_order) + 1
    cue_positions = {
        value: cue_start + index for index, value in enumerate(keep_values)
    }
    partial_plot["x_position"] = partial_plot["keep_prob"].map(cue_positions)
    partial_plot = partial_plot.rename(
        columns={"target_recovery_gain_vs_S0": "value"}
    )
    partial_plot["series"] = (
        "Cue " + partial_plot["target_item"].astype(str)
    )

    delay = context.store.read("pair.delay_contrast")
    delay_values = tuple(sorted(delay["delay2_ms"].astype(float).unique()))
    delay_plot = network_means(
        delay,
        ("delay2_ms",),
        "completion_gain_SAB_minus_SB",
    )
    delay_start = cue_start + len(keep_values) + 1
    delay_positions = {
        value: delay_start + index for index, value in enumerate(delay_values)
    }
    delay_plot["x_position"] = delay_plot["delay2_ms"].map(delay_positions)
    delay_plot = delay_plot.rename(
        columns={"completion_gain_SAB_minus_SB": "value"}
    ).assign(series="Delay")
    access_data = pd.concat(
        [
            neutral_plot[["network_seed", "x_position", "series", "value"]],
            partial_plot[["network_seed", "x_position", "series", "value"]],
            delay_plot[["network_seed", "x_position", "series", "value"]],
        ],
        ignore_index=True,
    )
    series_order = tuple(
        ["Ping"]
        + [f"Cue {item}" for item in partial["target_item"].drop_duplicates()]
        + ["Delay"]
    )
    axis = data_axis(slots["c"], left=0.09, right=0.02, bottom=0.24, top=0.32)
    plotted = network_line(
        axis,
        access_data,
        x="x_position",
        value="value",
        group="series",
        group_order=series_order,
        labels=series_order,
        colors=(TEAL, NAVY, CORAL, CYAN)[: len(series_order)],
        linestyles=("-", "-", "--", "-.")[: len(series_order)],
        xlabel="Ping state  |  cue strength  |  delay (ms)",
        ylabel="Access gain",
        show_networks=False,
        null=0.0,
    )
    keep_tick_values = tuple(
        keep_values[index]
        for index in sorted({0, len(keep_values) // 3, 2 * len(keep_values) // 3, len(keep_values) - 1})
    )
    delay_tick_values = tuple(
        delay_values[index]
        for index in sorted({0, len(delay_values) // 2, len(delay_values) - 1})
    )
    tick_positions = (
        tuple(neutral_positions[value] for value in neutral_order)
        + tuple(cue_positions[value] for value in keep_tick_values)
        + tuple(delay_positions[value] for value in delay_tick_values)
    )
    tick_labels = (
        tuple(str(value).replace("_", "") for value in neutral_order)
        + tuple(f"{value:g}" for value in keep_tick_values)
        + tuple(f"{value:g}" for value in delay_tick_values)
    )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_yscale("symlog", linthresh=0.1)
    context.capture_panel(
        "fig7",
        "c",
        plotted,
        metrics=("value",),
        groups=("series",),
    )

    multi = context.store.read("p0.multi_network")
    axis = data_axis(slots["d"], left=0.21, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        multi,
        x="seq_len",
        value="n_eff",
        colors=(NAVY,),
        xlabel="Sequence length K",
        ylabel=r"$N_{\mathrm{eff}}$",
        show_networks=True,
    )
    identity_x = np.sort(multi["seq_len"].astype(float).unique())
    axis.plot(
        identity_x,
        identity_x,
        color=GRAY,
        linestyle=":",
        linewidth=0.8,
        label=r"$N_{\mathrm{eff}}=K$",
    )
    axis.legend(frameon=False)
    context.capture_panel(
        "fig7",
        "d",
        plotted,
        metrics=("n_eff",),
        groups=("seq_len",),
    )

    weights = context.store.read("p0.multi_item_weights").copy()
    weights["relative_position"] = (
        weights["item_position"].astype(float)
        / weights["seq_len"].astype(float)
    ).round(2)
    axis = data_axis(slots["e"], left=0.21, right=0.04, bottom=0.22, top=0.27)
    plotted = network_line(
        axis,
        weights,
        x="relative_position",
        value="item_weight",
        colors=(NAVY,),
        xlabel="Relative serial position",
        ylabel="Constituent weight",
        show_networks=False,
    )
    latest = weights.loc[weights["is_latest"].astype(bool)]
    latest_summary = (
        latest.groupby("network_seed", as_index=False)["item_weight"].mean()
    )
    mean, low, high = mean_ci(latest_summary["item_weight"].to_numpy(float))
    axis.errorbar(
        [1.0],
        [mean],
        yerr=[[mean - low], [high - mean]],
        color=TEAL,
        marker="o",
        markeredgecolor=INK,
        markeredgewidth=0.5,
        capsize=2,
        zorder=5,
    )
    axis.text(
        0.98,
        0.96,
        "latest",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=TEAL,
        fontsize=7.2,
    )
    context.capture_panel(
        "fig7",
        "e",
        plotted,
        metrics=("item_weight",),
        groups=("relative_position",),
    )

    boundary = context.store.read("prog.boundary")
    boundary_network = (
        boundary.groupby(
            ["network_seed", "seq_len", "delay_ms"],
            as_index=False,
        )["rescued_fraction"]
        .mean()
    )
    axis = data_axis(slots["f"], left=0.20, right=0.10, bottom=0.22, top=0.27)
    plotted = heatmap(
        axis,
        boundary_network,
        row="seq_len",
        column="delay_ms",
        value="rescued_fraction",
        cmap=sequential_cmap(),
        vmin=0.0,
        vmax=1.0,
        colorbar_label="",
        annotate=True,
    )
    axis.set_xlabel("Delay (ms)")
    axis.set_ylabel("K")
    context.capture_panel(
        "fig7",
        "f",
        boundary_network,
        metrics=("rescued_fraction",),
        groups=("seq_len", "delay_ms"),
    )
    return fig


MAIN_BUILDERS: Mapping[
    str,
    Callable[[BuildContext, FigureContract], Figure],
] = {
    "fig1": build_fig1,
    "fig2": build_fig2,
    "fig3": build_fig3,
    "fig4": build_fig4,
    "fig5": build_fig5,
    "fig6": build_fig6,
    "fig7": build_fig7,
}
