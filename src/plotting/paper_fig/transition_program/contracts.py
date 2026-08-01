from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CANVAS_WIDTH_MM = 165.0
OUTER_MARGIN_MM = 2.0
CONTENT_WIDTH_MM = 161.0
ROW_HEIGHT_MM = 48.0
GUTTER_MM = 2.0

FULL = (2.0, 161.0)
HALVES = ((2.0, 79.5), (83.5, 79.5))
THIRDS = (
    (2.0, 52.333),
    (56.333, 52.333),
    (110.667, 52.333),
)
ROW_Y = (2.0, 52.0, 102.0)


@dataclass(frozen=True)
class PanelContract:
    panel_id: str
    title: str
    chart_family: str
    datasets: tuple[str, ...]
    statistic: str
    renderer: str
    position_mm: tuple[float, float, float, float]


@dataclass(frozen=True)
class FigureContract:
    figure_id: str
    display_id: str
    kind: str
    title: str
    takeaway: str
    canvas_mm: tuple[float, float]
    panels: tuple[PanelContract, ...]


def _positions(rows: Iterable[str]) -> tuple[tuple[float, float, float, float], ...]:
    positions: list[tuple[float, float, float, float]] = []
    row_list = tuple(rows)
    for row_index, layout in enumerate(row_list):
        y = ROW_Y[row_index]
        if layout == "full":
            positions.append((FULL[0], y, FULL[1], ROW_HEIGHT_MM))
        elif layout == "halves":
            positions.extend((x, y, width, ROW_HEIGHT_MM) for x, width in HALVES)
        elif layout == "thirds":
            positions.extend((x, y, width, ROW_HEIGHT_MM) for x, width in THIRDS)
        else:
            raise ValueError(f"Unsupported row layout: {layout}")
    return tuple(positions)


def _figure(
    figure_id: str,
    display_id: str,
    kind: str,
    title: str,
    takeaway: str,
    height_mm: float,
    row_layouts: tuple[str, ...],
    panels: tuple[
        tuple[str, str, str, tuple[str, ...], str, str],
        ...,
    ],
) -> FigureContract:
    positions = _positions(row_layouts)
    if len(positions) != len(panels):
        raise ValueError(
            f"{figure_id}: {len(panels)} panels do not match "
            f"{len(positions)} layout slots"
        )
    return FigureContract(
        figure_id=figure_id,
        display_id=display_id,
        kind=kind,
        title=title,
        takeaway=takeaway,
        canvas_mm=(CANVAS_WIDTH_MM, height_mm),
        panels=tuple(
            PanelContract(
                panel_id=panel_id,
                title=panel_title,
                chart_family=chart_family,
                datasets=datasets,
                statistic=statistic,
                renderer=renderer,
                position_mm=position,
            )
            for (
                panel_id,
                panel_title,
                chart_family,
                datasets,
                statistic,
                renderer,
            ), position in zip(panels, positions)
        ),
    )


FIGURE_CONTRACTS: tuple[FigureContract, ...] = (
    _figure(
        "fig1",
        "Fig. 1",
        "main",
        "An activity-silent STSP state provides a content-resolved initial condition",
        "STSP carries content without sustained firing and causally sets a time-dependent initial condition.",
        152.0,
        ("full", "halves", "thirds"),
        (
            ("a", "State-boundary protocol", "schematic", ("p0.phase_firing",), "design identity", "fig1"),
            ("b", "Population firing across phases", "trend", ("p0.phase_firing",), "network mean and 95% CI", "fig1"),
            ("c", "Delay-state content decoding", "matrix", ("fig1.delay_decode",), "network decoder accuracy", "fig1"),
            ("d", "Delay-dependent retained influence", "trend", ("fig1.delay_contrast",), "network mean and 95% CI", "fig1"),
            ("e", "STSP-state interventions", "uncertainty", ("fig1.condition",), "network effect and 95% CI", "fig1"),
            ("f", "Original-to-donor attribution transfer", "paired comparison", ("fig1.attribution",), "paired network shift", "fig1"),
        ),
    ),
    _figure(
        "fig2",
        "Fig. 2",
        "main",
        "The same later input produces a common update with a history-dependent residual",
        "Identical B produces a dominant common update plus a smaller causal history residual.",
        152.0,
        ("full", "halves", "thirds"),
        (
            ("a", "Frozen exact-B branch design", "schematic", ("fixed.trajectory",), "protocol identity", "fig2"),
            ("b", "Same-B common update", "distribution", ("fixed.scalars",), "20-network distribution", "fig2"),
            ("c", "T, L and Gamma decomposition", "decomposition", ("fixed.decomp_summary", "fixed.decomp_cell"), "network components and closure error", "fig2"),
            ("d", "Event-aligned Gamma enrichment", "relationship", ("fixed.event_cell",), "cell relationship with network inference", "fig2"),
            ("e", "L1-only transfer to L2", "uncertainty", ("fixed.scalars", "fixed.swap_summary"), "20-network donor-transfer effect", "fig2"),
            ("f", "Early L3 functional transfer", "trend", ("fixed.trajectory", "fixed.swap_summary"), "checkpoint trajectory and endpoint", "fig2"),
        ),
    ),
    _figure(
        "fig3",
        "Fig. 3",
        "main",
        "Overlap-gated re-entry converts retained support into downstream STSP write-back",
        "Overlap-aligned support changes Layer 1 recruitment and redirects Layer 2 write-back.",
        152.0,
        ("full", "halves", "thirds"),
        (
            ("a", "Cross-layer re-entry path", "schematic", ("competition.support", "overlap.matched"), "mechanism identity", "fig3"),
            ("b", "Overlap-aligned reset", "uncertainty", ("overlap.perturb_contrast",), "paired network contrast", "fig3"),
            ("c", "Pre-input retained support", "distribution", ("competition.support",), "network-stratified site distribution", "fig3"),
            ("d", "L1 transition composition and competition", "composition", ("competition.transitions", "competition.winner"), "network transition probabilities", "fig3"),
            ("e", "L1 perturbation to L2 write-back", "uncertainty", ("competition.perturb_effect", "p0.writeback", "p0.same_trial_path"), "network causal effects", "fig3"),
            ("f", "Early L3 decision deflection", "trend", ("overlap.l3_time", "overlap.decision"), "network timecourse", "fig3"),
        ),
    ),
    _figure(
        "fig4",
        "Fig. 4",
        "main",
        "The state written by B biases processing of an identical subsequent C",
        "Post-B Layer 1 STSP biases same-C Layer 2 updating and early Layer 3 state.",
        152.0,
        ("halves", "halves", "halves"),
        (
            ("a", "History-B-boundary-C protocol", "schematic", ("bridge.cell",), "frozen protocol identity", "fig4"),
            ("b", "Post-B boundary and donor construction", "paired comparison", ("bridge.boundary", "bridge.cell"), "19-network boundary displacement", "fig4"),
            ("c", "C-induced L2 donor transfer", "uncertainty", ("bridge.network",), "19-network K1/K5 effect", "fig4"),
            ("d", "Early L3 donor transfer", "uncertainty", ("bridge.network", "bridge.cell"), "19-network early endpoint", "fig4"),
            ("e", "Joint positive endpoints", "uncertainty", ("bridge.inference",), "four-endpoint joint inference", "fig4"),
            ("f", "State-transition loop", "schematic", ("bridge.inference",), "evidence-linked loop", "fig4"),
        ),
    ),
    _figure(
        "fig5",
        "Fig. 5",
        "main",
        "History-conditioned state transitions recur across successive inputs",
        "Successive inputs repeatedly displace inherited state beyond passive evolution.",
        152.0,
        ("full", "thirds", "halves"),
        (
            ("a", "Observed-next versus matched-passive", "schematic", ("p0.progressive_stage",), "counterfactual identity", "fig5"),
            ("b", "Stage displacement beyond passive", "uncertainty", ("p0.progressive_stage",), "network stage effects", "fig5"),
            ("c", "Fixed-input recurrence at K1 and K5", "uncertainty", ("fixed.scalars",), "20-network endpoint effects", "fig5"),
            ("d", "Observed and passive L2 trajectories", "trend", ("p0.progressive_stage",), "network trajectories", "fig5"),
            ("e", "Early versus late increments", "paired comparison", ("p0.progressive_network",), "paired network contrast", "fig5"),
            ("f", "Stagewise network repeatability", "matrix", ("p0.progressive_stage",), "network-by-stage signed effect", "fig5"),
        ),
    ),
    _figure(
        "fig6",
        "Fig. 6",
        "main",
        "Overlap-gated recruitment generalizes to multi-input STSP landscapes",
        "Structured support predicts recruitment most strongly on overlapping pathways.",
        102.0,
        ("thirds", "thirds"),
        (
            ("a", "Content bias across support regions", "composition", ("multi.region",), "network composition", "fig6"),
            ("b", "Global-ping support and firing", "trend", ("multi.global_ping",), "network quantile curve", "fig6"),
            ("c", "Real-input firing deflection", "trend", ("multi.real_probe",), "network quantile curve", "fig6"),
            ("d", "Support-by-overlap interaction", "relationship", ("multi.interaction",), "network interaction contrast", "fig6"),
            ("e", "High-STSP-overlap ablation", "paired comparison", ("multi.ablation",), "paired network loss", "fig6"),
            ("f", "Shuffle, threshold and coverage controls", "uncertainty", ("multi.shuffle", "multi.threshold", "multi.availability"), "network robustness controls", "fig6"),
        ),
    ),
    _figure(
        "fig7",
        "Fig. 7",
        "main",
        "Repeated transitions produce additive-dominant, sublinear, cue-accessible organization",
        "Repeated updating yields additive-dominant, residual-structured, sublinear and cue-accessible states.",
        152.0,
        ("halves", "full", "thirds"),
        (
            ("a", "Dual constituent retention", "relationship", ("pair.dual",), "pair-level similarity", "fig7"),
            ("b", "Additive fit and residual specificity", "uncertainty", ("p0.pair_network", "pair.mixture", "pair.residual"), "network geometry effects", "fig7"),
            ("c", "Neutral-ping and partial-cue access", "uncertainty", ("pair.neutral_ping", "pair.partial_cue", "pair.delay_contrast", "prog.cue"), "network access effects and delay", "fig7"),
            ("d", "Effective constituents versus K", "trend", ("p0.multi_network",), "network N_eff", "fig7"),
            ("e", "Serial-position recency", "trend", ("p0.multi_item_weights",), "network constituent weights", "fig7"),
            ("f", "K-by-delay access boundary", "matrix", ("prog.boundary",), "network rescued fraction", "fig7"),
        ),
    ),
    _figure(
        "s1",
        "Fig. S1",
        "supplement",
        "Activity-silent content is distributed, time-resolved and substrate-specific",
        "Silent content and influence are distributed and substrate-specific within a finite timescale.",
        152.0,
        ("full", "halves", "thirds"),
        (
            ("a", "Behavioral competence and errors", "matrix", ("fig1.baseline", "fig1.confusion", "fig1.class_recall"), "network behavior", "s1"),
            ("b", "Full-phase firing", "matrix", ("p0.phase_firing", "fig1.phase_rates"), "network phase firing", "s1"),
            ("c", "Decoder accuracy and macro-F1", "matrix", ("fig1.delay_curve",), "network decoder metrics", "s1"),
            ("d", "Delay interference and attribution bias", "trend", ("fig1.delay_metrics", "fig1.delay_contrast"), "network delay curves", "s1"),
            ("e", "Substrate-specific interventions", "uncertainty", ("fig1.substrate",), "network substrate effects", "s1"),
            ("f", "Original and donor attribution by substrate", "paired comparison", ("fig1.attribution", "fig1.substrate"), "paired network attribution", "s1"),
        ),
    ),
    _figure(
        "s2",
        "Fig. S2",
        "supplement",
        "Exact-B updating contains reproducible common, prestate and processing components",
        "Common prestate and processing components recur across K1/K5 histories and networks.",
        102.0,
        ("thirds", "thirds"),
        (
            ("a", "Common and residual cell distributions", "distribution", ("fixed.decomp_cell",), "cell distribution with network inference", "s2"),
            ("b", "Common cosine across histories", "matrix", ("fixed.decomp_cell",), "network-by-history matrix", "s2"),
            ("c", "Gamma fraction and directional alignment", "relationship", ("fixed.decomp_cell", "fixed.swap_cell"), "cell relationship with network summaries", "s2"),
            ("d", "Component magnitudes and closure", "decomposition", ("fixed.decomp_cell",), "network norms and closure error", "s2"),
            ("e", "Event-Gamma enrichment and coverage", "distribution", ("fixed.event_cell", "fixed.event_summary"), "network enrichment and coverage", "s2"),
            ("f", "L2-to-L3 transfer consistency", "relationship", ("fixed.scalars",), "20-network transfer relationship", "s2"),
        ),
    ),
    _figure(
        "s3",
        "Fig. S3",
        "supplement",
        "Overlap-aligned support defines a spatially selective and graded re-entry route",
        "Matched intervention, event chains and graded perturbations support a spatial re-entry route.",
        152.0,
        ("halves", "thirds", "halves"),
        (
            ("a", "Similarity and overlap entry regime", "relationship", ("overlap.entry", "overlap.matched"), "pair-level density", "s3"),
            ("b", "Natural, matched and intervention effects", "uncertainty", ("overlap.matched", "overlap.perturb_contrast"), "network contrasts", "s3"),
            ("c", "Accuracy, DPI and decision endpoints", "uncertainty", ("overlap.perturb_summary", "overlap.decision"), "network endpoints", "s3"),
            ("d", "Class-pair distribution", "matrix", ("overlap.class_pair",), "class-pair network effect", "s3"),
            ("e", "Support and spike transitions by site group", "matrix", ("competition.support", "competition.transitions"), "site-group network effects", "s3"),
            ("f", "Event-chain robustness", "uncertainty", ("p0.event_chain", "competition.nulls", "competition.window", "competition.radius"), "network robustness contrasts", "s3"),
            ("g", "Dose-dependent cross-layer path", "trend", ("competition.perturb_contrast", "competition.perturb_effect", "competition.same_winner", "competition.writeback"), "network dose-response endpoints", "s3"),
        ),
    ),
    _figure(
        "s4",
        "Fig. S4",
        "supplement",
        "The post-B Layer 1 state transmits a next-input bias at two sequence depths",
        "Same-C transfer is positive across K1/K5 families, mappings and networks.",
        152.0,
        ("halves", "halves", "halves"),
        (
            ("a", "Frozen bridge design", "schematic", ("bridge.cell",), "protocol coverage", "s4"),
            ("b", "Boundary separation and donor construction", "paired comparison", ("bridge.boundary", "bridge.cell"), "network boundary effect", "s4"),
            ("c", "L2 donor transfer hierarchy", "distribution", ("bridge.cell", "bridge.network"), "cell and network effects", "s4"),
            ("d", "Early L3 donor transfer hierarchy", "distribution", ("bridge.cell", "bridge.network"), "cell and network effects", "s4"),
            ("e", "Family, mapping and prefix strata", "matrix", ("bridge.cell",), "family-by-mapping effects", "s4"),
            ("f", "Joint endpoint inference", "uncertainty", ("bridge.inference",), "19-network conjunction", "s4"),
        ),
    ),
    _figure(
        "s5",
        "Fig. S5",
        "supplement",
        "Successive inputs drive distributed and order-dependent evolution beyond passive decay",
        "Repeated displacement spans layers and variables with stage and order dependence.",
        102.0,
        ("thirds", "thirds"),
        (
            ("a", "Layer and state-variable displacement", "matrix", ("prog.update",), "network stage effects", "s5"),
            ("b", "L2 observed and passive trajectories", "trend", ("p0.progressive_stage",), "network trajectories", "s5"),
            ("c", "Early-minus-late increments", "paired comparison", ("p0.progressive_network",), "paired network effects", "s5"),
            ("d", "Constituent weights across stages", "matrix", ("prog.weights",), "network item weights", "s5"),
            ("e", "K-by-delay order sensitivity", "matrix", ("prog.order",), "network order index", "s5"),
            ("f", "Network distributions across stages", "distribution", ("p0.progressive_stage",), "network effects", "s5"),
        ),
    ),
    _figure(
        "s6",
        "Fig. S6",
        "supplement",
        "Multi-input support fields preserve spatial content structure and overlap-gated recruitment",
        "Support-to-recruitment remains spatially organized and overlap-gated across controls.",
        102.0,
        ("thirds", "thirds"),
        (
            ("a", "Peak, valley and random content bias", "composition", ("multi.region",), "network composition", "s6"),
            ("b", "Score-quantile firing endpoints", "trend", ("multi.global_ping",), "network quantile curves", "s6"),
            ("c", "Real-input early-window robustness", "trend", ("multi.window_probe",), "network window curves", "s6"),
            ("d", "Quantile-by-overlap interaction surface", "matrix", ("multi.threshold", "multi.interaction"), "network interaction surface", "s6"),
            ("e", "Ablation versus matched removal", "paired comparison", ("multi.ablation_pair",), "paired network loss", "s6"),
            ("f", "Shuffle, availability and complete case", "uncertainty", ("multi.shuffle", "multi.availability", "multi.threshold"), "network null and coverage", "s6"),
        ),
    ),
    _figure(
        "s7",
        "Fig. S7",
        "supplement",
        "Pair successor states are additive-dominant, residual-specific and accessible within a finite range",
        "Pair states are mostly additive with residual specificity and finite cue access.",
        152.0,
        ("halves", "thirds", "halves"),
        (
            ("a", "Layer-by-variable dual retention", "relationship", ("pair.dual", "pair.delay_layer"), "pair-level similarity", "s7"),
            ("b", "Full mixture-model comparison", "uncertainty", ("pair.model_comparison",), "network cross-validated fit", "s7"),
            ("c", "Pair and residual specificity", "uncertainty", ("pair.specificity", "pair.residual", "p0.pair_network"), "network specificity effects", "s7"),
            ("d", "Cross-fit interaction versus null", "distribution", ("pair.interaction", "pair.null"), "network calibrated null", "s7"),
            ("e", "Pair outcomes across delay", "trend", ("pair.delay_layer",), "network delay curves", "s7"),
            ("f", "Completion gain across delay", "trend", ("pair.delay_contrast",), "network completion gain", "s7"),
            ("g", "Ping sweep and cue asymmetry", "matrix", ("pair.ping_sweep", "pair.partial_cue"), "network access calibration", "s7"),
        ),
    ),
    _figure(
        "s8",
        "Fig. S8",
        "supplement",
        "Multi-input states have structured but nonuniform geometry and access",
        "Multi-input organization is non-flat, sublinear, recency-reorganized and condition dependent.",
        152.0,
        ("thirds", "halves", "halves"),
        (
            ("a", "Effective constituents: linear versus saturating", "trend", ("p0.multi_network",), "network N_eff and model reference", "s8"),
            ("b", "Peak-valley, Gini and structured sequences", "uncertainty", ("prog.peak_summary",), "network morphology endpoints", "s8"),
            ("c", "Latest and earlier mass across delay", "trend", ("prog.access",), "network delay curves", "s8"),
            ("d", "K-by-delay functional boundary", "matrix", ("prog.boundary",), "network count and rescued fraction", "s8"),
            ("e", "Cue-specific access hierarchy", "uncertainty", ("prog.cue",), "network cue effects", "s8"),
            ("f", "True-order index", "matrix", ("prog.order",), "network order effect", "s8"),
            ("g", "Morphology-function coupling", "relationship", ("prog.coupling",), "sequence relationship with network slopes", "s8"),
        ),
    ),
)


FIGURE_BY_ID = {contract.figure_id: contract for contract in FIGURE_CONTRACTS}


def validate_contracts() -> None:
    expected_ids = {
        "fig1",
        "fig2",
        "fig3",
        "fig4",
        "fig5",
        "fig6",
        "fig7",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
    }
    observed_ids = {contract.figure_id for contract in FIGURE_CONTRACTS}
    if observed_ids != expected_ids:
        raise ValueError(
            f"Figure contract identity mismatch: {sorted(observed_ids)}"
        )
    if len(FIGURE_BY_ID) != len(FIGURE_CONTRACTS):
        raise ValueError("Duplicate figure ids")
    total_panels = 0
    for contract in FIGURE_CONTRACTS:
        expected_height = 102.0 if len({p.position_mm[1] for p in contract.panels}) == 2 else 152.0
        if contract.canvas_mm != (CANVAS_WIDTH_MM, expected_height):
            raise ValueError(f"{contract.figure_id}: invalid canvas {contract.canvas_mm}")
        panel_ids = tuple(panel.panel_id for panel in contract.panels)
        expected_panel_ids = tuple(chr(ord("a") + index) for index in range(len(panel_ids)))
        if panel_ids != expected_panel_ids:
            raise ValueError(
                f"{contract.figure_id}: panel order {panel_ids} != {expected_panel_ids}"
            )
        _validate_row_fill(contract)
        total_panels += len(contract.panels)
    if total_panels != 93:
        raise ValueError(f"Expected 93 panels, observed {total_panels}")


def _validate_row_fill(contract: FigureContract) -> None:
    by_row: dict[float, list[PanelContract]] = {}
    for panel in contract.panels:
        by_row.setdefault(panel.position_mm[1], []).append(panel)
    for y, panels in by_row.items():
        ordered = sorted(panels, key=lambda panel: panel.position_mm[0])
        if abs(ordered[0].position_mm[0] - OUTER_MARGIN_MM) > 1e-6:
            raise ValueError(f"{contract.figure_id} row {y}: left edge is not 2 mm")
        right = ordered[-1].position_mm[0] + ordered[-1].position_mm[2]
        if abs(right - (OUTER_MARGIN_MM + CONTENT_WIDTH_MM)) > 1e-3:
            raise ValueError(f"{contract.figure_id} row {y}: right edge is not 163 mm")
        for left_panel, right_panel in zip(ordered, ordered[1:]):
            gap = (
                right_panel.position_mm[0]
                - left_panel.position_mm[0]
                - left_panel.position_mm[2]
            )
            if abs(gap - GUTTER_MM) > 2e-3:
                raise ValueError(
                    f"{contract.figure_id} row {y}: gutter {gap} != 2 mm"
                )


validate_contracts()
