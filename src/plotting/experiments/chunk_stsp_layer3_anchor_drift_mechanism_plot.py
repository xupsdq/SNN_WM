from __future__ import annotations

from src.plotting.experiments._common import main_for, read_bundle_csv
from src.plotting.experiments._plot_builders import grouped_bar_figure, line_figure, scatter_figure


def _num(df, preferred):
    return preferred if preferred in df.columns else df.select_dtypes("number").columns[-1]


def plot_bundle(input_dir):
    changed = read_bundle_csv(input_dir, "layer3_changed_synapse_metrics.csv")
    rank = read_bundle_csv(input_dir, "layer3_changed_rank_metrics.csv")
    ping = read_bundle_csv(input_dir, "layer3_ping_coupling_metrics.csv")
    anchor = read_bundle_csv(input_dir, "layer3_state_anchor_metrics.csv")
    return {
        "changed_synapse_fraction_vs_stage": line_figure(
            changed,
            x="stage_k" if "stage_k" in changed.columns else changed.select_dtypes("number").columns[0],
            y=_num(changed, "changed_synapse_fraction"),
            hue="layer" if "layer" in changed.columns else None,
            title="Changed synapse fraction vs stage",
            ylabel="Changed fraction",
        ),
        "positive_change_mass_vs_stage": line_figure(
            changed,
            x="stage_k" if "stage_k" in changed.columns else changed.select_dtypes("number").columns[0],
            y=_num(changed, "positive_change_mass"),
            hue="layer" if "layer" in changed.columns else None,
            title="Positive change mass vs stage",
            ylabel="Positive change mass",
        ),
        "changed_rank_enrichment": grouped_bar_figure(
            rank,
            group="rank_group" if "rank_group" in rank.columns else ("layer" if "layer" in rank.columns else rank.columns[0]),
            value=_num(rank, "changed_rank_enrichment"),
            title="Changed rank enrichment",
            ylabel="Enrichment",
        ),
        "ping_coupling_with_changed_topness": scatter_figure(
            ping,
            x=_num(ping, "changed_topness"),
            y=_num(ping, "ping_coupling"),
            title="Ping coupling with changed topness",
            trend=True,
        ),
        "changed_topness_vs_chance_corrected_latest_hit": scatter_figure(
            anchor,
            x=_num(anchor, "changed_topness"),
            y=_num(anchor, "chance_corrected_latest_hit"),
            title="Changed topness vs chance-corrected latest hit",
            trend=True,
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main_for("chunk_stsp_layer3_anchor_drift_mechanism", plot_bundle, title="Chunk STSP Layer3 Anchor Drift"))
