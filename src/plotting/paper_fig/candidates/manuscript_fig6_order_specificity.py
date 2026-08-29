from __future__ import annotations

"""Plot-only reader-first Fig.6 replacement with formal temporal-order panel b."""

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.paper_fig.final_six.renderer import (
    render_composed_figure,
)
from src.plotting.paper_fig.final_six.specs import get_figure_spec


CANDIDATE_ID = "manuscript_fig6_reader_first_v3"
CANVAS_MM = (165.0, 152.0)
MM_TO_INCH = 1.0 / 25.4


class BundleReader:
    """Strict CSV reader limited to explicitly supplied persisted-result roots."""

    def __init__(self, allowed_roots: tuple[Path, ...]):
        self.allowed_roots = tuple(path.resolve() for path in allowed_roots)
        self.accesses: list[dict[str, str]] = []

    def read_csv(self, path: Path, purpose: str) -> pd.DataFrame:
        resolved = path.resolve()
        allowed = any(
            resolved == root or root in resolved.parents for root in self.allowed_roots
        )
        if not allowed or resolved.suffix.lower() != ".csv" or not resolved.is_file():
            raise PermissionError(f"Candidate source allowlist rejected {resolved}")
        self.accesses.append(
            {"path": str(resolved), "purpose": purpose, "sha256": _sha256(resolved)}
        )
        return pd.read_csv(resolved)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _network_balanced_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrices = []
    for _, part in frame.groupby("network_seed", sort=True):
        if len(part) != 36:
            raise RuntimeError(f"Every network requires 36 confusion cells, found {len(part)}")
        matrix = np.zeros((6, 6), dtype=float)
        for row in part.itertuples(index=False):
            matrix[int(row.true_order), int(row.predicted_order)] = float(row.proportion)
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise RuntimeError("Confusion rows must sum to one")
        matrices.append(matrix)
    if len(matrices) != 20:
        raise RuntimeError(f"Fig.6b requires 20 network matrices, found {len(matrices)}")
    return np.mean(np.stack(matrices, axis=0), axis=0)


def _plot_order_confusion(
    fig: plt.Figure,
    axis: plt.Axes,
    frame: pd.DataFrame,
) -> tuple[plt.Axes, plt.Text]:
    matrix = _network_balanced_matrix(frame)
    mesh = axis.imshow(
        matrix,
        cmap=get_plot_cmap("stsp_support"),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Decoded order", labelpad=7.0)
    axis.set_ylabel("Input order")
    for spine in axis.spines.values():
        spine.set_visible(False)

    x_mm, y_mm, width_mm, height_mm = (94.5, 9.8, 65.5, 1.4)
    cax = fig.add_axes(
        (
            x_mm / CANVAS_MM[0],
            1.0 - (y_mm + height_mm) / CANVAS_MM[1],
            width_mm / CANVAS_MM[0],
            height_mm / CANVAS_MM[1],
        )
    )
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(axis="x", labelsize=7.5, length=2, pad=0.0)
    colorbar.outline.set_linewidth(0.6)
    colorbar_label = fig.text(
        127.25 / CANVAS_MM[0],
        1.0 - 4.25 / CANVAS_MM[1],
        "Proportion",
        ha="center",
        va="center",
        color=get_plot_color("ink"),
    )
    return cax, colorbar_label


def _materialize_sources(
    authority: Path,
    formal: Path,
    output: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    data_dir = output / "data"
    metrics_dir = output / "metrics"
    meta_dir = output / "meta"
    figures_dir = output / "figures"
    for directory in (data_dir, metrics_dir, meta_dir, figures_dir / "qa", output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    parent_paths: list[Path] = []
    reader = BundleReader((authority, formal))
    frames: dict[str, pd.DataFrame] = {}
    for panel_id in ("a", "c", "d", "e", "f"):
        source = authority / f"data/panel_{panel_id}_plot_data.csv"
        statistics = authority / f"metrics/panel_{panel_id}_statistics.csv"
        parent_paths.extend([source, statistics])
        frame = reader.read_csv(source, f"current authority Fig.6{panel_id} plot data")
        frames[panel_id] = frame
        frame.to_csv(data_dir / f"panel_{panel_id}_plot_data.csv", index=False)
        reader.read_csv(
            statistics, f"current authority Fig.6{panel_id} statistics"
        ).to_csv(metrics_dir / f"panel_{panel_id}_statistics.csv", index=False)

    confusion_source = formal / "metrics/confusion_matrix.csv"
    primary_source = formal / "metrics/formal_primary_statistics.csv"
    validation_source = formal / "metrics/formal_validation_metrics.csv"
    formal_summary_source = formal / "summary.json"
    formal_spec_source = formal / "meta/formal_analysis_spec.json"
    formal_spec_hash_source = formal / "meta/formal_analysis_spec.sha256"
    parent_paths.extend(
        [
            confusion_source,
            primary_source,
            validation_source,
            formal_summary_source,
            formal_spec_source,
            formal_spec_hash_source,
        ]
    )
    formal_summary = json.loads(formal_summary_source.read_text(encoding="utf-8"))
    if formal_summary.get("analysis_scope") != "formal" or formal_summary.get("analysis_status") != "PASS":
        raise RuntimeError("Fig.6b source must be the validated formal analysis")
    formal_spec_sha = str(formal_summary.get("formal_spec_sha256") or "")
    if formal_spec_sha not in formal_spec_hash_source.read_text(encoding="utf-8"):
        raise RuntimeError("Fig.6b formal-spec hash does not match the formal summary")
    confusion = reader.read_csv(confusion_source, "formal Fig.6b confusion")
    confusion = confusion.loc[confusion["network_seed"].astype(int).ge(0)].copy()
    confusion.to_csv(data_dir / "panel_b_plot_data.csv", index=False)
    primary = reader.read_csv(primary_source, "formal Fig.6b primary statistics")
    if len(primary) != 1 or int(primary.iloc[0]["n_networks"]) != 20:
        raise RuntimeError("Fig.6b primary statistics must contain the 20-network formal endpoint")
    primary.to_csv(metrics_dir / "panel_b_statistics.csv", index=False)
    validation = reader.read_csv(validation_source, "formal Fig.6b validation")
    if not validation["passed"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Fig.6b formal validation contains a failed check")
    validation.to_csv(metrics_dir / "panel_b_formal_validation.csv", index=False)
    (meta_dir / "formal_analysis_spec.json").write_bytes(formal_spec_source.read_bytes())
    (meta_dir / "formal_analysis_spec.sha256").write_bytes(formal_spec_hash_source.read_bytes())

    parent_hashes = {
        path.relative_to(_repo_root()).as_posix(): _sha256(path)
        for path in sorted(parent_paths)
    }
    _write_json(meta_dir / "parent_hashes_before.json", parent_hashes)
    manifest = pd.DataFrame(
        [
            {
                "candidate_id": CANDIDATE_ID,
                "source_path": path,
                "sha256": digest,
                "use": "formal Fig.6b" if "order_specificity_formal" in path else "current authority panel",
            }
            for path, digest in parent_hashes.items()
        ]
    )
    manifest.to_csv(meta_dir / "source_manifest.csv", index=False)
    return frames, confusion, primary, parent_hashes


def render_manuscript_fig6_order_specificity(
    output_dir: str | Path,
    *,
    authority_dir: str | Path | None = None,
    formal_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo = _repo_root()
    output = Path(output_dir).resolve()
    authority = Path(authority_dir).resolve() if authority_dir else (
        repo
        / "results/paper_figure_multi_seed/final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/fig5"
    ).resolve()
    formal = Path(formal_dir).resolve() if formal_dir else (
        repo
        / "results/paper_figure_candidates/manuscript_fig6b_order_specificity_formal"
    ).resolve()
    frames, confusion, primary, parent_hashes = _materialize_sources(
        authority, formal, output
    )
    mean_accuracy = float(primary.iloc[0]["mean_accuracy"])
    _write_json(
        output / "run_config.json",
        {
            "schema": "manuscript_fig6_reader_first_plot_only_v1",
            "task": "plot_only",
            "candidate_id": CANDIDATE_ID,
            "parent_recomputation": False,
            "authority_parent_root": authority.relative_to(repo).as_posix(),
            "formal_result_root": formal.relative_to(repo).as_posix(),
            "formal_spec_sha256": (output / "meta/formal_analysis_spec.sha256").read_text(encoding="utf-8").split()[0],
            "canvas_mm": list(CANVAS_MM),
            "outputs": ["png", "svg", "pdf"],
        },
    )
    (output / "logs/plot_only.log").write_text(
        "task=plot_only\nparent_recomputation=false\nsource_validation=required\n",
        encoding="utf-8",
    )

    spec = copy.deepcopy(get_figure_spec("fig5"))
    spec["source_figure_id"] = "fig5"
    spec["figure_id"] = CANDIDATE_ID
    spec["candidate_id"] = CANDIDATE_ID
    spec["user_facing_figure"] = "Fig.6"
    spec["panels"]["a"]["plot_bbox_mm"] = [13.0, 12.0, 65.5, 32.0]
    spec["panels"]["d"]["y_label"] = "Last-item weight"
    spec["panels"]["d"]["references"] = [{"value": 0.5, "label": "50%"}]
    spec["panels"]["f"]["colorbar_label"] = "Match advantage"
    for panel_id in ("e", "f"):
        spec["panels"][panel_id]["colorbar_label_pad_pt"] = 3.25
        spec["panels"][panel_id]["colorbar_tick_pad_pt"] = 0.0
    reader_contract = spec["reader_contract"]
    reader_contract["figure_question"] = (
        "After repeated transitions, does the terminal STSP state retain multiple "
        "constituents and their experienced temporal organization?"
    )
    reader_contract["terminal_inference"] = (
        "Across independent pair, four-item order, multi-item, and load-by-delay "
        "protocols, terminal STSP remains multi-component, identifies preceding order "
        "when the item set and latest input are fixed, avoids latest-item-only collapse, "
        "and retains bounded history-specific morphology."
    )
    reader_contract["forbidden_inferences"] = [
        "structural order identification is behavioral temporal-order recall",
        "N_eff is a capacity or accessible-item count",
        "latest-item weights establish method-independent primacy or recency",
        "Layer-2 joint-state metrics and Layer-1 support morphology track the same coordinates",
        "order identification establishes a unique nonlinear binding code",
        "Fig.7 accesses the morphology defined here",
    ]
    reader_contract["semantic_units"]["b"] = (
        "identify preceding temporal order with item set and latest input fixed"
    )
    reader_contract["task_graph"] = {
        "nodes": ["a", "b", "c", "d", "e", "f"],
        "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["e", "f"]],
    }
    reader_contract["comparison_obligations"] = [
        {
            "panels": ["c", "d"],
            "comparison_basis": "shared sequence-length K positions; distinct structural metrics",
            "reader_task": "read multi-component growth together with rejection of latest-item dominance",
        },
        {
            "panels": ["e", "f"],
            "comparison_basis": "shared K-by-delay grid; distinct Layer-1 support metrics",
            "reader_task": "read footprint and history specificity as complementary boundaries",
        },
    ]
    reader_contract["protocol_boundaries"] = {
        "a": "two-item constituent-similarity protocol",
        "b": "separate four-item fixed-set, fixed-latest order protocol",
        "c_d": "longer-history Layer-2 decomposition protocol",
        "e_f": "independent Layer-1 effective-support morphology protocol",
    }
    spec["panels"]["b"] = {
        "claim": "Preceding order remains identifiable when set and latest input are fixed",
        "chart": "order_confusion",
        "source": "data/panel_b_plot_data.csv",
        "x_label": "Decoded order",
        "y_label": "Input order",
        "colorbar_label": "Proportion",
        "plot_bbox_mm": [94.5, 12.0, 65.5, 32.0],
        "role": "establish structural temporal-order identification",
        "legend_owner": "colorbar",
        "apply_standard_axis_style": False,
    }
    layout_contract = spec["layout_contract"]
    layout_contract["status"] = "candidate"
    layout_contract["panel_geometry"]["a"]["plot_bbox_mm"] = [13.0, 12.0, 65.5, 32.0]
    layout_contract["panel_geometry"]["b"] = {
        "category_slots": 6,
        "chart_family": "network_balanced_order_confusion",
        "decoration_sides": ["left", "top", "bottom"],
        "natural_aspect": [2.0, 2.1],
        "plot_bbox_mm": [94.5, 12.0, 65.5, 32.0],
        "slot_bbox_mm": list(spec["slots"]["b"]),
        "visual_weight": "high",
    }
    for unit in layout_contract["semantic_units"]:
        if unit.get("panels") == ["b"]:
            unit["role"] = "establish structural temporal-order identification"
    layout_contract["comparison_groups"] = [
        {
            "group_id": "fig6_long_history_structure",
            "panels": ["c", "d"],
            "comparison_basis": "shared sequence-length K positions; distinct metrics",
            "reader_task": "combine component growth with rejection of latest-item dominance",
        },
        {
            "group_id": "fig6_load_delay_structure",
            "panels": ["e", "f"],
            "comparison_basis": "shared K-by-delay grid; distinct metrics",
            "reader_task": "combine support footprint with history-specific morphology",
        },
    ]
    for group in layout_contract["alignment_groups"]:
        if group.get("group_id") == "fig5_row_1_plot_axes":
            group["comparison_basis"] = (
                "Both panels occupy the same 65.5 x 32 mm first-row data region; "
                "panel b reserves its horizontal colorbar inside the shared 48 mm slot."
            )
            group["rationale"] = (
                "Align constituent retention with structural order identification."
            )
    layout_contract["topology"]["rationale"] = (
        "Read constituent retention, order identification, longer-history organization, "
        "and load-by-delay boundaries in row-major order."
    )
    layout_contract["hard_constraints"].append(
        "panel b fills the shared 65.5 x 32 mm data region with one network-balanced "
        "6 x 6 confusion matrix and an in-slot top colorbar"
    )
    _write_json(output / "meta/plot_spec.json", spec)

    canvas_width, canvas_height = CANVAS_MM
    order_artists: dict[str, Any] = {}

    def draw_order_confusion(
        figure: plt.Figure,
        axis: plt.Axes,
        frame: pd.DataFrame,
        _panel_spec: Mapping[str, Any],
    ) -> None:
        colorbar_axis, colorbar_label = _plot_order_confusion(figure, axis, frame)
        order_artists["colorbar_axis"] = colorbar_axis
        order_artists["colorbar_label"] = colorbar_label

    def verify_rendered_layout(
        figure: plt.Figure,
        axes: Mapping[str, plt.Axes],
        panel_labels: Mapping[str, plt.Text],
        auxiliary_axes: Mapping[str, tuple[plt.Axes, ...]],
    ) -> None:
        renderer = figure.canvas.get_renderer()
        colorbar_axes = {
            panel_id: auxiliary_axes[panel_id][-1] for panel_id in ("e", "f")
        }
        b_colorbar_axis = order_artists["colorbar_axis"]
        b_colorbar_label = order_artists["colorbar_label"]

        def rendered_bbox_mm(artists: list[Any]) -> dict[str, float]:
            extents = [artist.get_tightbbox(renderer).extents for artist in artists]
            x0 = min(float(extent[0]) for extent in extents)
            y0 = min(float(extent[1]) for extent in extents)
            x1 = max(float(extent[2]) for extent in extents)
            y1 = max(float(extent[3]) for extent in extents)
            scale = 25.4 / figure.dpi
            return {
                "left": x0 * scale,
                "right": x1 * scale,
                "top": CANVAS_MM[1] - y1 * scale,
                "bottom": CANVAS_MM[1] - y0 * scale,
                "width": (x1 - x0) * scale,
                "height": (y1 - y0) * scale,
            }

        rendered_a = rendered_bbox_mm([axes["a"], panel_labels["a"]])
        rendered_b = rendered_bbox_mm(
            [axes["b"], b_colorbar_axis, b_colorbar_label, panel_labels["b"]]
        )
        rendered_e = rendered_bbox_mm(
            [axes["e"], colorbar_axes["e"], panel_labels["e"]]
        )
        rendered_f = rendered_bbox_mm(
            [axes["f"], colorbar_axes["f"], panel_labels["f"]]
        )

        def colorbar_label_tick_gap_mm(
            label: plt.Text, colorbar_axis: plt.Axes
        ) -> float:
            label_box = label.get_tightbbox(renderer)
            tick_boxes = [
                tick.get_tightbbox(renderer)
                for tick in colorbar_axis.get_xticklabels()
                if tick.get_visible()
            ]
            return (
                float(label_box.y0) - max(float(box.y1) for box in tick_boxes)
            ) * 25.4 / figure.dpi

        colorbar_gaps = {
            "b": colorbar_label_tick_gap_mm(b_colorbar_label, b_colorbar_axis),
            "e": colorbar_label_tick_gap_mm(
                colorbar_axes["e"].xaxis.label, colorbar_axes["e"]
            ),
            "f": colorbar_label_tick_gap_mm(
                colorbar_axes["f"].xaxis.label, colorbar_axes["f"]
            ),
        }
        row_1_height_delta = abs(rendered_a["height"] - rendered_b["height"])
        row_1_top_delta = abs(rendered_a["top"] - rendered_b["top"])
        row_1_bottom_delta = abs(rendered_a["bottom"] - rendered_b["bottom"])
        rendered_layout_qa = {
            "status": "passed",
            "a_total_bbox_mm": rendered_a,
            "b_total_bbox_mm": rendered_b,
            "e_total_bbox_mm": rendered_e,
            "f_total_bbox_mm": rendered_f,
            "colorbar_label_tick_gaps_mm": colorbar_gaps,
            "minimum_colorbar_label_tick_gap_mm": 0.5,
            "row_1_plot_bboxes_mm": {
                "a": spec["panels"]["a"]["plot_bbox_mm"],
                "b": spec["panels"]["b"]["plot_bbox_mm"],
            },
            "row_1_total_height_delta_mm": row_1_height_delta,
            "row_1_total_top_delta_mm": row_1_top_delta,
            "row_1_total_bottom_delta_mm": row_1_bottom_delta,
            "tolerance_mm": 0.5,
        }
        if (
            max(row_1_height_delta, row_1_top_delta, row_1_bottom_delta) > 0.5
            or min(colorbar_gaps.values()) < 0.5
        ):
            rendered_layout_qa["status"] = "failed"
            raise RuntimeError(f"Rendered layout failed: {rendered_layout_qa}")
        _write_json(output / "meta/rendered_layout_qa.json", rendered_layout_qa)

    rendered = render_composed_figure(
        spec=spec,
        frames={**frames, "b": confusion},
        figure_dir=output,
        svg_hashsalt=CANDIDATE_ID,
        custom_renderers={"order_confusion": draw_order_confusion},
        after_draw=verify_rendered_layout,
        export_mode="matplotlib",
    )
    png = rendered["png"]
    svg = rendered["svg"]
    pdf = rendered["pdf"]

    image = Image.open(png)
    expected_pixels = (
        round(canvas_width * MM_TO_INCH * 300),
        round(canvas_height * MM_TO_INCH * 300),
    )
    svg_root = etree.parse(str(svg)).getroot()
    namespace = {"s": "http://www.w3.org/2000/svg"}
    qa = {
        "status": "passed",
        "candidate_id": CANDIDATE_ID,
        "canvas_mm": list(CANVAS_MM),
        "png_pixels": [image.width, image.height],
        "expected_png_pixels_at_300dpi": list(expected_pixels),
        "png_size_tolerance_pass": (
            abs(image.width - expected_pixels[0]) <= 2
            and abs(image.height - expected_pixels[1]) <= 2
        ),
        "svg_text_elements": len(svg_root.xpath(".//s:text", namespaces=namespace)),
        "svg_editable_text_pass": len(svg_root.xpath(".//s:text", namespaces=namespace)) > 0,
        "formal_panel_b_networks": 20,
        "formal_panel_b_mean_accuracy": mean_accuracy,
        "formal_panel_b_confusion_cells": int(len(confusion)),
        "parent_hashes_unchanged": False,
    }

    current_parent_hashes = {
        relative: _sha256(repo / relative) for relative in parent_hashes
    }
    _write_json(output / "meta/parent_hashes_after.json", current_parent_hashes)
    qa["parent_hashes_unchanged"] = current_parent_hashes == parent_hashes
    if not (
        qa["png_size_tolerance_pass"]
        and qa["svg_editable_text_pass"]
        and qa["parent_hashes_unchanged"]
    ):
        qa["status"] = "failed"
        raise RuntimeError(f"Fig.6 candidate QA failed: {qa}")
    _write_json(output / "meta/visual_qa.json", qa)

    stats = primary.iloc[0]
    caption = f"""Fig. 6 | Terminal states retain constituents and experienced temporal order.

a, Similarity of the terminal Layer-2 joint u/x state in the two-item protocol to item A and item B templates. b, Network-balanced confusion matrix for six-way identification of the preceding A/B/C order in a separate four-item protocol, with the item set and latest D fixed; rows and columns represent the same six possible input orders. Leave-one-set-out accuracy was {float(stats['mean_accuracy']) * 100.0:.2f}% (95% Student-t CI, {float(stats['ci95_low']) * 100.0:.2f}-{float(stats['ci95_high']) * 100.0:.2f}%; chance, 16.67%; t(19) = {float(stats['t_statistic']):.2f}; P = {float(stats['p_two_sided']):.2e}); all 20 networks exceeded chance and all seven errors were CBA classified as BCA. c, Effective component number, N_eff, across sequence lengths K = 3, 5, 7 and 10. d, Last-item weight across the same lengths; the dashed reference marks 50%. e, Effective area of the Layer-1 effective-STSP-support map across sequence length and delay. f, Match advantage, defined as matched-minus-mismatched-history centered-cosine similarity of Layer-1 effective-STSP-support morphology, across the same grid. Panels a-d quantify Layer-2 joint u/x states from separate stated protocols; e and f quantify coefficient-free Layer-1 morphology. Bars, lines and cells show means across n = 20 independently trained networks; error bars show two-sided 95% Student-t CIs. Panel b is structural order identification, not behavioral recall or functional readout.
"""
    (output / "caption_draft.md").write_text(caption, encoding="utf-8")
    summary = {
        "status": "complete_candidate_not_integrated",
        "candidate_id": CANDIDATE_ID,
        "user_facing_figure": "Fig.6",
        "replacement": "reader-first Fig.6 candidate with a full-width order matrix and simplified labels",
        "manuscript_modified": False,
        "formal_result_root": formal.relative_to(repo).as_posix(),
        "authority_parent_root": authority.relative_to(repo).as_posix(),
        "formal_mean_accuracy": mean_accuracy,
        "formal_chance_accuracy": float(primary.iloc[0]["chance_accuracy"]),
        "n_networks": int(primary.iloc[0]["n_networks"]),
        "parent_hashes_unchanged": True,
        "visual_qa": qa,
        "outputs": {
            "png": png.relative_to(repo).as_posix(),
            "svg": svg.relative_to(repo).as_posix(),
            "pdf": pdf.relative_to(repo).as_posix(),
        },
    }
    _write_json(output / "summary.json", summary)

    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    _write_json(
        output / "artifact_manifest.json",
        {
            "candidate_id": CANDIDATE_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
        },
    )
    return summary


__all__ = ["CANDIDATE_ID", "render_manuscript_fig6_order_specificity"]
