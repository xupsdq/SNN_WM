from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.plotting.paper_fig.final_six.renderer import render_composed_figure


def _single_panel_spec() -> dict[str, Any]:
    return {
        "figure_id": "composition_probe",
        "canvas_mm": [40.0, 30.0],
        "slots": {"a": [2.0, 2.0, 36.0, 26.0]},
        "panels": {
            "a": {
                "claim": "Persisted values are rendered",
                "chart": "probe_line",
                "source": "data/probe.csv",
                "role": "exercise the public composition seam",
                "legend_owner": "none",
                "plot_bbox_mm": [6.0, 6.0, 28.0, 18.0],
            }
        },
        "layout_contract": {
            "version": "practical_layout_v1",
            "status": "candidate",
            "grid_policy": {
                "equal_row_heights": True,
                "equal_width_within_row": True,
                "panel_atomicity": True,
            },
            "approved_exceptions": [],
            "semantic_units": [
                {
                    "unit_id": "probe",
                    "panels": ["a"],
                    "role": "exercise the public composition seam",
                }
            ],
            "comparison_groups": [],
            "alignment_groups": [
                {
                    "group_id": "probe_slot",
                    "panels": ["a"],
                    "target": "slot",
                    "edges": ["top", "bottom"],
                    "rationale": "freeze the single panel slot",
                }
            ],
            "panel_geometry": {
                "a": {
                    "chart_family": "probe_line",
                    "category_slots": 2,
                    "natural_aspect": [1.0, 2.0],
                    "decoration_sides": ["left", "bottom"],
                    "visual_weight": "medium",
                    "slot_bbox_mm": [2.0, 2.0, 36.0, 26.0],
                    "plot_bbox_mm": [6.0, 6.0, 28.0, 18.0],
                }
            },
            "bar_width_policy": {
                "mode": "within_panel_only",
                "scope": "single panel",
                "tradeoff": "not applicable",
            },
            "topology": {
                "reading_direction": "row_major",
                "unit_sequence": ["probe"],
                "rationale": "single-panel probe",
            },
            "hard_constraints": ["fixed canvas"],
            "soft_targets": ["readable line"],
            "qa": {
                "final_size_render": True,
                "collision_check": True,
                "clipping_check": True,
                "alignment_measurement": True,
                "grayscale_check": True,
            },
        },
    }


def test_render_composed_figure_owns_layout_export_and_panel_qa(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0.25, 0.75]})

    def draw_probe(
        _figure: plt.Figure,
        axis: plt.Axes,
        data: pd.DataFrame,
        _panel: Mapping[str, Any],
    ) -> None:
        axis.plot(data["x"], data["y"], color="#123456")
        axis.set_xlabel("Input")
        axis.set_ylabel("Output")

    result = render_composed_figure(
        spec=_single_panel_spec(),
        frames={"a": frame},
        figure_dir=tmp_path,
        svg_hashsalt="composition-probe",
        custom_renderers={"probe_line": draw_probe},
        export_mode="matplotlib",
    )

    assert result["plot_bboxes"] == {"a": (6.0, 6.0, 28.0, 18.0)}
    for kind in ("svg", "pdf", "png"):
        assert result[kind].is_file()
    assert (tmp_path / "figures/panels/composition_probea.png").is_file()
    assert (tmp_path / "figures/qa/composition_probe_grayscale.png").is_file()
    with Image.open(result["png"]) as image:
        assert image.size == (472, 354)


def test_external_composition_callers_do_not_import_final_six_private_names() -> None:
    callers = (
        Path("src/plotting/paper_fig/paper_fig1_fig2_redesign.py"),
        Path("src/plotting/paper_fig/candidates/manuscript_fig6_order_specificity.py"),
    )
    violations: list[str] = []
    for path in callers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {
                "src.plotting.paper_fig.final_six.renderer",
                "src.plotting.paper_fig.final_six.specs",
            }:
                continue
            violations.extend(
                f"{path}:{node.lineno}:{alias.name}"
                for alias in node.names
                if alias.name.startswith("_")
            )
    assert violations == []
