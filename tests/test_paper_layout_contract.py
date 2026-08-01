from __future__ import annotations

from src.plotting.paper_fig.layout_contract import validate_layout_contract


def _valid_spec():
    return {
        "panels": {"A": {}, "B": {}},
        "layout_contract": {
            "version": "practical_layout_v1",
            "status": "candidate",
            "semantic_units": [
                {"unit_id": "entry", "panels": ["A"], "role": "entry"},
                {"unit_id": "comparison", "panels": ["B"], "role": "evidence"},
            ],
            "comparison_groups": [],
            "alignment_groups": [
                {
                    "group_id": "outer_row",
                    "panels": ["A", "B"],
                    "target": "slot",
                    "edges": ["top", "bottom"],
                    "rationale": "Preserve the reading row without forcing data-area equality.",
                }
            ],
            "panel_geometry": {
                "A": {
                    "chart_family": "stacked_bar",
                    "category_slots": 3,
                    "natural_aspect": [1.2, 1.8],
                    "decoration_sides": ["left", "top", "bottom"],
                    "visual_weight": "high",
                },
                "B": {
                    "chart_family": "line",
                    "category_slots": 5,
                    "natural_aspect": [1.1, 1.8],
                    "decoration_sides": ["left", "bottom"],
                    "visual_weight": "low",
                },
            },
            "bar_width_policy": {
                "mode": "within_panel_only",
                "scope": "unrelated panels",
                "tradeoff": "Physical bar width may differ across unrelated panels.",
            },
            "topology": {
                "reading_direction": "row_major",
                "unit_sequence": ["entry", "comparison"],
                "rationale": "Read the entry before the evidence panel.",
            },
            "hard_constraints": ["no collision"],
            "soft_targets": ["balanced visual weight"],
            "qa": {
                "final_size_render": True,
                "collision_check": True,
                "clipping_check": True,
                "alignment_measurement": True,
                "grayscale_check": True,
            },
        },
    }


def test_valid_contract_passes():
    report = validate_layout_contract(_valid_spec())
    assert report.ok, report.failures


def test_plot_area_alignment_requires_comparison_basis():
    spec = _valid_spec()
    group = spec["layout_contract"]["alignment_groups"][0]
    group["target"] = "plot_area"
    report = validate_layout_contract(spec)
    assert not report.ok
    assert any("comparison_basis" in failure for failure in report.failures)


def test_semantic_units_must_cover_panels_once():
    spec = _valid_spec()
    spec["layout_contract"]["semantic_units"][1]["panels"] = ["A"]
    report = validate_layout_contract(spec)
    assert not report.ok
    assert any("multiple semantic units" in failure for failure in report.failures)
    assert any("missing from semantic units" in failure for failure in report.failures)
