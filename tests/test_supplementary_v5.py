from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.plotting.paper_fig.layout_contract import validate_layout_contract
from src.plotting.paper_fig.supplementary_v5.common import BundleReader, style_axis
from src.plotting.paper_fig.supplementary_v5.render import _plot_geometry_report
from src.plotting.paper_fig.supplementary_v5.specs import load_spec


FIGURE_IDS = tuple(f"s{index}" for index in range(1, 8))


def _inside(inner: list[float], outer: list[float]) -> bool:
    inner_left, inner_top, inner_width, inner_height = (float(value) for value in inner)
    outer_left, outer_top, outer_width, outer_height = (float(value) for value in outer)
    return (
        inner_left >= outer_left
        and inner_top >= outer_top
        and inner_left + inner_width <= outer_left + outer_width
        and inner_top + inner_height <= outer_top + outer_height
    )


def test_all_supplementary_specs_are_frozen_and_cover_32_panels() -> None:
    specs = [load_spec(figure_id) for figure_id in FIGURE_IDS]
    assert sum(len(spec["panels"]) for spec in specs) == 32
    assert [len(spec["panels"]) for spec in specs] == [4, 4, 4, 4, 4, 6, 6]
    for spec in specs:
        report = validate_layout_contract(spec)
        assert report.ok, report.failures
        assert spec["layout_contract"]["status"] == "frozen"
        assert spec["reader_contract"]["status"] == "frozen"
        assert spec["reader_contract"]["terminal_inference"]
        assert spec["reader_contract"]["forbidden_inferences"]
        assert set(spec["slots"]) == set(spec["panels"])


def test_slots_and_plot_areas_remain_inside_the_canvas() -> None:
    for figure_id in FIGURE_IDS:
        spec = load_spec(figure_id)
        canvas_bbox = [0.0, 0.0, *spec["canvas_mm"]]
        for panel_id, panel in spec["panels"].items():
            assert spec["slots"][panel_id] == panel["slot_bbox_mm"]
            assert _inside(panel["slot_bbox_mm"], canvas_bbox)
            assert _inside(panel["plot_bbox_mm"], panel["slot_bbox_mm"])
            if "colorbar_bbox_mm" in panel:
                assert _inside(panel["colorbar_bbox_mm"], panel["slot_bbox_mm"])


def test_visible_plot_geometry_is_equal_across_equal_slots() -> None:
    for figure_id in FIGURE_IDS:
        report = _plot_geometry_report(load_spec(figure_id))
        assert report["status"] == "passed"
        for check in report["equal_slot_width_checks"]:
            assert check["span_mm"] <= report["tolerance_mm"]
        for panel in report["panels"].values():
            if "colorbar_bbox_mm" not in panel:
                continue
            assert panel["colorbar_bbox_mm"][0] == panel["plot_bbox_mm"][0]
            assert panel["colorbar_bbox_mm"][2] == panel["plot_bbox_mm"][2]


def test_standard_multi_column_slots_are_not_internally_compressed() -> None:
    for figure_id in FIGURE_IDS:
        report = _plot_geometry_report(load_spec(figure_id))
        for panel in report["panels"].values():
            if "standard_two_column_width_ok" in panel:
                assert panel["standard_two_column_width_ok"] is True
            if "standard_three_column_width_ok" in panel:
                assert panel["standard_three_column_width_ok"] is True


def test_s2_uses_one_plus_three_with_one_atomic_transfer_panel() -> None:
    spec = load_spec("s2")
    assert spec["canvas_mm"] == [165.0, 102.0]
    assert spec["reading_order"] == ["a", "b", "c", "d"]
    assert spec["slots"]["a"] == [2.0, 2.0, 161.0, 48.0]
    assert [spec["slots"][panel_id][1] for panel_id in ("b", "c", "d")] == [52.0, 52.0, 52.0]
    assert [spec["slots"][panel_id][2] for panel_id in ("b", "c", "d")] == pytest.approx(
        [52.333, 52.334, 52.333], abs=0.001
    )
    assert spec["panels"]["b"]["row_order"] == ["L2 update", "Early L3"]
    assert spec["panels"]["b"]["panel_type"] == "two_endpoint_transfer"
    assert "e" not in spec["panels"]


def test_s4_uses_two_by_two_with_identity_audit_last() -> None:
    spec = load_spec("s4")
    assert spec["canvas_mm"] == [165.0, 102.0]
    assert spec["slots"] == {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 79.5, 48.0],
        "d": [83.5, 52.0, 79.5, 48.0],
    }
    assert spec["reading_order"] == ["a", "b", "c", "d"]
    assert spec["panels"]["a"]["panel_type"] == "c5_network_distribution"
    assert spec["panels"]["b"]["panel_type"] == "c5_network_distribution"
    assert spec["panels"]["c"]["panel_type"] == "c5_cohort_forest"
    assert spec["panels"]["d"]["panel_type"] == "identity_gate_matrix"
    assert spec["panels"]["d"]["column_order"] == [
        "L2 only",
        "Boundary",
        "STSP kept",
        "Fast reset",
        "Same C",
        "Sham out",
    ]
    assert spec["panels"]["d"]["column_labels"] == [
        "L2 only",
        "Bnd.",
        "STSP",
        "Fast",
        "Same C",
        "Sham",
    ]
    assert spec["panels"]["c"]["row_order"] == ["L2 K1", "L2 K5", "L3 K1", "L3 K5"]
    assert spec["layout_contract"]["topology"]["unit_sequence"] == [
        "processing",
        "successor",
        "cohort",
        "identity",
    ]


@pytest.mark.parametrize("figure_id", ["s6", "s7"])
def test_three_row_supplements_use_equal_two_column_slots(figure_id: str) -> None:
    spec = load_spec(figure_id)
    assert spec["canvas_mm"] == [165.0, 152.0]
    assert spec["reading_order"] == ["a", "b", "c", "d", "e", "f"]
    expected = {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 79.5, 48.0],
        "d": [83.5, 52.0, 79.5, 48.0],
        "e": [2.0, 102.0, 79.5, 48.0],
        "f": [83.5, 102.0, 79.5, 48.0],
    }
    assert spec["slots"] == expected


def test_style_axis_keeps_bottom_and_left_spines_visible() -> None:
    figure = Figure()
    axis = figure.subplots()
    style_axis(axis)
    assert axis.spines["bottom"].get_visible()
    assert axis.spines["left"].get_visible()


def test_supplementary_renderers_do_not_use_hatches() -> None:
    renderer_path = Path("src/plotting/paper_fig/supplementary_v5/renderers.py")
    renderer_source = renderer_path.read_text(encoding="utf-8")
    assert "hatch=" not in renderer_source
    assert "set_yticks([])" not in renderer_source


def test_vector_outputs_freeze_hash_salt_and_timestamp_metadata() -> None:
    source = Path("src/plotting/paper_fig/supplementary_v5/common.py").read_text(encoding="utf-8")
    assert 'rcparams["svg.hashsalt"] = "net_torch_supplementary_v5_v1"' in source
    assert '"CreationDate": None' in source
    assert '"ModDate": None' in source
    assert '"Date": None' in source


@pytest.mark.parametrize(
    ("figure_id", "panel_id", "row_label"),
    [("s5", "b", "Stage\nmin."), ("s6", "f", "Grid\nmin.")],
)
def test_single_row_strips_declare_their_categorical_y_label(
    figure_id: str,
    panel_id: str,
    row_label: str,
) -> None:
    panel = load_spec(figure_id)["panels"][panel_id]
    assert panel["row_label"] == row_label


def test_s3_clockwise_figure_preserves_the_declared_reading_path() -> None:
    spec = load_spec("s3")
    assert spec["reading_order"] == ["a", "b", "c", "d"]
    assert spec["slots"]["a"][:2] == [2.0, 2.0]
    assert spec["slots"]["b"][:2] == [83.5, 2.0]
    assert spec["slots"]["c"][:2] == [83.5, 52.0]
    assert spec["slots"]["d"][:2] == [2.0, 52.0]


def test_retired_c9_and_residual_self_inclusion_are_absent() -> None:
    spec_text = "\n".join(
        Path(f"src/plotting/paper_fig/specs/supplementary_v5/{figure_id}.json").read_text(encoding="utf-8")
        for figure_id in FIGURE_IDS
    ).lower()
    renderer_text = Path("src/plotting/paper_fig/supplementary_v5/renderers.py").read_text(encoding="utf-8").lower()
    builder_text = Path("src/experiments/paper_figures/supplementary_v5/builders.py").read_text(encoding="utf-8").lower()
    combined = "\n".join((spec_text, renderer_text, builder_text))
    assert "c9" not in combined
    assert "self-inclusion" not in combined
    assert "free weights" not in combined


def test_bundle_reader_allows_only_internal_frozen_tables(tmp_path: Path) -> None:
    bundle = tmp_path / "supplementary_v5"
    table = bundle / "data" / "source_data" / "s1_a.csv"
    table.parent.mkdir(parents=True)
    pd.DataFrame({"network_seed": [1000], "value": [1.0]}).to_csv(table, index=False)
    reader = BundleReader(bundle)
    frame = reader.read_csv("data/source_data/s1_a.csv", "test table")
    assert frame.to_dict(orient="records") == [{"network_seed": 1000, "value": 1.0}]
    assert reader.accesses[0]["path"] == "data/source_data/s1_a.csv"
    with pytest.raises(PermissionError):
        reader.read_csv("../outside.csv", "escape attempt")
    with pytest.raises(ValueError):
        reader.read_csv(str(table.resolve()), "absolute attempt")
