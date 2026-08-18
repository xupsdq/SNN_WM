from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting.paper_fig.layout_contract import validate_layout_contract


EXPECTED_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
}

# Reader-first profiles with three 48 mm-high rows on a 165 x 152 mm canvas.
FIG5_V3_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 52.333, 48.0],
    "d": [56.333, 52.0, 52.334, 48.0],
    "e": [110.667, 52.0, 52.333, 48.0],
}
FIG5_V3_PLOTS = {
    "a": [14.0, 13.0, 65.5, 30.0],
    "b": [95.5, 13.0, 65.5, 30.0],
    "c": [17.0, 62.0, 34.333, 28.0],
    "d": [69.333, 62.0, 36.334, 28.0],
    "e": [123.667, 62.0, 36.333, 28.0],
}
FIG5_V2_SLOTS = {
    "a": [2.0, 2.0, 161.0, 48.0],
    "b": [2.0, 52.0, 79.5, 48.0],
    "c": [83.5, 52.0, 79.5, 48.0],
    "d": [2.0, 102.0, 79.5, 48.0],
    "e": [83.5, 102.0, 79.5, 48.0],
}
FIG7_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
    "e": [2.0, 102.0, 79.5, 48.0],
    "f": [83.5, 102.0, 79.5, 48.0],
}
SUPP_2_2_1_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
    "e": [2.0, 102.0, 161.0, 48.0],
}


def _matches_slots(slots: Mapping[str, Any], expected: Mapping[str, list[float]]) -> bool:
    return set(slots) == set(expected) and all(
        [float(value) for value in slots[panel_id]] == bbox
        for panel_id, bbox in expected.items()
    )


def _plots_align(left: Any, right: Any, *, match_width: bool = True, tolerance: float = 1.1e-3) -> bool:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    indices = (1, 2, 3) if match_width else (1, 3)
    return len(left_values) == 4 and len(right_values) == 4 and all(
        abs(left_values[index] - right_values[index]) <= tolerance
        for index in indices
    )


def _profile_for(spec: Mapping[str, Any]) -> tuple[dict[str, list[float]], list[float], list[tuple[str, str]]]:
    canvas = [float(value) for value in spec.get("canvas_mm", [])]
    slots = spec.get("slots") or {}
    if canvas == [165.0, 152.0] and _matches_slots(slots, SUPP_2_2_1_SLOTS):
        return SUPP_2_2_1_SLOTS, canvas, [("a", "b"), ("c", "d")]
    if canvas == [165.0, 102.0] and _matches_slots(slots, FIG5_V3_SLOTS):
        return FIG5_V3_SLOTS, canvas, [("a", "b"), ("c", "d"), ("d", "e")]
    if canvas == [165.0, 152.0] and _matches_slots(slots, FIG5_V2_SLOTS):
        return FIG5_V2_SLOTS, canvas, [("b", "c"), ("d", "e")]
    if canvas == [165.0, 152.0] and _matches_slots(slots, FIG7_SLOTS):
        return FIG7_SLOTS, canvas, [("c", "d"), ("e", "f")]
    return EXPECTED_SLOTS, [165.0, 102.0], [("a", "b"), ("c", "d")]


def audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    contract_report = validate_layout_contract(spec)
    failures = list(contract_report.failures)
    expected_slots, expected_canvas, aligned_rows = _profile_for(spec)
    canvas = [float(value) for value in spec.get("canvas_mm", [])]
    if canvas != expected_canvas:
        failures.append(f"canvas must be exactly {expected_canvas[0]:g} x {expected_canvas[1]:g} mm")
    slots = spec.get("slots") or {}
    for panel_id, expected in expected_slots.items():
        actual = [float(value) for value in slots.get(panel_id, [])]
        if actual != expected:
            failures.append(f"slot {panel_id} must be {expected}")
        panel = (spec.get("panels") or {}).get(panel_id) or {}
        plot = [float(value) for value in panel.get("plot_bbox_mm", [])]
        if len(plot) != 4 or len(actual) != 4:
            failures.append(f"panel {panel_id} must declare four plot and slot coordinates")
            continue
        if not (
            plot[0] >= actual[0]
            and plot[1] >= actual[1]
            and plot[0] + plot[2] <= actual[0] + actual[2]
            and plot[1] + plot[3] <= actual[1] + actual[3]
        ):
            failures.append(f"panel {panel_id} plot area escapes its slot")
        if expected_canvas == [165.0, 102.0] and _matches_slots(slots, FIG5_V3_SLOTS) and plot != FIG5_V3_PLOTS[panel_id]:
            failures.append(f"plot {panel_id} must be exactly {FIG5_V3_PLOTS[panel_id]}")
    for left, right in aligned_rows:
        left_plot = (spec.get("panels") or {}).get(left, {}).get("plot_bbox_mm", [])
        right_plot = (spec.get("panels") or {}).get(right, {}).get("plot_bbox_mm", [])
        match_width = not (
            expected_canvas == [165.0, 102.0]
            and _matches_slots(slots, FIG5_V3_SLOTS)
            and (left, right) == ("c", "d")
        )
        if not _plots_align(left_plot, right_plot, match_width=match_width):
            target = "top/bottom/width" if match_width else "top/bottom"
            failures.append(f"plot {target} must align for row {left}|{right}")
    if expected_canvas == [165.0, 152.0] and set(slots) == set(FIG7_SLOTS):
        a_plot = (spec.get("panels") or {}).get("a", {}).get("plot_bbox_mm", [])
        b_plot = (spec.get("panels") or {}).get("b", {}).get("plot_bbox_mm", [])
        children = (spec.get("panels") or {}).get("a", {}).get("child_plot_bboxes_mm", [])
        for child in children:
            if float(child[1]) != float(b_plot[1]) or float(child[3]) != float(b_plot[3]):
                failures.append("panel a child axes must align (top/bottom) with panel b plot area")
        if float(a_plot[1]) != float(b_plot[1]) or float(a_plot[3]) != float(b_plot[3]):
            failures.append("panel a parent plot area must align (top/bottom) with panel b plot area")
    if canvas == [165.0, 102.0] and _matches_slots(slots, FIG5_V3_SLOTS):
        alignment_pairs = (("a", "b"), ("c", "d"), ("d", "e"))
    elif canvas == [165.0, 152.0] and _matches_slots(slots, SUPP_2_2_1_SLOTS):
        alignment_pairs = (("a", "b"), ("c", "d"))
    elif canvas == [165.0, 152.0] and _matches_slots(slots, FIG5_V2_SLOTS):
        alignment_pairs = (("b", "c"), ("d", "e"))
    elif canvas == [165.0, 152.0] and _matches_slots(slots, FIG7_SLOTS):
        alignment_pairs = (("c", "d"), ("e", "f"))
    else:
        alignment_pairs = (("a", "b"), ("c", "d"))
    plot_alignment = {}
    for left, right in alignment_pairs:
        match_width = not (
            canvas == [165.0, 102.0]
            and _matches_slots(slots, FIG5_V3_SLOTS)
            and (left, right) == ("c", "d")
        )
        plot_alignment[f"{left}_{right}"] = _plots_align(
            (spec.get("panels") or {}).get(left, {}).get("plot_bbox_mm", []),
            (spec.get("panels") or {}).get(right, {}).get("plot_bbox_mm", []),
            match_width=match_width,
        )
    return {
        "schema": "layout_profile_audit_v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "contract_passes": contract_report.passes,
        "contract_warnings": contract_report.warnings,
        "canvas_mm": canvas,
        "slots": slots,
        "plot_alignment": plot_alignment,
        "plot_bboxes": {
            panel_id: (spec.get("panels") or {}).get(panel_id, {}).get("plot_bbox_mm", [])
            for panel_id in expected_slots
        },
        "fig5_v3_bottom_geometry": (
            {panel_id: (spec.get("panels") or {}).get(panel_id, {}).get("plot_bbox_mm", []) for panel_id in ("c", "d", "e")} ==
            {panel_id: FIG5_V3_PLOTS[panel_id] for panel_id in ("c", "d", "e")}
            if canvas == [165.0, 102.0] and _matches_slots(slots, FIG5_V3_SLOTS)
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a paper-figure layout profile.")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.spec.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    report = audit(spec)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
