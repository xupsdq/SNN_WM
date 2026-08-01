from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any, Mapping

from matplotlib.container import BarContainer

from src.plotting.paper_fig.utils import write_json


DEFAULT_TOLERANCE_MM = 0.05


def apply_row_bar_width_contract(
    fig,
    axes: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Set vertical bar patches to declared physical widths after layout solving.

    The operation changes rectangle width around each existing bar centre.  It
    deliberately leaves axes positions, limits, category centres, and all
    non-bar artists untouched.
    """
    contract = spec.get("row_bar_width_contract")
    if contract in (None, {}):
        return {}
    if not isinstance(contract, Mapping):
        raise TypeError("row_bar_width_contract must be a mapping")
    if str(contract.get("unit", "mm")).lower() != "mm":
        raise ValueError("row_bar_width_contract.unit must be 'mm'")
    groups = contract.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("row_bar_width_contract.groups must be a non-empty list")

    tolerance_mm = float(contract.get("tolerance_mm", DEFAULT_TOLERANCE_MM))
    if tolerance_mm <= 0:
        raise ValueError("row_bar_width_contract.tolerance_mm must be positive")

    fig.canvas.draw()
    axes_bounds_before = {panel_id: tuple(ax.get_position().bounds) for panel_id, ax in axes.items()}
    panels = spec.get("panels") or {}
    report_groups: list[dict[str, Any]] = []

    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise TypeError(f"row_bar_width_contract.groups[{index}] must be a mapping")
        group_id = str(group.get("group_id") or f"group_{index + 1}")
        panel_ids = [str(panel_id).upper() for panel_id in (group.get("panels") or [])]
        if len(panel_ids) < 2:
            raise ValueError(f"row bar-width group {group_id} must contain at least two panels")
        unknown = [panel_id for panel_id in panel_ids if panel_id not in axes or panel_id not in panels]
        if unknown:
            raise KeyError(f"row bar-width group {group_id} has unavailable panels: {unknown}")
        _require_same_row(group_id, panel_ids, panels)

        target_mm = float(group.get("target_mm", 0.0))
        if target_mm <= 0:
            raise ValueError(f"row bar-width group {group_id} requires positive target_mm")

        panel_reports: list[dict[str, Any]] = []
        for panel_id in panel_ids:
            root_ax = axes[panel_id]
            bar_axes = _bar_axes(fig, root_ax)
            patch_entries = _bar_patch_entries(bar_axes)
            if not patch_entries:
                raise RuntimeError(f"row bar-width group {group_id} panel {panel_id} has no vertical BarContainer")

            axes_limits_before = {id(ax): tuple(ax.get_xlim()) for ax, _patch in patch_entries}
            centers_before = [_patch_center(patch) for _ax, patch in patch_entries]
            widths_before = [_patch_width_mm(fig, ax, patch) for ax, patch in patch_entries]

            for ax, patch in patch_entries:
                center = _patch_center(patch)
                width_data = _data_width_for_mm(fig, ax, center, target_mm)
                patch.set_x(center - width_data / 2.0)
                patch.set_width(width_data)

            fig.canvas.draw()
            widths_after = [_patch_width_mm(fig, ax, patch) for ax, patch in patch_entries]
            centers_after = [_patch_center(patch) for _ax, patch in patch_entries]
            axes_limits_after = {id(ax): tuple(ax.get_xlim()) for ax, _patch in patch_entries}
            max_error_mm = max(abs(width - target_mm) for width in widths_after)
            if max_error_mm > tolerance_mm:
                raise RuntimeError(
                    f"row bar-width group {group_id} panel {panel_id} misses target by "
                    f"{max_error_mm:.4f} mm (tolerance {tolerance_mm:.4f} mm)"
                )
            if any(abs(before - after) > 1e-12 for before, after in zip(centers_before, centers_after)):
                raise RuntimeError(f"row bar-width normalization moved category centres in {panel_id}")
            if axes_limits_before != axes_limits_after:
                raise RuntimeError(f"row bar-width normalization changed x limits in {panel_id}")

            panel_reports.append(
                {
                    "panel_id": panel_id,
                    "bar_count": len(patch_entries),
                    "before_mm": _width_summary(widths_before),
                    "after_mm": _width_summary(widths_after),
                    "max_target_error_mm": float(max_error_mm),
                    "centres_unchanged": True,
                    "x_limits_unchanged": True,
                }
            )

        report_groups.append(
            {
                "group_id": group_id,
                "panels": panel_ids,
                "target_mm": target_mm,
                "selection_rule": str(group.get("selection_rule") or "declared"),
                "panels_report": panel_reports,
            }
        )

    axes_bounds_after = {panel_id: tuple(ax.get_position().bounds) for panel_id, ax in axes.items()}
    if axes_bounds_before != axes_bounds_after:
        raise RuntimeError("row bar-width normalization changed panel axes positions")
    return {
        "figure_id": str(spec.get("figure_id", "")),
        "contract_id": str(contract.get("contract_id") or "row_local_physical_bar_width_v1"),
        "unit": "mm",
        "tolerance_mm": tolerance_mm,
        "layout_axes_positions_unchanged": True,
        "groups": report_groups,
    }


def write_row_bar_width_report(report: Mapping[str, Any], output_dir: str | Path) -> Path | None:
    if not report:
        return None
    return write_json(report, Path(output_dir) / "row_bar_width_measurements.json")


def _require_same_row(
    group_id: str,
    panel_ids: list[str],
    panels: Mapping[str, Any],
    *,
    tolerance_mm: float = 0.05,
) -> None:
    y_values = []
    for panel_id in panel_ids:
        position = (panels.get(panel_id) or {}).get("position_mm") or {}
        if "y" not in position:
            raise ValueError(f"row bar-width group {group_id} panel {panel_id} lacks position_mm.y")
        y_values.append(float(position["y"]))
    if max(y_values) - min(y_values) > tolerance_mm:
        raise ValueError(f"row bar-width group {group_id} crosses rows: {dict(zip(panel_ids, y_values))}")


def _bar_axes(fig, root_ax) -> list[Any]:
    candidates = [root_ax]
    candidates.extend(list(getattr(root_ax, "child_axes", [])))
    candidates.extend(list(getattr(root_ax, "paper_fig_child_axes", [])))
    root_bbox = root_ax.bbox
    for other in fig.axes:
        if other is root_ax:
            continue
        if root_bbox.contains(other.bbox.x0, other.bbox.y0) and root_bbox.contains(other.bbox.x1, other.bbox.y1):
            candidates.append(other)
    out: list[Any] = []
    for ax in candidates:
        if ax not in out and any(isinstance(container, BarContainer) for container in ax.containers):
            out.append(ax)
    return out


def _bar_patch_entries(bar_axes: list[Any]) -> list[tuple[Any, Any]]:
    entries: list[tuple[Any, Any]] = []
    for ax in bar_axes:
        if str(ax.get_xscale()).lower() != "linear":
            raise ValueError("physical bar-width normalization currently requires a linear x axis")
        for container in ax.containers:
            if not isinstance(container, BarContainer):
                continue
            orientation = str(getattr(container, "orientation", "vertical") or "vertical").lower()
            if orientation != "vertical":
                continue
            entries.extend((ax, patch) for patch in container.patches if patch.get_visible())
    return entries


def _patch_center(patch) -> float:
    return float(patch.get_x()) + float(patch.get_width()) / 2.0


def _data_width_for_mm(fig, ax, center_x: float, target_mm: float) -> float:
    y0, y1 = ax.get_ylim()
    y_ref = (float(y0) + float(y1)) / 2.0
    center_px = ax.transData.transform((center_x, y_ref))
    half_width_px = target_mm / 25.4 * float(fig.dpi) / 2.0
    inverse = ax.transData.inverted()
    left = inverse.transform((center_px[0] - half_width_px, center_px[1]))[0]
    right = inverse.transform((center_px[0] + half_width_px, center_px[1]))[0]
    return abs(float(right) - float(left))


def _patch_width_mm(fig, ax, patch) -> float:
    y0, y1 = ax.get_ylim()
    y_ref = (float(y0) + float(y1)) / 2.0
    left_px = ax.transData.transform((float(patch.get_x()), y_ref))[0]
    right_px = ax.transData.transform((float(patch.get_x()) + float(patch.get_width()), y_ref))[0]
    return abs(float(right_px) - float(left_px)) / float(fig.dpi) * 25.4


def _width_summary(widths: list[float]) -> dict[str, float]:
    return {
        "minimum": float(min(widths)),
        "median": float(median(widths)),
        "maximum": float(max(widths)),
    }


__all__ = ["apply_row_bar_width_contract", "write_row_bar_width_report"]
