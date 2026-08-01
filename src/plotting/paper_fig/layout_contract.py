from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "practical_layout_v1"
ALLOWED_ALIGNMENT_TARGETS = {"slot", "plot_area", "legend", "panel_label"}
ALLOWED_EDGES = {"left", "right", "top", "bottom"}
ALLOWED_VISUAL_WEIGHTS = {"low", "medium", "high"}
ALLOWED_BAR_MODES = {"preserve_slots", "proportional_panel_width", "within_panel_only"}
ALLOWED_READING_DIRECTIONS = {"row_major", "column_major", "explicit"}


@dataclass
class LayoutContractReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_layout_contract(spec: Mapping[str, Any]) -> LayoutContractReport:
    """Validate a practical layout contract before axes are constructed."""
    report = LayoutContractReport()
    contract = spec.get("layout_contract")
    if contract is None:
        report.warnings.append("layout_contract is not declared")
        return report
    if not isinstance(contract, Mapping):
        report.failures.append("layout_contract must be a mapping")
        return report

    panels = set(str(panel_id) for panel_id in (spec.get("panels") or {}))
    if not panels:
        report.failures.append("spec must define panels before layout_contract validation")
        return report

    _check_version(contract, report)
    unit_ids = _check_semantic_units(contract, panels, report)
    _check_comparison_groups(contract, panels, report)
    _check_alignment_groups(contract, panels, report)
    _check_panel_geometry(contract, panels, report)
    _check_bar_policy(contract, report)
    _check_topology(contract, unit_ids, report)
    _check_constraint_lists(contract, report)
    _check_qa(contract, report)
    return report


def _check_version(contract: Mapping[str, Any], report: LayoutContractReport) -> None:
    version = str(contract.get("version") or "")
    if version == CONTRACT_VERSION:
        report.passes.append(f"layout contract version is {CONTRACT_VERSION}")
    else:
        report.failures.append(f"layout_contract.version must be {CONTRACT_VERSION!r}")
    status = str(contract.get("status") or "")
    if status in {"candidate", "frozen", "approved"}:
        report.passes.append(f"layout status is {status}")
    else:
        report.failures.append("layout_contract.status must be candidate, frozen, or approved")


def _check_semantic_units(
    contract: Mapping[str, Any],
    panels: set[str],
    report: LayoutContractReport,
) -> set[str]:
    units = contract.get("semantic_units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)) or not units:
        report.failures.append("layout_contract.semantic_units must be a non-empty list")
        return set()
    unit_ids: set[str] = set()
    assigned: list[str] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            report.failures.append(f"semantic_units[{index}] must be a mapping")
            continue
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            report.failures.append(f"semantic_units[{index}] requires unit_id")
        elif unit_id in unit_ids:
            report.failures.append(f"semantic unit id is duplicated: {unit_id}")
        else:
            unit_ids.add(unit_id)
        members = _panel_list(unit.get("panels"), f"semantic unit {unit_id or index}", report)
        assigned.extend(members)
        unknown = set(members) - panels
        if unknown:
            report.failures.append(f"semantic unit {unit_id or index} has unknown panels: {sorted(unknown)}")
        if not str(unit.get("role") or "").strip():
            report.failures.append(f"semantic unit {unit_id or index} requires role")
    duplicates = sorted({panel for panel in assigned if assigned.count(panel) > 1})
    if duplicates:
        report.failures.append(f"panels occur in multiple semantic units: {duplicates}")
    missing = panels - set(assigned)
    if missing:
        report.failures.append(f"panels missing from semantic units: {sorted(missing)}")
    if not duplicates and not missing and set(assigned) == panels:
        report.passes.append("semantic units cover every panel exactly once")
    return unit_ids


def _check_comparison_groups(
    contract: Mapping[str, Any],
    panels: set[str],
    report: LayoutContractReport,
) -> None:
    groups = contract.get("comparison_groups") or []
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        report.failures.append("layout_contract.comparison_groups must be a list")
        return
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            report.failures.append(f"comparison_groups[{index}] must be a mapping")
            continue
        group_id = str(group.get("group_id") or index)
        members = _panel_list(group.get("panels"), f"comparison group {group_id}", report)
        if len(members) < 2:
            report.failures.append(f"comparison group {group_id} must contain at least two panels")
        unknown = set(members) - panels
        if unknown:
            report.failures.append(f"comparison group {group_id} has unknown panels: {sorted(unknown)}")
        if not str(group.get("comparison_basis") or "").strip():
            report.failures.append(f"comparison group {group_id} requires comparison_basis")
        if not str(group.get("reader_task") or "").strip():
            report.failures.append(f"comparison group {group_id} requires reader_task")


def _check_alignment_groups(
    contract: Mapping[str, Any],
    panels: set[str],
    report: LayoutContractReport,
) -> None:
    groups = contract.get("alignment_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
        report.failures.append("layout_contract.alignment_groups must be a non-empty list")
        return
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            report.failures.append(f"alignment_groups[{index}] must be a mapping")
            continue
        group_id = str(group.get("group_id") or index)
        members = _panel_list(group.get("panels"), f"alignment group {group_id}", report)
        unknown = set(members) - panels
        if unknown:
            report.failures.append(f"alignment group {group_id} has unknown panels: {sorted(unknown)}")
        target = str(group.get("target") or "")
        if target not in ALLOWED_ALIGNMENT_TARGETS:
            report.failures.append(
                f"alignment group {group_id} target must be one of {sorted(ALLOWED_ALIGNMENT_TARGETS)}"
            )
        edges = {str(edge) for edge in (group.get("edges") or [])}
        if not edges or not edges.issubset(ALLOWED_EDGES):
            report.failures.append(f"alignment group {group_id} has invalid edges: {sorted(edges)}")
        if not str(group.get("rationale") or "").strip():
            report.failures.append(f"alignment group {group_id} requires rationale")
        if target == "plot_area" and len(members) > 1 and not str(group.get("comparison_basis") or "").strip():
            report.failures.append(
                f"plot-area alignment group {group_id} requires an explicit comparison_basis"
            )


def _check_panel_geometry(
    contract: Mapping[str, Any],
    panels: set[str],
    report: LayoutContractReport,
) -> None:
    geometry = contract.get("panel_geometry")
    if not isinstance(geometry, Mapping):
        report.failures.append("layout_contract.panel_geometry must be a mapping")
        return
    missing = panels - {str(panel_id) for panel_id in geometry}
    if missing:
        report.failures.append(f"panel_geometry is missing panels: {sorted(missing)}")
    for panel_id in panels.intersection({str(key) for key in geometry}):
        item = geometry.get(panel_id)
        if not isinstance(item, Mapping):
            report.failures.append(f"panel_geometry.{panel_id} must be a mapping")
            continue
        if not str(item.get("chart_family") or "").strip():
            report.failures.append(f"panel_geometry.{panel_id} requires chart_family")
        slots = item.get("category_slots")
        if slots is not None and (not isinstance(slots, int) or slots < 1):
            report.failures.append(f"panel_geometry.{panel_id}.category_slots must be a positive integer")
        aspect = item.get("natural_aspect")
        if not _valid_aspect(aspect):
            report.failures.append(
                f"panel_geometry.{panel_id}.natural_aspect must be [positive_min, positive_max]"
            )
        sides = {str(side) for side in (item.get("decoration_sides") or [])}
        if not sides.issubset(ALLOWED_EDGES):
            report.failures.append(f"panel_geometry.{panel_id} has invalid decoration_sides")
        weight = str(item.get("visual_weight") or "")
        if weight not in ALLOWED_VISUAL_WEIGHTS:
            report.failures.append(
                f"panel_geometry.{panel_id}.visual_weight must be one of {sorted(ALLOWED_VISUAL_WEIGHTS)}"
            )


def _check_bar_policy(contract: Mapping[str, Any], report: LayoutContractReport) -> None:
    policy = contract.get("bar_width_policy")
    if not isinstance(policy, Mapping):
        report.failures.append("layout_contract.bar_width_policy must be a mapping")
        return
    mode = str(policy.get("mode") or "")
    if mode not in ALLOWED_BAR_MODES:
        report.failures.append(f"bar_width_policy.mode must be one of {sorted(ALLOWED_BAR_MODES)}")
    if not str(policy.get("scope") or "").strip():
        report.failures.append("bar_width_policy.scope is required")
    if not str(policy.get("tradeoff") or "").strip():
        report.failures.append("bar_width_policy.tradeoff is required")


def _check_topology(
    contract: Mapping[str, Any],
    unit_ids: set[str],
    report: LayoutContractReport,
) -> None:
    topology = contract.get("topology")
    if not isinstance(topology, Mapping):
        report.failures.append("layout_contract.topology must be a mapping")
        return
    direction = str(topology.get("reading_direction") or "")
    if direction not in ALLOWED_READING_DIRECTIONS:
        report.failures.append(
            f"topology.reading_direction must be one of {sorted(ALLOWED_READING_DIRECTIONS)}"
        )
    sequence = [str(unit) for unit in (topology.get("unit_sequence") or [])]
    if set(sequence) != unit_ids or len(sequence) != len(unit_ids):
        report.failures.append("topology.unit_sequence must contain every semantic unit exactly once")
    else:
        report.passes.append("topology sequence covers every semantic unit")
    if not str(topology.get("rationale") or "").strip():
        report.failures.append("topology.rationale is required")


def _check_constraint_lists(contract: Mapping[str, Any], report: LayoutContractReport) -> None:
    for key in ("hard_constraints", "soft_targets"):
        value = contract.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            report.failures.append(f"layout_contract.{key} must be a non-empty list")


def _check_qa(contract: Mapping[str, Any], report: LayoutContractReport) -> None:
    qa = contract.get("qa")
    if not isinstance(qa, Mapping):
        report.failures.append("layout_contract.qa must be a mapping")
        return
    required = {
        "final_size_render",
        "collision_check",
        "clipping_check",
        "alignment_measurement",
        "grayscale_check",
    }
    missing = sorted(key for key in required if qa.get(key) is not True)
    if missing:
        report.failures.append(f"layout_contract.qa must enable: {missing}")
    else:
        report.passes.append("layout QA gates are enabled")


def _panel_list(value: Any, label: str, report: LayoutContractReport) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        report.failures.append(f"{label} requires a non-empty panels list")
        return []
    return [str(panel) for panel in value]


def _valid_aspect(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return False
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return False
    return 0 < low <= high
