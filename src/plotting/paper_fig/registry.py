from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.plotting.paper_fig.data_resolver import resolve_repo_path
from src.plotting.paper_fig.utils import load_yaml, paper_fig_root, repo_root_from_here


SPECS_DIR = paper_fig_root() / "specs"


def get_figure_index() -> dict[str, Any]:
    """Load the paper figure index."""
    return load_yaml(SPECS_DIR / "paper_figures.yaml")


def get_figure_spec(fig_id: str) -> dict[str, Any]:
    """Load a figure spec by figure id."""
    normalized = fig_id.lower()
    path = SPECS_DIR / f"{normalized}.yaml"
    if not path.is_file():
        raise KeyError(f"Unknown paper figure spec: {fig_id}")
    spec = load_yaml(path)
    spec.setdefault("figure_id", normalized)
    return spec


def get_panel_spec(fig_id: str, panel_id: str) -> dict[str, Any]:
    """Return one panel spec from a figure spec."""
    spec = get_figure_spec(fig_id)
    panels = spec.get("panels", {})
    if not isinstance(panels, Mapping):
        raise ValueError(f"{fig_id} panels must be a mapping")
    key = panel_id.upper()
    if key not in panels:
        raise KeyError(f"Unknown panel {fig_id}{panel_id}")
    panel = dict(panels[key])
    panel.setdefault("figure_id", spec.get("figure_id", fig_id.lower()))
    panel.setdefault("panel_id", key)
    return panel


def resolve_source_mapping(fig_id: str, panel_id: str) -> dict[str, Any]:
    """Return the explicit source mapping for a panel."""
    panel = get_panel_spec(fig_id, panel_id)
    mapping = dict(panel.get("source_mapping") or {})
    if panel.get("source"):
        mapping.setdefault("manual_asset", panel.get("source"))
    mapping.setdefault("figure_id", fig_id.lower())
    mapping.setdefault("panel_id", panel_id.upper())
    return mapping


def list_missing_sources(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """List panel source candidates that are not currently present."""
    root = repo_root or repo_root_from_here()
    missing: list[dict[str, Any]] = []
    for fig_id in _indexed_fig_ids():
        spec = get_figure_spec(fig_id)
        for panel_id, panel in (spec.get("panels") or {}).items():
            mapping = panel.get("source_mapping") or {}
            for key in ("required_files", "candidate_files"):
                for raw_path in mapping.get(key, []) or []:
                    path = resolve_repo_path(root, raw_path)
                    if path is not None and not path.exists():
                        missing.append({"figure_id": fig_id, "panel_id": panel_id, "path": str(raw_path), "kind": key})
            manual_asset = panel.get("source") or mapping.get("manual_asset")
            if panel.get("panel_type") == "manual_schematic" and manual_asset:
                path = resolve_repo_path(paper_fig_root(), manual_asset)
                if path is not None and not path.exists():
                    missing.append({"figure_id": fig_id, "panel_id": panel_id, "path": str(manual_asset), "kind": "manual_asset"})
    return missing


def list_manual_assets(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """List manual asset requirements from all specs."""
    _ = repo_root
    assets: list[dict[str, Any]] = []
    for fig_id in _indexed_fig_ids():
        spec = get_figure_spec(fig_id)
        for panel_id, panel in (spec.get("panels") or {}).items():
            if panel.get("panel_type") == "manual_schematic":
                raw_path = panel.get("source") or (panel.get("source_mapping") or {}).get("manual_asset")
                if raw_path:
                    path = resolve_repo_path(paper_fig_root(), raw_path)
                    assets.append(
                        {
                            "figure_id": fig_id,
                            "panel_id": panel_id,
                            "path": str(raw_path),
                            "exists": bool(path and path.exists()),
                        }
                    )
    return assets


def validate_registry(repo_root: Path | None = None) -> dict[str, list[str]]:
    """Validate that indexed specs and panel metadata are coherent."""
    _ = repo_root
    passes: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []
    index = get_figure_index()
    figures = index.get("figures", {})
    if not isinstance(figures, Mapping) or not figures:
        failures.append("paper_figures.yaml must define a non-empty figures mapping")
        return {"passes": passes, "warnings": warnings, "failures": failures}
    for fig_id, entry in figures.items():
        try:
            spec = get_figure_spec(str(fig_id))
        except Exception as exc:
            failures.append(f"{fig_id}: cannot load spec: {exc}")
            continue
        if spec.get("figure_id") != fig_id:
            warnings.append(f"{fig_id}: spec figure_id differs from index key")
        panels = spec.get("panels", {})
        if not isinstance(panels, Mapping):
            failures.append(f"{fig_id}: panels must be a mapping")
            continue
        for panel_id, panel in panels.items():
            if not panel.get("claim"):
                failures.append(f"{fig_id}{panel_id}: missing claim")
            if panel.get("panel_type") != "manual_schematic" and panel.get("data_adapter") in (None, ""):
                warnings.append(f"{fig_id}{panel_id}: data-driven panel has no adapter")
            if not panel.get("renderer"):
                warnings.append(f"{fig_id}{panel_id}: missing renderer")
        passes.append(f"{fig_id}: loaded {len(panels)} panel specs ({entry.get('status', 'unknown')})")
    return {"passes": passes, "warnings": warnings, "failures": failures}


def _indexed_fig_ids() -> list[str]:
    index = get_figure_index()
    figures = index.get("figures", {})
    if not isinstance(figures, Mapping):
        return []
    return [str(k) for k in figures.keys()]

