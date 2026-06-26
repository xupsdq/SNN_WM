from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import matplotlib.image as mpimg

from src.plotting.paper_fig.utils import repo_root_from_here


def parse_svg_viewbox(path: str | Path) -> dict[str, float]:
    """Return SVG viewBox dimensions in source units."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")[:8192]
    match = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', text)
    if match:
        values = [float(v) for v in re.split(r"[\s,]+", match.group(1).strip()) if v]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return {"x": values[0], "y": values[1], "width": values[2], "height": values[3]}
    width = _svg_length(text, "width")
    height = _svg_length(text, "height")
    if width and height:
        return {"x": 0.0, "y": 0.0, "width": width, "height": height}
    raise ValueError(f"Cannot parse SVG viewBox or width/height: {path}")


def render_svg_asset_panel(ax, spec: Mapping[str, Any], *, dpi: int = 300) -> None:
    """Draw an external SVG as a contained raster image while preserving aspect ratio."""
    ax.set_axis_off()
    path = resolve_svg_asset_path(spec)
    viewbox = parse_svg_viewbox(path)
    panel = spec.get("position_mm") or spec.get("size_mm") or {}
    panel_w = float(panel.get("w", panel.get("width", 1.0)))
    panel_h = float(panel.get("h", panel.get("height", 1.0)))
    aspect = viewbox["width"] / viewbox["height"]
    box_w, box_h = _contained_size(panel_w, panel_h, aspect)
    png = _rasterize_svg(path, box_w, box_h, dpi=dpi)
    img = mpimg.imread(png)
    frac_w = box_w / panel_w if panel_w else 1.0
    frac_h = box_h / panel_h if panel_h else 1.0
    x0 = (1.0 - frac_w) / 2.0
    y0 = (1.0 - frac_h) / 2.0
    ax.imshow(img, extent=(x0, x0 + frac_w, y0, y0 + frac_h), transform=ax.transAxes, aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.paper_fig_plot_form = "manual_svg_asset"
    ax.paper_fig_svg_asset = str(path)
    ax.paper_fig_svg_viewbox = viewbox
    ax.paper_fig_svg_aspect_ratio = float(aspect)
    ax.paper_fig_svg_rendered_size_mm = {"w": float(box_w), "h": float(box_h)}
    ax.paper_fig_svg_raster_cache = str(png)


def resolve_svg_asset_path(spec: Mapping[str, Any]) -> Path:
    raw = spec.get("source") or (spec.get("source_mapping") or {}).get("manual_asset")
    if not raw:
        raise ValueError(f"{spec.get('figure_id', '?')}{spec.get('panel_id', '?')}: SVG source missing")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = repo_root_from_here()
    for candidate in (root / path, Path(__file__).resolve().parent / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(root / path)


def _contained_size(panel_w: float, panel_h: float, aspect: float) -> tuple[float, float]:
    if panel_w / panel_h > aspect:
        return panel_h * aspect, panel_h
    return panel_w, panel_w / aspect


def _rasterize_svg(path: Path, width_mm: float, height_mm: float, *, dpi: int) -> Path:
    root = repo_root_from_here()
    cache_dir = root / ".codex" / "tmp" / "paper_fig_svg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    width_px = max(16, int(round(width_mm / 25.4 * dpi)))
    height_px = max(16, int(round(height_mm / 25.4 * dpi)))
    out = cache_dir / f"{path.stem}_{digest}_{width_px}x{height_px}.png"
    if out.exists():
        return out
    browser = _browser_exe()
    html_path = cache_dir / f"{path.stem}_{digest}_{width_px}x{height_px}.html"
    html_path.write_text(
        "<html><body style='margin:0;background:white;overflow:hidden'>"
        f"<img src='{html.escape(path.as_uri())}' style='width:{width_px}px;height:{height_px}px;object-fit:contain;display:block'>"
        "</body></html>",
        encoding="utf-8",
    )
    user_data_dir = cache_dir / "chrome_profile"
    user_data_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--allow-file-access-from-files",
            f"--user-data-dir={user_data_dir}",
            f"--window-size={width_px},{height_px}",
            f"--screenshot={out}",
            html_path.as_uri(),
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"SVG rasterization failed for {path}: {result.stderr.strip()}")
    return out


def _svg_length(text: str, attr: str) -> float | None:
    match = re.search(rf'{attr}\s*=\s*["\']([0-9.]+)', text)
    return float(match.group(1)) if match else None


def _node_exe() -> Path:
    candidates = []
    if os.environ.get("NODE_EXE"):
        candidates.append(Path(os.environ["NODE_EXE"]))
    candidates.append(Path(r"C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    from shutil import which

    found = which("node")
    if found:
        return Path(found)
    raise RuntimeError("Node.js is required to rasterize SVG assets for paper figures.")


def _bundled_node_modules() -> Path | None:
    path = Path(r"C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules")
    return path if path.exists() else None


def _browser_exe() -> Path:
    candidates = [
        Path(os.environ.get("CHROME_EXE", "")) if os.environ.get("CHROME_EXE") else None,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_file():
            return candidate
    raise RuntimeError("Chrome or Edge is required to rasterize SVG assets for paper figures.")
