from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .builders import FIGURE_BUILDERS
from .common import SourceBuildContext


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not locate Net_torch repository root")


def build_supplementary_v5_source_data(
    *,
    output_dir: Path,
    source_root: Path | None = None,
    figures: Sequence[str] = tuple(f"s{index}" for index in range(1, 8)),
    command: str = "",
) -> dict[str, object]:
    repo_root = _repo_root()
    resolved_source_root = (source_root or repo_root / "results" / "paper_figure_multi_seed").resolve()
    resolved_output = output_dir.resolve()
    if not resolved_source_root.is_dir():
        raise FileNotFoundError(resolved_source_root)
    requested = tuple(str(figure).lower() for figure in figures)
    unknown = sorted(set(requested) - set(FIGURE_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown supplementary figure ids: {unknown}")
    context = SourceBuildContext(
        repo_root=repo_root,
        source_root=resolved_source_root,
        output_dir=resolved_output,
    )
    for figure_id in requested:
        FIGURE_BUILDERS[figure_id](context)
    return context.finalize(figures=requested, command=command)


__all__ = ["build_supplementary_v5_source_data"]
