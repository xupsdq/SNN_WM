from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROGRESS_EVENT_PREFIX = "PAPER_FIG_PROGRESS "


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _enabled(ctx: Any) -> bool:
    cfg = getattr(ctx, "cfg", None)
    return bool(getattr(cfg, "show_progress", True))


def emit_progress_event(
    ctx: Any,
    *,
    phase: str,
    status: str,
    fig_id: str = "",
    phase_index: int | None = None,
    total_phases: int | None = None,
    elapsed_seconds: float | None = None,
    detail: str = "",
) -> None:
    if not _enabled(ctx):
        return
    cfg = getattr(ctx, "cfg", None)
    payload: dict[str, Any] = {
        "event": "paper_fig_progress",
        "fig_id": str(fig_id),
        "network_seed": int(getattr(cfg, "network_seed", 0)),
        "phase": str(phase),
        "status": str(status),
    }
    if phase_index is not None:
        payload["phase_index"] = int(phase_index)
    if total_phases is not None:
        payload["total_phases"] = int(total_phases)
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    if detail:
        payload["detail"] = str(detail)
    sys.stdout.write(PROGRESS_EVENT_PREFIX + json.dumps(_json_ready(payload), sort_keys=True) + "\n")
    sys.stdout.flush()


def planned_phases(candidates: Iterable[tuple[str, bool]]) -> tuple[str, ...]:
    return tuple(str(name) for name, enabled in candidates if bool(enabled))


class ProgressTracker:
    def __init__(self, ctx: Any, phases: Sequence[str], *, fig_id: str = "") -> None:
        self.ctx = ctx
        self.fig_id = str(fig_id)
        self.phases = tuple(str(phase) for phase in phases)
        self._index = {phase: index for index, phase in enumerate(self.phases, start=1)}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        with progress_phase(
            self.ctx,
            name,
            fig_id=self.fig_id,
            phase_index=self._index.get(str(name)),
            total_phases=len(self.phases) or None,
        ):
            yield


@contextmanager
def progress_phase(
    ctx: Any,
    phase: str,
    *,
    fig_id: str = "",
    phase_index: int | None = None,
    total_phases: int | None = None,
) -> Iterator[None]:
    started = time.time()
    emit_progress_event(ctx, phase=phase, status="start", fig_id=fig_id, phase_index=phase_index, total_phases=total_phases)
    try:
        yield
    except Exception as exc:
        emit_progress_event(
            ctx,
            phase=phase,
            status="failed",
            fig_id=fig_id,
            phase_index=phase_index,
            total_phases=total_phases,
            elapsed_seconds=time.time() - started,
            detail=str(exc),
        )
        raise
    else:
        emit_progress_event(
            ctx,
            phase=phase,
            status="done",
            fig_id=fig_id,
            phase_index=phase_index,
            total_phases=total_phases,
            elapsed_seconds=time.time() - started,
        )


__all__ = [
    "PROGRESS_EVENT_PREFIX",
    "ProgressTracker",
    "emit_progress_event",
    "planned_phases",
    "progress_phase",
]
