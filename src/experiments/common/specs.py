from __future__ import annotations

from src.config.units import ms


class StepSpecMixin:
    """Shared conversion helper for experiment timing dataclasses."""

    dt: float

    def ms_to_steps(self, duration_ms: float) -> int:
        return int(round((float(duration_ms) * ms) / float(self.dt)))

