from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ProfileStats:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    durations: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += int(amount)

    @contextlib.contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] += time.perf_counter() - start


GLOBAL_PROFILE_STATS = ProfileStats()

