from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


PARALLEL_AXES = ("auto", "run", "seed", "subtask")
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@dataclass(frozen=True)
class ResourcePlan:
    device: str
    requested_jobs: int
    effective_jobs: int
    seed_jobs: int | None
    parallel_axis: str
    cpu_workers: int
    cpu_threads_per_worker: int | None
    max_build_workers: int
    available_cpu_count: int
    notes: tuple[str, ...] = ()

    def thread_env(self) -> dict[str, str]:
        env = {
            "PAPER_FIG_PARALLEL_AXIS": str(self.parallel_axis),
            "PAPER_FIG_CPU_WORKERS": str(int(self.cpu_workers)),
            "PAPER_FIG_MAX_BUILD_WORKERS": str(int(self.max_build_workers)),
        }
        if self.cpu_threads_per_worker is not None:
            value = str(int(self.cpu_threads_per_worker))
            env["PAPER_FIG_CPU_THREADS_PER_WORKER"] = value
            for key in THREAD_ENV_KEYS:
                env[key] = value
        return env

    def apply_to_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env.update(self.thread_env())
        return env

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "requested_jobs": int(self.requested_jobs),
            "effective_jobs": int(self.effective_jobs),
            "seed_jobs": self.seed_jobs,
            "parallel_axis": self.parallel_axis,
            "cpu_workers": int(self.cpu_workers),
            "cpu_threads_per_worker": self.cpu_threads_per_worker,
            "max_build_workers": int(self.max_build_workers),
            "available_cpu_count": int(self.available_cpu_count),
            "thread_env": self.thread_env(),
            "notes": list(self.notes),
        }


def resolve_resource_plan(
    *,
    device: str,
    jobs: int,
    seed_jobs: int | None,
    cpu_workers: int | None,
    cpu_threads_per_worker: int | None,
    parallel_axis: str,
    max_build_workers: int,
) -> ResourcePlan:
    if parallel_axis not in PARALLEL_AXES:
        raise ValueError(f"Unsupported parallel axis: {parallel_axis}")
    available_cpu_count = int(os.cpu_count() or 1)
    requested_jobs = int(jobs)
    effective_jobs = int(seed_jobs) if seed_jobs is not None else requested_jobs
    if effective_jobs < 1:
        raise ValueError("--jobs/--seed-jobs must be >= 1")
    resolved_cpu_workers = int(cpu_workers) if cpu_workers is not None else 1
    resolved_cpu_workers = max(1, min(resolved_cpu_workers, available_cpu_count))
    resolved_build_workers = max(1, int(max_build_workers))
    resolved_threads = None if cpu_threads_per_worker is None else max(1, int(cpu_threads_per_worker))
    notes: list[str] = []
    if str(device).lower() == "cuda" and effective_jobs > 1:
        notes.append("Multiple run jobs will share one CUDA device; prefer jobs=1 for single-GPU benchmark runs.")
    return ResourcePlan(
        device=str(device),
        requested_jobs=requested_jobs,
        effective_jobs=effective_jobs,
        seed_jobs=None if seed_jobs is None else int(seed_jobs),
        parallel_axis=str(parallel_axis),
        cpu_workers=resolved_cpu_workers,
        cpu_threads_per_worker=resolved_threads,
        max_build_workers=resolved_build_workers,
        available_cpu_count=available_cpu_count,
        notes=tuple(notes),
    )
