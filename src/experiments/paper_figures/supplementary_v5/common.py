from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_NETWORK_SEEDS = tuple(range(1000, 1020))


def exact_sign_flip_p(values: Iterable[float], *, alternative: str = "two-sided") -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    if len(array) > 24:
        raise ValueError("Exact sign-flip is limited to 24 independent values")
    signed_sums = np.array([0.0], dtype=np.float64)
    for value in array:
        signed_sums = np.concatenate((signed_sums + value, signed_sums - value))
    observed = float(array.sum())
    tolerance = max(1e-14, abs(observed) * 1e-14)
    if alternative == "greater":
        return float(np.mean(signed_sums >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(signed_sums <= observed + tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(signed_sums) >= abs(observed) - tolerance))
    raise ValueError(f"Unknown alternative: {alternative}")


def _bootstrap_ci(values: np.ndarray, *, seed: int, n_resamples: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, len(values), size=(int(n_resamples), len(values)))]
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_values(
    values: Iterable[float],
    *,
    null: float = 0.0,
    ci_method: str = "student_t",
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("Cannot summarize an empty value array")
    mean = float(array.mean())
    if len(array) == 1:
        low = high = mean
    elif ci_method == "student_t":
        half = float(stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
        low, high = mean - half, mean + half
    elif ci_method == "bootstrap_percentile":
        low, high = _bootstrap_ci(array, seed=bootstrap_seed)
    elif ci_method == "none":
        low = high = float("nan")
    else:
        raise ValueError(f"Unknown CI method: {ci_method}")
    centered = array - float(null)
    return {
        "n": int(len(array)),
        "mean": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "null": float(null),
        "fraction_above_null": float(np.mean(centered > 0.0)),
        "ci_method": ci_method,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _seed_from_path(path: Path) -> int:
    return int(next(part for part in path.parts if part.startswith("seed_")).split("_", 1)[1])


@dataclass
class SourceBuildContext:
    repo_root: Path
    source_root: Path
    output_dir: Path
    input_paths: set[Path] = field(default_factory=set)
    panel_records: list[dict[str, Any]] = field(default_factory=list)
    statistic_records: list[dict[str, Any]] = field(default_factory=list)
    test_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def source_data_dir(self) -> Path:
        return self.output_dir / "data" / "source_data"

    def track(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.input_paths.add(resolved)
        return resolved

    def read_csv(self, path: Path, *, usecols: Sequence[str] | None = None) -> pd.DataFrame:
        return pd.read_csv(self.track(path), usecols=usecols)

    def read_many(
        self,
        pattern: str,
        *,
        usecols: Sequence[str] | None = None,
        seed_from_path: bool = False,
    ) -> pd.DataFrame:
        paths = sorted(self.source_root.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No persisted inputs matched: {self.source_root / pattern}")
        frames: list[pd.DataFrame] = []
        for path in paths:
            frame = self.read_csv(path, usecols=usecols)
            if seed_from_path and "network_seed" not in frame:
                frame.insert(0, "network_seed", _seed_from_path(path))
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def paths(self, pattern: str) -> list[Path]:
        paths = sorted(self.source_root.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No persisted inputs matched: {self.source_root / pattern}")
        return [self.track(path) for path in paths]

    def require_networks(self, frame: pd.DataFrame, *, label: str, expected: Sequence[int] = EXPECTED_NETWORK_SEEDS) -> None:
        observed = tuple(sorted(int(value) for value in frame["network_seed"].dropna().unique()))
        expected_tuple = tuple(int(value) for value in expected)
        if observed != expected_tuple:
            raise ValueError(f"{label} network cohort mismatch: observed={observed}, expected={expected_tuple}")

    def write_panel(self, figure_id: str, panel_id: str, frame: pd.DataFrame, *, suffix: str = "") -> Path:
        stem = f"{figure_id}_{panel_id}{suffix}"
        path = self.source_data_dir / f"{stem}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8", float_format="%.12g")
        self.panel_records.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "suffix": suffix,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "path": path.relative_to(self.output_dir).as_posix(),
            }
        )
        return path

    def add_summaries(
        self,
        figure_id: str,
        panel_id: str,
        frame: pd.DataFrame,
        *,
        value: str = "value",
        groups: Sequence[str] = (),
        null: float = 0.0,
        ci_method: str = "student_t",
        bootstrap_seed: int = 20260801,
        role: str = "display",
    ) -> None:
        iterator = frame.groupby(list(groups), dropna=False, observed=True, sort=True) if groups else [((), frame)]
        for labels, part in iterator:
            if not isinstance(labels, tuple):
                labels = (labels,)
            record: dict[str, Any] = {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": value,
                "role": role,
            }
            record.update(
                summarize_values(
                    pd.to_numeric(part[value], errors="coerce"),
                    null=null,
                    ci_method=ci_method,
                    bootstrap_seed=bootstrap_seed,
                )
            )
            for key, label in zip(groups, labels):
                record[key] = label.item() if hasattr(label, "item") else label
            self.statistic_records.append(record)

    def add_frozen_summary(
        self,
        figure_id: str,
        panel_id: str,
        *,
        metric: str,
        source_row: pd.Series,
        groups: dict[str, Any] | None = None,
        role: str = "display",
    ) -> None:
        record: dict[str, Any] = {
            "figure_id": figure_id,
            "panel_id": panel_id,
            "metric": metric,
            "role": role,
            "n": int(source_row["n_networks"]),
            "mean": float(source_row["mean"]),
            "ci95_low": float(source_row["ci95_low"]),
            "ci95_high": float(source_row["ci95_high"]),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "null": float(source_row.get("threshold", 0.0)),
            "fraction_above_null": float(source_row.get("fraction_above_zero", np.nan)),
            "ci_method": "persisted_confirmatory_bootstrap",
        }
        if groups:
            record.update(groups)
        self.statistic_records.append(record)

    def add_test(
        self,
        figure_id: str,
        panel_id: str,
        endpoint: str,
        values: Iterable[float],
        *,
        family: str,
        alternative: str = "two-sided",
        null: float = 0.0,
        role: str = "primary",
    ) -> None:
        array = np.asarray(list(values), dtype=float)
        array = array[np.isfinite(array)]
        centered = array - float(null)
        self.test_records.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "endpoint": endpoint,
                "family": family,
                "role": role,
                "alternative": alternative,
                "null": float(null),
                "n": int(len(array)),
                "mean": float(array.mean()),
                "p_raw": exact_sign_flip_p(centered, alternative=alternative),
                "p_holm": float("nan"),
            }
        )

    def add_frozen_test(
        self,
        figure_id: str,
        panel_id: str,
        endpoint: str,
        *,
        family: str,
        source_row: pd.Series,
        role: str = "confirmatory",
    ) -> None:
        self.test_records.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "endpoint": endpoint,
                "family": family,
                "role": role,
                "alternative": "greater",
                "null": float(source_row.get("threshold", 0.0)),
                "n": int(source_row["n_networks"]),
                "mean": float(source_row["mean"]),
                "p_raw": float(source_row["p_one_sided"]),
                "p_holm": float(source_row["holm_adjusted_p"]),
            }
        )

    def finalize(self, *, figures: Sequence[str], command: str) -> dict[str, Any]:
        self._adjust_holm()
        metrics_dir = self.output_dir / "metrics"
        meta_dir = self.output_dir / "meta"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        stats_frame = pd.DataFrame(self.statistic_records)
        tests_frame = pd.DataFrame(self.test_records)
        if stats_frame.empty or tests_frame.empty:
            raise ValueError("Supplementary Source Data must include statistics and inference records")
        invalid_raw = tests_frame.loc[~tests_frame["p_raw"].between(0.0, 1.0, inclusive="right")]
        invalid_holm = tests_frame.loc[~tests_frame["p_holm"].between(0.0, 1.0, inclusive="both")]
        if not invalid_raw.empty or not invalid_holm.empty or bool((tests_frame["p_raw"] == 0.0).any()):
            raise ValueError("Supplementary inference contains invalid, missing, or zero-valued p-values")
        decisive = tests_frame.loc[tests_frame["role"].isin(["primary", "confirmatory"])]
        failed_decisive = decisive.loc[decisive["p_holm"] >= 0.05]
        if not failed_decisive.empty:
            failed = failed_decisive[["figure_id", "panel_id", "endpoint", "p_holm"]].to_dict("records")
            raise ValueError(f"A conclusion-bearing supplementary endpoint is not significant: {failed}")
        current_source_paths = {
            (self.output_dir / str(record["path"])).resolve()
            for record in self.panel_records
        }
        for figure_id in figures:
            for path in self.source_data_dir.glob(f"{str(figure_id).lower()}_*.csv"):
                if path.resolve() not in current_source_paths:
                    path.unlink()
        boundary = tests_frame.loc[tests_frame["role"].eq("boundary")]
        stats_frame.to_csv(metrics_dir / "panel_statistics.csv", index=False, encoding="utf-8", float_format="%.12g")
        tests_frame.to_csv(metrics_dir / "primary_tests.csv", index=False, encoding="utf-8", float_format="%.12g")
        write_json(metrics_dir / "panel_statistics.json", self.statistic_records)
        write_json(metrics_dir / "primary_tests.json", self.test_records)
        source_qa = {
            "status": "passed",
            "independent_unit": "independently trained network",
            "network_cohort": list(EXPECTED_NETWORK_SEEDS),
            "panels": int(len({(record["figure_id"], record["panel_id"]) for record in self.panel_records})),
            "source_tables": int(len(self.panel_records)),
            "statistic_rows": int(len(stats_frame)),
            "inference_rows": int(len(tests_frame)),
            "decisive_rows": int(len(decisive)),
            "decisive_holm_below_0_05": int((decisive["p_holm"] < 0.05).sum()),
            "boundary_rows": int(len(boundary)),
            "zero_p_values": int((tests_frame["p_raw"] == 0.0).sum()),
            "boundary_endpoints": boundary[
                ["figure_id", "panel_id", "endpoint", "p_raw", "p_holm"]
            ].to_dict("records"),
        }
        write_json(metrics_dir / "source_data_qa.json", source_qa)
        pd.DataFrame(self.panel_records).to_csv(
            self.output_dir / "data" / "source_data_manifest.csv", index=False, encoding="utf-8"
        )

        input_manifest = [
            {
                "path": path.relative_to(self.repo_root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in sorted(self.input_paths)
        ]
        write_json(meta_dir / "input_manifest.json", input_manifest)
        run_config = {
            "task": "supplementary_v5_persisted_data_reanalysis",
            "figures": list(figures),
            "source_root": self.source_root.relative_to(self.repo_root).as_posix(),
            "independent_unit": "independently trained network",
            "expected_network_seeds": list(EXPECTED_NETWORK_SEEDS),
            "command": command,
            "simulation_rerun": False,
        }
        write_json(self.output_dir / "run_config.json", run_config)
        summary = {
            "status": "complete",
            "figures": list(figures),
            "panels": int(len({(record["figure_id"], record["panel_id"]) for record in self.panel_records})),
            "source_tables": int(len(self.panel_records)),
            "statistic_rows": int(len(self.statistic_records)),
            "test_rows": int(len(self.test_records)),
            "decisive_tests": int(len(decisive)),
            "boundary_tests": int(len(boundary)),
            "input_files": int(len(input_manifest)),
            "network_cohort": list(EXPECTED_NETWORK_SEEDS),
        }
        write_json(self.output_dir / "summary.json", summary)

        output_paths = sorted(self.source_data_dir.glob("*.csv"))
        output_paths.extend(
            [
                self.output_dir / "data" / "source_data_manifest.csv",
                metrics_dir / "panel_statistics.csv",
                metrics_dir / "panel_statistics.json",
                metrics_dir / "primary_tests.csv",
                metrics_dir / "primary_tests.json",
                metrics_dir / "source_data_qa.json",
                meta_dir / "input_manifest.json",
                self.output_dir / "run_config.json",
                self.output_dir / "summary.json",
            ]
        )
        output_paths = sorted(path for path in output_paths if path.is_file())
        artifact_manifest = {
            "schema_version": 1,
            "producer": "src.experiments.runners.supplementary_v5",
            "scope": "source_data_only",
            "inputs": input_manifest,
            "outputs": [
                {
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
                for path in output_paths
            ],
        }
        write_json(self.output_dir / "artifact_manifest.json", artifact_manifest)
        return summary

    def _adjust_holm(self) -> None:
        frame = pd.DataFrame(self.test_records)
        if frame.empty:
            return
        for (_, family), indices in frame.groupby(["figure_id", "family"]).groups.items():
            mutable = [int(index) for index in indices if not np.isfinite(float(self.test_records[int(index)]["p_holm"]))]
            if not mutable:
                continue
            ordered = sorted(mutable, key=lambda index: float(self.test_records[index]["p_raw"]))
            running = 0.0
            total = len(ordered)
            for rank, index in enumerate(ordered):
                candidate = min(1.0, (total - rank) * float(self.test_records[index]["p_raw"]))
                running = max(running, candidate)
                self.test_records[index]["p_holm"] = running
