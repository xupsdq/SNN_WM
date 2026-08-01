from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .plotting import mean_ci
from .sources import SourceStore


@dataclass
class BuildContext:
    store: SourceStore
    output_root: Path | None
    panel_data_records: list[dict[str, object]] = field(default_factory=list)
    statistic_records: list[dict[str, object]] = field(default_factory=list)
    qc_records: list[dict[str, object]] = field(default_factory=list)

    def capture_panel(
        self,
        figure_id: str,
        panel_id: str,
        frame: pd.DataFrame,
        *,
        metrics: Sequence[str] = (),
        groups: Sequence[str] = (),
        max_rows: int = 25_000,
    ) -> None:
        data = frame.copy()
        if len(data) > max_rows:
            indices = np.linspace(0, len(data) - 1, max_rows).astype(int)
            data = data.iloc[indices].copy()
            sampled = True
        else:
            sampled = False
        data.insert(0, "panel_id", panel_id)
        data.insert(0, "figure_id", figure_id)
        if self.output_root is not None:
            path = (
                self.output_root
                / "data"
                / "panel_data"
                / f"{figure_id}_{panel_id}.csv"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False, encoding="utf-8")
            relative_path = path.relative_to(self.output_root).as_posix()
        else:
            relative_path = ""
        self.panel_data_records.append(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "rows": int(len(data)),
                "source_rows": int(len(frame)),
                "sampled": sampled,
                "path": relative_path,
            }
        )
        if metrics:
            self._record_statistics(
                figure_id,
                panel_id,
                frame,
                metrics=metrics,
                groups=groups,
            )

    def add_qc(
        self,
        figure_id: str,
        check: str,
        status: str,
        detail: str,
    ) -> None:
        self.qc_records.append(
            {
                "figure_id": figure_id,
                "check": check,
                "status": status,
                "detail": detail,
            }
        )

    def write_tables(self) -> None:
        if self.output_root is None:
            return
        data_dir = self.output_root / "data"
        metrics_dir = self.output_root / "metrics"
        data_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.panel_data_records).to_csv(
            data_dir / "panel_data_manifest.csv",
            index=False,
            encoding="utf-8",
        )
        pd.DataFrame(self.statistic_records).to_csv(
            metrics_dir / "panel_statistics.csv",
            index=False,
            encoding="utf-8",
        )
        pd.DataFrame(self.qc_records).to_csv(
            metrics_dir / "figure_qc.csv",
            index=False,
            encoding="utf-8",
        )
        with (metrics_dir / "panel_statistics.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.statistic_records,
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_scalar,
            )

    def _record_statistics(
        self,
        figure_id: str,
        panel_id: str,
        frame: pd.DataFrame,
        *,
        metrics: Sequence[str],
        groups: Sequence[str],
    ) -> None:
        group_columns = [column for column in groups if column in frame.columns]
        for metric in metrics:
            if metric not in frame.columns:
                continue
            columns = [
                *(
                    ["network_seed"]
                    if "network_seed" in frame.columns
                    else []
                ),
                *group_columns,
                metric,
            ]
            data = frame.loc[:, list(dict.fromkeys(columns))].copy()
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
            data = data.dropna(subset=[metric])
            if data.empty:
                continue
            if "network_seed" in data.columns:
                by = ["network_seed", *group_columns]
                data = (
                    data.groupby(by, as_index=False, observed=True)[metric]
                    .mean()
                )
            iterator: Iterable[tuple[object, pd.DataFrame]]
            if group_columns:
                iterator = data.groupby(
                    group_columns,
                    dropna=False,
                    observed=True,
                )
            else:
                iterator = [((), data)]
            for group_key, part in iterator:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                mean, low, high = mean_ci(part[metric].to_numpy(float))
                record: dict[str, object] = {
                    "figure_id": figure_id,
                    "panel_id": panel_id,
                    "metric": metric,
                    "n_networks": (
                        int(part["network_seed"].nunique())
                        if "network_seed" in part.columns
                        else None
                    ),
                    "n_observations": int(len(part)),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
                for column, value in zip(group_columns, group_key):
                    record[column] = value
                self.statistic_records.append(record)


def _json_scalar(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )
