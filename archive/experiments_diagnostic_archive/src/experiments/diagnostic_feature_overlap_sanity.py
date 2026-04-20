from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd


def run_sanity_checks(runners: Mapping[str, Callable[[], pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for sanity_name, runner in runners.items():
        try:
            frame = runner()
        except FileNotFoundError as exc:
            rows.append(
                {
                    "sanity_check": str(sanity_name),
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue
        if frame is None or frame.empty:
            rows.append(
                {
                    "sanity_check": str(sanity_name),
                    "status": "skipped",
                    "reason": "empty",
                }
            )
            continue
        local = frame.copy()
        local["sanity_check"] = str(sanity_name)
        local["status"] = "ok"
        rows.extend(local.to_dict("records"))
    return pd.DataFrame(rows)

