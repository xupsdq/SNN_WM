from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.smoke.registry import SMOKE_SPECS
from src.smoke.run_one import run_one


def main() -> int:
    results_root = Path(__file__).resolve().parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for experiment_id in sorted(SMOKE_SPECS):
        rows.append(run_one(experiment_id, results_root))
    (results_root / "summary.json").write_text(
        json.dumps({"experiments": rows}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(results_root / "summary.csv", index=False, encoding="utf-8")
    failures = [row for row in rows if not row["calc_ok"] or not row["plot_ok"]]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
