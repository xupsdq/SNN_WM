from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.bundle_io import save_csv_with_registry
from src.experiments.paper_figures.fig1.subexperiments.helpers import (
    _balanced_disjoint_delay_trials,
    _balanced_image_trials,
    _build_dms_trials,
)
from src.experiments.paper_figures.fig1.types import ExperimentContext


def build_trial_specs(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.cfg
    rng = np.random.default_rng(int(cfg.network_seed))
    baseline = _balanced_image_trials(
        ctx.class_index,
        per_class=cfg.baseline_eval_per_class,
        rng=rng,
        network_seed=cfg.network_seed,
        split=cfg.split,
        id_prefix="baseline",
    )
    baseline = baseline[["network_seed", "trial_id", "image_id", "label", "split"]]

    train, test, overlap = _balanced_disjoint_delay_trials(
        ctx.class_index,
        train_per_class=cfg.delay_decode_train_per_class,
        test_per_class=cfg.delay_decode_test_per_class,
        rng=rng,
        network_seed=cfg.network_seed,
    )
    if overlap:
        ctx.warnings.append(f"Delay train/test image overlap was unavoidable for {overlap} image IDs.")

    dms, audit_rows = _build_dms_trials(
        ctx.class_index,
        n_trials=cfg.dms_num_trials,
        rng=rng,
        network_seed=cfg.network_seed,
    )

    save_csv_with_registry(ctx, baseline, ctx.trial_specs_dir / "baseline_eval_trials.csv")
    save_csv_with_registry(ctx, train, ctx.trial_specs_dir / "delay_decode_train_trials.csv")
    save_csv_with_registry(ctx, test, ctx.trial_specs_dir / "delay_decode_test_trials.csv")
    save_csv_with_registry(ctx, dms, ctx.trial_specs_dir / "dms_shuffle_trials.csv")
    save_csv_with_registry(ctx, pd.DataFrame(audit_rows), ctx.metrics_dir / "supp_trial_condition_audit.csv")
    ctx.n_trials.update(
        {
            "baseline": len(baseline),
            "delay_train": len(train),
            "delay_test": len(test),
            "dms": len(dms),
        }
    )
    ctx.completed_modules["trial_specs"] = True
    return {"baseline": baseline, "delay_train": train, "delay_test": test, "dms": dms}


__all__ = ["build_trial_specs"]
