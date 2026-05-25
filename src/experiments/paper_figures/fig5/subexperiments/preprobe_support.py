from __future__ import annotations

from src.experiments.paper_figures import fig5_local_support_competition_experiment as _legacy

# Keep module-level names identical while Fig.5 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_preprobe_support_metrics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> None:
    rows: list[dict[str, Any]] = []
    groups = list(bank.unit_groups.groupby("trial_id", sort=False))
    for trial_id, part in _progress(groups, total=len(groups), desc="fig5 preprobe metrics", enabled=ctx.cfg.show_progress):
        overall = float(pd.to_numeric(part["support_value"], errors="coerce").mean())
        overlap_mean = _mean_for_group(part, "overlap_dominant")
        probe_mean = _mean_for_group(part, "probe_only_dominant")
        for group in UNIT_GROUPS:
            subset = part[part["unit_group"].eq(group)]
            values = pd.to_numeric(subset["support_value"], errors="coerce")
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_id),
                    "unit_group": group,
                    "layer": PRIMARY_LAYER,
                    "state_variable": "g",
                    "mean_support": float(values.mean()) if not values.empty else float("nan"),
                    "total_support": float(values.sum()) if not values.empty else 0.0,
                    "support_area": int(values.count()),
                    "support_enrichment": float(values.mean() / (overall + 1e-9)) if not values.empty else float("nan"),
                    "overlap_minus_probe_only_support": float(overlap_mean - probe_mean),
                    "n_units": int(values.count()),
                }
            )
    _save_csv(ctx, pd.DataFrame(rows, columns=PANEL_A_COLUMNS), ctx.metrics_dir / "panel_a_preprobe_support_metrics.csv")
    ctx.completed_modules["preprobe_support"] = True
