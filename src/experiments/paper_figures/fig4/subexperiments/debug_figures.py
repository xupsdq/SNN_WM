from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    debug_specs = [
        ("fig4_debug_similarity_entry", ctx.metrics_dir / "panel_b_similarity_bin_summary.csv", "similarity_bin", "mean_acc_drop"),
        ("fig4_debug_overlap_localization", ctx.metrics_dir / "panel_c_overlap_localization_metrics.csv", "dice_overlap", "acc_drop"),
        ("fig4_debug_s7_iso_similarity_overlap_contrast", ctx.metrics_dir / "supp_s7_iso_similarity_overlap_contrast.csv", "network_seed", "delta_drop_rate"),
        ("fig4_debug_l3_trace_displacement", ctx.metrics_dir / "panel_e_time_resolved_l3_displacement.csv", "time_ms", "DPI_L3_t"),
        ("fig4_debug_l3_accumulator", ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv", "replacement_push_kstar", "replacement_pullback_kstar"),
        ("fig4_debug_overlap_perturbation_contrast", ctx.metrics_dir / "panel_d_overlap_perturbation_contrast.csv", "network_seed", "overlap_minus_nonoverlap_DPI"),
        ("fig4_debug_s8_decision_deflection", ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv", "condition", "mean_decision_deflection_score"),
    ]
    for stem, path, x_col, y_col in debug_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=150)
        if x_col == "condition":
            ax.bar(np.arange(len(df)), pd.to_numeric(df[y_col], errors="coerce"))
            ax.set_xticks(np.arange(len(df)), [CONDITION_LABELS.get(str(v), str(v)) for v in df[x_col]], rotation=30, ha="right")
        else:
            x_num = pd.to_numeric(df[x_col], errors="coerce")
            if x_num.notna().any():
                ax.scatter(x_num, pd.to_numeric(df[y_col], errors="coerce"), s=16)
            else:
                labels = [str(v) for v in df[x_col]]
                ax.scatter(np.arange(len(df)), pd.to_numeric(df[y_col], errors="coerce"), s=16)
                ax.set_xticks(np.arange(len(df)), labels, rotation=30, ha="right")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    ctx.completed_modules["debug_figures"] = True
