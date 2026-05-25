from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def save_debug_figures(ctx: ExperimentContext) -> None:
    apply_publication_style()
    jobs = [
        ("panel_b_progressive_update_metrics.csv", "stepwise_update_ratio", "fig3_debug_progressive_update"),
        ("panel_c_example_landscape_summary.csv", "peak_mean_support", "fig3_debug_example_landscape"),
        ("panel_d_ping_position_distribution.csv", "readout_mass", "fig3_debug_ping_distribution"),
        ("panel_e_weak_probe_metrics.csv", "P_target", "fig3_debug_weak_probe_target"),
        ("panel_e_weak_probe_memory_gain.csv", "target_recovery_gain", "fig3_debug_weak_probe_gain"),
    ]
    for filename, column, stem in jobs:
        path = ctx.metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if column not in df.columns or df.empty:
            continue
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        ax.hist(values, bins=min(20, max(3, len(values))), color="#4C78A8", alpha=0.8)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        save_figure_all_formats(fig, ctx.debug_dir / stem)
        plt.close(fig)
    _save_debug_category_plot(ctx, "panel_f_peak_cue_memory_gain.csv", "cue_condition", "memory_gain", "panel_f_peak_cue_memory_gain")
    matching_path = ctx.metrics_dir / "panel_f_peak_cue_matching_diagnostics.csv"
    if matching_path.exists():
        matching = pd.read_csv(matching_path)
        y_column = "cue_energy" if "cue_energy" in matching.columns else "encoded_spike_count"
        if y_column in matching.columns:
            _save_debug_category_plot(ctx, "panel_f_peak_cue_matching_diagnostics.csv", "cue_condition", y_column, "panel_f_peak_cue_matching")
    serial_path = ctx.metrics_dir / "supp_peak_cue_serial_position_gain.csv"
    if serial_path.exists():
        serial = pd.read_csv(serial_path)
        x_column = "target_position_bin" if "target_position_bin" in serial.columns else "relative_position"
        if x_column in serial.columns and "memory_gain" in serial.columns:
            _save_debug_category_plot(ctx, "supp_peak_cue_serial_position_gain.csv", x_column, "memory_gain", "supp_peak_cue_serial_position_gain")
    ctx.completed_modules["debug_figures"] = True

def _save_debug_category_plot(ctx: ExperimentContext, filename: str, x_column: str, y_column: str, stem: str) -> None:
    path = ctx.metrics_dir / filename
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        return
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_column])
    if plot_df.empty:
        return
    grouped = plot_df.groupby(x_column, sort=True)[y_column].mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=150)
    ax.bar(grouped[x_column].astype(str), grouped[y_column].astype(float), color="#4C78A8", alpha=0.85)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=30)
    save_figure_all_formats(fig, ctx.debug_dir / stem)
    plt.close(fig)
