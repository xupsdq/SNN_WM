from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_peak_source_attribution(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank, loo_bank: dict[int, list[dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    proxy_mode = _is_proxy_mode(ctx)
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 source attribution", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        seq_len = int(meta.seq_len)
        peak = bank.peak_mask[seq_idx].reshape(-1)
        nonpeak = bank.nonpeak_mask[seq_idx].reshape(-1)
        prior = bank.prior_updated_mask[seq_idx].reshape(-1)
        seq_rows = loo_bank.get(seq_id, [])
        peak_losses = np.asarray([float(np.sum(r["loss_map_i"][peak])) for r in seq_rows], dtype=float)
        nonpeak_losses = np.asarray([float(np.sum(r["loss_map_i"][nonpeak])) for r in seq_rows], dtype=float)
        peak_total = float(np.nansum(peak_losses))
        nonpeak_total = float(np.nansum(nonpeak_losses))
        for i, replay in enumerate(seq_rows):
            loss = np.asarray(replay["loss_map_i"], dtype=float)
            peak_loss = float(peak_losses[i])
            nonpeak_loss = float(nonpeak_losses[i])
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "removed_position": int(replay["removed_position"]),
                    "removed_label": int(replay["removed_label"]),
                    "removed_image_id": int(replay["removed_image_id"]),
                    "peak_loss": peak_loss,
                    "nonpeak_loss": nonpeak_loss,
                    "prior_updated_loss": float(np.sum(loss[prior])),
                    "peak_loss_fraction": _safe_div(peak_loss, peak_total),
                    "nonpeak_loss_fraction": _safe_div(nonpeak_loss, nonpeak_total),
                    "peak_vs_nonpeak_loss_ratio": _safe_div(peak_loss, max(nonpeak_loss, 1e-12)),
                    "support_loss_total": float(np.sum(loss)),
                    "leave_one_out_mode": str(ctx.cfg.leave_one_out_mode),
                    "proxy_mode": bool(proxy_mode),
                }
            )
    df = pd.DataFrame(rows, columns=PANEL_A_SOURCE_COLUMNS)
    _save_csv(ctx, df, ctx.metrics_dir / "panel_a_peak_source_attribution.csv")
    _save_csv(ctx, df, ctx.raw_dir / "panel_a_peak_source_attribution.csv")
    if not df.empty:
        for (network_seed, seq_len, pos), part in df.groupby(["network_seed", "seq_len", "removed_position"], sort=True):
            vals = pd.to_numeric(part["peak_loss_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
            ratios = pd.to_numeric(part["peak_vs_nonpeak_loss_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "network_seed": int(network_seed),
                    "seq_len": int(seq_len),
                    "removed_position": int(pos),
                    "relative_position_from_end": int(seq_len) - int(pos),
                    "mean_peak_loss_fraction": float(np.mean(vals)) if vals.size else np.nan,
                    "sem_peak_loss_fraction": _sem(vals) if vals.size else np.nan,
                    "mean_peak_vs_nonpeak_loss_ratio": float(np.mean(ratios)) if ratios.size else np.nan,
                    "n_sequences": int(part["sequence_id"].nunique()),
                }
            )
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=PANEL_A_SOURCE_SUMMARY_COLUMNS), ctx.metrics_dir / "panel_a_peak_source_attribution_summary.csv")
    ctx.completed_modules["peak_source_attribution"] = True
    if proxy_mode:
        ctx.warnings.append("Fig.6A leave-one-out attribution used proxy support replay; use real model replay for final scientific evidence.")
