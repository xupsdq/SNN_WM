from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_peak_input_overlap_origin(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    rows: list[dict[str, Any]] = []
    example_payload: dict[str, np.ndarray | str] = {}
    for seq_idx, meta in _progress(enumerate(bank.sequence_meta.itertuples(index=False)), total=len(bank.sequence_meta), desc="fig6 overlap origin", enabled=ctx.cfg.show_progress):
        seq_id = int(meta.sequence_id)
        seq_len = int(meta.seq_len)
        item_maps = bank.item_activation_history[seq_idx, :seq_len, :] > 0
        peak = bank.peak_mask[seq_idx].reshape(-1)
        delta = bank.delta_support[seq_idx].reshape(-1)
        maps: list[tuple[str, str, int, int, np.ndarray]] = [("all", "all", 1, seq_len, item_maps.sum(axis=0))]
        for k in tuple(int(v) for v in ctx.cfg.recent_overlap_windows):
            start = max(0, seq_len - k)
            recent_map = item_maps[start:seq_len, :].sum(axis=0)
            old_map = item_maps[:start, :].sum(axis=0) if start > 0 else np.zeros_like(recent_map)
            maps.append((f"recent_{k}", "recent", start + 1, seq_len, recent_map))
            maps.append((f"old_{k}", "old", 1, start, old_map))
        for window_name, overlap_type, start_pos, end_pos, overlap in maps:
            high, fallback = _high_overlap_mask(overlap, int(np.sum(peak)))
            inter = peak & high
            n_overlap = int(np.sum(overlap >= 2))
            n_peak = int(np.sum(peak))
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "overlap_window": window_name,
                    "window_start_position": int(start_pos),
                    "window_end_position": int(end_pos),
                    "n_items_in_window": int(max(0, end_pos - start_pos + 1)),
                    "overlap_type": overlap_type,
                    "n_overlap_pixels": n_overlap,
                    "n_peak_pixels": n_peak,
                    "dice_peak_overlap": _dice(peak, high),
                    "jaccard_peak_overlap": _jaccard(peak, high),
                    "peak_coverage": _safe_div(float(np.sum(inter)), float(n_peak)),
                    "overlap_precision": _safe_div(float(np.sum(inter)), float(max(1, np.sum(high)))),
                    "cosine_delta_support_overlap_count": _plain_cosine(delta, overlap),
                    "spearman_delta_support_overlap_count": _spearman(delta, overlap),
                    "fallback_used": bool(fallback),
                }
            )
        if not example_payload:
            recent2 = item_maps[max(0, seq_len - 2) : seq_len, :].sum(axis=0).reshape(28, 28)
            recent3 = item_maps[max(0, seq_len - 3) : seq_len, :].sum(axis=0).reshape(28, 28)
            old = item_maps[: max(0, seq_len - 2), :].sum(axis=0).reshape(28, 28) if seq_len > 2 else np.zeros((28, 28), dtype=np.float32)
            example_payload = {
                "peak_mask": peak.reshape(28, 28).astype(np.uint8),
                "delta_support_map": delta.reshape(28, 28).astype(np.float32),
                "all_input_overlap_count": item_maps.sum(axis=0).reshape(28, 28).astype(np.float32),
                "recent_2_overlap_count": recent2.astype(np.float32),
                "recent_3_overlap_count": recent3.astype(np.float32),
                "old_overlap_count": old.astype(np.float32),
                "high_overlap_mask_recent_2": _high_overlap_mask(recent2.reshape(-1), int(np.sum(peak)))[0].reshape(28, 28).astype(np.uint8),
                "high_overlap_mask_recent_3": _high_overlap_mask(recent3.reshape(-1), int(np.sum(peak)))[0].reshape(28, 28).astype(np.uint8),
                "selected_sequence_metadata": json.dumps(_json_safe(meta._asdict()), sort_keys=True),
            }
    df = pd.DataFrame(rows, columns=PANEL_C_ORIGIN_COLUMNS)
    _save_csv(ctx, df, ctx.metrics_dir / "panel_c_peak_input_overlap_similarity.csv")
    summary_rows = []
    if not df.empty:
        for (network_seed, window), part in df.groupby(["network_seed", "overlap_window"], sort=False):
            dice = pd.to_numeric(part["dice_peak_overlap"], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "network_seed": int(network_seed),
                    "overlap_window": str(window),
                    "mean_dice": float(np.mean(dice)) if dice.size else np.nan,
                    "sem_dice": _sem(dice) if dice.size else np.nan,
                    "mean_peak_coverage": _mean_col(part, "peak_coverage"),
                    "mean_cosine": _mean_col(part, "cosine_delta_support_overlap_count"),
                    "n_sequences": int(part["sequence_id"].nunique()),
                }
            )
    summary = pd.DataFrame(summary_rows, columns=PANEL_C_ORIGIN_SUMMARY_COLUMNS)
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_c_peak_input_overlap_similarity_summary.csv")
    _save_csv(ctx, summary, ctx.metrics_dir / "panel_c_peak_input_overlap_summary.csv")
    if example_payload:
        np.savez_compressed(ctx.raw_dir / "panel_c_peak_input_overlap_example.npz", **example_payload)
        ctx.output_files["panel_c_peak_input_overlap_example"] = "data/raw/panel_c_peak_input_overlap_example.npz"
    ctx.completed_modules["peak_input_overlap_origin"] = True
