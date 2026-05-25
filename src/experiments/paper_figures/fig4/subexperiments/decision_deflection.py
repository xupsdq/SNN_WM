from __future__ import annotations

from src.experiments.paper_figures import fig4_overlap_reentry_experiment as _legacy

# Keep module-level names identical while Fig.4 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def compute_decision_deflection_metrics(ctx: ExperimentContext, bank: OverlapReentryDMSBank | OverlapPerturbationCompatibleBank) -> None:
    missing_reason = ""
    rows = []
    try:
        for _, pair in bank.pair_trials.iterrows():
            pair_id = int(pair["pair_id"])
            v_dyn = _vector(bank, pair_id, "full_dynamic")
            v_sta = _vector(bank, pair_id, "full_static")
            pred_dyn = int(_cond_row(bank.condition_metrics, pair_id, "full_dynamic")["prediction"])
            pred_sta = int(_cond_row(bank.condition_metrics, pair_id, "full_static")["prediction"])
            for condition in CORE_CONDITIONS:
                v_cond = _vector(bank, pair_id, condition)
                s_dyn = _cosine(v_cond, v_dyn)
                s_sta = _cosine(v_cond, v_sta)
                push = _projection(v_cond - v_sta, v_dyn - v_sta)
                recovery = s_dyn - s_sta
                pred_cond = int(_cond_row(bank.condition_metrics, pair_id, condition)["prediction"])
                rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "pair_id": pair_id,
                        "condition": condition,
                        "similarity_bin": str(pair["similarity_bin"]),
                        "overlap_bin": str(pair["overlap_bin"]),
                        "trajectory_distance_dynamic_static": float(np.linalg.norm(v_dyn - v_sta)),
                        "condition_to_dynamic_similarity": s_dyn,
                        "condition_to_static_similarity": s_sta,
                        "dynamic_like_recovery": recovery,
                        "static_to_dynamic_push": push,
                        "decision_deflection_score": push,
                        "prediction_dynamic": pred_dyn,
                        "prediction_static": pred_sta,
                        "prediction_condition": pred_cond,
                        "condition_matches_dynamic": int(pred_cond == pred_dyn),
                        "condition_matches_static": int(pred_cond == pred_sta),
                        "x0": 0.0,
                        "y0": 0.0,
                        "x1": push,
                        "y1": recovery,
                    }
                )
    except KeyError as exc:
        missing_reason = f"overlap_vectors_missing:{exc}"
        ctx.warnings.append(f"Decision deflection metrics unavailable: {missing_reason}")
    df = pd.DataFrame(rows)
    _save_csv(ctx, df, ctx.metrics_dir / "supp_decision_deflection_metrics.csv")
    _save_csv(ctx, df.copy(), ctx.metrics_dir / "supp_s8_decision_deflection_metrics.csv")
    summary = _decision_deflection_summary(df)
    _save_csv(ctx, summary, ctx.metrics_dir / "supp_s8_decision_deflection_summary.csv")
    available = bool(not df.empty)
    ctx.availability["decision_deflection_available"] = available
    ctx.availability["decision_deflection_missing_reason"] = None if available else (missing_reason or "decision_deflection_metrics_empty")
    ctx.completed_modules["simplified_decision_deflection_supplement"] = True

def compute_l3_accumulator_region_replay_metrics(
    ctx: ExperimentContext,
    pair_trials: pd.DataFrame,
) -> L3AccumulatorReplayBank:
    cfg = ctx.cfg
    readout_step = _resolve_fig4_readout_step(ctx)
    work_pairs = pair_trials.copy()
    if cfg.smoke:
        work_pairs = work_pairs.head(min(4, len(work_pairs))).copy()
    images_cache = _image_cache(ctx, work_pairs)
    pair_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    pair_ids: list[int] = []
    delta_v_rows: list[np.ndarray] = []
    delta_hat_plus_rows: list[np.ndarray] = []
    delta_hat_minus_rows: list[np.ndarray] = []
    v_dynamic_rows: list[np.ndarray] = []
    v_static_rows: list[np.ndarray] = []
    pred_dynamic_rows: list[int] = []
    pred_static_rows: list[int] = []
    region_effect_payload: dict[str, list[np.ndarray]] = {
        "pair_id": [],
        "region_id": [],
        "D_dyn": [],
        "D_sta": [],
        "E_dyn": [],
        "E_sta": [],
        "R_plus": [],
        "R_minus": [],
        "R_plus_tilde": [],
        "R_minus_tilde": [],
    }
    regions = None
    for case_idx, row in enumerate(
        _progress(work_pairs.itertuples(index=False), total=len(work_pairs), desc="fig4 L3 accumulator replay", enabled=cfg.show_progress)
    ):
        sample_image = images_cache[int(row.sample_image_id)].unsqueeze(0)
        probe_image = images_cache[int(row.probe_image_id)].unsqueeze(0)
        sample_spikes = _encode_batch(ctx, sample_image, cfg.sample_steps)
        probe_spikes = _encode_batch(ctx, probe_image, cfg.probe_steps)
        dynamic_capture = run_dms_with_l3_trace_capture(
            ctx.net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=cfg.delay_steps,
            stsp_mode="dynamic",
            readout_step=readout_step,
            phase_reset=True,
        )
        static_capture = run_dms_with_l3_trace_capture(
            ctx.net,
            sample_spikes=sample_spikes,
            probe_spikes=probe_spikes,
            delay_steps=cfg.delay_steps,
            stsp_mode="static_frozen",
            readout_step=readout_step,
            phase_reset=True,
        )
        if regions is None:
            regions = make_l3_region_masks(
                int(dynamic_capture.probe_s2p_trace.shape[-2]),
                int(dynamic_capture.probe_s2p_trace.shape[-1]),
                mask_mode=str(cfg.l3_mask_mode),
            )
        num_classes = int(getattr(ctx.net.layer3, "num_classes", NUM_CLASSES))
        if bool(cfg.run_l3_region_deletion):
            deletion = run_l3_deletion_analysis_for_pair(
                ctx.net,
                dynamic_capture=dynamic_capture,
                static_capture=static_capture,
                regions=regions,
                batch_size=int(cfg.l3_region_batch_size),
            )
        else:
            nan_matrix = np.full((len(regions), num_classes), np.nan, dtype=np.float64)
            deletion = {"D_dyn": nan_matrix.copy(), "D_sta": nan_matrix.copy(), "E_dyn": nan_matrix.copy(), "E_sta": nan_matrix.copy()}
        if bool(cfg.run_l3_region_replacement):
            replacement = run_l3_replacement_analysis_for_pair(
                ctx.net,
                dynamic_capture=dynamic_capture,
                static_capture=static_capture,
                regions=regions,
                batch_size=int(cfg.l3_region_batch_size),
            )
        else:
            nan_matrix = np.full((len(regions), num_classes), np.nan, dtype=np.float64)
            replacement = {
                "R_plus": nan_matrix.copy(),
                "R_minus": nan_matrix.copy(),
                "R_plus_tilde": nan_matrix.copy(),
                "R_minus_tilde": nan_matrix.copy(),
            }
        v_dyn = np.asarray(dynamic_capture.grouped_voltage[0], dtype=np.float64)
        v_sta = np.asarray(static_capture.grouped_voltage[0], dtype=np.float64)
        delta_v = _legacy_center_vector(v_dyn) - _legacy_center_vector(v_sta)
        bias_direction = int(np.argmax(np.abs(delta_v))) if delta_v.size else -1
        bias_magnitude = float(delta_v[bias_direction]) if bias_direction >= 0 else float("nan")
        delta_hat_plus = np.nansum(np.asarray(replacement["R_plus_tilde"], dtype=np.float64), axis=0)
        delta_hat_minus = np.nansum(np.asarray(replacement["R_minus_tilde"], dtype=np.float64), axis=0)
        sim_plus = _legacy_vector_similarity(delta_hat_plus, delta_v)
        sim_minus = _legacy_vector_similarity(delta_hat_minus, delta_v)
        e_dyn_k = np.asarray(deletion["E_dyn"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        e_sta_k = np.asarray(deletion["E_sta"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        r_plus_k = np.asarray(replacement["R_plus_tilde"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        r_minus_k = np.asarray(replacement["R_minus_tilde"][:, bias_direction], dtype=np.float64) if bias_direction >= 0 else np.asarray([])
        pair_id = int(row.pair_id)
        pair_rows.append(
            {
                "network_seed": int(cfg.network_seed),
                "pair_id": pair_id,
                "sample_image_id": int(row.sample_image_id),
                "probe_image_id": int(row.probe_image_id),
                "sample_label": int(row.sample_label),
                "probe_label": int(row.probe_label),
                "prediction_dynamic": int(dynamic_capture.prediction_probe[0]),
                "prediction_static": int(static_capture.prediction_probe[0]),
                "correct_dynamic": int(int(dynamic_capture.prediction_probe[0]) == int(row.probe_label)),
                "correct_static": int(int(static_capture.prediction_probe[0]) == int(row.probe_label)),
                "first_fire_dynamic": int(dynamic_capture.first_fire_t_probe[0]),
                "first_fire_static": int(static_capture.first_fire_t_probe[0]),
                "bias_direction": int(bias_direction),
                "bias_magnitude": float(bias_magnitude),
                "replacement_push_kstar": float(np.nanmean(r_plus_k)) if r_plus_k.size else float("nan"),
                "replacement_pullback_kstar": float(np.nanmean(r_minus_k)) if r_minus_k.size else float("nan"),
                "deletion_dynamic_minus_static_kstar": float(np.nanmean(e_dyn_k - e_sta_k)) if e_dyn_k.size and e_sta_k.size else float("nan"),
                "reconstruction_cosine_plus": float(sim_plus["cosine"]),
                "reconstruction_cosine_minus": float(sim_minus["cosine"]),
                "direction_match_plus": int(np.nanargmax(delta_hat_plus) == bias_direction) if np.isfinite(delta_hat_plus).any() and bias_direction >= 0 else 0,
                "direction_match_minus": int(np.nanargmax(delta_hat_minus) == bias_direction) if np.isfinite(delta_hat_minus).any() and bias_direction >= 0 else 0,
                "n_regions": int(len(regions)),
                "l3_mask_mode": str(cfg.l3_mask_mode),
                "readout_step": int(readout_step),
            }
        )
        if case_idx < int(cfg.save_case_count):
            case_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "case_id": int(case_idx),
                    "pair_id": pair_id,
                    "selection_reason": "first_cases_smoke" if cfg.smoke else "first_cases",
                    "sample_image_id": int(row.sample_image_id),
                    "probe_image_id": int(row.probe_image_id),
                }
            )
        pair_ids.append(pair_id)
        delta_v_rows.append(delta_v)
        delta_hat_plus_rows.append(delta_hat_plus)
        delta_hat_minus_rows.append(delta_hat_minus)
        v_dynamic_rows.append(v_dyn)
        v_static_rows.append(v_sta)
        pred_dynamic_rows.append(int(dynamic_capture.prediction_probe[0]))
        pred_static_rows.append(int(static_capture.prediction_probe[0]))
        for effect_name in ("D_dyn", "D_sta", "E_dyn", "E_sta"):
            pass
        for region_idx, region in enumerate(regions):
            region_effect_payload["pair_id"].append(np.asarray([pair_id], dtype=np.int64))
            region_effect_payload["region_id"].append(np.asarray([int(region.region_id)], dtype=np.int64))
            for name, source in (
                ("D_dyn", deletion["D_dyn"]),
                ("D_sta", deletion["D_sta"]),
                ("E_dyn", deletion["E_dyn"]),
                ("E_sta", deletion["E_sta"]),
                ("R_plus", replacement["R_plus"]),
                ("R_minus", replacement["R_minus"]),
                ("R_plus_tilde", replacement["R_plus_tilde"]),
                ("R_minus_tilde", replacement["R_minus_tilde"]),
            ):
                region_effect_payload[name].append(np.asarray(source[region_idx], dtype=np.float64))
    results = pd.DataFrame(pair_rows)
    summary_dict = summarize_l3_mechanism_results(results) if not results.empty else {}
    summary_rows = _l3_summary_rows(results, summary_dict, int(cfg.network_seed))
    pair_payload = {
        "pair_id": np.asarray(pair_ids, dtype=np.int64),
        "delta_v": np.stack(delta_v_rows, axis=0) if delta_v_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "Delta_hat_plus": np.stack(delta_hat_plus_rows, axis=0) if delta_hat_plus_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "Delta_hat_minus": np.stack(delta_hat_minus_rows, axis=0) if delta_hat_minus_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "v_dynamic": np.stack(v_dynamic_rows, axis=0) if v_dynamic_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "v_static": np.stack(v_static_rows, axis=0) if v_static_rows else np.zeros((0, NUM_CLASSES), dtype=np.float64),
        "prediction_dynamic": np.asarray(pred_dynamic_rows, dtype=np.int64),
        "prediction_static": np.asarray(pred_static_rows, dtype=np.int64),
    }
    region_payload = {
        name: (np.concatenate(values, axis=0) if name in {"pair_id", "region_id"} and values else np.stack(values, axis=0) if values else np.zeros((0, NUM_CLASSES), dtype=np.float64))
        for name, values in region_effect_payload.items()
    }
    _save_csv(ctx, results, ctx.metrics_dir / "panel_f_l3_accumulator_region_replay_metrics.csv")
    _save_csv(ctx, pd.DataFrame(summary_rows), ctx.metrics_dir / "panel_f_l3_accumulator_summary.csv")
    _save_csv(ctx, pd.DataFrame(case_rows), ctx.raw_dir / "panel_f_l3_accumulator_case_metadata.csv")
    np.savez_compressed(ctx.raw_dir / "panel_f_l3_accumulator_pair_vectors.npz", **pair_payload)
    np.savez_compressed(ctx.raw_dir / "panel_f_l3_accumulator_region_effects.npz", **region_payload)
    ctx.output_files["panel_f_l3_accumulator_pair_vectors"] = "data/raw/panel_f_l3_accumulator_pair_vectors.npz"
    ctx.output_files["panel_f_l3_accumulator_region_effects"] = "data/raw/panel_f_l3_accumulator_region_effects.npz"
    ctx.completed_modules["decision_deflection"] = True
    return L3AccumulatorReplayBank(pair_trials, results, pd.DataFrame(summary_rows), pair_payload, region_payload)
