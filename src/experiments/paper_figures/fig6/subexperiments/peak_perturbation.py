from __future__ import annotations

from src.experiments.paper_figures import fig6_peak_amplified_reentry_experiment as _legacy

# Keep module-level names identical while Fig.6 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _write_standardized_peak_perturbation_outputs(ctx: ExperimentContext) -> None:
    d_source = _read_csv_if_exists(ctx.raw_dir / "panel_d_route_peak_perturbation_trial_readout.csv")
    e_source = _read_csv_if_exists(ctx.raw_dir / "panel_e_route_peak_downstream_trial_readout.csv")
    metric_columns = ["network_seed", "sequence_id", "condition", "perturbation_target", "reentry_strength", "decision_deflection_score", "downstream_metric", "metric_value", "overlap_aligned_peak", "control_peak", "random_matched_peak", "n_units_perturbed"]
    summary_columns = ["network_seed", "metric", "intact_value", "overlap_peak_perturb_value", "control_peak_perturb_value", "random_peak_perturb_value", "overlap_peak_reduction", "control_peak_reduction", "overlap_minus_control_reduction", "n_sequences", "claim_upgrade_allowed"]
    if d_source is None or e_source is None:
        ctx.warnings.append("Peak perturbation requested but route-peak perturbation source outputs are missing.")
        return
    rows = []
    for _, r in d_source.iterrows():
        unit_set = str(r.get("perturbation_unit_set", ""))
        target = _perturbation_target(unit_set)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.get("sequence_id", -1)) if pd.notna(r.get("sequence_id", np.nan)) else -1,
                "condition": unit_set,
                "perturbation_target": target,
                "reentry_strength": _num(r.get("reentry_strength_perturbed")),
                "decision_deflection_score": np.nan,
                "downstream_metric": "normalized_reentry_loss",
                "metric_value": _num(r.get("normalized_reentry_loss")),
                "overlap_aligned_peak": bool(target == "overlap_aligned_peak"),
                "control_peak": bool(target == "control_peak"),
                "random_matched_peak": bool(target == "random_matched_peak"),
                "n_units_perturbed": int(r.get(f"{unit_set}_unit_count", r.get("route_peak_unit_count", 0))) if unit_set in PERTURBATION_UNIT_SET_ORDER else 0,
            }
        )
    for _, r in e_source.iterrows():
        unit_set = str(r.get("perturbation_unit_set", ""))
        target = _perturbation_target(unit_set)
        for metric in ("output_switch", "response_displacement_loss", "decision_deflection_loss"):
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(r.get("sequence_id", -1)) if pd.notna(r.get("sequence_id", np.nan)) else -1,
                    "condition": unit_set,
                    "perturbation_target": target,
                    "reentry_strength": np.nan,
                    "decision_deflection_score": _num(r.get("decision_deflection_loss")),
                    "downstream_metric": metric,
                    "metric_value": float(_bool_value(r.get(metric))) if metric == "output_switch" else _num(r.get(metric)),
                    "overlap_aligned_peak": bool(target == "overlap_aligned_peak"),
                    "control_peak": bool(target == "control_peak"),
                    "random_matched_peak": bool(target == "random_matched_peak"),
                    "n_units_perturbed": 0,
                }
            )
    metrics = pd.DataFrame(rows, columns=metric_columns)
    _save_csv(ctx, metrics, ctx.metrics_dir / "supp_s12_peak_perturbation_metrics.csv")
    summary_rows = []
    for metric, part in metrics.groupby("downstream_metric", sort=True):
        intact = pd.to_numeric(part[part["perturbation_target"].eq("intact")]["metric_value"], errors="coerce").mean()
        overlap = pd.to_numeric(part[part["perturbation_target"].eq("overlap_aligned_peak")]["metric_value"], errors="coerce").mean()
        control = pd.to_numeric(part[part["perturbation_target"].eq("control_peak")]["metric_value"], errors="coerce").mean()
        random = pd.to_numeric(part[part["perturbation_target"].eq("random_matched_peak")]["metric_value"], errors="coerce").mean()
        overlap_reduction = intact - overlap
        control_reduction = np.nanmax([intact - control, intact - random])
        claim_upgrade = bool(np.isfinite(overlap_reduction) and np.isfinite(control_reduction) and overlap_reduction > control_reduction and not part.empty)
        summary_rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "metric": str(metric),
                "intact_value": float(intact),
                "overlap_peak_perturb_value": float(overlap),
                "control_peak_perturb_value": float(control),
                "random_peak_perturb_value": float(random),
                "overlap_peak_reduction": float(overlap_reduction),
                "control_peak_reduction": float(control_reduction),
                "overlap_minus_control_reduction": float(overlap_reduction - control_reduction),
                "n_sequences": int(part["sequence_id"].nunique()),
                "claim_upgrade_allowed": claim_upgrade,
            }
        )
    _save_csv(ctx, pd.DataFrame(summary_rows, columns=summary_columns), ctx.metrics_dir / "supp_s12_peak_perturbation_summary.csv")

def compute_route_peak_perturbation_outputs(ctx: ExperimentContext, bank: PeakAmplifiedReentryBank) -> None:
    if bank.probe_trials.empty:
        build_later_probe_peak_overlap_trials(ctx, bank)
    if bank.probe_trials.empty:
        raise RuntimeError("Fig.6 D/E route-peak perturbation requires non-empty probe trials.")
    if _is_proxy_mode(ctx):
        raise RuntimeError("Fig.6 D/E route-peak perturbation requires a loaded real network; proxy mode is disabled.")
    rows_d: list[dict[str, Any]] = []
    rows_e: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    output_distribution_rows: list[dict[str, Any]] = []
    encode_cache: dict[tuple[Any, ...], Any] = {}
    for r in _progress(bank.probe_trials.itertuples(index=False), total=len(bank.probe_trials), desc="fig6 route-peak perturbation", enabled=ctx.cfg.show_progress):
        seq_idx = _sequence_index(bank, int(r.sequence_id))
        boundary = bank.boundaries.get(int(r.sequence_id))
        if not boundary:
            raise RuntimeError(f"Fig.6 D/E route-peak perturbation missing S_final boundary for sequence={int(r.sequence_id)}.")
        probe_mask = _foreground_mask(ctx.dataset, int(r.probe_image_id), ctx.cfg.foreground_threshold).reshape(-1)
        peak = bank.peak_mask[seq_idx].reshape(-1).astype(bool)
        prior = bank.prior_updated_mask[seq_idx].reshape(-1).astype(bool)
        route = probe_mask.astype(bool) & prior
        set_masks = {
            "route_peak": route & peak,
            "route_nonpeak": route & ~peak,
            "nonroute_peak": peak & ~route,
        }
        set_masks["random_matched"] = _matched_random_unit_mask(~set_masks["route_peak"], int(set_masks["route_peak"].sum()), int(ctx.cfg.network_seed) + int(r.probe_id))
        counts = {name: int(mask.sum()) for name, mask in set_masks.items()}
        insufficient = {name: bool(count <= 0) for name, count in counts.items()}
        for name, mask in set_masks.items():
            for unit_id in np.flatnonzero(mask)[:200]:
                unit_rows.append(
                    {
                        "network_seed": int(ctx.cfg.network_seed),
                        "sequence_id": int(r.sequence_id),
                        "probe_id": int(r.probe_id),
                        "perturbation_unit_set": name,
                        "unit_id": int(unit_id),
                        "notes": "route_peak = probe foreground intersect prior-updated foreground intersect final peak mask",
                    }
                )
        try:
            probe_spikes = _encode_sequence_cached(ctx, [int(r.probe_image_id)], ctx.cfg.probe_steps, encode_cache)
            condition_results = _run_real_probe_conditions_batch(
                ctx,
                int(r.probe_image_id),
                (boundary, None),
                ("S_final", "S0"),
                probe_spikes=probe_spikes,
            )
            intact_trace, intact_pred, intact_fire, intact_vec = condition_results["S_final"]
            s0_trace, s0_pred, s0_fire, s0_vec = condition_results["S0"]
        except Exception as exc:
            raise RuntimeError(
                f"Fig.6 D/E baseline or intact rollout failed for sequence={int(r.sequence_id)} probe={int(r.probe_id)}: {exc}"
            ) from exc
        reentry_intact = float(np.linalg.norm(intact_trace.reshape(-1) - s0_trace.reshape(-1)))
        reentry_s0 = 0.0
        response_intact = float(np.linalg.norm(intact_vec.reshape(-1) - s0_vec.reshape(-1)))
        response_s0 = 0.0
        deflection_intact = float(_label_evidence(intact_vec, int(r.probe_label)) - _label_evidence(s0_vec, int(r.probe_label)))
        deflection_s0 = 0.0
        output_distribution_rows.append(_output_distribution_row(ctx, r, "intact", intact_vec, intact_vec, js=0.0))
        for unit_set in PERTURBATION_UNIT_SET_ORDER:
            selected = np.flatnonzero(set_masks[unit_set]).astype(np.int64)
            perturb_ok = False
            reset_record: dict[str, Any] = {}
            if insufficient[unit_set]:
                pert_trace = np.full_like(intact_trace, np.nan)
                pert_vec = np.full_like(intact_vec, np.nan)
                pert_pred = -1
                pert_fire = -1
                failure_reason = "insufficient_units"
            else:
                try:
                    pert_trace, pert_pred, pert_fire, pert_vec, reset_record = _run_real_probe_with_route_peak_reset(
                        ctx,
                        int(r.probe_image_id),
                        boundary,
                        selected,
                        probe_spikes=probe_spikes,
                    )
                    perturb_ok = bool(reset_record.get("restore_ok", False))
                    failure_reason = "" if perturb_ok else "perturbation_failed:stsp_reset_restore_not_ok"
                except Exception as exc:
                    pert_trace = np.full_like(intact_trace, np.nan)
                    pert_vec = np.full_like(intact_vec, np.nan)
                    pert_pred = -1
                    pert_fire = -1
                    failure_reason = f"perturbation_failed:{exc}"
                    ctx.warnings.append(f"Fig.6 {unit_set} reset failed for sequence={int(r.sequence_id)} probe={int(r.probe_id)}: {exc}")
            reentry_pert = float(np.linalg.norm(pert_trace.reshape(-1) - s0_trace.reshape(-1))) if perturb_ok else np.nan
            reentry_loss = float(reentry_intact - reentry_pert) if perturb_ok else np.nan
            normalized_loss = float(reentry_loss / max(abs(reentry_intact), 1e-9)) if perturb_ok else np.nan
            response_pert = float(np.linalg.norm(pert_vec.reshape(-1) - s0_vec.reshape(-1))) if perturb_ok else np.nan
            response_loss = float(response_intact - response_pert) if perturb_ok else np.nan
            deflection_pert = float(_label_evidence(pert_vec, int(r.probe_label)) - _label_evidence(s0_vec, int(r.probe_label))) if perturb_ok else np.nan
            deflection_loss = float(deflection_intact - deflection_pert) if perturb_ok else np.nan
            js = _js_divergence(intact_vec, pert_vec) if perturb_ok else np.nan
            output_distribution_rows.append(_output_distribution_row(ctx, r, unit_set, intact_vec, pert_vec, js=js))
            common = {
                "network_seed": int(ctx.cfg.network_seed),
                "sequence_id": int(r.sequence_id),
                "probe_id": int(r.probe_id),
                "probe_label": int(r.probe_label),
                "seq_len": int(bank.sequence_meta.iloc[seq_idx]["seq_len"]),
                "perturbation_unit_set": unit_set,
                "perturbation_condition": f"{unit_set}_reset",
                "perturbation_mode": "reset_u_x_to_s0",
                "state_condition": "S_final_then_reset_before_probe",
                "raw_overlap": float(r.raw_overlap),
                "peak_weighted_overlap": float(r.peak_weighted_overlap),
                "route_unit_count": int(route.sum()),
                "peak_unit_count": int(peak.sum()),
                "route_peak_unit_count": counts["route_peak"],
                "route_nonpeak_unit_count": counts["route_nonpeak"],
                "nonroute_peak_unit_count": counts["nonroute_peak"],
                "random_unit_count": counts["random_matched"],
                "insufficient_units": bool(insufficient[unit_set]),
                "restore_ok": bool(reset_record.get("restore_ok", perturb_ok)),
                "perturbation_ok": bool(perturb_ok),
                "failure_reason": failure_reason,
            }
            rows_d.append(
                {
                    **common,
                    "reentry_strength_intact": reentry_intact,
                    "reentry_strength_perturbed": reentry_pert,
                    "reentry_strength_s0": reentry_s0,
                    "reentry_loss": reentry_loss,
                    "normalized_reentry_loss": normalized_loss,
                    "prediction_intact": int(intact_pred),
                    "prediction_perturbed": int(pert_pred),
                    "prediction_s0": int(s0_pred),
                    "first_fire_time_intact": int(intact_fire),
                    "first_fire_time_perturbed": int(pert_fire),
                    "first_fire_time_s0": int(s0_fire),
                    "denominator_choice": "max(abs(reentry_strength_intact), eps)",
                    "reset_variables": str(reset_record.get("reset_variables", "u,x")),
                    "probe_input_unchanged": True,
                }
            )
            rows_e.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "sequence_id": int(r.sequence_id),
                    "probe_id": int(r.probe_id),
                    "probe_label": int(r.probe_label),
                    "perturbation_unit_set": unit_set,
                    "perturbation_condition": f"{unit_set}_reset",
                    "response_displacement_intact": response_intact,
                    "response_displacement_perturbed": response_pert,
                    "response_displacement_s0": response_s0,
                    "response_displacement_loss": response_loss,
                    "decision_deflection_intact": deflection_intact,
                    "decision_deflection_perturbed": deflection_pert,
                    "decision_deflection_s0": deflection_s0,
                    "decision_deflection_loss": deflection_loss,
                    "prediction_intact": int(intact_pred),
                    "prediction_perturbed": int(pert_pred),
                    "prediction_s0": int(s0_pred),
                    "output_switch": bool(perturb_ok and int(intact_pred) != int(pert_pred)),
                    "output_distribution_JS": js,
                    "perturbation_ok": bool(perturb_ok),
                    "insufficient_units": bool(insufficient[unit_set]),
                    "failure_reason": failure_reason,
                }
            )
    df_d = pd.DataFrame(rows_d, columns=PANEL_D_ROUTE_PEAK_TRIAL_COLUMNS)
    df_e = pd.DataFrame(rows_e, columns=PANEL_E_ROUTE_PEAK_TRIAL_COLUMNS)
    success = _route_peak_success(df_d, df_e)
    failure_reason = "" if success else _route_peak_failure_reason(df_d, df_e, n_total_probe_trials=int(len(bank.probe_trials)))
    _save_csv(ctx, df_d, ctx.raw_dir / "panel_d_route_peak_perturbation_trial_readout.csv")
    _save_csv(ctx, df_e, ctx.raw_dir / "panel_e_route_peak_downstream_trial_readout.csv")
    _save_csv(ctx, _route_peak_reentry_summary(ctx, df_d), ctx.metrics_dir / "panel_d_route_peak_reentry_loss_summary.csv")
    _save_csv(ctx, _route_peak_reentry_contrast(ctx, df_d), ctx.metrics_dir / "panel_d_route_peak_reentry_loss_contrast.csv")
    _save_csv(ctx, _route_peak_perturbation_audit(ctx, df_d, df_e, reason=failure_reason), ctx.metrics_dir / "panel_d_route_peak_perturbation_audit.csv")
    _save_csv(ctx, _route_peak_downstream_summary(ctx, df_e), ctx.metrics_dir / "panel_e_route_peak_downstream_summary.csv")
    _save_csv(ctx, _route_peak_downstream_contrast(ctx, df_e), ctx.metrics_dir / "panel_e_route_peak_downstream_contrast.csv")
    _save_csv(ctx, pd.DataFrame(output_distribution_rows, columns=PANEL_E_ROUTE_PEAK_OUTPUT_DISTRIBUTION_COLUMNS), ctx.metrics_dir / "panel_e_route_peak_output_distribution.csv")
    _save_csv(ctx, _route_peak_scientific_use_audit(ctx, df_d, df_e, reason=failure_reason), ctx.metrics_dir / "panel_de_route_peak_perturbation_scientific_use_audit.csv")
    _save_csv(ctx, pd.DataFrame(unit_rows, columns=ROUTE_PEAK_UNIT_SET_COLUMNS), ctx.trial_specs_dir / "peak_perturbation_unit_sets.csv")
    ctx.completed_modules["peak_perturbation"] = bool(success)
    if not success:
        raise RuntimeError(f"Fig.6 D/E route-peak perturbation produced invalid outputs: {failure_reason}")

def _run_real_probe_with_route_peak_reset(
    ctx: ExperimentContext,
    probe_image_id: int,
    boundary: Mapping[str, Mapping[str, Any]],
    selected_units: np.ndarray,
    *,
    probe_spikes: Any,
) -> tuple[np.ndarray, int, int, np.ndarray, dict[str, Any]]:
    if ctx.net is None or torch is None:
        raise RuntimeError("route-peak perturbation requires a real network")
    _, _, channels, height, width = probe_spikes.shape
    prepare_network_state(ctx.net, 1, int(channels), int(height), int(width))
    _restore_boundary_state(ctx.net, boundary)
    reset_record = _reset_layer1_stsp_units_to_s0(ctx.net, selected_units)
    trace, pred, fire, vector = _run_real_probe_from_condition(ctx, probe_image_id, snapshot_boundary_state(ctx.net), "S_final_route_peak_reset", probe_spikes=probe_spikes)
    return trace, pred, fire, vector, reset_record

def _reset_layer1_stsp_units_to_s0(net: Any, selected_units: np.ndarray) -> dict[str, Any]:
    if torch is None:
        return {"restore_ok": False, "reset_variables": ""}
    unit_ids = np.asarray(selected_units, dtype=np.int64).reshape(-1)
    layer = net.layer1
    variables: list[str] = []
    with torch.no_grad():
        if unit_ids.size == 0:
            return {"restore_ok": False, "reset_variables": ""}
        if getattr(layer, "u_pre", None) is None or getattr(layer, "x_pre", None) is None:
            return {"restore_ok": False, "reset_variables": ""}
        h = int(layer.u_pre.shape[-2])
        w = int(layer.u_pre.shape[-1])
        rr = torch.as_tensor(unit_ids // w, device=layer.u_pre.device, dtype=torch.long)
        cc = torch.as_tensor(unit_ids % w, device=layer.u_pre.device, dtype=torch.long)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr = rr[valid]
        cc = cc[valid]
        if int(rr.numel()) == 0:
            return {"restore_ok": False, "reset_variables": ""}
        layer.u_pre[..., rr, cc] = float(layer.stsp_U)
        layer.x_pre[..., rr, cc] = 1.0
        variables.extend(["u", "x"])
        if getattr(layer, "g_e", None) is not None and layer.g_e.ndim >= 4 and int(layer.g_e.shape[-2]) == h and int(layer.g_e.shape[-1]) == w:
            layer.g_e[..., rr, cc] = 0.0
            variables.append("g_e")
    return {"restore_ok": True, "reset_variables": ",".join(variables), "n_reset_units": int(unit_ids.size)}

def _matched_random_unit_mask(pool: np.ndarray, n_units: int, seed: int) -> np.ndarray:
    arr = np.asarray(pool, dtype=bool).reshape(-1)
    out = np.zeros(arr.size, dtype=bool)
    candidates = np.flatnonzero(arr)
    if int(n_units) <= 0 or candidates.size == 0:
        return out
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(candidates, size=min(int(n_units), int(candidates.size)), replace=False)
    out[chosen] = True
    return out

def _route_peak_reentry_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit_set in PERTURBATION_UNIT_SET_ORDER:
        part = df[df.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str).eq(unit_set)] if not df.empty else df
        valid = part[_bool_col(part, "perturbation_ok") & ~_bool_col(part, "insufficient_units")] if not part.empty else part
        loss = pd.to_numeric(valid.get("reentry_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        norm = pd.to_numeric(valid.get("normalized_reentry_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "perturbation_unit_set": unit_set,
                "condition_label": PERTURBATION_UNIT_SET_LABELS.get(unit_set, unit_set),
                "mean_reentry_loss": float(np.mean(loss)) if loss.size else np.nan,
                "sem_reentry_loss": _sem(loss) if loss.size else np.nan,
                "mean_normalized_reentry_loss": float(np.mean(norm)) if norm.size else np.nan,
                "sem_normalized_reentry_loss": _sem(norm) if norm.size else np.nan,
                "n_trials": int(len(part)),
                "n_valid_trials": int(len(valid)),
                "n_skipped_missing_boundary": int(part.get("failure_reason", pd.Series(dtype=str)).astype(str).str.contains("missing_sfinal_boundary", na=False).sum()) if len(part) else 0,
                "n_skipped_insufficient_units": int(_bool_col(part, "insufficient_units").sum()) if len(part) else 0,
                "n_perturbation_failed": int(part.get("failure_reason", pd.Series(dtype=str)).astype(str).str.contains("perturbation_failed", na=False).sum()) if len(part) else 0,
                "insufficient_fraction": float(_bool_col(part, "insufficient_units").mean()) if len(part) else np.nan,
                "denominator_choice": "max(abs(reentry_strength_intact), eps)",
            }
        )
    return pd.DataFrame(rows, columns=PANEL_D_ROUTE_PEAK_SUMMARY_COLUMNS)

def _route_peak_reentry_contrast(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric = "normalized_reentry_loss"
    for control in ("route_nonpeak", "nonroute_peak", "random_matched"):
        diff, n_pairs = _paired_unit_set_difference(df, "route_peak", control, metric)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "contrast": f"route_peak_minus_{control}",
                "metric": metric,
                "route_peak_minus_control": diff,
                "route_peak_minus_route_nonpeak": diff if control == "route_nonpeak" else np.nan,
                "route_peak_minus_nonroute_peak": diff if control == "nonroute_peak" else np.nan,
                "route_peak_minus_random": diff if control == "random_matched" else np.nan,
                "route_peak_effect_size": diff,
                "n_valid_pairs": int(n_pairs),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_D_ROUTE_PEAK_CONTRAST_COLUMNS)

def _route_peak_downstream_summary(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit_set in PERTURBATION_UNIT_SET_ORDER:
        part = df[df.get("perturbation_unit_set", pd.Series(dtype=str)).astype(str).eq(unit_set)] if not df.empty else df
        valid = part[_bool_col(part, "perturbation_ok") & ~_bool_col(part, "insufficient_units")] if not part.empty else part
        switch = _bool_col(valid, "output_switch")
        switch_vals = switch.astype(float).to_numpy(dtype=float) if len(valid) else np.asarray([], dtype=float)
        resp = pd.to_numeric(valid.get("response_displacement_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        dec = pd.to_numeric(valid.get("decision_deflection_loss", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "perturbation_unit_set": unit_set,
                "condition_label": PERTURBATION_UNIT_SET_LABELS.get(unit_set, unit_set),
                "P_output_switch": float(switch.mean()) if len(valid) else np.nan,
                "sem_output_switch": _sem(switch_vals) if switch_vals.size else np.nan,
                "mean_response_displacement_loss": float(np.mean(resp)) if resp.size else np.nan,
                "sem_response_displacement_loss": _sem(resp) if resp.size else np.nan,
                "mean_decision_deflection_loss": float(np.mean(dec)) if dec.size else np.nan,
                "sem_decision_deflection_loss": _sem(dec) if dec.size else np.nan,
                "n_trials": int(len(part)),
                "n_valid_trials": int(len(valid)),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_E_ROUTE_PEAK_SUMMARY_COLUMNS)

def _route_peak_downstream_contrast(ctx: ExperimentContext, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("output_switch", "response_displacement_loss", "decision_deflection_loss", "output_distribution_JS"):
        for control in ("route_nonpeak", "nonroute_peak", "random_matched"):
            diff, n_pairs = _paired_unit_set_difference(df, "route_peak", control, metric)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "metric": metric,
                    "contrast": f"route_peak_minus_{control}",
                    "route_peak_minus_route_nonpeak": diff if control == "route_nonpeak" else np.nan,
                    "route_peak_minus_nonroute_peak": diff if control == "nonroute_peak" else np.nan,
                    "route_peak_minus_random": diff if control == "random_matched" else np.nan,
                    "n_valid_pairs": int(n_pairs),
                }
            )
    return pd.DataFrame(rows, columns=PANEL_E_ROUTE_PEAK_CONTRAST_COLUMNS)

def _paired_unit_set_difference(df: pd.DataFrame, left: str, right: str, metric: str) -> tuple[float, int]:
    if df.empty or metric not in df.columns:
        return np.nan, 0
    use = df[_bool_col(df, "perturbation_ok") & ~_bool_col(df, "insufficient_units")].copy()
    if use.empty:
        return np.nan, 0
    piv = use.pivot_table(index=["sequence_id", "probe_id"], columns="perturbation_unit_set", values=metric, aggfunc="mean")
    if left not in piv.columns or right not in piv.columns:
        return np.nan, 0
    diff = pd.to_numeric(piv[left], errors="coerce") - pd.to_numeric(piv[right], errors="coerce")
    diff = diff.dropna()
    return (float(diff.mean()) if len(diff) else np.nan, int(len(diff)))

def _route_peak_perturbation_audit(ctx: ExperimentContext, df_d: pd.DataFrame, df_e: pd.DataFrame, *, reason: str) -> pd.DataFrame:
    success = _route_peak_success(df_d, df_e)
    valid = df_d[_bool_col(df_d, "perturbation_ok") & ~_bool_col(df_d, "insufficient_units")] if not df_d.empty else df_d
    missing_sets = _missing_route_peak_unit_sets(df_d, df_e)
    failure_reasons = _common_failure_reasons(df_d, df_e)
    return pd.DataFrame(
        [
            {
                "network_seed": int(ctx.cfg.network_seed),
                "route_peak_perturbation_implemented": True,
                "route_peak_perturbation_success": bool(success),
                "uses_real_rollout": bool(not _is_proxy_mode(ctx)),
                "proxy_mode": bool(_is_proxy_mode(ctx)),
                "probe_input_unchanged": True,
                "s0_baseline_available": bool(not df_d.empty and pd.to_numeric(df_d.get("reentry_strength_s0", pd.Series(dtype=float)), errors="coerce").notna().any()),
                "intact_sfinal_available": bool(not df_d.empty and pd.to_numeric(df_d.get("reentry_strength_intact", pd.Series(dtype=float)), errors="coerce").notna().any()),
                "route_peak_control_available": bool(_unit_set_valid(df_d, "route_peak")),
                "route_nonpeak_control_available": bool(_unit_set_valid(df_d, "route_nonpeak")),
                "nonroute_peak_control_available": bool(_unit_set_valid(df_d, "nonroute_peak")),
                "random_control_available": bool(_unit_set_valid(df_d, "random_matched")),
                "final_scientific_use": bool(success),
                "allowed_claim_strength": "causal_route_peak_gain" if success else "predictive_peak_amplified_only",
                "n_valid_trials": int(valid[["sequence_id", "probe_id"]].drop_duplicates().shape[0]) if not valid.empty else 0,
                "n_total_probe_trials": int(df_d[["sequence_id", "probe_id"]].drop_duplicates().shape[0]) if not df_d.empty else 0,
                "n_valid_trial_rows": int(len(valid)),
                "missing_unit_sets": ";".join(missing_sets),
                "all_unit_sets_valid": bool(not missing_sets),
                "missing_boundary_count": int(_failure_count(df_d, "missing_sfinal_boundary") + _failure_count(df_e, "missing_sfinal_boundary")),
                "insufficient_route_peak_count": int(_insufficient_count(df_d, "route_peak") + _insufficient_count(df_e, "route_peak")),
                "perturbation_failed_count": int(_failure_count(df_d, "perturbation_failed") + _failure_count(df_e, "perturbation_failed")),
                "common_failure_reasons": ";".join(failure_reasons),
                "failure_reason": reason if not success else "",
            }
        ]
    )

def _route_peak_scientific_use_audit(ctx: ExperimentContext, df_d: pd.DataFrame, df_e: pd.DataFrame, *, reason: str) -> pd.DataFrame:
    return _route_peak_perturbation_audit(ctx, df_d, df_e, reason=reason)

def _route_peak_success(df_d: pd.DataFrame, df_e: pd.DataFrame) -> bool:
    if df_d.empty or df_e.empty:
        return False
    return all(_unit_set_valid(df_d, unit_set) for unit_set in PERTURBATION_UNIT_SET_ORDER) and all(_unit_set_valid(df_e, unit_set) for unit_set in PERTURBATION_UNIT_SET_ORDER)

def _route_peak_failure_reason(df_d: pd.DataFrame, df_e: pd.DataFrame, *, n_total_probe_trials: int) -> str:
    missing_sets = _missing_route_peak_unit_sets(df_d, df_e)
    valid_pairs = df_d[_bool_col(df_d, "perturbation_ok") & ~_bool_col(df_d, "insufficient_units")][["sequence_id", "probe_id"]].drop_duplicates()
    failure_reasons = _common_failure_reasons(df_d, df_e)
    return (
        f"missing_unit_sets={','.join(missing_sets) or 'none'}; "
        f"n_total_probe_trials={int(n_total_probe_trials)}; "
        f"n_valid_trials={int(len(valid_pairs))}; "
        f"missing_boundary_count={int(_failure_count(df_d, 'missing_sfinal_boundary') + _failure_count(df_e, 'missing_sfinal_boundary'))}; "
        f"insufficient_route_peak_count={int(_insufficient_count(df_d, 'route_peak') + _insufficient_count(df_e, 'route_peak'))}; "
        f"perturbation_failed_count={int(_failure_count(df_d, 'perturbation_failed') + _failure_count(df_e, 'perturbation_failed'))}; "
        f"common_failure_reasons={','.join(failure_reasons) or 'none'}"
    )

def _missing_route_peak_unit_sets(df_d: pd.DataFrame, df_e: pd.DataFrame) -> list[str]:
    missing = []
    for unit_set in PERTURBATION_UNIT_SET_ORDER:
        if not (_unit_set_valid(df_d, unit_set) and _unit_set_valid(df_e, unit_set)):
            missing.append(unit_set)
    return missing

def _common_failure_reasons(df_d: pd.DataFrame, df_e: pd.DataFrame) -> list[str]:
    values: list[str] = []
    for df in (df_d, df_e):
        if df.empty or "failure_reason" not in df.columns:
            continue
        values.extend(str(v) for v in df["failure_reason"].dropna().tolist() if str(v).strip())
    counts = pd.Series(values, dtype=str).value_counts() if values else pd.Series(dtype=int)
    return [f"{reason}:{int(count)}" for reason, count in counts.head(8).items()]

def _failure_count(df: pd.DataFrame, token: str) -> int:
    if df.empty or "failure_reason" not in df.columns:
        return 0
    return int(df["failure_reason"].astype(str).str.contains(token, na=False).sum())

def _insufficient_count(df: pd.DataFrame, unit_set: str) -> int:
    if df.empty or "perturbation_unit_set" not in df.columns:
        return 0
    part = df[df["perturbation_unit_set"].astype(str).eq(unit_set)]
    return int(_bool_col(part, "insufficient_units").sum()) if not part.empty else 0

def _unit_set_valid(df: pd.DataFrame, unit_set: str) -> bool:
    if df.empty or "perturbation_unit_set" not in df.columns:
        return False
    part = df[df["perturbation_unit_set"].astype(str).eq(unit_set)]
    return bool(len(part) > 0 and (_bool_col(part, "perturbation_ok") & ~_bool_col(part, "insufficient_units")).any())

def _output_distribution_row(ctx: ExperimentContext, r: Any, unit_set: str, intact_vec: np.ndarray, condition_vec: np.ndarray, *, js: float) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "sequence_id": int(r.sequence_id),
        "probe_id": int(r.probe_id),
        "perturbation_unit_set": unit_set,
        "output_distribution_JS": float(js),
        "intact_entropy": _entropy_from_logits(intact_vec),
        "condition_entropy": _entropy_from_logits(condition_vec),
    }

def _softmax_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=float)
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    return exp / max(float(np.sum(exp)), 1e-12)

def _entropy_from_logits(values: np.ndarray) -> float:
    p = _softmax_np(values)
    if p.size == 0:
        return np.nan
    return float(-np.sum(p * np.log2(np.maximum(p, 1e-12))))

def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = _softmax_np(left)
    q = _softmax_np(right)
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return np.nan
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12))))
    kl_qm = float(np.sum(q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12))))
    return float(0.5 * (kl_pm + kl_qm))
