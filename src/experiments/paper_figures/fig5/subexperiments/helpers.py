from __future__ import annotations

from src.experiments.paper_figures import fig5_local_support_competition_experiment as _legacy
from src.experiments.common.input_masks import entry_mask_from_image, overlap_mask

# Keep module-level names identical while Fig.5 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def _entry_mask_cache(ctx: ExperimentContext) -> dict[tuple[Any, ...], np.ndarray]:
    cache = getattr(ctx, "_entry_mask_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_entry_mask_cache", cache)
    return cache

def _entry_masks_for_trial(ctx: ExperimentContext, sample_image_id: int, probe_image_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = _entry_mask_cache(ctx)
    sample_mask = entry_mask_from_image(
        ctx.dataset[int(sample_image_id)][0],
        mode=str(ctx.cfg.overlap_mask_mode),
        encoder=ctx.encoder,
        steps=int(ctx.cfg.sample_steps),
        device=ctx.device,
        foreground_threshold=float(ctx.cfg.foreground_threshold),
        cache=cache,
        image_id=int(sample_image_id),
    )
    probe_mask = entry_mask_from_image(
        ctx.dataset[int(probe_image_id)][0],
        mode=str(ctx.cfg.overlap_mask_mode),
        encoder=ctx.encoder,
        steps=int(ctx.cfg.probe_steps),
        device=ctx.device,
        foreground_threshold=float(ctx.cfg.foreground_threshold),
        cache=cache,
        image_id=int(probe_image_id),
    )
    overlap = overlap_mask(sample_mask, probe_mask)
    probe_only = probe_mask & (~sample_mask)
    return sample_mask, probe_mask, overlap, probe_only

def _copy_csv_alias(ctx: ExperimentContext, src: Path, dst: Path, *, empty_columns: Sequence[str], reason: str) -> None:
    copy_csv_alias(ctx, src, dst, empty_columns=empty_columns, reason=reason, message_label="Fig.5 supplement alias")

def _write_empty_csv(ctx: ExperimentContext, dst: Path, columns: Sequence[str], reason: str) -> None:
    write_empty_csv_with_warning(ctx, dst, columns, reason, message_label="Fig.5 supplement alias")

def _record_optional_missing(ctx: ExperimentContext, output_name: str, reason: str) -> None:
    record_optional_missing(ctx, output_name, reason, message_label="Fig.5 supplement alias")

def _mean_existing(df: pd.DataFrame, columns: Sequence[str]) -> float:
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else float("nan")
    return float("nan")

def _run_batch_network_checked(ctx: ExperimentContext, batch: pd.DataFrame) -> dict[str, Any]:
    if ctx.net is None or ctx.encoder is None:
        raise RuntimeError("Fig.5 requires a loaded real network and encoder before rollout.")
    try:
        return _run_batch_network(ctx, batch)
    except Exception as exc:
        raise RuntimeError("Fig.5 network rollout failed; no fallback traces are permitted.") from exc

def _run_batch_network(ctx: ExperimentContext, batch: pd.DataFrame) -> dict[str, Any]:
    assert ctx.net is not None and ctx.encoder is not None
    sample_images = _images_for_ids(ctx.dataset, batch["sample_image_id"].to_numpy()).to(ctx.device)
    probe_images = _images_for_ids(ctx.dataset, batch["probe_image_id"].to_numpy()).to(ctx.device)
    sample_spikes = encode_images(ctx.encoder, sample_images, ctx.cfg.sample_steps)
    probe_spikes = encode_images(ctx.encoder, probe_images, ctx.cfg.probe_steps)
    batch_size, _, channels, height, width = sample_spikes.shape
    prepare_network_state(ctx.net, batch_size, channels, height, width)
    current_time = 0
    with torch.no_grad():
        for t in range(ctx.cfg.sample_steps):
            current_time = _step_network_once(ctx.net, sample_spikes[:, t], current_time, stsp_mode="dynamic")
        zero = torch.zeros((batch_size, channels, height, width), device=ctx.device)
        for _ in range(ctx.cfg.delay_steps):
            current_time = _step_network_once(ctx.net, zero, current_time, stsp_mode="dynamic")
    boundary = snapshot_boundary_state(ctx.net)
    support_by_batch = _support_maps_from_boundary(boundary, batch_size)
    baseline_traces_by_local: dict[int, dict[str, BranchTrace]] = {}
    if ctx.cfg.enable_branch_batch:
        baseline_traces_by_local = _run_unperturbed_probe_branches_batch(
            ctx,
            boundary,
            probe_spikes,
            ("dynamic_intact", "static_frozen"),
        )
    l1_traces_by_local: dict[int, dict[str, BranchTrace]] = {}
    l1_audit_by_local: dict[int, list[dict[str, Any]]] = {}
    if ctx.cfg.enable_branch_batch:
        l1_traces_by_local, l1_audit_by_local = _run_l1_stsp_probe_branches_batch(
            ctx,
            boundary,
            probe_spikes,
            L1_STSP_PERTURBATION_CONDITIONS,
        )

    support_maps: dict[int, np.ndarray] = {}
    branch_traces: dict[int, dict[str, BranchTrace]] = {}
    boundary_states: dict[int, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    perturb_audit_rows: list[dict[str, Any]] = []
    l1_perturb_audit_rows: list[dict[str, Any]] = []
    trial_rows = list(batch.reset_index(drop=True).itertuples(index=False))
    psets_by_local: dict[int, pd.DataFrame] = {}
    for local_idx, trial in enumerate(trial_rows):
        trial_id = int(trial.trial_id)
        support_maps[trial_id] = support_by_batch[local_idx]
        single_boundary = _slice_boundary(boundary, local_idx)
        boundary_states[trial_id] = single_boundary
        single_probe = probe_spikes[local_idx : local_idx + 1]
        if baseline_traces_by_local:
            dynamic = baseline_traces_by_local[int(local_idx)]["dynamic_intact"]
            static = baseline_traces_by_local[int(local_idx)]["static_frozen"]
        else:
            dynamic, audit = _run_probe_branch(ctx, single_boundary, single_probe, "dynamic_intact")
            perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
            static, audit = _run_probe_branch(ctx, single_boundary, single_probe, "static_frozen")
            perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
        traces = {"dynamic_intact": dynamic, "static_frozen": static}
        if l1_traces_by_local:
            traces.update(l1_traces_by_local.get(int(local_idx), {}))
            l1_perturb_audit_rows.extend(
                dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id)
                for row in l1_audit_by_local.get(int(local_idx), [])
            )
        groups = _unit_group_rows(ctx, pd.Series(trial._asdict()), support_maps[trial_id])
        psets = _perturbation_unit_rows(ctx, pd.Series(trial._asdict()), support_maps[trial_id], groups)
        psets_by_local[int(local_idx)] = psets
        branch_traces[trial_id] = traces

    support_conditions = tuple(
        condition
        for condition in dict.fromkeys(LEGACY_REGION_PERTURBATION_CONDITIONS + SUPP_CONDITIONS)
        if condition != "dynamic_intact"
    )
    support_traces_by_local: dict[int, dict[str, BranchTrace]] = {}
    support_audit_by_local: dict[int, list[dict[str, Any]]] = {}
    if ctx.cfg.enable_branch_batch:
        support_traces_by_local, support_audit_by_local = _run_support_perturb_probe_branches_batch(
            ctx,
            boundary,
            probe_spikes,
            support_conditions,
            psets_by_local,
        )

    for local_idx, trial in enumerate(trial_rows):
        trial_id = int(trial.trial_id)
        single_boundary = boundary_states[trial_id]
        single_probe = probe_spikes[local_idx : local_idx + 1]
        psets = psets_by_local[int(local_idx)]
        traces = branch_traces[trial_id]
        if support_traces_by_local:
            traces.update(support_traces_by_local.get(int(local_idx), {}))
            perturb_audit_rows.extend(
                dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id)
                for row in support_audit_by_local.get(int(local_idx), [])
            )
        branch_conditions = list(dict.fromkeys(MAIN_CONDITIONS + LEGACY_REGION_PERTURBATION_CONDITIONS + SUPP_CONDITIONS))
        for condition in branch_conditions:
            if condition in traces:
                continue
            trace, audit = _run_probe_branch(ctx, single_boundary, single_probe, condition, perturb_units=psets[psets["condition"].eq(condition)])
            traces[condition] = trace
            if condition in L1_STSP_PERTURBATION_CONDITIONS:
                l1_perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
            else:
                perturb_audit_rows.extend(dict(row, network_seed=int(ctx.cfg.network_seed), trial_id=trial_id) for row in audit)
        branch_traces[trial_id] = traces
    return {
        "support_maps": support_maps,
        "branch_traces": branch_traces,
        "boundary_states": boundary_states,
        "perturbation_ux_audit": perturb_audit_rows,
        "l1_stsp_perturbation_audit": l1_perturb_audit_rows,
    }

def _run_probe_branch(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    condition: str,
    perturb_units: pd.DataFrame | None = None,
) -> tuple[BranchTrace, list[dict[str, Any]]]:
    assert ctx.net is not None
    batch_size, _, channels, height, width = probe_spikes.shape
    prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
    _restore_boundary_state(ctx.net, boundary)
    audit_rows: list[dict[str, Any]] = []
    if condition in L1_STSP_PERTURBATION_CONDITIONS:
        audit_rows = _apply_l1_stsp_perturbation(
            ctx.net,
            condition,
            attenuation_factor=float(ctx.cfg.perturbation_attenuation_factor),
        )
    elif condition not in {"dynamic_intact", "static_frozen"}:
        audit_rows = _apply_support_perturbation(
            ctx.net,
            condition,
            perturb_units,
            attenuation_factor=float(ctx.cfg.perturbation_attenuation_factor),
        )
    stsp_mode = "static_frozen" if condition == "static_frozen" else "dynamic"
    with torch.no_grad():
        ctx.net.layer3.reset_decision_state()
        ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
        ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
        layer1_spikes = []
        layer1_v = []
        layer1_inh = []
        layer3_spikes = []
        for t in range(int(probe_spikes.shape[1])):
            s1, m1 = ctx.net.layer1.forward_step(probe_spikes[:, t], t, training=False, monitor=True, stsp_mode=stsp_mode)
            s1p = ctx.net.pool1(s1.float())
            s2, _ = ctx.net.layer2.forward_step(s1p, t, training=False, monitor=False, stsp_mode=stsp_mode)
            s2p = ctx.net.pool2(s2.float())
            s3, _ = ctx.net.layer3.forward_step(s2p, t, training=False, monitor=False, stsp_mode=stsp_mode)
            layer1_spikes.append(s1.detach().clone())
            layer1_v.append(m1.get("v_effective", m1.get("v_mem_snapshot")).detach().clone())
            layer1_inh.append(m1.get("inh_after", torch.zeros_like(s1, dtype=torch.float32)).detach().clone())
            layer3_spikes.append(s3.detach().clone())
        pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, 1)
    spikes = torch.stack(layer1_spikes, dim=0).to(torch.bool).cpu().numpy()
    v = torch.stack(layer1_v, dim=0).to(torch.float32).cpu().numpy()
    inh = torch.stack(layer1_inh, dim=0).to(torch.float32).cpu().numpy()
    l3 = torch.stack(layer3_spikes, dim=0).to(torch.bool).cpu().numpy()
    trace = BranchTrace(
        spikes=spikes[:, 0].any(axis=1).astype(np.float32),
        v_effective=v[:, 0].mean(axis=1).astype(np.float32),
        inhibition=inh[:, 0].mean(axis=1).astype(np.float32),
        layer3_spikes=l3[:, 0].reshape(l3.shape[0], -1).astype(np.float32),
        prediction=int(pred[0].item()),
        first_fire_time=int(fire[0].item()),
    )
    return trace, audit_rows

def _run_unperturbed_probe_branches_batch(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    conditions: Sequence[str],
) -> dict[int, dict[str, BranchTrace]]:
    assert ctx.net is not None
    batch_size, _, channels, height, width = probe_spikes.shape
    out: dict[int, dict[str, BranchTrace]] = {idx: {} for idx in range(int(batch_size))}
    for condition in conditions:
        if condition not in {"dynamic_intact", "static_frozen"}:
            raise ValueError(f"Only unperturbed Fig.5 branches can be batched, got {condition!r}.")
        prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
        _restore_boundary_state(ctx.net, boundary)
        stsp_mode = "static_frozen" if condition == "static_frozen" else "dynamic"
        with torch.no_grad():
            ctx.net.layer3.reset_decision_state()
            ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
            ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
            layer1_spikes = []
            layer1_v = []
            layer1_inh = []
            layer3_spikes = []
            for t in range(int(probe_spikes.shape[1])):
                s1, m1 = ctx.net.layer1.forward_step(probe_spikes[:, t], t, training=False, monitor=True, stsp_mode=stsp_mode)
                s1p = ctx.net.pool1(s1.float())
                s2, _ = ctx.net.layer2.forward_step(s1p, t, training=False, monitor=False, stsp_mode=stsp_mode)
                s2p = ctx.net.pool2(s2.float())
                s3, _ = ctx.net.layer3.forward_step(s2p, t, training=False, monitor=False, stsp_mode=stsp_mode)
                layer1_spikes.append(s1.detach().clone())
                layer1_v.append(m1.get("v_effective", m1.get("v_mem_snapshot")).detach().clone())
                layer1_inh.append(m1.get("inh_after", torch.zeros_like(s1, dtype=torch.float32)).detach().clone())
                layer3_spikes.append(s3.detach().clone())
            pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, int(batch_size))
        spikes = torch.stack(layer1_spikes, dim=0).to(torch.bool).cpu().numpy()
        v = torch.stack(layer1_v, dim=0).to(torch.float32).cpu().numpy()
        inh = torch.stack(layer1_inh, dim=0).to(torch.float32).cpu().numpy()
        l3 = torch.stack(layer3_spikes, dim=0).to(torch.bool).cpu().numpy()
        for local_idx in range(int(batch_size)):
            out[local_idx][condition] = BranchTrace(
                spikes=spikes[:, local_idx].any(axis=1).astype(np.float32),
                v_effective=v[:, local_idx].mean(axis=1).astype(np.float32),
                inhibition=inh[:, local_idx].mean(axis=1).astype(np.float32),
                layer3_spikes=l3[:, local_idx].reshape(l3.shape[0], -1).astype(np.float32),
                prediction=int(pred[local_idx].item()),
                first_fire_time=int(fire[local_idx].item()),
            )
    return out

def _run_l1_stsp_probe_branches_batch(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    conditions: Sequence[str],
) -> tuple[dict[int, dict[str, BranchTrace]], dict[int, list[dict[str, Any]]]]:
    assert ctx.net is not None
    batch_size, _, channels, height, width = probe_spikes.shape
    out: dict[int, dict[str, BranchTrace]] = {idx: {} for idx in range(int(batch_size))}
    audits: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(int(batch_size))}
    for condition in conditions:
        if condition not in L1_STSP_PERTURBATION_CONDITIONS:
            raise ValueError(f"Only Fig.5 L1 STSP perturbation branches can be batched, got {condition!r}.")
        prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
        _restore_boundary_state(ctx.net, boundary)
        audit_rows = _apply_l1_stsp_perturbation_batch(
            ctx,
            condition,
            batch_size=int(batch_size),
            attenuation_factor=float(ctx.cfg.perturbation_attenuation_factor),
        )
        with torch.no_grad():
            ctx.net.layer3.reset_decision_state()
            ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
            ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
            layer1_spikes = []
            layer1_v = []
            layer1_inh = []
            layer3_spikes = []
            for t in range(int(probe_spikes.shape[1])):
                s1, m1 = ctx.net.layer1.forward_step(probe_spikes[:, t], t, training=False, monitor=True, stsp_mode="dynamic")
                s1p = ctx.net.pool1(s1.float())
                s2, _ = ctx.net.layer2.forward_step(s1p, t, training=False, monitor=False, stsp_mode="dynamic")
                s2p = ctx.net.pool2(s2.float())
                s3, _ = ctx.net.layer3.forward_step(s2p, t, training=False, monitor=False, stsp_mode="dynamic")
                layer1_spikes.append(s1.detach().clone())
                layer1_v.append(m1.get("v_effective", m1.get("v_mem_snapshot")).detach().clone())
                layer1_inh.append(m1.get("inh_after", torch.zeros_like(s1, dtype=torch.float32)).detach().clone())
                layer3_spikes.append(s3.detach().clone())
            pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, int(batch_size))
        spikes = torch.stack(layer1_spikes, dim=0).to(torch.bool).cpu().numpy()
        v = torch.stack(layer1_v, dim=0).to(torch.float32).cpu().numpy()
        inh = torch.stack(layer1_inh, dim=0).to(torch.float32).cpu().numpy()
        l3 = torch.stack(layer3_spikes, dim=0).to(torch.bool).cpu().numpy()
        for local_idx in range(int(batch_size)):
            out[local_idx][condition] = BranchTrace(
                spikes=spikes[:, local_idx].any(axis=1).astype(np.float32),
                v_effective=v[:, local_idx].mean(axis=1).astype(np.float32),
                inhibition=inh[:, local_idx].mean(axis=1).astype(np.float32),
                layer3_spikes=l3[:, local_idx].reshape(l3.shape[0], -1).astype(np.float32),
                prediction=int(pred[local_idx].item()),
                first_fire_time=int(fire[local_idx].item()),
            )
            audits[local_idx].append(audit_rows[local_idx])
    return out, audits

def _run_support_perturb_probe_branches_batch(
    ctx: ExperimentContext,
    boundary: Mapping[str, Mapping[str, torch.Tensor]],
    probe_spikes: torch.Tensor,
    conditions: Sequence[str],
    perturbation_sets_by_local: Mapping[int, pd.DataFrame],
) -> tuple[dict[int, dict[str, BranchTrace]], dict[int, list[dict[str, Any]]]]:
    assert ctx.net is not None
    batch_size, _, channels, height, width = probe_spikes.shape
    out: dict[int, dict[str, BranchTrace]] = {idx: {} for idx in range(int(batch_size))}
    audits: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(int(batch_size))}
    for condition in conditions:
        if condition in {"dynamic_intact", "static_frozen"} or condition in L1_STSP_PERTURBATION_CONDITIONS:
            raise ValueError(f"Fig.5 support perturbation batch got incompatible condition {condition!r}.")
        prepare_network_state(ctx.net, int(batch_size), int(channels), int(height), int(width))
        _restore_boundary_state(ctx.net, boundary)
        audit_rows_by_local = _apply_support_perturbation_batch(
            ctx,
            condition,
            perturbation_sets_by_local,
            batch_size=int(batch_size),
            attenuation_factor=float(ctx.cfg.perturbation_attenuation_factor),
        )
        with torch.no_grad():
            ctx.net.layer3.reset_decision_state()
            ctx.net.layer3.v_mem.fill_(ctx.net.layer3.V_L)
            ctx.net.layer3.lateral_inh.reset_state(ctx.net.layer3.output_shape)
            layer1_spikes = []
            layer1_v = []
            layer1_inh = []
            layer3_spikes = []
            for t in range(int(probe_spikes.shape[1])):
                s1, m1 = ctx.net.layer1.forward_step(probe_spikes[:, t], t, training=False, monitor=True, stsp_mode="dynamic")
                s1p = ctx.net.pool1(s1.float())
                s2, _ = ctx.net.layer2.forward_step(s1p, t, training=False, monitor=False, stsp_mode="dynamic")
                s2p = ctx.net.pool2(s2.float())
                s3, _ = ctx.net.layer3.forward_step(s2p, t, training=False, monitor=False, stsp_mode="dynamic")
                layer1_spikes.append(s1.detach().clone())
                layer1_v.append(m1.get("v_effective", m1.get("v_mem_snapshot")).detach().clone())
                layer1_inh.append(m1.get("inh_after", torch.zeros_like(s1, dtype=torch.float32)).detach().clone())
                layer3_spikes.append(s3.detach().clone())
            pred, fire = decode_prediction_and_fire_time_from_layer3(ctx.net, int(batch_size))
        spikes = torch.stack(layer1_spikes, dim=0).to(torch.bool).cpu().numpy()
        v = torch.stack(layer1_v, dim=0).to(torch.float32).cpu().numpy()
        inh = torch.stack(layer1_inh, dim=0).to(torch.float32).cpu().numpy()
        l3 = torch.stack(layer3_spikes, dim=0).to(torch.bool).cpu().numpy()
        for local_idx in range(int(batch_size)):
            out[local_idx][condition] = BranchTrace(
                spikes=spikes[:, local_idx].any(axis=1).astype(np.float32),
                v_effective=v[:, local_idx].mean(axis=1).astype(np.float32),
                inhibition=inh[:, local_idx].mean(axis=1).astype(np.float32),
                layer3_spikes=l3[:, local_idx].reshape(l3.shape[0], -1).astype(np.float32),
                prediction=int(pred[local_idx].item()),
                first_fire_time=int(fire[local_idx].item()),
            )
            audits[local_idx].extend(audit_rows_by_local.get(local_idx, []))
    return out, audits

def _run_probe_branches_batch(
    ctx: ExperimentContext,
    boundary_states: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    probe_spikes: Mapping[int, torch.Tensor],
    conditions: Sequence[str],
    perturbation_sets: Mapping[tuple[int, str], pd.DataFrame],
) -> dict[tuple[int, str], BranchTrace]:
    if ctx.cfg.enable_branch_batch:
        ctx.warnings.append("Fig.5 branch batch helper is scaffolded; falling back to order-preserving single-branch rollouts.")
    out: dict[tuple[int, str], BranchTrace] = {}
    for trial_id, boundary in boundary_states.items():
        for condition in conditions:
            trace, _ = _run_probe_branch(
                ctx,
                boundary,
                probe_spikes[int(trial_id)],
                str(condition),
                perturb_units=perturbation_sets.get((int(trial_id), str(condition))),
            )
            out[(int(trial_id), str(condition))] = trace
    return out

def _unit_group_rows(ctx: ExperimentContext, trial: Any, support: np.ndarray) -> pd.DataFrame:
    trial_map = _trial_mapping(trial)
    sample_mask, probe_mask, overlap, probe_only = _entry_masks_for_trial(
        ctx,
        int(trial_map["sample_image_id"]),
        int(trial_map["probe_image_id"]),
    )
    h, w = support.shape
    overlap = _resize_mask(overlap, h, w)
    probe_only = _resize_mask(probe_only, h, w)
    rng = np.random.default_rng(int(trial_map["trial_seed"]) + 17)
    high_support = support >= np.nanquantile(support, max(0.0, 1.0 - float(ctx.cfg.peak_support_q)))
    random_pool = np.flatnonzero(high_support.reshape(-1))
    random_take = set(rng.choice(random_pool, size=min(int(overlap.sum()), len(random_pool)), replace=False).tolist()) if len(random_pool) else set()
    rows = []
    for r in range(h):
        for c in range(w):
            unit_id = int(r * w + c)
            if bool(overlap[r, c]) and support[r, c] >= float(ctx.cfg.drive_score_threshold):
                group = "overlap_dominant"
            elif bool(probe_only[r, c]):
                group = "probe_only_dominant"
            elif unit_id in random_take:
                group = "random_matched"
            else:
                group = "balanced"
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_map["trial_id"]),
                    "layer": PRIMARY_LAYER,
                    "unit_id": unit_id,
                    "row": int(r),
                    "col": int(c),
                    "unit_group": group,
                    "overlap_drive_score": float(overlap[r, c]) * float(support[r, c]),
                    "probe_only_drive_score": float(probe_only[r, c]) * float(support[r, c]),
                    "support_value": float(support[r, c]),
                    "is_overlap_dominant": bool(group == "overlap_dominant"),
                    "is_probe_only_dominant": bool(group == "probe_only_dominant"),
                    "is_random_matched": bool(group == "random_matched"),
                }
            )
    return pd.DataFrame(rows, columns=UNIT_GROUP_COLUMNS)

def _perturbation_unit_rows(ctx: ExperimentContext, trial: Any, support: np.ndarray, groups: pd.DataFrame) -> pd.DataFrame:
    trial_map = _trial_mapping(trial)
    q = np.nanquantile(support, max(0.0, 1.0 - float(ctx.cfg.peak_support_q)))
    base = groups[pd.to_numeric(groups["support_value"], errors="coerce") >= q].copy()
    overlap = base[base["unit_group"].eq("overlap_dominant")].copy()
    condition_sets = {
        "attenuate_overlap_high_support": overlap,
        "reset_overlap_high_support": overlap,
        "sham_perturbation": overlap.head(0),
    }
    rows = []
    for condition, part in condition_sets.items():
        for row in part.itertuples(index=False):
            original = float(row.support_value)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_map["trial_id"]),
                    "condition": condition,
                    "unit_id": int(row.unit_id),
                    "unit_group": str(row.unit_group),
                    "original_support": original,
                    "perturbed_support": np.nan,
                    "support_delta": np.nan,
                    "row": int(row.row),
                    "col": int(row.col),
                    "matched_to_condition": "",
                    "matching_error_support": np.nan,
                    "matching_error_spike_count": np.nan,
                    "intervention_timing": "pre_probe_boundary",
                    "probe_input_changed": False,
                }
            )
    return pd.DataFrame(rows, columns=PERTURBATION_UNIT_COLUMNS)

def _node_metrics_for_condition(ctx: ExperimentContext, condition: str, trace: BranchTrace, dynamic: BranchTrace, static: BranchTrace, first: np.ndarray, dyn_first: np.ndarray, sta_first: np.ndarray, unit_set: pd.DataFrame) -> dict[str, Any]:
    transitions = [_transition_type(int(first[r, c]), int(sta_first[r, c])) for r in range(first.shape[0]) for c in range(first.shape[1])]
    n = max(1, len(transitions))
    early_delta = float(trace.spikes[: ctx.cfg.early_window_steps].sum() - static.spikes[: ctx.cfg.early_window_steps].sum())
    latency_vals = [_latency_delta(int(first[r, c]), int(sta_first[r, c])) for r in range(first.shape[0]) for c in range(first.shape[1]) if int(first[r, c]) >= 0 or int(sta_first[r, c]) >= 0]
    dyn_like = _pattern_similarity(trace.spikes, dynamic.spikes)
    sta_like = _pattern_similarity(trace.spikes, static.spikes)
    winner_boost = float(np.nanmean(trace.v_effective[: ctx.cfg.early_window_steps] - static.v_effective[: ctx.cfg.early_window_steps]))
    loser_inh = float(np.nanmean(trace.inhibition[ctx.cfg.early_window_steps :] - static.inhibition[ctx.cfg.early_window_steps :]))
    return {
        "P_advance": transitions.count("advance") / n,
        "P_recruit": transitions.count("recruit") / n,
        "P_advance_plus_recruit": (transitions.count("advance") + transitions.count("recruit")) / n,
        "delta_early_spike_count": early_delta,
        "delta_first_spike_latency": float(np.nanmean(latency_vals)) if latency_vals else float("nan"),
        "winner_pre_spike_delta_v_mean": winner_boost,
        "winner_pre_spike_boost": float(winner_boost > 0.0),
        "loser_post_winner_inh_rise": loser_inh,
        "loser_post_winner_delta_v_mean": float(np.nanmean(trace.v_effective[ctx.cfg.early_window_steps :] - dynamic.v_effective[ctx.cfg.early_window_steps :])),
        "loser_post_winner_suppressed": float(loser_inh > 0.0),
        "spike_pattern_displacement": float(1.0 - sta_like),
        "dynamic_like_spike_similarity": float(dyn_like),
        "decision_deflection_score": float(_decision_deflection(trace, dynamic, static)),
        "dynamic_like_readout_recovery": float(_pattern_similarity(trace.layer3_spikes, dynamic.layer3_spikes)),
    }

def _transition_summary(network_seed: int, metrics: pd.DataFrame, early_window_ms: int) -> pd.DataFrame:
    rows = []
    for (trial_id, group), part in metrics.groupby(["trial_id", "unit_group"], sort=False):
        transitions = part["transition_type"].astype(str)
        n = max(1, len(part))
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "unit_group": str(group),
                "early_window_ms": int(early_window_ms),
                "P_advance": float((transitions == "advance").mean()),
                "P_recruit": float((transitions == "recruit").mean()),
                "P_loss": float((transitions == "loss").mean()),
                "P_unchanged": float((transitions == "unchanged").mean()),
                "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                "mean_delta_early_spike_count": float(pd.to_numeric(part["delta_early_spike_count"], errors="coerce").mean()),
                "mean_delta_first_spike_latency": float(pd.to_numeric(part["delta_first_spike_latency"], errors="coerce").mean()),
                "n_units": int(n),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_B_SUMMARY_COLUMNS)

def _summarize_perturbation_transitions(ctx: ExperimentContext, unit_df: pd.DataFrame) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame(columns=PANEL_D_TRANSITION_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (network_seed, trial_id, condition, unit_group), part in unit_df.groupby(["network_seed", "trial_id", "condition", "unit_group"], sort=False):
        transitions = part["transition_vs_static"].astype(str)
        same_winner = part["same_winner"].astype(bool)
        n_same = int(same_winner.sum())
        denom_same = max(1, n_same)
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "condition": str(condition),
                "unit_group": str(unit_group),
                "P_advance": float((transitions == "advance").mean()),
                "P_recruit": float((transitions == "recruit").mean()),
                "P_loss": float((transitions == "loss").mean()),
                "P_unchanged": float((transitions == "unchanged").mean()),
                "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                "P_same_winner_preserved": float((part["same_winner_preserved"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_delayed": float((part["same_winner_delayed"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_lost": float((part["same_winner_lost"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_reverted_to_static": float((part["same_winner_reverted_to_static"].astype(bool) & same_winner).sum() / denom_same),
                "P_same_winner_lost_or_delayed": float((part["same_winner_lost_or_delayed"].astype(bool) & same_winner).sum() / denom_same),
                "mean_delta_latency_vs_static": float(pd.to_numeric(part["delta_latency_vs_static"], errors="coerce").mean()),
                "mean_delta_latency_vs_same": float(pd.to_numeric(part["delta_latency_vs_same"], errors="coerce").mean()),
                "mean_delta_early_spike_count_vs_static": float(pd.to_numeric(part["delta_early_spike_count_vs_static"], errors="coerce").mean()),
                "mean_delta_early_spike_count_vs_same": float(pd.to_numeric(part["delta_early_spike_count_vs_same"], errors="coerce").mean()),
                "n_units": int(len(part)),
                "n_same_winner_units": n_same,
            }
        )
    _ = ctx
    return pd.DataFrame(rows, columns=PANEL_D_TRANSITION_SUMMARY_COLUMNS)

def _compute_perturbation_transition_contrasts(ctx: ExperimentContext, summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=PANEL_D_TRANSITION_CONTRAST_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (network_seed, trial_id, unit_group), part in summary_df.groupby(["network_seed", "trial_id", "unit_group"], sort=False):
        by_cond = {str(row.condition): row for row in part.itertuples(index=False)}
        base = by_cond.get("dynamic_intact")
        attenuate = by_cond.get("attenuate_overlap_high_support")
        reset = by_cond.get("reset_overlap_high_support")
        if base is None:
            continue
        attenuate_delta_recruit = _delta_field(attenuate, base, "P_advance_plus_recruit")
        reset_delta_recruit = _delta_field(reset, base, "P_advance_plus_recruit")
        rows.append(
            {
                "network_seed": int(network_seed),
                "trial_id": int(trial_id),
                "unit_group": str(unit_group),
                "attenuate_delta_P_advance_plus_recruit": attenuate_delta_recruit,
                "reset_delta_P_advance_plus_recruit": reset_delta_recruit,
                "attenuate_delta_P_loss": _delta_field(attenuate, base, "P_loss"),
                "reset_delta_P_loss": _delta_field(reset, base, "P_loss"),
                "attenuate_delta_P_same_winner_lost_or_delayed": _delta_field(attenuate, base, "P_same_winner_lost_or_delayed"),
                "reset_delta_P_same_winner_lost_or_delayed": _delta_field(reset, base, "P_same_winner_lost_or_delayed"),
                "reset_minus_attenuate_delta_P_advance_plus_recruit": _finite_delta(reset_delta_recruit, attenuate_delta_recruit),
                "attenuate_delta_latency_vs_same": _delta_field(attenuate, base, "mean_delta_latency_vs_same"),
                "reset_delta_latency_vs_same": _delta_field(reset, base, "mean_delta_latency_vs_same"),
                "n_units": int(getattr(base, "n_units", 0)),
                "n_trials": 1,
            }
        )
    _ = ctx
    return pd.DataFrame(rows, columns=PANEL_D_TRANSITION_CONTRAST_COLUMNS)

def _summarize_l1_stsp_perturbation(ctx: ExperimentContext, unit_df: pd.DataFrame, included_groups: Sequence[str]) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame(columns=PANEL_D_L1_STSP_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    included = unit_df[unit_df["included_in_main"].astype(bool)].copy()
    for (network_seed, condition), part in included.groupby(["network_seed", "condition"], sort=False):
        transitions = part["transition_vs_static"].astype(str)
        transition_mass = float(((transitions == "advance") | (transitions == "recruit") | (transitions == "loss")).mean())
        rows.append(
            {
                "network_seed": int(network_seed),
                "condition": str(condition),
                "condition_label": _fig5d_condition_label(str(condition)),
                "P_advance": float((transitions == "advance").mean()),
                "P_recruit": float((transitions == "recruit").mean()),
                "P_loss": float((transitions == "loss").mean()),
                "P_unchanged": float((transitions == "unchanged").mean()),
                "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                "transition_mass": transition_mass,
                "n_units": int(len(part)),
                "n_trials": int(part["trial_id"].nunique()),
                "included_unit_groups": ";".join(included_groups),
                "perturbation_mode": _l1_stsp_perturbation_mode(str(condition)),
                "perturbed_layer": PRIMARY_LAYER if str(condition) in L1_STSP_PERTURBATION_CONDITIONS else "none",
                "perturbed_variables": "u_pre;x_pre" if str(condition) in L1_STSP_PERTURBATION_CONDITIONS else "none",
            }
        )
    _ = ctx
    return pd.DataFrame(rows, columns=PANEL_D_L1_STSP_SUMMARY_COLUMNS)

def _compute_l1_stsp_perturbation_contrast(ctx: ExperimentContext, summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=PANEL_D_L1_STSP_CONTRAST_COLUMNS)
    rows: list[dict[str, Any]] = []
    for network_seed, part in summary_df.groupby("network_seed", sort=False):
        by_cond = {str(row.condition): row for row in part.itertuples(index=False)}
        dynamic = by_cond.get("dynamic_intact")
        attenuate = by_cond.get("attenuate_l1_stsp")
        reset = by_cond.get("reset_l1_stsp")
        rows.append(
            {
                "network_seed": int(network_seed),
                "dynamic_transition_mass": _row_value(dynamic, "transition_mass"),
                "attenuate_transition_mass": _row_value(attenuate, "transition_mass"),
                "reset_transition_mass": _row_value(reset, "transition_mass"),
                "dynamic_minus_attenuate_transition_mass": _finite_delta(_row_value(dynamic, "transition_mass"), _row_value(attenuate, "transition_mass")),
                "dynamic_minus_reset_transition_mass": _finite_delta(_row_value(dynamic, "transition_mass"), _row_value(reset, "transition_mass")),
                "attenuate_minus_reset_transition_mass": _finite_delta(_row_value(attenuate, "transition_mass"), _row_value(reset, "transition_mass")),
                "dynamic_P_advance": _row_value(dynamic, "P_advance"),
                "attenuate_P_advance": _row_value(attenuate, "P_advance"),
                "reset_P_advance": _row_value(reset, "P_advance"),
                "dynamic_P_recruit": _row_value(dynamic, "P_recruit"),
                "attenuate_P_recruit": _row_value(attenuate, "P_recruit"),
                "reset_P_recruit": _row_value(reset, "P_recruit"),
                "dynamic_P_loss": _row_value(dynamic, "P_loss"),
                "attenuate_P_loss": _row_value(attenuate, "P_loss"),
                "reset_P_loss": _row_value(reset, "P_loss"),
            }
        )
    _ = ctx
    return pd.DataFrame(rows, columns=PANEL_D_L1_STSP_CONTRAST_COLUMNS)

def _delta_field(condition_row: Any | None, base_row: Any, field: str) -> float:
    if condition_row is None:
        return float("nan")
    condition_value = float(getattr(condition_row, field, np.nan))
    base_value = float(getattr(base_row, field, np.nan))
    return float(condition_value - base_value) if np.isfinite(condition_value) and np.isfinite(base_value) else float("nan")

def _event_trace_summary(ctx: ExperimentContext, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PANEL_C_TRACE_COLUMNS)
    out = []
    for (time_ms, trace_type), part in df.groupby(["time_ms", "trace_type"], sort=True):
        values = pd.to_numeric(part["value"], errors="coerce").dropna()
        out.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "time_ms": float(time_ms),
                "trace_type": str(trace_type),
                "mean_value": float(values.mean()) if not values.empty else np.nan,
                "sem_value": float(values.sem()) if len(values) > 1 else 0.0,
                "n_events": int(values.count()),
            }
        )
    return pd.DataFrame(out, columns=PANEL_C_TRACE_COLUMNS)

def _early_window_robustness(ctx: ExperimentContext, bank: LocalSupportCompetitionBank) -> pd.DataFrame:
    base = pd.read_csv(ctx.metrics_dir / "panel_b_early_firing_transition_metrics.csv")
    rows = []
    for window in (5, 10, 15, 20, 30):
        for group, part in base.groupby("unit_group", sort=False):
            transitions = part["transition_type"].astype(str)
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "early_window_ms": int(window),
                    "unit_group": str(group),
                    "P_advance": float((transitions == "advance").mean()),
                    "P_recruit": float((transitions == "recruit").mean()),
                    "P_loss": float((transitions == "loss").mean()),
                    "P_unchanged": float((transitions == "unchanged").mean()),
                    "P_advance_plus_recruit": float(((transitions == "advance") | (transitions == "recruit")).mean()),
                    "delta_early_spike_count": float(pd.to_numeric(part["delta_early_spike_count"], errors="coerce").mean()) * min(1.0, window / max(1.0, ctx.cfg.early_window_ms)),
                    "n_units": int(len(part)),
                }
            )
    return pd.DataFrame(rows)

def _neighborhood_radius_robustness(ctx: ExperimentContext, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in (1, 2, 3):
        part = events[pd.to_numeric(events.get("local_distance", pd.Series(dtype=float)), errors="coerce") <= radius * 2]
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "neighborhood_radius": int(radius),
                "n_events": int(len(part)),
                "winner_pre_spike_delta_v_mean": float(pd.to_numeric(part.get("winner_pre_spike_delta_v_mean", pd.Series(dtype=float)), errors="coerce").mean()) if not part.empty else np.nan,
                "loser_post_winner_inh_rise": float(pd.to_numeric(part.get("loser_post_winner_inh_rise", pd.Series(dtype=float)), errors="coerce").mean()) if not part.empty else np.nan,
                "loser_post_winner_suppressed": float(part.get("loser_post_winner_suppressed", pd.Series(dtype=bool)).astype(bool).mean()) if not part.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)

def _support_perturbation_controls(node_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in node_df.groupby("condition", sort=False):
        for metric in ["P_advance_plus_recruit", "winner_pre_spike_delta_v_mean", "loser_post_winner_inh_rise", "dynamic_like_spike_similarity", "decision_deflection_score"]:
            rows.append({"network_seed": int(part["network_seed"].iloc[0]), "condition": condition, "metric": metric, "value": float(pd.to_numeric(part[metric], errors="coerce").mean()), "n_trials": int(part["trial_id"].nunique())})
    return pd.DataFrame(rows)

def _perturbation_matching_diagnostics(ctx: ExperimentContext, bank: LocalSupportCompetitionBank, node_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for part in bank.perturbation_sets.groupby(["trial_id", "condition"], sort=False):
        (trial_id, condition), df = part
        rows.append(
            {
                "network_seed": int(ctx.cfg.network_seed),
                "trial_id": int(trial_id),
                "condition": str(condition),
                "n_perturbed_units": int(len(df)),
                "mean_pre_support": float(pd.to_numeric(df["original_support"], errors="coerce").mean()) if len(df) else np.nan,
                "mean_post_support": float(pd.to_numeric(df["perturbed_support"], errors="coerce").mean()) if len(df) else np.nan,
                "expected_spike_count": float(len(df)),
                "actual_spike_count": float(node_df[(node_df["trial_id"].eq(trial_id)) & (node_df["condition"].eq(condition))]["delta_early_spike_count"].mean()),
                "active_unit_count": int(len(df)),
                "matching_error_support": float(pd.to_numeric(df["matching_error_support"], errors="coerce").mean()) if len(df) else np.nan,
                "matching_error_spike_count": float(pd.to_numeric(df["matching_error_spike_count"], errors="coerce").mean()) if len(df) else np.nan,
            }
        )
    return pd.DataFrame(rows)

def _apply_l1_stsp_perturbation(
    net,
    mode: str,
    attenuation_factor: float,
) -> list[dict[str, Any]]:
    layer = getattr(net, PRIMARY_LAYER, None)
    if layer is None:
        raise RuntimeError("Fig.5D requires net.layer1 for Layer1 STSP perturbation.")
    u = getattr(layer, "u_pre", None)
    x = getattr(layer, "x_pre", None)
    if u is None or x is None:
        raise RuntimeError("Fig.5D requires net.layer1.u_pre and net.layer1.x_pre at the pre-probe boundary.")
    with torch.no_grad():
        u_before = u.detach().clone()
        x_before = x.detach().clone()
        u0 = _layer_stsp_baseline_u(layer, u_before)
        if mode == "attenuate_l1_stsp":
            u.copy_(u0 + float(attenuation_factor) * (u - u0))
            x.copy_(1.0 + float(attenuation_factor) * (x - 1.0))
        elif mode == "reset_l1_stsp":
            u.fill_(float(u0))
            x.fill_(1.0)
        else:
            raise RuntimeError(f"Unsupported Layer1 STSP perturbation condition: {mode}")
        u_after = u.detach().clone()
        x_after = x.detach().clone()
    return [{
        "condition": mode,
        "perturbation_mode": _l1_stsp_perturbation_mode(mode),
        "perturbed_layer": PRIMARY_LAYER,
        "perturbed_variables": "u_pre;x_pre",
        "n_l1_stsp_sites": int(u.numel()),
        "l1_u_before_mean": _tensor_mean(u_before),
        "l1_u_after_mean": _tensor_mean(u_after),
        "l1_u_delta_mean": _tensor_delta_mean(u_after, u_before),
        "l1_x_before_mean": _tensor_mean(x_before),
        "l1_x_after_mean": _tensor_mean(x_after),
        "l1_x_delta_mean": _tensor_delta_mean(x_after, x_before),
        "l1_u_before_std": _tensor_std(u_before),
        "l1_u_after_std": _tensor_std(u_after),
        "l1_x_before_std": _tensor_std(x_before),
        "l1_x_after_std": _tensor_std(x_after),
        "layer1_perturbed": True,
        "layer2_perturbed": False,
        "layer3_perturbed": False,
        "restore_ok": True,
        "perturbation_ok": True,
    }]

def _apply_l1_stsp_perturbation_batch(
    ctx: ExperimentContext,
    mode: str,
    *,
    batch_size: int,
    attenuation_factor: float,
) -> list[dict[str, Any]]:
    net = ctx.net
    layer = getattr(net, PRIMARY_LAYER, None)
    if layer is None:
        raise RuntimeError("Fig.5D requires net.layer1 for Layer1 STSP perturbation.")
    u = getattr(layer, "u_pre", None)
    x = getattr(layer, "x_pre", None)
    if u is None or x is None:
        raise RuntimeError("Fig.5D requires net.layer1.u_pre and net.layer1.x_pre at the pre-probe boundary.")
    with torch.no_grad():
        u_before = u.detach().clone()
        x_before = x.detach().clone()
        u0 = _layer_stsp_baseline_u(layer, u_before)
        if mode == "attenuate_l1_stsp":
            u.copy_(u0 + float(attenuation_factor) * (u - u0))
            x.copy_(1.0 + float(attenuation_factor) * (x - 1.0))
        elif mode == "reset_l1_stsp":
            u.fill_(float(u0))
            x.fill_(1.0)
        else:
            raise RuntimeError(f"Unsupported Layer1 STSP perturbation condition: {mode}")
        u_after = u.detach().clone()
        x_after = x.detach().clone()
    rows: list[dict[str, Any]] = []
    for local_idx in range(int(batch_size)):
        u_before_i = u_before[local_idx : local_idx + 1]
        x_before_i = x_before[local_idx : local_idx + 1]
        u_after_i = u_after[local_idx : local_idx + 1]
        x_after_i = x_after[local_idx : local_idx + 1]
        rows.append(
            {
                "condition": mode,
                "perturbation_mode": _l1_stsp_perturbation_mode(mode),
                "perturbed_layer": PRIMARY_LAYER,
                "perturbed_variables": "u_pre;x_pre",
                "n_l1_stsp_sites": int(u_before_i.numel()),
                "l1_u_before_mean": _tensor_mean(u_before_i),
                "l1_u_after_mean": _tensor_mean(u_after_i),
                "l1_u_delta_mean": _tensor_delta_mean(u_after_i, u_before_i),
                "l1_x_before_mean": _tensor_mean(x_before_i),
                "l1_x_after_mean": _tensor_mean(x_after_i),
                "l1_x_delta_mean": _tensor_delta_mean(x_after_i, x_before_i),
                "l1_u_before_std": _tensor_std(u_before_i),
                "l1_u_after_std": _tensor_std(u_after_i),
                "l1_x_before_std": _tensor_std(x_before_i),
                "l1_x_after_std": _tensor_std(x_after_i),
                "layer1_perturbed": True,
                "layer2_perturbed": False,
                "layer3_perturbed": False,
                "restore_ok": True,
                "perturbation_ok": True,
            }
        )
    return rows

def _apply_support_perturbation(
    net,
    condition: str,
    perturb_units: pd.DataFrame | None,
    attenuation_factor: float = 0.5,
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    if perturb_units is None or perturb_units.empty or not hasattr(net.layer1, "u_pre") or net.layer1.u_pre is None:
        return audit_rows
    with torch.no_grad():
        u = net.layer1.u_pre
        x = net.layer1.x_pre
        u0 = float(net.layer1.stsp_U)
        for row in perturb_units.itertuples(index=False):
            rr = min(int(row.row), u.shape[-2] - 1)
            cc = min(int(row.col), u.shape[-1] - 1)
            u_before = u[..., rr, cc].detach().clone()
            x_before = x[..., rr, cc].detach().clone()
            g_before = u_before * x_before
            if condition.startswith("attenuate"):
                u[..., rr, cc] = u0 + float(attenuation_factor) * (u[..., rr, cc] - u0)
            elif condition.startswith("reset"):
                u[..., rr, cc] = u0
                x[..., rr, cc] = 1.0
            elif condition == "sham_perturbation":
                pass
            u_after = u[..., rr, cc].detach().clone()
            x_after = x[..., rr, cc].detach().clone()
            g_after = u_after * x_after
            audit_rows.append(
                {
                    "condition": condition,
                    "unit_id": int(row.unit_id),
                    "row": int(row.row),
                    "col": int(row.col),
                    "u_before_mean": float(u_before.float().mean().cpu()),
                    "x_before_mean": float(x_before.float().mean().cpu()),
                    "g_before_mean": float(g_before.float().mean().cpu()),
                    "u_after_mean": float(u_after.float().mean().cpu()),
                    "x_after_mean": float(x_after.float().mean().cpu()),
                    "g_after_mean": float(g_after.float().mean().cpu()),
                    "u_delta_mean": float((u_after - u_before).float().mean().cpu()),
                    "x_delta_mean": float((x_after - x_before).float().mean().cpu()),
                    "g_delta_mean": float((g_after - g_before).float().mean().cpu()),
                }
            )
    return audit_rows

def _apply_support_perturbation_batch(
    ctx: ExperimentContext,
    condition: str,
    perturbation_sets_by_local: Mapping[int, pd.DataFrame],
    *,
    batch_size: int,
    attenuation_factor: float = 0.5,
) -> dict[int, list[dict[str, Any]]]:
    net = ctx.net
    audit_rows: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(int(batch_size))}
    if not hasattr(net.layer1, "u_pre") or net.layer1.u_pre is None:
        return audit_rows
    with torch.no_grad():
        u = net.layer1.u_pre
        x = net.layer1.x_pre
        u0 = float(net.layer1.stsp_U)
        for local_idx in range(int(batch_size)):
            perturb_units = perturbation_sets_by_local.get(local_idx)
            if perturb_units is None or perturb_units.empty:
                continue
            part = perturb_units[perturb_units["condition"].eq(condition)]
            if part.empty:
                continue
            for row in part.itertuples(index=False):
                rr = min(int(row.row), u.shape[-2] - 1)
                cc = min(int(row.col), u.shape[-1] - 1)
                u_site = u[local_idx : local_idx + 1, ..., rr, cc]
                x_site = x[local_idx : local_idx + 1, ..., rr, cc]
                u_before = u_site.detach().clone()
                x_before = x_site.detach().clone()
                g_before = u_before * x_before
                if condition.startswith("attenuate"):
                    u_site.copy_(u0 + float(attenuation_factor) * (u_site - u0))
                elif condition.startswith("reset"):
                    u_site.copy_(torch.as_tensor(float(u0), dtype=u_site.dtype, device=u_site.device))
                    x_site.copy_(torch.as_tensor(1.0, dtype=x_site.dtype, device=x_site.device))
                elif condition == "sham_perturbation":
                    pass
                u_after = u_site.detach().clone()
                x_after = x_site.detach().clone()
                g_after = u_after * x_after
                audit_rows[local_idx].append(
                    {
                        "condition": condition,
                        "unit_id": int(row.unit_id),
                        "row": int(row.row),
                        "col": int(row.col),
                        "u_before_mean": float(u_before.float().mean().cpu()),
                        "x_before_mean": float(x_before.float().mean().cpu()),
                        "g_before_mean": float(g_before.float().mean().cpu()),
                        "u_after_mean": float(u_after.float().mean().cpu()),
                        "x_after_mean": float(x_after.float().mean().cpu()),
                        "g_after_mean": float(g_after.float().mean().cpu()),
                        "u_delta_mean": float((u_after - u_before).float().mean().cpu()),
                        "x_delta_mean": float((x_after - x_before).float().mean().cpu()),
                        "g_delta_mean": float((g_after - g_before).float().mean().cpu()),
                    }
                )
    return audit_rows

def _support_maps_from_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int) -> dict[int, np.ndarray]:
    state = boundary.get("layer1", {})
    if "u" in state and "x" in state:
        support = (state["u"].to(torch.float32) * state["x"].to(torch.float32)).mean(dim=1).numpy()
    else:
        raise RuntimeError("Fig.5 requires Layer1 u/x STSP state at the pre-probe boundary.")
    return {idx: _resize_array(support[idx], 28, 28).astype(np.float32) for idx in range(batch_size)}

def _save_probe_trace_manifest(ctx: ExperimentContext, branch_traces: Mapping[int, Mapping[str, BranchTrace]]) -> None:
    rows = []
    for trial_id, traces in branch_traces.items():
        for condition, trace in traces.items():
            rows.append(
                {
                    "network_seed": int(ctx.cfg.network_seed),
                    "trial_id": int(trial_id),
                    "condition": condition,
                    "trace_kind": "layer1_spatial_collapsed",
                    "n_time_steps": int(trace.spikes.shape[0]),
                    "height": int(trace.spikes.shape[1]),
                    "width": int(trace.spikes.shape[2]),
                    "save_full_traces": bool(ctx.cfg.save_full_traces),
                }
            )
    _save_csv(ctx, pd.DataFrame(rows), ctx.raw_dir / "layer1_probe_trace_manifest.csv")

def _save_panel_a_example(ctx: ExperimentContext, trials: pd.DataFrame, support_maps: Mapping[int, np.ndarray], unit_groups: pd.DataFrame) -> None:
    first = trials.iloc[0]
    trial_id = int(first["trial_id"])
    sample_mask, probe_mask, overlap, probe_only = _entry_masks_for_trial(
        ctx,
        int(first["sample_image_id"]),
        int(first["probe_image_id"]),
    )
    groups = unit_groups[unit_groups["trial_id"].eq(trial_id)]
    np.savez_compressed(
        ctx.raw_dir / "panel_a_example_support_map.npz",
        support_map=support_maps[trial_id].astype(np.float32),
        sample_foreground_mask=sample_mask.astype(np.uint8),
        probe_foreground_mask=probe_mask.astype(np.uint8),
        sample_entry_mask=sample_mask.astype(np.uint8),
        probe_entry_mask=probe_mask.astype(np.uint8),
        overlap_mask_projected=overlap.astype(np.uint8),
        probe_only_mask_projected=probe_only.astype(np.uint8),
        overlap_dominant_units=groups[groups["unit_group"].eq("overlap_dominant")]["unit_id"].to_numpy(dtype=np.int64),
        probe_only_dominant_units=groups[groups["unit_group"].eq("probe_only_dominant")]["unit_id"].to_numpy(dtype=np.int64),
        selected_trial_metadata=json.dumps(first.to_dict(), sort_keys=True),
    )
    ctx.output_files["panel_a_example_support_map"] = _rel(ctx.raw_dir / "panel_a_example_support_map.npz", ctx.seed_dir)

def _save_trial_mask_npz(ctx: ExperimentContext, trials: pd.DataFrame) -> None:
    payload: dict[str, np.ndarray] = {}
    for row in trials.itertuples(index=False):
        sample_mask, probe_mask, overlap, probe_only = _entry_masks_for_trial(
            ctx,
            int(row.sample_image_id),
            int(row.probe_image_id),
        )
        payload[f"trial_{int(row.trial_id)}_sample_foreground_mask"] = sample_mask.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_probe_foreground_mask"] = probe_mask.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_sample_entry_mask"] = sample_mask.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_probe_entry_mask"] = probe_mask.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_overlap_mask"] = overlap.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_probe_only_mask"] = probe_only.astype(np.uint8)
        payload[f"trial_{int(row.trial_id)}_sample_nonoverlap_mask"] = (sample_mask & (~probe_mask)).astype(np.uint8)
    np.savez_compressed(ctx.raw_dir / "trial_masks.npz", **payload)
    ctx.output_files["trial_masks"] = _rel(ctx.raw_dir / "trial_masks.npz", ctx.seed_dir)

def _load_dataset_or_raise(dataset_root: str, split: str):
    return load_mnist_skeleton_dataset(dataset_root, split)

def _csv_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(not pd.read_csv(path).empty)
    except Exception:
        return False

def _steps_to_ms(value_steps: int | float, dt: float) -> float:
    value = float(value_steps)
    if not np.isfinite(value) or value < 0:
        return float("nan")
    return float(value * float(dt) / ms)

def _finite_delta(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")

def _row_value(row: Any | None, field: str) -> float:
    if row is None:
        return float("nan")
    value = float(getattr(row, field, np.nan))
    return value if np.isfinite(value) else float("nan")

def _fig5d_condition_label(condition: str) -> str:
    return {
        "dynamic_intact": "Dynamic",
        "attenuate_l1_stsp": "Attenuate L1 STSP",
        "reset_l1_stsp": "Reset L1 STSP",
        "static_frozen": "Static frozen",
        "attenuate_overlap_high_support": "Attenuate overlap support",
        "reset_overlap_high_support": "Reset overlap support",
        "sham_perturbation": "Sham perturbation",
    }.get(str(condition), str(condition))

def _l1_stsp_perturbation_mode(condition: str) -> str:
    if str(condition) == "attenuate_l1_stsp":
        return "attenuate"
    if str(condition) == "reset_l1_stsp":
        return "reset"
    if str(condition) == "dynamic_intact":
        return "none"
    return str(condition)

def _recovery_toward_static(dynamic: float, static: float, value: float) -> float:
    if not (np.isfinite(dynamic) and np.isfinite(static) and np.isfinite(value)):
        return float("nan")
    denom = float(static - dynamic)
    if abs(denom) < 1e-12:
        return float("nan")
    return float((value - dynamic) / denom)

def _layer_stsp_baseline_u(layer: Any, fallback: torch.Tensor) -> float:
    for attr in ("stsp_U", "U", "U0"):
        if hasattr(layer, attr):
            value = getattr(layer, attr)
            try:
                return float(value)
            except (TypeError, ValueError):
                try:
                    return float(torch.as_tensor(value).float().mean().item())
                except Exception:
                    pass
    return float(fallback.float().mean().item())

def _tensor_mean(value: torch.Tensor | None) -> float:
    return float(value.float().mean().cpu()) if value is not None else float("nan")

def _tensor_std(value: torch.Tensor | None) -> float:
    return float(value.float().std(unbiased=False).cpu()) if value is not None else float("nan")

def _tensor_delta_mean(after: torch.Tensor | None, before: torch.Tensor | None) -> float:
    if after is None or before is None:
        return float("nan")
    return float((after - before).float().mean().cpu())

def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), int(batch_size)):
        yield df.iloc[start : start + int(batch_size)].copy()

def _image_array(dataset, image_id: int) -> np.ndarray:
    image = dataset[int(image_id)][0].detach().cpu().to(torch.float32).squeeze().numpy()
    return np.asarray(image, dtype=np.float32)

def _images_for_ids(dataset, image_ids: Sequence[int]) -> torch.Tensor:
    return torch.stack([dataset[int(idx)][0].detach().to(torch.float32) for idx in image_ids], dim=0)

def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0

def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    return (arr - lo) / max(hi - lo, 1e-9)

def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return _resize_array(mask.astype(float), h, w) > 0.5

def _resize_array(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    src = np.asarray(arr)
    if src.shape == (h, w):
        return src
    rr = np.linspace(0, src.shape[0] - 1, h).round().astype(int)
    cc = np.linspace(0, src.shape[1] - 1, w).round().astype(int)
    return src[np.ix_(rr, cc)]

def _blur3(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, 1, mode="edge")
    out = np.zeros_like(arr, dtype=float)
    for dr in range(3):
        for dc in range(3):
            out += padded[dr : dr + arr.shape[0], dc : dc + arr.shape[1]]
    return out / 9.0

def _first_spike_map(spikes: np.ndarray) -> np.ndarray:
    arr = np.asarray(spikes)
    first = np.full(arr.shape[1:], -1, dtype=int)
    fired = arr > 0
    any_fire = fired.any(axis=0)
    if np.any(any_fire):
        first[any_fire] = np.argmax(fired, axis=0)[any_fire]
    return first

def _transition_type(dynamic_first: int, static_first: int) -> str:
    if dynamic_first >= 0 and static_first >= 0 and dynamic_first < static_first:
        return "advance"
    if dynamic_first >= 0 and static_first < 0:
        return "recruit"
    if dynamic_first < 0 and static_first >= 0:
        return "loss"
    return "unchanged"

def _transition_vs_same(first_cond: int, first_same: int, first_static: int) -> str:
    same_transition = _transition_type(first_same, first_static)
    cond_transition = _transition_type(first_cond, first_static)
    same_winner = same_transition in {"advance", "recruit"}
    cond_winner = cond_transition in {"advance", "recruit"}
    if not same_winner:
        return "not_same_winner"
    if first_cond < 0:
        return "lost"
    if not cond_winner:
        return "reverted_to_static"
    if first_same >= 0 and first_cond > first_same:
        return "delayed"
    return "preserved"

def _latency_delta(dynamic_first: int, static_first: int) -> float:
    if dynamic_first >= 0 and static_first >= 0:
        return float(dynamic_first - static_first)
    if dynamic_first >= 0 and static_first < 0:
        return float(-dynamic_first)
    if dynamic_first < 0 and static_first >= 0:
        return float(static_first)
    return float("nan")

def _spikes_earlier(dynamic_first: int, static_first: int) -> bool:
    return bool(dynamic_first >= 0 and (static_first < 0 or dynamic_first < static_first))

def _is_loser_suppressed(dynamic_first: int, static_first: int) -> bool:
    return bool(static_first >= 0 and (dynamic_first < 0 or dynamic_first > static_first))

def _advanced_or_recruited_units(first_dyn: np.ndarray, first_sta: np.ndarray) -> set[int]:
    out = set()
    h, w = first_dyn.shape
    for r in range(h):
        for c in range(w):
            if _transition_type(int(first_dyn[r, c]), int(first_sta[r, c])) in {"advance", "recruit"}:
                out.add(int(r * w + c))
    return out

def _delayed_or_lost_units(first_dyn: np.ndarray, first_sta: np.ndarray) -> set[int]:
    out = set()
    h, w = first_dyn.shape
    for r in range(h):
        for c in range(w):
            fd, fs = int(first_dyn[r, c]), int(first_sta[r, c])
            if (fs >= 0 and fd < 0) or (fd >= 0 and fs >= 0 and fd > fs):
                out.add(int(r * w + c))
    return out

def _nearest_loser(win: Any, losers: pd.DataFrame, radius: int):
    if losers.empty:
        return None
    part = losers.copy()
    part["dist"] = (part["row"].astype(int) - int(win.row)).abs() + (part["col"].astype(int) - int(win.col)).abs()
    part = part[part["dist"] <= int(radius) * 2]
    if part.empty:
        return None
    return next(part.sort_values("dist").itertuples(index=False))

def _aligned_delta(dynamic: np.ndarray, static: np.ndarray, t0: int, ctx: ExperimentContext) -> np.ndarray:
    vals = []
    for offset in range(-ctx.cfg.event_align_pre_steps, ctx.cfg.event_align_post_steps + 1):
        t = int(t0 + offset)
        if 0 <= t < len(dynamic):
            vals.append(float(dynamic[t] - static[t]))
        else:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float32)

def _trace_summary_row(ctx: ExperimentContext, time_ms: float, trace_type: str, value: float) -> dict[str, Any]:
    return {"network_seed": int(ctx.cfg.network_seed), "time_ms": float(time_ms), "trace_type": trace_type, "value": float(value)}

def _event_audit_row(ctx: ExperimentContext, trial_id: int, event_id: int, step: str, included: bool, reason: str, winner_group: str, loser_group: str, drive_winner: float, drive_loser: float) -> dict[str, Any]:
    return {
        "network_seed": int(ctx.cfg.network_seed),
        "trial_id": int(trial_id),
        "event_id": int(event_id),
        "selection_step": step,
        "included": bool(included),
        "exclusion_reason": reason,
        "winner_group": str(winner_group),
        "loser_group": str(loser_group),
        "neighborhood_radius": int(ctx.cfg.local_kernel_radius),
        "drive_score_winner": float(drive_winner),
        "drive_score_loser": float(drive_loser) if np.isfinite(drive_loser) else np.nan,
    }

def _mean_for_group(df: pd.DataFrame, group: str) -> float:
    part = df[df["unit_group"].eq(group)]
    return float(pd.to_numeric(part["support_value"], errors="coerce").mean()) if not part.empty else float("nan")

def _pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.dot(aa, bb) / denom)

def _decision_deflection(trace: BranchTrace, dynamic: BranchTrace, static: BranchTrace) -> float:
    dyn_sim = _pattern_similarity(trace.layer3_spikes, dynamic.layer3_spikes)
    sta_sim = _pattern_similarity(trace.layer3_spikes, static.layer3_spikes)
    return float(dyn_sim - sta_sim)

def _slice_boundary(boundary: Mapping[str, Mapping[str, torch.Tensor]], index: int) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in boundary.items():
        out[layer_key] = {}
        for key, tensor in state.items():
            out[layer_key][key] = tensor[index : index + 1].clone()
    return out

def _restore_boundary_state(net, boundary: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    with torch.no_grad():
        for layer_key, state in boundary.items():
            layer = getattr(net, layer_key)
            for src_key, attr in (("v_mem", "v_mem"), ("g_e", "g_e"), ("res", "res")):
                if src_key in state:
                    getattr(layer, attr).copy_(state[src_key].to(device=getattr(layer, attr).device, dtype=getattr(layer, attr).dtype))
            if "inh_trace" in state:
                layer.lateral_inh.inh_trace.copy_(state["inh_trace"].to(device=layer.lateral_inh.inh_trace.device, dtype=layer.lateral_inh.inh_trace.dtype))
            if "u" in state and getattr(layer, "u_pre", None) is not None:
                layer.u_pre.copy_(state["u"].to(device=layer.u_pre.device, dtype=layer.u_pre.dtype))
            if "x" in state and getattr(layer, "x_pre", None) is not None:
                layer.x_pre.copy_(state["x"].to(device=layer.x_pre.device, dtype=layer.x_pre.dtype))

def _step_network_once(net, input_t: torch.Tensor, current_time: int, *, stsp_mode: str = "dynamic") -> int:
    s1, _ = net.layer1.forward_step(input_t, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s1p = net.pool1(s1.float())
    s2, _ = net.layer2.forward_step(s1p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    s2p = net.pool2(s2.float())
    net.layer3.forward_step(s2p, current_time, training=False, monitor=False, stsp_mode=stsp_mode)
    return current_time + 1

def _trial_mapping(trial: Any) -> Mapping[str, Any]:
    if isinstance(trial, pd.Series):
        return trial.to_dict()
    if isinstance(trial, Mapping):
        return trial
    if hasattr(trial, "_asdict"):
        return trial._asdict()
    return dict(trial)

__all__ = ('_copy_csv_alias', '_write_empty_csv', '_record_optional_missing', '_mean_existing', '_run_batch_network_checked', '_run_batch_network', '_run_probe_branch', '_run_l1_stsp_probe_branches_batch', '_run_support_perturb_probe_branches_batch', '_run_probe_branches_batch', '_unit_group_rows', '_perturbation_unit_rows', '_node_metrics_for_condition', '_transition_summary', '_summarize_perturbation_transitions', '_compute_perturbation_transition_contrasts', '_summarize_l1_stsp_perturbation', '_compute_l1_stsp_perturbation_contrast', '_delta_field', '_event_trace_summary', '_early_window_robustness', '_neighborhood_radius_robustness', '_support_perturbation_controls', '_perturbation_matching_diagnostics', '_apply_l1_stsp_perturbation', '_apply_l1_stsp_perturbation_batch', '_apply_support_perturbation', '_apply_support_perturbation_batch', '_support_maps_from_boundary', '_save_probe_trace_manifest', '_save_panel_a_example', '_save_trial_mask_npz', '_load_dataset_or_raise', '_csv_nonempty', '_steps_to_ms', '_finite_delta', '_row_value', '_fig5d_condition_label', '_l1_stsp_perturbation_mode', '_recovery_toward_static', '_layer_stsp_baseline_u', '_tensor_mean', '_tensor_std', '_tensor_delta_mean', '_iter_batches', '_image_array', '_images_for_ids', '_centered_cosine', '_normalize', '_resize_mask', '_resize_array', '_blur3', '_first_spike_map', '_transition_type', '_transition_vs_same', '_latency_delta', '_spikes_earlier', '_is_loser_suppressed', '_advanced_or_recruited_units', '_delayed_or_lost_units', '_nearest_loser', '_aligned_delta', '_trace_summary_row', '_event_audit_row', '_mean_for_group', '_pattern_similarity', '_decision_deflection', '_slice_boundary', '_restore_boundary_state', '_step_network_once', '_trial_mapping')
