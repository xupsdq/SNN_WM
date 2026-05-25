from __future__ import annotations

from src.experiments.paper_figures import fig2_pair_fused_stsp_state_experiment as _legacy

# Keep module-level names identical while Fig.2 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_pair_episode_state_bank(ctx: ExperimentContext, pair_trials: pd.DataFrame) -> PairEpisodeStateBank:
    cfg = ctx.cfg
    arrays: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        cond: {layer: {"u": [], "x": [], "g": []} for layer in LAYER_KEYS} for cond in STATE_CONDITIONS
    }
    boundary_states: dict[str, Mapping[str, Mapping[str, torch.Tensor]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    all_layer_manifest_rows: list[dict[str, Any]] = []
    first_episode_saved = False

    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    batches = _iter_batches(pair_trials, cfg.batch_size)
    for batch in _progress(
        batches,
        total=math.ceil(len(pair_trials) / cfg.batch_size),
        desc="fig2 state batches",
        enabled=cfg.show_progress,
    ):
        a_spikes = _encode_cached(ctx, batch["A_image_id"].to_numpy(), cfg.sample_steps, cache=encode_cache)
        b_spikes = _encode_cached(ctx, batch["B_image_id"].to_numpy(), cfg.second_item_steps, cache=encode_cache)
        batch_bank, batch_boundaries = _capture_pair_batch(ctx, a_spikes, b_spikes)
        for cond in STATE_CONDITIONS:
            if cond not in boundary_states:
                boundary_states[cond] = batch_boundaries[cond]
            else:
                boundary_states[cond] = _concat_boundary_states(boundary_states[cond], batch_boundaries[cond])
            for layer in LAYER_KEYS:
                for variable in ("u", "x", "g"):
                    arrays[cond][layer][variable].append(batch_bank[cond][layer][variable])
        if not first_episode_saved and len(batch) > 0:
            first_episode_saved = True
            example = batch.iloc[0].to_dict()
            example_a_image = ctx.dataset[int(example["A_image_id"])][0].detach().cpu().to(torch.float32).numpy()
            example_b_image = ctx.dataset[int(example["B_image_id"])][0].detach().cpu().to(torch.float32).numpy()
            _write_json(_json_safe(example), ctx.raw_dir / "panel_a_example_episode_metadata.json")
            np.savez_compressed(
                ctx.raw_dir / "panel_a_example_episode.npz",
                A_image=example_a_image,
                B_image=example_b_image,
            )
            ctx.output_files["panel_a_example_episode_metadata"] = _rel(ctx.raw_dir / "panel_a_example_episode_metadata.json", ctx.seed_dir)
            ctx.output_files["panel_a_example_episode"] = _rel(ctx.raw_dir / "panel_a_example_episode.npz", ctx.seed_dir)

    final_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {
        cond: {layer: {} for layer in LAYER_KEYS} for cond in STATE_CONDITIONS
    }
    l3_payload: dict[str, np.ndarray] = {}
    all_layer_payload: dict[str, np.ndarray] = {}
    for cond in STATE_CONDITIONS:
        for layer in LAYER_KEYS:
            for variable in ("u", "x", "g"):
                arr = np.vstack(arrays[cond][layer][variable]).astype(np.float32, copy=False)
                final_arrays[cond][layer][variable] = arr
                storage_file = "state_bank_l3.npz" if layer == "layer3" else ("state_bank_all_layers.npz" if cfg.save_all_layer_state_bank else "")
                storage_key = f"{cond}_{variable}" if layer == "layer3" else f"{cond}_{layer}_{variable}"
                row = {
                    "network_seed": int(cfg.network_seed),
                    "pair_id": "all",
                    "state_condition": cond,
                    "layer": layer,
                    "state_variable": variable,
                    "shape": "x".join(str(v) for v in arr.shape),
                    "storage_file": storage_file,
                    "storage_key": storage_key if storage_file else "",
                    "captured_after": "A_delay1_B_delay2",
                    "sample_ms": int(cfg.sample_ms),
                    "delay1_ms": int(cfg.delay1_ms),
                    "second_item_ms": int(cfg.second_item_ms),
                    "delay2_ms": int(cfg.delay2_ms),
                }
                all_layer_manifest_rows.append(row)
                if layer == "layer3":
                    manifest_rows.append(row)
                    l3_payload[f"{cond}_{variable}"] = arr
                elif cfg.save_all_layer_state_bank:
                    all_layer_payload[storage_key] = arr
            final_arrays[cond][layer]["ux_concat"] = np.concatenate(
                [final_arrays[cond][layer]["u"], final_arrays[cond][layer]["x"]],
                axis=1,
            ).astype(np.float32, copy=False)
    np.savez_compressed(ctx.raw_dir / "state_bank_l3.npz", **l3_payload)
    ctx.output_files["state_bank_l3"] = _rel(ctx.raw_dir / "state_bank_l3.npz", ctx.seed_dir)
    if cfg.save_all_layer_state_bank:
        np.savez_compressed(ctx.raw_dir / "state_bank_all_layers.npz", **all_layer_payload)
        ctx.output_files["state_bank_all_layers"] = _rel(ctx.raw_dir / "state_bank_all_layers.npz", ctx.seed_dir)
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "state_bank_manifest.csv")
    _save_csv(ctx, pd.DataFrame(all_layer_manifest_rows), ctx.raw_dir / "state_bank_all_layers_manifest.csv")
    layer_input_shapes = _layer_input_shapes_from_boundary(boundary_states["S0"])
    ctx.completed_modules["state_bank"] = True
    return PairEpisodeStateBank(
        pair_trials=pair_trials.reset_index(drop=True),
        arrays=final_arrays,
        boundary_states=boundary_states,
        layer_input_shapes=layer_input_shapes,
        restore_mode=str(cfg.functional_restore_mode),
        episode_end_step=int(cfg.sample_steps + cfg.delay1_steps + cfg.second_item_steps + cfg.delay2_steps),
    )
