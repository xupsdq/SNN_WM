from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_multiitem_sequence_state_bank(ctx: ExperimentContext, sequence_trials: pd.DataFrame) -> MultiItemSequenceLandscapeBank:
    cfg = ctx.cfg
    arrays: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    singleton_refs: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]] = {}
    singleton_boundaries: dict[int, dict[int, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    boundaries: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]] = {}
    landscapes: dict[int, dict[str, np.ndarray]] = {}
    manifest_rows: list[dict[str, Any]] = []
    l1_payload: dict[str, np.ndarray] = {}
    l3_payload: dict[str, np.ndarray] = {}
    meta_rows: list[dict[str, Any]] = []
    s0_cache: dict[tuple[int, int, int, int], tuple[dict[str, dict[str, np.ndarray]], Mapping[str, Mapping[str, torch.Tensor]]]] = {}

    encode_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    groups = list(sequence_trials.groupby("sequence_id", sort=True))
    for sequence_id, group in _progress(groups, total=len(groups), desc="fig3 state sequences", enabled=cfg.show_progress):
        seq_id = int(sequence_id)
        group = group.sort_values("stage_k")
        seq_len = int(group["seq_len"].iloc[0])
        image_ids = group["item_image_id"].to_numpy(dtype=np.int64).tolist()
        labels = group["item_label"].to_numpy(dtype=np.int64).tolist()
        spikes = _encode_cached(ctx, image_ids, cfg.sample_steps, cache=encode_cache)
        state_arrays, state_boundaries = _capture_sequence(ctx, spikes, s0_cache=s0_cache)
        refs, ref_boundaries = _capture_singleton_refs_and_boundaries(ctx, spikes)
        arrays[seq_id] = state_arrays
        singleton_refs[seq_id] = refs
        singleton_boundaries[seq_id] = ref_boundaries
        boundaries[seq_id] = {"S0": state_boundaries["S0"], "S_final": state_boundaries["S_final"]}
        landscapes[seq_id] = _landscape_for_sequence(ctx, state_arrays, group)
        meta_rows.append({"sequence_id": seq_id, "seq_len": seq_len, "ordered_item_ids": ";".join(map(str, image_ids)), "ordered_item_labels": ";".join(map(str, labels))})
        for state, layer_map in state_arrays.items():
            stage_k = 0 if state == "S0" else (seq_len if state == "S_final" else int(state.split("_")[1]))
            for layer in ("layer1", "layer3"):
                for variable in STATE_VARIABLES:
                    arr = layer_map[layer][variable].astype(np.float32, copy=False)
                    key_state = state.replace("_", "")
                    storage_file = "state_bank_layer1.npz" if layer == "layer1" else "state_bank_layer3.npz"
                    storage_key = f"sequence_{seq_id}_{key_state}_{variable}"
                    manifest_rows.append(
                        {
                            "network_seed": int(cfg.network_seed),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "state_condition": state,
                            "stage_k": stage_k,
                            "layer": layer,
                            "state_variable": variable,
                            "shape": "x".join(str(v) for v in arr.shape),
                            "storage_file": storage_file,
                            "storage_key": storage_key,
                            "captured_after": "item_delay" if state not in {"S0", "S_final"} else state,
                            "sample_ms": int(cfg.sample_ms),
                            "delay_ms": int(cfg.delay_ms),
                        }
                    )
                    if layer == "layer1":
                        l1_payload[storage_key] = arr
                    else:
                        l3_payload[storage_key] = arr
        for pos, ref in refs.items():
            for layer in ("layer1", "layer3"):
                for variable in STATE_VARIABLES:
                    arr = ref[layer][variable].astype(np.float32, copy=False)
                    storage_file = "state_bank_layer1.npz" if layer == "layer1" else "state_bank_layer3.npz"
                    storage_key = f"sequence_{seq_id}_singleton_reference_{pos}_{variable}"
                    manifest_rows.append(
                        {
                            "network_seed": int(cfg.network_seed),
                            "sequence_id": seq_id,
                            "seq_len": seq_len,
                            "state_condition": "singleton_reference",
                            "stage_k": int(pos),
                            "layer": layer,
                            "state_variable": variable,
                            "shape": "x".join(str(v) for v in arr.shape),
                            "storage_file": storage_file,
                            "storage_key": storage_key,
                            "captured_after": f"temporal_slot_{pos}",
                            "sample_ms": int(cfg.sample_ms),
                            "delay_ms": int(cfg.delay_ms),
                        }
                    )
                    if layer == "layer1":
                        l1_payload[storage_key] = arr
                    else:
                        l3_payload[storage_key] = arr
            manifest_rows.append(
                {
                    "network_seed": int(cfg.network_seed),
                    "sequence_id": seq_id,
                    "seq_len": seq_len,
                    "state_condition": "singleton_boundary",
                    "stage_k": int(pos),
                    "layer": "",
                    "state_variable": "",
                    "shape": "",
                    "storage_file": "",
                    "storage_key": "",
                    "captured_after": f"temporal_slot_{pos}_singleton_end",
                    "sample_ms": int(cfg.sample_ms),
                    "delay_ms": int(cfg.delay_ms),
                    "restore_mode": str(cfg.functional_restore_mode),
                }
            )

    np.savez_compressed(ctx.raw_dir / "state_bank_layer1.npz", **l1_payload)
    np.savez_compressed(ctx.raw_dir / "state_bank_layer3.npz", **l3_payload)
    _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "state_bank_manifest.csv")
    ctx.output_files["state_bank_layer1"] = _rel(ctx.raw_dir / "state_bank_layer1.npz", ctx.seed_dir)
    ctx.output_files["state_bank_layer3"] = _rel(ctx.raw_dir / "state_bank_layer3.npz", ctx.seed_dir)
    bank = MultiItemSequenceLandscapeBank(
        sequence_trials=sequence_trials.reset_index(drop=True),
        sequence_meta=pd.DataFrame(meta_rows),
        arrays=arrays,
        singleton_refs=singleton_refs,
        singleton_boundaries=singleton_boundaries,
        boundaries=boundaries,
        landscapes=landscapes,
    )
    _save_example_landscape(ctx, bank)
    ctx.completed_modules["state_bank"] = True
    return bank
