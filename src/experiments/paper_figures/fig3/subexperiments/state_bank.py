from __future__ import annotations

from src.experiments.paper_figures import fig3_multiitem_peak_landscape_experiment as _legacy
from src.experiments.paper_figures.fig3.subexperiments.helpers_1 import _capture_sequences_same_length_batch

# Keep module-level names identical while Fig.3 is split into smaller files.
for _name, _value in vars(_legacy).items():
    if _name not in globals() and _name != "__builtins__":
        globals()[_name] = _value

def run_multiitem_sequence_state_bank(
    ctx: ExperimentContext,
    sequence_trials: pd.DataFrame,
    *,
    write_compat_outputs: bool = True,
) -> MultiItemSequenceLandscapeBank:
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
    jobs = [_sequence_job(sequence_id, group) for sequence_id, group in groups]
    if bool(getattr(cfg, "enable_state_bank_batch", False)):
        by_len: dict[int, list[dict[str, Any]]] = {}
        for job in jobs:
            by_len.setdefault(int(job["seq_len"]), []).append(job)
        progress_units = [chunk for seq_len in sorted(by_len) for chunk in _chunks(by_len[seq_len], max(1, int(cfg.batch_size)))]
        for chunk in _progress(progress_units, total=len(progress_units), desc="fig3 state sequence batches", enabled=cfg.show_progress):
            spikes_batch = torch.stack(
                [_encode_cached(ctx, job["image_ids"], cfg.sample_steps, cache=encode_cache) for job in chunk],
                dim=0,
            ).contiguous()
            batch_results = _capture_sequences_same_length_batch(ctx, spikes_batch)
            for job, (state_arrays, state_boundaries, refs, ref_boundaries) in zip(chunk, batch_results):
                _append_state_bank_sequence(
                    ctx,
                    job,
                    state_arrays,
                    state_boundaries,
                    refs,
                    ref_boundaries,
                    arrays=arrays,
                    singleton_refs=singleton_refs,
                    singleton_boundaries=singleton_boundaries,
                    boundaries=boundaries,
                    landscapes=landscapes,
                    manifest_rows=manifest_rows,
                    l1_payload=l1_payload,
                    l3_payload=l3_payload,
                    meta_rows=meta_rows,
                )
    else:
        for job in _progress(jobs, total=len(jobs), desc="fig3 state sequences", enabled=cfg.show_progress):
            spikes = _encode_cached(ctx, job["image_ids"], cfg.sample_steps, cache=encode_cache)
            state_arrays, state_boundaries = _capture_sequence(ctx, spikes, s0_cache=s0_cache)
            refs, ref_boundaries = _capture_singleton_refs_and_boundaries(ctx, spikes)
            _append_state_bank_sequence(
                ctx,
                job,
                state_arrays,
                state_boundaries,
                refs,
                ref_boundaries,
                arrays=arrays,
                singleton_refs=singleton_refs,
                singleton_boundaries=singleton_boundaries,
                boundaries=boundaries,
                landscapes=landscapes,
                manifest_rows=manifest_rows,
                l1_payload=l1_payload,
                l3_payload=l3_payload,
                meta_rows=meta_rows,
            )

    bank = MultiItemSequenceLandscapeBank(
        sequence_trials=sequence_trials.reset_index(drop=True),
        sequence_meta=pd.DataFrame(meta_rows),
        arrays=arrays,
        singleton_refs=singleton_refs,
        singleton_boundaries=singleton_boundaries,
        boundaries=boundaries,
        landscapes=landscapes,
    )
    if write_compat_outputs:
        np.savez_compressed(ctx.raw_dir / "state_bank_layer1.npz", **l1_payload)
        np.savez_compressed(ctx.raw_dir / "state_bank_layer3.npz", **l3_payload)
        _save_csv(ctx, pd.DataFrame(manifest_rows), ctx.raw_dir / "state_bank_manifest.csv")
        ctx.output_files["state_bank_layer1"] = _rel(ctx.raw_dir / "state_bank_layer1.npz", ctx.seed_dir)
        ctx.output_files["state_bank_layer3"] = _rel(ctx.raw_dir / "state_bank_layer3.npz", ctx.seed_dir)
        _save_example_landscape(ctx, bank)
    ctx.completed_modules["state_bank"] = True
    return bank


def _sequence_job(sequence_id: Any, group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("stage_k")
    return {
        "seq_id": int(sequence_id),
        "group": group,
        "seq_len": int(group["seq_len"].iloc[0]),
        "image_ids": group["item_image_id"].to_numpy(dtype=np.int64).tolist(),
        "labels": group["item_label"].to_numpy(dtype=np.int64).tolist(),
    }


def _chunks(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + int(chunk_size)] for idx in range(0, len(items), int(chunk_size))]


def _append_state_bank_sequence(
    ctx: ExperimentContext,
    job: dict[str, Any],
    state_arrays: dict[str, dict[str, dict[str, np.ndarray]]],
    state_boundaries: dict[str, Mapping[str, Mapping[str, torch.Tensor]]],
    refs: dict[int, dict[str, dict[str, np.ndarray]]],
    ref_boundaries: dict[int, Mapping[str, Mapping[str, torch.Tensor]]],
    *,
    arrays: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]],
    singleton_refs: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]],
    singleton_boundaries: dict[int, dict[int, Mapping[str, Mapping[str, torch.Tensor]]]],
    boundaries: dict[int, dict[str, Mapping[str, Mapping[str, torch.Tensor]]]],
    landscapes: dict[int, dict[str, np.ndarray]],
    manifest_rows: list[dict[str, Any]],
    l1_payload: dict[str, np.ndarray],
    l3_payload: dict[str, np.ndarray],
    meta_rows: list[dict[str, Any]],
) -> None:
    cfg = ctx.cfg
    seq_id = int(job["seq_id"])
    seq_len = int(job["seq_len"])
    image_ids = [int(value) for value in job["image_ids"]]
    labels = [int(value) for value in job["labels"]]
    group = job["group"]
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
