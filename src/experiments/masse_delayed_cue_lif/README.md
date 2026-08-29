# masse_delayed_cue_lif

Independent trainable recurrent LIF experiment for the Masse delayed-cue DMS+DMRS task.

This package is not part of the Tiddia / `recurrent_stsp` line and is not registered in the 13-experiment mainline catalog. Formal 500-unit training writes `results/masse_delayed_cue_lif/formal/`. The stripped STSP match (no 800 ms current, no SFA) writes `stripped_no_stsp/` and `stripped_stsp/` and must not overwrite `formal/`.

```powershell
python -m src.experiments.masse_delayed_cue_lif.run build-trials --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp --reuse-trials results/masse_delayed_cue_lif/formal/data/trials.csv
python -m src.experiments.masse_delayed_cue_lif.run train --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
python -m src.experiments.masse_delayed_cue_lif.run evaluate --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
python -m src.experiments.masse_delayed_cue_lif.run decode --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
python -m src.experiments.masse_delayed_cue_lif.run plot --profile stripped_stsp --output-directory results/masse_delayed_cue_lif/stripped_stsp
```

Contract details live in `docs/experiments/protocols/masse_delayed_cue_lif_task_spec.md` and `docs/experiments/protocols/masse_delayed_cue_lif_stsp_match_spec.md`.

```powershell
python -m src.experiments.masse_delayed_cue_lif.run build-trials --profile formal --output-directory results/masse_delayed_cue_lif/formal
python -m src.experiments.masse_delayed_cue_lif.run train --profile formal --output-directory results/masse_delayed_cue_lif/formal
python -m src.experiments.masse_delayed_cue_lif.run evaluate --profile formal --output-directory results/masse_delayed_cue_lif/formal
python -m src.experiments.masse_delayed_cue_lif.run plot --profile formal --output-directory results/masse_delayed_cue_lif/formal
```

Contract details live in `docs/experiments/protocols/masse_delayed_cue_lif_task_spec.md`.
