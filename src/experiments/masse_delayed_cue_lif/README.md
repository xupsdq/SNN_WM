# masse_delayed_cue_lif

Independent trainable recurrent LIF experiment for the Masse delayed-cue DMS+DMRS task.

This package is not part of the Tiddia / `recurrent_stsp` line and is not registered in the 13-experiment mainline catalog. Formal 500-unit training writes `results/masse_delayed_cue_lif/formal/`.

```powershell
python -m src.experiments.masse_delayed_cue_lif.run build-trials --profile formal --output-directory results/masse_delayed_cue_lif/formal
python -m src.experiments.masse_delayed_cue_lif.run train --profile formal --output-directory results/masse_delayed_cue_lif/formal
python -m src.experiments.masse_delayed_cue_lif.run evaluate --profile formal --output-directory results/masse_delayed_cue_lif/formal
python -m src.experiments.masse_delayed_cue_lif.run plot --profile formal --output-directory results/masse_delayed_cue_lif/formal
```

Contract details live in `docs/experiments/protocols/masse_delayed_cue_lif_task_spec.md`.
