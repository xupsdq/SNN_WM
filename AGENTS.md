# AGENTS.md

- Build experiment workflows as a DAG: persisted inputs → reusable artifacts → downstream outputs → plot-only leaf tasks.
- Each task must declare its dependencies. Plotting must read existing outputs and never rerun simulations.
