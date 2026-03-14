# Ride-Hailing Demand Forecasting Reproduction

Research-grade reimplementation scaffold for the paper **"Local Perception-Enhanced Spatial-Temporal Evolving Graph Transformer Network for Citywide Demand Prediction of Taxi and Ride-Hailing"**.

The repository is structured for a reproducible, staged implementation of the LPE-STGTN model on the **NYC Yellow Taxi 2018** dataset, with emphasis on clean engineering, documented assumptions, and terminal-first workflows.

## Scope

Current focus:

- establish a reproducible Python 3.11 environment
- validate and inspect the raw NYC Yellow Taxi data
- define a clean project layout for preprocessing, graph construction, modeling, training, and evaluation
- document paper-derived defaults and unresolved ambiguities before implementing the full model

Not implemented yet:

- full Manhattan zone preprocessing pipeline
- graph construction assets for the exact paper setting
- baseline training runs
- final LPE-STGTN architecture

## Paper Target

- Paper: `Local-Perception-Enhanced_SpatialTemporal_Evolving_Graph_Transformer_Network_Citywide_Demand_Prediction_of_Taxi_and_Ride-Hailing.pdf`
- Forecast task: multistep citywide taxi demand prediction
- New York setup from the paper:
  - Manhattan TLC zones
  - 15-minute aggregation
  - `T = 12` historical steps
  - `P = 12` forecast steps
  - metrics: MAE, RMSE, MAPE

## Repository Layout

```text
configs/        Config files for data, model, training, and experiments
data/           Raw data plus placeholders for external, interim, and processed assets
docs/           Reproduction log and project notes
scripts/        Terminal-first helper scripts
src/            Python package for data, graphs, models, training, and utilities
tests/          Unit tests for core utilities and early pipeline pieces
artifacts/      Checkpoints, logs, and generated reports
```

## Environment Setup

Python 3.11 is the intended runtime.

```bash
make install
```

This creates `.venv/` and installs the package in editable mode with development dependencies.

If you prefer manual setup:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

## Available Commands

```bash
make inspect-data
make test
make lint
make format
```

Raw data inspection can also be run directly:

```bash
.venv/bin/lpe-stgtn inspect-data --limit 1 --sample-rows 1
```

## Data Notes

- The raw monthly parquet files are expected under `data/`.
- Raw data is intentionally not tracked by git.
- The exact Manhattan zone lookup asset required for paper-faithful preprocessing is not yet present in the repository.
- Derived outputs should go to `data/interim/`, `data/processed/`, and `artifacts/`.

## Current Implementation Status

Implemented:

- project packaging via `pyproject.toml`
- config-driven scaffold under `configs/`
- CLI entrypoint for raw parquet inspection
- metric utilities for MAE, RMSE, and MAPE
- reproducibility helpers and project path utilities
- initial unit tests for config loading, data inspection, and metrics

Known ambiguities from the paper:

- distance graph threshold details are under-specified
- OD-flow graph sparsification details are under-specified
- some channel sizes and convolution details are not explicit
- the Manhattan study-area asset is missing locally

These are being tracked in `docs/reproduction_log.md`.

## Next Steps

1. Add the TLC zone lookup asset and Manhattan filtering pipeline.
2. Generate reproducible 15-minute demand tensors and train/validation/test splits.
3. Implement baseline models before the full LPE-STGTN architecture.
4. Add graph builders for distance and OD-flow semantics.
5. Implement the paper model in staged modules with tensor-shape tests.

## Reproducibility

- Keep assumptions and deviations recorded in `docs/reproduction_log.md`.
- Do not claim reproduction success before baseline and full-model evaluations are complete.
- Prefer small, reviewable commits and config-driven experiment execution.
