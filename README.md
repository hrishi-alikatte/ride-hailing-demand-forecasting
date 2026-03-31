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
make prepare-data
make download-road-network
make build-graphs
make run-baseline
make compare-zone-tracks
.venv/bin/lpe-stgtn inspect-study-area
make test
make lint
make format
```

Raw data inspection can also be run directly:

```bash
.venv/bin/lpe-stgtn inspect-data --limit 1 --sample-rows 1
.venv/bin/lpe-stgtn prepare-data
.venv/bin/lpe-stgtn download-road-network
.venv/bin/lpe-stgtn build-graphs --config configs/graphs/nyc_manhattan_default.yaml
.venv/bin/lpe-stgtn run-baseline --config configs/experiments/nyc_persistence_baseline.yaml
.venv/bin/lpe-stgtn compare-zone-tracks
```

Visualization notebook:

- `docs/stage1_stage2_visualization.ipynb` explains the stage-1 preprocessing artifacts and stage-2 baseline results using the generated repository-local outputs.
- `docs/stage_progress_visualization.ipynb` gives a staged project review from stage 1 through stage 4, including the 68-zone vs 67-zone comparison and the first graph-based baseline.

## Data Notes

- The raw monthly parquet files are expected under `data/`.
- Raw data is intentionally not tracked by git.
- The repository now includes `data/external/taxi_zone_lookup.csv` and `data/external/taxi_zones/` for Manhattan lookup and zone geometry.
- The official NYC street-centerline asset should be downloaded into `data/external/nyc_centerline/` with `make download-road-network` before rebuilding the default stage-3 distance graph.
- The exact paper-faithful Manhattan study-area rule remains unresolved because the current TLC lookup and geometry assets expose 69 Manhattan zones, while the paper reports 68.
- Derived outputs should go to `data/interim/`, `data/processed/`, and `artifacts/`.

## Current Implementation Status

Implemented:

- project packaging via `pyproject.toml`
- config-driven scaffold under `configs/`
- CLI entrypoint for raw parquet inspection
- lookup-backed CLI entrypoint for borough study-area inspection
- stage-1 preprocessing pipeline for reproducible Manhattan pickup-demand tensors and split metadata
- stage-2 baseline runner with persistence and LSTM baselines on the processed dataset
- stage-3 graph builders for official-road-network distance and OD-flow adjacency artifacts
- stage-4 first graph-based baseline using dual semantic graph convolution, semantic fusion, and a GRU temporal head
- parallel 68-zone and 67-zone Manhattan study-area configs for sensitivity comparison
- metric utilities for MAE, RMSE, and MAPE
- taxi-zone lookup utilities for reproducible borough filtering and explicit zone exclusions
- processed-dataset loader and lazy supervised window datasets
- pure-Python shapefile/DBF readers for the taxi-zone geometry assets
- official NYC centerline downloader plus road-network shortest-path distance graph support
- reproducibility helpers and project path utilities
- initial unit tests for config loading, data inspection, metrics, and preprocessing helpers

Known ambiguities from the paper:

- distance graph threshold details are under-specified
- OD-flow graph sparsification details are under-specified
- some channel sizes and convolution details are not explicit
- the exact Manhattan 68-zone filtering rule is not yet resolved from the current TLC assets

These are being tracked in `docs/reproduction_log.md`.

## Next Steps

1. Tune the stage-4 graph baseline beyond the initial quick CPU benchmark, or decide it has served its purpose as infrastructure validation.
2. Implement the full paper model in staged modules with tensor-shape tests.
3. Run controlled comparisons across the 68-zone mainline and 67-zone sensitivity track as needed.

## Reproducibility

- Keep assumptions and deviations recorded in `docs/reproduction_log.md`.
- Do not claim reproduction success before baseline and full-model evaluations are complete.
- Prefer small, reviewable commits and config-driven experiment execution.
