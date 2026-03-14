# AGENTS.md

## Project Identity

This repository is a research-grade reimplementation of the paper **"Local Perception-Enhanced Spatial-Temporal Evolving Graph Transformer Network for Citywide Demand Prediction of Taxi and Ride-Hailing"** (IEEE TITS, 2024).

The working dataset currently available in this repository is the **NYC Yellow Taxi 2018** trip dataset stored as monthly parquet files in `data/`.

The primary engineering goal is to build a **faithful, reproducible, maintainable** reproduction pipeline for the paper, starting with data ingestion, preprocessing, and baseline-ready infrastructure before implementing the full LPE-STGTN model.

## Required Session Startup Behavior

Every agent working in this repository must follow this order before editing code:

1. Read `AGENTS.md`.
2. Read `README.md`.
3. Read `Topic.md`.
4. Read the paper PDF: `Local-Perception-Enhanced_SpatialTemporal_Evolving_Graph_Transformer_Network_Citywide_Demand_Prediction_of_Taxi_and_Ride-Hailing.pdf` when architecture, training, or evaluation details are relevant.
5. Inspect the repository tree and existing configs under `configs/`, `src/`, `scripts/`, `tests/`, and `docs/`.
6. Read the latest entries in `docs/reproduction_log.md` before making assumption-heavy changes.

Do not skip the startup read path even if the repository looks simple.

## Engineering Principles

- Reproducibility first.
- Stay faithful to the paper unless there is a documented reason to diverge.
- Prefer small, reviewable changes over broad speculative rewrites.
- Keep data, graph construction, models, training, and evaluation clearly separated.
- Do not silently change assumptions, tensor layouts, data splits, or metrics.
- Treat undocumented paper details as explicit ambiguities to record, not as permission to guess invisibly.

## Coding Standards

- Target Python 3.11 unless a documented reason requires otherwise.
- Use type hints where they improve clarity without adding noise.
- Add docstrings for nontrivial modules, data transforms, graph builders, and model components.
- Avoid premature abstraction and framework-heavy indirection.
- Prefer explicit naming over short research-code shorthand.
- In model code, annotate important tensor shapes in comments or docstrings.
- Keep functions focused and testable.
- Use config files for experiment parameters instead of hardcoding run settings in scripts.

## Data And Experiment Discipline

- Never commit large raw datasets, generated features, checkpoints, or experiment artifacts.
- Keep raw data immutable once ingested.
- Save derived outputs in predictable locations such as `data/interim/`, `data/processed/`, and `artifacts/`.
- Record preprocessing assumptions in `docs/reproduction_log.md`.
- Record experiment configs, seeds, and notable environment details for each serious run.
- Keep train/validation/test splits reproducible and documented.
- If a required external asset is missing, document the missing dependency instead of inventing it.

## Model Implementation Discipline

- Build the system in stages.
- Start with data validation and simple sanity baselines before the full paper model.
- Add global-module support before implementing the most complex local dynamic graph machinery when that ordering reduces risk.
- Add tests for tensor shapes, graph construction logic, config loading, and metric computation as pieces become real.
- Preserve paper-faithful defaults where they are known.
- When the paper is ambiguous, choose the simplest defensible default and write it down in `docs/reproduction_log.md`.

## Documentation Expectations

- Maintain `docs/reproduction_log.md` as a chronological engineering log.
- Document deviations from the paper explicitly.
- Document unresolved ambiguities explicitly.
- Clearly separate implemented functionality from planned functionality in docs and status summaries.

## Preferred Workflow

- Terminal-first.
- Config-driven.
- Scriptable through `Makefile` targets and Python entrypoints.
- Friendly to continuation by future agents without requiring hidden local context.

## Safety And Integrity Rules

- Do not fabricate experimental results, missing assets, or paper details.
- Do not claim reproduction success without recorded evidence.
- Do not present planned modules as implemented modules.
- Do not overwrite or restructure raw data without necessity.
- If official code is consulted later for clarification, cite that use and avoid direct copying.

## Current Known Reproduction Facts

- The paper uses 15-minute aggregation.
- The New York experiments use Manhattan TLC zones and report 68 zones.
- The input history length is 12 steps and the forecast horizon is 12 steps.
- Time-of-day and day-of-week features are injected in the embedding layer.
- Metrics are MAE, RMSE, and MAPE.
- Training uses L1 loss and Adam with learning rate 0.001.
- The paper reports batch size 64, one GCRN layer with hidden size 64, AFT-local window size 4, and 4 attention heads in both global and fusion attention modules.

## Current Known Ambiguities

- The Manhattan zone lookup asset is not currently present in the repository.
- Several graph-construction hyperparameters in the paper are under-specified, including distance and OD sparsification thresholds.
- Some exact channel sizes, convolution details, and dynamic graph factorization choices are not fully recoverable from the paper text alone.
- MAPE zero-handling is not clearly specified in the paper and must be documented when implemented.

## Working Agreement For Future Agents

- Read first, edit second.
- Keep the codebase professional and research-ready.
- Explain major implementation decisions.
- Prefer correctness and traceability over speed of adding features.
- Leave the repository easier to continue than you found it.
