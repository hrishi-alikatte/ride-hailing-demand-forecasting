# Reproduction Log

## 2026-03-14 - Initial repository analysis and scaffold planning

### Source documents reviewed

- `README.md`
- `Topic.md`
- `Local-Perception-Enhanced_SpatialTemporal_Evolving_Graph_Transformer_Network_Citywide_Demand_Prediction_of_Taxi_and_Ride-Hailing.pdf`

### What is confirmed from the source of truth

- This is a course-style deep learning reproduction project with strong emphasis on reproducibility, analysis, and faithful reimplementation.
- The target task is multistep citywide taxi and ride-hailing demand forecasting.
- For the New York experiments, the paper uses Manhattan TLC zones with 15-minute aggregation.
- The paper states `T = 12` historical steps and `P = 12` forecast steps, corresponding to a 3-hour input window and 3-hour prediction horizon.
- The embedding layer concatenates demand with time-of-day and day-of-week information.
- The local path consists of a spatial-temporal evolving graph generator, AFT-local, and a one-layer graph convolution recurrent block.
- The global path uses distance and OD-flow graphs fused with multi-head attention.
- Reported training defaults include Adam, learning rate `0.001`, L1 loss, batch size `64`, local window size `4`, and hidden size `64`.
- Evaluation metrics are MAE, RMSE, and MAPE.

### Repository state at session start

- The repository contained only `README.md`, `Topic.md`, the paper PDF, and 12 monthly NYC Yellow Taxi parquet files for 2018.
- The local raw data footprint is approximately `1.4G`.
- Python `3.11.4` is available locally.
- `pandas`, `pyarrow`, `numpy`, `torch`, and `pypdf` are available in the current environment.
- `pytest` is not available in the current environment at session start.

### Data observations

- The parquet schema includes the expected Yellow Taxi fields, including `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`, and `DOLocationID`.
- The repository does not currently include a TLC taxi-zone lookup file or any Manhattan zone metadata needed to reproduce the exact 68-zone study area.

### Ambiguities and blocked details from the paper

- The exact values of some graph-construction thresholds are not clearly recoverable from the paper text alone, including distance and OD sparsification controls.
- The shared-pattern pool size and some factorization details for the period-varying graph are under-specified.
- The exact convolution kernel sizes and some embedding-channel choices are not fully specified.
- The paper defines MAPE but does not clarify zero-demand handling, which matters for implementation stability.

### Initial engineering decisions

- Use a Python 3.11 `src/` layout with a config-driven, terminal-first workflow.
- Keep the first implementation stage focused on environment setup, repo structure, data validation, metrics, and reproducibility utilities.
- Do not implement the full LPE-STGTN model until data preprocessing and baseline infrastructure are stable.
- Record all future paper deviations and assumption resolutions in this log.

### Scaffold decisions implemented in this session

- Added `AGENTS.md` as the repository contract for future coding agents.
- Added `pyproject.toml`, `Makefile`, config files under `configs/`, and a modular `src/lpe_stgtn/` layout.
- Added a raw-parquet inspection CLI and script for terminal-first sanity checks.
- Added metric helpers for MAE, RMSE, and MAPE.
- The repository implementation uses an explicit epsilon-clamped denominator for MAPE to avoid undefined behavior on zero-demand targets. This is an implementation default, not a confirmed paper detail.
- Created a local `.venv` and completed an editable install with dev dependencies.
- Verified the scaffold with `pytest`, `ruff check`, and a real-data inspection run against `yellow_tripdata_2018-01.parquet`.
