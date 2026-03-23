# Reproduction Log

## 2026-03-23 - External spatial assets organized for the NYC Yellow Taxi pipeline

### External assets now present

- Added `data/external/taxi_zone_lookup.csv` as the TLC zone lookup asset for mapping `LocationID` values to borough and zone names.
- Moved the taxi-zone shapefile bundle into `data/external/taxi_zones/` so the lookup and geometry assets live under the repository's external-data area.
- Verified that the lookup CSV has `265` rows and `69` rows with `Borough == "Manhattan"`.
- Verified that the taxi-zone shapefile has `263` geometries, CRS `EPSG:2263`, and fields including `LocationID`, `zone`, `borough`, and `geometry`.
- Verified that the taxi-zone shapefile also contains `69` Manhattan geometries.

### What this unblocks

- No additional NYC Yellow Taxi trip files are needed for the NYC-only reproduction track.
- The repository now has the minimum external assets needed to begin Manhattan filtering, derive OD-flow statistics from the raw trip table, and construct a geometry-based distance graph.

### Remaining ambiguity to resolve before paper-faithful preprocessing

- The paper reports `68` Manhattan zones, but the current TLC lookup and geometry assets both expose `69` Manhattan zones.
- Before claiming paper-faithful preprocessing, the exact exclusion rule or zone-set definition used by the paper must be identified and recorded.
- The paper describes the distance graph in terms of proximity in the real traffic network, but does not specify the exact distance source or thresholds. A centroid- or polygon-based zone distance derived from the TLC geometry is a defensible default if no road-network asset is introduced, but this should be documented as an implementation choice if used.

### Strongest current hypothesis for the 68-zone study area

- The strongest single-zone exclusion candidate is `LocationID 103`, labeled `Governor's Island/Ellis Island/Liberty Island`.
- Rationale:
  - Removing `103` from the current TLC-derived Manhattan set reduces `69` Manhattan zones to the paper-reported `68`.
  - In the full 2018 NYC Yellow Taxi raw data, zone `103` has `0` pickups and `0` dropoffs.
  - Under a stricter Manhattan-only filter where both pickup and dropoff endpoints are restricted to Manhattan TLC zones, zone `103` still has `0` pickups and `0` dropoffs.
  - The Manhattan set also contains `LocationID` values `104` and `105` with the same zone label, but these still show nonzero 2018 activity.
- This is currently a defensible implementation hypothesis, not a confirmed paper fact.

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
