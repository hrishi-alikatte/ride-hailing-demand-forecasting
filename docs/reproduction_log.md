# Reproduction Log

## 2026-03-27 - Stage-4 first graph-based baseline implemented

### What was implemented

- Added `src/lpe_stgtn/graphs/artifacts.py` to load saved stage-3 graph artifacts in a way that stays aligned to the processed dataset zone order.
- Added reusable stage-4 model components:
  - `src/lpe_stgtn/models/components/gcn.py`
  - `src/lpe_stgtn/models/components/attention.py`
- Added `src/lpe_stgtn/models/baselines/dual_graph_gru.py`, a first graph-based baseline that uses:
  - distance graph convolution
  - OD-flow graph convolution
  - lightweight semantic fusion attention across the two graph views
  - a per-zone GRU temporal encoder
- Extended `src/lpe_stgtn/training/baselines.py` so the existing baseline runner can train graph-based baselines with graph-artifact validation, graph-aware reports, and checkpoints.
- Added the stage-4 experiment config `configs/experiments/nyc_dual_graph_gru_baseline_sensitivity_67.yaml`.

### Why the semantic fusion was implemented this way

- The paper's global module uses multi-head attention to fuse semantic representations from the distance graph and OD graph.
- A literal token-level multi-head attention across all zones was too slow for practical CPU experimentation in the current repository environment.
- The implemented stage-4 baseline therefore uses a lightweight head-wise semantic attention across the **two graph views per node**, which preserves the dual-semantic fusion idea while staying trainable on CPU.
- This is a stage-4 baseline implementation choice, not a claim of full paper-faithful global-module reproduction.

### First completed stage-4 run

- Experiment: `nyc_dual_graph_gru_baseline_sensitivity_67_quickcpu`
- Track: 67-zone sensitivity dataset with official road-network distance graph
- Config choices:
  - graph hidden dim `16`
  - temporal hidden dim `32`
  - batch size `1024`
  - max epochs `2`
- Result:
  - validation: `MAE 17.496307`, `RMSE 30.099709`
  - test: `MAE 18.281325`, `RMSE 31.913867`

### Interpretation

- Stage 4 is now **implemented** as code and has completed a real graph-based run.
- The first completed CPU-budget benchmark is **not competitive** with the 67-zone LSTM baseline (`test MAE 6.560578`, `RMSE 11.924477`).
- This underperformance should not be overinterpreted:
  - the stage-4 graph baseline is still intentionally simpler than the paper's full global module
  - the completed benchmark used a deliberately tiny CPU training budget to keep the experiment practical in-session
- The next engineering move is to tune this stage-4 baseline more seriously or move onward to the full LPE-STGTN implementation once the graph-baseline path is considered sufficiently exercised.

## 2026-03-27 - Added a 67-zone sensitivity track and compared it against the 68-zone mainline

### What was implemented

- Added a parallel stage-1 config, `configs/data/nyc_yellow_2018_sensitivity_67.yaml`, that excludes both `103` and `104`.
- Added matching stage-3 and baseline configs for the 67-zone sensitivity track.
- Added `lpe-stgtn compare-zone-tracks` to summarize the main 68-zone track against the 67-zone sensitivity track from processed-data metadata, graph metadata, and baseline reports.

### Real comparison results

- Mainline track:
  - `68` zones
  - excluded `103`
  - total pickups: `93,162,565`
  - maximum road-network snap distance: `3859.50` feet
  - persistence test: `MAE 12.563956`, `RMSE 23.411007`
  - LSTM test: `MAE 6.654453`, `RMSE 12.291936`
- Sensitivity track:
  - `67` zones
  - excluded `103` and `104`
  - total pickups: `93,162,564`
  - maximum road-network snap distance: `469.33` feet
  - persistence test: `MAE 12.751478`, `RMSE 23.585068`
  - LSTM test: `MAE 6.560578`, `RMSE 11.924477`

### Interpretation

- Excluding `104` removes only `1` pickup from the full 2018 demand total.
- The 67-zone sensitivity track is much cleaner geometrically because it eliminates the `104` snap outlier from the road-network graph.
- The persistence baseline becomes slightly worse on the 67-zone track, but the LSTM baseline becomes modestly better on the test split.
- This makes the 67-zone track a strong practical development candidate for graph-based modeling, even though it is no longer aligned with the paper-reported Manhattan zone count of `68`.

## 2026-03-27 - Focused investigation of Manhattan zones 103, 104, and 105

### What was checked

- Verified the official TLC lookup rows for `103`, `104`, and `105`, all labeled `Governor's Island/Ellis Island/Liberty Island`.
- Recounted full-year 2018 Yellow Taxi pickup and dropoff activity for those three `LocationID` values across all 12 monthly parquet files.
- Re-checked their polygon geometry size and road-network snapping behavior against the official citywide NYC Open Data `Centerline` graph.

### Evidence collected

- `103`:
  - `0` pickups and `0` dropoffs in the full 2018 Yellow Taxi data.
  - Snaps to a disconnected road-network component, not the main connected citywide component.
- `104`:
  - `1` pickup and `1` dropoff in the full 2018 Yellow Taxi data, both from one short self-trip in May 2018 (`104 -> 104`).
  - Snaps to a disconnected road-network component when nearest-node snapping is unconstrained.
  - Under the current default `largest_connected_component` snap policy, it becomes the largest snap-distance outlier in the study area at about `3859.5` feet.
- `105`:
  - `135` pickups and `92` dropoffs in the full 2018 Yellow Taxi data.
  - Snaps directly into the main connected citywide road-network component with a small snap distance of about `13.7` feet.

### Recommendation from this investigation

- Do **not** replace `104` with `103` in the current provisional `68`-zone set.
- `103` remains the strongest single-zone exclusion candidate because it has zero 2018 activity and is also road-network disconnected.
- `104` is still a real problem for road-network distance construction, but swapping it out for `103` would make the study area worse, not better.
- The best current interpretation is:
  - keep `exclude 103` as the least-bad provisional `68`-zone rule for the mainline pipeline
  - keep the `104` road-network workaround explicit
  - continue to treat the exact paper-faithful Manhattan `68`-zone definition as unresolved

## 2026-03-27 - Stage-3 distance graph upgraded to the official NYC road network

### What was implemented

- Added an official-road-network downloader in `src/lpe_stgtn/graphs/road_network.py` and a new CLI entrypoint, `lpe-stgtn download-road-network`, to fetch the NYC Open Data `Centerline` dataset (`inkn-q76z`) into `data/external/nyc_centerline/`.
- Added a GeoJSON-based centerline parser, CRS projection step, road-network graph builder, nearest-node snapping logic, and shortest-path distance builder for the stage-3 distance graph.
- Added `pyproj` and explicit `networkx` dependency declarations so the road-network graph path runs inside the repository `.venv`, not just in ad hoc system environments.
- Updated the default graph config so stage 3 now uses `road_network_shortest_path` instead of the earlier centroid-Euclidean approximation.

### Important implementation decisions

- The road-network graph now uses the **citywide** official centerline asset rather than a borough-code-`1`-only subset. A Manhattan-only road subset left `LocationID 104` disconnected from the rest of the study area because some shortest paths for Manhattan zones rely on bridge segments outside borough code `1`.
- Zone centroids are still computed from the TLC taxi-zone polygons, projected into `EPSG:2263`, and then snapped onto the official road network before shortest-path distances are computed.
- The current default snap policy is `largest_connected_component`. This is explicit and config-driven because `LocationID 104` snaps onto a tiny disconnected island component in the official centerline graph under the current provisional 68-zone rule.

### Why the disconnected-zone workaround is documented explicitly

- Under the current stage-1 study area, `LocationID 104` is still included because only `103` was provisionally excluded to match the paper's reported `68` Manhattan zones.
- In the official citywide centerline graph, `104` is not road-connected to the main Manhattan component, while `105` is.
- Restricting snapping to the largest connected road component keeps the stage-1 `68`-zone tensor shape intact, but it is still a documented implementation workaround rather than a confirmed paper detail.
- This strengthens the case that the exact paper-faithful Manhattan zone set is still unresolved and may not be equivalent to the current `exclude 103 only` hypothesis.

## 2026-03-27 - Stage-3 graph construction pipeline implemented

### What was implemented

- Added a pure-Python taxi-zone shapefile/DBF reader in `src/lpe_stgtn/graphs/geometry_io.py` so graph construction does not depend on external GIS libraries in the current repository environment.
- Added a distance-graph builder in `src/lpe_stgtn/graphs/distance.py` that:
  - computes zone centroids from the taxi-zone polygons
  - derives pairwise centroid distances in the taxi-zone shapefile CRS
  - converts distances to weighted adjacency using the paper's exponential kernel form
  - saves a normalized adjacency using `I + D^{-1/2} A D^{-1/2}`
- Added an OD-flow graph builder in `src/lpe_stgtn/graphs/od_flow.py` that:
  - scans the raw monthly parquet files
  - filters trips whose pickup and dropoff endpoints are both inside the current stage-1 study area
  - aggregates full-year OD counts
  - converts counts to weighted adjacency using the paper's `(flow / flow_std)^2` form
- Added a config-driven graph pipeline in `src/lpe_stgtn/graphs/pipeline.py`, a `build-graphs` CLI command, and a default graph config under `configs/graphs/nyc_manhattan_default.yaml`.

### Stage-3 implementation choices

- The paper describes the distance graph using real traffic-network proximity, but the repository currently approximates this with centroid Euclidean distance from the TLC taxi-zone polygons because no road-network asset is present locally.
- The paper does not specify sigma and graph sparsification thresholds clearly. The default stage-3 build therefore:
  - resolves sigma from the off-diagonal centroid-distance standard deviation
  - leaves both graph thresholds at `0.0` rather than imposing undocumented sparsity
- The graph outputs are aligned to the zone order already fixed by the stage-1 processed dataset so later model code can consume them directly without hidden reindexing.

## 2026-03-27 - Stage-2 baseline infrastructure added

### What was implemented

- Added a processed-dataset loader in `src/lpe_stgtn/data/datasets.py` for reading the stage-1 `.npz` and `metadata.json` artifacts and exposing lazy split-specific window datasets.
- Added two baseline models:
  - `PersistenceBaseline`, which repeats the last observed timestep across the forecast horizon
  - `LSTMBaseline`, a simple demand-only recurrent encoder that predicts the full 12-step horizon at once
- Added a config-driven baseline runner in `src/lpe_stgtn/training/baselines.py` with support for:
  - persistence evaluation on validation and test splits
  - LSTM training with Adam, L1 loss, early stopping, report writing, and checkpoint saving
- Added baseline experiment configs for NYC persistence and NYC LSTM runs.
- Reports and checkpoints are separated into their respective `artifacts/reports/` and `artifacts/checkpoints/` areas.

### Why this stage was structured this way

- The persistence baseline gives a zero-training sanity floor that can catch dataset leakage or metric mistakes quickly.
- The LSTM baseline provides the first trainable benchmark on top of the processed demand tensors before introducing graph construction or the full paper architecture.
- The runner is intentionally config-driven and artifact-writing so future graph and paper-model experiments can reuse the same processed data contract and reporting pattern.

## 2026-03-27 - Stage-1 NYC preprocessing pipeline implemented

### What was implemented

- Added a config-driven preprocessing pipeline in `src/lpe_stgtn/data/preprocessing.py` to read the 12 monthly NYC Yellow Taxi parquet files, filter the configured Manhattan study area, aggregate pickup demand at 15-minute resolution, concatenate the full 2018 timeline, and write processed artifacts.
- Added a `prepare-data` CLI command and `make prepare-data` target to build the stage-1 dataset reproducibly from repository-local inputs.
- The stage-1 outputs now include:
  - a dense demand matrix parquet file
  - a compressed NumPy archive with raw demand, normalized demand, zone IDs, temporal indices, and split sample indices
  - JSON metadata with split boundaries, normalization statistics, study-area details, and monthly aggregation stats
- Added preprocessing tests covering config-driven preparation and supervised window extraction.

### Stage-1 implementation choices

- The current stage-1 default defines demand as pickup counts by `PULocationID` and pickup timestamp, aggregated into 15-minute bins.
- The current stage-1 default uses chronological `60/20/20` step splits and computes a single scalar Z-score mean/std from the training split, following the paper formula as closely as possible from the available detail.
- The default config now excludes `LocationID 103` as a provisional Manhattan 68-zone hypothesis so the processed study area matches the paper-reported zone count.
- The monthly raw parquet files are clipped to the filename month during aggregation because the local raw extract contains some pickup timestamps that fall outside their nominal month, and counting those rows would risk silent cross-month leakage or double counting.

### Why the 103 exclusion is still documented as provisional

- The paper states that Manhattan is divided into `68` TLC zones, but the current repository lookup asset exposes `69` Manhattan rows.
- In the local 2018 raw data, `LocationID 103` has `0` pickups and `0` dropoffs across the full year, which makes it the strongest current exclusion candidate.
- However, this remains an implementation hypothesis rather than a confirmed paper fact, so it is kept explicit in config and metadata rather than hidden inside code.

## 2026-03-27 - Lookup-backed Manhattan study-area inspection added

### What was implemented

- Added a lookup-backed study-area utility in `src/lpe_stgtn/data/taxi_zones.py` for loading `data/external/taxi_zone_lookup.csv` and summarizing borough-specific zone sets.
- Added a new CLI command, `lpe-stgtn inspect-study-area`, to report the current borough zone count, any explicitly excluded `LocationID` values, and whether the configured count matches the paper-derived expectation.
- Added `study_area_excluded_location_ids` to `configs/data/nyc_yellow_2018.yaml` so any eventual 68-zone exclusion rule stays config-driven and reviewable instead of being hardcoded invisibly.
- Added tests covering lookup parsing, explicit zone exclusion behavior, and the current Manhattan count mismatch from the repository lookup asset.

### Why this change was staged this way

- The repository currently has no geospatial reader dependency such as `geopandas`, `fiona`, `shapely`, or `pyogrio`, so this session intentionally stopped at the lookup-backed study-area layer instead of partially implementing shapefile ingestion.
- This keeps the next preprocessing stage unblocked for Manhattan zone filtering from the trip table while preserving the unresolved `69`-vs-`68` ambiguity as an explicit configuration and documentation concern.

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

## 2026-03-31 - Final LPE-STGTN GPU Implementation

### What was implemented

- Updated `src/lpe_stgtn/data/datasets.py` to optionally return `Time-of-Day` and `Day-of-Week` metadata as inputs alongside demand tensors.
- Created `src/lpe_stgtn/models/components/embedding.py` (`SpatioTemporalEmbedding`) to combine demand with temporal features globally across all nodes.
- Built a standard scaled dot-product temporal `DynamicGraphGenerator` under `src/lpe_stgtn/models/components/dynamic_graph.py` to represent the paper's spatial-temporal pattern pool. 
- Implemented `AttentionFreeTransformerLocal` in `src/lpe_stgtn/models/components/aft_local.py` simulating standard chronologically-bounded AFT over the local chronological window size of 4 without standard O(T^2) cost.
- Expanded the true token-wise self-attention in `src/lpe_stgtn/models/components/attention.py` (`GlobalMultiHeadAttention`) to properly fuse semantic features extracted from Distance and OD Graphs.
- Created the final integrated PyTorch `LPE_STGTN` class and a dedicated Python training runner (`lpe_stgtn_runner.py`) handling multi-input ingestion and ensuring explicit `.to(device)` mapping for seamless GPU orchestration on external infrastructure.
- Exposed `make run-full-model` locally to test the architecture integrity on CPU before pushing to Git.

### Why this addresses paper ambiguities

- Several complex math structures for the exact parameter pool of spatial-time dynamically evolving graphs and the strict exact bounds of AFT-local windowing were under-specified. 
- A scalable implementation for the attention-free transformer, symmetric sliding windows, and deterministic tensor concatenation provides an architecturally robust, research-defensible path forward.

## 2026-04-24 - Final Stage: MAPE Fix & Hyperparameter Sweep

### MAPE metric fix

The MAPE evaluation was producing meaningless values (36761%, 40771%) because:
- `epsilon=1e-6` in the denominator allowed denormalized near-zero pickup counts (e.g. 0.001) to produce individual MAPE contributions exceeding 500%.
- The zero-mask (`null_val=0.0`) correctly filters exact zeros, but floating-point denormalization artifacts pass through.

**Fix applied**: Changed `epsilon` default from `1e-6` to `1.0` and added `× 100` to return MAPE as a percentage. This matches the convention used by DCRNN, STGCN, ASTGCN, GWNet, and the major traffic forecasting literature. All previous MAPE numbers are now invalid; MAE and RMSE remain unaffected.

### Intelligence from collaborator's repository

Analysis of the collaborator's experiment configs revealed:
- `hidden_dim=128` was systematically tested (double the paper's 64)
- Dropout values 0.0, 0.1, 0.15, 0.2 were explored
- Learning rates 0.001, 0.0005, 0.0003 were tested
- Batch sizes 16, 32, 64 were used
- Graph epsilon 0.012 was used for `paper_strict` graphs
- A `paper_strict` dataset using `pickup_and_dropoff` filtering was created

### Hyperparameter sweep design (4-hour GPU budget)

Five experiments selected to form a 2×2 factorial on (hidden_dim, dropout) plus one interpolation:

| Run | hidden_dim | dropout | LR | batch_size | Rationale |
|-----|-----------|---------|------|-----------|-----------|
| run_baseline | 64 | 0.1 | 0.001 | 16 | Reproduce lost best run |
| run_h128_d01 | 128 | 0.1 | 0.001 | 16 | Isolate hidden_dim effect |
| run_h128_d015 | 128 | 0.15 | 0.0005 | 32 | Interpolated dropout |
| run_h128_d02 | 128 | 0.2 | 0.0005 | 32 | Collaborator's most-tested combo |
| run_h64_d02 | 64 | 0.2 | 0.001 | 16 | Isolate dropout effect |

All on 67-zone track, existing dense graphs (ε=0.0), max_epochs=100, patience=15.
