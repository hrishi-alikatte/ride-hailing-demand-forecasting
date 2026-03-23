# Data Layout

This repository currently stores the raw NYC Yellow Taxi 2018 parquet files directly under `data/`.

Planned layout:

- `data/` for the raw monthly parquet files already present locally
- `data/external/` for small reference assets such as zone lookup tables and taxi-zone geometry files
- `data/interim/` for filtered or partially transformed data
- `data/processed/` for model-ready tensors, demand matrices, graph assets, and cached splits

Notes:

- Do not move or rewrite the raw parquet files unless there is a clear reason and the change is documented.
- Do not commit large downloaded datasets or generated artifacts.
- The repository now includes `data/external/taxi_zone_lookup.csv` and `data/external/taxi_zones/` as the NYC Yellow Taxi spatial reference assets.
- The lookup and geometry assets currently indicate 69 Manhattan TLC zones, while the paper reports 68 Manhattan zones, so the exact study-area filtering rule remains unresolved.
