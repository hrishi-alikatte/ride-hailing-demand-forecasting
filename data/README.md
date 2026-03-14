# Data Layout

This repository currently stores the raw NYC Yellow Taxi 2018 parquet files directly under `data/`.

Planned layout:

- `data/` for the raw monthly parquet files already present locally
- `data/external/` for small reference assets such as zone lookup tables
- `data/interim/` for filtered or partially transformed data
- `data/processed/` for model-ready tensors, demand matrices, graph assets, and cached splits

Notes:

- Do not move or rewrite the raw parquet files unless there is a clear reason and the change is documented.
- Do not commit large downloaded datasets or generated artifacts.
- Full Manhattan preprocessing will require a TLC zone lookup asset that is not currently present in the repository.
