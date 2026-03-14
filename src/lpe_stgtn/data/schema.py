"""Dataset schema and paper-derived data constants."""

from __future__ import annotations

YELLOW_TAXI_REQUIRED_COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
)

PAPER_TIME_INTERVAL_MINUTES = 15
PAPER_HISTORY_STEPS = 12
PAPER_FORECAST_STEPS = 12
