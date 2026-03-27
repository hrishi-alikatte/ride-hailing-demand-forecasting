"""Config-driven preprocessing for the NYC Yellow Taxi 2018 demand dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpe_stgtn.config import load_yaml
from lpe_stgtn.data.taxi_zones import StudyAreaSummary, summarize_study_area


@dataclass(frozen=True)
class SplitSummary:
    """Step- and sample-level boundaries for one chronological split."""

    name: str
    step_start: int
    step_end: int
    sample_start: int
    sample_end: int

    @property
    def num_steps(self) -> int:
        return self.step_end - self.step_start

    @property
    def num_samples(self) -> int:
        return self.sample_end - self.sample_start


@dataclass(frozen=True)
class PreparedDatasetSummary:
    """Top-level summary returned after preparing the demand dataset."""

    dataset_name: str
    output_dir: Path
    demand_matrix_path: Path
    arrays_path: Path
    metadata_path: Path
    study_area_summary: StudyAreaSummary
    num_time_steps: int
    total_pickups: int
    normalization_mean: float
    normalization_std: float
    train_split: SplitSummary
    validation_split: SplitSummary
    test_split: SplitSummary
    monthly_stats: tuple[dict[str, Any], ...]


def prepare_dataset_from_config(config_path: Path, *, project_root: Path) -> PreparedDatasetSummary:
    """Build processed demand artifacts from the configured raw parquet files."""
    config = load_yaml(config_path)
    dataset_name = _require_string(config, "dataset_name")
    raw_glob = _require_string(config, "raw_glob")
    pickup_time_column = _require_string(config, "pickup_time_column")
    origin_zone_column = _require_string(config, "origin_zone_column")
    study_area = _require_string(config, "study_area")
    time_interval_minutes = _require_int(config, "time_interval_minutes")
    history_steps = _require_int(config, "history_steps")
    forecast_steps = _require_int(config, "forecast_steps")
    expected_zone_count = _require_int(config, "manhattan_zone_count_from_paper")
    train_ratio = _require_float(config, "train_ratio")
    validation_ratio = _require_float(config, "validation_ratio")
    test_ratio = _require_float(config, "test_ratio")
    demand_definition = _require_string(config, "demand_definition")
    study_area_filter_mode = _require_string(config, "study_area_filter_mode")
    output_subdir = _require_string(config, "processed_output_subdir")
    lookup_path = project_root / _require_string(config, "taxi_zone_lookup_path")
    interval_start = pd.Timestamp(_require_string(config, "time_range_start"))
    interval_end = pd.Timestamp(_require_string(config, "time_range_end"))

    if demand_definition != "pickup_counts":
        raise ValueError(f"Unsupported demand_definition: {demand_definition}")
    if study_area_filter_mode != "pickup_only":
        raise ValueError(f"Unsupported study_area_filter_mode: {study_area_filter_mode}")

    _validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    excluded_location_ids = tuple(
        int(value) for value in config.get("study_area_excluded_location_ids", [])
    )
    study_area_summary = summarize_study_area(
        lookup_path,
        borough=study_area,
        expected_zone_count=expected_zone_count,
        excluded_location_ids=excluded_location_ids,
    )
    zone_ids = study_area_summary.included_location_ids
    if not zone_ids:
        raise ValueError("Configured study area produced zero included zones.")

    raw_paths = sorted((project_root / raw_glob).parent.glob(Path(raw_glob).name))
    if not raw_paths:
        raise FileNotFoundError(f"No raw parquet files matched {raw_glob}")

    monthly_counts: list[pd.DataFrame] = []
    monthly_stats: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        counts, stats = aggregate_pickup_counts_for_file(
            raw_path,
            pickup_time_column=pickup_time_column,
            origin_zone_column=origin_zone_column,
            interval_minutes=time_interval_minutes,
            allowed_zone_ids=zone_ids,
            min_timestamp=interval_start,
            max_timestamp=interval_end + pd.Timedelta(minutes=time_interval_minutes),
        )
        monthly_counts.append(counts)
        monthly_stats.append(stats)

    full_index = build_full_time_index(
        start=interval_start,
        end=interval_end,
        interval_minutes=time_interval_minutes,
    )
    demand_matrix = build_demand_matrix(monthly_counts, full_index=full_index, zone_ids=zone_ids)
    demand_values = demand_matrix.to_numpy(dtype=np.int32)
    time_of_day, day_of_week = build_temporal_indices(demand_matrix.index, time_interval_minutes)

    train_split, validation_split, test_split = build_split_summaries(
        total_steps=demand_matrix.shape[0],
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        history_steps=history_steps,
        forecast_steps=forecast_steps,
    )

    train_values = demand_values[train_split.step_start : train_split.step_end].astype(np.float32)
    normalization_mean = float(train_values.mean())
    normalization_std = float(train_values.std())
    if normalization_std == 0.0:
        normalization_std = 1.0
    normalized_demand = (demand_values.astype(np.float32) - normalization_mean) / normalization_std
    total_pickups = int(demand_values.astype(np.int64).sum())

    output_dir = project_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    demand_matrix_path = output_dir / "demand_matrix.parquet"
    arrays_path = output_dir / "dataset_arrays.npz"
    metadata_path = output_dir / "metadata.json"

    save_demand_matrix(
        demand_matrix,
        path=demand_matrix_path,
    )
    np.savez_compressed(
        arrays_path,
        demand=demand_values,
        normalized_demand=normalized_demand.astype(np.float32),
        zone_ids=np.asarray(zone_ids, dtype=np.int32),
        time_of_day=time_of_day,
        day_of_week=day_of_week,
        train_sample_start_indices=np.arange(
            train_split.sample_start,
            train_split.sample_end,
            dtype=np.int32,
        ),
        validation_sample_start_indices=np.arange(
            validation_split.sample_start,
            validation_split.sample_end,
            dtype=np.int32,
        ),
        test_sample_start_indices=np.arange(
            test_split.sample_start, test_split.sample_end, dtype=np.int32
        ),
    )

    metadata_payload = {
        "dataset_name": dataset_name,
        "raw_glob": raw_glob,
        "study_area": study_area,
        "demand_definition": demand_definition,
        "study_area_filter_mode": study_area_filter_mode,
        "study_area_definition_status": config.get("study_area_definition_status"),
        "study_area_definition_notes": config.get("study_area_definition_notes", []),
        "time_interval_minutes": time_interval_minutes,
        "history_steps": history_steps,
        "forecast_steps": forecast_steps,
        "time_range_start": interval_start.isoformat(),
        "time_range_end": interval_end.isoformat(),
        "zone_ids": list(zone_ids),
        "num_zones": len(zone_ids),
        "num_time_steps": int(demand_matrix.shape[0]),
        "total_pickups": total_pickups,
        "normalization": {
            "method": "zscore_train_scalar",
            "train_mean": normalization_mean,
            "train_std": normalization_std,
        },
        "splits": {
            "train": split_summary_to_dict(train_split),
            "validation": split_summary_to_dict(validation_split),
            "test": split_summary_to_dict(test_split),
        },
        "study_area_summary": {
            "lookup_path": str(study_area_summary.lookup_path),
            "borough": study_area_summary.borough,
            "available_zone_count": study_area_summary.available_zone_count,
            "included_zone_count": study_area_summary.included_zone_count,
            "expected_zone_count": study_area_summary.expected_zone_count,
            "matches_expected_zone_count": study_area_summary.matches_expected_zone_count,
            "excluded_location_ids": list(study_area_summary.excluded_location_ids),
            "included_location_ids": list(study_area_summary.included_location_ids),
        },
        "monthly_stats": monthly_stats,
        "output_files": {
            "demand_matrix_parquet": str(demand_matrix_path),
            "dataset_arrays_npz": str(arrays_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return PreparedDatasetSummary(
        dataset_name=dataset_name,
        output_dir=output_dir,
        demand_matrix_path=demand_matrix_path,
        arrays_path=arrays_path,
        metadata_path=metadata_path,
        study_area_summary=study_area_summary,
        num_time_steps=int(demand_matrix.shape[0]),
        total_pickups=total_pickups,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        monthly_stats=tuple(monthly_stats),
    )


def aggregate_pickup_counts_for_file(
    path: Path,
    *,
    pickup_time_column: str,
    origin_zone_column: str,
    interval_minutes: int,
    allowed_zone_ids: tuple[int, ...],
    min_timestamp: pd.Timestamp,
    max_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate one raw parquet file into `(timestamp, location_id, demand)` counts."""
    allowed_zone_set = set(allowed_zone_ids)
    frame = pd.read_parquet(path, columns=[pickup_time_column, origin_zone_column])
    raw_row_count = int(len(frame))

    filtered = frame.dropna(subset=[pickup_time_column, origin_zone_column]).copy()
    filtered[origin_zone_column] = filtered[origin_zone_column].astype(int)
    filtered = filtered.loc[filtered[origin_zone_column].isin(allowed_zone_set)]
    filtered[pickup_time_column] = pd.to_datetime(filtered[pickup_time_column])

    file_month_start, file_month_end = infer_month_bounds_from_filename(path)
    filtered = filtered.loc[
        (filtered[pickup_time_column] >= min_timestamp)
        & (filtered[pickup_time_column] < max_timestamp)
        & (filtered[pickup_time_column] >= file_month_start)
        & (filtered[pickup_time_column] < file_month_end)
    ]
    filtered["timestamp"] = filtered[pickup_time_column].dt.floor(f"{interval_minutes}min")

    counts = (
        filtered.groupby(["timestamp", origin_zone_column])
        .size()
        .rename("demand")
        .reset_index()
        .rename(columns={origin_zone_column: "location_id"})
        .sort_values(["timestamp", "location_id"], kind="stable")
        .reset_index(drop=True)
    )

    stats = {
        "file_name": path.name,
        "raw_row_count": raw_row_count,
        "study_area_pickup_count": int(len(filtered)),
        "aggregated_row_count": int(len(counts)),
        "active_zone_count": int(filtered[origin_zone_column].nunique()),
        "min_timestamp": counts["timestamp"].min().isoformat() if not counts.empty else None,
        "max_timestamp": counts["timestamp"].max().isoformat() if not counts.empty else None,
    }
    return counts, stats


def infer_month_bounds_from_filename(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the valid month interval from a TLC monthly parquet file name."""
    match = re.search(r"(\d{4})-(\d{2})", path.name)
    if match is None:
        raise ValueError(f"Could not infer year-month from file name: {path.name}")

    month_start = pd.Timestamp(f"{match.group(1)}-{match.group(2)}-01T00:00:00")
    month_end = month_start + pd.offsets.MonthBegin(1)
    return month_start, month_end


def build_full_time_index(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_minutes: int,
) -> pd.DatetimeIndex:
    """Create a complete, inclusive time index for the configured study period."""
    if end < start:
        raise ValueError("time_range_end must be greater than or equal to time_range_start")
    return pd.date_range(start=start, end=end, freq=f"{interval_minutes}min")


def build_demand_matrix(
    monthly_counts: list[pd.DataFrame],
    *,
    full_index: pd.DatetimeIndex,
    zone_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Combine monthly counts into a dense time-by-zone matrix."""
    if monthly_counts:
        combined = pd.concat(monthly_counts, ignore_index=True)
        aggregated = (
            combined.groupby(["timestamp", "location_id"], as_index=False)["demand"]
            .sum()
            .sort_values(["timestamp", "location_id"], kind="stable")
        )
        pivoted = aggregated.pivot(index="timestamp", columns="location_id", values="demand")
    else:
        pivoted = pd.DataFrame(index=full_index)

    matrix = (
        pivoted.reindex(index=full_index, columns=list(zone_ids), fill_value=0)
        .fillna(0)
        .astype(np.int32)
    )
    matrix.index.name = "timestamp"
    return matrix


def build_temporal_indices(
    index: pd.DatetimeIndex,
    interval_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time-of-day slot indices and day-of-week indices for the full timeline."""
    time_of_day = ((index.hour * 60 + index.minute) // interval_minutes).astype(np.int16)
    day_of_week = index.dayofweek.astype(np.int16)
    return time_of_day.to_numpy(dtype=np.int16), day_of_week.to_numpy(dtype=np.int16)


def build_split_summaries(
    *,
    total_steps: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    history_steps: int,
    forecast_steps: int,
) -> tuple[SplitSummary, SplitSummary, SplitSummary]:
    """Create chronological split boundaries for both step and sample spaces."""
    _validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    train_end = int(total_steps * train_ratio)
    validation_end = train_end + int(total_steps * validation_ratio)
    test_end = total_steps

    split_specs = (
        ("train", 0, train_end),
        ("validation", train_end, validation_end),
        ("test", validation_end, test_end),
    )

    split_summaries: list[SplitSummary] = []
    for name, step_start, step_end in split_specs:
        num_samples = max(0, step_end - step_start - history_steps - forecast_steps + 1)
        sample_start = step_start
        sample_end = sample_start + num_samples
        split_summaries.append(
            SplitSummary(
                name=name,
                step_start=step_start,
                step_end=step_end,
                sample_start=sample_start,
                sample_end=sample_end,
            )
        )
    return tuple(split_summaries)  # type: ignore[return-value]


def build_supervised_windows(
    demand: np.ndarray,
    *,
    sample_start_indices: np.ndarray,
    history_steps: int,
    forecast_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize model-ready `(x, y)` windows from a full demand matrix."""
    x_windows = np.stack(
        [demand[start : start + history_steps] for start in sample_start_indices],
        axis=0,
    )
    y_windows = np.stack(
        [
            demand[start + history_steps : start + history_steps + forecast_steps]
            for start in sample_start_indices
        ],
        axis=0,
    )
    return x_windows, y_windows


def format_prepared_dataset_summary(summary: PreparedDatasetSummary) -> str:
    """Render a terminal-friendly summary after preprocessing completes."""
    excluded_ids = ", ".join(
        str(value) for value in summary.study_area_summary.excluded_location_ids
    )
    lines = [
        f"Dataset: {summary.dataset_name}",
        f"Output directory: {summary.output_dir}",
        f"Included Manhattan zones: {summary.study_area_summary.included_zone_count}",
        f"Excluded Manhattan location IDs: {excluded_ids or 'none'}",
        f"Demand time steps: {summary.num_time_steps}",
        f"Total study-area pickups: {summary.total_pickups:,}",
        (
            "Train split: "
            f"{summary.train_split.num_steps} steps, {summary.train_split.num_samples} samples"
        ),
        (
            "Validation split: "
            f"{summary.validation_split.num_steps} steps, "
            f"{summary.validation_split.num_samples} samples"
        ),
        (
            "Test split: "
            f"{summary.test_split.num_steps} steps, {summary.test_split.num_samples} samples"
        ),
        f"Train normalization mean: {summary.normalization_mean:.6f}",
        f"Train normalization std: {summary.normalization_std:.6f}",
        f"Wrote demand matrix: {summary.demand_matrix_path}",
        f"Wrote dataset arrays: {summary.arrays_path}",
        f"Wrote metadata: {summary.metadata_path}",
    ]
    if summary.study_area_summary.matches_expected_zone_count is False:
        lines.append("Warning: included zone count does not match the expected paper count.")
    return "\n".join(lines)


def save_demand_matrix(matrix: pd.DataFrame, *, path: Path) -> None:
    """Persist the dense demand matrix in a human-inspectable parquet file."""
    renamed = matrix.copy()
    renamed.columns = [f"zone_{column}" for column in renamed.columns]
    renamed.reset_index().to_parquet(path, index=False)


def split_summary_to_dict(split: SplitSummary) -> dict[str, int | str]:
    """Convert split metadata into JSON-serializable form."""
    return {
        "name": split.name,
        "step_start": split.step_start,
        "step_end": split.step_end,
        "num_steps": split.num_steps,
        "sample_start": split.sample_start,
        "sample_end": split.sample_end,
        "num_samples": split.num_samples,
    }


def _validate_split_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for config key '{key}'")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected int for config key '{key}'")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected float for config key '{key}'")
    return float(value)
