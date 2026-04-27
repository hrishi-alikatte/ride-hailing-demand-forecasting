"""Utilities for loading and aligning historical weather covariates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeatherAlignmentResult:
    feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    source_path: Path
    timestamp_column: str
    original_frequency: str
    aligned_frequency_minutes: int
    row_count_before_alignment: int
    row_count_after_alignment: int


def load_and_align_weather_from_config(
    config: dict[str, Any],
    *,
    full_index: pd.DatetimeIndex,
    project_root: Path,
    time_interval_minutes: int,
) -> WeatherAlignmentResult:
    path = project_root / _require_string(config, "path")
    timestamp_column = _require_string(config, "timestamp_column")
    feature_columns = _require_string_list(config, "feature_columns")
    file_format = str(config.get("file_format", path.suffix.lower().lstrip('.'))).lower()
    timezone = str(config.get("timezone", "UTC"))
    aggregation = str(config.get("aggregation", "ffill")).lower()

    if file_format in {"csv", "txt"}:
        frame = pd.read_csv(path)
    elif file_format in {"parquet", "pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported weather file format: {file_format}")

    if timestamp_column not in frame.columns:
        raise ValueError(f"Weather timestamp column '{timestamp_column}' not found in {path}")
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Weather feature columns missing from {path}: {missing}")

    frame = frame[[timestamp_column, *feature_columns]].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    if timezone and timezone.upper() != 'UTC':
        frame[timestamp_column] = frame[timestamp_column].dt.tz_convert(timezone)
    frame[timestamp_column] = frame[timestamp_column].dt.tz_localize(None)
    frame = frame.sort_values(timestamp_column).drop_duplicates(subset=[timestamp_column], keep='last')
    original_frequency = _infer_frequency(frame[timestamp_column])

    frame = frame.set_index(timestamp_column)
    frame = frame.apply(pd.to_numeric, errors='coerce')

    # Aggregate duplicates and align to the model's full 15-minute timeline.
    if aggregation == 'mean':
        aligned = frame.resample(f'{time_interval_minutes}min').mean()
    elif aggregation == 'sum':
        aligned = frame.resample(f'{time_interval_minutes}min').sum()
    else:
        # Typical case for hourly weather: carry each observed hour forward until the next hour.
        aligned = frame.resample(f'{time_interval_minutes}min').ffill()

    aligned = aligned.reindex(full_index).ffill().bfill()
    if aligned.isna().any().any():
        raise ValueError('Weather alignment produced NaN values after fill operations.')

    return WeatherAlignmentResult(
        feature_matrix=aligned.to_numpy(dtype=np.float32),
        feature_names=tuple(feature_columns),
        source_path=path,
        timestamp_column=timestamp_column,
        original_frequency=original_frequency,
        aligned_frequency_minutes=time_interval_minutes,
        row_count_before_alignment=int(len(frame)),
        row_count_after_alignment=int(len(aligned)),
    )


def normalize_weather_features_train_only(
    weather_features: np.ndarray,
    *,
    train_step_start: int,
    train_step_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_slice = weather_features[train_step_start:train_step_end].astype(np.float32)
    mean = train_slice.mean(axis=0)
    std = train_slice.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    normalized = (weather_features.astype(np.float32) - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for weather config key '{key}'")
    return value


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"Expected non-empty string list for weather config key '{key}'")
    return value


def _infer_frequency(series: pd.Series) -> str:
    if len(series) < 3:
        return 'unknown'
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 'unknown'
    minutes = int(diffs.mode().iloc[0].total_seconds() // 60)
    return f'{minutes}min'
