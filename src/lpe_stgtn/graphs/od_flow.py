"""OD-flow graph builders for the NYC taxi demand study area."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpe_stgtn.data.preprocessing import infer_month_bounds_from_filename
from lpe_stgtn.graphs.normalize import normalize_adjacency_with_self_loop


@dataclass(frozen=True)
class OdFlowGraphArtifacts:
    """OD-flow graph tensors aligned to the processed zone order."""

    flow_counts: np.ndarray
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    flow_std: float
    epsilon_threshold: float
    monthly_stats: tuple[dict[str, Any], ...]


def build_od_flow_graph_from_files(
    raw_paths: list[Path],
    *,
    ordered_zone_ids: tuple[int, ...],
    pickup_time_column: str,
    origin_zone_column: str,
    destination_zone_column: str,
    min_timestamp: pd.Timestamp,
    max_timestamp: pd.Timestamp,
    epsilon_threshold: float = 0.0,
) -> OdFlowGraphArtifacts:
    """Aggregate OD flows across raw monthly parquet files and derive weighted adjacency."""
    zone_to_index = {location_id: index for index, location_id in enumerate(ordered_zone_ids)}
    flow_counts = np.zeros((len(ordered_zone_ids), len(ordered_zone_ids)), dtype=np.int64)
    monthly_stats: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        monthly_counts, stats = aggregate_od_counts_for_file(
            raw_path,
            zone_to_index=zone_to_index,
            pickup_time_column=pickup_time_column,
            origin_zone_column=origin_zone_column,
            destination_zone_column=destination_zone_column,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )
        flow_counts += monthly_counts
        monthly_stats.append(stats)

    adjacency, flow_std = od_counts_to_adjacency(
        flow_counts,
        epsilon_threshold=epsilon_threshold,
    )
    normalized = normalize_adjacency_with_self_loop(adjacency)
    return OdFlowGraphArtifacts(
        flow_counts=flow_counts,
        adjacency=adjacency,
        normalized_adjacency=normalized,
        flow_std=flow_std,
        epsilon_threshold=epsilon_threshold,
        monthly_stats=tuple(monthly_stats),
    )


def aggregate_od_counts_for_file(
    path: Path,
    *,
    zone_to_index: dict[int, int],
    pickup_time_column: str,
    origin_zone_column: str,
    destination_zone_column: str,
    min_timestamp: pd.Timestamp,
    max_timestamp: pd.Timestamp,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate one monthly parquet file into an OD count matrix."""
    frame = pd.read_parquet(
        path,
        columns=[pickup_time_column, origin_zone_column, destination_zone_column],
    )
    raw_row_count = int(len(frame))
    filtered = frame.dropna(
        subset=[pickup_time_column, origin_zone_column, destination_zone_column]
    ).copy()
    filtered[origin_zone_column] = filtered[origin_zone_column].astype(int)
    filtered[destination_zone_column] = filtered[destination_zone_column].astype(int)
    filtered[pickup_time_column] = pd.to_datetime(filtered[pickup_time_column])

    file_month_start, file_month_end = infer_month_bounds_from_filename(path)
    filtered = filtered.loc[
        (filtered[pickup_time_column] >= min_timestamp)
        & (filtered[pickup_time_column] < max_timestamp)
        & (filtered[pickup_time_column] >= file_month_start)
        & (filtered[pickup_time_column] < file_month_end)
        & (filtered[origin_zone_column].isin(zone_to_index))
        & (filtered[destination_zone_column].isin(zone_to_index))
    ]

    grouped = (
        filtered.groupby([origin_zone_column, destination_zone_column])
        .size()
        .rename("flow")
        .reset_index()
    )
    matrix = np.zeros((len(zone_to_index), len(zone_to_index)), dtype=np.int64)
    for row in grouped.itertuples(index=False):
        origin_index = zone_to_index[getattr(row, origin_zone_column)]
        destination_index = zone_to_index[getattr(row, destination_zone_column)]
        matrix[origin_index, destination_index] = int(row.flow)

    stats = {
        "file_name": path.name,
        "raw_row_count": raw_row_count,
        "study_area_trip_count": int(len(filtered)),
        "nonzero_od_pairs": int(np.count_nonzero(matrix)),
        "max_flow": int(matrix.max(initial=0)),
        "min_timestamp": (
            filtered[pickup_time_column].min().isoformat() if not filtered.empty else None
        ),
        "max_timestamp": (
            filtered[pickup_time_column].max().isoformat() if not filtered.empty else None
        ),
    }
    return matrix, stats


def od_counts_to_adjacency(
    flow_counts: np.ndarray,
    *,
    epsilon_threshold: float,
) -> tuple[np.ndarray, float]:
    """Convert raw OD counts into the paper's weighted OD adjacency."""
    counts = np.asarray(flow_counts, dtype=np.float64)
    off_diagonal_mask = ~np.eye(counts.shape[0], dtype=bool)
    flow_std = float(np.std(counts[off_diagonal_mask]))
    if flow_std <= 0:
        raise ValueError("Resolved non-positive OD flow standard deviation.")

    adjacency = np.square(counts / flow_std)
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < epsilon_threshold] = 0.0
    return adjacency.astype(np.float64), flow_std
