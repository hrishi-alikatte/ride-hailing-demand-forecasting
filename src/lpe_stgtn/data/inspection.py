"""Raw-parquet inspection utilities for sanity checks and schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lpe_stgtn.data.schema import YELLOW_TAXI_REQUIRED_COLUMNS


@dataclass(frozen=True)
class RawParquetSummary:
    """Concise metadata and schema summary for a raw parquet file."""

    path: Path
    num_rows: int
    num_row_groups: int
    columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    sample_rows: tuple[dict[str, Any], ...]

    @property
    def has_required_columns(self) -> bool:
        return not self.missing_required_columns


def summarize_parquet_file(path: Path, *, sample_rows: int = 3) -> RawParquetSummary:
    """Inspect a single parquet file without reading it fully into memory."""
    parquet = pq.ParquetFile(path)
    columns = tuple(parquet.schema.names)
    missing = tuple(sorted(set(YELLOW_TAXI_REQUIRED_COLUMNS) - set(columns)))

    rows: tuple[dict[str, Any], ...] = ()
    if sample_rows > 0:
        batch_iterator = parquet.iter_batches(batch_size=sample_rows)
        first_batch = next(batch_iterator, None)
        if first_batch is not None:
            rows = tuple(_batch_to_rows(first_batch.to_pydict()))

    metadata = parquet.metadata
    return RawParquetSummary(
        path=path,
        num_rows=metadata.num_rows,
        num_row_groups=metadata.num_row_groups,
        columns=columns,
        missing_required_columns=missing,
        sample_rows=rows,
    )


def summarize_dataset(
    *,
    data_dir: Path,
    pattern: str = "yellow_tripdata_2018-*.parquet",
    limit: int | None = None,
    sample_rows: int = 3,
) -> list[RawParquetSummary]:
    """Inspect a collection of parquet files from the data directory."""
    matched_paths = sorted(data_dir.glob(pattern))
    if limit is not None:
        matched_paths = matched_paths[:limit]
    return [summarize_parquet_file(path, sample_rows=sample_rows) for path in matched_paths]


def format_dataset_report(summaries: list[RawParquetSummary]) -> str:
    """Render a human-readable report for terminal use."""
    lines: list[str] = []
    for summary in summaries:
        lines.append(f"File: {summary.path.name}")
        lines.append(f"  Rows: {summary.num_rows:,}")
        lines.append(f"  Row groups: {summary.num_row_groups}")
        lines.append(f"  Required columns present: {summary.has_required_columns}")
        if summary.missing_required_columns:
            missing = ", ".join(summary.missing_required_columns)
            lines.append(f"  Missing required columns: {missing}")
        else:
            core_columns = ", ".join(YELLOW_TAXI_REQUIRED_COLUMNS)
            lines.append(f"  Core columns: {core_columns}")
        if summary.sample_rows:
            lines.append(f"  Sample rows shown: {len(summary.sample_rows)}")
            for index, row in enumerate(summary.sample_rows, start=1):
                lines.append(f"    [{index}] {row}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _batch_to_rows(columnar: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Convert a batch-level column mapping into row-wise dictionaries."""
    if not columnar:
        return []

    columns = list(columnar.keys())
    row_count = len(columnar[columns[0]])
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        rows.append({column: columnar[column][index] for column in columns})
    return rows
