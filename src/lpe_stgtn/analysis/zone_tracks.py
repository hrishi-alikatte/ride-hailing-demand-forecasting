"""Comparison helpers for parallel Manhattan study-area tracks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineMetricSummary:
    """Validation and test metrics for one baseline model."""

    validation_mae: float
    validation_rmse: float
    test_mae: float
    test_rmse: float


@dataclass(frozen=True)
class ZoneTrackSummary:
    """One processed Manhattan study-area track plus its baseline results."""

    label: str
    processed_data_dir: Path
    graph_dir: Path
    report_paths: dict[str, str]
    num_zones: int
    total_pickups: int
    excluded_location_ids: tuple[int, ...]
    included_location_ids: tuple[int, ...]
    graph_distance_method: str
    graph_snap_component_policy: str | None
    graph_sigma: float
    graph_max_distance: float
    max_snap_distance: float | None
    persistence: BaselineMetricSummary
    lstm: BaselineMetricSummary


@dataclass(frozen=True)
class ZoneTrackComparison:
    """Side-by-side comparison of the mainline and sensitivity study-area tracks."""

    primary: ZoneTrackSummary
    secondary: ZoneTrackSummary
    removed_zone_ids: tuple[int, ...]
    added_zone_ids: tuple[int, ...]
    total_pickups_delta: int
    max_snap_distance_delta: float | None
    persistence_test_mae_delta: float
    persistence_test_rmse_delta: float
    lstm_test_mae_delta: float
    lstm_test_rmse_delta: float


def build_zone_track_comparison(
    *,
    primary_label: str,
    primary_processed_data_dir: Path,
    primary_persistence_report_path: Path,
    primary_lstm_report_path: Path,
    secondary_label: str,
    secondary_processed_data_dir: Path,
    secondary_persistence_report_path: Path,
    secondary_lstm_report_path: Path,
) -> ZoneTrackComparison:
    """Load two study-area tracks and compute their comparison summary."""
    primary = load_zone_track_summary(
        label=primary_label,
        processed_data_dir=primary_processed_data_dir,
        persistence_report_path=primary_persistence_report_path,
        lstm_report_path=primary_lstm_report_path,
    )
    secondary = load_zone_track_summary(
        label=secondary_label,
        processed_data_dir=secondary_processed_data_dir,
        persistence_report_path=secondary_persistence_report_path,
        lstm_report_path=secondary_lstm_report_path,
    )

    removed_zone_ids = tuple(
        sorted(set(primary.included_location_ids) - set(secondary.included_location_ids))
    )
    added_zone_ids = tuple(
        sorted(set(secondary.included_location_ids) - set(primary.included_location_ids))
    )
    primary_snap = primary.max_snap_distance
    secondary_snap = secondary.max_snap_distance
    max_snap_distance_delta = None
    if primary_snap is not None and secondary_snap is not None:
        max_snap_distance_delta = secondary_snap - primary_snap

    return ZoneTrackComparison(
        primary=primary,
        secondary=secondary,
        removed_zone_ids=removed_zone_ids,
        added_zone_ids=added_zone_ids,
        total_pickups_delta=secondary.total_pickups - primary.total_pickups,
        max_snap_distance_delta=max_snap_distance_delta,
        persistence_test_mae_delta=secondary.persistence.test_mae - primary.persistence.test_mae,
        persistence_test_rmse_delta=secondary.persistence.test_rmse - primary.persistence.test_rmse,
        lstm_test_mae_delta=secondary.lstm.test_mae - primary.lstm.test_mae,
        lstm_test_rmse_delta=secondary.lstm.test_rmse - primary.lstm.test_rmse,
    )


def load_zone_track_summary(
    *,
    label: str,
    processed_data_dir: Path,
    persistence_report_path: Path,
    lstm_report_path: Path,
) -> ZoneTrackSummary:
    """Load one processed track and the corresponding baseline reports."""
    metadata = _load_json(processed_data_dir / "metadata.json")
    graph_dir = processed_data_dir / "graphs"
    graph_metadata = _load_json(graph_dir / "metadata.json")
    max_snap_distance = load_max_snap_distance(graph_dir / "zone_road_network_snaps.csv")
    persistence_report = _load_json(persistence_report_path)
    lstm_report = _load_json(lstm_report_path)

    study_area_summary = metadata["study_area_summary"]
    return ZoneTrackSummary(
        label=label,
        processed_data_dir=processed_data_dir,
        graph_dir=graph_dir,
        report_paths={
            "persistence": str(persistence_report_path),
            "lstm": str(lstm_report_path),
        },
        num_zones=int(metadata["num_zones"]),
        total_pickups=int(metadata["total_pickups"]),
        excluded_location_ids=tuple(
            int(value) for value in study_area_summary["excluded_location_ids"]
        ),
        included_location_ids=tuple(
            int(value) for value in study_area_summary["included_location_ids"]
        ),
        graph_distance_method=str(graph_metadata["distance_graph"]["distance_method"]),
        graph_snap_component_policy=_maybe_string(
            graph_metadata["distance_graph"].get("snap_component_policy")
        ),
        graph_sigma=float(graph_metadata["distance_graph"]["sigma"]),
        graph_max_distance=float(graph_metadata["distance_graph"]["max_distance"]),
        max_snap_distance=max_snap_distance,
        persistence=_baseline_metric_summary(persistence_report),
        lstm=_baseline_metric_summary(lstm_report),
    )


def format_zone_track_comparison(comparison: ZoneTrackComparison) -> str:
    """Render a terminal-friendly summary for the 68-vs-67 track comparison."""
    primary = comparison.primary
    secondary = comparison.secondary
    lines = [
        (
            f"Primary track: {primary.label} "
            f"({primary.num_zones} zones, excluded {list(primary.excluded_location_ids)})"
        ),
        (
            f"Secondary track: {secondary.label} "
            f"({secondary.num_zones} zones, excluded {list(secondary.excluded_location_ids)})"
        ),
        (
            "Zone-set delta: "
            f"removed={list(comparison.removed_zone_ids)} "
            f"added={list(comparison.added_zone_ids)}"
        ),
        (
            "Total pickups delta: "
            f"{comparison.total_pickups_delta:+d} "
            f"({primary.total_pickups} -> {secondary.total_pickups})"
        ),
        (
            "Graph max snap distance: "
            f"{_format_optional(primary.max_snap_distance)} -> "
            f"{_format_optional(secondary.max_snap_distance)}"
        ),
        (
            "Persistence test MAE/RMSE: "
            f"{primary.persistence.test_mae:.6f}/{primary.persistence.test_rmse:.6f} -> "
            f"{secondary.persistence.test_mae:.6f}/{secondary.persistence.test_rmse:.6f}"
        ),
        (
            "LSTM test MAE/RMSE: "
            f"{primary.lstm.test_mae:.6f}/{primary.lstm.test_rmse:.6f} -> "
            f"{secondary.lstm.test_mae:.6f}/{secondary.lstm.test_rmse:.6f}"
        ),
        (
            "LSTM test delta: "
            f"MAE {comparison.lstm_test_mae_delta:+.6f}, "
            f"RMSE {comparison.lstm_test_rmse_delta:+.6f}"
        ),
    ]
    return "\n".join(lines)


def zone_track_comparison_to_dict(comparison: ZoneTrackComparison) -> dict[str, Any]:
    """Convert the comparison dataclass tree into a JSON-serializable payload."""
    return _jsonify(asdict(comparison))


def load_max_snap_distance(path: Path) -> float | None:
    """Return the maximum snap distance if the graph build wrote snap metadata."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if "snap_distance" not in header:
            return None
        snap_distance_index = header.index("snap_distance")
        max_value: float | None = None
        for line in handle:
            values = line.strip().split(",")
            if len(values) <= snap_distance_index:
                continue
            value = float(values[snap_distance_index])
            max_value = value if max_value is None else max(max_value, value)
    return max_value


def _baseline_metric_summary(report: dict[str, Any]) -> BaselineMetricSummary:
    validation = report["metrics"]["validation"]
    test = report["metrics"]["test"]
    return BaselineMetricSummary(
        validation_mae=float(validation["mae"]),
        validation_rmse=float(validation["rmse"]),
        test_mae=float(test["mae"]),
        test_rmse=float(test["rmse"]),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison input artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _maybe_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    return value
