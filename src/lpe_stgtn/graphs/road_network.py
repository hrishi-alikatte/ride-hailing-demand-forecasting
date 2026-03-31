"""Road-network asset helpers for stage-3 Manhattan distance graphs."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np

OFFICIAL_CENTERLINE_DATASET_ID = "inkn-q76z"
OFFICIAL_CENTERLINE_DATASET_NAME = "Centerline"
OFFICIAL_CENTERLINE_BASE_URL = "https://data.cityofnewyork.us/resource/inkn-q76z"
DEFAULT_CENTERLINE_SELECT = "the_geom,physicalid,trafdir,segmentlength,boroughcode"


@dataclass(frozen=True)
class CenterlineFeature:
    """One official NYC centerline feature in its source CRS coordinates."""

    physical_id: int | None
    borough_code: int | None
    traffic_direction: str | None
    segment_length: float | None
    parts: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class ProjectedCenterlineFeature:
    """One centerline feature projected into the target CRS."""

    physical_id: int | None
    borough_code: int | None
    traffic_direction: str | None
    segment_length: float | None
    parts: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class RoadNetworkGraph:
    """Weighted undirected road graph plus deterministic node coordinates."""

    graph: nx.Graph
    node_coordinates: dict[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class ZoneRoadNetworkSnap:
    """One zone centroid snapped onto the road-network graph."""

    location_id: int
    borough: str
    zone: str
    centroid_x: float
    centroid_y: float
    snapped_node_key: tuple[float, float]
    snapped_x: float
    snapped_y: float
    snap_distance: float


@dataclass(frozen=True)
class RoadNetworkDownloadSummary:
    """Download summary for the official centerline subset."""

    output_path: Path
    metadata_path: Path
    borough_code: int | None
    feature_count: int
    request_url: str


def format_road_network_download_summary(summary: RoadNetworkDownloadSummary) -> str:
    """Render a terminal-friendly summary for the downloaded road-network asset."""
    borough_label = (
        f"borough code {summary.borough_code}"
        if summary.borough_code is not None
        else "citywide"
    )
    lines = [
        f"Road-network asset: {summary.output_path}",
        f"Subset: {borough_label}",
        f"Feature count: {summary.feature_count}",
        f"Request URL: {summary.request_url}",
        f"Wrote metadata: {summary.metadata_path}",
    ]
    return "\n".join(lines)


def download_centerline_geojson(
    output_path: Path,
    *,
    borough_code: int | None = None,
    timeout_seconds: float = 120.0,
) -> RoadNetworkDownloadSummary:
    """Download the official NYC centerline dataset or one borough subset."""
    feature_count = query_centerline_feature_count(
        borough_code=borough_code,
        timeout_seconds=timeout_seconds,
    )
    request_url = build_centerline_geojson_url(
        borough_code=borough_code,
        limit=feature_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": "lpe-stgtn-reproduction/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    output_path.write_bytes(payload)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "source": "NYC Open Data",
        "dataset_id": OFFICIAL_CENTERLINE_DATASET_ID,
        "dataset_name": OFFICIAL_CENTERLINE_DATASET_NAME,
        "borough_code": borough_code,
        "feature_count": feature_count,
        "request_url": request_url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return RoadNetworkDownloadSummary(
        output_path=output_path,
        metadata_path=metadata_path,
        borough_code=borough_code,
        feature_count=feature_count,
        request_url=request_url,
    )


def download_manhattan_centerline_geojson(
    output_path: Path,
    *,
    borough_code: int = 1,
    timeout_seconds: float = 120.0,
) -> RoadNetworkDownloadSummary:
    """Backward-compatible wrapper for the borough-code-1 subset."""
    return download_centerline_geojson(
        output_path,
        borough_code=borough_code,
        timeout_seconds=timeout_seconds,
    )


def query_centerline_feature_count(
    *,
    borough_code: int | None,
    timeout_seconds: float = 30.0,
) -> int:
    """Query the official dataset for the feature count of one borough subset."""
    params = {"$select": "count(*)"}
    if borough_code is not None:
        params["$where"] = f"boroughcode='{borough_code}'"
    url = f"{OFFICIAL_CENTERLINE_BASE_URL}.json?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "lpe-stgtn-reproduction/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload or "count" not in payload[0]:
        raise ValueError(f"Unexpected centerline count response from {url}")
    return int(payload[0]["count"])


def build_centerline_geojson_url(
    *,
    borough_code: int | None,
    limit: int,
) -> str:
    """Build the official GeoJSON export URL for one borough subset."""
    params = {
        "$select": DEFAULT_CENTERLINE_SELECT,
        "$order": "physicalid",
        "$limit": str(limit),
    }
    if borough_code is not None:
        params["$where"] = f"boroughcode='{borough_code}'"
    return f"{OFFICIAL_CENTERLINE_BASE_URL}.geojson?{urllib.parse.urlencode(params)}"


def load_centerline_features(geojson_path: Path) -> tuple[CenterlineFeature, ...]:
    """Load centerline features from a downloaded GeoJSON file."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"Missing road-network GeoJSON asset: {geojson_path}")
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON file does not contain a feature list: {geojson_path}")

    records: list[CenterlineFeature] = []
    for feature in features:
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "MultiLineString":
            continue
        parts = _parse_multiline_coordinates(geometry.get("coordinates"))
        if not parts:
            continue
        properties = feature.get("properties", {})
        records.append(
            CenterlineFeature(
                physical_id=_maybe_int(properties.get("physicalid")),
                borough_code=_maybe_int(properties.get("boroughcode")),
                traffic_direction=_maybe_string(properties.get("trafdir")),
                segment_length=_maybe_float(properties.get("segmentlength")),
                parts=parts,
            )
        )
    return tuple(records)


def project_centerline_features(
    features: tuple[CenterlineFeature, ...],
    *,
    source_crs: str,
    target_crs: str,
) -> tuple[ProjectedCenterlineFeature, ...]:
    """Project centerline coordinates into the target CRS."""
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - guarded by dependency declaration.
        raise ImportError(
            "Road-network distance graphs require pyproj. Install project dependencies first."
        ) from exc

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected: list[ProjectedCenterlineFeature] = []
    for feature in features:
        projected_parts: list[tuple[tuple[float, float], ...]] = []
        for part in feature.parts:
            xs, ys = transformer.transform(
                [coordinate[0] for coordinate in part],
                [coordinate[1] for coordinate in part],
            )
            projected_parts.append(tuple(zip(xs, ys, strict=True)))
        projected.append(
            ProjectedCenterlineFeature(
                physical_id=feature.physical_id,
                borough_code=feature.borough_code,
                traffic_direction=feature.traffic_direction,
                segment_length=feature.segment_length,
                parts=tuple(projected_parts),
            )
        )
    return tuple(projected)


def build_road_network_graph(
    features: tuple[ProjectedCenterlineFeature, ...],
) -> RoadNetworkGraph:
    """Build an undirected weighted road graph from projected centerline segments."""
    graph = nx.Graph()
    node_coordinates: dict[tuple[float, float], tuple[float, float]] = {}

    for feature in features:
        for part in feature.parts:
            if len(part) < 2:
                continue
            keys = tuple(_coordinate_key(x, y) for x, y in part)
            for key, coordinate in zip(keys, part, strict=True):
                node_coordinates.setdefault(key, coordinate)
                if key not in graph:
                    graph.add_node(key)

            chord_lengths = [
                math.dist(part[index], part[index + 1])
                for index in range(len(part) - 1)
            ]
            total_chord_length = sum(chord_lengths)
            if total_chord_length <= 0.0:
                continue
            feature_length = (
                feature.segment_length
                if feature.segment_length is not None and feature.segment_length > 0.0
                else total_chord_length
            )
            scale = feature_length / total_chord_length

            for index, chord_length in enumerate(chord_lengths):
                if chord_length <= 0.0:
                    continue
                start_key = keys[index]
                end_key = keys[index + 1]
                weight = chord_length * scale
                if graph.has_edge(start_key, end_key):
                    existing_weight = float(graph[start_key][end_key]["weight"])
                    if weight < existing_weight:
                        graph[start_key][end_key]["weight"] = weight
                    continue
                graph.add_edge(start_key, end_key, weight=weight)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Road-network graph is empty after parsing the centerline features.")

    return RoadNetworkGraph(graph=graph, node_coordinates=node_coordinates)


def snap_points_to_road_network(
    *,
    location_ids: tuple[int, ...],
    boroughs: tuple[str, ...],
    zones: tuple[str, ...],
    point_coordinates: np.ndarray,
    road_network: RoadNetworkGraph,
    component_policy: str = "nearest_any_component",
) -> tuple[ZoneRoadNetworkSnap, ...]:
    """Snap point coordinates to the nearest road-network node."""
    if point_coordinates.ndim != 2 or point_coordinates.shape[1] != 2:
        raise ValueError("Expected point coordinates with shape (N, 2).")

    if component_policy == "nearest_any_component":
        node_keys = tuple(road_network.node_coordinates)
    elif component_policy == "largest_connected_component":
        largest_component = max(nx.connected_components(road_network.graph), key=len)
        node_keys = tuple(
            key for key in road_network.node_coordinates if key in largest_component
        )
    else:
        raise ValueError(f"Unsupported road-network component policy: {component_policy}")

    node_coordinates = np.asarray(
        [road_network.node_coordinates[key] for key in node_keys],
        dtype=np.float64,
    )

    snaps: list[ZoneRoadNetworkSnap] = []
    for index, (location_id, borough, zone) in enumerate(
        zip(location_ids, boroughs, zones, strict=True)
    ):
        point = point_coordinates[index]
        deltas = node_coordinates - point
        squared_distances = np.sum(np.square(deltas), axis=1)
        nearest_index = int(np.argmin(squared_distances))
        snapped_key = node_keys[nearest_index]
        snapped_x, snapped_y = road_network.node_coordinates[snapped_key]
        snap_distance = float(math.sqrt(float(squared_distances[nearest_index])))
        snaps.append(
            ZoneRoadNetworkSnap(
                location_id=location_id,
                borough=borough,
                zone=zone,
                centroid_x=float(point[0]),
                centroid_y=float(point[1]),
                snapped_node_key=snapped_key,
                snapped_x=snapped_x,
                snapped_y=snapped_y,
                snap_distance=snap_distance,
            )
        )
    return tuple(snaps)


def _parse_multiline_coordinates(
    raw_coordinates: object,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(raw_coordinates, list):
        return ()
    parts: list[tuple[tuple[float, float], ...]] = []
    for raw_part in raw_coordinates:
        if not isinstance(raw_part, list):
            continue
        coordinates: list[tuple[float, float]] = []
        for raw_coordinate in raw_part:
            if (
                isinstance(raw_coordinate, list)
                and len(raw_coordinate) >= 2
                and isinstance(raw_coordinate[0], (int, float))
                and isinstance(raw_coordinate[1], (int, float))
            ):
                coordinates.append((float(raw_coordinate[0]), float(raw_coordinate[1])))
        if coordinates:
            parts.append(tuple(coordinates))
    return tuple(parts)


def _coordinate_key(x: float, y: float, *, precision: int = 12) -> tuple[float, float]:
    return (round(x, precision), round(y, precision))


def _maybe_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _maybe_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _maybe_string(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)
