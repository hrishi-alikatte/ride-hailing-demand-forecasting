"""Config-driven graph construction pipeline for stage 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpe_stgtn.config import load_yaml
from lpe_stgtn.data.datasets import load_processed_dataset
from lpe_stgtn.graphs.distance import (
    DistanceGraphArtifacts,
    build_distance_graph,
    build_road_network_distance_graph,
)
from lpe_stgtn.graphs.geometry_io import load_taxi_zone_geometries
from lpe_stgtn.graphs.od_flow import (
    OdFlowGraphArtifacts,
    build_od_flow_graph_from_files,
)
from lpe_stgtn.graphs.road_network import (
    build_road_network_graph,
    load_centerline_features,
    project_centerline_features,
)


@dataclass(frozen=True)
class GraphBuildSummary:
    """Top-level summary returned after graph construction completes."""

    output_dir: Path
    zone_ids: tuple[int, ...]
    distance_graph: DistanceGraphArtifacts
    od_flow_graph: OdFlowGraphArtifacts
    metadata_path: Path


def build_graph_artifacts_from_config(
    config_path: Path,
    *,
    project_root: Path,
) -> GraphBuildSummary:
    """Build distance and OD-flow graph artifacts from repository-local assets."""
    config = load_yaml(config_path)
    processed_data_dir = project_root / _require_string(config, "processed_data_dir")
    raw_glob = _require_string(config, "raw_glob")
    shapefile_path = project_root / _require_string(config, "taxi_zone_shapefile_path")
    output_subdir = _require_string(config, "output_subdir")
    pickup_time_column = _require_string(config, "pickup_time_column")
    origin_zone_column = _require_string(config, "origin_zone_column")
    destination_zone_column = _require_string(config, "destination_zone_column")

    distance_config = _require_mapping(config, "distance_graph")
    od_config = _require_mapping(config, "od_flow_graph")

    bundle = load_processed_dataset(processed_data_dir)
    zone_ids = tuple(int(value) for value in bundle.zone_ids.tolist())
    time_range_start = pd.Timestamp(_require_string(bundle.metadata, "time_range_start"))
    time_range_end = pd.Timestamp(_require_string(bundle.metadata, "time_range_end"))
    raw_paths = sorted((project_root / raw_glob).parent.glob(Path(raw_glob).name))
    if not raw_paths:
        raise FileNotFoundError(f"No raw parquet files matched {raw_glob}")

    geometries = load_taxi_zone_geometries(shapefile_path)

    sigma_value = distance_config.get("sigma")
    resolved_sigma = float(sigma_value) if isinstance(sigma_value, (int, float)) else None
    distance_method = _require_string(distance_config, "distance_method")
    if distance_method == "centroid_euclidean_epsg2263":
        distance_graph = build_distance_graph(
            geometries,
            ordered_zone_ids=zone_ids,
            sigma=resolved_sigma,
            epsilon_threshold=float(distance_config.get("epsilon_threshold", 0.0)),
        )
    elif distance_method == "road_network_shortest_path":
        road_network_relative_path = _require_string(
            distance_config,
            "road_network_geojson_path",
        )
        road_network_path = project_root / road_network_relative_path
        centerline_features = load_centerline_features(road_network_path)
        projected_centerline = project_centerline_features(
            centerline_features,
            source_crs=_require_string(distance_config, "road_network_source_crs"),
            target_crs=_require_string(distance_config, "road_network_target_crs"),
        )
        road_network = build_road_network_graph(projected_centerline)
        distance_graph = build_road_network_distance_graph(
            geometries,
            ordered_zone_ids=zone_ids,
            road_network=road_network,
            snap_component_policy=str(
                distance_config.get("snap_component_policy", "nearest_any_component")
            ),
            sigma=resolved_sigma,
            epsilon_threshold=float(distance_config.get("epsilon_threshold", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported distance graph method: {distance_method}")

    od_flow_graph = build_od_flow_graph_from_files(
        raw_paths,
        ordered_zone_ids=zone_ids,
        pickup_time_column=pickup_time_column,
        origin_zone_column=origin_zone_column,
        destination_zone_column=destination_zone_column,
        min_timestamp=time_range_start,
        max_timestamp=time_range_end + pd.Timedelta(minutes=15),
        epsilon_threshold=float(od_config.get("epsilon_threshold", 0.0)),
    )

    output_dir = project_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_graph_artifacts(
        output_dir,
        zone_ids=zone_ids,
        distance_graph=distance_graph,
        od_flow_graph=od_flow_graph,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_payload = {
        "processed_data_dir": str(processed_data_dir),
        "zone_ids": list(zone_ids),
        "num_zones": len(zone_ids),
        "distance_graph": {
            "distance_units": distance_graph.distance_units,
            "distance_method": distance_graph.distance_method,
            "sigma": distance_graph.sigma,
            "epsilon_threshold": distance_graph.epsilon_threshold,
            "nonzero_edges": int(np.count_nonzero(distance_graph.adjacency)),
            "max_weight": float(distance_graph.adjacency.max(initial=0.0)),
            "max_distance": float(distance_graph.distance_matrix.max(initial=0.0)),
            "snap_component_policy": distance_config.get(
                "snap_component_policy",
                "nearest_any_component",
            ),
            "road_network_geojson_path": distance_config.get("road_network_geojson_path"),
            "road_network_source_crs": distance_config.get("road_network_source_crs"),
            "road_network_target_crs": distance_config.get("road_network_target_crs"),
        },
        "od_flow_graph": {
            "epsilon_threshold": od_flow_graph.epsilon_threshold,
            "flow_std": od_flow_graph.flow_std,
            "nonzero_edges": int(np.count_nonzero(od_flow_graph.adjacency)),
            "max_weight": float(od_flow_graph.adjacency.max(initial=0.0)),
            "monthly_stats": list(od_flow_graph.monthly_stats),
        },
        "notes": config.get("notes", []),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return GraphBuildSummary(
        output_dir=output_dir,
        zone_ids=zone_ids,
        distance_graph=distance_graph,
        od_flow_graph=od_flow_graph,
        metadata_path=metadata_path,
    )


def write_graph_artifacts(
    output_dir: Path,
    *,
    zone_ids: tuple[int, ...],
    distance_graph: DistanceGraphArtifacts,
    od_flow_graph: OdFlowGraphArtifacts,
) -> None:
    """Persist graph arrays and supporting centroids metadata."""
    np.save(output_dir / "zone_ids.npy", np.asarray(zone_ids, dtype=np.int32))
    np.save(output_dir / "distance_matrix.npy", distance_graph.distance_matrix.astype(np.float32))
    np.save(output_dir / "distance_adjacency.npy", distance_graph.adjacency.astype(np.float32))
    np.save(
        output_dir / "distance_adjacency_normalized.npy",
        distance_graph.normalized_adjacency.astype(np.float32),
    )
    np.save(output_dir / "od_flow_counts.npy", od_flow_graph.flow_counts.astype(np.int64))
    np.save(output_dir / "od_flow_adjacency.npy", od_flow_graph.adjacency.astype(np.float32))
    np.save(
        output_dir / "od_flow_adjacency_normalized.npy",
        od_flow_graph.normalized_adjacency.astype(np.float32),
    )

    centroids_frame = pd.DataFrame(
        [
            {
                "location_id": centroid.location_id,
                "borough": centroid.borough,
                "zone": centroid.zone,
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
            }
            for centroid in distance_graph.centroids
        ]
    )
    centroids_frame.to_csv(output_dir / "zone_centroids.csv", index=False)
    if distance_graph.zone_snaps is not None:
        snaps_frame = pd.DataFrame(
            [
                {
                    "location_id": snap.location_id,
                    "borough": snap.borough,
                    "zone": snap.zone,
                    "centroid_x": snap.centroid_x,
                    "centroid_y": snap.centroid_y,
                    "snapped_node_key_x": snap.snapped_node_key[0],
                    "snapped_node_key_y": snap.snapped_node_key[1],
                    "snapped_x": snap.snapped_x,
                    "snapped_y": snap.snapped_y,
                    "snap_distance": snap.snap_distance,
                }
                for snap in distance_graph.zone_snaps
            ]
        )
        snaps_frame.to_csv(output_dir / "zone_road_network_snaps.csv", index=False)


def format_graph_build_summary(summary: GraphBuildSummary) -> str:
    """Render a terminal-friendly summary of the graph build outputs."""
    lines = [
        f"Output directory: {summary.output_dir}",
        f"Zone count: {len(summary.zone_ids)}",
        f"Distance graph sigma: {summary.distance_graph.sigma:.6f}",
        (
            "Distance graph nonzero edges: "
            f"{int(np.count_nonzero(summary.distance_graph.adjacency))}"
        ),
        f"OD flow graph std: {summary.od_flow_graph.flow_std:.6f}",
        (f"OD flow graph nonzero edges: {int(np.count_nonzero(summary.od_flow_graph.adjacency))}"),
        f"Wrote metadata: {summary.metadata_path}",
    ]
    return "\n".join(lines)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for config key '{key}'")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for config key '{key}'")
    return value
