"""Distance-graph builders for the NYC taxi-zone study area."""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from lpe_stgtn.graphs.geometry_io import (
    TaxiZoneGeometry,
    ZoneCentroid,
    compute_zone_centroid,
)
from lpe_stgtn.graphs.normalize import normalize_adjacency_with_self_loop
from lpe_stgtn.graphs.road_network import (
    RoadNetworkGraph,
    ZoneRoadNetworkSnap,
    snap_points_to_road_network,
)


@dataclass(frozen=True)
class DistanceGraphArtifacts:
    """Distance-derived graph tensors aligned to the processed zone order."""

    distance_method: str
    distance_units: str
    centroids: tuple[ZoneCentroid, ...]
    distance_matrix: np.ndarray
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    sigma: float
    epsilon_threshold: float
    zone_snaps: tuple[ZoneRoadNetworkSnap, ...] | None = None


def build_distance_graph(
    geometries: list[TaxiZoneGeometry],
    *,
    ordered_zone_ids: tuple[int, ...],
    sigma: float | None = None,
    epsilon_threshold: float = 0.0,
) -> DistanceGraphArtifacts:
    """Build the distance graph from centroid geometry in the shapefile CRS."""
    centroids = ordered_zone_centroids(geometries, ordered_zone_ids=ordered_zone_ids)
    coordinates = np.asarray(
        [(centroid.x, centroid.y) for centroid in centroids],
        dtype=np.float64,
    )
    distance_matrix = pairwise_euclidean_distances(coordinates)
    resolved_sigma = sigma if sigma is not None else resolve_distance_sigma(distance_matrix)
    adjacency = distance_matrix_to_adjacency(
        distance_matrix,
        sigma=resolved_sigma,
        epsilon_threshold=epsilon_threshold,
    )
    normalized = normalize_adjacency_with_self_loop(adjacency)
    return DistanceGraphArtifacts(
        distance_method="centroid_euclidean_epsg2263",
        distance_units="US survey foot in EPSG:2263 centroid coordinates",
        centroids=centroids,
        distance_matrix=distance_matrix,
        adjacency=adjacency,
        normalized_adjacency=normalized,
        sigma=resolved_sigma,
        epsilon_threshold=epsilon_threshold,
    )


def build_road_network_distance_graph(
    geometries: list[TaxiZoneGeometry],
    *,
    ordered_zone_ids: tuple[int, ...],
    road_network: RoadNetworkGraph,
    snap_component_policy: str = "nearest_any_component",
    sigma: float | None = None,
    epsilon_threshold: float = 0.0,
) -> DistanceGraphArtifacts:
    """Build the distance graph from shortest-path distances on the road network."""
    centroids = ordered_zone_centroids(geometries, ordered_zone_ids=ordered_zone_ids)
    coordinates = np.asarray(
        [(centroid.x, centroid.y) for centroid in centroids],
        dtype=np.float64,
    )
    snaps = snap_points_to_road_network(
        location_ids=tuple(centroid.location_id for centroid in centroids),
        boroughs=tuple(centroid.borough for centroid in centroids),
        zones=tuple(centroid.zone for centroid in centroids),
        point_coordinates=coordinates,
        road_network=road_network,
        component_policy=snap_component_policy,
    )
    distance_matrix = road_network_distance_matrix(snaps, road_network=road_network)
    resolved_sigma = sigma if sigma is not None else resolve_distance_sigma(distance_matrix)
    adjacency = distance_matrix_to_adjacency(
        distance_matrix,
        sigma=resolved_sigma,
        epsilon_threshold=epsilon_threshold,
    )
    normalized = normalize_adjacency_with_self_loop(adjacency)
    return DistanceGraphArtifacts(
        distance_method="road_network_shortest_path",
        distance_units="US survey foot in EPSG:2263 road-network shortest-path distance",
        centroids=centroids,
        distance_matrix=distance_matrix,
        adjacency=adjacency,
        normalized_adjacency=normalized,
        sigma=resolved_sigma,
        epsilon_threshold=epsilon_threshold,
        zone_snaps=snaps,
    )


def ordered_zone_centroids(
    geometries: list[TaxiZoneGeometry],
    *,
    ordered_zone_ids: tuple[int, ...],
) -> tuple[ZoneCentroid, ...]:
    """Load centroids in the exact processed dataset zone order."""
    geometry_map = {geometry.location_id: geometry for geometry in geometries}
    missing = sorted(set(ordered_zone_ids) - set(geometry_map))
    if missing:
        raise ValueError(f"Missing zone geometries for LocationID values: {missing}")
    return tuple(
        compute_zone_centroid(geometry_map[location_id]) for location_id in ordered_zone_ids
    )


def pairwise_euclidean_distances(coordinates: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for `(N, 2)` coordinates."""
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(np.square(deltas), axis=2))


def road_network_distance_matrix(
    snaps: tuple[ZoneRoadNetworkSnap, ...],
    *,
    road_network: RoadNetworkGraph,
) -> np.ndarray:
    """Compute all-pairs zone distances through the snapped road network."""
    count = len(snaps)
    distance_matrix = np.zeros((count, count), dtype=np.float64)
    source_lengths: list[dict[tuple[float, float], float]] = []
    for snap in snaps:
        lengths = nx.single_source_dijkstra_path_length(
            road_network.graph,
            snap.snapped_node_key,
            weight="weight",
        )
        source_lengths.append({key: float(value) for key, value in lengths.items()})

    for source_index, source_snap in enumerate(snaps):
        for target_index, target_snap in enumerate(snaps):
            if source_index == target_index:
                continue
            target_length = source_lengths[source_index].get(target_snap.snapped_node_key)
            if target_length is None:
                raise ValueError(
                    "Road-network graph does not connect all snapped Manhattan zones. "
                    f"Missing path from {source_snap.location_id} to {target_snap.location_id}."
                )
            distance_matrix[source_index, target_index] = (
                source_snap.snap_distance + target_length + target_snap.snap_distance
            )
    return distance_matrix


def resolve_distance_sigma(distance_matrix: np.ndarray) -> float:
    """Default sigma choice used when the paper does not provide one."""
    off_diagonal = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)]
    positive = off_diagonal[off_diagonal > 0]
    if positive.size == 0:
        raise ValueError("Cannot resolve sigma from an empty distance matrix.")
    sigma = float(np.std(positive))
    if sigma <= 0:
        raise ValueError("Resolved non-positive sigma from the distance matrix.")
    return sigma


def distance_matrix_to_adjacency(
    distance_matrix: np.ndarray,
    *,
    sigma: float,
    epsilon_threshold: float,
) -> np.ndarray:
    """Convert pairwise centroid distances into the paper's weighted distance adjacency."""
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError(f"Expected a positive finite sigma, got {sigma}.")
    adjacency = np.exp(-np.square(distance_matrix) / (sigma**2))
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < epsilon_threshold] = 0.0
    return adjacency.astype(np.float64)
