from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lpe_stgtn.graphs.distance import (
    build_road_network_distance_graph,
    distance_matrix_to_adjacency,
    pairwise_euclidean_distances,
    resolve_distance_sigma,
)
from lpe_stgtn.graphs.geometry_io import (
    PolygonShape,
    TaxiZoneGeometry,
    load_taxi_zone_geometries,
    polygon_centroid,
)
from lpe_stgtn.graphs.normalize import normalize_adjacency_with_self_loop
from lpe_stgtn.graphs.od_flow import od_counts_to_adjacency
from lpe_stgtn.graphs.road_network import (
    ProjectedCenterlineFeature,
    build_road_network_graph,
    load_centerline_features,
    snap_points_to_road_network,
)


class GraphTests(unittest.TestCase):
    def test_polygon_centroid_for_unit_square(self) -> None:
        centroid = polygon_centroid(
            (
                (
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                ),
            )
        )
        self.assertAlmostEqual(centroid[0], 0.5)
        self.assertAlmostEqual(centroid[1], 0.5)

    def test_pairwise_distance_and_adjacency_helpers(self) -> None:
        coordinates = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        distance_matrix = pairwise_euclidean_distances(coordinates)
        self.assertAlmostEqual(distance_matrix[0, 1], 5.0)
        sigma = resolve_distance_sigma(distance_matrix)
        self.assertGreater(sigma, 0.0)
        adjacency = distance_matrix_to_adjacency(
            distance_matrix,
            sigma=sigma,
            epsilon_threshold=0.1,
        )
        self.assertEqual(adjacency.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(adjacency), 0.0))

    def test_normalize_adjacency_with_self_loop(self) -> None:
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        normalized = normalize_adjacency_with_self_loop(adjacency)
        expected = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        np.testing.assert_allclose(normalized, expected)

    def test_od_counts_to_adjacency(self) -> None:
        flow_counts = np.array([[0, 2], [4, 0]], dtype=np.int64)
        adjacency, flow_std = od_counts_to_adjacency(flow_counts, epsilon_threshold=0.0)
        self.assertGreater(flow_std, 0.0)
        self.assertEqual(adjacency.shape, (2, 2))
        self.assertEqual(adjacency[0, 0], 0.0)
        self.assertGreater(adjacency[1, 0], adjacency[0, 1])

    def test_real_taxi_zone_asset_loads(self) -> None:
        geometries = load_taxi_zone_geometries(Path("data/external/taxi_zones/taxi_zones.shp"))
        manhattan = [geometry for geometry in geometries if geometry.borough == "Manhattan"]

        self.assertEqual(len(geometries), 263)
        self.assertEqual(len(manhattan), 69)
        self.assertIn(103, {geometry.location_id for geometry in manhattan})

    def test_load_centerline_features_from_geojson(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [[[0.0, 0.0], [1.0, 0.0]]],
                    },
                    "properties": {
                        "physicalid": "10",
                        "boroughcode": "1",
                        "trafdir": "TW",
                        "segmentlength": "1.0",
                    },
                }
            ],
        }
        with TemporaryDirectory() as temp_dir:
            geojson_path = Path(temp_dir) / "centerline.geojson"
            geojson_path.write_text(json.dumps(payload), encoding="utf-8")
            features = load_centerline_features(geojson_path)

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].physical_id, 10)
        self.assertEqual(features[0].borough_code, 1)
        self.assertEqual(features[0].traffic_direction, "TW")
        self.assertEqual(features[0].segment_length, 1.0)

    def test_build_road_network_distance_graph(self) -> None:
        features = (
            ProjectedCenterlineFeature(
                physical_id=1,
                borough_code=1,
                traffic_direction="TW",
                segment_length=5.0,
                parts=(((0.0, 0.0), (5.0, 0.0)),),
            ),
            ProjectedCenterlineFeature(
                physical_id=2,
                borough_code=1,
                traffic_direction="TW",
                segment_length=5.0,
                parts=(((5.0, 0.0), (5.0, 5.0)),),
            ),
        )
        road_network = build_road_network_graph(features)
        geometries = [
            TaxiZoneGeometry(
                location_id=1,
                borough="Manhattan",
                zone="Zone A",
                shape=PolygonShape(
                    shape_type=5,
                    parts=(
                        (
                            (-0.5, -0.5),
                            (0.5, -0.5),
                            (0.5, 0.5),
                            (-0.5, 0.5),
                            (-0.5, -0.5),
                        ),
                    ),
                    bbox=(-0.5, -0.5, 0.5, 0.5),
                ),
            ),
            TaxiZoneGeometry(
                location_id=2,
                borough="Manhattan",
                zone="Zone B",
                shape=PolygonShape(
                    shape_type=5,
                    parts=(
                        (
                            (4.5, 4.5),
                            (5.5, 4.5),
                            (5.5, 5.5),
                            (4.5, 5.5),
                            (4.5, 4.5),
                        ),
                    ),
                    bbox=(4.5, 4.5, 5.5, 5.5),
                ),
            ),
        ]

        artifacts = build_road_network_distance_graph(
            geometries,
            ordered_zone_ids=(1, 2),
            road_network=road_network,
            sigma=1.0,
            epsilon_threshold=0.0,
        )

        self.assertEqual(artifacts.distance_method, "road_network_shortest_path")
        self.assertIsNotNone(artifacts.zone_snaps)
        np.testing.assert_allclose(artifacts.distance_matrix, np.array([[0.0, 10.0], [10.0, 0.0]]))
        self.assertGreater(artifacts.adjacency[0, 1], 0.0)

    def test_snap_points_to_largest_connected_component(self) -> None:
        features = (
            ProjectedCenterlineFeature(
                physical_id=1,
                borough_code=1,
                traffic_direction="TW",
                segment_length=2.0,
                parts=(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),),
            ),
            ProjectedCenterlineFeature(
                physical_id=2,
                borough_code=1,
                traffic_direction="TW",
                segment_length=1.0,
                parts=(((10.0, 0.0), (11.0, 0.0)),),
            ),
        )
        road_network = build_road_network_graph(features)
        snaps = snap_points_to_road_network(
            location_ids=(1,),
            boroughs=("Manhattan",),
            zones=("Zone A",),
            point_coordinates=np.array([[10.2, 0.0]], dtype=np.float64),
            road_network=road_network,
            component_policy="largest_connected_component",
        )

        self.assertEqual(snaps[0].snapped_x, 2.0)
        self.assertEqual(snaps[0].snapped_y, 0.0)


if __name__ == "__main__":
    unittest.main()
