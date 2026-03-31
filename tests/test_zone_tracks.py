from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lpe_stgtn.analysis.zone_tracks import (
    build_zone_track_comparison,
    format_zone_track_comparison,
)


class ZoneTrackComparisonTests(unittest.TestCase):
    def test_build_zone_track_comparison(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary_dir = root / "primary"
            secondary_dir = root / "secondary"
            self._write_track(
                primary_dir,
                excluded_location_ids=[103],
                included_location_ids=[4, 104, 105],
                num_zones=3,
                total_pickups=100,
                sigma=10.0,
                max_distance=1000.0,
                max_snap_distance=400.0,
            )
            self._write_track(
                secondary_dir,
                excluded_location_ids=[103, 104],
                included_location_ids=[4, 105],
                num_zones=2,
                total_pickups=99,
                sigma=9.0,
                max_distance=900.0,
                max_snap_distance=25.0,
            )
            primary_persistence = self._write_report(
                root / "primary_persistence.json",
                validation_mae=1.0,
                validation_rmse=2.0,
                test_mae=3.0,
                test_rmse=4.0,
            )
            primary_lstm = self._write_report(
                root / "primary_lstm.json",
                validation_mae=0.5,
                validation_rmse=1.0,
                test_mae=1.5,
                test_rmse=2.5,
            )
            secondary_persistence = self._write_report(
                root / "secondary_persistence.json",
                validation_mae=0.9,
                validation_rmse=1.8,
                test_mae=2.7,
                test_rmse=3.6,
            )
            secondary_lstm = self._write_report(
                root / "secondary_lstm.json",
                validation_mae=0.4,
                validation_rmse=0.9,
                test_mae=1.2,
                test_rmse=2.2,
            )

            comparison = build_zone_track_comparison(
                primary_label="primary",
                primary_processed_data_dir=primary_dir,
                primary_persistence_report_path=primary_persistence,
                primary_lstm_report_path=primary_lstm,
                secondary_label="secondary",
                secondary_processed_data_dir=secondary_dir,
                secondary_persistence_report_path=secondary_persistence,
                secondary_lstm_report_path=secondary_lstm,
            )

        self.assertEqual(comparison.removed_zone_ids, (104,))
        self.assertEqual(comparison.added_zone_ids, ())
        self.assertEqual(comparison.total_pickups_delta, -1)
        self.assertAlmostEqual(comparison.max_snap_distance_delta, -375.0)
        self.assertAlmostEqual(comparison.lstm_test_mae_delta, -0.3)
        formatted = format_zone_track_comparison(comparison)
        self.assertIn("removed=[104]", formatted)
        self.assertIn("LSTM test MAE/RMSE", formatted)

    def _write_track(
        self,
        root_dir: Path,
        *,
        excluded_location_ids: list[int],
        included_location_ids: list[int],
        num_zones: int,
        total_pickups: int,
        sigma: float,
        max_distance: float,
        max_snap_distance: float,
    ) -> None:
        root_dir.mkdir(parents=True)
        metadata = {
            "num_zones": num_zones,
            "total_pickups": total_pickups,
            "study_area_summary": {
                "excluded_location_ids": excluded_location_ids,
                "included_location_ids": included_location_ids,
            },
        }
        (root_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        graph_dir = root_dir / "graphs"
        graph_dir.mkdir()
        graph_metadata = {
            "distance_graph": {
                "distance_method": "road_network_shortest_path",
                "snap_component_policy": "largest_connected_component",
                "sigma": sigma,
                "max_distance": max_distance,
            }
        }
        (graph_dir / "metadata.json").write_text(json.dumps(graph_metadata), encoding="utf-8")
        (graph_dir / "zone_road_network_snaps.csv").write_text(
            "location_id,snap_distance\n1," + str(max_snap_distance) + "\n",
            encoding="utf-8",
        )

    def _write_report(
        self,
        path: Path,
        *,
        validation_mae: float,
        validation_rmse: float,
        test_mae: float,
        test_rmse: float,
    ) -> Path:
        payload = {
            "metrics": {
                "validation": {"mae": validation_mae, "rmse": validation_rmse},
                "test": {"mae": test_mae, "rmse": test_rmse},
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
