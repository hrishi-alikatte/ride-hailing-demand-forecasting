from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lpe_stgtn.data.preprocessing import (
    aggregate_pickup_counts_for_file,
    build_supervised_windows,
    prepare_dataset_from_config,
)


class PreprocessingTests(unittest.TestCase):
    def test_prepare_dataset_from_config_builds_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "data" / "external").mkdir(parents=True)
            (root / "data" / "processed").mkdir(parents=True)
            (root / "configs" / "data").mkdir(parents=True)

            lookup_path = root / "data" / "external" / "taxi_zone_lookup.csv"
            lookup_path.write_text(
                "\n".join(
                    [
                        "LocationID,Borough,Zone,service_zone",
                        "1,Manhattan,Alpha,Yellow Zone",
                        "2,Manhattan,Beta,Yellow Zone",
                        "103,Manhattan,Gamma,Yellow Zone",
                        "4,Brooklyn,Delta,Boro Zone",
                    ]
                ),
                encoding="utf-8",
            )

            january = pd.DataFrame(
                {
                    "tpep_pickup_datetime": [
                        "2018-01-01 00:01:00",
                        "2018-01-01 00:14:00",
                        "2018-01-01 00:16:00",
                        "2018-01-01 00:35:00",
                    ],
                    "PULocationID": [1, 2, 1, 103],
                }
            )
            (root / "data" / "yellow_tripdata_2018-01.parquet").parent.mkdir(
                parents=True, exist_ok=True
            )
            january.to_parquet(root / "data" / "yellow_tripdata_2018-01.parquet")

            config_path = root / "configs" / "data" / "nyc_yellow_2018.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_name: sample_dataset",
                        "raw_glob: data/yellow_tripdata_2018-*.parquet",
                        "taxi_zone_lookup_path: data/external/taxi_zone_lookup.csv",
                        "pickup_time_column: tpep_pickup_datetime",
                        "dropoff_time_column: tpep_dropoff_datetime",
                        "origin_zone_column: PULocationID",
                        "destination_zone_column: DOLocationID",
                        "demand_definition: pickup_counts",
                        "study_area_filter_mode: pickup_only",
                        "time_interval_minutes: 15",
                        "study_area: Manhattan",
                        "spatial_partition: TLC taxi zones",
                        "manhattan_zone_count_from_paper: 2",
                        "study_area_definition_status: provisional",
                        "study_area_excluded_location_ids:",
                        "  - 103",
                        'time_range_start: "2018-01-01T00:00:00"',
                        'time_range_end: "2018-01-01T00:45:00"',
                        "processed_output_subdir: data/processed/sample_dataset",
                        "normalization: zscore_train_only",
                        "train_ratio: 0.5",
                        "validation_ratio: 0.25",
                        "test_ratio: 0.25",
                        "history_steps: 1",
                        "forecast_steps: 1",
                    ]
                ),
                encoding="utf-8",
            )

            summary = prepare_dataset_from_config(config_path, project_root=root)

            self.assertEqual(summary.study_area_summary.included_location_ids, (1, 2))
            self.assertEqual(summary.num_time_steps, 4)
            self.assertEqual(summary.total_pickups, 3)
            self.assertTrue(summary.demand_matrix_path.exists())
            self.assertTrue(summary.arrays_path.exists())
            self.assertTrue(summary.metadata_path.exists())

            arrays = np.load(summary.arrays_path)
            np.testing.assert_array_equal(arrays["zone_ids"], np.array([1, 2], dtype=np.int32))
            np.testing.assert_array_equal(
                arrays["demand"],
                np.array(
                    [
                        [1.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=np.int32,
                ),
            )

            metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["splits"]["train"]["num_steps"], 2)
            self.assertEqual(metadata["splits"]["train"]["num_samples"], 1)

    def test_build_supervised_windows_returns_expected_shapes(self) -> None:
        demand = np.arange(12, dtype=np.float32).reshape(6, 2)
        sample_start_indices = np.array([0, 1, 2], dtype=np.int32)

        x_windows, y_windows = build_supervised_windows(
            demand,
            sample_start_indices=sample_start_indices,
            history_steps=2,
            forecast_steps=1,
        )

        self.assertEqual(x_windows.shape, (3, 2, 2))
        self.assertEqual(y_windows.shape, (3, 1, 2))
        np.testing.assert_array_equal(x_windows[0], demand[0:2])
        np.testing.assert_array_equal(y_windows[2], demand[4:5])

    def test_monthly_aggregation_clips_rows_to_file_month(self) -> None:
        frame = pd.DataFrame(
            {
                "tpep_pickup_datetime": [
                    "2018-01-15 00:01:00",
                    "2018-02-01 00:01:00",
                ],
                "PULocationID": [1, 1],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "yellow_tripdata_2018-01.parquet"
            frame.to_parquet(path)
            counts, stats = aggregate_pickup_counts_for_file(
                path,
                pickup_time_column="tpep_pickup_datetime",
                origin_zone_column="PULocationID",
                interval_minutes=15,
                allowed_zone_ids=(1,),
                min_timestamp=pd.Timestamp("2018-01-01T00:00:00"),
                max_timestamp=pd.Timestamp("2018-02-02T00:00:00"),
            )

        self.assertEqual(stats["study_area_pickup_count"], 1)
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts.iloc[0]["timestamp"], pd.Timestamp("2018-01-15 00:00:00"))


if __name__ == "__main__":
    unittest.main()
