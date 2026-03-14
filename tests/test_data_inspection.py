from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lpe_stgtn.data.inspection import summarize_parquet_file


class DataInspectionTests(unittest.TestCase):
    def test_required_columns_are_detected(self) -> None:
        frame = pd.DataFrame(
            {
                "tpep_pickup_datetime": ["2018-01-01 00:00:00"],
                "tpep_dropoff_datetime": ["2018-01-01 00:10:00"],
                "PULocationID": [1],
                "DOLocationID": [2],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.parquet"
            frame.to_parquet(path)
            summary = summarize_parquet_file(path, sample_rows=1)

        self.assertTrue(summary.has_required_columns)
        self.assertEqual(summary.num_rows, 1)
        self.assertEqual(len(summary.sample_rows), 1)

    def test_missing_columns_are_reported(self) -> None:
        frame = pd.DataFrame({"tpep_pickup_datetime": ["2018-01-01 00:00:00"]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_sample.parquet"
            frame.to_parquet(path)
            summary = summarize_parquet_file(path, sample_rows=1)

        self.assertIn("DOLocationID", summary.missing_required_columns)


if __name__ == "__main__":
    unittest.main()
