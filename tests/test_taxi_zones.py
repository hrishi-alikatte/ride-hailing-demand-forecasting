from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lpe_stgtn.data.taxi_zones import load_taxi_zone_lookup, summarize_study_area


class TaxiZoneLookupTests(unittest.TestCase):
    def test_lookup_rows_load_with_expected_types(self) -> None:
        csv_payload = """LocationID,Borough,Zone,service_zone
1,Manhattan,Alpha,Yellow Zone
2,Queens,Beta,Boro Zone
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "taxi_zone_lookup.csv"
            path.write_text(csv_payload, encoding="utf-8")
            rows = load_taxi_zone_lookup(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].location_id, 1)
        self.assertEqual(rows[0].borough, "Manhattan")
        self.assertEqual(rows[0].zone, "Alpha")

    def test_study_area_summary_applies_exclusions(self) -> None:
        csv_payload = """LocationID,Borough,Zone,service_zone
1,Manhattan,Alpha,Yellow Zone
2,Manhattan,Beta,Yellow Zone
3,Manhattan,Gamma,Yellow Zone
4,Brooklyn,Delta,Boro Zone
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "taxi_zone_lookup.csv"
            path.write_text(csv_payload, encoding="utf-8")
            summary = summarize_study_area(
                path,
                borough="Manhattan",
                expected_zone_count=2,
                excluded_location_ids=[2],
            )

        self.assertEqual(summary.available_zone_count, 3)
        self.assertEqual(summary.included_zone_count, 2)
        self.assertEqual(summary.included_location_ids, (1, 3))
        self.assertTrue(summary.matches_expected_zone_count)

    def test_repo_lookup_shows_documented_manhattan_count_mismatch(self) -> None:
        path = Path("data/external/taxi_zone_lookup.csv")
        summary = summarize_study_area(path, borough="Manhattan", expected_zone_count=68)

        self.assertEqual(summary.available_zone_count, 69)
        self.assertEqual(summary.included_zone_count, 69)
        self.assertFalse(summary.matches_expected_zone_count)


if __name__ == "__main__":
    unittest.main()
