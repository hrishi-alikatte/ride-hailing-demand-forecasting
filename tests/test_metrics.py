from __future__ import annotations

import unittest

from lpe_stgtn.evaluation.metrics import mae, mape, rmse


class MetricTests(unittest.TestCase):
    def test_metrics_match_expected_values(self) -> None:
        truth = [1.0, 2.0, 3.0]
        pred = [2.0, 2.0, 1.0]

        self.assertAlmostEqual(mae(truth, pred), 1.0)
        self.assertAlmostEqual(rmse(truth, pred), (5.0 / 3.0) ** 0.5)
        # MAPE returns percentage with epsilon=1.0 denominator clipping.
        # truth=1 → denom=max(1,1)=1, err=|2-1|/1=1.0
        # truth=2 → denom=max(2,1)=2, err=|2-2|/2=0.0
        # truth=3 → denom=max(3,1)=3, err=|1-3|/3=0.667
        # mean = (1.0 + 0.0 + 0.667) / 3 = 0.5556 → × 100 = 55.56%
        expected_mape = (1.0 + 0.0 + (2.0 / 3.0)) / 3.0 * 100
        self.assertAlmostEqual(mape(truth, pred), expected_mape)

    def test_mape_handles_zero_targets(self) -> None:
        # With null_val=0.0, the zero target is masked out.
        # Only truth=2 remains: |1-2|/max(2,1.0) = 0.5 → × 100 = 50%
        score = mape([0.0, 2.0], [1.0, 1.0])
        self.assertAlmostEqual(score, 50.0)

    def test_mape_returns_percentage(self) -> None:
        """MAPE should return values in percentage form (e.g. 31.82 not 0.3182)."""
        score = mape([10.0, 20.0, 30.0], [12.0, 22.0, 33.0])
        # |12-10|/10 + |22-20|/20 + |33-30|/30 = 0.2+0.1+0.1 → mean=0.1333 → 13.33%
        self.assertAlmostEqual(score, 13.33, places=1)
        self.assertGreater(score, 1.0)  # Must be > 1 since it's a percentage


if __name__ == "__main__":
    unittest.main()
