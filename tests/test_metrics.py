from __future__ import annotations

import unittest

from lpe_stgtn.evaluation.metrics import mae, mape, rmse


class MetricTests(unittest.TestCase):
    def test_metrics_match_expected_values(self) -> None:
        truth = [1.0, 2.0, 3.0]
        pred = [2.0, 2.0, 1.0]

        self.assertAlmostEqual(mae(truth, pred), 1.0)
        self.assertAlmostEqual(rmse(truth, pred), (5.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(mape(truth, pred), (1.0 + 0.0 + (2.0 / 3.0)) / 3.0)

    def test_mape_handles_zero_targets(self) -> None:
        score = mape([0.0, 2.0], [1.0, 1.0], epsilon=1e-3)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
