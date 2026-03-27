from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lpe_stgtn.data.datasets import WindowedDemandDataset, load_processed_dataset
from lpe_stgtn.models.baselines.lstm import LSTMBaseline
from lpe_stgtn.models.baselines.persistence import PersistenceBaseline
from lpe_stgtn.training.baselines import compute_metrics, deep_merge


class DatasetAndBaselineTests(unittest.TestCase):
    def test_load_processed_dataset_and_windowed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            np.savez_compressed(
                root / "dataset_arrays.npz",
                demand=np.arange(24, dtype=np.int32).reshape(6, 4),
                normalized_demand=np.arange(24, dtype=np.float32).reshape(6, 4) / 10.0,
                zone_ids=np.array([1, 2, 3, 4], dtype=np.int32),
                time_of_day=np.array([0, 1, 2, 3, 4, 5], dtype=np.int16),
                day_of_week=np.array([0, 0, 0, 0, 0, 0], dtype=np.int16),
                train_sample_start_indices=np.array([0, 1], dtype=np.int32),
                validation_sample_start_indices=np.array([2], dtype=np.int32),
                test_sample_start_indices=np.array([3], dtype=np.int32),
            )
            metadata = {
                "history_steps": 2,
                "forecast_steps": 1,
                "normalization": {"train_mean": 1.5, "train_std": 2.0},
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            bundle = load_processed_dataset(root)
            dataset = WindowedDemandDataset(bundle, split="train", normalized=True)

            self.assertEqual(bundle.num_zones, 4)
            self.assertEqual(len(dataset), 2)
            x_window, y_window = dataset[0]
            self.assertEqual(tuple(x_window.shape), (2, 4))
            self.assertEqual(tuple(y_window.shape), (1, 4))

    def test_persistence_baseline_repeats_last_step(self) -> None:
        history = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ]
        )
        baseline = PersistenceBaseline(forecast_steps=2)
        prediction = baseline.predict(history)

        expected = torch.tensor([[[5.0, 6.0], [5.0, 6.0]]])
        self.assertTrue(torch.equal(prediction, expected))

    def test_lstm_baseline_forward_shape(self) -> None:
        model = LSTMBaseline(input_dim=5, hidden_dim=8, forecast_steps=3)
        history = torch.randn(4, 12, 5)
        prediction = model(history)
        self.assertEqual(tuple(prediction.shape), (4, 3, 5))

    def test_compute_metrics_and_deep_merge(self) -> None:
        metrics = compute_metrics(
            np.array([[[1.0], [2.0]]]),
            np.array([[[2.0], [2.0]]]),
        )
        self.assertAlmostEqual(metrics["mae"], 0.5)

        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4})
        self.assertEqual(merged, {"a": {"b": 1, "c": 3}, "d": 4})


if __name__ == "__main__":
    unittest.main()
