from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from lpe_stgtn.data.datasets import ProcessedDatasetBundle
from lpe_stgtn.graphs.artifacts import load_graph_artifacts
from lpe_stgtn.models.baselines.dual_graph_gru import DualGraphGRUBaseline
from lpe_stgtn.training.baselines import validate_graph_alignment


class Stage4GraphBaselineTests(unittest.TestCase):
    def test_load_graph_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "zone_ids.npy", np.array([4, 105], dtype=np.int32))
            np.save(root / "distance_adjacency.npy", np.eye(2, dtype=np.float32))
            np.save(root / "distance_adjacency_normalized.npy", np.eye(2, dtype=np.float32))
            np.save(root / "od_flow_counts.npy", np.zeros((2, 2), dtype=np.int64))
            np.save(root / "od_flow_adjacency.npy", np.eye(2, dtype=np.float32))
            np.save(root / "od_flow_adjacency_normalized.npy", np.eye(2, dtype=np.float32))
            (root / "metadata.json").write_text(
                json.dumps({"distance_graph": {"distance_method": "road_network_shortest_path"}}),
                encoding="utf-8",
            )

            bundle = load_graph_artifacts(root)

        self.assertEqual(bundle.num_zones, 2)
        np.testing.assert_array_equal(bundle.zone_ids, np.array([4, 105], dtype=np.int32))
        self.assertEqual(
            bundle.metadata["distance_graph"]["distance_method"],
            "road_network_shortest_path",
        )

    def test_validate_graph_alignment_raises_on_zone_mismatch(self) -> None:
        dataset_bundle = ProcessedDatasetBundle(
            root_dir=Path("."),
            demand=np.zeros((4, 2), dtype=np.int32),
            normalized_demand=np.zeros((4, 2), dtype=np.float32),
            zone_ids=np.array([4, 105], dtype=np.int32),
            time_of_day=np.zeros(4, dtype=np.int32),
            day_of_week=np.zeros(4, dtype=np.int32),
            train_sample_start_indices=np.array([0], dtype=np.int32),
            validation_sample_start_indices=np.array([0], dtype=np.int32),
            test_sample_start_indices=np.array([0], dtype=np.int32),
            metadata={"normalization": {"train_mean": 0.0, "train_std": 1.0}},
            history_steps=1,
            forecast_steps=1,
            normalization_mean=0.0,
            normalization_std=1.0,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "zone_ids.npy", np.array([4, 104], dtype=np.int32))
            np.save(root / "distance_adjacency.npy", np.eye(2, dtype=np.float32))
            np.save(root / "distance_adjacency_normalized.npy", np.eye(2, dtype=np.float32))
            np.save(root / "od_flow_counts.npy", np.zeros((2, 2), dtype=np.int64))
            np.save(root / "od_flow_adjacency.npy", np.eye(2, dtype=np.float32))
            np.save(root / "od_flow_adjacency_normalized.npy", np.eye(2, dtype=np.float32))
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            graph_bundle = load_graph_artifacts(root)

            with self.assertRaises(ValueError):
                validate_graph_alignment(dataset_bundle, graph_bundle)

    def test_dual_graph_gru_baseline_forward_shape(self) -> None:
        model = DualGraphGRUBaseline(
            num_zones=3,
            forecast_steps=2,
            distance_adjacency=torch.eye(3),
            od_flow_adjacency=torch.eye(3),
            graph_hidden_dim=8,
            temporal_hidden_dim=16,
            attention_heads=4,
        )
        history = torch.randn(5, 12, 3)
        forecast = model(history)
        self.assertEqual(tuple(forecast.shape), (5, 2, 3))

    def test_dual_graph_gru_baseline_starts_as_persistence(self) -> None:
        model = DualGraphGRUBaseline(
            num_zones=2,
            forecast_steps=3,
            distance_adjacency=torch.eye(2),
            od_flow_adjacency=torch.eye(2),
            graph_hidden_dim=8,
            temporal_hidden_dim=8,
            attention_heads=4,
        )
        history = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        forecast = model(history)
        expected = history[:, -1:, :].repeat(1, 3, 1)
        torch.testing.assert_close(forecast, expected)


if __name__ == "__main__":
    unittest.main()
