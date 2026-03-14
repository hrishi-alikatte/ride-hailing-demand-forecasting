from __future__ import annotations

import unittest
from pathlib import Path

from lpe_stgtn.config import load_yaml


class ConfigLoadTests(unittest.TestCase):
    def test_default_config_loads_as_mapping(self) -> None:
        payload = load_yaml(Path("configs/default.yaml"))
        self.assertIn("project", payload)
        self.assertEqual(payload["project"]["name"], "lpe_stgtn_reproduction")


if __name__ == "__main__":
    unittest.main()
