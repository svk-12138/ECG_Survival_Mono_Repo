import unittest
from pathlib import Path

import yaml


class StrokeThesisConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = Path(__file__).resolve().parents[1] / "configs" / "stroke_survival_thesis.yaml"
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_config_exists(self) -> None:
        self.assertTrue(self.config_path.exists())

    def test_core_fields(self) -> None:
        self.assertIn(self.cfg.get("task_mode"), {"prediction", "classification"})
        self.assertIn(self.cfg.get("lead_mode"), {"8lead", "12lead"})
        self.assertIn("manifest", self.cfg)
        self.assertTrue("xml_dir" in self.cfg or "csv_dir" in self.cfg)


if __name__ == "__main__":
    unittest.main()
