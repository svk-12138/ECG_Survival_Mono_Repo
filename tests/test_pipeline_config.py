import unittest
from pathlib import Path

import yaml


class PipelineConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline.default.yaml"
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_sections_present(self) -> None:
        for section in ("vae", "pearson", "survival"):
            self.assertIn(section, self.cfg, f"Missing {section} section in pipeline config.")

    def test_checkpoint_exists(self) -> None:
        checkpoint = Path(__file__).resolve().parents[1] / self.cfg["pearson"]["checkpoint"]
        self.assertTrue(checkpoint.exists(), "Pearson checkpoint not found.")

    def test_survival_task_mode(self) -> None:
        self.assertIn(self.cfg["survival"].get("task_mode"), {"prediction", "classification"})
        self.assertIn(self.cfg["survival_pred"].get("task_mode"), {"auto", "prediction", "classification"})

    def test_survival_lead_mode(self) -> None:
        self.assertIn(self.cfg["survival"].get("lead_mode"), {"8lead", "12lead"})
        self.assertIn(self.cfg["survival_pred"].get("lead_mode"), {"8lead", "12lead"})


if __name__ == "__main__":
    unittest.main()
