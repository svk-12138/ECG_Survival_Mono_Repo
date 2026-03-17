import json
import unittest
from pathlib import Path


class DataManifestTemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.template_path = Path(__file__).resolve().parents[1] / "configs" / "data_manifest.template.json"
        self.template = json.loads(self.template_path.read_text(encoding="utf-8"))

    def test_is_list(self) -> None:
        self.assertIsInstance(self.template, list)
        self.assertGreaterEqual(len(self.template), 1)

    def test_required_fields(self) -> None:
        sample = self.template[0]
        for field in ("patient_id", "xml_path", "time_to_event", "event"):
            self.assertIn(field, sample)


if __name__ == "__main__":
    unittest.main()
