import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.infer_survival_risk import _row_xml_file_value


class InferRiskScoreMetadataTest(unittest.TestCase):
    def test_prefers_manifest_xml_file_when_present(self) -> None:
        row = {"patient_id": "P001", "xml_file": "folder/sample_a.xml"}
        self.assertEqual(_row_xml_file_value(row, {"P001": Path("/tmp/real.xml")}), "folder/sample_a.xml")

    def test_falls_back_to_resolved_patient_index_path_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "sample_b.xml"
            xml_path.write_text("<xml />", encoding="utf-8")
            row = {"patient_id": "P002", "xml_file": ""}
            self.assertEqual(_row_xml_file_value(row, {"P002": xml_path}), str(xml_path.resolve()))


if __name__ == "__main__":
    unittest.main()
