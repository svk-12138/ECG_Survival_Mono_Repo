import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_manifest_from_event_csv as builder


def write_xml(path: Path, patient_id: str) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<RestingECG>
  <PatientDemographics>
    <PatientID>{patient_id}</PatientID>
  </PatientDemographics>
</RestingECG>
""",
        encoding="iso-8859-1",
    )


class BuildManifestFromEventCSVTest(unittest.TestCase):
    def test_preserves_multiple_ecgs_for_same_patient(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_dir = root / "xml"
            xml_dir.mkdir()

            write_xml(xml_dir / "exam_a.xml", "P001")
            write_xml(xml_dir / "exam_b.xml", "P001")

            labels = pd.DataFrame(
                [
                    {"patient_SN": "SN001", "event": 1, "time": 120.0, "xml_file": "exam_a.xml"},
                    {"patient_SN": "SN001", "event": 0, "time": 365.0, "xml_file": "exam_b.xml"},
                ]
            )

            rows, report = builder.make_manifest_rows(
                labels=labels,
                xml_dir=xml_dir,
                patient_sn_col="patient_SN",
                event_col="event",
                time_col="time",
                xml_file_col="xml_file",
                xml_encodings=["iso-8859-1"],
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["patient_SN"], "SN001")
            self.assertEqual(rows[1]["patient_SN"], "SN001")
            self.assertEqual(rows[0]["patient_id"], "P001")
            self.assertEqual(rows[1]["patient_id"], "P001")
            self.assertEqual(rows[0]["xml_file"], "exam_a.xml")
            self.assertEqual(rows[1]["xml_file"], "exam_b.xml")
            self.assertEqual(report["manifest_rows"], 2)
            self.assertEqual(report["unique_patient_sn"], 1)
            self.assertEqual(report["unique_patient_id"], 1)
            self.assertEqual(report["repeated_patient_sn_rows_preserved"], 1)


if __name__ == "__main__":
    unittest.main()
