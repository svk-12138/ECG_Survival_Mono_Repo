import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_survival_dataset import _waveform_leads, build_lead_audit


class PrepareSurvivalDatasetLeadAuditTest(unittest.TestCase):
    def test_waveform_leads_treats_augmented_leads_case_insensitively(self) -> None:
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <Waveform>
    <WaveformType>Rhythm</WaveformType>
    <LeadData><LeadID>I</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>II</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>III</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>AVR</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>AVL</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>AVF</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V1</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V2</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V3</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V4</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V5</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V6</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
  </Waveform>
</RestingECG>
"""
        root = ET.fromstring(xml_text)
        waveforms = _waveform_leads(root, Path("sample.xml"))

        self.assertEqual(len(waveforms), 1)
        self.assertTrue(waveforms[0]["supports_12lead"])
        self.assertIn("aVR", waveforms[0]["leads"])
        self.assertIn("aVL", waveforms[0]["leads"])
        self.assertIn("aVF", waveforms[0]["leads"])

    def test_build_lead_audit_falls_back_to_best_waveform_when_requested_type_missing(self) -> None:
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <Waveform>
    <NumberofLeads>8</NumberofLeads>
    <SampleBase>500</SampleBase>
    <LeadData><LeadID>I</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>II</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>III</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>AVR</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>AVL</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>AVF</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V1</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V2</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V3</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V4</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V5</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>V6</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
  </Waveform>
</RestingECG>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_dir = Path(tmpdir)
            xml_path = xml_dir / "sample.xml"
            xml_path.write_text(xml_text, encoding="utf-8")

            rows = [
                {
                    "xml_file": "sample.xml",
                    "patient_id": "P001",
                    "patient_SN": "SN001",
                }
            ]
            audit_rows = build_lead_audit(
                rows=rows,
                xml_dir=xml_dir,
                waveform_type="Rhythm",
                xml_encodings=["utf-8"],
            )

            self.assertEqual(len(audit_rows), 1)
            row = audit_rows[0]
            self.assertFalse(row["requested_waveform_found"])
            self.assertEqual(row["effective_waveform_source"], "best")
            self.assertTrue(row["effective_supports_12lead"])
            self.assertEqual(row["effective_lead_count"], 12)

    def test_waveform_leads_counts_only_decodable_leads(self) -> None:
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <Waveform>
    <WaveformType>Rhythm</WaveformType>
    <LeadData><LeadID>I</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
    <LeadData><LeadID>II</LeadID><WaveFormData>!!!</WaveFormData></LeadData>
    <LeadData><LeadID>V1</LeadID><WaveFormData>0 100 -50 25</WaveFormData></LeadData>
  </Waveform>
</RestingECG>
"""
        root = ET.fromstring(xml_text)
        waveforms = _waveform_leads(root, Path("sample.xml"))

        self.assertEqual(waveforms[0]["leads"], ["I", "V1"])
        self.assertFalse(waveforms[0]["supports_8lead"])


if __name__ == "__main__":
    unittest.main()
