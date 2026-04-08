import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_survival_dataset import _waveform_leads


class PrepareSurvivalDatasetLeadAuditTest(unittest.TestCase):
    def test_waveform_leads_treats_augmented_leads_case_insensitively(self) -> None:
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <Waveform>
    <WaveformType>Rhythm</WaveformType>
    <LeadData><LeadID>I</LeadID></LeadData>
    <LeadData><LeadID>II</LeadID></LeadData>
    <LeadData><LeadID>III</LeadID></LeadData>
    <LeadData><LeadID>AVR</LeadID></LeadData>
    <LeadData><LeadID>AVL</LeadID></LeadData>
    <LeadData><LeadID>AVF</LeadID></LeadData>
    <LeadData><LeadID>V1</LeadID></LeadData>
    <LeadData><LeadID>V2</LeadID></LeadData>
    <LeadData><LeadID>V3</LeadID></LeadData>
    <LeadData><LeadID>V4</LeadID></LeadData>
    <LeadData><LeadID>V5</LeadID></LeadData>
    <LeadData><LeadID>V6</LeadID></LeadData>
  </Waveform>
</RestingECG>
"""
        root = ET.fromstring(xml_text)
        waveforms = _waveform_leads(root)

        self.assertEqual(len(waveforms), 1)
        self.assertTrue(waveforms[0]["supports_12lead"])
        self.assertIn("aVR", waveforms[0]["leads"])
        self.assertIn("aVL", waveforms[0]["leads"])
        self.assertIn("aVF", waveforms[0]["leads"])


if __name__ == "__main__":
    unittest.main()
