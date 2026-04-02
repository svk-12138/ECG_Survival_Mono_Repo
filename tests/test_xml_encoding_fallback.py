import base64
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.ecg_preprocessing import ECGPreprocessingConfig, LEADS_KEEP_8, load_xml_ecg


def _lead_waveform_base64() -> str:
    signal = np.array([0, 100, -50, 25], dtype="<i2")
    return base64.b64encode(signal.tobytes()).decode("ascii")


def write_xml(path: Path, encoding: str) -> None:
    lead_data = []
    for lead in LEADS_KEEP_8:
        lead_data.append(
            f"""
    <LeadData>
      <LeadID>{lead}</LeadID>
      <LeadAmplitudeUnitsPerBit>1.0</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{_lead_waveform_base64()}</WaveFormData>
    </LeadData>"""
        )

    xml_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <Waveform>
    <WaveformType>Rhythm</WaveformType>
    <SampleBase>500</SampleBase>
    <SampleExponent>0</SampleExponent>
    {''.join(lead_data)}
  </Waveform>
</RestingECG>
"""
    path.write_text(xml_text, encoding=encoding)


class XMLParsingFallbackTest(unittest.TestCase):
    def test_load_xml_ecg_supports_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "bom_utf8.xml"
            write_xml(xml_path, encoding="utf-8-sig")

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                apply_filters=False,
                normalize=False,
            )
            x = load_xml_ecg(xml_path, cfg)
            self.assertEqual(x.shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
