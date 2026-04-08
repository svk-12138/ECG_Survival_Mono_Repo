import base64
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.ecg_preprocessing import ECGPreprocessingConfig, LEADS_KEEP_8, LEADS_KEEP_12, load_xml_ecg


def _lead_waveform_base64(strip_padding: bool = False) -> str:
    signal = np.array([0, 100, -50, 25], dtype="<i2")
    encoded = base64.b64encode(signal.tobytes()).decode("ascii")
    return encoded.rstrip("=") if strip_padding else encoded


def _lead_waveform_plaintext() -> str:
    signal = np.array([0, 100, -50, 25], dtype=np.int16)
    return " ".join(str(int(value)) for value in signal.tolist())


def write_xml(path: Path, encoding: str, strip_padding: bool = False, invalid_lead: str | None = None) -> None:
    lead_data = []
    for lead in LEADS_KEEP_8:
        waveform = "!!!" if lead == invalid_lead else _lead_waveform_base64(strip_padding=strip_padding)
        lead_data.append(
            f"""
    <LeadData>
      <LeadID>{lead}</LeadID>
      <LeadAmplitudeUnitsPerBit>1.0</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{waveform}</WaveFormData>
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


def write_xml_with_plaintext_waveform(
    path: Path,
    encoding: str,
    *,
    leads: tuple[str, ...] = LEADS_KEEP_8,
    lead_id_aliases: dict[str, str] | None = None,
) -> None:
    lead_data = []
    aliases = lead_id_aliases or {}
    for lead in leads:
        waveform = _lead_waveform_plaintext()
        xml_lead_id = aliases.get(lead, lead)
        lead_data.append(
            f"""
    <LeadData>
      <LeadID>{xml_lead_id}</LeadID>
      <LeadAmplitudeUnitsPerBit>1.0</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{waveform}</WaveFormData>
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


def write_xml_with_odd_waveform_bytes(path: Path, encoding: str) -> None:
    lead_data = []
    for lead in LEADS_KEEP_8:
        signal = np.array([0, 100, -50, 25], dtype="<i2").tobytes()
        if lead == "I":
            signal = signal + b"\x00"
        waveform = base64.b64encode(signal).decode("ascii")
        lead_data.append(
            f"""
    <LeadData>
      <LeadID>{lead}</LeadID>
      <LeadAmplitudeUnitsPerBit>1.0</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{waveform}</WaveFormData>
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


def write_xml_with_non_base64_noise(path: Path, encoding: str) -> None:
    lead_data = []
    for lead in LEADS_KEEP_8:
        waveform = _lead_waveform_base64()
        if lead == "I":
            waveform = waveform[:4] + "#" + waveform[4:] + "!"
        lead_data.append(
            f"""
    <LeadData>
      <LeadID>{lead}</LeadID>
      <LeadAmplitudeUnitsPerBit>1.0</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{waveform}</WaveFormData>
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


def write_xml_with_base64_len_mod_4_eq_1(path: Path, encoding: str) -> None:
    lead_data = []
    for lead in LEADS_KEEP_8:
        waveform = _lead_waveform_base64()
        if lead == "V1":
            waveform = waveform + "A"
        lead_data.append(
            f"""
    <LeadData>
      <LeadID>{lead}</LeadID>
      <LeadAmplitudeUnitsPerBit>1.0</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{waveform}</WaveFormData>
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
    def test_load_xml_ecg_supports_plaintext_integer_waveform(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "plaintext_waveform.xml"
            write_xml_with_plaintext_waveform(xml_path, encoding="utf-8-sig")

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                resample_hz=500.0,
                apply_filters=False,
                normalize=False,
            )
            x = load_xml_ecg(xml_path, cfg)
            self.assertEqual(x.shape, (8, 4))
            np.testing.assert_allclose(x[0], np.array([0.0, 100.0, -50.0, 25.0], dtype=np.float32))

    def test_load_xml_ecg_supports_uppercase_augmented_lead_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "uppercase_augmented_leads.xml"
            write_xml_with_plaintext_waveform(
                xml_path,
                encoding="utf-8",
                leads=LEADS_KEEP_12,
                lead_id_aliases={"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"},
            )

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_12,
                target_len=4,
                resample_hz=500.0,
                apply_filters=False,
                normalize=False,
            )
            x = load_xml_ecg(xml_path, cfg)
            self.assertEqual(x.shape, (12, 4))

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

    def test_load_xml_ecg_supports_missing_base64_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "missing_padding.xml"
            write_xml(xml_path, encoding="utf-8", strip_padding=True)

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                apply_filters=False,
                normalize=False,
            )
            x = load_xml_ecg(xml_path, cfg)
            self.assertEqual(x.shape, (8, 4))

    def test_load_xml_ecg_reports_invalid_waveform_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "invalid_waveform.xml"
            write_xml(xml_path, encoding="utf-8", invalid_lead="V3")

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                apply_filters=False,
                normalize=False,
            )
            with self.assertRaisesRegex(ValueError, r"invalid_waveform\.xml .* lead=V3"):
                load_xml_ecg(xml_path, cfg)

    def test_load_xml_ecg_repairs_odd_waveform_byte_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "odd_waveform.xml"
            write_xml_with_odd_waveform_bytes(xml_path, encoding="utf-8")

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                apply_filters=False,
                normalize=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x = load_xml_ecg(xml_path, cfg)

            self.assertEqual(x.shape, (8, 4))
            self.assertTrue(any("已自动裁掉最后 1 个字节" in str(item.message) for item in caught))

    def test_load_xml_ecg_repairs_non_base64_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "noisy_waveform.xml"
            write_xml_with_non_base64_noise(xml_path, encoding="utf-8")

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                apply_filters=False,
                normalize=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x = load_xml_ecg(xml_path, cfg)

            self.assertEqual(x.shape, (8, 4))
            self.assertTrue(any("已自动清洗后解码" in str(item.message) for item in caught))

    def test_load_xml_ecg_repairs_len_mod_4_eq_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "len_mod_4_eq_1.xml"
            write_xml_with_base64_len_mod_4_eq_1(xml_path, encoding="utf-8")

            cfg = ECGPreprocessingConfig(
                leads=LEADS_KEEP_8,
                target_len=4,
                apply_filters=False,
                normalize=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x = load_xml_ecg(xml_path, cfg)

            self.assertEqual(x.shape, (8, 4))
            self.assertTrue(any("已自动裁掉末尾 1 个字符后解码" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
