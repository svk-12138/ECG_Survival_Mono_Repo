import argparse
import base64
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


LEADS_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _resample(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] == target_len:
        return signal
    x_old = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def _parse_rhythm(
    xml_path: Path, leads: list[str], target_len: int, max_ms: float | None
) -> tuple[np.ndarray, np.ndarray]:
    root = ET.fromstring(xml_path.read_text(encoding="iso-8859-1"))
    waveform = None
    for wf in root.findall(".//Waveform"):
        if (wf.findtext("WaveformType") or "").strip().lower() == "rhythm":
            waveform = wf
            break
    if waveform is None:
        raise ValueError("Rhythm waveform not found")

    sample_base = float(waveform.findtext("SampleBase") or 500)
    sample_exp = float(waveform.findtext("SampleExponent") or 0)
    sample_rate = sample_base * (10 ** sample_exp)
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        sample_rate = 500.0

    lead_signals: dict[str, np.ndarray] = {}
    for ld in waveform.findall("LeadData"):
        lead_id = (ld.findtext("LeadID") or "").strip()
        if lead_id not in leads:
            continue
        wf_text = ld.findtext("WaveFormData") or ""
        raw = base64.b64decode(wf_text)
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        units_per_bit = float(ld.findtext("LeadAmplitudeUnitsPerBit") or 1.0)
        arr = arr * units_per_bit
        lead_signals[lead_id] = arr

    missing = [lead for lead in leads if lead not in lead_signals]
    if missing:
        raise ValueError(f"Missing leads: {missing}")

    if max_ms and max_ms > 0:
        max_samples = int(max_ms * sample_rate / 1000.0) + 1
        for lead in list(lead_signals.keys()):
            lead_signals[lead] = lead_signals[lead][:max_samples]
        duration_ms = float(max_ms)
    else:
        any_lead = next(iter(lead_signals.values()))
        duration_ms = float((len(any_lead) - 1) / sample_rate * 1000.0)

    stacked = []
    for lead in leads:
        sig = _resample(lead_signals[lead], target_len)
        stacked.append(sig)
    time_ms = np.linspace(0.0, duration_ms, target_len, dtype=np.float32)
    return time_ms, np.stack(stacked, axis=0)


def _load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest must be a list")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rhythm ECG from XML to CSV (4096).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--target-len", type=int, default=4096)
    parser.add_argument("--max-ms", type=float, default=0.0, help="仅保留前 max_ms 毫秒")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_index = {}
    for xml_path in args.xml_dir.glob("*.xml"):
        try:
            root = ET.fromstring(xml_path.read_text(encoding="iso-8859-1"))
            pid = root.findtext(".//PatientDemographics/PatientID")
        except Exception:
            continue
        if pid:
            if pid not in xml_index or xml_path.stat().st_mtime > xml_index[pid].stat().st_mtime:
                xml_index[pid] = xml_path

    written = []
    missing_xml = []
    failed = []

    for row in rows:
        pid = str(row.get("patient_id"))
        if pid not in xml_index:
            missing_xml.append(pid)
            continue
        out_path = out_dir / f"rhythm.{pid}.csv"
        if out_path.exists() and not args.overwrite:
            written.append(row)
            continue
        try:
            time_ms, signals = _parse_rhythm(xml_index[pid], LEADS_8, args.target_len, args.max_ms)
        except Exception as exc:
            failed.append({"patient_id": pid, "reason": str(exc)})
            continue
        df = pd.DataFrame(signals.T, columns=LEADS_8)
        df.insert(0, "time_ms", time_ms)
        df.to_csv(out_path, index=False)
        written.append(row)
        if args.max_samples and len(written) >= args.max_samples:
            break

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(written, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "manifest_total": len(rows),
        "xml_total": len(xml_index),
        "written": len(written),
        "missing_xml": len(missing_xml),
        "failed": len(failed),
        "max_ms": args.max_ms,
        "target_len": args.target_len,
        "output_dir": str(out_dir),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if failed:
        pd.DataFrame(failed).to_csv(out_dir / "failed.csv", index=False, encoding="utf-8")
    if missing_xml:
        pd.DataFrame({"patient_id": sorted(set(missing_xml))}).to_csv(
            out_dir / "missing_xml.csv", index=False, encoding="utf-8"
        )

    print(f"[OK] wrote {len(written)} csvs to {out_dir}")
    print(f"[OK] manifest: {args.output_manifest}")
    print(f"[OK] report: {args.report_path}")


if __name__ == "__main__":
    main()
