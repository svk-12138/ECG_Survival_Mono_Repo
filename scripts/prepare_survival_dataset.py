#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Doctor-friendly single-entry data preparation pipeline for survival training.

Inputs:
- one label file containing patient_SN / event / time / xml_file
- one XML directory

Outputs under output-dir:
- manifest.json
- 处理报告.txt
- process_report.json
- training_inputs.json
- audit/*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_manifest_from_event_csv as manifest_builder
from modules.survival_model.torch_survival.ecg_preprocessing import (
    LEADS_KEEP_8,
    LEADS_KEEP_12,
    _decode_waveform_signal,
)


def _normalize_lead_id(lead_id: str) -> str:
    """将导联名转成大小写无关的比较形式。"""

    return str(lead_id).strip().upper()


def _canonical_lead_id(lead_id: str) -> str:
    """把 XML 里的导联名统一成项目内部常用显示形式。"""

    raw = str(lead_id).strip()
    mapping = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }
    return mapping.get(_normalize_lead_id(raw), raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare survival manifest and audits from doctor labels + XML.")
    parser.add_argument("--labels", type=Path, required=True, help="Doctor CSV/Excel labels file.")
    parser.add_argument("--xml-dir", type=Path, required=True, help="XML root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Processed output directory.")
    parser.add_argument("--patient-sn-col", type=str, default="patient_SN")
    parser.add_argument("--event-col", type=str, default="event")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--xml-file-col", type=str, default="xml_file")
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--waveform-type", type=str, default="Rhythm")
    parser.add_argument(
        "--xml-encodings",
        nargs="*",
        default=["iso-8859-1", "utf-8", "utf-8-sig"],
        help="XML text encodings to try.",
    )
    return parser.parse_args()


def _parse_xml_root(xml_path: Path, encodings: list[str]) -> ET.Element:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return ET.fromstring(xml_path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Unable to parse XML {xml_path}: {last_error}")


def _waveform_leads(root: ET.Element, xml_path: Path) -> list[dict]:
    waveforms: list[dict] = []
    normalized_leads_keep_8 = {_normalize_lead_id(lead) for lead in LEADS_KEEP_8}
    normalized_leads_keep_12 = {_normalize_lead_id(lead) for lead in LEADS_KEEP_12}
    for idx, waveform in enumerate(root.findall(".//Waveform"), start=1):
        waveform_type = (waveform.findtext("WaveformType") or "").strip() or f"waveform_{idx}"
        readable_leads: set[str] = set()
        for lead_data in waveform.findall(".//LeadData"):
            raw_lead_id = (lead_data.findtext("LeadID") or "").strip()
            if not raw_lead_id:
                continue
            waveform_data = lead_data.findtext("WaveFormData") or ""
            try:
                signal = _decode_waveform_signal(waveform_data, xml_path, raw_lead_id)
            except Exception:
                continue
            if signal.size == 0:
                continue
            readable_leads.add(_canonical_lead_id(raw_lead_id))

        leads = sorted(readable_leads)
        normalized_leads = {_normalize_lead_id(lead) for lead in leads}
        waveforms.append(
            {
                "waveform_type": waveform_type,
                "lead_count": len(leads),
                "leads": leads,
                "supports_8lead": normalized_leads_keep_8.issubset(normalized_leads),
                "supports_12lead": normalized_leads_keep_12.issubset(normalized_leads),
            }
        )
    return waveforms


def build_lead_audit(rows: list[dict], xml_dir: Path, waveform_type: str, xml_encodings: list[str]) -> list[dict]:
    requested_type_lower = waveform_type.strip().lower()
    seen: set[str] = set()
    audit_rows: list[dict] = []
    for row in rows:
        xml_file = str(row["xml_file"])
        if xml_file in seen:
            continue
        seen.add(xml_file)
        xml_path = Path(xml_file)
        if not xml_path.is_absolute():
            xml_path = (xml_dir / xml_path).resolve()

        root = _parse_xml_root(xml_path, xml_encodings)
        waveform_rows = _waveform_leads(root, xml_path)
        requested = None
        best = None
        for item in waveform_rows:
            if best is None or int(item["lead_count"]) > int(best["lead_count"]):
                best = item
            if str(item["waveform_type"]).strip().lower() == requested_type_lower:
                requested = item
        effective = requested or best
        effective_source = "requested" if requested is not None else ("best" if best is not None else "none")

        audit_rows.append(
            {
                "xml_file": xml_file,
                "xml_abspath": str(xml_path),
                "patient_id": row["patient_id"],
                "patient_SN": row["patient_SN"],
                "requested_waveform_type": waveform_type,
                "requested_waveform_found": bool(requested),
                "requested_leads": ",".join(requested["leads"]) if requested else "",
                "requested_lead_count": int(requested["lead_count"]) if requested else 0,
                "requested_supports_8lead": bool(requested["supports_8lead"]) if requested else False,
                "requested_supports_12lead": bool(requested["supports_12lead"]) if requested else False,
                "best_waveform_type": best["waveform_type"] if best else "",
                "best_leads": ",".join(best["leads"]) if best else "",
                "best_lead_count": int(best["lead_count"]) if best else 0,
                "best_supports_8lead": bool(best["supports_8lead"]) if best else False,
                "best_supports_12lead": bool(best["supports_12lead"]) if best else False,
                "effective_waveform_source": effective_source,
                "effective_waveform_type": effective["waveform_type"] if effective else "",
                "effective_leads": ",".join(effective["leads"]) if effective else "",
                "effective_lead_count": int(effective["lead_count"]) if effective else 0,
                "effective_supports_8lead": bool(effective["supports_8lead"]) if effective else False,
                "effective_supports_12lead": bool(effective["supports_12lead"]) if effective else False,
                "all_waveforms_json": json.dumps(waveform_rows, ensure_ascii=False),
            }
        )
    return audit_rows


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "ECG Survival Data Preparation Report",
        "",
        f"status={report['status']}",
        f"labels_file={report['labels_file']}",
        f"xml_dir={report['xml_dir']}",
        f"output_dir={report['output_dir']}",
        f"waveform_type={report['waveform_type']}",
        "",
        "[summary]",
        f"input_rows={report['input_rows']}",
        f"manifest_rows={report['manifest_rows']}",
        f"unique_patient_sn={report['unique_patient_sn']}",
        f"unique_patient_id={report['unique_patient_id']}",
        f"repeated_patient_sn_rows_preserved={report['repeated_patient_sn_rows_preserved']}",
        f"missing_xml_rows={report['missing_xml_rows']}",
        f"invalid_label_rows={report['invalid_label_rows']}",
        f"invalid_patient_rows={report['invalid_patient_rows']}",
        f"matched_unique_xml={report['matched_unique_xml']}",
        f"requested_waveform_supports_8lead={report['requested_waveform_supports_8lead']}",
        f"requested_waveform_supports_12lead={report['requested_waveform_supports_12lead']}",
        f"effective_waveform_supports_8lead={report['effective_waveform_supports_8lead']}",
        f"effective_waveform_supports_12lead={report['effective_waveform_supports_12lead']}",
        f"recommended_lead_mode={report['recommended_lead_mode']}",
        "",
        "[doctor_advice]",
        report["doctor_advice"],
    ]

    example_sections = [
        ("missing_xml_examples", report.get("missing_xml_examples", [])),
        ("invalid_label_examples", report.get("invalid_label_examples", [])),
        ("invalid_patient_examples", report.get("invalid_patient_examples", [])),
    ]
    for title, rows in example_sections:
        if not rows:
            continue
        lines.extend(["", f"[{title}]"])
        for item in rows[:20]:
            lines.append(json.dumps(item, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    audit_dir = output_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    labels = manifest_builder.read_table(args.labels, args.encoding, args.delimiter, args.sheet)
    rows, join_report = manifest_builder.make_manifest_rows(
        labels=labels,
        xml_dir=args.xml_dir,
        patient_sn_col=args.patient_sn_col,
        event_col=args.event_col,
        time_col=args.time_col,
        xml_file_col=args.xml_file_col,
        xml_encodings=list(args.xml_encodings),
    )

    manifest_path = output_dir / "manifest.json"
    report_json_path = output_dir / "process_report.json"
    report_txt_path = output_dir / "处理报告.txt"
    training_inputs_path = output_dir / "training_inputs.json"

    critical_issue_count = (
        int(join_report["missing_xml_rows"])
        + int(join_report["invalid_label_rows"])
        + int(join_report["invalid_patient_rows"])
    )

    lead_audit_rows = build_lead_audit(
        rows=rows,
        xml_dir=args.xml_dir.resolve(),
        waveform_type=args.waveform_type,
        xml_encodings=list(args.xml_encodings),
    ) if rows else []
    lead_audit_df = pd.DataFrame(lead_audit_rows)
    lead_audit_path = audit_dir / "lead_audit.csv"
    lead_audit_df.to_csv(lead_audit_path, index=False, encoding="utf-8-sig")

    if join_report.get("missing_xml_examples"):
        pd.DataFrame(join_report["missing_xml_examples"]).to_csv(
            audit_dir / "missing_xml_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if join_report.get("invalid_label_examples"):
        pd.DataFrame(join_report["invalid_label_examples"]).to_csv(
            audit_dir / "invalid_label_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if join_report.get("invalid_patient_examples"):
        pd.DataFrame(join_report["invalid_patient_examples"]).to_csv(
            audit_dir / "invalid_patient_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    requested_8_count = int(lead_audit_df["requested_supports_8lead"].sum()) if not lead_audit_df.empty else 0
    requested_12_count = int(lead_audit_df["requested_supports_12lead"].sum()) if not lead_audit_df.empty else 0
    effective_8_count = int(lead_audit_df["effective_supports_8lead"].sum()) if not lead_audit_df.empty else 0
    effective_12_count = int(lead_audit_df["effective_supports_12lead"].sum()) if not lead_audit_df.empty else 0
    requested_found_count = int(lead_audit_df["requested_waveform_found"].sum()) if not lead_audit_df.empty else 0
    matched_unique_xml = int(len(lead_audit_df))
    recommended_lead_mode = "12lead" if matched_unique_xml > 0 and effective_12_count == matched_unique_xml else "8lead"

    status = "ready" if critical_issue_count == 0 else "failed"
    if status == "ready":
        if 0 < requested_found_count < matched_unique_xml:
            doctor_advice = (
                "可以开始训练。部分 XML 没有找到请求的 WaveformType，系统已自动回退到导联数最多的波形做判断；"
                "建议按 recommended_lead_mode 配置训练脚本。"
            )
        elif requested_found_count == 0 and matched_unique_xml > 0:
            doctor_advice = (
                "可以开始训练。当前 XML 未提供请求的 WaveformType，系统已自动回退到导联数最多的波形做判断；"
                "建议按 recommended_lead_mode 配置训练脚本。"
            )
        else:
            doctor_advice = "可以开始训练。建议按 recommended_lead_mode 配置训练脚本。"
    else:
        doctor_advice = "请先修正报告中的缺失 XML / 非法标签 / XML 解析问题，再重新运行本脚本。"

    report = {
        "status": status,
        "labels_file": str(args.labels.resolve()),
        "xml_dir": str(args.xml_dir.resolve()),
        "output_dir": str(output_dir),
        "waveform_type": args.waveform_type,
        **join_report,
        "matched_unique_xml": matched_unique_xml,
        "requested_waveform_supports_8lead": requested_8_count,
        "requested_waveform_supports_12lead": requested_12_count,
        "effective_waveform_supports_8lead": effective_8_count,
        "effective_waveform_supports_12lead": effective_12_count,
        "requested_waveform_found_count": requested_found_count,
        "recommended_lead_mode": recommended_lead_mode,
        "doctor_advice": doctor_advice,
        "manifest_path": str(manifest_path),
        "lead_audit_path": str(lead_audit_path),
    }

    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_report(report_txt_path, report)

    if status != "ready":
        if manifest_path.exists():
            manifest_path.unlink()
        if training_inputs_path.exists():
            training_inputs_path.unlink()
        print(f"[error] data preparation failed; see report: {report_txt_path}")
        return 1

    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    training_inputs = {
        "manifest": str(manifest_path),
        "xml_dir": str(args.xml_dir.resolve()),
        "csv_dir": None,
        "recommended_lead_mode": recommended_lead_mode,
        "waveform_type": args.waveform_type,
        "group_field": "patient_SN",
        "manifest_fields": ["patient_SN", "patient_id", "event", "time", "xml_file"],
    }
    training_inputs_path.write_text(json.dumps(training_inputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] manifest written: {manifest_path}")
    print(f"[ok] report written: {report_txt_path}")
    print(f"[ok] training inputs written: {training_inputs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
