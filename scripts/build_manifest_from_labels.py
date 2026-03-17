import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def normalize_id(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        return text[:-2]
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest.json from XML + labels.xlsx.")
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument("--sheet", type=str, default="总表")
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--id-col", type=str, default="住院号")
    parser.add_argument("--event-col", type=str, default="end")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--missing-labels", type=Path, default=None)
    parser.add_argument("--missing-xml", type=Path, default=None)
    args = parser.parse_args()

    labels = pd.read_excel(args.labels_xlsx, sheet_name=args.sheet)
    labels = labels.rename(columns=lambda c: str(c).strip())
    for col in (args.id_col, args.event_col, args.time_col):
        if col not in labels.columns:
            raise SystemExit(f"Missing column: {col}")

    labels = labels[[args.id_col, args.event_col, args.time_col]].copy()
    labels[args.id_col] = labels[args.id_col].apply(normalize_id)
    labels = labels.dropna(subset=[args.id_col])

    labels[args.event_col] = pd.to_numeric(labels[args.event_col], errors="coerce")
    labels[args.time_col] = pd.to_numeric(labels[args.time_col], errors="coerce")
    labels = labels.dropna(subset=[args.event_col, args.time_col])

    labels[args.event_col] = labels[args.event_col].astype(int)

    if labels.duplicated(subset=[args.id_col]).any():
        labels = labels.drop_duplicates(subset=[args.id_col], keep="first")

    xml_ids = {}
    for path in args.xml_dir.glob("*.xml"):
        try:
            root = ET.fromstring(path.read_text(encoding="iso-8859-1"))
            pid = root.findtext(".//PatientDemographics/PatientID")
        except Exception:
            continue
        if pid:
            if pid not in xml_ids or path.stat().st_mtime > (args.xml_dir / xml_ids[pid]).stat().st_mtime:
                xml_ids[pid] = path.name

    rows = []
    missing_xml = []
    for _, row in labels.iterrows():
        pid = row[args.id_col]
        if pid in xml_ids:
            rows.append(
                {
                    "patient_id": pid,
                    "time": float(row[args.time_col]),
                    "event": int(row[args.event_col]),
                }
            )
        else:
            missing_xml.append(pid)

    output_manifest = args.output_manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "labels_total": int(len(labels)),
        "xml_total": int(len(xml_ids)),
        "intersect": int(len(rows)),
        "labels_missing_xml": int(len(missing_xml)),
        "xml_missing_labels": int(len(set(xml_ids) - set(labels[args.id_col]))),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.missing_labels:
        missing_labels = sorted(set(labels[args.id_col]) - set(xml_ids))
        pd.DataFrame({"patient_id": missing_labels}).to_csv(args.missing_labels, index=False, encoding="utf-8")

    if args.missing_xml:
        pd.DataFrame({"patient_id": missing_xml}).to_csv(args.missing_xml, index=False, encoding="utf-8")

    print(f"[OK] manifest: {output_manifest}")
    print(f"[OK] report: {args.report_path}")


if __name__ == "__main__":
    main()
