#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build manifest JSON from a doctor-provided CSV label table.

Expected source columns:
- patient_SN: patient-level grouping key used to avoid leakage
- event: outcome event flag
- time: outcome time
- xml_file: XML filename or relative path for a specific ECG exam

Output rows preserve input order and intentionally keep repeated patient_id rows
when the same patient has multiple ECG exams. Each output row contains:
- patient_SN
- patient_id
- time
- event
- xml_file
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest JSON from CSV(event,time,xml_file) labels plus XML files."
    )
    parser.add_argument("--labels-csv", type=Path, required=True, help="Doctor CSV/Excel labels file.")
    parser.add_argument("--xml-dir", type=Path, required=True, help="Directory containing XML ECG files.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--patient-sn-col", type=str, default="patient_SN")
    parser.add_argument("--event-col", type=str, default="event")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--xml-file-col", type=str, default="xml_file")
    parser.add_argument("--encoding", type=str, default=None, help="CSV encoding; auto-detect when omitted.")
    parser.add_argument("--delimiter", type=str, default=None, help="CSV separator; auto-detect when omitted.")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name/index when input is .xls/.xlsx.")
    parser.add_argument(
        "--xml-encodings",
        nargs="*",
        default=["iso-8859-1", "utf-8", "utf-8-sig"],
        help="XML text encodings to try when reading patient IDs.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
    return df


def read_table(path: Path, encoding: str | None, delimiter: str | None, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return normalize_columns(pd.read_excel(path, sheet_name=sheet or 0))

    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "gbk", "latin1"]
    delimiters = [delimiter] if delimiter else [None, ",", "\t", ";"]
    last_error: Exception | None = None
    for enc in encodings:
        for sep in delimiters:
            try:
                return normalize_columns(pd.read_csv(path, encoding=enc, sep=sep, engine="python"))
            except Exception as exc:  # pragma: no cover - exercised indirectly
                last_error = exc
    raise RuntimeError(f"Unable to read label table: {path}; last error: {last_error}")


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def build_xml_reference_index(xml_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for xml_path in sorted(xml_dir.rglob("*.xml")):
        by_name.setdefault(xml_path.name.lower(), []).append(xml_path)
        by_stem.setdefault(xml_path.stem.lower(), []).append(xml_path)
    return by_name, by_stem


def _pick_unique_candidate(candidates: Iterable[Path], ref_text: str) -> Path:
    unique = sorted({path.resolve() for path in candidates})
    if not unique:
        raise FileNotFoundError(ref_text)
    if len(unique) > 1:
        raise RuntimeError(
            f"Ambiguous xml_file reference {ref_text!r}: " + ", ".join(str(path) for path in unique)
        )
    return unique[0]


def resolve_xml_reference(
    ref_value,
    xml_dir: Path,
    by_name: dict[str, list[Path]],
    by_stem: dict[str, list[Path]],
) -> Path:
    ref_text = str(ref_value).strip()
    if not ref_text:
        raise FileNotFoundError("empty xml_file")

    ref_path = Path(ref_text)
    if ref_path.is_absolute():
        if ref_path.exists():
            return ref_path.resolve()
        raise FileNotFoundError(ref_text)

    direct = (xml_dir / ref_path)
    if direct.exists():
        return direct.resolve()

    candidates: list[Path] = []
    name_key = ref_path.name.lower()
    stem_key = ref_path.stem.lower()

    candidates.extend(by_name.get(name_key, []))
    if not ref_path.suffix:
        candidates.extend(by_name.get(f"{name_key}.xml", []))
    candidates.extend(by_stem.get(stem_key, []))

    return _pick_unique_candidate(candidates, ref_text)


def extract_patient_id(xml_path: Path, encodings: list[str]) -> str:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            root = ET.fromstring(xml_path.read_text(encoding=encoding))
            patient_id = root.findtext(".//PatientDemographics/PatientID")
            patient_id = str(patient_id).strip() if patient_id is not None else ""
            if patient_id:
                return patient_id
        except Exception as exc:  # pragma: no cover - exercised indirectly
            last_error = exc
            continue
    raise ValueError(f"Unable to extract PatientID from {xml_path}: {last_error}")


def make_manifest_rows(
    labels: pd.DataFrame,
    xml_dir: Path,
    patient_sn_col: str,
    event_col: str,
    time_col: str,
    xml_file_col: str,
    xml_encodings: list[str],
) -> tuple[list[dict], dict]:
    missing = [column for column in (patient_sn_col, event_col, time_col, xml_file_col) if column not in labels.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}; available columns: {labels.columns.tolist()}")

    by_name, by_stem = build_xml_reference_index(xml_dir)
    rows: list[dict] = []
    missing_xml_refs: list[dict] = []
    invalid_label_rows: list[dict] = []
    invalid_patient_rows: list[dict] = []

    for row_index, row in labels.iterrows():
        patient_sn = str(row[patient_sn_col]).strip()
        event_value = to_int(row[event_col])
        time_value = to_float(row[time_col])
        xml_ref = row[xml_file_col]

        if (not patient_sn) or event_value is None or time_value is None:
            invalid_label_rows.append(
                {
                    "row_index": int(row_index),
                    "patient_SN": patient_sn,
                    "event": None if event_value is None else int(event_value),
                    "time": None if time_value is None else float(time_value),
                    "xml_file": str(xml_ref),
                }
            )
            continue

        try:
            xml_path = resolve_xml_reference(xml_ref, xml_dir, by_name, by_stem)
        except Exception as exc:
            missing_xml_refs.append(
                {
                    "row_index": int(row_index),
                    "xml_file": str(xml_ref),
                    "error": str(exc),
                }
            )
            continue

        try:
            patient_id = extract_patient_id(xml_path, xml_encodings)
        except Exception as exc:
            invalid_patient_rows.append(
                {
                    "row_index": int(row_index),
                    "xml_file": str(xml_ref),
                    "error": str(exc),
                }
            )
            continue

        try:
            xml_file_value = xml_path.resolve().relative_to(xml_dir.resolve()).as_posix()
        except ValueError:
            xml_file_value = str(xml_path.resolve())

        rows.append(
            {
                "patient_SN": patient_sn,
                "patient_id": patient_id,
                "time": float(time_value),
                "event": int(event_value),
                "xml_file": xml_file_value,
            }
        )

    report = {
        "input_rows": int(len(labels)),
        "manifest_rows": int(len(rows)),
        "missing_xml_rows": int(len(missing_xml_refs)),
        "invalid_label_rows": int(len(invalid_label_rows)),
        "invalid_patient_rows": int(len(invalid_patient_rows)),
        "unique_patient_sn": int(len({row["patient_SN"] for row in rows})),
        "unique_patient_id": int(len({row["patient_id"] for row in rows})),
        "repeated_patient_sn_rows_preserved": int(len(rows) - len({row["patient_SN"] for row in rows})),
        "missing_xml_examples": missing_xml_refs[:20],
        "invalid_label_examples": invalid_label_rows[:20],
        "invalid_patient_examples": invalid_patient_rows[:20],
    }
    return rows, report


def main() -> None:
    args = parse_args()
    labels = read_table(args.labels_csv, args.encoding, args.delimiter, args.sheet)
    rows, report = make_manifest_rows(
        labels=labels,
        xml_dir=args.xml_dir,
        patient_sn_col=args.patient_sn_col,
        event_col=args.event_col,
        time_col=args.time_col,
        xml_file_col=args.xml_file_col,
        xml_encodings=list(args.xml_encodings),
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] manifest written: {args.output_manifest}")
    print(f"[ok] rows: {len(rows)}")

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] report written: {args.report_path}")


if __name__ == "__main__":
    main()
