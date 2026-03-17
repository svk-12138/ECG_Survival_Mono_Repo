#!/usr/bin/env python3
"""
Build a label CSV by matching manifest patient_ids with ECG CSV files.

Example:
  python scripts/build_labels.py \
      --manifest /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/manifest.json \
      --csv-dir /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/median_csv \
      --output /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/labels.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate label CSV from manifest + ECG CSV directory.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json containing patient_id/time/event.")
    parser.add_argument("--csv-dir", required=True, help="Directory containing *_<patient_id>.csv files.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    return parser


def extract_patient_id(file: Path) -> str:
    stem = file.stem
    parts = stem.split("_")
    return parts[-1] if parts else stem


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    csv_dir = Path(args.csv_dir)
    output_path = Path(args.output)

    entries: List[Dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_map = {str(item["patient_id"]): item for item in entries}

    records: List[Dict] = []
    unmatched_patients = set(manifest_map.keys())
    missing_manifest: List[str] = []

    for csv_file in sorted(csv_dir.glob("*.csv")):
        patient_id = extract_patient_id(csv_file)
        entry = manifest_map.get(patient_id)
        if entry is None:
            missing_manifest.append(csv_file.name)
            continue
        unmatched_patients.discard(patient_id)
        records.append(
            {
                "sample_id": csv_file.stem,
                "patient_id": patient_id,
                "csv_path": str(csv_file),
                "time": entry.get("time"),
                "event": entry.get("event"),
            }
        )

    if not records:
        raise RuntimeError("No matching CSV files found for patient IDs in manifest.")

    df = pd.DataFrame(records).sort_values("sample_id")
    df.to_csv(output_path, index=False)
    print(f"[labels] Wrote {len(df)} rows to {output_path}")

    if missing_manifest:
        print(f"[labels] Warning: {len(missing_manifest)} CSV files without manifest entry (skipped).")
    if unmatched_patients:
        print(f"[labels] Warning: {len(unmatched_patients)} manifest entries had no CSV file.")


if __name__ == "__main__":
    main()
