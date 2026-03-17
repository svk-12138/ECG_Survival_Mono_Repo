#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load a Braveheart-style median MAT file (e.g., 1.653896_medians.mat) and export the
median 12-lead signals into a CSV. Assumes structure similar to previous median_12L
objects: mat["data"]["median_12L"] contains custom ECG12 objects serialized via MCOS.
"""

import argparse
import csv
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _extract_leads_from_workspace(mat_dict):
    fw_raw = mat_dict["__function_workspace__"].flatten().tobytes()
    header = bytearray(b"PYWORKSPACE MATFILE".ljust(116, b" "))
    header += b"\x00" * 8
    header += fw_raw[:2]
    header += fw_raw[2:4]
    body = fw_raw[8:]
    mini = BytesIO(bytes(header) + body)
    reader = MatFile5Reader(mini)
    mini.seek(0)
    reader.initialize_read()
    reader.read_file_header()
    hdr, next_pos = reader.read_var_header()
    mcos_array = reader.read_var_array(hdr)
    mini.seek(next_pos)
    entry = mcos_array["MCOS"][0, 0]
    data_list = list(entry["arr"][0])
    for idx, block in enumerate(data_list):
        if isinstance(block, np.ndarray) and block.size == 1:
            cell = block[0]
            if isinstance(cell, np.ndarray):
                if cell.size == 1 and cell.dtype.kind in {"U", "S"}:
                    text = str(cell.item())
                else:
                    continue
            else:
                text = str(cell)
            if text == "mV":
                sample_entry = data_list[idx - 1][0]
                sample_rate = float(sample_entry.item()) if isinstance(sample_entry, np.ndarray) else float(sample_entry)
                leads = []
                raw_order = [
                    "I",
                    "II",
                    "III",
                    "aVF",
                    "aVL",
                    "aVR",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6",
                ]
                for offset in range(1, 13):
                    arr = data_list[idx + offset][0]
                    leads.append(arr.reshape(-1))
                lead_map = {name: leads[i] for i, name in enumerate(raw_order)}
                ordered = [lead_map[name] for name in LEADS]
                return sample_rate, np.vstack(ordered)
    raise ValueError("无法在 __function_workspace__ 中找到 12 导 median 数据。")


def _coerce_numeric_array(value: np.ndarray) -> np.ndarray:
    """Convert MATLAB value into a 1-D float array."""
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _extract_from_medians_struct(medians_obj) -> Tuple[np.ndarray, float | None]:
    """Handle MAT files that store the median signals in a MATLAB struct."""
    field_map = {}
    if hasattr(medians_obj, "_fieldnames"):
        for name in medians_obj._fieldnames:
            field_map[name] = getattr(medians_obj, name)
    elif isinstance(medians_obj, np.ndarray) and medians_obj.dtype.names:
        for name in medians_obj.dtype.names:
            field_map[name] = medians_obj[name]
    else:
        raise ValueError("未找到兼容的 medians 结构。")

    lead_arrays = {}
    normalized_keys = {}
    for key, value in field_map.items():
        try:
            lead_arrays[key] = _coerce_numeric_array(value)
            normalized_keys[key.lower()] = key
        except Exception:
            continue

    ordered_rows = []
    for lead in LEADS:
        actual_key = normalized_keys.get(lead.lower())
        if actual_key is None or actual_key not in lead_arrays:
            raise ValueError(f"medians 结构缺少导联 {lead}")
        ordered_rows.append(lead_arrays[actual_key])

    lengths = {row.shape[0] for row in ordered_rows}
    if len(lengths) != 1:
        raise ValueError("导联长度不一致，无法对齐为矩阵。")

    fs = None
    for candidate in ("Fs", "fs", "sampling_rate", "sample_rate", "Hz", "hz"):
        actual_key = normalized_keys.get(candidate.lower())
        if actual_key and actual_key in lead_arrays:
            try:
                fs = float(lead_arrays[actual_key].reshape(-1)[0])
                break
            except Exception:
                continue

    stacked = np.vstack(ordered_rows)
    return stacked, fs


def extract_median_12l(mat_path: Path) -> Tuple[np.ndarray, float | None]:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True, simplify_cells=False)
    if "__function_workspace__" in mat:
        sample_rate, leads = _extract_leads_from_workspace(mat)
        return leads, sample_rate
    if "medians" in mat:
        leads, sample_rate = _extract_from_medians_struct(mat["medians"])
        return leads, sample_rate
    raise ValueError("MAT 文件中既没有 __function_workspace__ 也没有 medians 字段，无法解析。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract median 12-lead from MAT to CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Median MAT file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path; defaults to same name with .csv extension.",
    )
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".csv")

    data, sample_rate = extract_median_12l(args.input)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx"] + LEADS[: data.shape[0]])
        for idx in range(data.shape[1]):
            row = [idx] + [float(data[lead_idx, idx]) for lead_idx in range(data.shape[0])]
            writer.writerow(row)
    rate_display = f"{sample_rate} Hz" if sample_rate is not None else "unknown Hz"
    print(f"Saved CSV ({rate_display}) to {output}")


if __name__ == "__main__":
    main()
