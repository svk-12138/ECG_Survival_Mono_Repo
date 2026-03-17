#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取 XML 目录 + 标签表（CSV/Excel），生成对齐后的 JSON manifest。
- 以 XML 为主：保留所有 XML 中出现的 PatientID；若标签缺失则补 event=0, time=默认（5 年）。
- 标签表必含 PatientID/time/event 列（列名可自定义）。
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DEFAULT_TIME_FILL = 1825  # 5年（天）


def parse_args():
    p = argparse.ArgumentParser(description="对齐 XML 与标签，生成 JSON manifest")
    p.add_argument("--xml_dir", type=Path, required=True, help="XML 目录")
    p.add_argument("--labels", type=Path, required=True, help="标签文件 CSV/Excel")
    p.add_argument("--out", type=Path, required=True, help="输出 JSON 路径")
    p.add_argument("--patient_field", type=str, default="PatientID")
    p.add_argument("--time_field", type=str, default="time")
    p.add_argument("--event_field", type=str, default="end")
    p.add_argument("--encoding", type=str, default=None, help="CSV 编码，空则自动尝试 utf-8/gbk")
    p.add_argument("--delimiter", type=str, default=None, help="CSV 分隔符，空则自动尝试逗号/制表")
    p.add_argument("--sheet", type=str, default=None, help="Excel 工作表名/索引")
    p.add_argument("--time_fill", type=float, default=DEFAULT_TIME_FILL, help="当 event=0 且 time 为空时的填充值")
    return p.parse_args()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def read_table(path: Path, encoding: Optional[str], delimiter: Optional[str], sheet: Optional[str]) -> pd.DataFrame:
    if path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path, sheet_name=sheet or 0)
        return normalize_cols(df)
    encs = [encoding] if encoding else ["utf-8", "gbk", "latin1"]
    seps = [delimiter] if delimiter else [None, "\t", ";"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                return normalize_cols(df)
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"无法读取标签文件，最后错误: {last_err}")


def to_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except Exception:
        return None


def to_int(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return int(val)
    except Exception:
        try:
            return int(float(val))
        except Exception:
            return None


def load_labels(path: Path, patient_field: str, time_field: str, event_field: str,
               encoding: Optional[str], delimiter: Optional[str], sheet: Optional[str], time_fill: float) -> Dict[str, Tuple[float, int]]:
    df = read_table(path, encoding, delimiter, sheet)
    missing = [c for c in [patient_field, time_field, event_field] if c not in df.columns]
    if missing:
        raise KeyError(f"列缺失: {missing}; 现有列: {df.columns.tolist()}")

    label_map: Dict[str, Tuple[float, int]] = {}
    for _, r in df.iterrows():
        pid = str(r[patient_field]).strip()
        if not pid:
            continue
        ev = to_int(r[event_field])
        ev = 0 if ev is None else ev
        t = to_float(r[time_field])
        if (t is None) and (ev == 0):
            t = time_fill
        if t is None:
            t = time_fill
        # 重复 pid：优先事件=1 或时间更早
        if pid in label_map:
            old_t, old_ev = label_map[pid]
            if ev > old_ev or (ev == old_ev and t < old_t):
                label_map[pid] = (t, ev)
        else:
            label_map[pid] = (t, ev)
    return label_map


def build_pid_index(xml_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for xml_file in xml_dir.glob('*.xml'):
        try:
            root = ET.fromstring(xml_file.read_text(encoding="iso-8859-1"))
            pid = root.findtext('.//PatientDemographics/PatientID')
            if pid:
                if pid not in index or xml_file.stat().st_mtime > index[pid].stat().st_mtime:
                    index[pid] = xml_file
        except Exception:
            continue
    if not index:
        raise ValueError("XML 目录未找到任何 PatientID")
    return index


def main():
    args = parse_args()
    label_map = load_labels(args.labels, args.patient_field, args.time_field, args.event_field,
                            args.encoding, args.delimiter, args.sheet, args.time_fill)
    xml_index = build_pid_index(args.xml_dir)

    out_rows = []
    for pid in xml_index.keys():
        if pid in label_map:
            t, ev = label_map[pid]
        else:
            t, ev = args.time_fill, 0
        out_rows.append({"patient_id": pid, "time": t, "event": ev})

    args.out.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出 {len(out_rows)} 条记录到 {args.out}")

if __name__ == "__main__":
    main()
