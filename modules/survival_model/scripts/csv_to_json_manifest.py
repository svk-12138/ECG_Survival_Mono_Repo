#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 CSV/Excel 提取 PatientID/time/event 三列写入 JSON（不校验 XML）。"""
import argparse
import json
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="提取 PatientID/time/event 三列，输出 JSON")
    p.add_argument("--input", type=Path, required=True, help="输入 CSV 或 Excel 文件")
    p.add_argument("--out", type=Path, required=True, help="输出 JSON")
    p.add_argument("--patient_field", type=str, default="PatientID")
    p.add_argument("--time_field", type=str, default="time")
    p.add_argument("--event_field", type=str, default="end")
    p.add_argument("--encoding", type=str, default=None, help="CSV 编码，空则自动尝试")
    p.add_argument("--delimiter", type=str, default=None, help="CSV 分隔符，空则自动尝试逗号/制表")
    p.add_argument("--sheet", type=str, default=None, help="Excel 工作表名/索引")
    return p.parse_args()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def read_table(path: Path, encoding: str, delimiter: str, sheet: str) -> pd.DataFrame:
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
    raise RuntimeError(f"无法解析文件，最后错误: {last_err}")


def main():
    args = parse_args()
    df = read_table(args.input, args.encoding, args.delimiter, args.sheet)
    missing = [c for c in [args.patient_field, args.time_field, args.event_field] if c not in df.columns]
    if missing:
        raise KeyError(f"列缺失: {missing}; 现有列: {df.columns.tolist()}")
    df = df[[args.patient_field, args.time_field, args.event_field]]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "patient_id": str(r[args.patient_field]).strip(),
            "time": r[args.time_field],
            "event": r[args.event_field],
        })
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {len(rows)} records to {args.out}")

if __name__ == "__main__":
    main()
