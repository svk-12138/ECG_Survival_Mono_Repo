#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量将 Braveheart 的 median MAT 文件转为 CSV。
示例：
    python batch_export_medians.py ^
        --input-dir "C:/Users/17615/Documents/xwechat_files/.../2025-12" ^
        --output-dir "/home/admin123/use/Program/ECG_Survival_Mono_Repo/data/median_csv"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from tools.export_median_mat_to_csv import LEADS, extract_median_12l
except ModuleNotFoundError:  # fallback when run from tools/ directory
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from export_median_mat_to_csv import LEADS, extract_median_12l


def write_csv(data: np.ndarray, out_path: Path, sample_rate: float | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx"] + LEADS[: data.shape[0]])
        for idx in range(data.shape[1]):
            row = [idx] + [float(data[lead_idx, idx]) for lead_idx in range(data.shape[0])]
            writer.writerow(row)
    rate_display = f"{sample_rate} Hz" if sample_rate is not None else "unknown Hz"
    print(f"[OK] Saved {out_path} ({rate_display})")


def process_directory(input_dir: Path, output_dir: Path | None, overwrite: bool) -> None:
    mats = sorted(input_dir.glob("*.mat"))
    if not mats:
        print(f"[WARN] 目录 {input_dir} 中没有 .mat 文件")
        return

    for mat_path in mats:
        rel_name = mat_path.stem + ".csv"
        out_path = (output_dir or mat_path.parent).joinpath(rel_name)
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} 已存在，使用 --overwrite 才会重新生成")
            continue
        try:
            data, sample_rate = extract_median_12l(mat_path)
            write_csv(data, out_path, sample_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR ] 处理 {mat_path} 失败：{exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量导出 median MAT -> CSV")
    parser.add_argument("--input-dir", required=True, type=Path, help="含 .mat 的目录")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV 输出目录（默认写回原目录）",
    )
    parser.add_argument("--overwrite", action="store_true", help="已存在则覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_directory(args.input_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
