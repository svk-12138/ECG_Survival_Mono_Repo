#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理指定目录下的 ECG XML 文件，调用 extract_median_ecg.py 对每个文件生成中位心搏 CSV 与对比图。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm

def run_single(
    script_path: Path,
    input_xml: Path,
    output_csv: Path,
    plot_path: Path | None,
    reference_lead: str,
    waveform_type: str,
    window_ms: int,
) -> None:
    """调用单个脚本处理一个 XML。"""
    cmd = [
        "python",
        str(script_path),
        "--input-xml",
        str(input_xml),
        "--output-csv",
        str(output_csv),
        "--reference-lead",
        reference_lead,
        "--waveform-type",
        waveform_type,
        "--window-ms",
        str(window_ms),
    ]
    if plot_path:
        cmd.extend(["--plot-path", str(plot_path)])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量提取中位心搏 CSV")
    parser.add_argument("--input-dir", required=True, type=Path, help="存放 XML 的目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出 CSV 的根目录")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="可选：输出对比图目录，若不指定则不保存图像",
    )
    parser.add_argument("--reference-lead", default="II", help="R 峰参考导联")
    parser.add_argument("--waveform-type", default="Rhythm", help="WaveformType (Rhythm/Median 等)")
    parser.add_argument("--window-ms", type=int, default=300, help="R 峰前后窗口毫秒数")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("extract_median_ecg.py"),
        help="单文件处理脚本路径",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(args.input_dir.glob("*.xml"))
    success = 0
    failures: list[Path] = []
    for xml_path in tqdm(xml_files, desc="Processing XML files", unit="file"):
        out_csv = args.output_dir / f"{xml_path.stem}_median.csv"
        plot_path = None
        if args.plot_dir:
            plot_path = args.plot_dir / f"{xml_path.stem}_median.png"
        try:
            run_single(
                args.script,
                xml_path,
                out_csv,
                plot_path,
                reference_lead=args.reference_lead,
                waveform_type=args.waveform_type,
                window_ms=args.window_ms,
            )
            success += 1
        except subprocess.CalledProcessError:
            failures.append(xml_path)

    print(f"\nProcessed {success}/{len(xml_files)} files successfully.")
    if failures:
        print("Failed files:")
        for item in failures:
            print(f" - {item}")


if __name__ == "__main__":
    main()
