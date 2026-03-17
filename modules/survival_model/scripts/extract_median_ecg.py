#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从心电 XML 文件中提取所有导联的 R 峰，并按照 R 峰对齐的方式截取 ±window_ms 毫秒的波形段，
最终对每个采样点取中位数，得到稳健的中位心搏。该脚本主要依赖 numpy、scipy、matplotlib 和 xml.etree.ElementTree。
输出为 CSV 文件，首列为采样点索引，其余列为各导联的中位波形。同时可选地输出与原始波形对齐的可视化对比图。
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from base64 import b64decode
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def _dtype_from_sample_size(sample_size: int) -> np.dtype:
    """根据 SampleSize 返回合适的 numpy dtype（默认小端）。"""
    if sample_size == 1:
        return np.int8
    if sample_size == 2:
        return np.dtype("<i2")
    if sample_size == 3:
        # 3 字节不常见，转为 int32 再右移
        return np.dtype("<i4")
    if sample_size == 4:
        return np.dtype("<i4")
    raise ValueError(f"不支持的 SampleSize: {sample_size}")


def parse_leads(xml_path: Path, waveform_type: str) -> Dict[str, Dict[str, np.ndarray]]:
    """解析指定 WaveformType 的导联数据，返回 {lead_name: {"sample_rate": float, "data": np.ndarray}}。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    leads: Dict[str, Dict[str, np.ndarray]] = {}

    waveform_node = None
    for wf in root.findall(".//Waveform"):
        w_type = wf.findtext("WaveformType", "").strip()
        if w_type.lower() == waveform_type.lower():
            waveform_node = wf
            break
    if waveform_node is None:
        raise ValueError(f"XML 中未找到 WaveformType = {waveform_type} 的节点")

    sample_base_text = waveform_node.findtext("SampleBase")
    if not sample_base_text:
        raise ValueError("Waveform 节点缺少 SampleBase 字段，无法确定采样率")
    sample_rate = float(sample_base_text)

    for lead in waveform_node.findall(".//LeadData"):
        lead_id = lead.findtext("LeadID")
        if not lead_id:
            continue
        sample_count_text = lead.findtext("LeadSampleCountTotal")
        sample_size_text = lead.findtext("LeadSampleSize")
        units_per_bit = float(lead.findtext("LeadAmplitudeUnitsPerBit", "1"))
        data_text = lead.findtext("WaveFormData")
        if not data_text or not sample_count_text or not sample_size_text:
            continue
        sample_count = int(sample_count_text)
        sample_size = int(sample_size_text)
        raw_bytes = b64decode(re.sub(r"\\s+", "", data_text.strip()))

        dtype = _dtype_from_sample_size(sample_size)
        arr = np.frombuffer(raw_bytes, dtype=dtype)
        if sample_size == 3:
            arr = arr.view(np.uint8).reshape(-1, 4)
            arr = arr[:, :3].astype(np.int32)
        if len(arr) < sample_count:
            raise ValueError(f"导联 {lead_id} 数据长度不足 {sample_count}")
        arr = arr[:sample_count].astype(np.float32) * units_per_bit
        leads[lead_id] = {
            "sample_rate": sample_rate,
            "data": arr,
        }
    if not leads:
        raise ValueError("XML 文件中未解析到任何导联数据，请检查结构。")
    return leads


def detect_r_peaks(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """在参考导联上检测 R 峰，返回峰值索引。"""
    distance = max(1, int(0.25 * sample_rate))  # 最小 RR 间隔约 250ms
    prominence = max(0.1, 0.5 * np.std(signal))  # 根据波形振幅自适应
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    return peaks


def extract_segments(data: np.ndarray, peaks: Sequence[int], half_window: int) -> List[np.ndarray]:
    """基于 R 峰索引截取等长心搏片段，过短/越界的片段会被丢弃。"""
    segments: List[np.ndarray] = []
    total = len(data)
    for peak in peaks:
        start = peak - half_window
        end = peak + half_window
        if start < 0 or end > total:
            continue
        segments.append(data[start:end])
    return segments


def compute_median_waveforms(
    leads: Dict[str, Dict[str, np.ndarray]],
    reference_lead: str,
    window_ms: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """根据参考导联的 R 峰，对所有导联计算中位心搏。"""
    if reference_lead not in leads:
        reference_lead = next(iter(leads))
    sample_rate = leads[reference_lead]["sample_rate"]
    half_window_samples = int(window_ms * sample_rate / 1000)
    if half_window_samples <= 0:
        raise ValueError("窗口长度过小，无法截取片段。")

    peaks = detect_r_peaks(leads[reference_lead]["data"], sample_rate)
    if len(peaks) == 0:
        raise ValueError("参考导联未检测到 R 峰，请检查波形或调整参数。")

    waveforms: Dict[str, np.ndarray] = {}
    reference_segment: np.ndarray | None = None
    selected_peak = int(peaks[0])
    for lead_name, payload in leads.items():
        data = payload["data"]
        sr = payload["sample_rate"]
        if not math.isclose(sr, sample_rate):
            raise ValueError(f"导联 {lead_name} 的采样率 {sr} 与参考导联 {sample_rate} 不一致。")
        segments = extract_segments(data, peaks, half_window_samples)
        if not segments:
            raise ValueError(f"导联 {lead_name} 没有可用的心搏片段。")
        stacked = np.vstack(segments)
        waveforms[lead_name] = np.median(stacked, axis=0)
        if lead_name == reference_lead and reference_segment is None:
            reference_segment = stacked[0]

    time_axis = (np.arange(-half_window_samples, half_window_samples) / sample_rate) * 1000.0
    if reference_segment is None:
        reference_segment = np.zeros(2 * half_window_samples, dtype=np.float32)
    reference_signal = leads[reference_lead]["data"]
    return waveforms, reference_segment, time_axis, reference_signal, sample_rate, selected_peak, half_window_samples


def save_csv(output_path: Path, waveforms: Dict[str, np.ndarray], time_axis_ms: np.ndarray) -> None:
    """将结果写入 CSV 文件，首列为相对时间（ms），其他列为导联。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lead_names = sorted(waveforms.keys())
    length = len(time_axis_ms)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(",".join(["time_ms"] + lead_names) + "\n")
        for idx in range(length):
            row = [f"{time_axis_ms[idx]:.3f}"] + [f"{waveforms[name][idx]:.6f}" for name in lead_names]
            f.write(",".join(row) + "\n")


def plot_comparison(
    raw_segment: np.ndarray,
    median_waveform: np.ndarray,
    time_axis_ms: np.ndarray,
    output_path: Path,
    full_signal: np.ndarray,
    sample_rate: float,
    peak_index: int,
    half_window_samples: int,
) -> None:
    """绘制原始参考心搏与中位心搏，并保持相同坐标轴。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))

    time_full = np.arange(len(full_signal)) / sample_rate * 1000.0
    start_ms = (peak_index - half_window_samples) / sample_rate * 1000.0
    end_ms = (peak_index + half_window_samples) / sample_rate * 1000.0
    start_ms = max(0.0, start_ms)
    end_ms = min(time_full[-1], end_ms)
    peak_ms = peak_index / sample_rate * 1000.0

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time_full, full_signal, color="tab:blue", linewidth=0.8)
    ax1.axvspan(start_ms, end_ms, color="orange", alpha=0.2, label="Window around R peak")
    ax1.axvline(peak_ms, color="tab:red", linestyle="--", linewidth=1.2, label="Detected R peak")
    ax1.set_title("Reference Lead - Full Rhythm")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(time_axis_ms[: len(raw_segment)], raw_segment, color="tab:blue", label="Raw Beat")
    ax2.plot(time_axis_ms, median_waveform, color="tab:red", label="Median Beat", linewidth=2)
    ax2.set_title("Reference Lead - Median vs Raw Beat")
    ax2.set_xlabel("Time relative to R peak (ms)")
    ax2.set_ylabel("Amplitude (uV)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="从心电 XML 中提取中位心搏波形。")
    parser.add_argument("--input-xml", required=True, type=Path, help="输入 XML 文件路径")
    parser.add_argument("--output-csv", required=True, type=Path, help="输出 CSV 文件路径")
    parser.add_argument("--waveform-type", type=str, default="Rhythm", help="解析的 WaveformType（如 Rhythm、Median）")
    parser.add_argument("--reference-lead", type=str, default="II", help="R 峰检测参考导联名称")
    parser.add_argument("--window-ms", type=int, default=300, help="R 峰前后窗口长度（毫秒）")
    parser.add_argument("--plot-path", type=Path, help="可选：保存对比图的路径")
    args = parser.parse_args()

    leads = parse_leads(args.input_xml, args.waveform_type)
    waveforms, ref_segment, time_axis, ref_signal, sr, peak_idx, half_window = compute_median_waveforms(
        leads, args.reference_lead, args.window_ms
    )
    save_csv(args.output_csv, waveforms, time_axis)
    if args.plot_path:
        if args.reference_lead in waveforms:
            ref_waveform = waveforms[args.reference_lead]
        else:
            ref_waveform = next(iter(waveforms.values()))
        plot_comparison(
            ref_segment,
            ref_waveform,
            time_axis,
            args.plot_path,
            full_signal=ref_signal,
            sample_rate=sr,
            peak_index=peak_idx,
            half_window_samples=half_window,
        )


if __name__ == "__main__":
    main()
