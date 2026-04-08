#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将单个 ECG XML 样本画成多导联波形图，便于人工检查数据是否正常。

本脚本用于排查医生提供的 XML 是否能被正确解析，以及波形形态是否像真实心电。
它会自动兼容当前项目已经支持的两类 WaveFormData：
1. base64 编码波形
2. 空格分隔的明文整数波形

输出:
- 一张 PNG 预览图
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.survival_model.torch_survival.ecg_preprocessing import (  # noqa: E402
    LEADS_KEEP_12,
    _decode_waveform_signal,
    _find_waveform_node,
    _parse_xml_root,
    _waveform_sample_rate,
)


def _normalize_lead_id(lead_id: str) -> str:
    return str(lead_id).strip().upper()


def _canonical_lead_id(lead_id: str) -> str:
    raw = str(lead_id).strip()
    mapping = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }
    return mapping.get(_normalize_lead_id(raw), raw)


def _preferred_lead_order(lead_names: list[str]) -> list[str]:
    preferred = {name.upper(): idx for idx, name in enumerate(LEADS_KEEP_12)}
    return sorted(
        lead_names,
        key=lambda name: (preferred.get(name.upper(), 10_000), name.upper()),
    )


def _load_waveform_signals(xml_path: Path, waveform_type: str, xml_encoding: str) -> tuple[float | None, dict[str, np.ndarray]]:
    root = _parse_xml_root(xml_path, xml_encoding)
    waveform = _find_waveform_node(root, waveform_type)
    if waveform is None:
        raise ValueError(f"XML 中未找到可解析的 Waveform 节点: {xml_path}")

    sample_rate = _waveform_sample_rate(waveform)
    lead_signals: dict[str, np.ndarray] = {}
    for lead_data in waveform.findall(".//LeadData"):
        raw_lead_id = (lead_data.findtext("LeadID") or "").strip()
        if not raw_lead_id:
            continue
        lead_id = _canonical_lead_id(raw_lead_id)
        signal = _decode_waveform_signal(lead_data.findtext("WaveFormData") or "", xml_path, raw_lead_id)
        units_per_bit = float(lead_data.findtext("LeadAmplitudeUnitsPerBit") or 1.0)
        signal = signal.astype(np.float32) * units_per_bit
        previous = lead_signals.get(lead_id)
        if previous is None or signal.shape[0] > previous.shape[0]:
            lead_signals[lead_id] = signal
    if not lead_signals:
        raise ValueError(f"XML 中未解析到任何导联: {xml_path}")
    return sample_rate, {lead: lead_signals[lead] for lead in _preferred_lead_order(list(lead_signals))}


def _plot_waveforms(
    xml_path: Path,
    output_path: Path,
    sample_rate: float | None,
    lead_signals: dict[str, np.ndarray],
    seconds: float | None,
) -> None:
    lead_names = list(lead_signals.keys())
    cols = 2 if len(lead_names) <= 8 else 3
    rows = math.ceil(len(lead_names) / cols)
    panel_w = 760
    panel_h = 220
    margin_x = 28
    margin_y = 28
    title_h = 60
    canvas_w = cols * panel_w + (cols + 1) * margin_x
    canvas_h = title_h + rows * panel_h + (rows + 1) * margin_y
    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    sample_rate_text = f"{sample_rate:.2f} Hz" if sample_rate and sample_rate > 0 else "unknown"
    draw.text(
        (margin_x, 10),
        f"{xml_path.name}\nleads={len(lead_names)} | sample_rate={sample_rate_text}",
        fill="black",
        font=font,
    )

    for idx, lead_name in enumerate(lead_names):
        row = idx // cols
        col = idx % cols
        left = margin_x + col * (panel_w + margin_x)
        top = title_h + margin_y + row * (panel_h + margin_y)
        right = left + panel_w
        bottom = top + panel_h

        inner_left = left + 42
        inner_top = top + 22
        inner_right = right - 14
        inner_bottom = bottom - 28

        signal = lead_signals[lead_name]
        if sample_rate and sample_rate > 0:
            max_samples = signal.shape[0]
            if seconds and seconds > 0:
                max_samples = min(max_samples, int(round(sample_rate * seconds)))
            signal = signal[:max_samples]
            x_label = "time (s)"
        else:
            if seconds and seconds > 0:
                signal = signal[: int(seconds)]
            x_label = "samples"

        if signal.size == 0:
            continue

        ymin = float(np.min(signal))
        ymax = float(np.max(signal))
        if not math.isfinite(ymin) or not math.isfinite(ymax):
            ymin, ymax = -1.0, 1.0
        if abs(ymax - ymin) < 1e-6:
            ymin -= 1.0
            ymax += 1.0
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad

        draw.rectangle((left, top, right, bottom), outline="#999999", width=1)
        draw.rectangle((inner_left, inner_top, inner_right, inner_bottom), outline="#d0d0d0", width=1)
        draw.text((left + 8, top + 4), lead_name, fill="black", font=font)
        draw.text((left + 8, bottom - 18), x_label, fill="#555555", font=font)

        if ymin <= 0 <= ymax:
            zero_y = inner_bottom - (0.0 - ymin) / (ymax - ymin) * (inner_bottom - inner_top)
            draw.line((inner_left, zero_y, inner_right, zero_y), fill="#dddddd", width=1)

        points: list[tuple[float, float]] = []
        width = max(inner_right - inner_left, 1)
        height = max(inner_bottom - inner_top, 1)
        if signal.size == 1:
            points.append((inner_left, inner_top + height / 2))
        else:
            for point_idx, value in enumerate(signal):
                x = inner_left + point_idx / (signal.size - 1) * width
                y = inner_bottom - (float(value) - ymin) / (ymax - ymin) * height
                points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill="#0b57d0", width=1)

        draw.text((inner_left, inner_top - 14), f"max={ymax:.1f}", fill="#666666", font=font)
        draw.text((inner_left, inner_bottom + 2), f"min={ymin:.1f}", fill="#666666", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview ECG XML waveform and save a PNG.")
    parser.add_argument("--xml", type=Path, required=True, help="要预览的 XML 文件路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 PNG 路径")
    parser.add_argument("--waveform-type", type=str, default="Rhythm", help="优先选择的 WaveformType")
    parser.add_argument("--xml-encoding", type=str, default="iso-8859-1", help="首选 XML 文本编码")
    parser.add_argument("--seconds", type=float, default=10.0, help="仅显示前多少秒；<=0 表示全长")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_rate, lead_signals = _load_waveform_signals(args.xml.resolve(), args.waveform_type, args.xml_encoding)
    preview_seconds = None if args.seconds <= 0 else float(args.seconds)
    _plot_waveforms(args.xml.resolve(), args.output.resolve(), sample_rate, lead_signals, preview_seconds)
    print(f"[OK] 预览图已保存: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
