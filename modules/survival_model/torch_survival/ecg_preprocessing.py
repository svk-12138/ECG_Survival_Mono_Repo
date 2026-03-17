"""ECG 预处理工具。

统一处理 XML/CSV 波形读取、导联选择、滤波、重采样和补零。
训练和推理都应复用本文件，避免不同入口使用不同预处理口径。
"""

from __future__ import annotations

import base64
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample

LEADS_KEEP_8 = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
LEADS_KEEP_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

_TIME_COLUMN_CANDIDATES = ("time_ms", "time", "time_s", "t")


@dataclass
class ECGPreprocessingConfig:
    leads: tuple[str, ...] = LEADS_KEEP_8
    waveform_type: str = "Rhythm"
    target_len: int = 4096
    resample_hz: float = 400.0
    apply_filters: bool = True
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 100.0
    notch_hz: float | None = 60.0
    notch_q: float = 30.0
    normalize: bool = True
    xml_encoding: str = "iso-8859-1"


def resolve_leads(lead_mode: str) -> tuple[str, ...]:
    """将导联模式字符串解析成固定导联顺序。

    `8lead`:
      I, II, V1-V6
    `12lead`:
      I, II, III, aVR, aVL, aVF, V1-V6
    """

    normalized = str(lead_mode).strip().lower()
    if normalized in {"8", "8lead", "8_lead", "lead8"}:
        return LEADS_KEEP_8
    if normalized in {"12", "12lead", "12_lead", "lead12"}:
        return LEADS_KEEP_12
    raise ValueError(f"不支持的 lead_mode: {lead_mode}")


def _resample_to_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] == target_len:
        return signal.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def _pad_or_trim(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] == target_len:
        return signal.astype(np.float32, copy=False)
    if signal.shape[0] > target_len:
        return signal[:target_len].astype(np.float32, copy=False)
    out = np.zeros(target_len, dtype=np.float32)
    out[: signal.shape[0]] = signal.astype(np.float32, copy=False)
    return out


def _normalize(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    signal = signal - float(signal.mean())
    std = float(signal.std())
    if std < 1e-6:
        return signal
    return signal / std


def _apply_bandpass(signal: np.ndarray, sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        return signal.astype(np.float32, copy=False)
    if low_hz <= 0 or high_hz <= low_hz or high_hz >= nyquist:
        return signal.astype(np.float32, copy=False)
    b, a = butter(3, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")
    return filtfilt(b, a, signal).astype(np.float32)


def _apply_notch(signal: np.ndarray, sample_rate: float, notch_hz: float | None, notch_q: float) -> np.ndarray:
    if notch_hz is None or notch_hz <= 0:
        return signal.astype(np.float32, copy=False)
    nyquist = 0.5 * sample_rate
    if not math.isfinite(sample_rate) or sample_rate <= 0 or notch_hz >= nyquist:
        return signal.astype(np.float32, copy=False)
    b, a = iirnotch(notch_hz / nyquist, notch_q)
    return filtfilt(b, a, signal).astype(np.float32)


def _resample_by_rate(signal: np.ndarray, sample_rate: float, resample_hz: float) -> np.ndarray:
    if not math.isfinite(sample_rate) or sample_rate <= 0 or not math.isfinite(resample_hz) or resample_hz <= 0:
        return signal.astype(np.float32, copy=False)
    if abs(sample_rate - resample_hz) < 1e-6:
        return signal.astype(np.float32, copy=False)
    target_len = max(int(round(signal.shape[0] * resample_hz / sample_rate)), 1)
    return resample(signal, target_len).astype(np.float32)


def _preprocess_signal(signal: np.ndarray, sample_rate: float | None, cfg: ECGPreprocessingConfig) -> np.ndarray:
    processed = signal.astype(np.float32, copy=False)
    if cfg.apply_filters and sample_rate is not None:
        processed = _apply_bandpass(processed, sample_rate, cfg.bandpass_low_hz, cfg.bandpass_high_hz)
        processed = _apply_notch(processed, sample_rate, cfg.notch_hz, cfg.notch_q)
    if sample_rate is not None and cfg.resample_hz:
        processed = _resample_by_rate(processed, sample_rate, cfg.resample_hz)
        processed = _pad_or_trim(processed, cfg.target_len)
    else:
        processed = _resample_to_length(processed, cfg.target_len)
    if cfg.normalize:
        processed = _normalize(processed)
        processed = _pad_or_trim(processed, cfg.target_len)
    return processed.astype(np.float32, copy=False)


def _find_waveform_node(root: ET.Element, waveform_type: str) -> ET.Element | None:
    waveform_type = (waveform_type or "").strip().lower()
    fallback = None
    fallback_score = -1
    for waveform in root.findall(".//Waveform"):
        current_type = (waveform.findtext("WaveformType") or "").strip().lower()
        if waveform_type and current_type == waveform_type:
            return waveform
        score = len(waveform.findall(".//LeadData"))
        if score > fallback_score:
            fallback = waveform
            fallback_score = score
    return fallback


def _waveform_sample_rate(waveform: ET.Element) -> float | None:
    sample_base = waveform.findtext("SampleBase")
    sample_exp = waveform.findtext("SampleExponent")
    if sample_base is None:
        return None
    try:
        base = float(sample_base)
        exponent = float(sample_exp) if sample_exp is not None else 0.0
        sample_rate = base * (10 ** exponent)
        return sample_rate if math.isfinite(sample_rate) and sample_rate > 0 else None
    except (TypeError, ValueError):
        return None


def _decode_xml_leads(xml_path: Path, leads: Sequence[str], waveform_type: str, xml_encoding: str) -> tuple[float | None, Dict[str, np.ndarray]]:
    root = ET.fromstring(xml_path.read_text(encoding=xml_encoding))
    waveform = _find_waveform_node(root, waveform_type)
    if waveform is None:
        raise ValueError(f"XML 中未找到可解析的 Waveform 节点: {xml_path}")
    sample_rate = _waveform_sample_rate(waveform)

    lead_signals: Dict[str, np.ndarray | None] = {lead: None for lead in leads}
    for lead_data in waveform.findall(".//LeadData"):
        lead_id = (lead_data.findtext("LeadID") or "").strip()
        if lead_id not in lead_signals:
            continue
        waveform_data = lead_data.findtext("WaveFormData") or ""
        raw = base64.b64decode(re.sub(r"\s+", "", waveform_data))
        signal = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        units_per_bit = float(lead_data.findtext("LeadAmplitudeUnitsPerBit") or 1.0)
        signal = signal * units_per_bit
        previous = lead_signals[lead_id]
        if previous is None or signal.shape[0] > previous.shape[0]:
            lead_signals[lead_id] = signal

    missing = [lead for lead, signal in lead_signals.items() if signal is None]
    if missing:
        raise ValueError(f"缺少导联 {missing} in {xml_path}")
    return sample_rate, {lead: lead_signals[lead] for lead in leads if lead_signals[lead] is not None}


def load_xml_ecg(xml_path: Path, cfg: ECGPreprocessingConfig) -> np.ndarray:
    sample_rate, lead_signals = _decode_xml_leads(xml_path, cfg.leads, cfg.waveform_type, cfg.xml_encoding)
    stacked = [_preprocess_signal(lead_signals[lead], sample_rate, cfg) for lead in cfg.leads]
    return np.stack(stacked, axis=0)


def _infer_sample_rate_from_time(values: np.ndarray, column_name: str) -> float | None:
    if values.shape[0] < 2:
        return None
    diffs = np.diff(values.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    step = float(np.median(diffs))
    if step <= 0:
        return None
    column_name = column_name.lower()
    if column_name.endswith("_ms"):
        return 1000.0 / step
    if column_name.endswith("_s"):
        return 1.0 / step
    if step > 5.0:
        return 1000.0 / step
    return 1.0 / step


def _find_time_column(columns: Iterable[str]) -> str | None:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in _TIME_COLUMN_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return None


def load_csv_ecg(csv_path: Path, cfg: ECGPreprocessingConfig) -> np.ndarray:
    df = pd.read_csv(csv_path)
    cols = {str(col).strip().upper(): col for col in df.columns}
    time_col = _find_time_column(df.columns)
    sample_rate = None
    if time_col is not None:
        sample_rate = _infer_sample_rate_from_time(df[time_col].to_numpy(dtype=np.float32), time_col)

    lead_signals: Dict[str, np.ndarray] = {}
    for lead in cfg.leads:
        col = cols.get(lead.upper())
        if col is None:
            raise ValueError(f"缺少导联 {lead} in {csv_path}")
        lead_signals[lead] = df[col].to_numpy(dtype=np.float32)
    stacked = [_preprocess_signal(lead_signals[lead], sample_rate, cfg) for lead in cfg.leads]
    return np.stack(stacked, axis=0)


__all__ = [
    "ECGPreprocessingConfig",
    "LEADS_KEEP_8",
    "LEADS_KEEP_12",
    "load_csv_ecg",
    "load_xml_ecg",
    "resolve_leads",
]
