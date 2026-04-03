"""ECG 预处理工具。

统一处理 XML/CSV 波形读取、导联选择、滤波、重采样和补零。
训练和推理都应复用本文件，避免不同入口使用不同预处理口径。
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample

LEADS_KEEP_8 = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
LEADS_KEEP_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_XML_ENCODING_FALLBACKS = ("iso-8859-1", "utf-8", "utf-8-sig")

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


def _xml_encoding_candidates(preferred: str) -> list[str]:
    ordered = [(preferred or "").strip(), * _XML_ENCODING_FALLBACKS]
    result: list[str] = []
    seen: set[str] = set()
    for encoding in ordered:
        if not encoding or encoding in seen:
            continue
        seen.add(encoding)
        result.append(encoding)
    return result


def _parse_xml_root(xml_path: Path, xml_encoding: str) -> ET.Element:
    errors: list[str] = []
    for encoding in _xml_encoding_candidates(xml_encoding):
        try:
            return ET.fromstring(xml_path.read_text(encoding=encoding))
        except Exception as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    raise ValueError(f"XML 解析失败: {xml_path} | tried={'; '.join(errors)}")


def _decode_waveform_bytes(waveform_data: str, xml_path: Path, lead_id: str) -> bytes:
    compact = re.sub(r"\s+", "", waveform_data or "")
    if not compact:
        raise ValueError(f"WaveFormData 为空: {xml_path} | lead={lead_id}")

    normalized = compact.replace("-", "+").replace("_", "/")

    def _decode_base64_text(text: str, *, validate: bool) -> bytes:
        padded = text
        padding = (-len(padded)) % 4
        if padding:
            padded = padded + ("=" * padding)
        return base64.b64decode(padded, validate=validate)

    def _repair_len_mod_4_eq_1(text: str) -> bytes | None:
        if len(text) % 4 != 1 or len(text) <= 1:
            return None
        repaired = text[:-1]
        try:
            raw_bytes = _decode_base64_text(repaired, validate=False)
        except binascii.Error:
            return None
        warnings.warn(
            f"WaveFormData base64 长度为 4n+1，已自动裁掉末尾 1 个字符后解码: "
            f"{xml_path} | lead={lead_id} | text_len={len(text)}->{len(repaired)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return raw_bytes

    try:
        raw = _decode_base64_text(normalized, validate=True)
    except binascii.Error as exc:
        cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", normalized)
        removed = len(normalized) - len(cleaned)
        if not cleaned:
            raise ValueError(
                f"WaveFormData base64 解码失败，清洗后为空: "
                f"{xml_path} | lead={lead_id} | text_len={len(compact)} | {exc}"
            ) from exc
        repaired_raw = _repair_len_mod_4_eq_1(cleaned)
        if repaired_raw is not None:
            raw = repaired_raw
        else:
            try:
                raw = _decode_base64_text(cleaned, validate=False)
            except binascii.Error as cleaned_exc:
                raise ValueError(
                    f"WaveFormData base64 解码失败: {xml_path} | lead={lead_id} | "
                    f"text_len={len(compact)} | cleaned_text_len={len(cleaned)} | {cleaned_exc}"
                ) from cleaned_exc
            warnings.warn(
                f"WaveFormData 含有非 base64 字符，已自动清洗后解码: "
                f"{xml_path} | lead={lead_id} | removed_chars={removed}",
                RuntimeWarning,
                stacklevel=2,
            )

    if len(raw) % 2 != 0:
        # 医疗设备导出的 XML 偶尔会在波形末尾多出 1 个脏字节。
        # 对 16-bit little-endian ECG 而言，裁掉最后 1 个字节即可恢复可解析的采样流，
        # 比直接中断整批训练更符合医生使用场景。
        repaired = raw[:-1]
        if len(repaired) < 2:
            raise ValueError(
                f"WaveFormData 解码后字节数不是 2 的倍数，且无法修复: "
                f"{xml_path} | lead={lead_id} | bytes={len(raw)}"
            )
        warnings.warn(
            f"WaveFormData 字节数为奇数，已自动裁掉最后 1 个字节: "
            f"{xml_path} | lead={lead_id} | bytes={len(raw)}->{len(repaired)}",
            RuntimeWarning,
            stacklevel=2,
        )
        raw = repaired
    return raw


def _decode_xml_leads(xml_path: Path, leads: Sequence[str], waveform_type: str, xml_encoding: str) -> tuple[float | None, Dict[str, np.ndarray]]:
    root = _parse_xml_root(xml_path, xml_encoding)
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
        raw = _decode_waveform_bytes(waveform_data, xml_path, lead_id)
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
