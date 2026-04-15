#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ECG 风险推理入口。

功能：
- 读取训练好的 checkpoint
- 读取 XML/CSV ECG
- 统一走与训练相同的预处理
- 按 `prediction/classification` 两种模式导出风险分数
- 支持 8 导和 12 导输入
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecg_survival.data_utils import SurvivalBreaks
from torch_survival.ecg_preprocessing import (
    ECGPreprocessingConfig,
    resolve_leads,
    load_csv_ecg,
    load_xml_ecg,
)
from torch_survival.model_builder import build_survival_resnet


def _load_manifest(json_path: Path) -> List[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON manifest 格式需为列表")
    return data


def _build_preprocessing_config(args: argparse.Namespace) -> ECGPreprocessingConfig:
    leads = resolve_leads(args.lead_mode)
    return ECGPreprocessingConfig(
        leads=leads,
        waveform_type=args.waveform_type,
        target_len=args.target_len,
        resample_hz=args.resample_hz,
        apply_filters=args.apply_filters,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        notch_hz=args.notch_hz,
        notch_q=args.notch_q,
        normalize=True,
    )


def _build_patient_index(xml_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for xml_file in xml_dir.rglob("*.xml"):
        try:
            root = ET.fromstring(xml_file.read_text(encoding="iso-8859-1"))
            pid = root.findtext(".//PatientDemographics/PatientID")
            if pid:
                if pid not in index or xml_file.stat().st_mtime > index[pid].stat().st_mtime:
                    index[pid] = xml_file
        except Exception:
            continue
    if not index:
        raise ValueError("XML 目录未找到任何 PatientID")
    return index


def _build_csv_index(csv_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for csv_file in csv_dir.rglob("*.csv"):
        stem = csv_file.stem
        if stem.endswith("_median"):
            stem = stem[: -len("_median")]
        if stem.endswith("_rhythm"):
            stem = stem[: -len("_rhythm")]
        pid = stem.split(".")[-1]
        pid = pid.split("_")[-1]
        if pid:
            if pid not in index or csv_file.stat().st_mtime > index[pid].stat().st_mtime:
                index[pid] = csv_file
    if not index:
        raise ValueError("CSV 目录未找到任何样本文件")
    return index


def _resolve_manifest_waveform_path(base_dir: Path, row: dict, field_name: str) -> Path | None:
    raw_value = row.get(field_name)
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    candidate = Path(text)
    path = candidate if candidate.is_absolute() else (base_dir / candidate)
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} 指向的文件不存在: {path}")
    return path


def _row_has_manifest_waveform_path(row: dict, field_name: str) -> bool:
    raw_value = row.get(field_name)
    if raw_value is None:
        return False
    return bool(str(raw_value).strip())


class ECGXMLInferDataset(Dataset):
    def __init__(self, manifest_json: Path, xml_dir: Path, preprocessing: ECGPreprocessingConfig):
        self.rows = _load_manifest(manifest_json)
        self.preprocessing = preprocessing
        self.xml_dir = xml_dir
        self.use_patient_index_fallback = any(
            not _row_has_manifest_waveform_path(row, "xml_file") for row in self.rows
        )
        self.patient_index = _build_patient_index(xml_dir) if self.use_patient_index_fallback else {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pid = str(row["patient_id"])
        xml_path = _resolve_manifest_waveform_path(self.xml_dir, row, "xml_file")
        if xml_path is None:
            if pid not in self.patient_index:
                raise FileNotFoundError(f"PatientID {pid} 在 XML 目录中未找到")
            xml_path = self.patient_index[pid]
        x = load_xml_ecg(xml_path, self.preprocessing)
        return pid, torch.from_numpy(x).float()


class ECGCSVInferDataset(Dataset):
    def __init__(self, manifest_json: Path, csv_dir: Path, preprocessing: ECGPreprocessingConfig):
        self.rows = _load_manifest(manifest_json)
        self.preprocessing = preprocessing
        self.csv_dir = csv_dir
        self.use_csv_index_fallback = any(
            not _row_has_manifest_waveform_path(row, "csv_file") for row in self.rows
        )
        self.csv_index = _build_csv_index(csv_dir) if self.use_csv_index_fallback else {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pid = str(row["patient_id"])
        csv_path = _resolve_manifest_waveform_path(self.csv_dir, row, "csv_file")
        if csv_path is None:
            if pid not in self.csv_index:
                raise FileNotFoundError(f"PatientID {pid} 在 CSV 目录中未找到")
            csv_path = self.csv_index[pid]
        x = load_csv_ecg(csv_path, self.preprocessing)
        return pid, torch.from_numpy(x).float()


def _normalize_state_dict(state: dict) -> dict:
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("0.") for k in state.keys()):
        state = {k[2:] if k.startswith("0.") else k: v for k, v in state.items()}
    if any("resblock1d_" in k for k in state.keys()):
        normalized = {}
        for key, value in state.items():
            normalized[re.sub(r"resblock1d_(\d+)", r"res_blocks.\1", key)] = value
        state = normalized
    return state


def _infer_output_dim(state: dict) -> int | None:
    for key, value in state.items():
        if key.endswith("lin.weight") and hasattr(value, "shape"):
            return int(value.shape[0])
    return None


def _resolve_prediction_interval(breaks: SurvivalBreaks, prediction_horizon: float | None) -> tuple[int, float]:
    if breaks.n_intervals <= 1:
        return 0, float(breaks.breaks[-1])
    if prediction_horizon is None:
        idx = breaks.n_intervals - 1
    else:
        idx = int(np.searchsorted(breaks.breaks[1:], prediction_horizon, side="left"))
        idx = max(0, min(idx, breaks.n_intervals - 1))
    return idx, float(breaks.breaks[idx + 1])


def _resolve_task_mode(requested: str, output_dim: int) -> str:
    if requested == "auto":
        return "classification" if output_dim == 1 else "prediction"
    if requested == "classification" and output_dim != 1:
        raise ValueError("classification 模式要求 checkpoint 输出维度为 1")
    if requested == "prediction" and output_dim <= 1:
        raise ValueError("prediction 模式要求 checkpoint 输出维度大于 1")
    return requested


def _resolve_checkpoint_from_log_dir(log_dir: Path) -> Path:
    log_dir = log_dir.resolve()
    summary_path = log_dir / "run_summary.json"
    candidates: list[Path] = []

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        for key in (
            "preferred_checkpoint",
            "legacy_checkpoint",
            "archived_best_checkpoint",
            "latest_checkpoint",
            "archived_last_checkpoint",
        ):
            raw = summary.get(key)
            if raw:
                candidates.append(Path(raw))

    candidates.extend(
        [
            log_dir / "model_best.pt",
            log_dir / "model_final.pt",
            log_dir / "model_last.pt",
        ]
    )

    checkpoint_dir = log_dir / "checkpoints"
    if checkpoint_dir.exists():
        candidates.extend(sorted(checkpoint_dir.glob("model_best_epoch_*.pt"), reverse=True))
        candidates.extend(sorted(checkpoint_dir.glob("model_last_epoch_*.pt"), reverse=True))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"在 log_dir 中未找到可用 checkpoint: {log_dir} | "
        "已尝试 model_best.pt / model_final.pt / model_last.pt / checkpoints/*.pt"
    )


def _resolve_checkpoint_path(checkpoint: str | None, log_dir: str | None) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir() and not log_dir:
            return _resolve_checkpoint_from_log_dir(checkpoint_path)
        resolved = checkpoint_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {resolved}")
        return resolved
    if log_dir:
        return _resolve_checkpoint_from_log_dir(Path(log_dir))
    raise ValueError("请提供 --checkpoint，或提供 --log-dir 让脚本自动选择最佳模型")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ECG 风险分数")
    parser.add_argument("--checkpoint", default=None, help="模型权重路径；若不填，可改用 --log-dir 自动选择最佳模型")
    parser.add_argument("--log-dir", default=None, help="训练输出目录；未显式传 --checkpoint 时，自动优先选择 model_best.pt")
    parser.add_argument("--manifest", required=True, help="JSON manifest 路径")
    parser.add_argument("--xml-dir", default=None, help="XML 目录")
    parser.add_argument("--csv-dir", default=None, help="CSV 目录（可选，优先生效）")
    parser.add_argument("--task-mode", choices=["auto", "prediction", "classification"], default="auto")
    parser.add_argument("--lead-mode", choices=["8lead", "12lead"], default="8lead", help="选择 8 导或 12 导输入")
    parser.add_argument("--n-intervals", type=int, default=40, help="离散时间区间数")
    parser.add_argument("--max-time", type=float, default=3650.0, help="训练时使用的最大时间窗口")
    parser.add_argument("--prediction-horizon", type=float, default=365.25 * 5.0, help="风险输出对应的预测时间点")
    parser.add_argument("--target-len", type=int, default=4096, help="导联长度")
    parser.add_argument("--waveform-type", default="Rhythm", help="优先使用的 XML WaveformType")
    parser.add_argument("--resample-hz", type=float, default=400.0, help="先重采样到该频率，再补零到 target_len")
    parser.add_argument("--apply-filters", dest="apply_filters", action="store_true", help="启用 ECG 滤波")
    parser.add_argument("--no-apply-filters", dest="apply_filters", action="store_false", help="禁用 ECG 滤波")
    parser.set_defaults(apply_filters=True)
    parser.add_argument("--bandpass-low-hz", type=float, default=0.5)
    parser.add_argument("--bandpass-high-hz", type=float, default=100.0)
    parser.add_argument("--notch-hz", type=float, default=60.0)
    parser.add_argument("--notch-q", type=float, default=30.0)
    parser.add_argument("--batch", type=int, default=16, help="推理 batch size")
    parser.add_argument("--device", default=None, help="设备，例如 cuda:0 / cpu")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument(
        "--save-full-curve",
        action="store_true",
        help="prediction 模式下额外输出整条 survival curve；classification 模式下输出事件概率列。",
    )
    args = parser.parse_args()
    if not args.xml_dir and not args.csv_dir:
        raise ValueError("xml_dir 和 csv_dir 至少提供一个")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    preprocessing = _build_preprocessing_config(args)
    leads = resolve_leads(args.lead_mode)
    if args.csv_dir:
        dataset = ECGCSVInferDataset(Path(args.manifest), Path(args.csv_dir), preprocessing)
    else:
        dataset = ECGXMLInferDataset(Path(args.manifest), Path(args.xml_dir), preprocessing)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, args.log_dir)
    print(f"[infer] using checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint 格式不支持，未找到 state_dict")
    state = _normalize_state_dict(state)
    output_dim = _infer_output_dim(state) or args.n_intervals
    task_mode = _resolve_task_mode(args.task_mode, output_dim)
    n_intervals = output_dim if task_mode == "prediction" else 1
    breaks = SurvivalBreaks.from_uniform(args.max_time, n_intervals)
    interval_idx, resolved_horizon = _resolve_prediction_interval(
        breaks, args.prediction_horizon if task_mode == "prediction" else None
    )

    model = build_survival_resnet(n_intervals, input_dim=(len(leads), args.target_len))
    model.load_state_dict(state)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    rows: list[dict] = []
    with torch.no_grad():
        for pids, xb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            if task_mode == "classification":
                event_prob = probs.view(-1).detach().cpu().numpy()
                for pid, score in zip(pids, event_prob):
                    row = {
                        "sample_id": pid,
                        "task_mode": task_mode,
                        "risk_horizon": np.nan,
                        "risk_score": float(score),
                    }
                    if args.save_full_curve:
                        row["p_event"] = float(score)
                    rows.append(row)
                continue

            probs = torch.clamp(probs.view(xb.size(0), n_intervals), min=1e-7, max=1.0 - 1e-7)
            survival_curve = torch.cumprod(probs, dim=1).detach().cpu().numpy()
            risk_curve = 1.0 - survival_curve
            risk_score = risk_curve[:, interval_idx]
            final_risk = risk_curve[:, -1]
            for row_idx, pid in enumerate(pids):
                row = {
                    "sample_id": pid,
                    "task_mode": task_mode,
                    "risk_horizon": float(resolved_horizon),
                    "risk_score": float(risk_score[row_idx]),
                    "risk_score_final": float(final_risk[row_idx]),
                }
                if args.save_full_curve:
                    for interval in range(n_intervals):
                        row[f"p_surv_t{interval + 1}"] = float(survival_curve[row_idx, interval])
                rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(
        f"[infer] saved {len(rows)} rows to {out_path} | "
        f"task_mode={task_mode} | leads={args.lead_mode} | "
        f"horizon={resolved_horizon if task_mode == 'prediction' else 'NA'}"
    )


if __name__ == "__main__":
    main()
