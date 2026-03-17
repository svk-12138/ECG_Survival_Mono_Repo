# -*- coding: utf-8 -*-
"""从 XML (base64) 读取 8 导联 ECG，训练 PyTorch 生存模型。
- 默认使用 8 个物理导联：I, II, V1-6（跳过导出/推导导联）。
- 支持 manifest 中使用 PatientID/住院号匹配：若未提供文件名列，会在 XML 内部读取 <PatientID> 建立索引。
- manifest 可为 CSV 或 Excel（通过 `--manifest_format` 指定），必须包含时间列与事件列（默认 `time`, `event`）。
"""
import argparse
import base64
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from ecg_survival.data_utils import SurvivalBreaks
from torch_survival.model_builder import build_survival_resnet

LEADS_KEEP = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]  # 8 个物理导联


def _read_manifest(csv_path: Path, fmt: str, encoding: str, sheet: Optional[str]) -> List[dict]:
    if fmt == "excel":
        df = pd.read_excel(csv_path, sheet_name=sheet or 0)
        rows = df.to_dict("records")
    else:
        rows = []
        with csv_path.open("r", newline="", encoding=encoding, errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    if not rows:
        raise ValueError("manifest 解析为空，请检查编码/格式/列名")
    return rows


def _resample(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] == target_len:
        return signal
    x_old = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, signal)


def _parse_xml_waveforms(xml_path: Path, target_len: int) -> np.ndarray:
    root = ET.fromstring(xml_path.read_text(encoding="iso-8859-1"))
    lead_signals = {lead: None for lead in LEADS_KEEP}
    for ld in root.findall('.//LeadData'):
        lead_id = ld.findtext('LeadID')
        if lead_id not in LEADS_KEEP:
            continue
        wf_text = ld.findtext('WaveFormData') or ""
        raw = base64.b64decode(wf_text)
        arr = np.frombuffer(raw, dtype="<i2")  # GE-MUSE 小端 int16
        prev = lead_signals[lead_id]
        if prev is None or arr.size > prev.size:
            lead_signals[lead_id] = arr

    for k, v in lead_signals.items():
        if v is None:
            raise ValueError(f"缺少导联 {k} 在 {xml_path}")

    stacked = []
    for lead in LEADS_KEEP:
        sig = lead_signals[lead].astype(np.float32)
        sig = _resample(sig, target_len)
        sig = sig - sig.mean()
        std = sig.std() + 1e-6
        sig = sig / std
        stacked.append(sig)
    return np.stack(stacked, axis=0)  # (8, target_len)


def _build_patient_index(xml_dir: Path) -> Dict[str, Path]:
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
        raise ValueError("XML 目录中未找到任何 PatientID，可检查文件格式或编码")
    return index


class ECGXMLSurvDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        manifest_format: str,
        xml_dir: Path,
        breaks: SurvivalBreaks,
        target_len: int = 4096,
        file_field: Optional[str] = "file",
        time_field: str = "time",
        event_field: str = "event",
        patient_id_field: Optional[str] = None,
        manifest_encoding: str = "utf-8",
        manifest_sheet: Optional[str] = None,
    ):
        self.xml_dir = xml_dir
        self.breaks = breaks
        self.target_len = target_len
        self.rows = _read_manifest(manifest_path, manifest_format, manifest_encoding, manifest_sheet)
        self.file_field = file_field if file_field else None
        self.time_field = time_field
        self.event_field = event_field
        self.patient_id_field = patient_id_field

        times = [float(r[self.time_field]) for r in self.rows]
        events = [int(r[self.event_field]) for r in self.rows]
        self.labels = np.array(events, dtype="float32")

        self.patient_index: Dict[str, Path] = {}
        if (not self.file_field) and self.patient_id_field:
            self.patient_index = _build_patient_index(self.xml_dir)

    def __len__(self):
        return len(self.rows)

    def _resolve_xml_path(self, row: dict) -> Path:
        if self.file_field and self.file_field in row and row[self.file_field]:
            return self.xml_dir / row[self.file_field]
        if self.patient_id_field:
            pid = row.get(self.patient_id_field)
            if pid and pid in self.patient_index:
                return self.patient_index[pid]
        raise FileNotFoundError("无法从 manifest 解析 XML 路径：需要 file 列或 patient_id 匹配")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        xml_path = self._resolve_xml_path(row)
        x = _parse_xml_waveforms(xml_path, target_len=self.target_len)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(float(y), dtype=torch.float32)

def parse_args():
    p = argparse.ArgumentParser(description="PyTorch ECG 生存分析训练（从 XML base64 导联）")
    p.add_argument("--xml_dir", type=Path, required=True, help="XML 数据目录")
    p.add_argument("--manifest", type=Path, required=True, help="CSV 或 Excel 文件")
    p.add_argument("--manifest_format", choices=["csv", "excel"], default="csv", help="manifest 格式")
    p.add_argument("--manifest_encoding", type=str, default="utf-8", help="CSV 编码，例如 utf-8 或 gbk")
    p.add_argument("--manifest_sheet", type=str, default=None, help="Excel 工作表名称/索引")
    p.add_argument("--n_intervals", type=int, default=8, help="离散时间区间数")
    p.add_argument("--max_time", type=float, default=365.0, help="最大随访时间（天）")
    p.add_argument(
        "--max_time_years",
        type=float,
        default=None,
        help="若提供则覆盖 --max_time，按年数×365.25 转换（论文示例为10年）",
    )
    p.add_argument("--target_len", type=int, default=4096, help="重采样后的每导联采样点数")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--inspect", action="store_true", help="仅打印模型结构")
    p.add_argument("--file_field", type=str, default="file", help="文件名列；为空时走 PatientID 映射")
    p.add_argument("--time_field", type=str, default="time", help="时间列名（天）")
    p.add_argument("--event_field", type=str, default="event", help="事件列名，0=无事件，1=事件")
    p.add_argument("--patient_id_field", type=str, default=None, help="PatientID 列名；当无文件名列时用于匹配 XML 内的 PatientID")
    return p.parse_args()

def main():
    args = parse_args()
    if args.n_intervals != 1:
        print(f"[binary] n_intervals={args.n_intervals} -> override to 1")
        args.n_intervals = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_time_days = float(args.max_time)
    if args.max_time_years is not None:
        max_time_days = float(args.max_time_years) * 365.25
        print(f"[info] max_time_years={args.max_time_years} -> {max_time_days:.1f} 天")
    breaks = SurvivalBreaks.from_uniform(max_time_days, args.n_intervals)
    dataset = ECGXMLSurvDataset(
        args.manifest,
        args.manifest_format,
        args.xml_dir,
        breaks,
        target_len=args.target_len,
        file_field=args.file_field,
        time_field=args.time_field,
        event_field=args.event_field,
        patient_id_field=args.patient_id_field,
        manifest_encoding=args.manifest_encoding,
        manifest_sheet=args.manifest_sheet,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)

    model = build_survival_resnet(args.n_intervals, input_dim=(len(LEADS_KEEP), args.target_len)).to(device)
    if args.inspect:
        print(model)
        return

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.float32).view(-1)
            optimizer.zero_grad()
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        avg = total / len(loader.dataset)
        print(f"Epoch {epoch+1}: loss={avg:.4f}")

if __name__ == "__main__":
    main()
