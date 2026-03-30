# -*- coding: utf-8 -*-
"""ECG 生存/分类训练主入口。

功能：
1. 读取 `patient_id + time + event` 的 JSON manifest。
   若 manifest 额外包含 `xml_file`/`csv_file`，则按该条目精确绑定每次 ECG 检查。
2. 从 XML 或 CSV 读取 ECG，并走统一预处理。
3. 通过 `task_mode` 控制：
   - `prediction`: 离散时间生存预测
   - `classification`: 单输出二分类
4. 支持 8 导或 12 导输入。

说明：
- 该文件是当前项目的主训练实现，后续医生/学生优先使用这里对应的封装脚本
  `scripts/run_survival_training.py`，不要再维护多套训练逻辑。
"""
import csv
import json
import re
import time
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Literal, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))

from ecg_survival.data_utils import SurvivalBreaks, make_surv_targets
from torch_survival.ecg_preprocessing import (
    ECGPreprocessingConfig,
    load_csv_ecg,
    load_xml_ecg,
    resolve_leads,
)
from torch_survival.losses import SurvLikelihoodLoss
from torch_survival.model_builder import build_survival_resnet

# ====================== 超参数区域 ======================
# 把你的数据路径与训练配置集中放在这里。
DEFAULT_XML_DIR = Path("data/sample_xml")  # TODO: 替换为你自己的 ECG XML 目录
DEFAULT_MANIFEST = next(
    (p for p in Path("data").glob("*.json") if "manifest" in p.stem.lower()),
    Path("data/sample_manifest.json"),
)
DEFAULT_NUM_INTERVALS = 40  # 论文默认离散时间区间数
DEFAULT_MAX_TIME = 3650.0  # 默认 10 年窗口（天）
DEFAULT_TARGET_LEN = 4096  # 每条导联重采样后的长度
DEFAULT_BATCH_SIZE = 10 # batchsize批处理数
DEFAULT_EPOCHS = 0 # 训练轮次
DEFAULT_LR = 5e-4 # 起始学习率
DEFAULT_NUM_WORKERS = 1 # 数据加载线程数，以你电脑能够承受的数值进行设置
DEFAULT_DROPOUT = 0.5  # ResNet1d dropout
DEFAULT_WEIGHT_DECAY = 0.0  # 优化器权重衰减
DEFAULT_SCHED_TMAX = DEFAULT_EPOCHS  # CosineAnnealingLR T_max
DEFAULT_EVAL_THRESHOLD = 0.5  # 评估阶段将风险概率转二分类的阈值
DEFAULT_POS_WEIGHT_MULT = 3.0  # 正类权重倍率（基于 neg/pos 再乘该系数）
DEFAULT_EARLY_STOP_PATIENCE = 10
DEFAULT_EARLY_STOP_MIN_DELTA = 1e-4
DEFAULT_EARLY_STOP_METRIC = "auto"  # classification->PR AUC, prediction->C-index
DEFAULT_DEVICE = None  # 例如 "cuda:0"，None 表示自动检测
DEFAULT_INSPECT = False  # 是否打印模型结构后退出
DEFAULT_LOG_DIR = Path("training_logs")  # 训练日志与曲线保存目录
DEFAULT_USE_DATA_PARALLEL = False  # 是否启用 DataParallel 多 GPU
DEFAULT_DEVICE_IDS = None  # 例如 [0,1,2,3]，为空时使用所有可见 GPU
DEFAULT_CV_FOLDS = 1  # K-fold cross validation (1 = disable)
DEFAULT_CV_SEED = 42  # Random seed for CV splits
DEFAULT_TRAIN_RATIO = 0.8  # 留出法训练集比例
DEFAULT_VAL_RATIO = 0.2  # 留出法验证集比例
DEFAULT_TEST_RATIO = 0.0  # 留出法测试集比例，允许为 0
DEFAULT_CSV_DIR = None
DEFAULT_TASK_MODE = "prediction"
DEFAULT_LEAD_MODE = "8lead"
DEFAULT_PREDICTION_HORIZON = 365.25 * 5.0  # 论文预测评估使用 5 年风险
DEFAULT_WAVEFORM_TYPE = "Rhythm"
DEFAULT_RESAMPLE_HZ = 400.0
DEFAULT_APPLY_FILTERS = True
DEFAULT_BANDPASS_LOW_HZ = 0.5
DEFAULT_BANDPASS_HIGH_HZ = 100.0
DEFAULT_NOTCH_HZ = 60.0
DEFAULT_NOTCH_Q = 30.0
BEST_PARAMS_PATH = DEFAULT_LOG_DIR / "best_params.json"
# =====================================================

_THRESH_GRID = np.linspace(0.05, 0.95, 19)

@dataclass
class TrainConfig:
    xml_dir: Path = DEFAULT_XML_DIR
    csv_dir: Path | None = DEFAULT_CSV_DIR
    manifest: Path = DEFAULT_MANIFEST
    task_mode: Literal["prediction", "classification"] = DEFAULT_TASK_MODE
    lead_mode: str = DEFAULT_LEAD_MODE
    n_intervals: int = DEFAULT_NUM_INTERVALS
    max_time: float = DEFAULT_MAX_TIME
    prediction_horizon: float | None = DEFAULT_PREDICTION_HORIZON
    target_len: int = DEFAULT_TARGET_LEN
    waveform_type: str = DEFAULT_WAVEFORM_TYPE
    resample_hz: float = DEFAULT_RESAMPLE_HZ
    apply_filters: bool = DEFAULT_APPLY_FILTERS
    bandpass_low_hz: float = DEFAULT_BANDPASS_LOW_HZ
    bandpass_high_hz: float = DEFAULT_BANDPASS_HIGH_HZ
    notch_hz: float | None = DEFAULT_NOTCH_HZ
    notch_q: float = DEFAULT_NOTCH_Q
    batch: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    num_workers: int = DEFAULT_NUM_WORKERS
    dropout: float = DEFAULT_DROPOUT
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    sched_tmax: int = DEFAULT_SCHED_TMAX
    eval_threshold: float = DEFAULT_EVAL_THRESHOLD
    pos_weight_mult: float = DEFAULT_POS_WEIGHT_MULT
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    early_stop_min_delta: float = DEFAULT_EARLY_STOP_MIN_DELTA
    early_stop_metric: str = DEFAULT_EARLY_STOP_METRIC
    log_dir: Path = DEFAULT_LOG_DIR
    device: str | None = DEFAULT_DEVICE
    inspect: bool = DEFAULT_INSPECT
    use_data_parallel: bool = DEFAULT_USE_DATA_PARALLEL
    device_ids: list[int] | None = DEFAULT_DEVICE_IDS
    cv_folds: int = DEFAULT_CV_FOLDS
    cv_seed: int = DEFAULT_CV_SEED
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO

def _apply_best_params(cfg: TrainConfig) -> None:
    if not BEST_PARAMS_PATH.exists():
        return
    try:
        params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if "epochs" in params and "sched_tmax" not in params:
        cfg.sched_tmax = cfg.epochs


def get_default_config() -> TrainConfig:
    cfg = TrainConfig()
    _apply_best_params(cfg)
    return cfg

def _load_manifest(json_path: Path) -> List[dict]:
    """读取 manifest 文件，返回所有样本条目的列表。"""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON manifest 格式需为列表")
    return data


def _build_preprocessing_config(cfg: TrainConfig) -> ECGPreprocessingConfig:
    leads = resolve_leads(cfg.lead_mode)
    return ECGPreprocessingConfig(
        leads=leads,
        waveform_type=cfg.waveform_type,
        target_len=cfg.target_len,
        resample_hz=cfg.resample_hz,
        apply_filters=cfg.apply_filters,
        bandpass_low_hz=cfg.bandpass_low_hz,
        bandpass_high_hz=cfg.bandpass_high_hz,
        notch_hz=cfg.notch_hz,
        notch_q=cfg.notch_q,
        normalize=True,
    )


def _build_patient_index(xml_dir: Path) -> Dict[str, Path]:
    """遍历 XML 目录，建立 patient_id → 最新 XML 文件的映射。"""

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


def _build_csv_index(csv_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for csv_file in csv_dir.glob("*.csv"):
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


def _compute_pos_weight(events: np.ndarray, indices: list[int] | np.ndarray | None = None) -> float:
    if indices is not None:
        subset = events[indices]
    else:
        subset = events
    pos = float(np.sum(subset))
    neg = float(len(subset) - pos)
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


def _metrics_at_threshold(events_arr: np.ndarray, scores_arr: np.ndarray, threshold: float) -> dict:
    preds_binary = (scores_arr >= threshold).astype(int)
    precision = float(precision_score(events_arr, preds_binary, zero_division=0))
    recall = float(recall_score(events_arr, preds_binary, zero_division=0))
    f1 = float(f1_score(events_arr, preds_binary, zero_division=0))
    accuracy = float(accuracy_score(events_arr, preds_binary))
    balanced_acc = float(balanced_accuracy_score(events_arr, preds_binary))
    tn, fp, fn, tp = confusion_matrix(events_arr, preds_binary, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_acc": balanced_acc,
        "specificity": specificity,
    }


def _find_best_threshold(events_arr: np.ndarray, scores_arr: np.ndarray) -> dict:
    if len(np.unique(events_arr)) < 2:
        return {"threshold": 0.5, "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "accuracy": float("nan"), "balanced_acc": float("nan"),
                "specificity": float("nan")}
    best = None
    for thr in _THRESH_GRID:
        cur = _metrics_at_threshold(events_arr, scores_arr, thr)
        if best is None or cur["f1"] > best["f1"]:
            best = cur
    return best or {"threshold": 0.5, "precision": float("nan"), "recall": float("nan"),
                    "f1": float("nan"), "accuracy": float("nan"), "balanced_acc": float("nan"),
                    "specificity": float("nan")}


def _get_early_stop_score(val_metrics: dict, metric: str, task_mode: str) -> float:
    if metric == "auto":
        metric = "val_c_index" if task_mode == "prediction" else "val_pr_auc"
    if metric == "val_loss":
        loss = val_metrics.get("loss", float("nan"))
        return -float(loss) if loss is not None and not np.isnan(loss) else float("-inf")
    if metric == "val_c_index":
        return float(val_metrics.get("c_index", float("-inf")))
    if metric == "val_best_f1":
        return float(val_metrics.get("best_f1", float("-inf")))
    if metric == "val_auc":
        return float(val_metrics.get("auc", float("-inf")))
    # default: val_pr_auc
    return float(val_metrics.get("pr_auc", float("-inf")))


class _BaseECGDataset(Dataset):
    def __init__(
        self,
        manifest_json: Path,
        breaks: SurvivalBreaks,
        preprocessing: ECGPreprocessingConfig,
        task_mode: str,
    ):
        self.breaks = breaks
        self.preprocessing = preprocessing
        self.task_mode = task_mode
        self.rows = _load_manifest(manifest_json)
        required = {"patient_id", "time", "event"}
        patient_ids: list[str] = []
        group_ids: list[str] = []
        used_patient_sn = False
        missing_patient_sn = False
        for row in self.rows:
            if not required.issubset(row.keys()):
                raise KeyError(f"JSON 条目缺少字段: {required}")
            patient_id = str(row["patient_id"]).strip()
            if not patient_id:
                raise ValueError("JSON 条目中的 patient_id 不能为空")
            row["patient_id"] = patient_id
            patient_sn = str(row.get("patient_SN", "")).strip() if "patient_SN" in row else ""
            if patient_sn:
                row["patient_SN"] = patient_sn
                group_ids.append(patient_sn)
                used_patient_sn = True
            else:
                group_ids.append(patient_id)
                missing_patient_sn = True
            patient_ids.append(patient_id)
        times = np.array([float(r["time"]) for r in self.rows], dtype=np.float32)
        events = np.array([int(r["event"]) for r in self.rows], dtype=np.int64)
        self.events = events.astype(np.int64)
        self.times = times
        self.patient_ids = np.array(patient_ids, dtype=object)
        self.group_ids = np.array(group_ids, dtype=object)
        if used_patient_sn and not missing_patient_sn:
            self.group_field = "patient_SN"
        elif used_patient_sn:
            self.group_field = "patient_SN_or_patient_id"
        else:
            self.group_field = "patient_id"
        self.group_to_indices: Dict[str, list[int]] = {}
        for idx, group_id in enumerate(group_ids):
            self.group_to_indices.setdefault(group_id, []).append(idx)
        self.unique_group_ids = np.array(list(self.group_to_indices.keys()), dtype=object)
        # Backward compatibility for older helper code/tests.
        self.unique_patient_ids = self.unique_group_ids
        self.group_events = np.array(
            [int(self.events[self.group_to_indices[group_id]].max()) for group_id in self.unique_group_ids],
            dtype=np.int64,
        )
        if task_mode == "prediction":
            self.targets = make_surv_targets(times, events, breaks).astype(np.float32)
        else:
            self.targets = events.astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def _target_tensor(self, idx: int) -> torch.Tensor:
        target = self.targets[idx]
        return torch.from_numpy(target).float() if isinstance(target, np.ndarray) else torch.tensor(float(target), dtype=torch.float32)

    def subset_indices_for_groups(self, group_ids: np.ndarray | list[str]) -> np.ndarray:
        indices: list[int] = []
        for group_id in group_ids:
            indices.extend(self.group_to_indices[str(group_id)])
        return np.array(sorted(indices), dtype=np.int64)

    def subset_indices_for_patients(self, patient_ids: np.ndarray | list[str]) -> np.ndarray:
        return self.subset_indices_for_groups(patient_ids)


class ECGXMLSurvDataset(_BaseECGDataset):
    """将 manifest 元数据与 XML ECG 波形配对的自定义数据集。"""

    def __init__(
        self,
        manifest_json: Path,
        xml_dir: Path,
        breaks: SurvivalBreaks,
        preprocessing: ECGPreprocessingConfig,
        task_mode: str,
    ):
        super().__init__(manifest_json, breaks, preprocessing, task_mode)
        self.xml_dir = xml_dir
        self.patient_index = _build_patient_index(self.xml_dir)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        pid = str(row["patient_id"])
        xml_path = _resolve_manifest_waveform_path(self.xml_dir, row, "xml_file")
        if xml_path is None:
            if pid not in self.patient_index:
                raise FileNotFoundError(f"PatientID {pid} 在 XML 目录中未找到")
            xml_path = self.patient_index[pid]
        x = load_xml_ecg(xml_path, self.preprocessing)
        event = self.events[idx]
        time_value = self.times[idx]
        return (
            torch.from_numpy(x).float(),
            self._target_tensor(idx),
            torch.tensor(event, dtype=torch.float32),
            torch.tensor(time_value, dtype=torch.float32),
        )


class ECGCSVSurvDataset(_BaseECGDataset):
    """将 manifest 与 CSV 波形配对的自定义数据集。"""

    def __init__(
        self,
        manifest_json: Path,
        csv_dir: Path,
        breaks: SurvivalBreaks,
        preprocessing: ECGPreprocessingConfig,
        task_mode: str,
    ):
        super().__init__(manifest_json, breaks, preprocessing, task_mode)
        self.csv_dir = csv_dir
        self.csv_index = _build_csv_index(self.csv_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        pid = str(row["patient_id"])
        csv_path = _resolve_manifest_waveform_path(self.csv_dir, row, "csv_file")
        if csv_path is None:
            if pid not in self.csv_index:
                raise FileNotFoundError(f"PatientID {pid} 在 CSV 目录中未找到")
            csv_path = self.csv_index[pid]
        x = load_csv_ecg(csv_path, self.preprocessing)
        event = self.events[idx]
        time_value = self.times[idx]
        return (
            torch.from_numpy(x).float(),
            self._target_tensor(idx),
            torch.tensor(event, dtype=torch.float32),
            torch.tensor(time_value, dtype=torch.float32),
        )


def _empty_metrics() -> dict:
    return {
        metric: float("nan")
        for metric in (
            "loss",
            "auc",
            "pr_auc",
            "c_index",
            "accuracy",
            "balanced_acc",
            "precision",
            "recall",
            "specificity",
            "f1",
            "brier",
            "best_threshold",
            "best_precision",
            "best_recall",
            "best_specificity",
            "best_f1",
            "best_accuracy",
            "best_balanced_acc",
        )
    }


def _resolve_prediction_interval(breaks: SurvivalBreaks, prediction_horizon: float | None) -> tuple[int, float]:
    if breaks.n_intervals <= 1:
        return 0, float(breaks.breaks[-1])
    if prediction_horizon is None:
        idx = breaks.n_intervals - 1
    else:
        idx = int(np.searchsorted(breaks.breaks[1:], prediction_horizon, side="left"))
        idx = max(0, min(idx, breaks.n_intervals - 1))
    return idx, float(breaks.breaks[idx + 1])


def _event_target_for_metrics(events_arr: np.ndarray, time_arr: np.ndarray, horizon: float | None) -> np.ndarray:
    if horizon is None or not np.isfinite(horizon):
        return events_arr.astype(int)
    return ((events_arr == 1) & (time_arr <= horizon)).astype(int)


def _scores_from_logits(
    logits: torch.Tensor,
    task_mode: str,
    breaks: SurvivalBreaks,
    prediction_horizon: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    probs = torch.sigmoid(logits)
    if task_mode == "classification":
        score = probs.view(-1).detach().cpu().numpy()
        return score, score.reshape(-1, 1)
    interval_idx, _ = _resolve_prediction_interval(breaks, prediction_horizon)
    surv_probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    survival_curve = torch.cumprod(surv_probs, dim=1)
    risk_curve = 1.0 - survival_curve
    score = risk_curve[:, interval_idx].detach().cpu().numpy()
    return score, survival_curve.detach().cpu().numpy()


def evaluate(
    model,
    loader,
    criterion,
    device,
    task_mode: str,
    breaks: SurvivalBreaks,
    prediction_horizon: float | None,
    threshold: float = DEFAULT_EVAL_THRESHOLD,
):
    """计算指定数据加载器上的 loss、AUC、C-index、F1、Recall、Brier 指标。"""

    if loader is None or len(loader.dataset) == 0:
        return _empty_metrics()

    model.eval()
    total_loss = 0.0
    total_samples = 0
    scores: List[float] = []
    events: List[int] = []
    times: List[float] = []
    with torch.no_grad():
        for xb, yb, eb, tb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            eb = eb.to(device=device)
            tb = tb.to(device=device)
            logits = model(xb).view(-1)
            if task_mode == "prediction":
                logits = logits.view(xb.size(0), breaks.n_intervals)
                loss = criterion(torch.sigmoid(logits), yb)
            else:
                logits = logits.view(-1)
                loss = criterion(logits, yb.view(-1))
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            batch_scores, _ = _scores_from_logits(logits, task_mode, breaks, prediction_horizon)
            scores.extend(batch_scores.tolist())
            events.extend(eb.detach().cpu().numpy().tolist())
            times.extend(tb.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    events_arr = np.array(events)
    times_arr = np.array(times)
    scores_arr = np.array(scores)
    _, horizon = _resolve_prediction_interval(breaks, prediction_horizon)
    metric_targets = _event_target_for_metrics(events_arr, times_arr, horizon if task_mode == "prediction" else None)
    if len(np.unique(metric_targets)) > 1:
        auc = float(roc_auc_score(metric_targets, scores_arr))
        pr_auc = float(average_precision_score(metric_targets, scores_arr))
    else:
        auc = float("nan")
        pr_auc = float("nan")
    fixed = _metrics_at_threshold(metric_targets, scores_arr, threshold)
    best = _find_best_threshold(metric_targets, scores_arr)
    brier = float(np.mean((scores_arr - metric_targets) ** 2))
    c_index = _concordance_index(times_arr, events_arr, scores_arr)
    return {
        "loss": avg_loss,
        "auc": auc,
        "pr_auc": pr_auc,
        "c_index": c_index,
        "accuracy": fixed["accuracy"],
        "balanced_acc": fixed["balanced_acc"],
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "specificity": fixed["specificity"],
        "f1": fixed["f1"],
        "brier": brier,
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_specificity": best["specificity"],
        "best_f1": best["f1"],
        "best_accuracy": best["accuracy"],
        "best_balanced_acc": best["balanced_acc"],
    }


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    """校验留出法比例配置。"""

    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"{name}_ratio 不能小于 0，收到: {value}")
    if ratios["train"] <= 0:
        raise ValueError("train_ratio 必须大于 0")
    if ratios["val"] <= 0:
        raise ValueError("val_ratio 必须大于 0")
    total = sum(ratios.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio 必须等于 1.0，当前为 {total:.6f}"
        )
    return ratios["train"], ratios["val"], ratios["test"]



def _compute_split_lengths(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """根据比例计算 train/val/test 长度，允许 test 为 0。"""

    if n < 2:
        raise ValueError("数据量至少需要 2 条，才能同时划分训练集和验证集")

    names = ("train", "val", "test")
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    mins = {
        "train": 1 if train_ratio > 0 else 0,
        "val": 1 if val_ratio > 0 else 0,
        "test": 1 if test_ratio > 0 else 0,
    }
    required = sum(mins.values())
    if required > n:
        raise ValueError(
            f"当前样本数 {n} 无法满足 train/val/test 的非空划分要求；"
            "请减小 test_ratio 或增加数据量。"
        )

    raw = {name: ratios[name] * n for name in names}
    lengths = {name: int(np.floor(raw[name])) for name in names}
    for name in names:
        lengths[name] = max(lengths[name], mins[name])

    total = sum(lengths.values())
    while total > n:
        candidates = [name for name in ("train", "test", "val") if lengths[name] > mins[name]]
        if not candidates:
            raise ValueError("划分失败，请检查 train/val/test 比例设置。")
        remove_name = max(candidates, key=lambda name: (lengths[name] - raw[name], lengths[name]))
        lengths[remove_name] -= 1
        total -= 1

    while total < n:
        add_name = max(
            names,
            key=lambda name: (raw[name] - lengths[name], raw[name], -names.index(name)),
        )
        lengths[add_name] += 1
        total += 1

    return lengths["train"], lengths["val"], lengths["test"]



def _split_dataset(
    dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
    """按患者分组切分训练/验证/测试集，避免同一患者泄露到多个子集。"""

    train_ratio, val_ratio, test_ratio = _validate_split_ratios(
        train_ratio,
        val_ratio,
        test_ratio,
    )
    groups = _dataset_groups(dataset)
    group_events = dataset.group_events
    n_groups = len(groups)
    train_group_len, val_group_len, test_group_len = _compute_split_lengths(
        n_groups,
        train_ratio,
        val_ratio,
        test_ratio,
    )

    group_indices = np.arange(n_groups)

    def _split_group_indices(source_indices: np.ndarray, train_size: int, test_size: int, split_seed: int):
        if train_size == 0:
            return np.array([], dtype=np.int64), source_indices.copy()
        if test_size == 0:
            return source_indices.copy(), np.array([], dtype=np.int64)
        source_labels = group_events[source_indices]
        unique, counts = np.unique(source_labels, return_counts=True)
        use_stratify = (
            len(unique) > 1
            and counts.min() >= 2
            and train_size >= len(unique)
            and test_size >= len(unique)
        )
        stratify = source_labels if use_stratify else None
        train_idx, test_idx = train_test_split(
            source_indices,
            train_size=train_size,
            test_size=test_size,
            random_state=split_seed,
            shuffle=True,
            stratify=stratify,
        )
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)

    train_group_idx, remaining_group_idx = _split_group_indices(
        group_indices,
        train_group_len,
        val_group_len + test_group_len,
        seed,
    )

    if test_group_len > 0:
        val_group_idx, test_group_idx = _split_group_indices(
            remaining_group_idx,
            val_group_len,
            test_group_len,
            seed + 1,
        )
    else:
        val_group_idx = remaining_group_idx
        test_group_idx = np.array([], dtype=np.int64)

    train_idx = _subset_indices_for_groups(dataset, groups[train_group_idx])
    val_idx = _subset_indices_for_groups(dataset, groups[val_group_idx])
    test_idx = _subset_indices_for_groups(dataset, groups[test_group_idx])
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def _normalize_device_ids(value) -> list[int] | None:
    """Parse device ids from list or comma/space separated string."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
        return [int(p) for p in parts] if parts else None
    return None


def _validate_device_ids(device_ids: list[int], available: int) -> None:
    invalid = [i for i in device_ids if i < 0 or i >= available]
    if invalid:
        raise ValueError(f"device_ids 包含非法 GPU 索引: {invalid} (available: 0..{available-1})")


def _event_stats(events: np.ndarray, indices: np.ndarray) -> Tuple[int, int, float]:
    """统计某个子集的样本数、事件数、事件比例。"""

    subset = events[indices]
    count = int(len(subset))
    num_events = int(subset.sum())
    ratio = num_events / count if count else 0.0
    return count, num_events, ratio


def _concordance_index(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    """简单 C-index 计算（ties 计 0.5）。"""

    n = len(time)
    if n == 0:
        return float("nan")
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 0 and event[j] == 0:
                continue
            if time[i] == time[j]:
                continue
            den += 1
            if time[i] < time[j]:
                ci, cj = i, j
            else:
                ci, cj = j, i
            if score[ci] > score[cj]:
                num += 1
            elif score[ci] == score[cj]:
                num += 0.5
    return num / den if den > 0 else float("nan")


def _dataset_group_field(dataset) -> str:
    return str(getattr(dataset, "group_field", "patient_id"))


def _dataset_groups(dataset) -> np.ndarray:
    groups = getattr(dataset, "unique_group_ids", None)
    if groups is not None:
        return groups
    return getattr(dataset, "unique_patient_ids")


def _subset_indices_for_groups(dataset, groups: np.ndarray | list[str]) -> np.ndarray:
    if hasattr(dataset, "subset_indices_for_groups"):
        return dataset.subset_indices_for_groups(groups)
    return dataset.subset_indices_for_patients(groups)


def _make_cv_splits(dataset: _BaseECGDataset, n_splits: int, seed: int):
    """按患者分组生成 K-fold indices，避免同一患者泄露到多个折。"""

    groups = _dataset_groups(dataset)
    group_events = dataset.group_events
    n_groups = len(groups)
    if n_groups < n_splits:
        group_field = _dataset_group_field(dataset)
        raise ValueError(
            f"唯一 {group_field} 分组数量不足以做 {n_splits} 折交叉验证："
            f"当前分组数={n_groups}"
        )

    group_indices = np.arange(n_groups)
    unique, counts = np.unique(group_events, return_counts=True)
    use_stratified = len(unique) > 1 and counts.min() >= n_splits
    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        group_splits = list(splitter.split(group_indices, group_events))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        group_splits = list(splitter.split(group_indices))

    sample_splits = []
    for train_group_idx, val_group_idx in group_splits:
        train_idx = _subset_indices_for_groups(dataset, groups[train_group_idx])
        val_idx = _subset_indices_for_groups(dataset, groups[val_group_idx])
        sample_splits.append((train_idx, val_idx))
    return sample_splits, use_stratified


def _group_count(dataset: _BaseECGDataset, indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    values = getattr(dataset, "group_ids", getattr(dataset, "patient_ids"))
    return int(len(set(str(values[int(idx)]) for idx in indices)))


def _patient_count(dataset: _BaseECGDataset, indices: np.ndarray) -> int:
    return _group_count(dataset, indices)


def _train_model(
    cfg: TrainConfig,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log_dir: Path,
    fold_label: str | None = None,
    device_ids: list[int] | None = None,
    pos_weight: float | None = None,
):
    """训练单个划分（train/val），返回模型与最后一轮指标。"""

    fold_prefix = f"[Fold {fold_label}] " if fold_label else ""
    breaks = SurvivalBreaks.from_uniform(cfg.max_time, cfg.n_intervals)
    leads = resolve_leads(cfg.lead_mode)
    model = build_survival_resnet(
        cfg.n_intervals,
        input_dim=(len(leads), cfg.target_len),
        dropout_rate=cfg.dropout,
    )
    model = model.to(device=device, dtype=torch.float32)
    if cfg.use_data_parallel and device.type == "cuda":
        device_ids = device_ids or list(range(torch.cuda.device_count()))
        if device_ids and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            print(f"{fold_prefix}[GPU] DataParallel enabled on devices: {device_ids}")
        else:
            print(f"{fold_prefix}[GPU] DataParallel requested but only one GPU is visible.")
    if cfg.inspect:
        print(model)
        return {"model": model, "train": None, "val": None, "history": []}

    if cfg.task_mode == "prediction":
        criterion = SurvLikelihoodLoss(cfg.n_intervals)
    elif pos_weight is not None:
        pos_weight_t = torch.tensor([float(pos_weight)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t_max = cfg.sched_tmax if cfg.sched_tmax > 0 else max(cfg.epochs, 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    metrics_history: List[dict] = []
    model.train()

    best_threshold_info: dict | None = None
    best_score = float("-inf")
    no_improve = 0

    if cfg.epochs <= 0:
        train_metrics = evaluate(
            model,
            train_eval_loader,
            criterion,
            device,
            task_mode=cfg.task_mode,
            breaks=breaks,
            prediction_horizon=cfg.prediction_horizon,
            threshold=cfg.eval_threshold,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            task_mode=cfg.task_mode,
            breaks=breaks,
            prediction_horizon=cfg.prediction_horizon,
            threshold=cfg.eval_threshold,
        )
        selection_score = val_metrics["c_index"] if cfg.task_mode == "prediction" else val_metrics["best_f1"]
        if val_metrics:
            best_threshold_info = {
                "epoch": 0,
                "best_threshold": val_metrics["best_threshold"],
                "best_precision": val_metrics["best_precision"],
                "best_recall": val_metrics["best_recall"],
                "best_specificity": val_metrics["best_specificity"],
                "best_f1": val_metrics["best_f1"],
                "best_accuracy": val_metrics["best_accuracy"],
                "best_balanced_acc": val_metrics["best_balanced_acc"],
                "val_c_index": val_metrics["c_index"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_auc": val_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "selection_score": selection_score,
            }
    else:
        train_metrics = None
        val_metrics = None
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0
            seen = 0
            desc = f"{fold_prefix}Epoch {epoch}/{cfg.epochs}"
            progress = tqdm(train_loader, desc=desc, unit="batch")
            for xb, yb, _, _ in progress:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                logits = model(xb)
                if cfg.task_mode == "prediction":
                    logits = logits.view(xb.size(0), cfg.n_intervals)
                    loss = criterion(torch.sigmoid(logits), yb)
                else:
                    logits = logits.view(-1)
                    loss = criterion(logits, yb.view(-1))
                loss.backward()
                optimizer.step()
                batch_size = xb.size(0)
                running_loss += loss.item() * batch_size
                seen += batch_size
                progress.set_postfix(loss=f"{loss.item():.4f}")
            epoch_loss = running_loss / max(seen, 1)
            elapsed = time.time() - epoch_start

            train_metrics = evaluate(
                model,
                train_eval_loader,
                criterion,
                device,
                task_mode=cfg.task_mode,
                breaks=breaks,
                prediction_horizon=cfg.prediction_horizon,
                threshold=cfg.eval_threshold,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                task_mode=cfg.task_mode,
                breaks=breaks,
                prediction_horizon=cfg.prediction_horizon,
                threshold=cfg.eval_threshold,
            )

            if val_metrics:
                selection_score = val_metrics["c_index"] if cfg.task_mode == "prediction" else val_metrics["best_f1"]
                if best_threshold_info is None or selection_score > best_threshold_info.get("selection_score", float("-inf")):
                    best_threshold_info = {
                        "epoch": epoch,
                        "best_threshold": val_metrics["best_threshold"],
                        "best_precision": val_metrics["best_precision"],
                        "best_recall": val_metrics["best_recall"],
                        "best_specificity": val_metrics["best_specificity"],
                        "best_f1": val_metrics["best_f1"],
                        "best_accuracy": val_metrics["best_accuracy"],
                        "best_balanced_acc": val_metrics["best_balanced_acc"],
                        "val_c_index": val_metrics["c_index"],
                        "val_pr_auc": val_metrics["pr_auc"],
                        "val_auc": val_metrics["auc"],
                        "val_loss": val_metrics["loss"],
                        "selection_score": selection_score,
                    }

                current_score = _get_early_stop_score(val_metrics, cfg.early_stop_metric, cfg.task_mode)
                if current_score > best_score + cfg.early_stop_min_delta:
                    best_score = current_score
                    no_improve = 0
                else:
                    no_improve += 1

            metrics_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "train_pr_auc": train_metrics["pr_auc"],
                "train_c_index": train_metrics["c_index"],
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_acc": train_metrics["balanced_acc"],
                "train_precision": train_metrics["precision"],
                "train_f1": train_metrics["f1"],
                "train_recall": train_metrics["recall"],
                "train_specificity": train_metrics["specificity"],
                "train_brier": train_metrics["brier"],
                "train_best_threshold": train_metrics["best_threshold"],
                "train_best_precision": train_metrics["best_precision"],
                "train_best_recall": train_metrics["best_recall"],
                "train_best_specificity": train_metrics["best_specificity"],
                "train_best_f1": train_metrics["best_f1"],
                "train_best_accuracy": train_metrics["best_accuracy"],
                "train_best_balanced_acc": train_metrics["best_balanced_acc"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_c_index": val_metrics["c_index"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_precision": val_metrics["precision"],
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
                "val_brier": val_metrics["brier"],
                "val_best_threshold": val_metrics["best_threshold"],
                "val_best_precision": val_metrics["best_precision"],
                "val_best_recall": val_metrics["best_recall"],
                "val_best_specificity": val_metrics["best_specificity"],
                "val_best_f1": val_metrics["best_f1"],
                "val_best_accuracy": val_metrics["best_accuracy"],
                "val_best_balanced_acc": val_metrics["best_balanced_acc"],
            }
            metrics_history.append(metrics_record)

            print(
                f"{fold_prefix}[Epoch {epoch}/{cfg.epochs}] train_loss={train_metrics['loss']:.4f} "
                f"train_auc={train_metrics['auc']:.4f} train_pr_auc={train_metrics['pr_auc']:.4f} "
                f"train_cidx={train_metrics['c_index']:.4f} train_f1={train_metrics['f1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
                f"val_cidx={val_metrics['c_index']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_best_thr={val_metrics['best_threshold']:.2f} "
                f"val_best_f1={val_metrics['best_f1']:.4f} | time={elapsed:.1f}s"
            )

            model.train()
            scheduler.step()

            if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
                print(
                    f"{fold_prefix}[early-stop] metric={cfg.early_stop_metric} "
                    f"patience={cfg.early_stop_patience} best_score={best_score:.4f}"
                )
                break

    if metrics_history:
        _log_and_plot(metrics_history, log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = log_dir / "model_final.pt"
    save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(save_model.state_dict(), ckpt_path)
    print(f"{fold_prefix}[save] model saved to {ckpt_path}")
    if best_threshold_info is not None:
        thresh_path = log_dir / "best_threshold.json"
        thresh_path.write_text(
            json.dumps(best_threshold_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{fold_prefix}[save] best threshold saved to {thresh_path}")

    return {"model": model, "train": train_metrics, "val": val_metrics, "history": metrics_history}

def _log_and_plot(history: List[dict], log_dir: Path):
    """将历史指标写入 CSV 并绘制训练曲线。"""

    import matplotlib.pyplot as plt

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_metrics.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_pr_auc",
        "train_c_index",
        "train_accuracy",
        "train_balanced_acc",
        "train_precision",
        "train_f1",
        "train_recall",
        "train_specificity",
        "train_brier",
        "train_best_threshold",
        "train_best_precision",
        "train_best_recall",
        "train_best_specificity",
        "train_best_f1",
        "train_best_accuracy",
        "train_best_balanced_acc",
        "val_loss",
        "val_auc",
        "val_pr_auc",
        "val_c_index",
        "val_accuracy",
        "val_balanced_acc",
        "val_precision",
        "val_f1",
        "val_recall",
        "val_specificity",
        "val_brier",
        "val_best_threshold",
        "val_best_precision",
        "val_best_recall",
        "val_best_specificity",
        "val_best_f1",
        "val_best_accuracy",
        "val_best_balanced_acc",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    epochs = [row["epoch"] for row in history]
    for metric in ["loss", "auc", "pr_auc", "c_index", "f1", "recall", "precision", "specificity", "brier", "best_f1"]:
        fig, ax = plt.subplots()
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="train")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} over epochs")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(log_dir / f"{metric}_curve.png")
        plt.close(fig)


def _run_cross_validation(
    cfg: TrainConfig,
    dataset: _BaseECGDataset,
    device: torch.device,
    device_ids: list[int] | None,
) -> dict:
    """执行 K-fold 交叉验证训练与汇总。"""

    splits, stratified = _make_cv_splits(dataset, cfg.cv_folds, cfg.cv_seed)
    group_field = _dataset_group_field(dataset)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CV] {cfg.cv_folds}-fold | stratified={stratified} | seed={cfg.cv_seed} | group_field={group_field}")

    fold_results: List[dict] = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, 1):
        torch.manual_seed(cfg.cv_seed + fold_id)
        fold_dir = cfg.log_dir / f"fold_{fold_id:02d}"

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        train_eval_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        train_count, train_events, train_ratio = _event_stats(dataset.events, train_idx)
        val_count, val_events, val_ratio = _event_stats(dataset.events, val_idx)
        train_groups = _group_count(dataset, train_idx)
        val_groups = _group_count(dataset, val_idx)
        print(
            f"[Fold {fold_id}] train={train_count} samples / {train_groups} groups({group_field}) "
            f"events={train_events} ({train_ratio:.2%}) | "
            f"val={val_count} samples / {val_groups} groups({group_field}) "
            f"events={val_events} ({val_ratio:.2%})"
        )

        pos_weight = _compute_pos_weight(dataset.events, np.array(train_idx)) * cfg.pos_weight_mult
        print(f"[Fold {fold_id}] pos_weight={pos_weight:.3f} (mult={cfg.pos_weight_mult})")
        result = _train_model(
            cfg,
            train_loader,
            train_eval_loader,
            val_loader,
            device,
            fold_dir,
            fold_label=str(fold_id),
            device_ids=device_ids,
            pos_weight=pos_weight,
        )
        fold_results.append(
            {
                "fold": fold_id,
                "train": result["train"],
                "val": result["val"],
                "log_dir": str(fold_dir),
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {}
    for metric in (
        "loss",
        "auc",
        "pr_auc",
        "c_index",
        "accuracy",
        "balanced_acc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "brier",
        "best_threshold",
        "best_f1",
    ):
        values = [r["val"][metric] for r in fold_results if r["val"] is not None]
        metrics[f"val_{metric}_mean"] = float(np.nanmean(values)) if values else float("nan")
        metrics[f"val_{metric}_std"] = float(np.nanstd(values)) if values else float("nan")

    best_fold = None
    best_score = -np.inf
    for row in fold_results:
        val = row.get("val") or {}
        preferred = val.get("c_index") if cfg.task_mode == "prediction" else val.get("auc")
        loss = val.get("loss")
        score = -np.inf
        if preferred is not None and not np.isnan(preferred):
            score = float(preferred)
        elif loss is not None and not np.isnan(loss):
            score = -float(loss)
        if score > best_score:
            best_score = score
            best_fold = row

    best_checkpoint = None
    if best_fold:
        ckpt_candidate = Path(best_fold["log_dir"]) / "model_final.pt"
        if ckpt_candidate.exists():
            best_checkpoint = cfg.log_dir / "model_final.pt"
            shutil.copy2(ckpt_candidate, best_checkpoint)
            print(f"[CV] best fold -> {best_fold['fold']} (checkpoint copied to {best_checkpoint})")

    summary = {
        "cv_folds": cfg.cv_folds,
        "stratified": bool(stratified),
        "seed": cfg.cv_seed,
        "metrics": metrics,
        "fold_results": fold_results,
        "best_fold": best_fold["fold"] if best_fold else None,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
    }
    summary_path = cfg.log_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CV] summary saved to {summary_path}")
    for metric in (
        "loss",
        "auc",
        "pr_auc",
        "c_index",
        "accuracy",
        "balanced_acc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "brier",
        "best_threshold",
        "best_f1",
    ):
        mean = metrics.get(f"val_{metric}_mean", float("nan"))
        std = metrics.get(f"val_{metric}_std", float("nan"))
        print(f"[CV] val_{metric}: mean={mean:.4f} std={std:.4f}")
    return summary


def run_training(cfg: TrainConfig) -> dict:
    """训练主流程：数据加载 -> 划分 -> 训练 -> 评估 -> 记录。"""

    cfg.task_mode = str(cfg.task_mode).strip().lower()
    if cfg.task_mode not in {"prediction", "classification"}:
        raise ValueError(f"task_mode 必须为 prediction 或 classification，收到: {cfg.task_mode}")
    leads = resolve_leads(cfg.lead_mode)
    if cfg.task_mode == "classification":
        if cfg.n_intervals != 1:
            print(f"[classification] n_intervals={cfg.n_intervals} -> override to 1")
            cfg.n_intervals = 1
    elif cfg.n_intervals <= 1:
        raise ValueError("prediction 模式要求 n_intervals > 1")

    device_ids = _normalize_device_ids(cfg.device_ids)
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.use_data_parallel and device.type == "cuda":
        available = torch.cuda.device_count()
        if device_ids is None:
            device_ids = list(range(available))
        else:
            _validate_device_ids(device_ids, available)
        if device_ids:
            device = torch.device(f"cuda:{device_ids[0]}")
    breaks = SurvivalBreaks.from_uniform(cfg.max_time, cfg.n_intervals)
    preprocessing = _build_preprocessing_config(cfg)
    if cfg.csv_dir:
        dataset = ECGCSVSurvDataset(cfg.manifest, cfg.csv_dir, breaks, preprocessing, cfg.task_mode)
    else:
        dataset = ECGXMLSurvDataset(cfg.manifest, cfg.xml_dir, breaks, preprocessing, cfg.task_mode)
    group_field = _dataset_group_field(dataset)

    if cfg.cv_folds and cfg.cv_folds > 1:
        print("[split] 已启用交叉验证，train_ratio/val_ratio/test_ratio 将被忽略。")
        return _run_cross_validation(cfg, dataset, device, device_ids)

    train_set, val_set, test_set = _split_dataset(
        dataset,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.cv_seed,
    )

    # 第二步：构建数据加载器，包含训练、评估、验证、测试
    train_loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers)
    train_eval_loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers)

    # 打印数据规模与事件比例，方便快速了解数据集
    num_samples = len(dataset)
    num_groups = len(_dataset_groups(dataset))
    num_events = int(dataset.events.sum())
    event_ratio = num_events / num_samples if num_samples else 0.0
    print(
        f"Dataset: {num_samples} samples | groups({group_field}): {num_groups} | events: {num_events} ({event_ratio:.2%})"
        f" | leads:{len(leads)} ({cfg.lead_mode})"
        f" | split_ratio -> train:{cfg.train_ratio:.2f} val:{cfg.val_ratio:.2f} test:{cfg.test_ratio:.2f}"
        f" | splits -> train:{len(train_set)} val:{len(val_set)} test:{len(test_set)}"
    )
    if isinstance(train_set, Subset) and isinstance(val_set, Subset) and isinstance(test_set, Subset):
        print(
            f"[split] groups({group_field}) -> train:{_group_count(dataset, np.array(train_set.indices))} "
            f"val:{_group_count(dataset, np.array(val_set.indices))} "
            f"test:{_group_count(dataset, np.array(test_set.indices))}"
        )

    train_indices = train_set.indices if isinstance(train_set, Subset) else None
    pos_weight = _compute_pos_weight(
        dataset.events, np.array(train_indices) if train_indices is not None else None
    ) * cfg.pos_weight_mult
    if cfg.task_mode == "classification":
        print(f"[train] pos_weight={pos_weight:.3f} (mult={cfg.pos_weight_mult})")
    result = _train_model(
        cfg,
        train_loader,
        train_eval_loader,
        val_loader,
        device,
        cfg.log_dir,
        device_ids=device_ids,
        pos_weight=pos_weight,
    )
    if cfg.inspect:
        return {}
    model = result["model"]
    train_metrics = result["train"]
    val_metrics = result["val"]

    criterion = SurvLikelihoodLoss(cfg.n_intervals) if cfg.task_mode == "prediction" else nn.BCEWithLogitsLoss()
    if len(test_set) == 0:
        test_metrics = None
        print("[Test] 已跳过：test_ratio=0，当前未划分测试集。")
    else:
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            task_mode=cfg.task_mode,
            breaks=breaks,
            prediction_horizon=cfg.prediction_horizon,
            threshold=cfg.eval_threshold,
        )
        print(
            f"[Test] loss={test_metrics['loss']:.4f} auc={test_metrics['auc']:.4f} "
            f"pr_auc={test_metrics['pr_auc']:.4f} c_index={test_metrics['c_index']:.4f} "
            f"f1={test_metrics['f1']:.4f} "
            f"brier={test_metrics['brier']:.4f}"
        )
    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


def main():
    run_training(get_default_config())

if __name__ == "__main__":
    main()
