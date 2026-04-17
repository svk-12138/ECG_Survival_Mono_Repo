#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Survival Model Wrapper / 生存模型训练封装
=============================================

本脚本封装 `modules.survival_model.torch_survival.train_survival_from_json`
暴露命令行参数与 YAML 配置，方便在 pipeline 中调用。

推荐使用方式：
  Win11 主入口：scripts\train_stroke_thesis.bat
  Linux/macOS/WSL 主入口：bash scripts/train_stroke_thesis.sh
  底层调试用法：python3 scripts/run_survival_training.py --help

设计目标：
- 论文场景优先通过脚本入口统一收敛启动参数
- Win11 医生用户优先走 bat + PowerShell
- 底层 Python 入口保留给开发者调试与 pipeline 调用
- 通过 task_mode / lead_mode 控制主要实验分支
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def configure_console_encoding() -> None:
    """Prefer UTF-8 console IO so Windows users do not see garbled Chinese logs."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

configure_console_encoding()

from modules.survival_model.torch_survival.train_survival_from_json import (
    TrainConfig,
    BEST_PARAMS_PATH,
    get_default_config,
    run_training,
)


# ==================== 模型预设系统 ====================
# 基于模型选型规则，为不同样本量推荐合适的模型架构和超参数
MODEL_PRESETS = {
    # 传统ML基线（不使用深度学习，仅供对比）
    "xgboost": {
        "description": "XGBoost基线（需单独运行 scripts/baseline_ml.py）",
        "model_type": None,  # 不适用
    },

    # 小样本场景（1200样本）
    "tcn_light": {
        "description": "TCN轻量版（参数量~25k，适合1200样本）",
        "model_type": "tcn_light",
        "batch": 16,
        "epochs": 100,
        "lr": 0.001,
        "dropout": 0.3,
        "weight_decay": 0.0001,
        "n_intervals": 15,
    },

    # 中等样本场景（1万样本）
    "resnet_small": {
        "description": "ResNet1d小版（参数量~13.5万，适合1万样本）",
        "model_type": "resnet",
        "resnet_blocks_dim": [[12, 1024], [24, 256], [32, 64], [48, 16]],
        "batch": 32,
        "epochs": 80,
        "lr": 0.0005,
        "dropout": 0.5,
        "weight_decay": 0.0001,
        "n_intervals": 20,
    },

    # 大样本场景（10万+样本）
    "resnet_standard": {
        "description": "ResNet1d标准版（参数量~386万，适合10万+样本，论文同款）",
        "model_type": "resnet",
        "resnet_blocks_dim": None,
        "batch": 64,
        "epochs": 100,
        "lr": 0.0003,
        "dropout": 0.8,
        "weight_decay": 0.0001,
        "n_intervals": 40,
    },

    # 实验性模型
    "cnn_transformer": {
        "description": "CNN+Transformer（参数量~69万，实验性，需要足够样本）",
        "model_type": "cnn_transformer",
        "batch": 32,
        "epochs": 80,
        "lr": 0.0005,
        "dropout": 0.3,
        "weight_decay": 0.0001,
        "n_intervals": 20,
    },
}


def get_model_preset(name: str) -> dict:
    """获取模型预设配置"""
    if name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"未知的模型预设: {name}。可用预设: {available}")
    return MODEL_PRESETS[name].copy()


def list_model_presets():
    """列出所有可用的模型预设"""
    print("\n可用的模型预设：")
    print("=" * 80)
    for name, preset in MODEL_PRESETS.items():
        desc = preset.get("description", "")
        model_type = preset.get("model_type", "N/A")
        print(f"  {name:20s} - {desc}")
        if model_type:
            print(f"                       模型架构: {model_type}")
    print("=" * 80)
    print("\n使用方法：")
    print("  1. 在 configs/train_stroke_thesis.env 中设置: MODEL_NAME=tcn_light")
    print("  2. 或命令行: python scripts/run_survival_training.py --model-name tcn_light ...")
    print()
# ======================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survival training wrapper loading TrainConfig overrides."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file containing TrainConfig overrides.")
    # 模型预设（优先级高于 --model-type，会自动展开为对应超参数）
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "模型预设名称，自动设置 model_type 和对应超参数。"
            f" 可选: {', '.join(MODEL_PRESETS.keys())}。"
            " 设置后会覆盖 --model-type / --batch / --dropout / --n-intervals 等参数。"
        ),
    )
    parser.add_argument("--list-models", action="store_true", default=False,
                        help="列出所有可用的模型预设后退出。")
    # 固定划分文件
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "固定数据集划分文件路径（JSON）。"
            " 文件存在时直接加载，确保每次训练使用完全相同的 train/val/test 组成；"
            " 文件不存在时随机划分后自动保存，供后续复用。"
        ),
    )
    parser.add_argument("--xml-dir", type=Path, default=None, help="ECG XML directory.")
    parser.add_argument("--csv-dir", type=Path, default=None, help="ECG CSV directory (optional).")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON manifest path.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["resnet", "cnn_transformer", "tcn_light"],
        default=None,
        help="模型架构：resnet（论文同款）| cnn_transformer（实验性）| tcn_light（轻量版，适合小样本）",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=["prediction", "classification"],
        default=None,
        help="prediction=离散时间风险预测；classification=单输出二分类。",
    )
    parser.add_argument(
        "--lead-mode",
        type=str,
        choices=["8lead", "12lead"],
        default=None,
        help="选择 8 导或 12 导输入。",
    )
    parser.add_argument("--n-intervals", type=int, default=None, help="Number of survival intervals.")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum time horizon.")
    parser.add_argument("--prediction-horizon", type=float, default=None, help="Risk horizon used for prediction evaluation/inference.")
    parser.add_argument("--target-len", type=int, default=None, help="Resampled lead length.")
    parser.add_argument("--waveform-type", type=str, default=None, help="Preferred ECG waveform type, default Rhythm.")
    parser.add_argument("--resample-hz", type=float, default=None, help="Resample ECG to this frequency before padding.")
    parser.add_argument("--apply-filters", dest="apply_filters", action="store_true", help="Enable bandpass + notch filtering.")
    parser.add_argument("--no-apply-filters", dest="apply_filters", action="store_false", help="Disable ECG filtering.")
    parser.set_defaults(apply_filters=None)
    parser.add_argument("--bandpass-low-hz", type=float, default=None, help="Bandpass lower cutoff.")
    parser.add_argument("--bandpass-high-hz", type=float, default=None, help="Bandpass upper cutoff.")
    parser.add_argument("--notch-hz", type=float, default=None, help="Notch frequency, e.g. 60.")
    parser.add_argument("--notch-q", type=float, default=None, help="Notch filter Q factor.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--dropout", type=float, default=None, help="ResNet1d dropout.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optimizer weight decay.")
    parser.add_argument("--sched-tmax", type=int, default=None, help="CosineAnnealingLR T_max.")
    parser.add_argument("--eval-threshold", type=float, default=None, help="Risk threshold for evaluation.")
    parser.add_argument("--pos-weight-mult", type=float, default=None, help="Positive class weight multiplier.")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--early-stop-min-delta", type=float, default=None, help="Early stopping min delta.")
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default=None,
        help="Early stopping metric: auto | val_c_index | val_pr_auc | val_best_f1 | val_auc | val_loss.",
    )
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store training logs/metrics.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda:0 / cpu.")
    parser.add_argument("--use-data-parallel", action="store_true", help="Enable DataParallel across visible GPUs.")
    parser.add_argument("--device-ids", type=str, default=None, help="Comma separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--cv-folds", type=int, default=None, help="K-fold cross validation folds (1=disable).")
    parser.add_argument("--cv-seed", type=int, default=None, help="Random seed for CV splits.")
    parser.add_argument("--train-seed", type=int, default=None, help="Random seed for model init / shuffle / deterministic training.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Holdout train split ratio, used when cv_folds=1.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Holdout validation split ratio, used when cv_folds=1.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Holdout test split ratio, can be 0 when cv_folds=1.")
    parser.add_argument("--use-best-params", action="store_true", help="Override config with best params.")
    parser.add_argument("--best-params", type=Path, default=None, help="Path to best_params.json.")
    parser.add_argument("--inspect", dest="inspect", action="store_true", help="Print model then exit.")
    parser.add_argument("--no-inspect", dest="inspect", action="store_false", help="Disable inspect flag.")
    parser.set_defaults(inspect=None)
    return parser


def _coerce_value(current: Any, new_value: Any):
    if isinstance(current, Path):
        return Path(new_value)
    if isinstance(current, bool):
        return bool(new_value)
    return new_value


def apply_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            continue
        # resnet_blocks_dim 允许显式设为 None（表示使用默认大版）
        if key == "resnet_blocks_dim":
            setattr(cfg, key, value)
            continue
        if value is None:
            continue
        current = getattr(cfg, key)
        if key in {"xml_dir", "csv_dir", "manifest", "log_dir", "split_file"} and not isinstance(value, Path):
            value = Path(value)
        setattr(cfg, key, _coerce_value(current, value))
    return cfg


def apply_best_params(cfg: TrainConfig, path: Path) -> None:
    if not path or not path.exists():
        print(f"[survival] 未找到 best_params: {path}")
        return
    try:
        params = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[survival] best_params 解析失败: {exc}")
        return
    if not isinstance(params, dict):
        print("[survival] best_params 格式不正确，应为 JSON/YAML 字典")
        return
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, _coerce_value(getattr(cfg, key), value))


def load_yaml_overrides(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"找不到 YAML 配置：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML 配置需要是字典结构。")
    return data


def main() -> None:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()

    # 列出模型预设后退出
    if args.list_models:
        list_model_presets()
        return

    cfg = get_default_config()
    yaml_overrides = load_yaml_overrides(args.config)

    # 处理模型预设：优先级高于单独的 model_type/batch/dropout 等参数
    model_preset_overrides = {}
    model_name = args.model_name or yaml_overrides.get("model_name")
    if model_name:
        preset = get_model_preset(model_name)
        print(f"[model_preset] 使用预设: {model_name} - {preset.get('description', '')}")
        # 移除 description 字段，只保留配置参数
        preset.pop("description", None)
        model_preset_overrides = preset

    cli_overrides = {k: getattr(args, k) for k in (
        "xml_dir", "csv_dir", "manifest", "task_mode", "model_type", "lead_mode", "n_intervals", "max_time", "prediction_horizon",
        "target_len", "waveform_type", "resample_hz", "apply_filters", "bandpass_low_hz",
        "bandpass_high_hz", "notch_hz", "notch_q", "batch", "epochs",
        "lr", "num_workers", "dropout", "weight_decay", "sched_tmax",
        "eval_threshold", "pos_weight_mult", "early_stop_patience", "early_stop_min_delta",
        "early_stop_metric", "log_dir", "device", "use_data_parallel", "device_ids",
        "inspect", "cv_folds", "cv_seed", "train_seed", "train_ratio", "val_ratio", "test_ratio", "split_file"
    )}

    # 合并优先级：yaml < model_preset < cli（cli 优先级最高）
    combined = {**yaml_overrides, **model_preset_overrides, **{k: v for k, v in cli_overrides.items() if v is not None}}
    cfg = apply_overrides(cfg, combined)

    use_best = args.use_best_params or bool(yaml_overrides.get("use_best_params"))
    best_params_path = args.best_params or yaml_overrides.get("best_params")
    if best_params_path:
        best_params_path = Path(best_params_path)
    elif use_best:
        best_params_path = BEST_PARAMS_PATH
    if use_best and best_params_path:
        apply_best_params(cfg, best_params_path)

    print("[survival] 启动训练，配置如下：")
    for field in cfg.__dataclass_fields__.keys():
        print(f"  - {field}: {getattr(cfg, field)}")

    run_training(cfg)


if __name__ == "__main__":
    main()
