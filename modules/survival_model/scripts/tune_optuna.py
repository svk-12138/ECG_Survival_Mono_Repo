#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optuna 自动化调参脚本：针对 PyTorch ECG 生存分析模型搜索超参数。"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np
import optuna
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[1]  # modules/survival_model
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_survival.train_survival_from_json import (
    TrainConfig,
    run_training,
    BEST_PARAMS_PATH,
    get_default_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna tuning for survival model.")
    parser.add_argument("--config", type=Path, default=None, help="YAML overrides for TrainConfig.")
    parser.add_argument("--xml-dir", type=Path, default=None, help="ECG XML directory.")
    parser.add_argument("--csv-dir", type=Path, default=None, help="ECG CSV directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON manifest path.")
    parser.add_argument("--task-mode", choices=["prediction", "classification"], default=None)
    parser.add_argument("--lead-mode", choices=["8lead", "12lead"], default=None)
    parser.add_argument("--n-intervals", type=int, default=None, help="Number of survival intervals.")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum time horizon.")
    parser.add_argument("--prediction-horizon", type=float, default=None, help="Risk horizon used for prediction evaluation.")
    parser.add_argument("--target-len", type=int, default=None, help="Resampled lead length.")
    parser.add_argument("--waveform-type", type=str, default=None, help="Preferred ECG waveform type.")
    parser.add_argument("--resample-hz", type=float, default=None, help="Resample ECG to this frequency before padding.")
    parser.add_argument("--apply-filters", dest="apply_filters", action="store_true", help="Enable bandpass + notch filtering.")
    parser.add_argument("--no-apply-filters", dest="apply_filters", action="store_false", help="Disable ECG filtering.")
    parser.set_defaults(apply_filters=None)
    parser.add_argument("--bandpass-low-hz", type=float, default=None)
    parser.add_argument("--bandpass-high-hz", type=float, default=None)
    parser.add_argument("--notch-hz", type=float, default=None)
    parser.add_argument("--notch-q", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Output directory for tuning logs.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda:0 / cpu.")
    parser.add_argument("--use-data-parallel", action="store_true", help="Enable DataParallel across visible GPUs.")
    parser.add_argument("--device-ids", type=str, default=None, help="Comma separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--cv-folds", type=int, default=None, help="K-fold cross validation folds.")
    parser.add_argument("--cv-seed", type=int, default=None, help="Random seed for CV splits.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler.")
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--dropout-min", type=float, default=0.2)
    parser.add_argument("--dropout-max", type=float, default=0.8)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=5e-4)
    parser.add_argument("--batch-choices", type=str, default="64,80,100")
    parser.add_argument("--eval-threshold-min", type=float, default=0.2)
    parser.add_argument("--eval-threshold-max", type=float, default=0.6)
    parser.add_argument("--epochs-min", type=int, default=40)
    parser.add_argument("--epochs-max", type=int, default=80)
    return parser


def _coerce_value(current: Any, new_value: Any):
    if isinstance(current, Path):
        return Path(new_value)
    if isinstance(current, bool):
        return bool(new_value)
    return new_value


def apply_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    for key, value in overrides.items():
        if value is None or not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        setattr(cfg, key, _coerce_value(current, value))
    return cfg


def load_yaml_overrides(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"找不到 YAML 配置：{path}")
    data = __import__("yaml").safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML 配置需要是字典结构。")
    return data


def _parse_batch_choices(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts] if parts else [64, 80, 100]


def _extract_val_score(result: dict) -> float:
    if isinstance(result.get("val"), dict):
        val = result["val"]
        cidx = val.get("c_index")
        if cidx is not None and not np.isnan(cidx):
            return float(cidx)
        auc = val.get("auc")
        if auc is not None and not np.isnan(auc):
            return float(auc)
        loss = val.get("loss")
        if loss is not None and not np.isnan(loss):
            return -float(loss)
    metrics = result.get("metrics") or {}
    cidx_mean = metrics.get("val_c_index_mean")
    if cidx_mean is not None and not np.isnan(cidx_mean):
        return float(cidx_mean)
    auc_mean = metrics.get("val_auc_mean")
    if auc_mean is not None and not np.isnan(auc_mean):
        return float(auc_mean)
    loss_mean = metrics.get("val_loss_mean")
    if loss_mean is not None and not np.isnan(loss_mean):
        return -float(loss_mean)
    return float("nan")


def main():
    args = build_parser().parse_args()
    base_cfg = get_default_config()
    yaml_overrides = load_yaml_overrides(args.config)
    cli_overrides = {k: getattr(args, k) for k in (
        "xml_dir", "csv_dir", "manifest", "task_mode", "lead_mode", "n_intervals", "max_time", "prediction_horizon",
        "target_len", "waveform_type", "resample_hz", "apply_filters", "bandpass_low_hz",
        "bandpass_high_hz", "notch_hz", "notch_q", "num_workers",
        "log_dir", "device", "use_data_parallel", "device_ids", "cv_folds", "cv_seed"
    )}
    combined = {**yaml_overrides, **{k: v for k, v in cli_overrides.items() if v is not None}}
    base_cfg = apply_overrides(base_cfg, combined)

    batch_choices = _parse_batch_choices(args.batch_choices)
    log_root = base_cfg.log_dir / "optuna_trials"
    log_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        cfg = replace(base_cfg)
        cfg.lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        cfg.dropout = trial.suggest_float("dropout", args.dropout_min, args.dropout_max)
        cfg.weight_decay = trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True)
        cfg.batch = trial.suggest_categorical("batch", batch_choices)
        cfg.eval_threshold = trial.suggest_float("eval_threshold", args.eval_threshold_min, args.eval_threshold_max)
        cfg.epochs = trial.suggest_int("epochs", args.epochs_min, args.epochs_max)
        cfg.sched_tmax = cfg.epochs
        cfg.log_dir = log_root / f"trial_{trial.number:03d}"

        result = run_training(cfg)
        score = _extract_val_score(result)
        trial.report(score, step=cfg.epochs)
        return score
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)
    print("Best value (val_auc or -loss):", study.best_value)
    print("Best params:", study.best_params)

    best_path = base_cfg.log_dir / "best_params.json"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    print(f"Saved best params to {best_path}")
    if best_path.resolve() != BEST_PARAMS_PATH.resolve():
        BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        print(f"[compat] Saved best params to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
