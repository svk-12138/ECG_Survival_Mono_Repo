#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-click pipeline orchestrator for VAE + Pearson + Survival training.

设计目标：即便是不懂行的同事，也可以直接运行该脚本并通过输出了解
目前执行到哪一步、遇到什么问题、产物保存在哪里。

说明：
- 该文件偏“全流程自动化”，适合项目维护者。
- 若当前任务是毕业论文主实验，优先使用
  `scripts/run_survival_training.py --config configs/stroke_survival_thesis.yaml`
  作为唯一训练入口，更容易收敛实验管理。
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import json
import re
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parents[1]

# 用来显示更友好的阶段标题
STAGE_TITLES = {
    "vae": "VAE 表征学习 / VAE training",
    "vae_infer": "VAE 推理导出潜变量 / VAE inference (latents)",
    "latent_traversal_mean": "潜因子聚合可视化 / Latent traversal (mean)",
    "pearson": "潜变量皮尔逊评估 / Pearson analysis",
    "survival_tune": "生存模型自动调参 / Survival tuning",
    "survival": "生存模型训练 / Survival training",
    "survival_pred": "生存模型推理 / Survival inference",
    "linear": "线性回归解释 / Linear t-value scoring",
}
STAGE_ORDER = [
    "vae",
    "pearson",
    "survival_tune",
    "survival",
    "survival_pred",
    "vae_infer",
    "linear",
    "latent_traversal_mean",
]


def _latest_vae_version() -> Path:
    """Return the most recently updated VAE version directory."""
    base = ROOT / "logs" / "MedianBeatVAE"
    if not base.exists():
        raise FileNotFoundError(f"找不到 VAE 日志目录：{base}")
    versions = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("version_")]
    if not versions:
        raise FileNotFoundError(f"在 {base} 下未找到 version_* 目录")
    return max(versions, key=lambda p: p.stat().st_mtime)


def _latest_vae_checkpoint(version_dir: Path) -> Path:
    """Return last.ckpt if exists, otherwise the newest checkpoint in the version."""
    ckpt_dir = version_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"找不到 checkpoints：{ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"在 {ckpt_dir} 下未找到 .ckpt 文件")
    return ckpts[0]


def _resolve_path(value: str | Path, kind: str | None = None) -> Path:
    """Resolve config path, supporting 'latest' for VAE artifacts."""
    if isinstance(value, Path):
        return value
    if value in ("latest", "auto"):
        latest = _latest_vae_version()
        if kind == "checkpoint":
            return _latest_vae_checkpoint(latest)
        if kind in ("latents_survival", "output_dir"):
            return latest / "latents_survival"
        return latest
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run end-to-end ECG pipeline.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/pipeline.default.yaml",
        help="YAML config defining pipeline steps."
    )
    parser.add_argument(
        "--python-bin", type=str, default=None,
        help="Override python executable (defaults to config.python_bin)."
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma/space separated stage list (default: run all).",
    )
    parser.add_argument(
        "--skip", type=str, default=None,
        help="Comma/space separated stage list to skip.",
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="List available stages and exit.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    """Read YAML config and fail early if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"找不到 pipeline 配置：{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_stage_list(value: str | None) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[,\s]+", value.strip())
    return [p.strip().lower() for p in parts if p.strip()]


def _resolve_stages(include: list[str], exclude: list[str]) -> list[str]:
    all_stages = [s.lower() for s in STAGE_ORDER]
    unknown = set(include + exclude) - set(all_stages)
    if unknown:
        raise ValueError(f"未知阶段：{', '.join(sorted(unknown))}")
    selected = include if include else list(all_stages)
    selected = [s for s in selected if s not in exclude]
    if not selected:
        raise ValueError("可执行阶段为空，请检查 --stages / --skip 参数。")
    return selected


def _print_stage_list() -> None:
    print("可用阶段 / Available stages:")
    for stage in STAGE_ORDER:
        print(f"- {stage}: {STAGE_TITLES.get(stage, stage)}")


def _stream_process(name: str, cmd: List[str], log_file: Path) -> None:
    """
    Run a subprocess, stream stdout to console and to a log file,
    方便在出问题时对照日志定位。
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        header = f"\n[{name}] CMD: {' '.join(cmd)}\n"
        print(header.strip())
        fh.write(header)
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fh.write(line)
        ret = proc.wait()
        if ret != 0:
            fh.write(f"[{name}] FAILED with exit code {ret}\n")
            raise subprocess.CalledProcessError(ret, cmd)
        fh.write(f"[{name}] SUCCESS\n")
        print(f"[{name}] 完成")


def build_log_path(base_dir: Path) -> Path:
    """Create a timestamped log path."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"pipeline_{timestamp}.log"


def _print_stage_intro(stage_key: str, stage_cfg: dict) -> None:
    """Print a short description before executing a stage."""
    title = STAGE_TITLES.get(stage_key, stage_key.upper())
    desc = stage_cfg.get("description")
    output_hint = stage_cfg.get("expected_output")
    print("\n" + "=" * 80)
    print(f"[{title}] 即将开始")
    if desc:
        print(f"说明 / Notes: {desc}")
    if output_hint:
        print(f"输出 / Output: {output_hint}")
    print("=" * 80)


def _record_result(summary: list, stage_key: str, status: str, detail: str, stage_cfg: dict | None = None) -> None:
    """Append stage status to the summary list (will be written to report)."""
    record = {
        "stage": STAGE_TITLES.get(stage_key, stage_key),
        "status": status,
        "detail": detail,
    }
    if stage_cfg:
        record["description"] = stage_cfg.get("description")
        record["expected_output"] = stage_cfg.get("expected_output")
    summary.append(record)


def run_vae(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Kick off the VAE Lightning training script."""
    if not cfg.get("enabled", False):
        print("[vae] 跳过 VAE 训练")
        _record_result(summary, "vae", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("vae", cfg)
    script = ROOT / cfg.get("script", "modules/vae_model/run.py")
    config_path = ROOT / cfg.get("config", "modules/vae_model/configs/median_vae.yaml")
    extra = cfg.get("extra_args", [])
    cmd = [python_bin, str(script), "--config", str(config_path), *extra]
    _stream_process("vae_train", cmd, log_file)
    _record_result(summary, "vae", "DONE", f"配置：{config_path}", cfg)


def run_pearson(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Run latent Pearson analysis."""
    if not cfg.get("enabled", False):
        print("[pearson] 跳过相关性分析")
        _record_result(summary, "pearson", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("pearson", cfg)
    script = ROOT / cfg.get("script", "scripts/vae_latent_pearson.py")
    vae_root = ROOT / cfg.get("vae_root", "modules/vae_model")
    checkpoint = ROOT / cfg["checkpoint"]
    config_path = ROOT / cfg.get("config", "modules/vae_model/configs/median_vae.yaml")
    output = ROOT / cfg.get("output", "outputs/analysis/vae_latent")
    split = cfg.get("split", "val")
    cmd = [
        python_bin, str(script),
        "--vae-root", str(vae_root),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint),
        "--split", split,
        "--output", str(output),
    ]
    _stream_process("vae_pearson", cmd, log_file)
    _record_result(summary, "pearson", "DONE", f"检查点：{checkpoint}", cfg)


def run_survival(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Launch survival model training with collected overrides."""
    if not cfg.get("enabled", False):
        print("[survival] 跳过生存训练")
        _record_result(summary, "survival", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("survival", cfg)
    script = ROOT / cfg.get("script", "scripts/run_survival_training.py")
    cmd = [python_bin, str(script)]
    option_map = {
        "--config": "config",
        "--xml-dir": "xml_dir",
        "--csv-dir": "csv_dir",
        "--manifest": "manifest",
        "--task-mode": "task_mode",
        "--lead-mode": "lead_mode",
        "--epochs": "epochs",
        "--batch": "batch",
        "--lr": "lr",
        "--dropout": "dropout",
        "--weight-decay": "weight_decay",
        "--eval-threshold": "eval_threshold",
        "--n-intervals": "n_intervals",
        "--max-time": "max_time",
        "--prediction-horizon": "prediction_horizon",
        "--target-len": "target_len",
        "--waveform-type": "waveform_type",
        "--resample-hz": "resample_hz",
        "--bandpass-low-hz": "bandpass_low_hz",
        "--bandpass-high-hz": "bandpass_high_hz",
        "--notch-hz": "notch_hz",
        "--notch-q": "notch_q",
        "--num-workers": "num_workers",
        "--sched-tmax": "sched_tmax",
        "--log-dir": "log_dir",
        "--device": "device",
        "--device-ids": "device_ids",
        "--cv-folds": "cv_folds",
        "--cv-seed": "cv_seed",
        "--best-params": "best_params",
    }
    for flag, key in option_map.items():
        if key in cfg and cfg[key] is not None:
            value = cfg[key]
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            if isinstance(value, str) and (key.endswith("dir") or key in ("manifest", "config", "best_params")):
                value = ROOT / value
            cmd.extend([flag, str(value)])
    if cfg.get("use_best_params"):
        cmd.append("--use-best-params")
    if cfg.get("use_data_parallel"):
        cmd.append("--use-data-parallel")
    if "apply_filters" in cfg and cfg["apply_filters"] is not None:
        cmd.append("--apply-filters" if cfg["apply_filters"] else "--no-apply-filters")
    extra = cfg.get("extra_args", [])
    cmd.extend(extra)
    _stream_process("survival_train", cmd, log_file)
    manifest = cfg.get("manifest")
    detail = f"manifest: {manifest}" if manifest else "未在配置中提供 manifest"
    _record_result(summary, "survival", "DONE", detail, cfg)


def run_survival_tune(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Launch survival model hyper-parameter tuning with Optuna."""
    if not cfg.get("enabled", False):
        print("[survival_tune] 跳过自动调参")
        _record_result(summary, "survival_tune", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("survival_tune", cfg)
    script = ROOT / cfg.get("script", "modules/survival_model/scripts/tune_optuna.py")
    cmd = [python_bin, str(script)]
    option_map = {
        "--config": "config",
        "--xml-dir": "xml_dir",
        "--csv-dir": "csv_dir",
        "--manifest": "manifest",
        "--task-mode": "task_mode",
        "--lead-mode": "lead_mode",
        "--n-intervals": "n_intervals",
        "--max-time": "max_time",
        "--prediction-horizon": "prediction_horizon",
        "--target-len": "target_len",
        "--waveform-type": "waveform_type",
        "--resample-hz": "resample_hz",
        "--bandpass-low-hz": "bandpass_low_hz",
        "--bandpass-high-hz": "bandpass_high_hz",
        "--notch-hz": "notch_hz",
        "--notch-q": "notch_q",
        "--num-workers": "num_workers",
        "--log-dir": "log_dir",
        "--device": "device",
        "--device-ids": "device_ids",
        "--cv-folds": "cv_folds",
        "--cv-seed": "cv_seed",
        "--trials": "trials",
        "--seed": "seed",
        "--lr-min": "lr_min",
        "--lr-max": "lr_max",
        "--dropout-min": "dropout_min",
        "--dropout-max": "dropout_max",
        "--weight-decay-min": "weight_decay_min",
        "--weight-decay-max": "weight_decay_max",
        "--batch-choices": "batch_choices",
        "--eval-threshold-min": "eval_threshold_min",
        "--eval-threshold-max": "eval_threshold_max",
        "--epochs-min": "epochs_min",
        "--epochs-max": "epochs_max",
    }
    for flag, key in option_map.items():
        if key in cfg and cfg[key] is not None:
            value = cfg[key]
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            if isinstance(value, str) and (key.endswith("dir") or key in ("manifest", "config")):
                value = ROOT / value
            cmd.extend([flag, str(value)])
    if cfg.get("use_data_parallel"):
        cmd.append("--use-data-parallel")
    if "apply_filters" in cfg and cfg["apply_filters"] is not None:
        cmd.append("--apply-filters" if cfg["apply_filters"] else "--no-apply-filters")
    extra = cfg.get("extra_args", [])
    cmd.extend(extra)
    _stream_process("survival_tune", cmd, log_file)
    detail = f"log_dir: {cfg.get('log_dir', '')} | trials: {cfg.get('trials', '')}"
    _record_result(summary, "survival_tune", "DONE", detail, cfg)


def run_survival_pred(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Run survival model inference to export risk scores."""
    if not cfg.get("enabled", False):
        print("[survival_pred] 跳过生存推理")
        _record_result(summary, "survival_pred", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("survival_pred", cfg)
    script = ROOT / cfg.get("script", "modules/survival_model/torch_survival/infer_survival_risk.py")
    required_keys = ["checkpoint", "manifest", "output"]
    for k in required_keys:
        if k not in cfg or cfg[k] is None:
            raise ValueError(f"[survival_pred] 缺少必要配置项：{k}")
    if not cfg.get("xml_dir") and not cfg.get("csv_dir"):
        raise ValueError("[survival_pred] xml_dir 和 csv_dir 至少需要提供一个")
    cmd = [
        python_bin,
        str(script),
        "--checkpoint",
        str(ROOT / cfg["checkpoint"] if isinstance(cfg["checkpoint"], str) else cfg["checkpoint"]),
        "--manifest",
        str(ROOT / cfg["manifest"] if isinstance(cfg["manifest"], str) else cfg["manifest"]),
        "--output",
        str(ROOT / cfg["output"] if isinstance(cfg["output"], str) else cfg["output"]),
        "--task-mode",
        str(cfg.get("task_mode", "auto")),
        "--lead-mode",
        str(cfg.get("lead_mode", "8lead")),
        "--n-intervals",
        str(cfg.get("n_intervals", 40)),
        "--max-time",
        str(cfg.get("max_time", 3650.0)),
        "--target-len",
        str(cfg.get("target_len", 4096)),
        "--waveform-type",
        str(cfg.get("waveform_type", "Rhythm")),
        "--resample-hz",
        str(cfg.get("resample_hz", 400.0)),
        "--bandpass-low-hz",
        str(cfg.get("bandpass_low_hz", 0.5)),
        "--bandpass-high-hz",
        str(cfg.get("bandpass_high_hz", 100.0)),
        "--notch-q",
        str(cfg.get("notch_q", 30.0)),
        "--batch",
        str(cfg.get("batch", 16)),
    ]
    if cfg.get("xml_dir"):
        cmd.extend(["--xml-dir", str(ROOT / cfg["xml_dir"] if isinstance(cfg["xml_dir"], str) else cfg["xml_dir"])])
    if cfg.get("prediction_horizon") is not None:
        cmd.extend(["--prediction-horizon", str(cfg["prediction_horizon"])])
    if cfg.get("notch_hz") is not None:
        cmd.extend(["--notch-hz", str(cfg["notch_hz"])])
    if cfg.get("csv_dir"):
        cmd.extend(["--csv-dir", str(ROOT / cfg["csv_dir"] if isinstance(cfg["csv_dir"], str) else cfg["csv_dir"])])
    if cfg.get("device"):
        cmd.extend(["--device", str(cfg["device"])])
    if "apply_filters" in cfg and cfg["apply_filters"] is not None:
        cmd.append("--apply-filters" if cfg["apply_filters"] else "--no-apply-filters")
    _stream_process("survival_infer", cmd, log_file)
    _record_result(summary, "survival_pred", "DONE", f"risk_scores: {cfg['output']}", cfg)


def run_vae_infer(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Use trained VAE checkpoint to export latents on a target dataset."""
    if not cfg.get("enabled", False):
        print("[vae_infer] 跳过 VAE 推理导出潜变量")
        _record_result(summary, "vae_infer", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("vae_infer", cfg)
    script = ROOT / cfg.get("script", "modules/vae_model/tools/extract_latent_features.py")
    required = ["checkpoint", "config", "data_path", "output"]
    for k in required:
        if k not in cfg or cfg[k] is None:
            raise ValueError(f"[vae_infer] 缺少必要配置项：{k}")
    cmd = [
        python_bin,
        str(script),
        "--config",
        str(_resolve_path(cfg["config"])),
        "--checkpoint",
        str(_resolve_path(cfg["checkpoint"], kind="checkpoint")),
        "--data-path",
        str(cfg["data_path"]),
        "--output",
        str(_resolve_path(cfg["output"], kind="output_dir")),
    ]
    if cfg.get("file_pattern"):
        cmd.extend(["--pattern", str(cfg["file_pattern"])])
    extra = cfg.get("extra_args", [])
    cmd.extend(extra)
    _stream_process("vae_infer", cmd, log_file)
    _record_result(summary, "vae_infer", "DONE", f"latents: {cfg['output']}", cfg)


def run_linear(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Train linear regression on VAE latents using risk score as label."""
    if not cfg.get("enabled", False):
        print("[linear] 跳过线性回归")
        _record_result(summary, "linear", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("linear", cfg)
    script = ROOT / cfg.get("script", "modules/vae_model/scripts/train_linear_scores.py")
    required_keys = ["latents_dir", "labels_csv"]
    for k in required_keys:
        if k not in cfg or cfg[k] is None:
            raise ValueError(f"[linear] 缺少必要配置项：{k}")
    cmd = [
        python_bin,
        str(script),
        "--latents-dir",
        str(_resolve_path(cfg["latents_dir"], kind="latents_survival")),
        "--labels-csv",
        str(ROOT / cfg["labels_csv"] if isinstance(cfg["labels_csv"], str) else cfg["labels_csv"]),
        "--id-column",
        str(cfg.get("id_column", "sample_id")),
        "--label-column",
        str(cfg.get("label_column", "risk_score")),
        "--target-type",
        str(cfg.get("target_type", "regression")),
        "--top-k",
        str(cfg.get("top_k", 5)),
    ]
    if cfg.get("id_suffix"):
        cmd.extend(["--id-suffix", str(cfg["id_suffix"])])
    if cfg.get("output"):
        cmd.extend(["--output", str(_resolve_path(cfg["output"], kind="output_dir"))])
    _stream_process("linear_scores", cmd, log_file)
    _record_result(summary, "linear", "DONE", f"labels: {cfg['labels_csv']}", cfg)


def run_latent_traversal_mean(cfg: dict, python_bin: str, log_file: Path, summary: list) -> None:
    """Aggregate latent traversal visualization over a latent dataset."""
    if not cfg.get("enabled", False):
        print("[latent_traversal_mean] 跳过聚合可视化")
        _record_result(summary, "latent_traversal_mean", "SKIPPED", "配置中将该阶段设为 disabled", cfg)
        return
    _print_stage_intro("latent_traversal_mean", cfg)
    script = ROOT / cfg.get("script", "modules/vae_model/tools/latent_traversal_mean.py")
    required = ["config", "checkpoint", "latents_dir", "output_dir", "dims"]
    for k in required:
        if k not in cfg or cfg[k] is None:
            raise ValueError(f"[latent_traversal_mean] 缺少必要配置项：{k}")
    cmd = [
        python_bin,
        str(script),
        "--config",
        str(_resolve_path(cfg["config"])),
        "--checkpoint",
        str(_resolve_path(cfg["checkpoint"], kind="checkpoint")),
        "--latents-dir",
        str(_resolve_path(cfg["latents_dir"], kind="latents_survival")),
        "--output-dir",
        str(_resolve_path(cfg["output_dir"], kind="output_dir")),
        "--dims",
        *[str(v) for v in cfg["dims"]],
    ]
    steps = cfg.get("steps")
    if steps:
        cmd.extend(["--steps", str(steps[0]), str(steps[1]), str(steps[2])])
    if cfg.get("xmax") is not None:
        cmd.extend(["--xmax", str(cfg["xmax"])])
    if cfg.get("downsample") is not None:
        cmd.extend(["--downsample", str(cfg["downsample"])])
    if cfg.get("batch_size") is not None:
        cmd.extend(["--batch-size", str(cfg["batch_size"])])
    _stream_process("latent_traversal_mean", cmd, log_file)
    _record_result(summary, "latent_traversal_mean", "DONE", f"output: {cfg['output_dir']}", cfg)


def print_summary(summary: list, log_file: Path) -> None:
    """Print a compact summary table so novice users know what happened."""
    print("\n执行摘要 / Run Summary")
    print("-" * 80)
    for item in summary:
        print(f"{item['stage']:<32} {item['status']:<10} {item['detail']}")
    print("-" * 80)
    print(f"完整日志 / Full log: {log_file}")


def write_report(summary: list, log_file: Path, report_path: Path) -> None:
    """Write summary to a JSON report for later reference."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log_file": str(log_file),
        "stages": summary,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] 汇总报告已写入 {report_path}")


def main() -> None:
    args = parse_args()
    if args.list_stages:
        _print_stage_list()
        return
    cfg = load_yaml_config(args.config)

    python_bin = args.python_bin or cfg.get("python_bin", "python")
    log_dir = ROOT / cfg.get("log_dir", "outputs/logs")
    log_file = build_log_path(log_dir)
    report_path = ROOT / cfg.get("report", "outputs/pipeline_report.json")
    summary: list = []

    include = _parse_stage_list(args.stages)
    exclude = _parse_stage_list(args.skip)
    selected = _resolve_stages(include, exclude)
    selected_set = set(selected)

    stage_runners = {
        "vae": (run_vae, "vae"),
        "pearson": (run_pearson, "pearson"),
        "survival_tune": (run_survival_tune, "survival_tune"),
        "survival": (run_survival, "survival"),
        "survival_pred": (run_survival_pred, "survival_pred"),
        "vae_infer": (run_vae_infer, "vae_infer"),
        "linear": (run_linear, "linear"),
        "latent_traversal_mean": (run_latent_traversal_mean, "latent_traversal_mean"),
    }

    # Execute each stage in order, collecting success/skip information.
    for stage in STAGE_ORDER:
        runner, cfg_key = stage_runners[stage]
        if stage not in selected_set:
            _record_result(summary, stage, "SKIPPED", "未选择该阶段", cfg.get(cfg_key, {}))
            continue
        runner(cfg.get(cfg_key, {}), python_bin, log_file, summary)

    print_summary(summary, log_file)
    write_report(summary, log_file, report_path)
    print(f"[pipeline] 全部步骤完成，日志保存在 {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
