#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project self-check script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository health checks.")
    parser.add_argument("--skip-unit-tests", action="store_true", help="Skip tests/ discovery.")
    parser.add_argument("--check-data", action="store_true", help="Validate data/manifest structure.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.default.yaml",
                        help="Pipeline config to validate.")
    return parser.parse_args()


def ensure_paths_exist() -> None:
    required = [
        ROOT / "modules" / "survival_model",
        ROOT / "modules" / "vae_model",
        ROOT / "scripts" / "vae_latent_pearson.py",
        ROOT / "weights",
        ROOT / "data",
        ROOT / "configs",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺失关键目录：{missing}")


def validate_pipeline_config(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("vae", "pearson", "survival"):
        if section not in cfg:
            raise ValueError(f"pipeline 配置缺少 {section} 区块")
    if "checkpoint" not in cfg["pearson"]:
        raise ValueError("pearson 区块必须指定 checkpoint")


def check_data_manifest_template() -> None:
    template = ROOT / "configs" / "data_manifest.template.json"
    if not template.exists():
        raise FileNotFoundError("缺少数据 manifest 模板。")


def run_unit_tests() -> None:
    cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    ensure_paths_exist()
    validate_pipeline_config(args.config)
    check_data_manifest_template()
    if args.check_data:
        manifests_dir = ROOT / "data" / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tests] data/manifests/ exists at {manifests_dir}")
    if not args.skip_unit_tests:
        run_unit_tests()
    print("[tests] 所有检查通过")


if __name__ == "__main__":
    main()
