#!/usr/bin/env bash
# 统一入口（Linux/macOS）。对使用者的唯一要求是安装好 Python/依赖。
# 可选：设置 PYTHON_BIN=python3 或传入配置路径。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/pipeline.default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[pipeline] Using config: $CONFIG"
"$PYTHON_BIN" "$ROOT/scripts/run_pipeline.py" --config "$CONFIG"
