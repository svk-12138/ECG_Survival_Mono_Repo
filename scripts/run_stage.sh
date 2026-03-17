#!/usr/bin/env bash
# 单阶段入口：参数1为阶段列表（逗号分隔），参数2为可选 config
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGES="${1:-}"
CONFIG="${2:-$ROOT/configs/pipeline.default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$STAGES" ]]; then
  echo "Usage: bash scripts/run_stage.sh <stage[,stage]> [config]"
  exit 1
fi

echo "[stage] Using config: $CONFIG"
echo "[stage] Running stages: $STAGES"
"$PYTHON_BIN" "$ROOT/scripts/run_pipeline.py" --config "$CONFIG" --stages "$STAGES"
