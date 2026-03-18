#!/usr/bin/env bash
# 卒中论文训练启动脚本（Linux / macOS / WSL）。
#
# 用法：
#   1. 先修改本文件顶部参数
#   2. 再执行：bash scripts/train_stroke_thesis.sh
#
# 说明：
# - 如果医生在 Win11 上使用，优先改 scripts/train_stroke_thesis.ps1
#   然后运行 scripts\train_stroke_thesis.bat
# - Linux / macOS / WSL 用户再使用本脚本
#
# 设计目标：
# - 医生或学生只需要改这一份入口脚本
# - 不需要手动拼接很长的命令
# - 常用实验切换尽量只改 1-2 个参数

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

resolve_repo_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf '%s' ""
  elif [[ "$value" = /* ]]; then
    printf '%s' "$value"
  else
    printf '%s' "$ROOT/$value"
  fi
}

# ==================== 必改参数 ====================
# 标签文件：必须包含 patient_id / time / event
MANIFEST="/your/path/stroke_manifest.json"

# ECG 数据二选一：
# 1. 用 XML 就填 XML_DIR，把 CSV_DIR 留空
XML_DIR="/your/path/xml_dir"
CSV_DIR=""

# 2. 用 CSV 就填 CSV_DIR，把 XML_DIR 留空
# XML_DIR=""
# CSV_DIR="/your/path/csv_dir"

# 任务类型：
# prediction    = 生存预测，建议作为论文主实验
# classification = 二分类，建议作为对照实验
TASK_MODE="prediction"

# 导联类型：
# 8lead  = I, II, V1-V6
# 12lead = I, II, III, aVR, aVL, aVF, V1-V6
LEAD_MODE="12lead"
# ==================================================

# ==================== 常用参数 ====================
# 时间设置，单位要和 manifest 里的 time 一致
N_INTERVALS=20
MAX_TIME=1825.0

# 风险时间点：
# - 写 null 表示整个随访窗口风险
# - 写 365.0 / 1095.0 / 1825.0 表示 1年/3年/5年风险
PREDICTION_HORIZON="null"

# ECG 预处理：
# - APPLY_FILTERS=true 时，会做带通滤波 + 工频陷波
# - 这部分更贴近论文中的常见 ECG 预处理思路
WAVEFORM_TYPE="Rhythm"
RESAMPLE_HZ=400.0
APPLY_FILTERS=true
BANDPASS_LOW_HZ=0.5
BANDPASS_HIGH_HZ=100.0
NOTCH_HZ=60.0
NOTCH_Q=30.0
TARGET_LEN=4096

# 训练参数
BATCH=32
EPOCHS=80
LR=0.0005
DROPOUT=0.5
WEIGHT_DECAY=0.0001
NUM_WORKERS=0

# 留出法划分：
# - 仅在 CV_FOLDS=1 时生效
# - 默认固定为 0.8 / 0.2 / 0.0，即训练/验证，无测试集
TRAIN_RATIO=0.8
VAL_RATIO=0.2
TEST_RATIO=0.0

# 是否启用交叉验证：
# - 1 表示使用上面的 train/val/test 比例
# - >1 表示启用 K 折交叉验证，此时比例参数会被忽略
CV_FOLDS=1
CV_SEED=42

# 早停与评估
EVAL_THRESHOLD=0.5
EARLY_STOP_METRIC="auto"
EARLY_STOP_PATIENCE=15
EARLY_STOP_MIN_DELTA=0.0001
POS_WEIGHT_MULT=2.0

# 输出目录：
# - 可以写相对路径，脚本会自动保存到仓库目录下
LOG_DIR="outputs/stroke_survival_thesis"

# 设备设置
DEVICE=""
USE_DATA_PARALLEL=false
DEVICE_IDS=""

# 如已有 best_params.json，可打开这两项
USE_BEST_PARAMS=false
BEST_PARAMS="outputs/stroke_survival_thesis/best_params.json"
# ==================================================

if [[ -z "$MANIFEST" ]]; then
  echo "[error] MANIFEST 不能为空"
  exit 1
fi

if [[ -n "$XML_DIR" && -n "$CSV_DIR" ]]; then
  echo "[error] XML_DIR 和 CSV_DIR 只能填一个"
  exit 1
fi

if [[ -z "$XML_DIR" && -z "$CSV_DIR" ]]; then
  echo "[error] XML_DIR 和 CSV_DIR 至少要填一个"
  exit 1
fi

if [[ "$TASK_MODE" != "prediction" && "$TASK_MODE" != "classification" ]]; then
  echo "[error] TASK_MODE 只能是 prediction 或 classification"
  exit 1
fi

if [[ "$LEAD_MODE" != "8lead" && "$LEAD_MODE" != "12lead" ]]; then
  echo "[error] LEAD_MODE 只能是 8lead 或 12lead"
  exit 1
fi

MANIFEST_PATH="$(resolve_repo_path "$MANIFEST")"
XML_DIR_PATH="$(resolve_repo_path "$XML_DIR")"
CSV_DIR_PATH="$(resolve_repo_path "$CSV_DIR")"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[error] 找不到 MANIFEST: $MANIFEST_PATH"
  exit 1
fi

if [[ -n "$XML_DIR_PATH" && ! -d "$XML_DIR_PATH" ]]; then
  echo "[error] 找不到 XML_DIR: $XML_DIR_PATH"
  exit 1
fi

if [[ -n "$CSV_DIR_PATH" && ! -d "$CSV_DIR_PATH" ]]; then
  echo "[error] 找不到 CSV_DIR: $CSV_DIR_PATH"
  exit 1
fi

LOG_DIR_PATH="$(resolve_repo_path "$LOG_DIR")"
BEST_PARAMS_PATH="$(resolve_repo_path "$BEST_PARAMS")"
mkdir -p "$LOG_DIR_PATH"

CMD=(
  "$PYTHON_BIN"
  "$ROOT/scripts/run_survival_training.py"
  "--manifest" "$MANIFEST_PATH"
  "--task-mode" "$TASK_MODE"
  "--lead-mode" "$LEAD_MODE"
  "--n-intervals" "$N_INTERVALS"
  "--max-time" "$MAX_TIME"
  "--target-len" "$TARGET_LEN"
  "--waveform-type" "$WAVEFORM_TYPE"
  "--resample-hz" "$RESAMPLE_HZ"
  "--bandpass-low-hz" "$BANDPASS_LOW_HZ"
  "--bandpass-high-hz" "$BANDPASS_HIGH_HZ"
  "--notch-hz" "$NOTCH_HZ"
  "--notch-q" "$NOTCH_Q"
  "--batch" "$BATCH"
  "--epochs" "$EPOCHS"
  "--lr" "$LR"
  "--dropout" "$DROPOUT"
  "--weight-decay" "$WEIGHT_DECAY"
  "--num-workers" "$NUM_WORKERS"
  "--cv-folds" "$CV_FOLDS"
  "--cv-seed" "$CV_SEED"
  "--train-ratio" "$TRAIN_RATIO"
  "--val-ratio" "$VAL_RATIO"
  "--test-ratio" "$TEST_RATIO"
  "--eval-threshold" "$EVAL_THRESHOLD"
  "--early-stop-metric" "$EARLY_STOP_METRIC"
  "--early-stop-patience" "$EARLY_STOP_PATIENCE"
  "--early-stop-min-delta" "$EARLY_STOP_MIN_DELTA"
  "--pos-weight-mult" "$POS_WEIGHT_MULT"
  "--log-dir" "$LOG_DIR_PATH"
)

if [[ -n "$XML_DIR_PATH" ]]; then
  CMD+=("--xml-dir" "$XML_DIR_PATH")
fi

if [[ -n "$CSV_DIR_PATH" ]]; then
  CMD+=("--csv-dir" "$CSV_DIR_PATH")
fi

if [[ "$PREDICTION_HORIZON" != "null" ]]; then
  CMD+=("--prediction-horizon" "$PREDICTION_HORIZON")
fi

if [[ "$APPLY_FILTERS" == "true" ]]; then
  CMD+=("--apply-filters")
else
  CMD+=("--no-apply-filters")
fi

if [[ -n "$DEVICE" ]]; then
  CMD+=("--device" "$DEVICE")
fi

if [[ "$USE_DATA_PARALLEL" == "true" ]]; then
  CMD+=("--use-data-parallel")
fi

if [[ -n "$DEVICE_IDS" ]]; then
  CMD+=("--device-ids" "$DEVICE_IDS")
fi

if [[ "$USE_BEST_PARAMS" == "true" ]]; then
  CMD+=("--use-best-params" "--best-params" "$BEST_PARAMS_PATH")
fi

echo "[info] 即将启动训练，关键参数如下："
echo "  task_mode=$TASK_MODE"
echo "  lead_mode=$LEAD_MODE"
echo "  manifest=$MANIFEST_PATH"
echo "  xml_dir=$XML_DIR_PATH"
echo "  csv_dir=$CSV_DIR_PATH"
echo "  log_dir=$LOG_DIR_PATH"
echo "  prediction_horizon=$PREDICTION_HORIZON"
echo "  split_ratio=train:$TRAIN_RATIO val:$VAL_RATIO test:$TEST_RATIO"
echo "  cv_folds=$CV_FOLDS"

echo "[cmd] ${CMD[*]}"
exec "${CMD[@]}"
