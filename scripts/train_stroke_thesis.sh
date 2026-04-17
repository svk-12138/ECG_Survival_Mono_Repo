#!/usr/bin/env bash
# 卒中论文训练启动脚本（Linux / macOS / WSL）。
#
# 用法：
#   1. 先复制 configs/train_stroke_thesis.env.example
#      生成为 configs/train_stroke_thesis.env
#   2. 只修改 configs/train_stroke_thesis.env
#   3. 再执行：bash scripts/train_stroke_thesis.sh
#
# 说明：
# - 如果医生在 Win11 上使用，优先改 scripts/train_stroke_thesis.ps1
#   然后运行 scripts\train_stroke_thesis.bat
# - Linux / macOS / WSL 用户再使用本脚本
#
# 设计目标：
# - 医生或学生优先改 env 配置文件，不需要改脚本本身
# - 不需要手动拼接很长的命令
# - 常用实验切换尽量只改 1-2 个参数

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

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

resolve_training_env_file() {
  local configured="${TRAIN_STROKE_ENV_FILE:-}"
  if [[ -n "$configured" ]]; then
    local resolved
    resolved="$(resolve_repo_path "$configured")"
    if [[ ! -f "$resolved" ]]; then
      echo "[error] 找不到 TRAIN_STROKE_ENV_FILE 指定的配置文件: $resolved"
      exit 1
    fi
    printf '%s' "$resolved"
    return
  fi

  local default_env="$ROOT/configs/train_stroke_thesis.env"
  if [[ -f "$default_env" ]]; then
    printf '%s' "$default_env"
    return
  fi

  local example_env="$ROOT/configs/train_stroke_thesis.env.example"
  echo "[error] 未找到本地训练配置文件: $default_env"
  echo "[hint] 请先复制 $example_env 为 $default_env，然后只修改 .env 文件，不要再改 .sh 或 .ps1 脚本。"
  exit 1
}

# ==================== 默认参数 ====================
# 下面这些值只是参数模板，用来说明每个配置项的含义。
# 实际训练时，必须由 configs/train_stroke_thesis.env 覆盖。
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

# 模型预设（优先级高于 MODEL_TYPE）：
# tcn_light       = TCN轻量版（~25k参数，适合1200样本）
# resnet_small    = ResNet1d小版（~12万参数，适合1万样本）★ 医生端推荐
# resnet_standard = ResNet1d标准版（~294万参数，适合10万+样本）
# cnn_transformer = CNN+Transformer（~69万参数，实验性）
MODEL_NAME=""

# 若不使用预设，直接指定模型架构（MODEL_NAME 留空时生效）
MODEL_TYPE="resnet"

# 导联类型：
# 8lead  = I, II, V1-V6
# 12lead = I, II, III, aVR, aVL, aVF, V1-V6
LEAD_MODE="12lead"

# 固定划分文件：首次训练自动生成，后续自动复用，确保每次数据集组成完全一致
# 留空则每次随机划分（不推荐）
SPLIT_FILE="outputs/stroke_survival_thesis/dataset_split.json"
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
TRAIN_SEED=42

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

TRAINING_ENV_FILE="$(resolve_training_env_file)"
set -a
# shellcheck disable=SC1090
source "$TRAINING_ENV_FILE"
set +a
export PYTHONHASHSEED="${TRAIN_SEED}"
echo "[env] 已加载本地配置: $TRAINING_ENV_FILE"

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

MODEL_TYPE="${MODEL_TYPE:-resnet}"
# 若设置了 MODEL_NAME 预设，校验其合法性；否则校验 MODEL_TYPE
VALID_MODEL_NAMES="tcn_light resnet_small resnet_standard cnn_transformer"
if [[ -n "${MODEL_NAME:-}" ]]; then
  valid=false
  for name in $VALID_MODEL_NAMES; do
    [[ "$MODEL_NAME" == "$name" ]] && valid=true && break
  done
  if [[ "$valid" != "true" ]]; then
    echo "[error] MODEL_NAME 只能是: $VALID_MODEL_NAMES"
    exit 1
  fi
else
  if [[ "$MODEL_TYPE" != "resnet" && "$MODEL_TYPE" != "tcn_light" && "$MODEL_TYPE" != "cnn_transformer" ]]; then
    echo "[error] MODEL_TYPE 只能是 resnet、tcn_light 或 cnn_transformer"
    exit 1
  fi
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
SPLIT_FILE_PATH="$(resolve_repo_path "${SPLIT_FILE:-}")"
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
  "--train-seed" "$TRAIN_SEED"
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

# 优先使用 MODEL_NAME 预设，若为空则使用 MODEL_TYPE
if [[ -n "${MODEL_NAME:-}" ]]; then
  CMD+=("--model-name" "$MODEL_NAME")
else
  CMD+=("--model-type" "$MODEL_TYPE")
fi

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

if [[ -n "${SPLIT_FILE_PATH:-}" ]]; then
  CMD+=("--split-file" "$SPLIT_FILE_PATH")
fi

MODEL_DISPLAY="${MODEL_NAME:-$MODEL_TYPE}"
echo "[info] 即将启动训练，关键参数如下："
echo "  task_mode=$TASK_MODE"
echo "  model=$MODEL_DISPLAY"
echo "  lead_mode=$LEAD_MODE"
echo "  manifest=$MANIFEST_PATH"
echo "  xml_dir=$XML_DIR_PATH"
echo "  csv_dir=$CSV_DIR_PATH"
echo "  log_dir=$LOG_DIR_PATH"
echo "  prediction_horizon=$PREDICTION_HORIZON"
echo "  split_ratio=train:$TRAIN_RATIO val:$VAL_RATIO test:$TEST_RATIO"
echo "  split_file=${SPLIT_FILE_PATH:-（未设置，每次随机划分）}"
echo "  cv_folds=$CV_FOLDS"
if [[ -n "$TRAINING_ENV_FILE" ]]; then
  echo "  env_file=$TRAINING_ENV_FILE"
fi

echo "[cmd] ${CMD[*]}"
exec "${CMD[@]}"
