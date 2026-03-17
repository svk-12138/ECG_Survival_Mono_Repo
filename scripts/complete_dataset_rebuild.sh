#!/bin/bash
# 完整的数据集重建脚本 - 包含train/val/test分割
# 创建日期: 2025-11-07
# 用途: 从头重新生成数据集，确保正确的train/val/test分割

echo "=========================================="
echo "完整数据集重建流程"
echo "=========================================="
echo ""

# ============================================
# 步骤1: 环境准备
# ============================================
echo "步骤1: 环境准备"
echo "----------------------------------------"

# 激活conda环境
echo "激活Conda环境..."
source /home/admin123/data/miniconda3/etc/profile.d/conda.sh
conda activate Visionmodel

# 设置工作目录
WORK_DIR="/home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main"
DATA_OUTPUT="/home/admin123/workspace/combined_stenosis_new"

cd "$WORK_DIR" || { echo "❌ 无法进入工作目录: $WORK_DIR"; exit 1; }

echo "✅ 当前工作目录: $(pwd)"
echo "✅ Python版本: $(python --version)"
echo "✅ 数据输出路径: $DATA_OUTPUT"
echo ""

# ============================================
# 步骤2: 完全清理旧数据
# ============================================
echo "步骤2: 完全清理旧数据"
echo "----------------------------------------"

if [ -d "$DATA_OUTPUT" ]; then
    echo "删除旧数据目录: $DATA_OUTPUT"
    rm -rf "$DATA_OUTPUT"/*
    echo "✅ 清理完成"
else
    echo "创建数据输出目录: $DATA_OUTPUT"
    mkdir -p "$DATA_OUTPUT"
fi
echo ""

# ============================================
# 步骤3: 验证配置文件
# ============================================
echo "步骤3: 验证配置文件"
echo "----------------------------------------"

CONFIG_FILE="$WORK_DIR/scripts/dataset_generation/cfg_dsgen_combined.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "配置文件路径: $CONFIG_FILE"
echo "检查output_folder配置..."
grep "output_folder" "$CONFIG_FILE"
echo "✅ 配置文件验证通过"
echo ""

# ============================================
# 步骤4: 重新生成数据集
# ============================================
echo "步骤4: 重新生成数据集"
echo "----------------------------------------"

LOG_FILE="$DATA_OUTPUT/dataset_regeneration_$(date +%Y%m%d_%H%M%S).log"
echo "开始数据集生成..."
echo "日志文件: $LOG_FILE"
echo ""

python "$WORK_DIR/scripts/dataset_generation/generate_dataset.py" \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

GENERATION_EXIT_CODE=${PIPESTATUS[0]}

if [ $GENERATION_EXIT_CODE -ne 0 ]; then
    echo "❌ 数据集生成失败，退出码: $GENERATION_EXIT_CODE"
    exit 1
fi

echo "✅ 数据集生成完成"
echo ""

# ============================================
# 步骤5: 验证生成的文件
# ============================================
echo "步骤5: 验证生成的文件"
echo "----------------------------------------"

STENOSIS_DIR="$DATA_OUTPUT/stenosis_detection"
JSON_DIR="$STENOSIS_DIR/json"
YOLO_DIR="$STENOSIS_DIR/datasets/yolo"

# 检查JSON文件
echo "检查JSON文件..."
for json_file in combined_standardized.json planned_standardized.json processed.json splits.json; do
    if [ -f "$JSON_DIR/$json_file" ]; then
        size=$(du -h "$JSON_DIR/$json_file" | cut -f1)
        echo "  ✅ $json_file ($size)"
    else
        echo "  ❌ 缺失: $json_file"
    fi
done

# 检查YOLO数据集
echo ""
echo "检查YOLO数据集..."
TOTAL_IMAGES=$(find "$YOLO_DIR/images" -name "*.png" 2>/dev/null | wc -l)
TOTAL_LABELS=$(find "$YOLO_DIR/labels" -name "*.txt" 2>/dev/null | wc -l)

echo "  总图像数: $TOTAL_IMAGES"
echo "  总标签数: $TOTAL_LABELS"

if [ $TOTAL_IMAGES -eq 0 ]; then
    echo "❌ 未找到生成的图像文件"
    exit 1
fi

# 检查splits.json
if [ ! -f "$JSON_DIR/splits.json" ]; then
    echo "❌ splits.json不存在"
    exit 1
fi

echo "✅ 文件验证通过"
echo ""

# ============================================
# 步骤6: 从splits.json生成train/val/test文件列表
# ============================================
echo "步骤6: 生成train/val/test文件列表"
echo "----------------------------------------"

# 创建Python脚本生成文件列表
python << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

# 路径配置
json_dir = Path("/home/admin123/workspace/combined_stenosis_new/stenosis_detection/json")
yolo_dir = Path("/home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo")
images_dir = yolo_dir / "images"
labels_dir = yolo_dir / "labels"

# 读取splits.json
with open(json_dir / "splits.json", "r") as f:
    splits = json.load(f)

# 获取所有可用的图像文件
all_images = set()
for img_path in images_dir.glob("*.png"):
    all_images.add(img_path.stem)

print(f"找到 {len(all_images)} 张图像")

# 为每个split生成文件列表
for split_name in ["train", "val", "test"]:
    output_file = yolo_dir / f"{split_name}.txt"
    
    # 收集该split的所有患者ID的图像
    split_images = []
    for dataset_name, patient_ids in splits.get(split_name, {}).items():
        for patient_id in patient_ids:
            # 查找匹配该患者的所有图像
            for img_name in all_images:
                if img_name.startswith(patient_id):
                    # 使用相对路径
                    img_rel_path = f"images/{img_name}.png"
                    split_images.append(img_rel_path)
    
    # 去重并排序
    split_images = sorted(set(split_images))
    
    # 写入文件
    with open(output_file, "w") as f:
        for img_path in split_images:
            f.write(f"{img_path}\n")
    
    print(f"✅ {split_name}.txt: {len(split_images)} 张图像")

print("\n文件列表生成完成！")
PYTHON_SCRIPT

PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "❌ 文件列表生成失败"
    exit 1
fi

# 验证文件列表
echo ""
echo "验证文件列表..."
for split in train val test; do
    if [ -f "$YOLO_DIR/${split}.txt" ]; then
        count=$(wc -l < "$YOLO_DIR/${split}.txt")
        echo "  $split.txt: $count 行"
    else
        echo "  ❌ ${split}.txt 不存在"
    fi
done
echo ""

# ============================================
# 步骤7: 创建正确的YAML配置
# ============================================
echo "步骤7: 创建YAML配置"
echo "----------------------------------------"

YAML_FILE="$YOLO_DIR/yolo_ica_detection.yaml"

cat > "$YAML_FILE" << EOF
# YOLOv8 ICA Stenosis Detection Dataset Configuration
# Generated: $(date)
# Dataset path (absolute)
path: $YOLO_DIR

# Train/val/test splits (relative to 'path')
train: train.txt
val: val.txt
test: test.txt

# Classes
names:
  0: stenosis

# Number of classes
nc: 1
EOF

echo "✅ YAML配置已创建: $YAML_FILE"
echo ""
cat "$YAML_FILE"
echo ""

# ============================================
# 步骤8: 最终验证和统计
# ============================================
echo "=========================================="
echo "最终验证和统计"
echo "=========================================="

echo "数据集位置: $YOLO_DIR"
echo ""

echo "文件列表统计:"
for split in train val test; do
    if [ -f "$YOLO_DIR/${split}.txt" ]; then
        count=$(wc -l < "$YOLO_DIR/${split}.txt")
        echo "  $split: $count 张图像"
    fi
done

echo ""
echo "YAML配置: $YAML_FILE"
echo ""

# 检查train.txt中的第一张图像是否存在
FIRST_IMAGE=$(head -1 "$YOLO_DIR/train.txt" | sed 's/^images\///')
if [ -f "$YOLO_DIR/images/${FIRST_IMAGE}" ]; then
    echo "✅ 样本图像检查通过"
else
    echo "⚠️  警告: 第一张训练图像不存在: $FIRST_IMAGE"
fi

echo ""
echo "=========================================="
echo "✅ 数据集重建完成！"
echo "=========================================="
echo ""
echo "下一步: 启动训练"
echo "  cd $YOLO_DIR"
echo "  python << 'TRAIN_SCRIPT'"
echo "from ultralytics import YOLO"
echo "model = YOLO('yolov8x.pt')"
echo "results = model.train("
echo "    data='yolo_ica_detection.yaml',"
echo "    epochs=100,"
echo "    imgsz=512,"
echo "    batch=130,"
echo "    device=[0,1],"
echo "    name='train_fixed_split'"
echo ")"
echo "TRAIN_SCRIPT"
echo ""
