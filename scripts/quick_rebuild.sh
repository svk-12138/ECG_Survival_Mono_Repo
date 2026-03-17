#!/bin/bash
# 快速执行版本 - 适合直接在SSH终端中粘贴
# 所有操作在/home/admin123/workspace中完成

# 1. 环境准备
set -euo pipefail

source /home/admin123/data/miniconda3/etc/profile.d/conda.sh
conda activate Visionmodel
cd /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

export PYTHONPATH="/home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main:${PYTHONPATH:-}"

# 2. 清理旧数据
echo "清理旧数据..."
rm -rf /home/admin123/workspace/combined_stenosis_new/*

# 3. 重新生成数据集
echo "重新生成数据集..."
python scripts/dataset_generation/generate_dataset.py \
    --config scripts/dataset_generation/cfg_dsgen_combined.yaml \
    2>&1 | tee /home/admin123/workspace/combined_stenosis_new/regeneration.log

# 如果生成失败则立即退出
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 数据集生成脚本执行失败，已终止"
    exit 1
fi

json_dir="/home/admin123/workspace/combined_stenosis_new/stenosis_detection/json"
yolo_dir="/home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo"

# 确保关键目录存在
if [ ! -d "$json_dir" ]; then
    echo "❌ 缺少目录: $json_dir"
    echo "请检查cfg_dsgen_combined.yaml中的output_folder设置"
    exit 1
fi

if [ ! -d "$yolo_dir" ]; then
    echo "❌ 缺少目录: $yolo_dir"
    echo "生成脚本未创建YOLO数据集，请先修复配置文件再重试"
    exit 1
fi

# 检查splits.json是否存在
if [ ! -f "$json_dir/splits.json" ]; then
    echo "❌ 未找到splits.json: $json_dir/splits.json"
    echo "请确认生成脚本已输出train/val/test分割"
    exit 1
fi

# 4. 统计splits并创建YAML配置
echo "统计train/val/test样本数量..."

train_count=$(find -L "$yolo_dir/images/train" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
val_count=$(find -L "$yolo_dir/images/val" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
test_count=$(find -L "$yolo_dir/images/test" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')

echo "  train: $train_count 张"
echo "  val:   $val_count 张"
echo "  test:  $test_count 张"

# 清理旧的txt列表，避免混淆
rm -f "$yolo_dir"/*.txt

echo "创建YAML配置..."
cat > "$yolo_dir/yolo_ica_detection.yaml" <<EOF
path: /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo
train: images/train
val: images/val
test: images/test
names:
  0: stenosis
nc: 1
EOF

# 5. 验证结果
echo ""
echo "验证结果:"
stat "$yolo_dir/yolo_ica_detection.yaml"
echo ""
printf "train images: %s\n" "$train_count"
printf "val images:   %s\n" "$val_count"
printf "test images:  %s\n" "$test_count"

echo ""
echo "✅ 完成！可以开始训练了"
