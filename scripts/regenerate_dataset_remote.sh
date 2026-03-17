#!/bin/bash
# 重新生成修复后的数据集脚本
# 使用方法: ssh admin123@192.168.1.251 "bash /home/admin123/workspace/regenerate_dataset.sh"

echo "=========================================="
echo "开始重新生成数据集（Bug修复后）"
echo "=========================================="
echo ""

# 初始化conda
source /home/admin123/data/miniconda3/etc/profile.d/conda.sh

# 激活Visionmodel环境
conda activate Visionmodel
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "Python路径: $(which python)"
echo ""

# 设置环境变量
export PYTHONPATH=/home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

# 进入工作目录
cd /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

echo "当前目录: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# 清理旧数据
echo "清理旧的YOLO数据集..."
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/
echo "清理完成"
echo ""

# 重新生成
echo "开始生成数据集..."
python scripts/dataset_generation/generate_dataset.py \
    --config scripts/dataset_generation/cfg_dsgen_combined.yaml

echo ""
echo "=========================================="
echo "数据集生成完成！"
echo "=========================================="

# 统计文件数量
YOLO_IMAGES=$(find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/images/train -name "*.png" 2>/dev/null | wc -l)
YOLO_LABELS=$(find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/labels/train -name "*.txt" 2>/dev/null | wc -l)

echo ""
echo "生成的YOLO数据集统计:"
echo "  图像数量: $YOLO_IMAGES"
echo "  标签数量: $YOLO_LABELS"
echo ""

if [ $YOLO_IMAGES -eq $YOLO_LABELS ]; then
    echo "✅ 图像和标签数量匹配"
else
    echo "⚠️  警告: 图像和标签数量不匹配！"
fi
