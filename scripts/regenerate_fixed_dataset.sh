#!/bin/bash
# 重新生成修复后的数据集脚本
# 创建日期: 2025-01-XX
# 用途: 修复preprocessing.py的坐标bug后重新生成完整数据集

echo "=========================================="
echo "开始重新生成KEMEROVO数据集（Bug修复后）"
echo "=========================================="
echo ""

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Visionmodel

# 进入工作目录
cd /home/admin123/workspace/FRcnn/Coronary_Angiography_Detection-main

echo "当前工作目录: $(pwd)"
echo "Python版本: $(python --version)"
echo ""

# 删除旧的输出
echo "清理旧的数据集输出..."
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/images/
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/labels/
rm -f /home/admin123/workspace/combined_stenosis_new/stenosis_detection/combined_standardized.json
rm -rf /home/admin123/workspace/combined_stenosis_new/*
echo "清理完成"
echo ""

# 重新生成数据集
echo "开始运行数据集生成脚本..."
echo "配置文件:/home/admin123/use/home_data/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation/cfg_dsgen_combined.yaml"
echo ""

python /home/admin123/use/home_data/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation/generate_dataset.py \
    --config /home/admin123/use/home_data/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation/cfg_dsgen_combined.yaml \
    2>&1 | tee dataset_regeneration_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "数据集生成完成！"
echo "=========================================="
echo ""

# 统计生成的文件数量
YOLO_IMAGES=$(find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/images/train -name "*.png" 2>/dev/null | wc -l)
YOLO_LABELS=$(find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/labels/train -name "*.txt" 2>/dev/null | wc -l)

echo "生成的YOLO数据集统计:"
echo "  图像数量: $YOLO_IMAGES"
echo "  标签数量: $YOLO_LABELS"
echo ""

if [ $YOLO_IMAGES -eq $YOLO_LABELS ]; then
    echo "✅ 图像和标签数量匹配"
else
    echo "⚠️  警告: 图像和标签数量不匹配！"
fi

echo ""
echo "下一步: 运行验证脚本检查坐标准确性"
echo "  python d:/WORKING/remote_utils/compare_before_after_preprocessing.py"
