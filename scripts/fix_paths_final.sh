#!/bin/bash
# 修复配置文件 - 使用正确的数据集路径

CONFIG_FILE="/home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation/cfg_dsgen_combined.yaml"

echo "=========================================="
echo "修复配置文件路径"
echo "=========================================="

# 1. 备份配置文件
echo "步骤1: 备份配置文件..."
cp $CONFIG_FILE ${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)
echo "✅ 备份完成"
echo ""

# 2. 修复路径
echo "步骤2: 修复数据集路径..."

# KEMEROVO路径修复
sed -i 's|folder: /data/MEDDataset/KEMEROVO/dataset|folder: /home/admin123/workspace/MEDDataset/Stenosis_detection|g' $CONFIG_FILE
echo "  ✅ KEMEROVO -> /home/admin123/workspace/MEDDataset/Stenosis_detection"

# ARCADE路径修复
sed -i 's|folder: /data/ARCADE|folder: /home/admin123/workspace/ARCADE|g' $CONFIG_FILE
echo "  ✅ ARCADE -> /home/admin123/workspace/ARCADE"

# CADICA路径修复
sed -i 's|folder: /data/CADICA|folder: /home/admin123/workspace/CADICA|g' $CONFIG_FILE
echo "  ✅ CADICA -> /home/admin123/workspace/CADICA"

# 输出路径修复
sed -i 's|output_folder: /data/combined_stenosis|output_folder: /home/admin123/workspace/combined_stenosis_new|g' $CONFIG_FILE
echo "  ✅ output_folder -> /home/admin123/workspace/combined_stenosis_new"

echo ""

# 3. 验证修改
echo "步骤3: 验证修改后的配置..."
echo "----------------------------------------"
grep -E "folder:|output_folder" $CONFIG_FILE
echo "----------------------------------------"
echo ""

# 4. 验证数据集是否存在
echo "步骤4: 验证数据集路径是否存在..."
if [ -d "/home/admin123/workspace/MEDDataset/Stenosis_detection" ]; then
    echo "  ✅ KEMEROVO数据集存在"
else
    echo "  ❌ KEMEROVO数据集不存在: /home/admin123/workspace/MEDDataset/Stenosis_detection"
fi

if [ -d "/home/admin123/workspace/ARCADE" ]; then
    echo "  ✅ ARCADE数据集存在"
else
    echo "  ❌ ARCADE数据集不存在: /home/admin123/workspace/ARCADE"
fi

if [ -d "/home/admin123/workspace/CADICA" ]; then
    echo "  ✅ CADICA数据集存在"
else
    echo "  ❌ CADICA数据集不存在: /home/admin123/workspace/CADICA"
fi

echo ""
echo "=========================================="
echo "✅ 配置文件修复完成！"
echo "=========================================="
echo ""
echo "下一步: 重新生成数据集"
echo "  bash /home/admin123/workspace/quick_rebuild.sh"
