#!/bin/bash
# 修复配置文件 - 假设stenosis_detection在/home/admin123/workspace/home_data/data/

CONFIG_FILE="/home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation/cfg_dsgen_combined.yaml"

echo "备份配置文件..."
cp $CONFIG_FILE ${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)

echo "修复配置文件路径..."
# 修复KEMEROVO路径（实际是stenosis_detection）
sed -i 's|folder: /data/MEDDataset/KEMEROVO/dataset|folder: /home/admin123/workspace/home_data/data/stenosis_detection|g' $CONFIG_FILE

# 修复ARCADE路径
sed -i 's|folder: /data/ARCADE|folder: /home/admin123/workspace/home_data/data/ARCADE|g' $CONFIG_FILE

# 修复CADICA路径
sed -i 's|folder: /data/CADICA|folder: /home/admin123/workspace/home_data/data/CADICA|g' $CONFIG_FILE

# 确保output_folder正确
sed -i 's|output_folder: /data/combined_stenosis|output_folder: /home/admin123/workspace/combined_stenosis_new|g' $CONFIG_FILE

echo "修改后的配置:"
grep -E "folder:|output_folder" $CONFIG_FILE

echo ""
echo "✅ 配置文件已修复！"
echo "现在可以重新运行: bash /home/admin123/workspace/quick_rebuild.sh"
