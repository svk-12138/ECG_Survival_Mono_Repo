#!/bin/bash
# 修复配置文件并重新生成数据集
# 执行位置: SSH登录后在服务器上执行

echo "=========================================="
echo "修复配置文件并重新生成数据集"
echo "=========================================="
echo ""

# 1. 环境准备
source /home/admin123/data/miniconda3/etc/profile.d/conda.sh
conda activate Visionmodel
cd /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

CONFIG_FILE="scripts/dataset_generation/cfg_dsgen_combined.yaml"

# 2. 备份原配置
echo "步骤1: 备份原配置文件..."
cp $CONFIG_FILE ${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)

# 3. 检查当前配置中的数据集路径
echo "步骤2: 检查当前配置..."
echo "----------------------------------------"
grep -E "CADICA|ARCADE|KEMEROVO|output_folder" $CONFIG_FILE
echo "----------------------------------------"
echo ""

# 4. 查找实际数据集位置
echo "步骤3: 查找实际数据集位置..."
echo "检查可能的数据集路径:"

# 检查常见位置
for path in \
    "/home/admin123/workspace/MEDDataset/KEMEROVO/dataset" \
    "/home/admin123/workspace/KEMEROVO/dataset" \
    "/home/admin123/workspace/home_data/data/KEMEROVO/dataset" \
    "/home/admin123/data/KEMEROVO/dataset" \
    "/data/MEDDataset/KEMEROVO/dataset"; do
    if [ -d "$path" ]; then
        echo "  ✅ 找到: $path"
        KEMEROVO_PATH="$path"
    else
        echo "  ❌ 不存在: $path"
    fi
done

for path in \
    "/home/admin123/workspace/ARCADE" \
    "/home/admin123/workspace/home_data/data/ARCADE" \
    "/home/admin123/data/ARCADE" \
    "/data/ARCADE"; do
    if [ -d "$path" ]; then
        echo "  ✅ 找到: $path"
        ARCADE_PATH="$path"
    else
        echo "  ❌ 不存在: $path"
    fi
done

for path in \
    "/home/admin123/workspace/CADICA" \
    "/home/admin123/workspace/home_data/data/CADICA" \
    "/home/admin123/data/CADICA" \
    "/data/CADICA"; do
    if [ -d "$path" ]; then
        echo "  ✅ 找到: $path"
        CADICA_PATH="$path"
    else
        echo "  ❌ 不存在: $path"
    fi
done

echo ""

# 5. 如果没找到，列出可能的位置
if [ -z "$KEMEROVO_PATH" ]; then
    echo "⚠️  未找到KEMEROVO数据集，搜索包含'KEMEROVO'的目录..."
    find /home/admin123/workspace -type d -name "*KEMEROVO*" 2>/dev/null | head -5
    echo ""
fi

# 6. 创建修复后的配置文件
echo "步骤4: 请手动确认数据集路径并创建新配置文件"
echo "当前需要修改的配置项:"
echo "  KEMEROVO: ${KEMEROVO_PATH:-未找到，需手动指定}"
echo "  ARCADE: ${ARCADE_PATH:-未找到，需手动指定}"  
echo "  CADICA: ${CADICA_PATH:-未找到，需手动指定}"
echo "  output_folder: /home/admin123/workspace/combined_stenosis_new"
echo ""

# 7. 显示当前配置文件完整内容以便修改
echo "=========================================="
echo "当前配置文件完整内容:"
echo "=========================================="
cat $CONFIG_FILE
echo ""
echo "=========================================="
echo ""

echo "请执行以下操作:"
echo "1. 找到实际的数据集路径"
echo "2. 编辑配置文件: nano $CONFIG_FILE"
echo "3. 修改数据集路径为实际路径"
echo "4. 确认 output_folder: /home/admin123/workspace/combined_stenosis_new"
echo "5. 保存后重新运行 bash quick_rebuild.sh"
