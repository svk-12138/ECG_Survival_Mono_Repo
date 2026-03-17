#!/bin/bash
# 查找stenosis_detection数据集并修复配置文件

echo "=========================================="
echo "查找stenosis_detection数据集位置"
echo "=========================================="

# 查找stenosis_detection目录
echo "搜索stenosis_detection目录..."
find /home/admin123 -type d -name "stenosis_detection" 2>/dev/null | grep -v "combined_stenosis_new"

echo ""
echo "搜索包含dataset的KEMEROVO相关目录..."
find /home/admin123 -type d -path "*/KEMEROVO/*" -name "dataset" 2>/dev/null

echo ""
echo "搜索home_data/data目录结构..."
ls -la /home/admin123/workspace/home_data/data/ 2>/dev/null || echo "路径不存在"

echo ""
echo "搜索可能的数据目录..."
find /home/admin123/workspace -maxdepth 4 -type d -name "stenosis_detection" 2>/dev/null | head -10
