# 坐标Bug修复 - 快速执行清单

## ✅ 已完成

- [x] **Bug定位**: preprocessing.py 第350行
- [x] **代码修复**: 使用yolo_norm_width/height替代orig_width/height
- [x] **上传到服务器**: /home/.../preprocessing.py

---

## 🚀 立即执行 (按顺序)

### 1️⃣ 重新生成数据集 (SSH到服务器)

```bash
ssh admin123@192.168.1.251

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Visionmodel

# 进入项目目录
cd /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

# 清理旧数据(可选:先备份combined_standardized.json)
cp /home/admin123/workspace/combined_stenosis_new/stenosis_detection/combined_standardized.json \
   /home/admin123/workspace/combined_stenosis_new/stenosis_detection/combined_standardized.json.backup_$(date +%Y%m%d)

# 删除旧的YOLO数据集
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/

# 重新生成(约30-60分钟)
python ./scripts/dataset_generation/generate_dataset.py \
    --config ./scripts/dataset_generation/cfg_dsgen_combined.yaml

# 等待完成...
```

**检查点**:
```bash
# 验证文件数量
find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/images/train -name "*.png" | wc -l
find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/labels/train -name "*.txt" | wc -l

# 应该看到相同数量(约12000+)
```

---

### 2️⃣ 验证坐标准确性 (本地Windows)

```bash
# 在本地运行对比脚本
python d:/WORKING/remote_utils/compare_before_after_preprocessing.py

# 下载对比结果
scp -r admin123@192.168.1.251:/home/admin123/workspace/preprocessing_comparison d:/WORKING/preprocessing_comparison_results_FIXED/

# 对比修复前后
# 修复前: d:/WORKING/preprocessing_comparison_results/
# 修复后: d:/WORKING/preprocessing_comparison_results_FIXED/
```

**预期结果**:
- ✅ 平均误差: 从142.6px降至<2px
- ✅ 对比图: 蓝色框和绿色框完全重叠
- ✅ 表格: 坐标差异≈0

---

### 3️⃣ 重新训练模型 (SSH到服务器)

```bash
ssh admin123@192.168.1.251

cd /data/combined_stenosis/stenosis_detection

# 备份旧的训练结果(可选)
mv runs/detect/train runs/detect/train_before_fix_$(date +%Y%m%d)

# 启动训练(100 epochs,约4-6小时双GPU)
./tune_with_existing_dataset.sh train

# 监控训练(新开一个终端)
watch -n 5 "tail -20 runs/detect/train/train.log"
```

**检查点**:
```bash
# 查看训练进度
tensorboard --logdir runs/detect/train --port 6006

# 在本地浏览器访问
# http://192.168.1.251:6006
```

---

### 4️⃣ 评估模型性能

**训练完成后**:
```bash
# 查看最终指标
cat runs/detect/train/results.csv | tail -5

# 测试集推理
python val.py --weights runs/detect/train/weights/best.pt --data yolo_ica_detection.yaml

# 可视化检测结果
python predict.py --weights runs/detect/train/weights/best.pt --source test_images/
```

**预期指标**:
- mAP50: >0.30 (修复前0.08)
- mAP50-95: >0.15 (修复前0.03)
- Precision: >0.50
- Recall: >0.45

---

## 📊 关键验证点

| 阶段 | 检查项 | 预期值 | 实际值 |
|-----|-------|--------|--------|
| 数据集生成 | 图像数量 | ~12000 | ___ |
| 数据集生成 | 标签数量 | ~12000 | ___ |
| 坐标验证 | 平均误差 | <2px | ___ |
| 坐标验证 | 最大误差 | <5px | ___ |
| 模型训练 | mAP50 (epoch 50) | >0.20 | ___ |
| 模型训练 | mAP50 (epoch 100) | >0.30 | ___ |
| 模型训练 | Precision | >0.50 | ___ |
| 模型训练 | Recall | >0.45 | ___ |

---

## ⚠️ 故障排查

### 数据集生成失败
```bash
# 检查preprocessing.py是否正确更新
ssh admin123@192.168.1.251
grep -n "yolo_norm_width" /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main/ICA_Detection/preprocessing/preprocessing.py

# 应该看到第333-343行附近有yolo_norm_width的定义
```

### 坐标验证仍有误差
```bash
# 确认使用的是新生成的数据集
stat /home/admin123/workspace/combined_stenosis_new/stenosis_detection/combined_standardized.json

# Modified时间应该是今天
```

### 训练效果无改善
```bash
# 确认YAML配置使用新数据集路径
cat /data/combined_stenosis/stenosis_detection/datasets/yolo/yolo_ica_detection.yaml

# 检查train和val路径
```

---

## 📞 紧急联系

如果遇到无法解决的问题:
1. 查看完整文档: `d:/WORKING/docs/coordinate_bug_fix_guide.md`
2. 检查之前的对比报告: `d:/WORKING/docs/preprocessing_comparison_report.md`
3. 重新运行诊断: `python d:/WORKING/remote_utils/compare_before_after_preprocessing.py`

---

**更新**: 2025-01-XX  
**状态**: 代码已修复并上传,等待数据集重新生成
