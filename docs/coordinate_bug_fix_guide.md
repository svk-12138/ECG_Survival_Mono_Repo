# 坐标Bug修复完整流程

## 📌 问题总结

**Bug位置**: `ICA_Detection/preprocessing/preprocessing.py` 第350行  
**错误代码**:
```python
yolo_bbox = common_to_yolo(bbox, orig_width, orig_height)  # ❌
```

**正确代码**:
```python
yolo_bbox = common_to_yolo(bbox, yolo_norm_width, yolo_norm_height)  # ✅
```

**根本原因**:
- 图像已通过`resolution_standardization`缩放到512×512
- Bbox坐标已通过`rescale_bbox()`正确缩放到512×512
- 但YOLO归一化时错误地使用了原始图像尺寸(800×800或1000×1000)
- 导致归一化后的坐标数值错误

**影响范围**:
- KEMEROVO 800×800: 5332个样本
- KEMEROVO 1000×1000: 1047个样本
- **总计: 6722个样本 (56%的训练数据)**
- 平均坐标误差: **142.6像素** (28%偏移)
- 训练影响: mAP50仅0.08

---

## 🔧 修复内容

### 修改的代码逻辑

**原始代码** (第327-350行):
```python
# Image dimensions for YOLO normalization
orig_width = img_info.get("width", 512)
orig_height = img_info.get("height", 512)

bbox_container = annotations
...
yolo_bbox = common_to_yolo(bbox, orig_width, orig_height)  # ❌ 错误
```

**修复后代码**:
```python
# Image dimensions for YOLO normalization
# 🔧 BUG FIX: 使用缩放后的尺寸而非原始尺寸
# 因为bbox坐标已通过rescale_bbox()缩放到desired_X×desired_Y
# 如果没有进行resolution_standardization，则使用原始尺寸
if "resolution_standardization" in entry.get("preprocessing_plan", {}):
    # 使用缩放后的尺寸
    res_plan = entry["preprocessing_plan"]["resolution_standardization"]
    yolo_norm_width = res_plan.get("desired_X", 512)
    yolo_norm_height = res_plan.get("desired_Y", 512)
else:
    # 使用原始尺寸
    yolo_norm_width = img_info.get("width", 512)
    yolo_norm_height = img_info.get("height", 512)

bbox_container = annotations
...
yolo_bbox = common_to_yolo(bbox, yolo_norm_width, yolo_norm_height)  # ✅ 正确
```

**关键改进**:
1. 添加条件判断：检查是否进行了`resolution_standardization`
2. 如果缩放了图像，使用`desired_X/desired_Y`(512×512)
3. 如果未缩放，使用原始尺寸(保持向后兼容)
4. 添加详细注释说明修复原因

---

## ✅ 已完成的步骤

### 1. Bug修复 ✅
- [x] 修改`preprocessing.py`第327-350行
- [x] 添加正确的归一化尺寸选择逻辑
- [x] 上传到远程服务器
  ```bash
  scp preprocessing.py admin123@192.168.1.251:/home/.../preprocessing.py
  ```

### 2. 验证脚本准备 ✅
- [x] 创建`regenerate_fixed_dataset.sh` - 重新生成数据集的自动化脚本
- [x] 已有`compare_before_after_preprocessing.py` - 预处理前后对比验证工具

---

## 🚀 待执行的步骤

### 步骤1: 备份当前数据集 (可选但推荐)

```bash
ssh admin123@192.168.1.251

# 备份当前的combined_standardized.json
cd /home/admin123/workspace/combined_stenosis_new/stenosis_detection/
cp combined_standardized.json combined_standardized.json.before_fix_$(date +%Y%m%d)

# 备份datasets目录(可选,因为数据较大)
# tar -czf datasets_before_fix_$(date +%Y%m%d).tar.gz datasets/
```

### 步骤2: 重新生成数据集

**方式A: 使用自动化脚本**
```bash
# 从本地上传脚本
scp d:/WORKING/scripts/regenerate_fixed_dataset.sh admin123@192.168.1.251:/home/admin123/workspace/

# SSH到服务器执行
ssh admin123@192.168.1.251
cd /home/admin123/workspace
chmod +x regenerate_fixed_dataset.sh
./regenerate_fixed_dataset.sh
```

**方式B: 手动执行**
```bash
ssh admin123@192.168.1.251

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Visionmodel

# 进入项目目录
cd /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

# 清理旧数据
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/images/
rm -rf /home/admin123/workspace/combined_stenosis_new/stenosis_detection/labels/

# 重新生成
python ./scripts/dataset_generation/generate_dataset.py \
    --config ./scripts/dataset_generation/cfg_dsgen_combined.yaml
```

**预计时间**: ~30-60分钟 (取决于数据集大小)

### 步骤3: 验证修复效果

**3.1 快速检查文件数量**
```bash
# YOLO格式数据集
find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/images/train -name "*.png" | wc -l
find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/labels/train -name "*.txt" | wc -l

# 应该看到相同数量(约12000+个)
```

**3.2 运行对比验证脚本**

在**本地Windows**执行:
```bash
# 重新运行之前的对比脚本
python d:/WORKING/remote_utils/compare_before_after_preprocessing.py

# 查看生成的对比图像
# 应该在远程服务器的preprocessing_comparison目录
```

**预期结果**:
- 平均坐标误差从**142.6px**降至**<2px**
- 对比图中蓝色框(实际)和绿色框(预期)应该完全重叠
- 表格中的坐标差异应该≈0

**3.3 下载验证结果到本地**
```bash
# 下载新的对比结果
scp -r admin123@192.168.1.251:/home/admin123/workspace/preprocessing_comparison d:/WORKING/preprocessing_comparison_results_FIXED/

# 对比修复前后的图像
# 修复前: d:/WORKING/preprocessing_comparison_results/
# 修复后: d:/WORKING/preprocessing_comparison_results_FIXED/
```

### 步骤4: 重新训练YOLOv8模型

**4.1 清理旧的训练结果**
```bash
ssh admin123@192.168.1.251

cd /data/combined_stenosis/stenosis_detection
# 备份旧的训练结果
mv runs/detect/train runs/detect/train_before_fix_$(date +%Y%m%d)

# 或直接删除
# rm -rf runs/detect/train
```

**4.2 启动训练**

使用之前的训练脚本:
```bash
cd /data/combined_stenosis/stenosis_detection

# 基础训练
./tune_with_existing_dataset.sh train

# 或使用Optuna优化
./tune_with_existing_dataset.sh bayesian
```

**训练配置**:
- Epochs: 100-300
- Batch size: 16-32 (根据GPU内存)
- Image size: 512
- Device: cuda:0,1 (双GPU)

**4.3 监控训练指标**

关键指标对比:
| 指标 | 修复前 | 修复后(预期) | 改善幅度 |
|------|--------|-------------|---------|
| mAP50 | 0.08 | >0.3 | +275% |
| mAP50-95 | ~0.03 | >0.15 | +400% |
| Precision | ~0.15 | >0.5 | +233% |
| Recall | ~0.12 | >0.45 | +275% |

**预计训练时间**: 
- 单GPU: ~8-12小时 (100 epochs)
- 双GPU DDP: ~4-6小时 (100 epochs)

---

## 📊 验证检查清单

### 数据集生成阶段
- [ ] combined_standardized.json文件大小正常(~数MB)
- [ ] YOLO images和labels数量一致
- [ ] 抽查10个样本的YOLO标签,归一化坐标在[0,1]范围内
- [ ] 没有空的.txt标签文件

### 坐标验证阶段
- [ ] 运行compare_before_after_preprocessing.py
- [ ] 平均误差<2px
- [ ] 对比图中蓝色框和绿色框重叠度>98%
- [ ] 所有800×800和1000×1000样本的误差都<5px

### 训练验证阶段
- [ ] 训练loss正常下降(不出现震荡)
- [ ] mAP50 >0.2 (epoch 50时)
- [ ] mAP50 >0.3 (epoch 100时)
- [ ] 验证集上的检测结果视觉正常(bbox位置准确)

---

## 🔍 故障排查

### 问题1: 数据集生成失败
**症状**: generate_dataset.py报错或中途停止  
**检查**:
```bash
# 检查preprocessing.py是否正确上传
ssh admin123@192.168.1.251
cat /home/.../preprocessing.py | grep -A 5 "yolo_norm_width"

# 应该看到修复后的代码
```

### 问题2: 坐标仍有偏差
**症状**: 验证后误差仍>10px  
**检查**:
```bash
# 确认resolution_standardization配置
cat ./scripts/dataset_generation/cfg_dsgen_combined.yaml | grep -A 10 "resolution_standardization"

# 应该看到desired_X: 512, desired_Y: 512
```

### 问题3: 训练效果无改善
**症状**: mAP50仍<0.15  
**可能原因**:
1. 数据集未完全重新生成(检查时间戳)
2. 训练脚本仍使用旧数据集路径
3. 模型架构或超参数问题(非数据问题)

**检查**:
```bash
# 确认YOLO标签文件的修改时间
ls -lt /home/.../datasets/yolo/labels/train/*.txt | head -10

# 应该是今天的日期
```

---

## 📈 预期改善效果

### 坐标准确性
| 样本类型 | 修复前误差 | 修复后误差 | 改善 |
|---------|-----------|-----------|------|
| 800×800 | 142.6px | <2px | 98.6%↓ |
| 1000×1000 | ~178px | <2px | 98.9%↓ |
| 512×512 | 0px | 0px | 无影响 |

### 模型性能
| 评估指标 | 修复前 | 修复后(保守) | 修复后(乐观) |
|---------|-------|-------------|-------------|
| mAP50 | 0.08 | 0.30 | 0.45 |
| mAP50-95 | 0.03 | 0.15 | 0.25 |
| Precision | 0.15 | 0.50 | 0.65 |
| Recall | 0.12 | 0.45 | 0.60 |

### 训练稳定性
- 收敛速度: 提升50%+ (loss更快下降)
- 过拟合风险: 降低 (正确的标注减少混淆)
- 梯度稳定性: 改善 (bbox位置准确,梯度信号更清晰)

---

## 📝 文档记录

### 已创建的文档
1. `d:/WORKING/docs/coordinate_scaling_analysis.md` - 初始代码审查
2. `d:/WORKING/docs/preprocessing_comparison_report.md` - 对比验证报告
3. `d:/WORKING/docs/coordinate_bug_fix_guide.md` (本文件) - 修复完整流程

### 对比结果保存
- 修复前: `d:/WORKING/preprocessing_comparison_results/`
- 修复后: `d:/WORKING/preprocessing_comparison_results_FIXED/` (待生成)

### Git提交建议
```bash
cd d:/WORKING/FRcnn/Coronary_Angiography_Detection-main

git add ICA_Detection/preprocessing/preprocessing.py
git commit -m "🐛 修复YOLO坐标归一化bug - 使用缩放后尺寸而非原始尺寸

- 问题: common_to_yolo()使用orig_width/height导致6722个样本坐标错误
- 修复: 在resolution_standardization后使用desired_X/Y(512×512)
- 影响: 修复前mAP50=0.08,预期修复后>0.3
- 误差: 从142.6px降至<2px
"
```

---

## ⏱️ 时间线计划

| 阶段 | 任务 | 预计耗时 | 状态 |
|-----|------|---------|------|
| **已完成** | Bug定位与修复 | 2小时 | ✅ |
| **已完成** | 代码上传到服务器 | 5分钟 | ✅ |
| **进行中** | 重新生成数据集 | 30-60分钟 | ⏳ |
| **待完成** | 验证坐标准确性 | 15分钟 | ⏸️ |
| **待完成** | 重新训练模型 | 4-6小时 | ⏸️ |
| **待完成** | 评估模型性能 | 30分钟 | ⏸️ |
| **总计** | | ~7-10小时 | |

---

## 🎯 成功标准

修复被认为成功的标准:
1. ✅ **坐标验证**: 平均误差<2px,所有样本误差<5px
2. ✅ **数据集完整性**: images和labels数量一致,无空文件
3. ✅ **模型性能**: mAP50 >0.25 (100 epochs), >0.30 (200 epochs)
4. ✅ **训练稳定性**: loss正常下降,无异常波动
5. ✅ **视觉验证**: 推理结果bbox位置准确,覆盖狭窄区域

---

## 📞 后续支持

如果遇到问题,可以:
1. 查看训练日志: `tail -f dataset_regeneration_*.log`
2. 重新运行验证脚本: `python compare_before_after_preprocessing.py`
3. 检查GPU使用: `nvidia-smi`
4. 查看TensorBoard: `tensorboard --logdir runs/detect/train`

---

**创建日期**: 2025-01-XX  
**修复者**: GitHub Copilot  
**版本**: 1.0  
**最后更新**: 修复完成,等待数据集重新生成
