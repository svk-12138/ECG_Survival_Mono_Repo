# 预处理前后对比 - 手动步骤指南

由于本地Python脚本通过SSH调用存在输出问题，建议采用以下手动步骤：

## 步骤1：先重新生成数据集（最重要！）

```bash
# SSH到服务器
ssh admin123@192.168.1.251

# 激活环境
cd /home/admin123/workspace/home_data/Program/FRcnn/Coronary_Angiography_Detection-main

# 🔧 确认preprocessing.py已更新
grep -n "yolo_norm_width" ICA_Detection/preprocessing/preprocessing.py
# 应该看到第333-343行有yolo_norm_width的定义

# 重新生成数据集
python scripts/dataset_generation/generate_dataset.py \
    --config scripts/dataset_generation/cfg_dsgen_combined.yaml
```

**预计时间**: 30-60分钟

## 步骤2：等待数据集生成完成后验证

```bash
# 检查生成的文件数量
find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/images/train -name "*.png" | wc -l
find /home/admin123/workspace/combined_stenosis_new/stenosis_detection/datasets/yolo/labels/train -name "*.txt" | wc -l

# 应该看到相同数量(约12000+)
```

## 步骤3：快速验证坐标（抽查方式）

手动检查几个800×800样本的YOLO标签是否合理：

```bash
# 查看一个样本的YOLO标签
cat /home/admin123/workspace/combined_stenosis_new/stenosis_detection/labels/yolo/kemerovo_p032_v4_00055.txt

# 应该看到归一化坐标在[0,1]范围内
# 修复后预期: 0 0.213 0.217 0.066 0.037 (归一化到512×512)
# 修复前错误: 0 0.385 0.371 0.102 0.059 (归一化到800×800)
```

## 步骤4：重新训练模型

```bash
cd /data/combined_stenosis/stenosis_detection

# 备份旧训练结果
mv runs/detect/train runs/detect/train_before_fix_$(date +%Y%m%d)

# 启动训练
./tune_with_existing_dataset.sh train
```

## 当前状态总结

✅ **已完成**:
- Bug定位: preprocessing.py 第350行
- 代码修复: 使用yolo_norm_width/height
- 上传到服务器: ✓

⏳ **进行中**:
- 等待重新生成数据集

❓ **问题**:
- 本地Python脚本通过SSH调用在Windows PowerShell中输出被抑制
- 建议直接在服务器上操作

## 预期改善

| 指标 | 修复前 | 修复后预期 |
|------|--------|-----------|
| 平均坐标误差 | 142.6px | <2px |
| mAP50 | 0.08 | >0.30 |
| 受影响样本 | 6722个(56%) | 全部修复 |

## 下一步

**最优先**: 重新生成数据集（SSH到服务器执行上述命令）

验证方法可以简化为：
1. 对比修复前后的YOLO标签文件内容
2. 直接开始训练，观察mAP50是否提升
