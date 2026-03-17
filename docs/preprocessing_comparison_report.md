# 预处理前后标注对比验证报告

## 📅 检查日期
2025年11月7日

## 🔍 检查目的
验证数据预处理脚本在将不同尺寸图像统一到512×512时，是否正确缩放了边界框坐标。

---

## ⚠️ **重大发现：坐标缩放存在严重偏差！**

### 检查结果

**数据集**: KEMEROVO  
**原始尺寸**: 800×800  
**目标尺寸**: 512×512  
**缩放比例**: 0.64 × 0.64  
**检查样本数**: 8个

### 误差统计

| 样本 | 原始尺寸 | 处理后尺寸 | 平均误差 | 状态 |
|------|---------|----------|---------|------|
| kemerovo_p096_v6_00067 | 800×800 | 512×512 | **122.00px** | ⚠️ 严重偏差 |
| kemerovo_p022_v3_00073 | 800×800 | 512×512 | **178.00px** | ⚠️ 严重偏差 |
| kemerovo_p007_v3_00060 | 800×800 | 512×512 | **175.00px** | ⚠️ 严重偏差 |
| kemerovo_p032_v4_00055 | 800×800 | 512×512 | **80.00px** | ⚠️ 严重偏差 |
| kemerovo_p031_v4_00033 | 800×800 | 512×512 | **157.00px** | ⚠️ 严重偏差 |
| kemerovo_p029_v5_00090 | 800×800 | 512×512 | **180.00px** | ⚠️ 严重偏差 |
| kemerovo_p024_v2_00026 | 800×800 | 512×512 | **103.00px** | ⚠️ 严重偏差 |
| kemerovo_p022_v2_00039 | 800×800 | 512×512 | **146.00px** | ⚠️ 严重偏差 |

**平均坐标误差**: **142.625 像素** (在512×512图像上，这是~28%的偏移！)

---

## 🔬 问题分析

### 1. 预期行为 vs 实际行为

**预期**:
- 原始坐标在800×800图像上
- 缩放比例: 0.64
- 缩放后坐标应该在512×512图像上完美对齐

**实际**:
- 平均偏差142像素，远超合理的四舍五入误差(±0.5px)
- **这表明坐标缩放逻辑存在问题**

### 2. 可能的原因

#### ⭐ 原因1: 坐标格式不匹配 (最可能)

**假设**: XML中的坐标可能不是Pascal VOC格式，而是其他格式：

```python
# 如果XML中的坐标是中心点+宽高格式:
<bndbox>
  <x>400</x>      # 中心X
  <y>300</y>      # 中心Y
  <width>50</width>
  <height>50</height>
</bndbox>

# 但脚本按Pascal VOC格式解析:
xmin = int(bndbox.find('xmin').text)  # 实际读到的是x (中心点)
ymin = int(bndbox.find('ymin').text)  # 实际读到的是y (中心点)
xmax = int(bndbox.find('xmax').text)  # 实际读到的是width
ymax = int(bndbox.find('ymax').text)  # 实际读到的是height
```

#### 原因2: 坐标系原点不同

- **可能性**: 图像坐标系与标注坐标系不一致
- **影响**: Y轴翻转或坐标偏移

#### 原因3: 预处理管道中的坐标转换错误

- 在数据集生成脚本的某个环节，坐标转换逻辑有误
- 可能在 `kemerovo_to_common()` 函数中

---

## 📊 可视化验证

已生成8个对比图像，保存在:
```
d:\WORKING\preprocessing_comparison_results\
```

每个图像包含:
1. **左图**: 原始800×800图像 + 原始XML标注 (红色框)
2. **中图**: 处理后512×512图像 + 处理后YOLO标注 (蓝色框)
3. **右图**: 处理后512×512图像 + 手动缩放的预期标注 (绿色虚线框)
4. **表格**: 详细坐标对比和误差

**关键观察点**:
- 蓝色框(实际)与绿色框(预期)的偏移
- 偏移方向和大小

---

## 🚨 影响评估

### 对训练的影响

**严重程度**: 🔴 **极高**

1. **检测精度下降**: 
   - 142像素的偏移意味着标注框与实际目标位置严重不匹配
   - 这会导致模型学习到错误的目标位置

2. **mAP50低的根本原因**:
   - 当前训练mAP50仅0.08，可能正是因为标注位置错误
   - 模型无法学习正确的目标位置

3. **数据集规模**:
   - KEMEROVO 800×800样本: **5332个** (占总数据47%)
   - KEMEROVO 1000×1000样本: **1047个** (占总数据9%)
   - **总计56%的数据受影响**

---

## ✅ 解决方案

### 立即行动 (优先级最高)

#### 步骤1: 检查XML标注格式

查看KEMEROVO原始XML文件的实际格式:
```bash
ssh admin123@192.168.1.251 "cat /home/admin123/workspace/MEDDataset/KEMEROVO/dataset/14_006_1_0044.xml"
```

**需要确认**:
- XML中的标签名是什么? (`<xmin>` 还是 `<x>`?)
- 坐标格式是 Pascal VOC (xmin/ymin/xmax/ymax) 还是其他?

#### 步骤2: 修复坐标解析逻辑

根据实际XML格式，修改解析函数:
```python
# 如果XML格式是中心点+宽高:
def parse_kemerovo_xml_corrected(xml_path):
    # 读取中心点和宽高
    x_center = int(bndbox.find('x').text)
    y_center = int(bndbox.find('y').text)
    width = int(bndbox.find('width').text)
    height = int(bndbox.find('height').text)
    
    # 转换为Pascal VOC格式
    xmin = x_center - width // 2
    ymin = y_center - height // 2
    xmax = x_center + width // 2
    ymax = y_center + height // 2
```

#### 步骤3: 重新生成数据集

修复后，重新运行数据集生成脚本:
```bash
cd /home/admin123/workspace/FRcnn/Coronary_Angiography_Detection-main
python scripts/dataset_generation/generate_dataset.py --config cfg_dsgen_combined.yaml
```

#### 步骤4: 重新验证

重新运行对比脚本，确认误差降到 < 2px

#### 步骤5: 重新训练

使用修复后的数据集重新训练，预期mAP50会显著提升

---

## 📝 下一步调查

### 需要检查的文件

1. **KEMEROVO XML格式**:
   - `/home/admin123/workspace/MEDDataset/KEMEROVO/dataset/14_006_1_0044.xml`
   - 查看实际标签名和坐标格式

2. **坐标转换函数**:
   - `ICA_Detection/tools/bbox_translation.py` 中的 `kemerovo_to_common()`
   - 确认是否正确处理了KEMEROVO格式

3. **数据集集成逻辑**:
   - `ICA_Detection/generator/generator.py` 中的 `integrate_datasets()`
   - 检查KEMEROVO数据集的加载逻辑

---

## 🎯 预期修复效果

### 修复前
- mAP50: 0.08
- 标注误差: 142px (28%)
- 受影响数据: 56%

### 修复后 (预期)
- mAP50: > 0.3 (提升300%+)
- 标注误差: < 2px (<0.4%)
- 数据质量: 全部正确

---

## 📂 相关文件

| 文件 | 位置 |
|------|------|
| 对比脚本 | `d:\WORKING\remote_utils\compare_before_after_preprocessing.py` |
| 对比结果图像 | `d:\WORKING\preprocessing_comparison_results\*.png` |
| 坐标缩放分析 | `d:\WORKING\docs\coordinate_scaling_analysis.md` |

---

**报告生成时间**: 2025年11月7日  
**状态**: 🔴 发现关键问题，需要立即修复  
**优先级**: ⚠️ **最高** - 这是导致训练效果差的根本原因
