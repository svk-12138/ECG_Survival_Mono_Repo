# 数据集生成脚本坐标缩放逻辑分析报告

## 检查日期
2025年11月7日

## 问题描述
第三个数据集(KEMEROVO)中存在不同尺寸的图像(512×512, 800×800, 1000×1000)，需要确认在统一图像尺寸到512×512时，是否对标注坐标进行了相应的缩放。

---

## ✅ 结论：坐标缩放已正确实现

**关键发现**: 脚本在 `resolution_standarization` 步骤中**已经正确实现了坐标缩放**。

---

## 详细分析

### 1. 预处理流程

**文件位置**: `ICA_Detection/preprocessing/preprocessing.py`

**处理步骤** (按顺序):
1. `format_standarization`: 格式标准化 (转换为PNG)
2. `dtype_standarization`: 数据类型标准化
3. **`resolution_standarization`**: **分辨率标准化 + 坐标缩放** ⭐
4. `filtering_smoothing_equalization`: 滤波/平滑/均衡化

### 2. 分辨率标准化的实现

**代码位置**: `preprocessing.py` 第145-225行

#### 关键逻辑:

```python
# 3d. Resolution Standardization + Bbox/Mask Rescaling
if (
    "resolution_standarization" in entry.get("preprocessing_plan", {})
    and "resolution_standarization" in steps_order
):
    res_plan = entry["preprocessing_plan"]["resolution_standarization"]
    desired_X = res_plan.get("desired_X", 512)  # 目标宽度
    desired_Y = res_plan.get("desired_Y", 512)  # 目标高度
    method = res_plan.get("method", "bilinear")

    # 保存原始尺寸
    old_width = img_info.get("width")    # 例如: 800, 1000, 512
    old_height = img_info.get("height")  # 例如: 800, 1000, 512

    # 调整图像尺寸
    new_img_path = current_img_path
    ret = apply_resolution(current_img_path, new_img_path, desired_X, desired_Y, method)
    
    # 更新JSON中存储的尺寸
    img_info["width"] = desired_X   # 512
    img_info["height"] = desired_Y  # 512

    # ========== 关键: 缩放所有边界框 ==========
    annotations = entry.get("annotations", {})
    
    if task == "detection" and old_width and old_height:
        # 遍历所有bbox (bbox1, bbox2, bbox3, ...)
        for key in list(annotations.keys()):
            if key.startswith("bbox") and isinstance(annotations[key], dict):
                annotations[key] = rescale_bbox(
                    annotations[key], 
                    old_width,      # 原始尺寸 (800/1000/512)
                    old_height, 
                    desired_X,      # 目标尺寸 (512)
                    desired_Y
                )
        
        # 如果有stenosis字典包含多个bbox
        if "stenosis" in annotations and isinstance(annotations["stenosis"], dict):
            for bbox_key in list(annotations["stenosis"].keys()):
                if bbox_key.startswith("bbox"):
                    annotations["stenosis"][bbox_key] = rescale_bbox(
                        annotations["stenosis"][bbox_key],
                        old_width,
                        old_height,
                        desired_X,
                        desired_Y
                    )
```

### 3. 坐标缩放函数实现

**文件位置**: `ICA_Detection/tools/bbox_translation.py`

```python
def rescale_bbox(bbox, orig_width, orig_height, new_width, new_height):
    """
    根据图像尺寸变化缩放Pascal VOC格式的边界框
    
    Args:
        bbox (dict): 边界框 {"xmin", "ymin", "xmax", "ymax", "label"}
        orig_width (int): 原始图像宽度 (例如: 800, 1000)
        orig_height (int): 原始图像高度
        new_width (int): 新图像宽度 (512)
        new_height (int): 新图像高度 (512)
    
    Returns:
        dict: 缩放后的边界框
    """
    # 计算缩放比例
    scale_x = new_width / orig_width      # 512/800 = 0.64  或 512/1000 = 0.512
    scale_y = new_height / orig_height    # 512/800 = 0.64  或 512/1000 = 0.512
    
    # 应用缩放到所有坐标
    return {
        "xmin": np.round(bbox["xmin"] * scale_x, 0),  # 四舍五入到整数
        "ymin": np.round(bbox["ymin"] * scale_y, 0),
        "xmax": np.round(bbox["xmax"] * scale_x, 0),
        "ymax": np.round(bbox["ymax"] * scale_y, 0),
        "label": bbox.get("label", ""),
    }
```

### 4. 缩放示例计算

#### 场景 1: 800×800 → 512×512

**原始标注** (在800×800图像上):
```
xmin: 200, ymin: 300, xmax: 250, ymax: 350
宽度: 50, 高度: 50
```

**缩放比例**:
```
scale_x = 512 / 800 = 0.64
scale_y = 512 / 800 = 0.64
```

**缩放后坐标** (在512×512图像上):
```
xmin: 200 × 0.64 = 128
ymin: 300 × 0.64 = 192
xmax: 250 × 0.64 = 160
ymax: 350 × 0.64 = 224
宽度: 32, 高度: 32  (保持相对大小比例)
```

#### 场景 2: 1000×1000 → 512×512

**原始标注** (在1000×1000图像上):
```
xmin: 400, ymin: 500, xmax: 460, ymax: 550
宽度: 60, 高度: 50
```

**缩放比例**:
```
scale_x = 512 / 1000 = 0.512
scale_y = 512 / 1000 = 0.512
```

**缩放后坐标** (在512×512图像上):
```
xmin: 400 × 0.512 = 205 (四舍五入)
ymin: 500 × 0.512 = 256
xmax: 460 × 0.512 = 236
ymax: 550 × 0.512 = 282
宽度: 31, 高度: 26  (保持相对大小比例)
```

#### 场景 3: 512×512 → 512×512

**缩放比例**:
```
scale_x = 512 / 512 = 1.0
scale_y = 512 / 512 = 1.0
```

**结果**: 坐标不变 (scale = 1.0)

---

## 配置验证

### 配置文件: `cfg_dsgen_combined.yaml`

```yaml
preprocessing:
  detection:
    plan_name: "Stenosis_Detection"
    steps:
      format_standarization:
        desired_format: "png"
      dtype_standarization:
        desired: "uint8"
      resolution_standarization:
        desired_X: 512  # ⭐ 目标宽度
        desired_Y: 512  # ⭐ 目标高度
        method: "bilinear"
```

### 数据集处理配置

```yaml
dataset_processing:
  datasets_to_process:
    - "CADICA"    # 主要是512×512
    - "ARCADE"    # 主要是512×512
    - "KEMEROVO"  # ⭐ 混合尺寸: 512, 800, 1000
```

---

## 坐标格式转换流程

### KEMEROVO 数据集的完整处理流程:

```
1. 原始标注格式 (Pascal VOC):
   {"xmin": 400, "ymin": 500, "xmax": 460, "ymax": 550, "label": "stenosis"}
   图像尺寸: 1000×1000

2. 统一格式 (common format):
   kemerovo_to_common() → 
   {"xmin": 400, "ymin": 500, "xmax": 460, "ymax": 550, "label": "stenosis"}
   (KEMEROVO已经是Pascal VOC格式，无需转换)

3. 图像缩放:
   apply_resolution() → 
   图像从 1000×1000 resize到 512×512

4. 坐标缩放:
   rescale_bbox() → 
   {"xmin": 205, "ymin": 256, "xmax": 236, "ymax": 282, "label": "stenosis"}

5. 转换为YOLO格式 (归一化):
   common_to_yolo() → 
   x_center = (205 + 236) / 2 / 512 = 0.4307
   y_center = (256 + 282) / 2 / 512 = 0.5254
   width    = (236 - 205) / 512 = 0.0605
   height   = (282 - 256) / 512 = 0.0508
   
   YOLO格式: "0 0.4307 0.5254 0.0605 0.0508"
```

---

## 潜在的精度损失

### 四舍五入误差

**代码中使用**:
```python
np.round(bbox["xmin"] * scale_x, 0)  # 四舍五入到整数
```

**影响分析**:
- **误差范围**: ±0.5像素
- **对于512×512图像**: 最大误差 ±0.5/512 ≈ ±0.1%
- **影响程度**: **可忽略不计**

**示例**:
```
原始: xmin = 401 (在1000×1000上)
缩放: 401 × 0.512 = 205.312
四舍五入: 205
精确值: 205.312
误差: 0.312像素 (在512×512图像上)
归一化误差: 0.312/512 = 0.061%
```

### 为什么不是问题

1. **YOLO格式是归一化的**: 最终训练时使用[0,1]范围的坐标
2. **目标很小**: 平均目标仅48×45像素，±0.5像素误差在合理范围内
3. **训练时的容错性**: 深度学习模型对亚像素级误差有很好的鲁棒性

---

## 问题排查建议

虽然坐标缩放逻辑正确，但之前发现的**标注宽高比异常问题**可能来自:

### 1. 原始标注质量 (最可能 ⭐⭐⭐⭐⭐)

**KEMEROVO数据集的原始标注可能就存在问题**:
- 标注工具操作失误
- 血管段标注过长
- 非狭窄区域被错误标注

**验证方法**:
```bash
# 查看原始尺寸图像的标注
# 如果原始1000×1000图像的标注就是宽高比7:1，那缩放到512也会保持7:1
```

### 2. 缩放插值方法

**当前配置**: `method: "bilinear"`

**影响**: 
- 仅影响图像质量，不影响坐标
- 坐标缩放是纯数学计算，与插值方法无关

### 3. 不同数据集的标注标准不一致

**观察**:
- CADICA/ARCADE: 可能标注的是"狭窄点"周围的小区域
- KEMEROVO: 可能标注的是"整个狭窄段"

**证据**:
```
abnormal_aspect_ratio问题主要集中在:
- kemerovo_p006_v1 系列 (30+ 张)
- kemerovo_p029_v4 系列
- kemerovo_p053_v1 系列
```

---

## 代码改进建议

### 当前代码优点 ✅

1. **逻辑正确**: 缩放比例计算准确
2. **覆盖全面**: 处理所有bbox变体 (bbox1, bbox2, stenosis/bbox等)
3. **保持比例**: 使用相同的scale_x和scale_y (假设原图是正方形)

### 潜在改进 💡

#### 1. 添加坐标验证

```python
def rescale_bbox(bbox, orig_width, orig_height, new_width, new_height):
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    rescaled = {
        "xmin": np.round(bbox["xmin"] * scale_x, 0),
        "ymin": np.round(bbox["ymin"] * scale_y, 0),
        "xmax": np.round(bbox["xmax"] * scale_x, 0),
        "ymax": np.round(bbox["ymax"] * scale_y, 0),
        "label": bbox.get("label", ""),
    }
    
    # 验证缩放后的坐标是否合法
    assert 0 <= rescaled["xmin"] < new_width, f"xmin超出范围: {rescaled['xmin']}"
    assert 0 <= rescaled["ymin"] < new_height, f"ymin超出范围: {rescaled['ymin']}"
    assert rescaled["xmin"] < rescaled["xmax"], "xmin >= xmax"
    assert rescaled["ymin"] < rescaled["ymax"], "ymin >= ymax"
    
    return rescaled
```

#### 2. 记录缩放日志

```python
# 在preprocessing.py中添加
if old_width != desired_X or old_height != desired_Y:
    logger.info(
        f"{uid}: 图像从 {old_width}×{old_height} 缩放到 {desired_X}×{desired_Y}, "
        f"scale=({scale_x:.3f}, {scale_y:.3f})"
    )
```

#### 3. 保留原始坐标用于审计

```python
# 保存缩放前的坐标到元数据
annotations["_original_bbox_before_scaling"] = copy.deepcopy(annotations["bbox1"])
annotations["bbox1"] = rescale_bbox(...)
```

---

## 总结

### ✅ 确认事项

1. **坐标缩放已正确实现**: `rescale_bbox()` 函数按比例缩放所有坐标
2. **覆盖所有数据集**: CADICA, ARCADE, KEMEROVO 都经过相同的处理流程
3. **缩放比例正确**:
   - 512→512: scale=1.0 (无变化)
   - 800→512: scale=0.64
   - 1000→512: scale=0.512

### ⚠️ 需要注意

1. **原始标注质量**: 异常宽高比问题**不是坐标缩放导致的**，而是原始标注的问题
2. **四舍五入误差**: 存在但可忽略 (±0.5像素)
3. **数据集一致性**: 不同数据集的标注粒度可能不同

### 📝 建议行动

1. **验证原始标注**: 在KEMEROVO原始数据上检查宽高比异常的样本
2. **过滤异常标注**: 使用之前的 `check_annotation_quality.py` 结果过滤
3. **继续训练**: 坐标缩放逻辑无问题，可以放心使用

---

## 参考代码文件

| 文件 | 功能 |
|------|------|
| `generate_dataset.py` | 主脚本,调用DatasetGenerator |
| `preprocessing.py` | 核心预处理逻辑,包含resolution_standarization |
| `bbox_translation.py` | 坐标格式转换和缩放函数 |
| `cfg_dsgen_combined.yaml` | 配置文件,指定desired_X/Y=512 |

---

**分析完成日期**: 2025年11月7日  
**分析结论**: ✅ 坐标缩放逻辑正确，无需修改  
**后续行动**: 关注原始标注质量问题
