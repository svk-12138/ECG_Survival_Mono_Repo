# 卒中论文快速开始

这份说明按 Win11 医生用户来写。

## 第 1 步：改参数文件

请打开：

- [scripts/train_stroke_thesis.ps1](../scripts/train_stroke_thesis.ps1)

不要改下面这个 `.bat` 文件，它只是负责启动：

- [scripts/train_stroke_thesis.bat](../scripts/train_stroke_thesis.bat)

## 第 2 步：只改最上面的 4 个参数

必须改：

- `$Manifest`
- `$XmlDir` 或 `$CsvDir` 二选一
- `$TaskMode`
- `$LeadMode`

路径说明：

- 可以写绝对路径，例如 `D:\stroke_project\stroke_manifest.json`
- 也可以写相对项目根目录的路径，例如 `data\stroke_manifest.json`
- 如果路径里有空格或中文，保留外面的双引号即可

Win11 最常用写法：

```powershell
$Manifest = "D:\stroke_project\stroke_manifest.json"
$XmlDir = "D:\stroke_project\xml_dir"
$CsvDir = ""
$TaskMode = "prediction"
$LeadMode = "12lead"
```

如果你用的是 CSV：

```powershell
$XmlDir = ""
$CsvDir = "D:\stroke_project\csv_dir"
```

## 第 3 步：启动训练

在项目根目录双击或运行：

```bat
scripts\train_stroke_thesis.bat
```

如果你习惯 PowerShell，也可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_stroke_thesis.ps1
```

## 跑完后去哪里看结果

默认输出目录：

```text
outputs\stroke_survival_thesis
```

重点看：

- `model_final.pt`
- `training_metrics.csv`
- `best_threshold.json`
- `cv_summary.json`（如果开了交叉验证）

## 最常用的 4 组实验

主实验：

```powershell
$TaskMode = "prediction"
$LeadMode = "12lead"
```

导联对照：

```powershell
$TaskMode = "prediction"
$LeadMode = "8lead"
```

分类基线：

```powershell
$TaskMode = "classification"
$LeadMode = "12lead"
```

简化基线：

```powershell
$TaskMode = "classification"
$LeadMode = "8lead"
```

## 对 1200 条卒中数据的建议

先用这组参数作为论文主实验起点：

```powershell
$TaskMode = "prediction"
$LeadMode = "12lead"
$NIntervals = 20
$CVFolds = 5
$PredictionHorizon = "null"
$ApplyFilters = $true
```

含义是：

- 用生存预测作为主任务
- 先跑 12 导模型
- 用 5 折交叉验证应对小样本
- 打开 ECG 滤波，补上杂波处理

## Linux / macOS / WSL 用户怎么用

如果不是 Win11，而是 Linux / macOS / WSL，请改：

- [scripts/train_stroke_thesis.sh](../scripts/train_stroke_thesis.sh)

然后运行：

```bash
bash scripts/train_stroke_thesis.sh
```
