# 卒中论文数据处理统一流程

这份文档对应新的“单入口数据处理”方案，目标是把医生手里的原始标签表和 XML 数据，稳定转换为训练直接可用的 `manifest.json`。

适用场景：

- 标签来自 CSV / Excel
- 标签表至少包含 4 列：`patient_SN`、`event`、`time`、`xml_file`
- ECG 原始数据来自 XML
- 同一个患者可能有多次 ECG 检查，必须全部保留，不能去重

## 为什么要统一这条流程

统一后有 3 个好处：

1. `xml_file` 和具体 ECG 检查一一对应，不再靠“一个患者只取一条 ECG”的隐式假设
2. 训练切分按 `patient_SN` 分组，避免同一患者泄露到训练集和验证集
3. 处理阶段就能提前审计 XML 是否支持 8 导 / 12 导，减少训练时才发现导联缺失

## 医生需要准备什么

标签表必须至少包含下面 4 列：

- `patient_SN`
- `event`
- `time`
- `xml_file`

字段含义：

- `patient_SN`：患者分组键，用于训练划分，防止数据泄露
- `event`：结局事件，通常是 `0/1`
- `time`：结局时间
- `xml_file`：这次 ECG 对应的 XML 文件名或相对路径

处理后生成的 manifest 会保留：

- `patient_SN`
- `patient_id`
- `event`
- `time`
- `xml_file`

其中：

- `patient_id` 从 XML 中提取
- `patient_SN` 仍然保留，用于训练分组

## 处理命令

在仓库根目录运行：

```bash
python scripts/prepare_survival_dataset.py \
  --labels path/to/labels.csv \
  --xml-dir path/to/XML \
  --output-dir processed
```

Windows PowerShell 示例：

```powershell
python scripts/prepare_survival_dataset.py `
  --labels "D:\stroke_project\labels.csv" `
  --xml-dir "D:\stroke_project\XML" `
  --output-dir "D:\stroke_project\processed"
```

## 处理后会生成什么

输出目录下会生成：

- `manifest.json`
- `training_inputs.json`
- `处理报告.txt`
- `process_report.json`
- `audit\lead_audit.csv`
- 其他异常明细 CSV

重点文件说明：

- `manifest.json`
  训练直接读取的清单文件
- `training_inputs.json`
  给训练入口自动读取的路径配置
- `处理报告.txt`
  医生优先阅读的人类可读报告
- `lead_audit.csv`
  每份 XML 在指定 `waveform_type` 下的导联情况审计

## 什么时候会失败

脚本会在下面几类问题出现时直接失败，并且不保留可训练 manifest：

- 标签表缺列
- `xml_file` 找不到对应 XML
- `event/time/patient_SN` 非法或为空
- XML 无法解析或取不到 `PatientID`

这时请先看：

- `处理报告.txt`
- `process_report.json`
- `audit\missing_xml_rows.csv`
- `audit\invalid_label_rows.csv`
- `audit\invalid_patient_rows.csv`

## 与训练入口怎么衔接

如果 `processed/training_inputs.json` 存在，`scripts/train_stroke_thesis.ps1` 会自动优先读取：

- `manifest`
- `xml_dir / csv_dir`
- `recommended_lead_mode`
- `waveform_type`

也就是说，数据处理完成后，训练入口不需要再手工改这些路径。

## 分组规则

训练切分现在按下面规则执行：

1. 优先使用 `patient_SN`
2. 如果旧 manifest 没有 `patient_SN`，回退到 `patient_id`

因此：

- 新数据请统一提供 `patient_SN`
- 旧数据仍可兼容，但新论文实验应优先使用 `patient_SN`

## 关于 8 导和 12 导

处理脚本会基于 XML 波形审计给出：

- `requested_waveform_supports_8lead`
- `requested_waveform_supports_12lead`
- `recommended_lead_mode`

如果 `recommended_lead_mode=8lead`，说明你当前指定的波形类型下，并不是每份 XML 都稳定支持 12 导；这时不要强行按 12 导训练，先看 `lead_audit.csv`。
