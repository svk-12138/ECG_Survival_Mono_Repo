# 论文复现差距分析

论文：`European Heart Journal (2025), DOI: 10.1093/eurheartj/ehaf448`

## 已基本覆盖的部分

- 残差 1D CNN 风险模型主干仍在仓库中，训练主入口为 `scripts/run_survival_training.py` 和 `modules/survival_model/torch_survival/train_survival_from_json.py`。
- 内部评估脚本已经具备 C-index、风险四分位统计等基础能力，见 `scripts/survival_eval.py`。
- VAE 相关解释链路已存在，包括潜变量导出、线性回归打分、latent traversal 和 median beat 可视化。
- median beat 提取脚本已存在，见 `modules/survival_model/scripts/extract_median_ecg.py`。

## 本次修复前的关键偏差

- 主训练链路实际上是 `n_intervals=1 + BCEWithLogitsLoss` 的分类实验，不是论文的离散时间预测。
- 训练/推理预处理缺少论文明确写出的 `0.5-100Hz bandpass`、`60Hz notch`、`400Hz resample`、`zero padding to 4096`。
- XML 读取未显式优先 `WaveformType=Rhythm`。
- 推理输出默认是单概率，不是可配置的 horizon 风险。

## 本次已修复的部分

- 新增共享预处理模块 `modules/survival_model/torch_survival/ecg_preprocessing.py`：
  - 优先读取 `Rhythm` 波形
  - 带通滤波 `0.5-100Hz`
  - `60Hz` 陷波
  - 重采样到 `400Hz`
  - 补零/截断到 `4096`
- 训练主入口新增 `task_mode=prediction|classification`：
  - `prediction` 使用 `make_surv_targets()` + `SurvLikelihoodLoss`
  - `classification` 保留 `BCEWithLogitsLoss`
- 推理入口改为同样支持双模式，并支持 `prediction_horizon` 控制输出风险时间点。
- pipeline 配置和调度脚本已透传上述参数。

## 仍未完全复现的内容

这些差距主要来自数据规模限制或当前标签结构不完整，本次未强行伪复现：

- 论文是 3 个独立终点模型：`MR`、`AR`、`TR`；当前仓库仍是单终点 manifest 训练。
- 论文是 12 导联 ECG；当前主训练链路仍保留 8 导联输入。
- 论文数据对 ECG 和 TTE 先做 `<=60 天配对`，并区分 prevalent/future disease；当前 manifest 仅提供 `patient_id/time/event`，未包含完整配对逻辑。
- 论文训练/验证/测试按患者级划分，且评估时使用“每位患者首个 ECG”；当前主链路仍是随机样本切分。
- 论文包含外部验证（BIDMC）、亚组分析、竞争风险敏感性分析、Cox + echo 基线模型增益分析、serial ECG 风险轨迹分析；当前仓库没有足够数据完整复现。
- 论文包含 imaging association analysis；当前仓库仅有部分相关脚本，未形成完整复现结果。

## 建议的下一步

- 将单终点 manifest 扩展为多终点标签结构，分别训练 `MR/AR/TR`。
- 若原始 XML 可用，补齐 12 导联输入版本，与当前 8 导版本并行评估。
- 在 manifest 构建阶段补入 ECG-TTE 配对时间、基线是否已有中重度病变、患者级 split 信息。
- 若后续仍只有约 1200 条样本，建议把论文复现目标明确为：
  - 方法学复现
  - 工程链路复现
  - 小样本可运行验证
  而不是指标数值复现。
