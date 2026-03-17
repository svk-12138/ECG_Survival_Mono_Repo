# ECG_Survival_Mono_Repo 项目整理（截至 2026-01-15）

## 1. 数据信息
- 数据来源：私密数据，来源于大医一院董浩宇
- 数据量：1200 条
- 最长随访年限：10 年
- 病种：卒中
- 事件占比：16%

**数据字段示例（manifest.json 单条）**
```json
{
  "patient_id": "P000123",
  "time": 1825,
  "event": 1,
}
```
说明：`time` 为随访时间（单位与训练配置保持一致），`event` 为事件标签（0/1）。

## 2. 模型信息
### 2.1 主模型（ResNet1d）
- 位置：`modules/survival_model/torch_age/resnet_age.py`
- 架构：1D 卷积 + 4 个残差块（逐级下采样）+ 全连接输出
- 输入：8 导联 ECG（I, II, V1–V6），每导联重采样至 4096
- 输出：
  - 回归（离散时间生存）：`n_intervals` 维条件生存概率
  - 分类（二分类）：1 个 logit（通过 sigmoid 得到概率）

## 3. 预测方向与方法
### 3.1 回归：离散时间生存预测
- 预测目标：事件发生风险（时间序列上的生存概率曲线）
- 输出形式：离散时间区间的条件生存概率（随后可转换为风险分数）
- 评价指标：C-index（主）、AUROC、Brier 等

### 3.2 分类：事件二分类
- 预测目标：事件是否发生（0/1）
- 输出形式：事件概率（sigmoid(logit)）
- 评价指标：AUC、Brier

### 3.3 指标释义
- **C-index**：看“排序对不对”。模型给的风险越高，真实发生事件越早，越算“排序正确”。0.5≈随机，1.0=完美，**越高越好**。
- **AUROC / AUC**：看模型把“有事件”和“无事件”分开的能力。0.5≈随机，1.0=完美，**越高越好**。
- **Brier**：预测概率与真实结果的平均平方误差。**越低越好**，0 表示完美。
- **Accuracy（准确率）**：整体预测正确的比例。样本不平衡时可能“看起来高但没意义”。
- **Loss**：训练时优化的目标函数值，用于训练过程对比，**越低越好**。

## 4. 训练与评估流程（简要）
1. 读取 manifest 与 XML/CSV 波形，取 8 导联并重采样至 4096。
2. 每导联做标准化（减均值、除标准差）。
3. 训练/验证/测试按 7:1.5:1.5 划分（或 K-fold）。
4. 回归：输出离散时间生存概率；分类：输出 logit 并用 sigmoid 得到事件概率。
5. 评估：
   - 回归：C-index、AUROC、Brier、风险分组等。
   - 分类：AUC、PR-AUC、F1、Recall、Specificity、Brier，并扫描阈值获得 best_f1。

## 5. 损失函数
### 4.1 回归（离散时间生存损失）
- 损失：`SurvLikelihoodLoss`
- 位置：`modules/survival_model/torch_survival/losses.py`
- 说明：复刻 nnet-survival 的离散时间似然，对每个时间区间的生存/失败概率做负对数似然。

### 4.2 分类（BCE）
- 损失：`BCEWithLogitsLoss`
- 说明：对正类引入 `pos_weight`（按负/正样本比并乘倍率）以缓解类别不平衡。

## 6. 最终训练结果（回归 + 分类）

### 6.1 回归结果（离散时间生存）
来源文件：`outputs/survival_logs/survival_eval/metrics.json`
- 训练轮次：Epoch 100/100
- 样本数：1210
- Train训练表现：loss=0.000826，C-index：0.6254，AUROC：0.5098，Brier：0.1526
- Val验证集表现：loss=0.15，C-index：0.5625，AUROC：0.4921，Brier：0.1279

### 6.2 分类结果
来源文件：`outputs/logs/pipeline_20260115_171155.log`
- 训练轮次：Epoch 100/100
- Train：loss=0.0009，AUC=0.5293，Brier=0.1238
- Val：loss=0.2146，AUC=0.4258，Brier=0.1442

## 7. 待确认
- “最后一次回归训练”为 2026-01-15：当前仓库内可直接定位的回归评估文件为
  `outputs/survival_logs/survival_eval/metrics.json`（时间戳 2026-01-27）。
  若需严格使用 2026-01-15 的回归结果，请提供对应日志或服务器路径。
