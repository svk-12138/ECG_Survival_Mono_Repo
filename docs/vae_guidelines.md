# VAE 实践补充（基于 FactorECG 论文译文）

结合《利用变分自编码器提高基于深度神经网络的心电图判读可解释性》及补充材料，当前仓库的 VAE 流程建议如下：

1. **数据准备**
   - 使用 GE MUSE 原始电压 → 生成 12 导或 8 导 median beat，10 s × 4096 或 512 点。
   - 训练 VAE 前不需要标签；常规/任务标签仅在后续线性或 XGBoost 模型中使用。

2. **模型与训练**
   - `MedianBeatVAE` 对应 β-VAE 架构：1D 卷积编码器 + 解码器，latent ≤ 32（推荐 30）。
   - 默认重建损失为对称 MAPE；`--median-vae-profile` 会自动限制 latent≤30、β∈{0.1…10}。
   - 训练 50 epoch 左右即可收敛；可使用 `--export-latents` 导出 train/val/test 的潜向量。

3. **Explainable Pipeline**
   - 任务模型使用 FactorECG（latent 均值向量）作为特征：
     * 常规诊断：逐标签 Logistic Regression；
     * EF/死亡率等复杂任务：XGBoost + `SHAP`（后续可使用 `train_linear_scores.py --target-type classification` 快速跑基线）。
   - 可解释性来源：
     * `scripts/vae_latent_traversal.py`：保持其它潜因子为 0，对目标因子按 [-5,5] 可视化；
     * `scripts/vae_waveform_extremes.py`：高/低风险均值 + 标准差；
     * `scripts/risk_echo_correlation.py`：风险分数 vs echocardiography 指标的相关性/回归。

4. **生存模型配合**
   - `train_survival_from_xml.py` 支持 `--max_time_years 10`，对应论文中的 10 年离散时间窗口；
   - VAE 输出可直接作为下游 Cox/生存模型的输入以获得可解释系数。

5. **再现建议**
   - 训练顺序：`run.py --median-vae-profile --export-latents` → `train_linear_scores.py` → `vae_latent_traversal.py` / `vae_waveform_extremes.py`；
   - 若需与 FactorECG 完全一致，可在 traversal 脚本中启用 `--range-mode fixed --base-mode zero --scale-mode absolute`。

> 后续若要加入 XGBoost+SHAP，可在 `latents/*.npz` 读取后使用 `xgboost`/`shap` 包复现论文中的可解释特征重要性。
