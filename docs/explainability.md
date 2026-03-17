# Explainability Toolkit

本仓库新增 3 个脚本，覆盖论文中的可解释性环节：

1. **潜在因子 Traversal** – `scripts/vae_latent_traversal.py`  
   ```bash
   python scripts/vae_latent_traversal.py \
     --config modules/vae_model/configs/median.yaml \
     --checkpoint logs/MedianBeatVAE/version_0/checkpoints/last.ckpt \
     --latents-dir logs/MedianBeatVAE/version_0/latents \
     --linear-json logs/MedianBeatVAE/version_0/latents/linear_scores/linear_model.json \
     --output-dir outputs/explainability/latent_traversal \
     --range-mode fixed --fixed-min -5 --fixed-max 5 --fixed-step 1 \
     --base-mode zero --scale-mode absolute
   ```
   - 读取 VAE 配置与 checkpoint，沿着线性模型权重最大的潜在维度，输出 `.npz` 和可视化 PNG。
   - `--range-mode` 在 `std` 与 `fixed` 间切换；`fixed`=补充材料提到的 `[-5,5]`。
   - `--base-mode zero --scale-mode absolute` 可复现文中“其余因子置零、直接设定该因子数值”的 latent traversal。

2. **高/低风险平均波形** – `scripts/vae_waveform_extremes.py`  
   ```bash
   python scripts/vae_waveform_extremes.py \
     --scores-csv logs/.../linear_scores/test_scores.csv \
     --median-dir data/processed/median_beats \
     --output-dir outputs/explainability/waveform_extremes
   ```
   - 根据 `prob`（或 `prediction`）字段挑选前/后 `N` 条样本，加载对应 median-beat CSV，计算 mean±std 并生成 Figure 7 风格的图。

3. **影像相关性** – `scripts/risk_echo_correlation.py`  
   ```bash
   python scripts/risk_echo_correlation.py \
     --scores-csv logs/.../linear_scores/test_scores.csv \
     --score-column prob \
     --echo-csv data/manifests/echo_labels.csv \
     --target-columns LA_volume LVEDd LVEF \
     --covariates age sex \
     --output-dir outputs/explainability/echo_corr
   ```
   - 将风险分数与 echocardiography 指标做 Pearson 相关 + 多元线性回归（可选协变量），输出 CSV/JSON。

> 依赖：`numpy pandas torch yaml matplotlib (可选) scipy (可选)`，均已包含在当前环境。
