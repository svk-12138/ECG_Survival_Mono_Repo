# Median CSV Dataset Quickstart

本指南介绍如何将 Braveheart 导出的 median `.mat` 信号用于 `run.py` 训练。

## 1. 准备数据
1. 使用项目内的 `tools/batch_export_medians.py`（内部调用 `tools/export_median_mat_to_csv.py`）把所有 median `.mat` 转成 `.csv`：
   ```
   python tools/batch_export_medians.py ^
     --input-dir "<原始mat目录>" ^
     --output-dir data/median_csv --overwrite
   ```
2. 确保每个 CSV 里包含表头 `idx,I,II,...,V6`，默认输出目录为 `data/median_csv`。

## 2. 选择配置
项目新增了 `configs/median_vae.yaml`，关键字段如下：

| 字段 | 说明 |
| --- | --- |
| `model_params` | 1D VAE 超参，默认 `MedianBeatVAE`（1D Conv encoder/decoder，latent≤30）。 |
| `data_params.dataset_type` | 设为 `median_csv` 以启用新数据管道。 |
| `data_params.data_path` | 指向 CSV 所在目录。 |
| `max_length` | 如果不同 CSV 序列长度不一致，用于截断或补零（例如 512 点）。 |
| `representation` | 设为 `waveform` 时返回 12×时间的波形张量，适配 1D Conv VAE；`image` 模式则保持旧的 2D 图像。 |
| `normalize` | `zscore`、`minmax` 或 `none`。 |
| `val_fraction` | 验证集比例，默认 0.2。 |
| `num_workers` | Windows 下建议设为 `0`（避免多进程 spawn 报错）。 |

可根据需要调整 `train_batch_size`、`gpus` 等。

## 3. 启动训练
```
cd /home/admin123/use/Program/ECG_Survival_Mono_Repo/modules/vae_model
python run.py --config configs/median_vae.yaml --median-vae-profile
```

`--median-vae-profile` 会自动应用论文中的 latent/beta/epoch 限制。如果要运行自己的配置，也可以去掉该参数。

## 4. 常见问题
- **找不到 CSV**：确认 `data_params.data_path` 指向的目录存在，并且包含匹配 `*.csv` 的文件。
- **导联缺失**：`MedianBeatCSVDataset` 会检查每个 CSV 的导联表头，请确保与导出脚本一致。
- **长度不一致**：调整 `max_length` 或移除该字段改为原样长度（需自定义 collate_fn）。
- **Windows 多进程报错**：把 `num_workers` 设为 `0`。
- **输入/输出通道不匹配**：在 `model_params` 中添加 `out_channels`（默认等于 `in_channels`）。

## 5. 推理与可视化
1. 查找训练日志目录下的 checkpoint，例如 `logs/BetaVAE/version_11/checkpoints/last.ckpt`（具体名称视实际 run 而定）。
2. 准备任意一个 median CSV（可直接取 `data/median_csv/*.csv`）。
3. 运行：
   ```
   python tools/infer_median_csv.py ^
     --config configs/median_vae.yaml ^
     --checkpoint logs/MedianVAE/MedianVAE/version_10/checkpoints/last.ckpt ^
     --input-csv data/median_csv/1.653896_medians.csv ^
     --output-dir results/inference
   ```
   如有 GPU，可附加 `--device cuda`。
4. 脚本会生成输入/重构的 12 导波形对比 PNG 以及对应的 JSON 元信息，便于快速核对模型表现。***

## 6. 导出拉丁特征
如果需要按照论文做 latent 特征解析，可以用 `tools/extract_latent_features.py` 批量导出中位心搏的潜在向量以及对应的重构 sMAPE：
```
python tools/extract_latent_features.py ^
  --config configs/median_vae.yaml ^
  --checkpoint logs/MedianBeatVAE/version_2/checkpoints/last.ckpt ^
  --pattern "*.csv" ^
  --output results/latent_features.csv
```
输出 CSV 包括原始文件路径、每个样本的 sMAPE，以及 `latent_1 ... latent_n`。这份表可以用于筛选最具代表性的 latent 维度或与下游模型结合。***
