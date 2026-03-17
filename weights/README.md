# 权重目录说明 / Weights Directory Guide

```
weights/
├── README.md
├── vae/
│   └── MedianBeatVAE/version_15/last.ckpt
└── survival/
    └── (待同步)
```

## 现有权重 / Archived Checkpoints

| 模块 | 路径 | 来源 | 备注 |
| --- | --- | --- | --- |
| VAE (MedianBeatVAE) | `weights/vae/MedianBeatVAE/version_15/last.ckpt` | `modules/vae_model/logs/MedianBeatVAE/version_15/checkpoints/last.ckpt` | `version_15`，4096 采样长度，Alpha=1.0 |
| VAE (MedianBeatVAE) | `weights/vae/MedianBeatVAE/version_15/epoch=49-step=1950.ckpt` | 同上 | 训练后期节点，可用于快速对比 |
| VAE (BetaVAE) | `weights/vae/BetaVAE/version_11/last.ckpt` | `modules/vae_model/logs/BetaVAE/version_11/checkpoints/last.ckpt` | β-VAE 基线 |

> **提示**：原始 Lightning 日志（TensorBoard、Reconstructions 等）仍保留在 `modules/vae_model/logs/` 供排查使用。

## 生存模型权重 / Survival Checkpoints
- 当前仓库未包含最新的生存模型 `.pt` / `.ckpt` 文件。请在服务器训练完成后，将导出的权重复制至 `weights/survival/<model_name>/` 并在本文件更新记录。
- 推荐命名方式：`weights/survival/resnet1d/<YYYYMMDD_HHMM>/best.pt`。

## 同步策略 / Sync Guidelines
1. **复制**（而非移动）: 先将服务器上的 `.ckpt` / `.pt` 复制进 `weights/`，保留原始日志目录。
2. **更新表格**: 在上表新增行，描述训练参数（latent 维度、epochs、AUC/F1 等）。
3. **README 说明**: 若权重依赖额外配置，请在 `docs/pipelines.md` 添加“如何加载该权重”的说明。
4. **流水线引用**: 在 `configs/pipeline.default.yaml` 中，将 `pearson.checkpoint` 与 `survival.resume` 指向对应文件，确保一键脚本使用最新权重。
