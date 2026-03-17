# Project Overview / 项目概览

## 目标 / Goals
1. 统一 VAE 与生存模型至单一 mono repo，减少多仓同步成本。
2. 提供跨平台一键脚本（pipeline/test），包含中文指引。
3. 整理数据、权重、日志目录，便于审计与复现。

## 模块 / Modules
- **modules/vae_model**：Median-beat VAE、Beta-VAE 等结构，沿用 PyTorch Lightning。
- **modules/survival_model**：残差 1D CNN + 生存概率分段模型，含 Optuna 搜索脚本。
- **scripts/**：上层封装，包括 `pipeline`, `run_survival_training`, `run_tests` 等。

## 目录速览 / Directory Snapshot
```
configs/                 # pipeline 默认配置、manifest 模板
data/                    # raw/interim/processed/manifests
docs/                    # 说明文档
modules/                 # VAE & Survival 子模块
outputs/analysis         # Pearson & 指标
outputs/logs             # pipeline 日志
scripts/                 # bash/pwsh/python 脚本
tests/                   # unittest 集
weights/                 # 统一 checkpoint
```

## 环境 / Environment
- Python 3.10 / Conda 虚拟环境。
- PyTorch >= 2.1（CUDA 11.8 建议）。
- 其它依赖：NumPy、Pandas、scikit-learn、PyYAML、Matplotlib、Optuna。

## 工作流 / Workflow
1. **数据准备**：`data/raw/` + manifest（详见 `data/README.md`）。
2. **VAE 训练**：`modules/vae_model/run.py`，由 `scripts/run_pipeline.py` 调度。
3. **相关性评估**：`scripts/vae_latent_pearson.py` 输出到 `outputs/analysis/vae_latent/`。
4. **生存训练**：`scripts/run_survival_training.py` 调 `TrainConfig`。
5. **测试与报告**：`scripts/run_tests.py` + `docs/pipelines.md` 中的命令记录。

## 里程碑 / Roadmap
- ✅ 模块迁移与目录草图（integration_plan.md）。
- ✅ 数据/权重统一入口、pipeline/test 脚本。
- ⏳ Manifest/Optuna 输出的标准化（Phase 3）。
- ⏳ 发布完整复现文档（Phase 4）。
