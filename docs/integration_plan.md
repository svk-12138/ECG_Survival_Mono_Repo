# Integration Plan / 整合计划

## 当前要求 / Requirements
1. **Mono repo，多模块**：VAE 与生存分析共用一套仓库与脚本。
2. **保持当前 git 版本**：只在现有工作区上调整结构，不新增外部提交。
3. **双语文档**：`README`、`docs/*`、`scripts/*` 必须包含中文说明。
4. **统一数据与权重**：所有原始/处理数据进入 `data/`，训练 checkpoint 进入 `weights/` 并记录来源。
5. **一键化体验**：提供 `pipeline.(sh|ps1)` 与自动化测试入口，流程与命令需写入文档。

## 阶段划分 / Phase Breakdown
- **Phase 1（完成）**：迁入 `modules/survival_model` 与 `modules/vae_model`，初步清理依赖。
- **Phase 2（进行中）**：编排一键 pipeline、Pearson 分析脚本、测试脚本与多语言 README。
- **Phase 3（待办）**：整理数据/权重 manifest、统一配置入口、补充自动化校验。
- **Phase 4（待办）**：端到端回归测试、发布同步指令、交付复现报告。

## 目标目录结构草图 / Target Directory Layout
```
ECG-PIPELINE/
├── README.md
├── configs/
│   ├── pipeline.default.yaml        # 一键流程配置（VAE/生存模型路径、超参）
│   └── data_manifest.template.json  # 数据/权重登记模板
├── data/
│   ├── README.md                    # 数据脱敏、命名规范、拆分策略
│   ├── raw/                         # 原始 XML/CSV/波形
│   ├── interim/                     # 重采样、median-beat、VAE 输入
│   ├── processed/                   # 生存模型 tensors、特征
│   └── manifests/                   # manifest.jsonl、latents_meta.json
├── docs/
│   ├── overview.md
│   ├── pipelines.md
│   ├── data_guide.md
│   └── integration_plan.md
├── modules/
│   ├── survival_model/              # 旧 ecg-survival-project
│   └── vae_model/                   # 旧 PyTorch-VAE
├── outputs/
│   ├── analysis/                    # Pearson 结果、指标 CSV
│   └── logs/                        # pipeline 执行日志
├── scripts/
│   ├── pipeline.sh / pipeline.ps1   # VAE → Pearson → Survival
│   ├── run_tests.sh / run_tests.ps1 # 自动化测试/自检
│   └── utils/                       # 公共 shell/python 辅助函数
├── tests/
│   ├── test_data_manifest.py
│   ├── test_pipeline_config.py
│   └── __init__.py
├── weights/
│   ├── README.md                    # 权重来源、日期、依赖
│   ├── vae/MedianBeatVAE/*.ckpt
│   └── survival/resnet1d/*.pt
└── pipeline.(sh|ps1) → scripts/pipeline.sh # 便于调用的软链或拷贝
```

## 阻塞项 / Open Issues
1. **Pipeline 配置 Schema**：需要决定如何映射至 PyTorch Lightning 与 Optuna 参数。
2. **依赖共用策略**：两个模块共存时如何避免版本冲突，需要 `requirements` 合并或 lock。
3. **测试覆盖面**：自动化脚本需判定数据/权重存在性、配置字段合法性，并能局部运行模型前向。

## 下一步 / Next Steps
1. 按照上图创建缺失的 `configs/`、`data/README.md`、`weights/README.md` 等骨架文件。
2. 重写 `scripts/pipeline.sh`/`.ps1`，支持读取 `configs/pipeline.default.yaml` 并串联任务。
3. 迁移历史 checkpoint 至 `weights/`，在文档中登记。
4. 在 `docs/pipelines.md` 与 `README.md` 中同步更新命令示例与多平台写法。
