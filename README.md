# ECG Survival Mono Repo / 心电图生存分析一体化仓库

本项目整合 ECG 风险建模与相关分析脚本，可作为毕业论文或方法学复现的基础框架。

## 论文版从这里开始

如果你现在做的是“1200 条卒中 ECG + time + event”的毕业论文，默认按 Win11 使用方式来操作。

Win11 医生用户请直接看这 3 个文件：

- [scripts/train_stroke_thesis.ps1](scripts/train_stroke_thesis.ps1)
- [scripts/train_stroke_thesis.bat](scripts/train_stroke_thesis.bat)
- [docs/stroke_survival_thesis_framework.md](docs/stroke_survival_thesis_framework.md)

如果医生需要先把原始标签表处理成训练 manifest，再看：

- [docs/survival_data_preparation_workflow.md](docs/survival_data_preparation_workflow.md)
- [scripts/prepare_survival_dataset.py](scripts/prepare_survival_dataset.py)

最简单的启动方式：

```bat
scripts\train_stroke_thesis.bat
```

你只需要做两件事：

1. 打开 `scripts/train_stroke_thesis.ps1`
2. 修改脚本顶部参数后运行 `scripts\train_stroke_thesis.bat`

路径既支持绝对路径，也支持相对仓库根目录的路径；如果路径里有空格或中文，保留双引号即可。
默认留出法比例已经改成 train=0.8、val=0.2、test=0.0；如果把 `cv_folds` 改成大于 1，就会切换回交叉验证，此时比例参数会被忽略。

如果你是 Linux / macOS / WSL 用户，再使用：

- [scripts/train_stroke_thesis.sh](scripts/train_stroke_thesis.sh)
- `bash scripts/train_stroke_thesis.sh`

## 论文版训练目前支持什么

- `prediction / classification`
- `8lead / 12lead`
- XML 或 CSV 输入
- 小样本交叉验证
- ECG 滤波与基础预处理

## 环境依赖

- Python 3.10+
- PyTorch >= 2.1
- NumPy / Pandas / SciPy / matplotlib

建议环境：

```bash
conda create -n ecg-pipeline python=3.10 -y
conda activate ecg-pipeline
pip install -r modules/survival_model/requirements.txt
```

## 如果你要跑整条项目 pipeline

整条 pipeline 仍然保留，适合项目维护者，不是论文主入口。

```bash
python3 scripts/run_pipeline.py --list-stages
python3 scripts/run_pipeline.py --stages survival
```

## 其他文档

- [docs/stroke_survival_thesis_framework.md](docs/stroke_survival_thesis_framework.md)
- [docs/survival_data_preparation_workflow.md](docs/survival_data_preparation_workflow.md)
- [docs/paper_reproduction_gap_analysis_20260317.md](docs/paper_reproduction_gap_analysis_20260317.md)
- [docs/pipelines.md](docs/pipelines.md)
