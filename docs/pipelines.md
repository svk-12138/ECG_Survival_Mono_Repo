# Pipelines / 管线说明

## 一键脚本 / One-click Scripts
| 平台 | 命令 |
| --- | --- |
| Linux / macOS | `bash scripts/pipeline.sh --config configs/pipeline.default.yaml` |
| Windows PowerShell | `./scripts/pipeline.ps1 -Config configs/pipeline.default.yaml` |

> 可通过 `PYTHON_BIN=python3 bash scripts/pipeline.sh ...` 或 `./scripts/pipeline.ps1 -PythonBin C:\Python311\python.exe` 指定解释器。

### 步骤 / Flow（已对齐论文流程）
1) **VAE 训练**  
   `modules/vae_model/run.py --config <vae-config> --export-latents`  
   输出 `logs/MedianBeatVAE/version_*/`，并导出潜变量 CSV 供后续线性回归。
2) **Pearson 分析**  
   `scripts/vae_latent_pearson.py` 读取 VAE checkpoint，生成 `outputs/analysis/vae_latent/latent_pearson.*`。
3) **生存模型训练**  
   `scripts/run_survival_training.py --xml-dir ... --manifest ...`  
   基于 manifest + XML 训练残差 1D CNN，保存 `outputs/survival_logs/model_final.pt` 与指标曲线。
4) **生存模型推理（风险分数）**  
   `modules/survival_model/torch_survival/infer_survival_risk.py --checkpoint ... --manifest ... --xml-dir ... --output ...`  
   生成 `risk_scores.csv`（作为线性回归标签）。
5) **线性回归 + t 值解释**  
   `modules/vae_model/scripts/train_linear_scores.py --latents-dir ... --labels-csv risk_scores.csv --target-type regression`  
   计算 OLS 回归、标准误与 t 值，按 |t| 排序给出最具影响力的潜在因子。

所有 stdout/stderr 仍写入 `outputs/logs/pipeline_<timestamp>.log`，并额外生成汇总报告 `outputs/pipeline_report.json`。

### 配置 / Config
- 默认配置：`configs/pipeline.default.yaml`
  - `vae`: 脚本、配置、是否导出潜变量（默认 `--export-latents`）。
  - `pearson`: checkpoint、split、输出目录（请将 checkpoint 指向你希望解释的 VAE 版本）。
  - `survival`: `scripts/run_survival_training.py` 覆盖参数（epochs、batch、xml_dir 等），会自动保存 `model_final.pt`。
  - `survival_pred`: 使用生存模型推理生成 `risk_scores.csv`。
  - `linear`: 使用 VAE 潜变量与 `risk_scores.csv` 做回归并输出 t 值排序。
- 如需多套配置，可拷贝该 YAML 并传入 `--config`。

## 自动化测试 / Automated Tests
| 平台 | 命令 |
| --- | --- |
| Linux / macOS | `bash scripts/run_tests.sh [--skip-unit-tests] [--check-data]` |
| Windows | `./scripts/run_tests.ps1 [-PythonBin python] [--skip-unit-tests]` |

功能：
1. 校验目录结构、weights、pipeline 配置。
2. 可选：`--check-data` 确认 `data/manifests/` 存在。
3. 触发 `python -m unittest discover tests`，检查配置与模板。

## 常见命令 / Example CLI
```bash
# 使用默认配置运行整套流程（Linux/macOS）
bash scripts/pipeline.sh

# 仅测试配置（跳过单元测试）
python3 scripts/run_tests.py --skip-unit-tests

# Windows 下指定 Python（注意 PowerShell 用反引号 ` 续行）
./scripts/pipeline.ps1 -Config configs/pipeline.default.yaml -PythonBin C:/Users/me/miniconda3/python.exe

# 自定义 manifest / epochs 训练生存模型
python3 scripts/run_survival_training.py \
    --xml-dir data/raw/xml \
    --manifest data/manifests/survival_manifest.json \
    --epochs 120 --batch 64 --lr 2e-4 \
    --max_time_years 10

# Windows 示例：生存模型推理导出风险分数（PowerShell）
python modules/survival_model/torch_survival/infer_survival_risk.py `
  --checkpoint outputs/survival_logs/model_final.pt `
  --manifest /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/manifest.json `
  --xml-dir /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/XML `
  --output outputs/survival_logs/risk_scores.csv `
  --n-intervals 20 --target-len 4096 --batch 16
```

> 避免 `ParserError`：PowerShell 多行用反引号 `` ` ``，不要用 CMD 的 `^`。
- `--max_time_years` 可直接设定离散生存区间覆盖的年限（补充材料示例=10年）。如未指定则使用 `--max_time`（天）。
