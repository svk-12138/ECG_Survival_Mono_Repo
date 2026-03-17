# Data & Weights Guide / 数据与权重指南

## 数据目录 / Data Layout
```
data/
├── raw/          # 原始 XML、CSV、标签
├── interim/      # 预处理中间结果（median beat、降噪波形）
├── processed/    # 生存模型张量、特征
└── manifests/    # manifest.json(l) / split.json
```

详细规范见 `data/README.md`。建议：
- `data/raw/xml/` 保存脱敏后的 ECG XML，文件名包含 `patient_id`。
- `data/raw/labels/` 保存随访/事件 CSV。
- `data/manifests/` 使用 JSON/JSONL 描述样本，字段至少包括 `patient_id/xml_path/time_to_event/event`。

## Manifest 生成 / Manifest Creation
1. 使用 `configs/data_manifest.template.json` 作为示例。
2. 将路径写成相对仓库根目录的形式，便于跨平台。
3. 如需训练/验证/测试拆分，可：
   - 为每条记录添加 `split` 字段；
   - 或在 `manifests/` 下分别创建 `train.json`, `val.json`, `test.json`。
4. 使用 `scripts/run_tests.py --check-data` 快速验证路径是否存在。

## 权重目录 / Weights Layout
```
weights/
├── vae/MedianBeatVAE/version_15/last.ckpt
├── vae/BetaVAE/version_11/last.ckpt
└── survival/...(待添加)
```

说明：
- 所有 Lightning/PyTorch 权重集中保存在此，`weights/README.md` 记录来源与训练指标。
- VAE 日志仍位于 `modules/vae_model/logs/`，如需裁剪可仅复制 `.ckpt`。
- 生存模型训练后将 `.pt` 或 `.ckpt` 拷贝至 `weights/survival/<model>/<timestamp>/` 并更新 README。

## 日志与输出 / Logs & Outputs
- `outputs/logs/`: `scripts/run_pipeline.py` 生成的执行日志。
- `outputs/analysis/vae_latent/`: `scripts/vae_latent_pearson.py` 输出的皮尔逊矩阵。
- `outputs/survival_logs/`: 默认的生存模型训练日志（可在 `configs/pipeline.default.yaml` 中调整）。

## 脱敏与安全 / Privacy
- 在导入前完成 `patient_id` 脱敏，对应表保存在安全环境（不入库）。
- manifest 中若存在 `meta` 字段，仅存储分析所需的信息（年龄段、性别、诊断等），避免直接可识别字段。
- 若需上传数据至服务器，请使用压缩+校验流程并记录在 `data/README.md`。
