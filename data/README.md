# 数据目录说明 / Data Directory Guide

```
data/
├── raw/            # 原始 ECG XML、CSV、采集记录
├── interim/        # 预处理产物（median beat、去噪后的 numpy/tensor）
├── processed/      # 生存模型可直接加载的张量或特征
├── manifests/      # manifest.jsonl / split.json 等索引文件
└── README.md       # 当前文件
```

## 原始数据 / Raw Assets
- **ECG XML**：请放入 `data/raw/xml/`（可自建子目录），命名保持 `PatientID_timestamp.xml`。  
  - 现有生产数据实际存放在：`/home/admin123/use/Program/ECG_Survival_Mono_Repo/data/XML/`（请按需复制/挂载到 `data/raw/xml/` 或在配置中指向该路径）。
- **标签/存活信息**：推荐以 CSV 或 JSONL 放在 `data/raw/labels/`，字段建议包含：
  - `patient_id`
  - `time_to_event`（单位秒或天，请与配置 `DEFAULT_MAX_TIME` 对齐）
  - `event`（0/1 是否发生事件）
  - 若使用中位心搏特征，当前存放在：`/home/admin123/use/Program/ECG_Survival_Mono_Repo/data/median_beats/*.csv`，可复制到 `data/raw/labels/` 或在 manifest 中写全路径。

## Manifest 格式 / Manifest Format
- 文件路径建议 `data/manifests/survival_manifest.json`。
- 内容为 **列表**，每个元素示例：

```json
[
  {
    "patient_id": "123456",
    "xml_path": "data/raw/xml/123456_20220101.xml",
    "time_to_event": 360,
    "event": 1,
    "meta": {"age": 67, "sex": "M"}
  }
]
```

- 若需要多个拆分，可在 `manifests/` 下创建 `train.json`, `val.json`, `test.json`，或者在主 manifest 中添加 `split` 字段。
- 模板可参考 `configs/data_manifest.template.json`。

## 中间数据 / Interim Artifacts
- `data/interim/median_beats/`：VAE 重采样后的波形（`.npy` / `.pt`）。
- `data/interim/vae_latents/`：VAE 缩放后的潜变量；`scripts/vae_latent_pearson.py` 默认输入为配置中的 split 数据集，因此无需单独保存，但若希望缓存，则可在此目录下保存 `*.npy`。

## 处理数据 / Processed Data
- `data/processed/survival_tensors/` 可存储打包后的 `(signal, label)`；`modules/survival_model/torch_survival/train_survival_from_json.py` 仍支持直接从 manifest + XML 读取，如需缩短训练时间，可将处理后的 tensor 路径写入 manifest。

## 数据登记 / Registration Checklist
1. 将原始 ECG/XML 拷贝至 `data/raw/`，保持唯一 `patient_id`。
2. 根据 `configs/data_manifest.template.json` 生成 manifest，保存到 `data/manifests/`。
3. 更新 `configs/pipeline.default.yaml` 中的 `survival.manifest` 与 `survival.xml_dir`（若与默认不同）。
4. `scripts/run_tests.py --check-data` 可验证 manifest 中的路径是否存在。

## 脱敏 / Privacy
- 所有 `patient_id` 建议在导入前完成脱敏并保留映射表于安全环境。
- Manifest 中若包含额外临床指标，请确保符合所在机构的数据规范。
