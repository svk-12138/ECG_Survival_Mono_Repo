# ECG Survival Project

集成三套 ECG 变体：
1. **survival-tf**：TensorFlow + nnet-survival 离散时间生存分析（`ecg_survival/`，`train_demo.py`）。
2. **survival-torch**：PyTorch 生存分析，提供两种数据入口：
   - `train_survival_from_xml.py`：直读 CSV/Excel manifest（含文件名或 PatientID）
   - `train_survival_from_json.py`：使用 JSON manifest（仅需 `patient_id/time/event` 三列）
3. **age-torch**：PyTorch 年龄预测（`torch_age/`）。

## 依赖
```powershell
cd /home/admin123/use/Program/ECG_Survival_Mono_Repo
pip install -r requirements.txt
```

## 将 CSV/Excel manifest 转换为 JSON（仅保留所需列）
```powershell
python scripts\csv_to_json_manifest.py \
  --csv /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/manifest.csv \
  --out /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/manifest.json \
  --encoding utf-8 \
  --patient_field PatientID \
  --time_field time \
  --event_field end
```

## PyTorch 生存分析（JSON 清爽入口）
```powershell
python -m torch_survival.train_survival_from_json \
  --xml_dir /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/XML \
  --manifest /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/manifest.json \
  --n_intervals 8 \
  --max_time 365 \
  --target_len 4096 \
  --batch 8 \
  --epochs 10 \
  --lr 1e-3 \
  --num_workers 4
```
- JSON 中每条记录需包含：`{"patient_id": "653896", "time": 842, "event": 0}`。
- 仅保留 8 个物理导联（I, II, V1–V6），波形自动解码、重采样、标准化为 `(8, 4096)` 多通道序列。

## 其它脚本
- `python -m torch_survival.train_survival_demo --inspect`：随机数据 dry-run。
- `python -m torch_survival.train_survival_from_xml ...`：仍可使用 CSV/Excel manifest 与文件名/PatientID 匹配。
- `python -m torch_age.train_age_demo --inspect`：PyTorch 年龄预测 demo。
- `python train_demo.py --inspect`：TensorFlow 生存版 demo。

## 工作流建议
1. 使用 `scripts/csv_to_json_manifest.py` 从原始标签表提取 `PatientID/time/end` → `manifest.json`
2. 运行 `train_survival_from_json.py` 训练 PyTorch 生存模型（参数精简）。
3. 如需 CSV/Excel 原格式，可继续用 `train_survival_from_xml.py`。
- 训练脚本默认使用 tqdm 展示批次进度与 loss（安装依赖时已包含 tqdm）。
