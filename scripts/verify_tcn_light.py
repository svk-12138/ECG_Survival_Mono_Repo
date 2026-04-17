#!/usr/bin/env python3
"""快速验证脚本：检查 TCN 轻量版模型是否能正常构建和前向传播

运行方式：
  python3 scripts/verify_tcn_light.py

预期输出：
  ✓ TCN轻量版构建成功，参数量: 25,xxx
  ✓ 前向传播正常: (4, 20)
  ✓ 损失计算正常
  ✓ 反向传播正常
  全部通过，可以开始训练
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules" / "survival_model"))

import torch
import numpy as np

from torch_survival.model_builder import build_survival_tcn_light, build_survival_resnet
from torch_survival.losses import SurvLikelihoodLoss

N_INTERVALS = 20
BATCH = 4
IN_CHANNELS = 8
SEQ_LEN = 4096

print("=" * 50)
print("TCN 轻量版验证")
print("=" * 50)

# 1. 构建模型
model_tcn = build_survival_tcn_light(N_INTERVALS, input_dim=(IN_CHANNELS, SEQ_LEN))
n_params_tcn = sum(p.numel() for p in model_tcn.parameters())
print(f"✓ TCN轻量版构建成功，参数量: {n_params_tcn:,}")

model_resnet = build_survival_resnet(N_INTERVALS, input_dim=(IN_CHANNELS, SEQ_LEN))
n_params_resnet = sum(p.numel() for p in model_resnet.parameters())
print(f"  ResNet1d参数量（对比）: {n_params_resnet:,}")
print(f"  参数量比: ResNet1d / TCN = {n_params_resnet / n_params_tcn:.1f}x")

# 2. 前向传播
x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN)
out = model_tcn(x)
assert out.shape == (BATCH, N_INTERVALS), f"输出形状错误: {out.shape}"
assert not torch.isnan(out).any(), "输出包含 NaN"
print(f"✓ 前向传播正常: 输入{list(x.shape)} → 输出{list(out.shape)}")

# 3. 损失计算（模拟生存分析标签）
loss_fn = SurvLikelihoodLoss(N_INTERVALS)
# 构造假标签：(batch, 2*n_intervals)
y_true = torch.zeros(BATCH, 2 * N_INTERVALS)
for i in range(BATCH):
    t = np.random.randint(0, N_INTERVALS)
    e = np.random.randint(0, 2)
    y_true[i, :t+1] = 1.0          # surv_part
    if e == 1:
        y_true[i, N_INTERVALS + t] = 1.0  # fail_part

probs = torch.sigmoid(out)
loss = loss_fn(probs, y_true)
assert not torch.isnan(loss), f"损失为 NaN"
print(f"✓ 损失计算正常: loss={loss.item():.4f}")

# 4. 反向传播
loss.backward()
grad_norms = [p.grad.norm().item() for p in model_tcn.parameters() if p.grad is not None]
assert len(grad_norms) > 0, "没有梯度"
assert all(not np.isnan(g) for g in grad_norms), "梯度包含 NaN"
print(f"✓ 反向传播正常: {len(grad_norms)} 个参数有梯度")

print()
print("全部通过，可以开始训练")
print()
print("切换到 TCN 轻量版训练：")
print("  在 configs/train_stroke_thesis.env 中设置：")
print("    MODEL_TYPE=tcn_light")
print("  然后运行：")
print("    bash scripts/train_stroke_thesis.sh   # Linux/macOS")
print("    scripts\\train_stroke_thesis.bat       # Windows")
