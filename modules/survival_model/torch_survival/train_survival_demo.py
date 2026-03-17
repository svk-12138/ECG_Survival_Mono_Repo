# -*- coding: utf-8 -*-
"""PyTorch 版本 ECG 生存分析示例脚本（dry-run）。"""
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_survival.data_utils import SurvivalBreaks, demo_fake_targets
from .model_builder import build_survival_resnet
from .losses import SurvLikelihoodLoss

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ECG 生存分析 demo")
    parser.add_argument("--n_intervals", type=int, default=8, help="离散时间区间数")
    parser.add_argument("--max_time", type=float, default=365.0, help="最大随访时间")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="演示轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--inspect", action="store_true", help="仅打印网络结构")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breaks = SurvivalBreaks.from_uniform(args.max_time, args.n_intervals)
    model = build_survival_resnet(args.n_intervals).to(device)

    if args.inspect:
        print(model)
        return

    x = torch.randn(args.batch * 2, 12, 4096)
    y = demo_fake_targets(args.batch * 2, breaks).astype("float32")
    y = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True)

    criterion = SurvLikelihoodLoss(args.n_intervals)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

if __name__ == "__main__":
    main()
