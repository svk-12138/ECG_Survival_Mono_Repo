# -*- coding: utf-8 -*-
"""PyTorch 版本 ECG 年龄预测示例训练脚本。
- 复用 ecg-age-prediction-main 的 ResNet1d 实现。
- 这里使用随机数据/标签做 dry-run，真实训练时替换为实际 dataloader 即可。
"""
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model_builder import build_resnet_ecg_model

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ECG 年龄预测 demo")
    parser.add_argument("--n_classes", type=int, default=1, help="输出维度，年龄回归默认为 1")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="演示训练轮次")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--inspect", action="store_true", help="仅打印网络结构")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet_ecg_model(n_classes=args.n_classes)
    model.to(device)

    if args.inspect:
        print(model)
        return

    # 随机生成假数据，形状 (batch, channels=12, length=4096)
    x = torch.randn(args.batch * 2, 12, 4096)
    y = torch.randn(args.batch * 2, args.n_classes)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True)

    criterion = torch.nn.MSELoss()
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
