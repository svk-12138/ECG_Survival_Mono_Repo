#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the Beta-VAE model from PyTorch-VAE on CIFAR-10 (resized to 64x64)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_VAE_ROOT = PROJECT_ROOT.parent / "PyTorch-VAE"
if str(PYTORCH_VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_VAE_ROOT))

from models.beta_vae import BetaVAE  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Beta-VAE on CIFAR-10 (64x64).")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "cifar10")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "runs" / "beta_vae")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--loss_type", type=str, choices=["B", "H"], default="B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10, help="Checkpoint frequency (epochs).")
    return parser.parse_args()


def make_dataloaders(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader


def train_one_epoch(
    model: BetaVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset_len: int,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recons, inputs, mu, log_var = model(data)
        m_n = data.size(0) / dataset_len
        loss_dict = model.loss_function(recons, inputs, mu, log_var, M_N=m_n)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        running_recon += loss_dict["Reconstruction_Loss"].item() * data.size(0)
        running_kld += loss_dict["KLD"].item() * data.size(0)

    total = dataset_len
    return running_loss / total, running_recon / total, running_kld / total


def save_samples(model: BetaVAE, device: torch.device, output_dir: Path, epoch: int, num_samples: int = 16) -> None:
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device)
        samples = samples.mul(0.5).add(0.5).clamp(0, 1)
        grid = utils.make_grid(samples, nrow=4)
        output_dir.mkdir(parents=True, exist_ok=True)
        utils.save_image(grid, output_dir / f"samples_epoch_{epoch:03d}.png")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loader = make_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = BetaVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        beta=args.beta,
        gamma=args.gamma,
        loss_type=args.loss_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "beta_vae.pt"

    for epoch in range(1, args.epochs + 1):
        loss, recon, kld = train_one_epoch(model, loader, optimizer, device, len(loader.dataset))
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} recon={recon:.4f} kld={kld:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": vars(args),
                },
                ckpt_path,
            )
            save_samples(model, device, args.output_dir, epoch)


if __name__ == "__main__":
    main()
