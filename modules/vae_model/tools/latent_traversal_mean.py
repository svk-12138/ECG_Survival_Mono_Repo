#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute mean reconstructions over a latent traversal and save multi-lead plots.
The traversal is averaged across all provided latent npz files (e.g., latents_survival train/val/test).

Example:
  python modules/vae_model/tools/latent_traversal_mean.py \
    --config modules/vae_model/configs/median_vae.yaml \
    --checkpoint logs/MedianBeatVAE/version_9/checkpoints/last.ckpt \
    --latents-dir logs/MedianBeatVAE/version_9/latents_survival \
    --dims 11 14 20 \
    --steps -5 5 11 \
    --xmax 400 \
    --output-dir logs/MedianBeatVAE/version_9/latents_survival
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import VAEXperiment  # noqa: E402
from models import vae_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean latent traversal over saved latents.")
    parser.add_argument("--config", type=Path, required=True, help="VAE config YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VAE checkpoint path.")
    parser.add_argument("--latents-dir", type=Path, required=True, help="Directory containing *_latents.npz.")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        required=True,
        help="Latent dimensions to traverse, e.g. 11 14 20.",
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs=3,
        metavar=("START", "END", "COUNT"),
        default=(-5.0, 5.0, 11),
        help="Traversal range as start end count (linspace).",
    )
    parser.add_argument("--xmax", type=int, default=400, help="Max sample index to display on x-axis.")
    parser.add_argument("--downsample", type=int, default=800, help="Target number of points after downsample.")
    parser.add_argument("--batch-size", type=int, default=64, help="Decode batch size.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save plots (filename: latent{dim}_multilead_mean.png).",
    )
    return parser.parse_args()


def load_latents(latents_dir: Path) -> np.ndarray:
    lat_list: List[np.ndarray] = []
    for name in ["train_latents.npz", "val_latents.npz", "test_latents.npz"]:
        path = latents_dir / name
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        lat_list.append(data["latents"])
    if not lat_list:
        raise FileNotFoundError(f"No latents found under {latents_dir}")
    return np.concatenate(lat_list, axis=0)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cpu")
    model = vae_models[cfg["model_params"]["name"]](**cfg["model_params"])
    exp = VAEXperiment.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        vae_model=model,
        params=cfg["exp_params"],
        map_location=device,
    )
    exp.eval()
    exp.to(device)

    latents = load_latents(args.latents_dir)
    print(f"Loaded latents: {latents.shape}")

    lead_labels = cfg["data_params"].get("lead_order") or [f"lead{i+1}" for i in range(model.output_channels)]
    steps = np.linspace(args.steps[0], args.steps[1], int(args.steps[2]))
    batch_size = args.batch_size

    for dim in args.dims:
        mean_recons = {}
        for step in steps:
            z = torch.tensor(latents, dtype=torch.float32)
            z[:, dim] += step
            sums = None
            count = 0
            with torch.no_grad():
                for i in range(0, z.shape[0], batch_size):
                    zb = z[i : i + batch_size].to(device)
                    rec = exp.model.decode(zb).cpu().numpy()  # (b,C,L)
                    if sums is None:
                        sums = np.zeros_like(rec[0:1])
                    sums += rec.sum(axis=0, keepdims=True)
                    count += rec.shape[0]
            mean_recons[step] = (sums / count)[0]
            print(f"[dim {dim}] step {step:+.1f} done, count {count}")

        C, L = mean_recons[steps[0]].shape
        step_ds = max(1, L // args.downsample)
        x_full = np.arange(0, L, step_ds)
        mask = x_full <= args.xmax
        x = x_full[mask]

        cols = 4
        rows = int(np.ceil(C / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        for li in range(C):
            r, c = divmod(li, cols)
            ax = axes[r, c]
            lead_name = lead_labels[li] if li < len(lead_labels) else f"lead{li+1}"
            for step in steps:
                rec = mean_recons[step][li]
                ax.plot(x, rec[::step_ds][: len(x)], label=f"{step:+.1f}", lw=0.9)
            ax.set_xlim(0, args.xmax)
            ax.set_title(lead_name)
            ax.set_xlabel("sample index")
            ax.set_ylabel("amplitude")
            ax.legend(fontsize=6, ncol=3)
        for k in range(C, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")
        fig.suptitle(f"Latent {dim} traversal | mean over latents (x<={args.xmax})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        outfile = args.output_dir / f"latent{dim}_multilead_mean.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
