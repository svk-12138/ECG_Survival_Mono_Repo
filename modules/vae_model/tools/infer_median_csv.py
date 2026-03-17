#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load a trained checkpoint, run inference on a single median CSV, and visualize reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import MedianBeatCSVDataset  # noqa: E402
from experiment import VAEXperiment  # noqa: E402
from models import vae_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Median CSV inference + visualization")
    parser.add_argument("--config", default="configs/median_vae.yaml", type=Path, help="Training config YAML.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint (.ckpt) produced by Trainer.")
    parser.add_argument("--input-csv", required=True, type=Path, help="Median 12-lead CSV to analyze.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/inference"), help="Directory to store outputs."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' picks CUDA if available.",
    )
    return parser.parse_args()


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sample_tensor(csv_path: Path, data_params: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    dataset = MedianBeatCSVDataset(
        [csv_path],
        lead_order=data_params.get("lead_order"),
        max_length=data_params.get("max_length"),
        pad_value=float(data_params.get("pad_value", 0.0)),
        normalize=data_params.get("normalize", "zscore"),
        target_hw=data_params.get("target_hw"),
        representation=data_params.get("representation", "image"),
        return_meta=True,
    )
    tensor, _, meta = dataset[0]
    return tensor.unsqueeze(0).to(device), meta  # (1, C, H, W)


def load_experiment(config: dict, checkpoint: Path, device: torch.device) -> VAEXperiment:
    model = vae_models[config["model_params"]["name"]](**config["model_params"])
    experiment = VAEXperiment.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        vae_model=model,
        params=config["exp_params"],
        map_location=device,
    )
    experiment.eval()
    experiment.to(device)
    return experiment


def reconstruct_waveforms(
    recon: torch.Tensor, meta: dict
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    raw = meta["raw"]
    leads, length = raw.shape
    recon_tensor = recon.squeeze(0)
    if recon_tensor.shape[-1] != length:
        recon_tensor = F.interpolate(
            recon_tensor.unsqueeze(0),
            size=length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    recon_norm = recon_tensor.cpu().numpy()
    stats = meta.get("stats") or {}
    if "mean" in stats and "std" in stats:
        recon_wave = recon_norm * stats["std"] + stats["mean"]
    elif "min" in stats and "max" in stats:
        recon_wave = recon_norm * (stats["max"] - stats["min"]) + stats["min"]
    else:
        recon_wave = recon_norm
    lead_names = meta.get("lead_names") or [f"Lead {i+1}" for i in range(leads)]
    return raw, recon_wave, lead_names


def plot_waveforms(
    original: np.ndarray,
    reconstructed: np.ndarray,
    lead_names: List[str],
    out_path: Path,
) -> None:
    leads = original.shape[0]
    cols = 4
    rows = math.ceil(leads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2))
    axes = axes.flatten()
    for idx in range(rows * cols):
        ax = axes[idx]
        if idx >= leads:
            ax.axis("off")
            continue
        name = lead_names[idx] if idx < len(lead_names) else f"Lead {idx+1}"
        ax.plot(original[idx], label="Input", linewidth=1.0)
        ax.plot(reconstructed[idx], label="Recon", linewidth=1.0, alpha=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = args.device
    device = torch.device(dev_str)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment(config, args.checkpoint, device)
    sample, meta = build_sample_tensor(args.input_csv, config["data_params"], device)

    with torch.no_grad():
        recon, *_ = experiment.model(sample)

    orig_wave, recon_wave, lead_names = reconstruct_waveforms(recon, meta)
    vis_path = args.output_dir / f"{args.input_csv.stem}_waveforms.png"
    plot_waveforms(orig_wave, recon_wave, lead_names, vis_path)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "input_csv": str(args.input_csv),
        "device": str(device),
        "output_png": str(vis_path),
    }
    with (args.output_dir / f"{args.input_csv.stem}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved visualization to {vis_path}")


if __name__ == "__main__":
    main()
