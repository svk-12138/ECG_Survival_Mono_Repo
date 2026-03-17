#!/usr/bin/env python3
"""
Visualize latent traversal by decoding values from -5 to 5 for each factor.

Example:
  python scripts/visualize_latent_traversal.py \
      --config configs/median_vae.yaml \
      --checkpoint logs/MedianBeatVAE/version_x/checkpoints/last.ckpt \
      --output plots/latent_traversal
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import sys
import types
ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 兼容模型文件里的 `from models import BaseVAE` 写法：手动加载 base.py 放入假模块 models
dummy_models = types.ModuleType("models")
base_file = ROOT / "modules" / "vae_model" / "models" / "base.py"
types_file = ROOT / "modules" / "vae_model" / "models" / "types_.py"
types_code = types_file.read_text(encoding="utf-8")
exec(compile(types_code, str(types_file), "exec"), dummy_models.__dict__)
code = base_file.read_text(encoding="utf-8").replace("from .types_ import *", "")
exec(compile(code, str(base_file), "exec"), dummy_models.__dict__)
sys.modules["models"] = dummy_models

from modules.vae_model.models import vae_models


def load_model(config_path: Path, checkpoint_path: Path) -> torch.nn.Module:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    model_params: Dict = config.get("model_params", {})
    model_cls = vae_models[model_params["name"]]
    model = model_cls(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state[key.replace("model.", "", 1)] = value
        else:
            new_state[key] = value
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def decode_traversal(model: torch.nn.Module, latent_dim: int, steps: int, output_dir: Path, dims=None) -> None:
    """Decode traversal for specified dims; default is all dims."""
    values = np.linspace(-5, 5, steps)
    dim_list = list(range(latent_dim)) if dims is None else list(dims)
    for dim in dim_list:
        latents = []
        for val in values:
            z = torch.zeros(1, latent_dim)
            z[0, dim] = val
            with torch.no_grad():
                waveform = model.decode(z).cpu().numpy()[0]
            latents.append((val, waveform))

        fig, axes = plt.subplots(3, 4, figsize=(12, 6))
        axes = axes.flatten()
        leads = min(len(latents[0][1]), 12)
        for idx in range(leads):
            ax = axes[idx]
            for val, waveform in latents:
                ax.plot(waveform[idx], label=f"{val:.1f}", alpha=0.8)
            ax.set_title(f"Lead {idx+1}")
            ax.set_xlim(0, min(400, waveform.shape[-1]))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, title="z value")
        fig.suptitle(f"Latent dim {dim}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        fig.savefig(output_dir / f"latent_dim_{dim}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VAE latent traversal (-5 to 5).")
    parser.add_argument("--config", required=True, help="YAML config used for training VAE")
    parser.add_argument("--checkpoint", required=True, help="Path to trained VAE checkpoint (.ckpt)")
    parser.add_argument("--output", default="plots/latent_traversal", help="Directory to store traversal plots")
    parser.add_argument("--steps", type=int, default=11, help="Number of points between -5 and 5")
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="逗号分隔的潜在维度列表，例如 0,1,5；为空则遍历全部维度。",
    )
    parser.add_argument(
        "--linear-model",
        type=Path,
        default=None,
        help="可选：线性模型 JSON（含 top_factors），自动选择最具代表性的潜在因子进行可视化。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config_path, checkpoint_path)
    latent_dim = model.latent_dim

    dims = None
    if args.dims:
        dims = [int(x) for x in args.dims.split(",") if x.strip() != ""]
    if args.linear_model and args.linear_model.exists():
        import json

        top = json.loads(args.linear_model.read_text(encoding="utf-8")).get("top_factors", [])
        if top:
            dims = [int(item["latent_dim"]) for item in top]
            print(f"[viz] 使用线性模型 top_factors 维度: {dims}")
    if dims:
        dims = [d for d in dims if 0 <= d < latent_dim]
        print(f"[viz] 仅可视化维度 {dims}")
    decode_traversal(model, latent_dim, args.steps, output_dir, dims=dims)
    print(f"[viz] Latent traversal plots saved to {output_dir}")


if __name__ == "__main__":
    main()
