#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE 潜变量皮尔逊相关性分析 / Pearson Correlation for VAE Latents
=================================================================

功能 / Purpose
--------------
- 复用 `modules/vae_model`（源自 PyTorch-VAE）的配置与数据加载；
- 加载 Lightning checkpoint，抽取 `μ` 向量；
- 计算潜变量皮尔逊相关矩阵与平均绝对相关（越低代表独立性越好）。

输入参数 / CLI Arguments
------------------------
- ``--vae-root``：PyTorch-VAE 根目录（默认 `modules/vae_model`）。
- ``--config``：训练使用的 YAML 配置，缺省时读取 ``<vae-root>/configs/vae.yaml``。
- ``--checkpoint``：Lightning `.ckpt` 文件。
- ``--split``：`train` / `val` / `test`。
- ``--output``：结果目录（默认 `outputs/analysis/vae_latent`）。

输出 / Outputs
--------------
- `latent_pearson.npy`：皮尔逊相关矩阵。
- `latent_pearson.txt`：`latent_dim` 及 `mean_abs_corr_offdiag` 指标。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pearson correlation between latent dimensions.")
    default_root = Path(__file__).resolve().parents[1] / "modules" / "vae_model"
    parser.add_argument(
        "--vae-root", type=Path, default=default_root,
        help="Path to modules/vae_model (PyTorch-VAE fork)."
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config used for training. Defaults to <vae-root>/configs/vae.yaml."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Lightning checkpoint (*.ckpt) of the trained VAE."
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], default="val",
        help="Dataset split to encode for correlation analysis."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/analysis/vae_latent"),
        help="Directory to store the Pearson matrix and summary."
    )
    return parser.parse_args()


def _import_from_repo(vae_root: Path) -> None:
    vae_root = vae_root.resolve()
    if str(vae_root) not in sys.path:
        sys.path.insert(0, str(vae_root))


def load_config(config_path: Path | None, vae_root: Path) -> dict:
    if config_path is None:
        config_path = vae_root / "configs" / "vae.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 VAE 配置文件：{config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def build_experiment(vae_root: Path, config: dict):
    _import_from_repo(vae_root)
    from experiment import VAEXperiment  # type: ignore
    from models import vae_models  # type: ignore

    model_name = config["model_params"]["name"]
    model = vae_models[model_name](**config["model_params"])
    return VAEXperiment(model, config["exp_params"])


def build_dataset(vae_root: Path, config: dict):
    _import_from_repo(vae_root)
    from dataset import VAEDataset  # type: ignore

    accelerator = config.get("trainer_params", {}).get("accelerator", "auto")
    pin_memory = accelerator not in ("cpu", "mps")
    dataset = VAEDataset(**config["data_params"], pin_memory=pin_memory)
    dataset.setup()
    return dataset


def collect_latents(model, dataset) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    latents = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            tensor = batch[0] if isinstance(batch, tuple) else batch
            tensor = tensor.unsqueeze(0).to(device, dtype=torch.float32)
            mu, _ = model.encode(tensor)
            latents.append(mu.cpu().numpy())
    if not latents:
        raise RuntimeError("目标数据集中没有可用样本，无法计算皮尔逊相关系数。")
    return np.concatenate(latents, axis=0)


def main() -> None:
    args = parse_args()
    vae_root = args.vae_root.resolve()
    config = load_config(args.config, vae_root)

    experiment = build_experiment(vae_root, config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    experiment.load_state_dict(state_dict)

    data = build_dataset(vae_root, config)
    dataset = getattr(data, f"{args.split}_dataset", None)
    if dataset is None:
        raise RuntimeError(f"配置中不存在 {args.split}_dataset，请确认 data_params 划分。")

    latents = collect_latents(experiment.model, dataset)
    corr = np.corrcoef(latents, rowvar=False)
    off_diag = corr - np.eye(corr.shape[0])
    mean_abs_corr = float(np.mean(np.abs(off_diag)))

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "latent_pearson.npy", corr)
    (out_dir / "latent_pearson.txt").write_text(
        f"latent_dim={corr.shape[0]}\nmean_abs_corr_offdiag={mean_abs_corr:.6f}\n",
        encoding="utf-8",
    )

    print(f"[pearson] latent_dim={corr.shape[0]} mean|corr|={mean_abs_corr:.6f}")
    print(f"[pearson] 结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
