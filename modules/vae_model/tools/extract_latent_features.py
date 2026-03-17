#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract latent vectors from a trained MedianBeatVAE and dump them to CSV.
Each row corresponds to one CSV sample (median beat), with latent values and
an optional reconstruction sMAPE metric.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import MedianBeatCSVDataset  # noqa: E402
from experiment import VAEXperiment  # noqa: E402
from models import vae_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latent features from a trained VAE.")
    parser.add_argument("--config", default="configs/median_vae.yaml", type=Path, help="Config YAML path.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint file.")
    parser.add_argument("--data-path", type=Path, help="Override data_path in config for inference.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for input CSV files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples processed.")
    parser.add_argument("--val-fraction", type=float, default=None, help="Override val_fraction split.")
    parser.add_argument("--test-fraction", type=float, default=None, help="Override test_fraction split.")
    parser.add_argument("--output", default=Path("results/latents"), type=Path, help="Output directory.")
    parser.add_argument("--ignore-lead-order", action="store_true", help="忽略配置中的导联顺序校验，避免缺导联报错。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    parser.add_argument(
        "--id-after-dot",
        action="store_true",
        help="若文件名形如 prefix.sampleid_xxx.csv，则仅保留小数点后的部分作为 ID，便于与标签对齐。",
    )
    parser.add_argument(
        "--id-strip-suffix",
        default="",
        help="可选，移除 ID 末尾的后缀（例如 '_median'），便于与标签表的 sample_id 对齐。",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_model(config: dict, checkpoint: Path, device: torch.device) -> VAEXperiment:
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


def smape(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    numerator = torch.abs(a - b)
    denominator = torch.abs(a) + torch.abs(b) + eps
    return float((2.0 * numerator / denominator).mean().cpu())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    data_root = Path(args.data_path) if args.data_path else Path(cfg["data_params"]["data_path"])
    files = sorted(data_root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"在 {data_root} 未找到匹配 {args.pattern} 的 CSV。")
    if args.limit:
        files = files[: args.limit]

    lead_order = None if args.ignore_lead_order else cfg["data_params"].get("lead_order")
    dataset = MedianBeatCSVDataset(
        files,
        lead_order=lead_order,
        max_length=cfg["data_params"].get("max_length"),
        pad_value=float(cfg["data_params"].get("pad_value", 0.0)),
        normalize=cfg["data_params"].get("normalize", "zscore"),
        representation=cfg["data_params"].get("representation", "waveform"),
        return_meta=False,
    )

    experiment = init_model(cfg, args.checkpoint, dev)
    latent_dim = cfg["model_params"]["latent_dim"]

    # split files into train/val/test for downstream linear script
    val_frac = args.val_fraction if args.val_fraction is not None else cfg["data_params"].get("val_fraction", 0.1)
    test_frac = args.test_fraction if args.test_fraction is not None else cfg["data_params"].get("test_fraction", 0.1)
    n = len(files)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    file_to_idx = {f: i for i, f in enumerate(files)}

    def encode_split(split_name: str, split_files: List[Path]) -> None:
        if not split_files:
            return
        ids = []
        latents = []
        skipped = 0
        for f in split_files:
            idx = file_to_idx[f]
            try:
                tensor, _ = dataset[idx]
            except KeyError as exc:
                print(f"[vae_infer] 跳过 {f.name}: {exc}")
                skipped += 1
                continue
            # 调整导联数到模型需要的通道数
            c_expected = cfg["model_params"].get("in_channels", 12)
            c_now = tensor.shape[0]
            if c_now < c_expected:
                pad = torch.zeros((c_expected - c_now, tensor.shape[1]), dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            elif c_now > c_expected:
                tensor = tensor[:c_expected]
            signal = tensor.unsqueeze(0).to(dev)
            with torch.no_grad():
                try:
                    _, _, mu, _ = experiment.model(signal)
                except RuntimeError as exc:
                    print(f"[vae_infer] 跳过 {f.name}: {exc}")
                    skipped += 1
                    continue
            _id = f.stem
            if args.id_after_dot and "." in _id:
                _id = _id.split(".", 1)[1]
            if args.id_strip_suffix and _id.endswith(args.id_strip_suffix):
                _id = _id[: -len(args.id_strip_suffix)]
            ids.append(_id)
            latents.append(mu.squeeze(0).cpu().numpy())
        if not latents:
            print(f"[vae_infer] {split_name}: 0 samples kept (skipped {skipped})")
            return
        np.savez(args.output / f"{split_name}_latents.npz", ids=np.array(ids), latents=np.stack(latents))
        manifest_rows.append((split_name, len(ids), f"{split_name}_latents.npz"))
        print(f"[vae_infer] {split_name}: {len(ids)} samples (skipped {skipped}) -> {args.output / (split_name + '_latents.npz')}")

    encode_split("train", train_files)
    encode_split("val", val_files)
    encode_split("test", test_files)

    with (args.output / "manifest.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["split", "samples", "file"])
        for row in manifest_rows:
            writer.writerow(row)

    print(f"[DONE] Latent npz saved to {args.output}, manifest written.")


if __name__ == "__main__":
    main()
