#!/usr/bin/env python3
"""
Generate latent traversal reconstructions for Median-beat VAE checkpoints.

This script helps复现论文中的 VAE 可解释性分析：沿着线性模型权重最大的潜在维度，
采样若干步长并解码成波形，输出 npz/可视化图片。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "modules" / "vae_model"))
from models import vae_models  # noqa: E402


def _load_config(cfg_path: Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _instantiate_model(cfg: Dict) -> torch.nn.Module:
    model_name = cfg["model_params"]["name"]
    params = dict(cfg["model_params"])
    params.pop("name", None)
    model_cls = vae_models[model_name]
    model = model_cls(**params)
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for key, val in state.items():
        if key.startswith("model."):
            cleaned[key.replace("model.", "", 1)] = val
        else:
            cleaned[key] = val
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    model.to(device).eval()


def _load_latents(latent_dir: Path) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for name in ("train_latents.npz", "val_latents.npz", "test_latents.npz"):
        path = latent_dir / name
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        arrays.append(data["latents"])
    if not arrays:
        raise FileNotFoundError(f"未在 {latent_dir} 找到 *latents.npz")
    return np.concatenate(arrays, axis=0)


def _load_linear_model(linear_json: Path, top_k: int | None) -> List[Dict]:
    payload = json.loads(linear_json.read_text(encoding="utf-8"))
    candidates = payload.get("top_factors") or []
    if not candidates:
        coef = payload.get("coefficients")
        if coef is None:
            raise ValueError(f"{linear_json} 缺少 top_factors/coefficients")
        candidates = [
            {"latent_dim": idx, "weight": float(weight)}
            for idx, weight in enumerate(coef)
        ]
    if top_k is not None:
        return candidates[:top_k]
    return candidates


def _plot_waveforms(
    waveforms: np.ndarray,
    steps: Sequence[float],
    out_png: Path,
    lead_names: Sequence[str],
) -> None:
    if not HAS_MPL:
        print("[warn] matplotlib 不可用，跳过绘图")
        return
    leads = waveforms.shape[1]
    cols = 2
    rows = (leads + cols - 1) // cols
    time_axis = np.arange(waveforms.shape[-1])
    plt.figure(figsize=(cols * 4, rows * 2.5))
    for lead in range(leads):
        ax = plt.subplot(rows, cols, lead + 1)
        for idx, step in enumerate(steps):
            ax.plot(
                time_axis,
                waveforms[idx, lead],
                label=f"{step:.2f}σ",
                alpha=0.7,
            )
        ax.set_title(lead_names[lead] if lead < len(lead_names) else f"Lead {lead+1}")
        ax.set_xlim(0, time_axis[-1])
        ax.grid(True, alpha=0.2)
        if lead == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def traverse_latent(
    model: torch.nn.Module,
    device: torch.device,
    base: np.ndarray,
    std: np.ndarray,
    latent_idx: int,
    steps: np.ndarray,
    scale_mode: str = "std",
) -> np.ndarray:
    vectors = []
    for step in steps:
        vec = base.copy()
        if scale_mode == "std":
            vec[latent_idx] = base[latent_idx] + step * std[latent_idx]
        else:
            vec[latent_idx] = base[latent_idx] + step
        vectors.append(vec)
    tensor = torch.from_numpy(np.stack(vectors)).float().to(device)
    with torch.no_grad():
        decoded = model.decode(tensor).cpu().numpy()
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Median-beat VAE latent traversal")
    parser.add_argument("--config", required=True, help="与 VAE 训练一致的 YAML 配置")
    parser.add_argument("--checkpoint", required=True, help="VAE Lightning checkpoint 路径")
    parser.add_argument("--latents-dir", required=True, help="run.py --export-latents 的输出目录")
    parser.add_argument("--linear-json", required=True, help="train_linear_scores.py 生成的 linear_model.json")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--range-mode",
        choices=["std", "fixed"],
        default="std",
        help="std=按 σ 倍数取值；fixed=使用固定数值范围（补充材料的 -5..5）",
    )
    parser.add_argument("--num-steps", type=int, default=5, help="range-mode=std 时的采样步数")
    parser.add_argument("--std-range", type=float, default=2.0, help="range-mode=std 时沿 ±std_range*σ 采样")
    parser.add_argument("--fixed-min", type=float, default=-5.0, help="range-mode=fixed 时的最小值")
    parser.add_argument("--fixed-max", type=float, default=5.0, help="range-mode=fixed 时的最大值")
    parser.add_argument("--fixed-step", type=float, default=1.0, help="range-mode=fixed 时的步长")
    parser.add_argument(
        "--base-mode",
        choices=["mean", "zero"],
        default="mean",
        help="mean=以整体潜在均值为基线；zero=将未调节因子固定为 0（补充材料设置）",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["std", "absolute"],
        default="std",
        help="std=step×σ（旧逻辑）；absolute=直接将维度设置为 step 值",
    )
    parser.add_argument("--top-k", type=int, default=3, help="选择前 K 个潜在维度")
    parser.add_argument("--device", default="cpu", help="torch 设备，如 cuda:0")
    parser.add_argument(
        "--lead-names",
        nargs="*",
        default=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        help="绘图时的导联名称",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(args.config))
    model = _instantiate_model(config)
    device = torch.device(args.device)
    _load_checkpoint(model, Path(args.checkpoint), device)

    latents = _load_latents(Path(args.latents_dir))
    mean = latents.mean(axis=0)
    std = latents.std(axis=0)
    std[std == 0] = 1.0

    if args.range_mode == "std":
        steps = np.linspace(-args.std_range, args.std_range, args.num_steps)
    else:
        if args.fixed_step <= 0:
            raise ValueError("fixed-step 必须 > 0")
        steps = np.arange(args.fixed_min, args.fixed_max + 1e-9, args.fixed_step)

    if args.base_mode == "mean":
        base_vec = mean
    else:
        base_vec = np.zeros_like(mean)
    factors = _load_linear_model(Path(args.linear_json), args.top_k)

    manifest_rows = []
    for rank, factor in enumerate(factors, start=1):
        idx = int(factor["latent_dim"])
        decoded = traverse_latent(
            model,
            device,
            base_vec,
            std,
            idx,
            steps,
            scale_mode=args.scale_mode,
        )
        npz_path = output_dir / f"latent_{idx:02d}_traversal.npz"
        np.savez(
            npz_path,
            latent_dim=idx,
            weight=float(factor.get("weight", 0.0)),
            steps=steps,
            waveforms=decoded,
        )
        png_path = output_dir / f"latent_{idx:02d}_traversal.png"
        _plot_waveforms(decoded, steps, png_path, args.lead_names)
        manifest_rows.append(
            {
                "rank": rank,
                "latent_dim": idx,
                "weight": float(factor.get("weight", 0.0)),
                "npz": npz_path.name,
                "png": png_path.name if png_path.exists() else "",
            }
        )
        print(f"[latent] 维度 {idx} 完成，结果写入 {npz_path.parent}")

    if manifest_rows:
        import pandas as pd

        df = pd.DataFrame(manifest_rows)
        df.to_csv(output_dir / "latent_traversal_manifest.csv", index=False)
        print(f"[latent] 总览写入 {output_dir / 'latent_traversal_manifest.csv'}")


if __name__ == "__main__":
    main()
