"""
Generic runner for VAE models.

Supplementary AI-ECG protocol highlights:
- Median-beat ECG VAEs employ 1D convolutional encoders/decoders with
  progressively increasing filters and shrinking kernels.
- Latent space is constrained to <=30 factors to maintain disentanglement.
- Reconstruction loss uses symmetric MAPE plus KL divergence with a beta
  weight; beta candidates [0.1, 0.25, 0.5, 1, 3, 5, 10] were explored and
  beta=0.25 performed best (Pearson correlation + latent traversal review).
- Training typically runs for 50 epochs on a single RTX 6000 GPU.

Use --median-vae-profile to automatically enforce these defaults.
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
# from pytorch_lightning.plugins import DDPPlugin
from typing import List, Dict, Tuple, Optional


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--median-vae-profile', action='store_true',
                    help='Apply ECG median-beat VAE defaults (latent<=30, beta grid, sMAPE loss, max 50 epochs).')
parser.add_argument('--export-latents', action='store_true', help='导出训练/验证/测试的 VAE 潜在表征供后续线性模型使用。')
parser.add_argument('--latent-pearson', action='store_true',
                    help='在验证集潜在均值上计算皮尔逊相关性矩阵，并保存指标文件。')

DEFAULT_BETA_GRID: List[float] = [0.1, 0.25, 0.5, 1.0, 3.0, 5.0, 10.0]


def _export_latent_representations(experiment: VAEXperiment, datamodule: VAEDataset, log_dir: str) -> None:
    """Encode every sample with the trained VAE encoder and save latent vectors for linear models."""
    export_dir = Path(log_dir) / "latents"
    export_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = experiment.model.to(device)
    model.eval()

    splits: List[Tuple[str, Optional[Dataset], int]] = [
        ("train", getattr(datamodule, "train_dataset", None), getattr(datamodule, "train_batch_size", 32)),
        ("val", getattr(datamodule, "val_dataset", None), getattr(datamodule, "val_batch_size", 32)),
        ("test", getattr(datamodule, "test_dataset", None), getattr(datamodule, "val_batch_size", 32)),
    ]

    manifest: List[Tuple[str, int, str]] = []
    for split_name, dataset, _ in splits:
        if dataset is None:
            continue
        latents: List[np.ndarray] = []
        sample_ids: List[str] = []
        files = getattr(dataset, "files", None)
        with torch.no_grad():
            for idx in range(len(dataset)):
                batch = dataset[idx]
                if isinstance(batch, tuple):
                    tensor = batch[0]
                else:
                    tensor = batch
                tensor = tensor.unsqueeze(0).to(device, dtype=torch.float32)
                mu, _ = model.encode(tensor)
                latents.append(mu.cpu().numpy())
                if files:
                    sample_ids.append(Path(files[idx]).stem)
                else:
                    sample_ids.append(f"{split_name}_{idx}")
        if not latents:
            continue
        out_path = export_dir / f"{split_name}_latents.npz"
        np.savez(out_path, ids=np.array(sample_ids), latents=np.concatenate(latents, axis=0))
        manifest.append((split_name, len(sample_ids), out_path.name))
        print(f"[export] {split_name} 集合导出 {len(sample_ids)} 条样本 -> {out_path}")

    if manifest:
        with open(export_dir / "manifest.tsv", "w", encoding="utf-8") as fh:
            fh.write("split\tsamples\tfile\n")
            for split, count, name in manifest:
                fh.write(f"{split}\t{count}\t{name}\n")
        print(f"[export] 潜在表征总览写入 {export_dir / 'manifest.tsv'}")

args = parser.parse_args()
config = None
try:
    with open(args.filename, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError as exc:
    raise FileNotFoundError(f"无法找到配置文件 {args.filename}") from exc
except yaml.YAMLError as exc:
    raise RuntimeError(f"解析配置文件 {args.filename} 失败：{exc}") from exc

if config is None:
    raise RuntimeError(f"配置文件 {args.filename} 为空或未正确加载。")

if args.median_vae_profile:
    config.setdefault('model_params', {})
    config.setdefault('exp_params', {})
    config.setdefault('trainer_params', {})
    latent = config['model_params'].get('latent_dim', 30)
    if latent > 30:
        print(f"[median-vae] latent_dim {latent} clipped to 30.")
        latent = 30
    config['model_params']['latent_dim'] = latent
    beta = config['model_params'].get('beta', 0.25)
    if beta not in DEFAULT_BETA_GRID:
        print(f"[median-vae] beta {beta} not in grid {DEFAULT_BETA_GRID}; using 0.25.")
        beta = 0.25
    config['model_params']['beta'] = beta
    config['exp_params']['reconstruction_loss'] = config['exp_params'].get('reconstruction_loss', 'sMAPE')
    config['exp_params']['beta_grid'] = DEFAULT_BETA_GRID
    max_epochs = config['trainer_params'].get('max_epochs', 50)
    if max_epochs > 50:
        print(f"[median-vae] max_epochs {max_epochs} truncated to 50.")
        max_epochs = 50
    config['trainer_params']['max_epochs'] = max_epochs
    trainer_defaults = config['trainer_params']
    trainer_defaults.setdefault('accelerator', 'auto')
    trainer_defaults.setdefault('devices', 1)
    config['trainer_params']['accumulate_grad_batches'] = config['trainer_params'].get('accumulate_grad_batches', 1)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

accelerator = config['trainer_params'].get('accelerator', 'auto')
use_pinned = accelerator not in ('cpu', 'mps')
data = VAEDataset(**config["data_params"], pin_memory=use_pinned)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                #  strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
if args.median_vae_profile:
    print("[median-vae] Profile active:")
    print(f"  latent_dim: {config['model_params']['latent_dim']}")
    print(f"  beta: {config['model_params']['beta']}")
    print(f"  beta grid: {config['exp_params'].get('beta_grid')}")
    print(f"  reconstruction loss: {config['exp_params']['reconstruction_loss']}")
    print(f"  max_epochs: {config['trainer_params']['max_epochs']}")
runner.fit(experiment, datamodule=data)

best_path = None
if getattr(runner, "checkpoint_callback", None):
    best_path = runner.checkpoint_callback.best_model_path or None

try:
    test_results = runner.test(datamodule=data, ckpt_path=best_path)
    if test_results:
        print(f"Test evaluation results: {test_results}")
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] 测试阶段执行失败：{exc}")

if args.export_latents:
    try:
        data.setup()  # ensure datasets are ready
        _export_latent_representations(experiment, data, tb_logger.log_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 导出潜在表征失败：{exc}")
