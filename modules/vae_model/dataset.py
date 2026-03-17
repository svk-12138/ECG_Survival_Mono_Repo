import os
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.datasets.folder import default_loader
from torch import Tensor


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 


class MedianBeatCSVDataset(Dataset):
    """
    Dataset for median-beat ECG CSV files exported from Braveheart.
    """

    def __init__(
        self,
        files: Iterable[Union[str, Path]],
        lead_order: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        pad_value: float = 0.0,
        normalize: str = "zscore",
        dtype: torch.dtype = torch.float32,
        target_hw: Optional[Sequence[int]] = None,
        representation: str = "image",
        return_meta: bool = False,
    ) -> None:
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("MedianBeatCSVDataset 需要至少 1 个 CSV 文件。")
        self.lead_order = lead_order
        self.max_length = max_length
        self.pad_value = pad_value
        self.normalize = normalize.lower()
        self.dtype = dtype
        self.target_hw = tuple(target_hw) if target_hw else None
        self.return_meta = return_meta
        self.representation = representation.lower()

    def __len__(self) -> int:
        return len(self.files)

    def _load_csv(self, path: Path) -> Tuple[np.ndarray, List[str]]:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{path} 没有数据。")
        header = df.columns.tolist()
        decoded_header = [str(h).strip() for h in header]
        first_col = decoded_header[0].lower()
        has_meta_col = (
            first_col in {"", "index", "sample", "time", "t"}
            or first_col.startswith("unnamed")
        )
        if not has_meta_col:
            values = df.to_numpy(dtype=np.float32)
            lead_cols = decoded_header
        else:
            lead_cols = decoded_header[1:]
            values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if not np.isfinite(values).all():
            # Replace NaN/Inf with zeros to keep training numerically stable.
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        lead_cols = [str(c).strip() for c in lead_cols]
        if self.lead_order:
            lookup = {name.upper(): idx for idx, name in enumerate(lead_cols)}
            indices = []
            for lead in self.lead_order:
                key = lead.upper()
                if key not in lookup:
                    raise KeyError(f"{path} 缺少导联 {lead}")
                indices.append(lookup[key])
            values = values[:, indices]
            lead_cols = list(self.lead_order)
        signal = values.T
        return signal, lead_cols

    def _normalize(self, signal: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        mode = self.normalize
        if mode == "zscore":
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            return (signal - mean) / std, {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        if mode == "minmax":
            min_v = signal.min(axis=1, keepdims=True)
            max_v = signal.max(axis=1, keepdims=True)
            denom = max_v - min_v
            denom[denom == 0] = 1.0
            return (signal - min_v) / denom, {"min": min_v.astype(np.float32), "max": max_v.astype(np.float32)}
        return signal, None

    def _align_length(self, signal: np.ndarray) -> np.ndarray:
        if self.max_length is None:
            return signal
        seq_len = signal.shape[1]
        if seq_len > self.max_length:
            return signal[:, : self.max_length]
        if seq_len < self.max_length:
            pad = np.full(
                (signal.shape[0], self.max_length - seq_len),
                self.pad_value,
                dtype=signal.dtype,
            )
            return np.concatenate([signal, pad], axis=1)
        return signal

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Dict[str, Any]]]:
        signal, lead_names = self._load_csv(self.files[idx])
        aligned = self._align_length(signal)
        raw_copy = aligned.astype(np.float32, copy=True)
        normed, stats = self._normalize(aligned.astype(np.float32, copy=True))

        if self.representation == "waveform":
            tensor = torch.from_numpy(normed).to(self.dtype)
        else:
            tensor = torch.from_numpy(normed).to(self.dtype).unsqueeze(0)
            if self.target_hw:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=self.target_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        label = torch.tensor(0.0, dtype=self.dtype)
        if not self.return_meta:
            return tensor, label

        meta = {
            "raw": raw_copy,
            "stats": stats,
            "lead_names": lead_names,
        }
        return tensor, label, meta

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_type = kwargs.get("dataset_type", "celeba").lower()
        self.extra_params = kwargs
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        if self.dataset_type == "median_csv":
            self._setup_median_csv()
            return

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
        self.test_dataset = self.val_dataset
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def _setup_median_csv(self) -> None:
        data_dir = Path(self.data_dir)
        pattern = self.extra_params.get("file_pattern", "*.csv")
        files = sorted(data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 未找到匹配 {pattern} 的 CSV 文件。")

        shuffle = bool(self.extra_params.get("shuffle", False))
        split_seed = int(self.extra_params.get("split_seed", 1265))
        if shuffle:
            rng = np.random.default_rng(split_seed)
            rng.shuffle(files)

        val_fraction = float(self.extra_params.get("val_fraction", 0.1))
        test_fraction = float(self.extra_params.get("test_fraction", 0.1))
        if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1 and both non-negative.")
        total = len(files)
        val_count = max(1, int(total * val_fraction))
        test_count = max(1, int(total * test_fraction))
        if val_count + test_count >= total:
            val_count = max(1, min(val_count, total - 2))
            test_count = max(1, min(test_count, total - val_count - 1))
        train_count = total - val_count - test_count
        if train_count <= 0:
            raise ValueError("Not enough files to satisfy train/val/test split.")

        train_files = files[:train_count]
        val_files = files[train_count : train_count + val_count]
        test_files = files[train_count + val_count :] or files[-1:]

        median_kwargs = {
            "lead_order": self.extra_params.get("lead_order"),
            "max_length": self.extra_params.get("max_length"),
            "pad_value": float(self.extra_params.get("pad_value", 0.0)),
            "normalize": self.extra_params.get("normalize", "zscore"),
            "target_hw": self.extra_params.get("target_hw"),
            "representation": self.extra_params.get("representation", "image"),
            "return_meta": self.extra_params.get("return_meta", False),
        }

        self.train_dataset = MedianBeatCSVDataset(train_files, **median_kwargs)
        self.val_dataset = MedianBeatCSVDataset(val_files, **median_kwargs)
        self.test_dataset = MedianBeatCSVDataset(test_files, **median_kwargs)
     
