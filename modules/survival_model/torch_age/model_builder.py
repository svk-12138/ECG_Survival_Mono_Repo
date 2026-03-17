"""构建 PyTorch 版 ResNet1d ECG 模型，基于 ecg-age-prediction-main。"""
from __future__ import annotations

from typing import Sequence, Tuple
import torch.nn as nn

from .resnet_age import ResNet1d

DEFAULT_BLOCKS = [
    (64, 1024),
    (128, 256),
    (196, 64),
    (256, 16),
]


def build_resnet_ecg_model(
    n_classes: int = 1,
    input_dim: Tuple[int, int] = (12, 4096),
    blocks_dim: Sequence[Tuple[int, int]] = DEFAULT_BLOCKS,
    kernel_size: int = 17,
    dropout_rate: float = 0.8,
) -> nn.Module:
    """实例化 ResNet1d 模型。
    参数与原仓库保持一致，方便与 TensorFlow 版本对照。
    """
    return ResNet1d(input_dim=input_dim, blocks_dim=blocks_dim, n_classes=n_classes,
                    kernel_size=kernel_size, dropout_rate=dropout_rate)

__all__ = ["build_resnet_ecg_model", "DEFAULT_BLOCKS"]
