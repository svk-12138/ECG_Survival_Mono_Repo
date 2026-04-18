import torch.nn as nn
import torch
from typing import Sequence, Tuple

from torch_age.resnet_age import ResNet1d


class ConvTransformerSurvival(nn.Module):
    """1D CNN 提特征 + Transformer 编码，再输出离散时间存活概率。"""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        n_intervals: int,
        conv_channels: Sequence[int] = (64, 128, 256),
        conv_kernel: int = 7,
        conv_stride: int = 2,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_seq_len: int = 128,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        conv_layers = []
        cur_channels = in_channels
        for ch in conv_channels:
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        cur_channels,
                        ch,
                        kernel_size=conv_kernel,
                        stride=conv_stride,
                        padding=conv_kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            cur_channels = ch
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(transformer_seq_len)
        self.channel_proj = nn.Conv1d(cur_channels, transformer_dim, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        head_hidden = max(transformer_dim // 2, 32)
        self.head = nn.Sequential(
            nn.Linear(transformer_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden, n_intervals),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        feats = self.conv(x)
        feats = self.pool(feats)
        feats = self.channel_proj(feats)  # (B, d, T)
        feats = feats.transpose(1, 2)  # (B, T, d)
        encoded = self.transformer(feats)  # (B, T, d)
        pooled = encoded.mean(dim=1)  # (B, d)
        return self.head(pooled)


def build_survival_cnn_transformer(
    n_intervals: int,
    input_dim: Tuple[int, int] = (12, 4096),
    conv_channels: Sequence[int] = (64, 128, 256),
    conv_kernel: int = 7,
    conv_stride: int = 2,
    transformer_dim: int = 128,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    transformer_seq_len: int = 128,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """构建 CNN+Transformer 的离散生存模型。"""
    model = ConvTransformerSurvival(
        in_channels=input_dim[0],
        seq_len=input_dim[1],
        n_intervals=n_intervals,
        conv_channels=conv_channels,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        transformer_dim=transformer_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        transformer_seq_len=transformer_seq_len,
        dropout_rate=dropout_rate,
    )
    return model.float()


# 为了兼容旧导入，保留 build_survival_resnet 接口但指向新模型
def build_survival_resnet(
    n_intervals: int,
    input_dim: Tuple[int, int] = (12, 4096),
    blocks_dim: Sequence[Tuple[int, int]] = ((64, 1024), (128, 256), (196, 64), (256, 16)),
    kernel_size: int = 17,
    dropout_rate: float = 0.8,
) -> nn.Module:
    """保留原始 ResNet1d 生存模型实现。"""
    backbone = ResNet1d(
        input_dim=input_dim,
        blocks_dim=blocks_dim,
        n_classes=n_intervals,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
    )
    return backbone.float()


class TCNLightSurvival(nn.Module):
    """时序卷积网络（TCN）轻量版，用于生存分析

    特点：
    - 因果卷积（无信息泄露，padding在左侧）
    - 膨胀卷积（扩大感受野）
    - 残差连接
    - 参数量约25,000（适合1200样本）

    架构：
    - 输入：(batch, in_channels, seq_len)
    - TCN块1：16通道，膨胀率1
    - TCN块2：32通道，膨胀率2
    - TCN块3：64通道，膨胀率4
    - 全局平均池化
    - 输出层：64 → n_intervals
    """

    def __init__(
        self,
        in_channels: int,
        n_intervals: int,
        num_channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            # 因果卷积：padding只在左侧
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                nn.Sequential(
                    # 第一个卷积
                    nn.Conv1d(
                        in_ch, out_ch, kernel_size,
                        stride=1, padding=padding, dilation=dilation_size
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),

                    # 第二个卷积
                    nn.Conv1d(
                        out_ch, out_ch, kernel_size,
                        stride=1, padding=padding, dilation=dilation_size
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

            # 残差连接的shortcut
            if in_ch != out_ch:
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            else:
                layers.append(nn.Identity())

        self.tcn_blocks = nn.ModuleList(layers[::2])  # 主路径
        self.shortcuts = nn.ModuleList(layers[1::2])  # 残差连接

        # 全局平均池化 + 输出层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], n_intervals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        for tcn_block, shortcut in zip(self.tcn_blocks, self.shortcuts):
            residual = shortcut(x)
            out = tcn_block(x)
            # 裁剪因果padding产生的右侧多余部分
            out = out[:, :, :x.size(2)]
            x = out + residual

        # 全局池化
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        return self.fc(x)


def build_survival_tcn_light(
    n_intervals: int,
    input_dim: Tuple[int, int] = (8, 4096),
    num_channels: Sequence[int] = (16, 32, 64),
    kernel_size: int = 3,
    dropout: float = 0.3,
) -> nn.Module:
    """构建TCN轻量版生存模型

    参数量估算：~25,000
    适用场景：1200样本

    Args:
        n_intervals: 离散时间区间数
        input_dim: (n_channels, seq_len)，默认(8, 4096)
        num_channels: TCN各层通道数，默认(16, 32, 64)
        kernel_size: 卷积核大小，默认3
        dropout: Dropout率，默认0.3

    Returns:
        TCNLightSurvival模型
    """
    model = TCNLightSurvival(
        in_channels=input_dim[0],
        n_intervals=n_intervals,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    return model.float()


__all__ = ["build_survival_cnn_transformer", "build_survival_resnet", "build_survival_tcn_light",
           "build_survival_cnn_gru", "build_survival_cnn_transformer_small"]


class CNNGRUSurvival(nn.Module):
    """CNN 提特征 + 双向 GRU 时序建模，输出离散时间存活概率。

    参数量约 8 万，适合 1 万样本场景。
    """

    def __init__(
        self,
        in_channels: int,
        n_intervals: int,
        conv_channels: Sequence[int] = (32, 64, 64),
        conv_kernels: Sequence[int] = (7, 5, 3),
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_layers = []
        cur = in_channels
        for ch, k in zip(conv_channels, conv_kernels):
            conv_layers.append(nn.Sequential(
                nn.Conv1d(cur, ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ))
            cur = ch
        self.conv = nn.Sequential(*conv_layers)
        self.gru = nn.GRU(
            input_size=cur,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, n_intervals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)               # (B, C, L')
        x = x.transpose(1, 2)         # (B, L', C)
        _, h = self.gru(x)             # h: (2*layers, B, hidden)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden*2)
        return self.head(self.dropout(h))


def build_survival_cnn_gru(
    n_intervals: int,
    input_dim: Tuple[int, int] = (8, 4096),
    conv_channels: Sequence[int] = (32, 64, 64),
    conv_kernels: Sequence[int] = (7, 5, 3),
    gru_hidden: int = 64,
    gru_layers: int = 1,
    dropout: float = 0.3,
) -> nn.Module:
    """构建 CNN+GRU 生存模型，参数量约 8 万，适合 1 万样本。"""
    return CNNGRUSurvival(
        in_channels=input_dim[0],
        n_intervals=n_intervals,
        conv_channels=conv_channels,
        conv_kernels=conv_kernels,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        dropout=dropout,
    ).float()


class CNNTransformerSmallSurvival(nn.Module):
    """轻量 CNN+Transformer，参数量约 10 万，适合 1 万样本。

    相比 ConvTransformerSurvival（69万参数）大幅缩减：
    - conv 通道数减半（32/64）
    - transformer dim=64，1层，4头
    """

    def __init__(
        self,
        in_channels: int,
        n_intervals: int,
        conv_channels: Sequence[int] = (32, 64),
        transformer_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        seq_len: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_layers = []
        cur = in_channels
        for ch in conv_channels:
            conv_layers.append(nn.Sequential(
                nn.Conv1d(cur, ch, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ))
            cur = ch
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(seq_len)
        self.proj = nn.Conv1d(cur, transformer_dim, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, n_intervals),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.proj(x).transpose(1, 2)   # (B, T, d)
        x = self.transformer(x).mean(dim=1) # (B, d)
        return self.head(x)


def build_survival_cnn_transformer_small(
    n_intervals: int,
    input_dim: Tuple[int, int] = (8, 4096),
    conv_channels: Sequence[int] = (32, 64),
    transformer_dim: int = 64,
    transformer_heads: int = 4,
    transformer_layers: int = 1,
    seq_len: int = 64,
    dropout: float = 0.3,
) -> nn.Module:
    """构建轻量 CNN+Transformer 生存模型，参数量约 10 万，适合 1 万样本。"""
    return CNNTransformerSmallSurvival(
        in_channels=input_dim[0],
        n_intervals=n_intervals,
        conv_channels=conv_channels,
        transformer_dim=transformer_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        seq_len=seq_len,
        dropout=dropout,
    ).float()
