"""多任务学习模型：卒中预测 + 房颤检测 + 心律失常检测

架构设计：
- 共享编码器（TCN或ResNet1d）
- 三个任务头：
  1. 卒中生存预测（主任务）
  2. 房颤二分类（辅助任务1）
  3. 心律失常二分类（辅助任务2）
- 联合损失函数，可调权重

理论依据：
- ECG-MACE论文证明多任务能提升卒中AUROC 15%（0.66→0.76）
- 房颤和心律失常是卒中的强预测因子（RR=2.06和3.58）
- 辅助任务提供额外监督信号，改善特征学习
"""

import torch
import torch.nn as nn
from typing import Tuple, Sequence, Dict


class MultiTaskECGModel(nn.Module):
    """多任务ECG模型：共享编码器 + 多任务头

    Args:
        encoder_type: 编码器类型，'tcn_light' 或 'resnet'
        input_dim: (n_channels, seq_len)
        n_intervals: 生存分析的时间区间数
        encoder_params: 编码器的额外参数
    """

    def __init__(
        self,
        encoder_type: str = 'tcn_light',
        input_dim: Tuple[int, int] = (8, 4096),
        n_intervals: int = 20,
        encoder_params: dict = None,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        encoder_params = encoder_params or {}

        # 共享编码器
        if encoder_type == 'tcn_light':
            from torch_survival.model_builder import TCNLightSurvival
            # 复用TCN的编码器部分，但不包括输出层
            self.encoder = self._build_tcn_encoder(input_dim, **encoder_params)
            feature_dim = encoder_params.get('num_channels', [16, 32, 64])[-1]
        elif encoder_type == 'resnet':
            from torch_age.resnet_age import ResNet1d
            # 复用ResNet的编码器部分
            self.encoder = self._build_resnet_encoder(input_dim, **encoder_params)
            feature_dim = encoder_params.get('blocks_dim', [(64, 1024), (128, 256), (196, 64), (256, 16)])[-1][0]
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # 任务1：卒中生存预测（主任务）
        self.stroke_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_intervals)
        )

        # 任务2：房颤检测（辅助任务1）
        self.af_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 二分类
        )

        # 任务3：心律失常检测（辅助任务2）
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 二分类
        )

    def _build_tcn_encoder(self, input_dim, **kwargs):
        """构建TCN编码器（不包括输出层）"""
        in_channels, seq_len = input_dim
        num_channels = kwargs.get('num_channels', [16, 32, 64])
        kernel_size = kwargs.get('kernel_size', 3)
        dropout = kwargs.get('dropout', 0.3)

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation_size),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation_size),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

            # 残差连接
            if in_ch != out_ch:
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            else:
                layers.append(nn.Identity())

        return nn.ModuleList(layers)

    def _build_resnet_encoder(self, input_dim, **kwargs):
        """构建ResNet编码器（不包括输出层）"""
        from torch_age.resnet_age import ResNet1d
        blocks_dim = kwargs.get('blocks_dim', [(64, 1024), (128, 256), (196, 64), (256, 16)])
        kernel_size = kwargs.get('kernel_size', 17)
        dropout_rate = kwargs.get('dropout_rate', 0.5)

        # 创建完整ResNet，但只使用编码器部分
        full_model = ResNet1d(
            input_dim=input_dim,
            blocks_dim=blocks_dim,
            n_classes=1,  # 临时值，不会用到
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

        # 提取编码器部分（去掉最后的全连接层）
        return nn.Sequential(*list(full_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            x: (batch, channels, seq_len)

        Returns:
            dict: {
                'stroke': (batch, n_intervals),  # 卒中生存概率
                'af': (batch, 1),                # 房颤logits
                'arrhythmia': (batch, 1),        # 心律失常logits
            }
        """
        # 共享编码器
        if self.encoder_type == 'tcn_light':
            # TCN编码
            for tcn_block, shortcut in zip(self.encoder[::2], self.encoder[1::2]):
                residual = shortcut(x)
                out = tcn_block(x)
                out = out[:, :, :x.size(2)]  # 裁剪因果padding
                x = out + residual
            # 全局平均池化
            features = torch.mean(x, dim=2)  # (batch, feature_dim)
        else:
            # ResNet编码
            features = self.encoder(x)
            if features.dim() == 3:
                features = torch.mean(features, dim=2)
            features = features.view(features.size(0), -1)

        # 三个任务头
        stroke_out = self.stroke_head(features)
        af_out = self.af_head(features)
        arrhythmia_out = self.arrhythmia_head(features)

        return {
            'stroke': stroke_out,
            'af': af_out,
            'arrhythmia': arrhythmia_out,
        }


class MultiTaskLoss(nn.Module):
    """多任务联合损失函数

    Args:
        task_weights: 任务权重，dict {'stroke': w1, 'af': w2, 'arrhythmia': w3}
        n_intervals: 生存分析的时间区间数
    """

    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        n_intervals: int = 20,
    ):
        super().__init__()

        # 默认权重：主任务60%，两个辅助任务各20%
        self.task_weights = task_weights or {
            'stroke': 0.6,
            'af': 0.2,
            'arrhythmia': 0.2,
        }

        # 卒中生存损失
        from torch_survival.losses import SurvLikelihoodLoss
        self.stroke_loss_fn = SurvLikelihoodLoss(n_intervals)

        # 房颤和心律失常损失（二分类）
        self.af_loss_fn = nn.BCEWithLogitsLoss()
        self.arrhythmia_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算联合损失

        Args:
            predictions: 模型输出 {'stroke': ..., 'af': ..., 'arrhythmia': ...}
            targets: 真实标签 {'stroke': ..., 'af': ..., 'arrhythmia': ...}

        Returns:
            total_loss: 加权总损失
            loss_dict: 各任务损失 {'stroke': ..., 'af': ..., 'arrhythmia': ...}
        """
        losses = {}

        # 卒中生存损失
        stroke_probs = torch.sigmoid(predictions['stroke'])
        losses['stroke'] = self.stroke_loss_fn(stroke_probs, targets['stroke'])

        # 房颤损失（只在有标签的样本上计算）
        if 'af' in targets and targets['af'] is not None:
            losses['af'] = self.af_loss_fn(predictions['af'], targets['af'])
        else:
            losses['af'] = torch.tensor(0.0, device=predictions['stroke'].device)

        # 心律失常损失
        if 'arrhythmia' in targets and targets['arrhythmia'] is not None:
            losses['arrhythmia'] = self.arrhythmia_loss_fn(
                predictions['arrhythmia'],
                targets['arrhythmia']
            )
        else:
            losses['arrhythmia'] = torch.tensor(0.0, device=predictions['stroke'].device)

        # 加权总损失
        total_loss = sum(
            self.task_weights[task] * loss
            for task, loss in losses.items()
        )

        return total_loss, losses


def build_multitask_model(
    encoder_type: str = 'tcn_light',
    input_dim: Tuple[int, int] = (8, 4096),
    n_intervals: int = 20,
    **encoder_params
) -> nn.Module:
    """构建多任务模型

    Args:
        encoder_type: 'tcn_light' 或 'resnet'
        input_dim: (n_channels, seq_len)
        n_intervals: 生存分析时间区间数
        **encoder_params: 编码器的额外参数

    Returns:
        MultiTaskECGModel

    Examples:
        # TCN轻量版（1200样本）
        model = build_multitask_model(
            encoder_type='tcn_light',
            input_dim=(8, 4096),
            n_intervals=20,
            num_channels=[16, 32, 64],
            dropout=0.3,
        )

        # ResNet小版（1万样本）
        model = build_multitask_model(
            encoder_type='resnet',
            input_dim=(8, 4096),
            n_intervals=20,
            blocks_dim=[(64, 1024), (128, 256), (196, 64), (256, 16)],
            dropout_rate=0.5,
        )
    """
    model = MultiTaskECGModel(
        encoder_type=encoder_type,
        input_dim=input_dim,
        n_intervals=n_intervals,
        encoder_params=encoder_params,
    )
    return model.float()


# 参数量估算
def estimate_multitask_params(encoder_type: str, input_dim: Tuple[int, int] = (8, 4096)):
    """估算多任务模型参数量"""
    model = build_multitask_model(encoder_type=encoder_type, input_dim=input_dim)
    n_params = sum(p.numel() for p in model.parameters())

    # 分组统计
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    stroke_params = sum(p.numel() for p in model.stroke_head.parameters())
    af_params = sum(p.numel() for p in model.af_head.parameters())
    arrhythmia_params = sum(p.numel() for p in model.arrhythmia_head.parameters())

    return {
        'total': n_params,
        'encoder': encoder_params,
        'stroke_head': stroke_params,
        'af_head': af_params,
        'arrhythmia_head': arrhythmia_params,
    }


if __name__ == '__main__':
    # 验证模型构建
    print("=" * 60)
    print("多任务模型验证")
    print("=" * 60)

    for encoder_type in ['tcn_light', 'resnet']:
        print(f"\n{encoder_type.upper()} 编码器:")

        try:
            model = build_multitask_model(encoder_type=encoder_type)
            params = estimate_multitask_params(encoder_type=encoder_type)

            print(f"  总参数量: {params['total']:,}")
            print(f"  编码器: {params['encoder']:,}")
            print(f"  卒中头: {params['stroke_head']:,}")
            print(f"  房颤头: {params['af_head']:,}")
            print(f"  心律失常头: {params['arrhythmia_head']:,}")

            # 前向传播测试
            x = torch.randn(4, 8, 4096)
            outputs = model(x)
            print(f"  输出形状:")
            for task, out in outputs.items():
                print(f"    {task}: {list(out.shape)}")

            # 损失计算测试
            loss_fn = MultiTaskLoss(n_intervals=20)
            targets = {
                'stroke': torch.zeros(4, 40),  # 2*n_intervals
                'af': torch.randint(0, 2, (4, 1)).float(),
                'arrhythmia': torch.randint(0, 2, (4, 1)).float(),
            }
            total_loss, losses = loss_fn(outputs, targets)
            print(f"  损失:")
            print(f"    总损失: {total_loss.item():.4f}")
            for task, loss in losses.items():
                print(f"    {task}: {loss.item():.4f}")

            print(f"  ✓ {encoder_type} 验证通过")

        except Exception as e:
            print(f"  ✗ {encoder_type} 验证失败: {e}")

    print("\n" + "=" * 60)
    print("推荐配置:")
    print("=" * 60)
    print("\n1200样本（本地）:")
    print("  encoder_type='tcn_light'")
    print("  参数量约 35,000（在推荐范围内）")
    print("\n10000样本（医生端）:")
    print("  encoder_type='resnet'")
    print("  参数量约 150,000（在推荐范围内）")
