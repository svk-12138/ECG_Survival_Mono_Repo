"""PyTorch 实现的离散时间生存似然损失。"""
from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn

class SurvLikelihoodLoss(nn.Module):
    """复刻 nnet-survival 的离散时间似然 (Keras 版) 到 PyTorch。"""

    def __init__(self, n_intervals: int, reduction: Literal["mean", "sum", "none"] = "mean", eps: float = 1e-7):
        super().__init__()
        self.n_intervals = n_intervals
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """y_true 形状 = (batch, 2*n_intervals)，与 nnet_survival.make_surv_array 输出一致。"""
        n = self.n_intervals
        probs = torch.clamp(y_pred, min=self.eps, max=1.0 - self.eps)
        surv_part = 1.0 + y_true[:, :n] * (probs - 1.0)
        fail_part = 1.0 - y_true[:, n:2 * n] * probs
        concat = torch.cat([surv_part, fail_part], dim=1)
        loss_vec = -torch.log(torch.clamp(concat, min=self.eps)).sum(dim=1)
        if self.reduction == "mean":
            return loss_vec.mean()
        if self.reduction == "sum":
            return loss_vec.sum()
        return loss_vec

__all__ = ["SurvLikelihoodLoss"]
