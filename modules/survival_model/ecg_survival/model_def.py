import sys
from pathlib import Path
from typing import Optional

import tensorflow as tf

# 将外部项目加入路径，便于直接复用其实现
ROOT = Path(__file__).resolve().parent.parent
AUTOMATIC_ECG_PATH = Path("/home/admin123/use/Program/automatic-ecg-diagnosis")
NNET_SURVIVAL_PATH = Path("/home/admin123/use/Program/nnet-survival")
for p in (AUTOMATIC_ECG_PATH, NNET_SURVIVAL_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model import get_model as get_ecg_model  # type: ignore
from nnet_survival import surv_likelihood  # type: ignore

def build_ecg_survival_model(
    n_intervals: int,
    lr: float = 1e-3,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
) -> tf.keras.Model:
    """
    构建 ECG + 生存分析模型：
    - Backbone: 来自 automatic-ecg-diagnosis/get_model，输出维度设为 n_intervals。
    - 激活: sigmoid（离散时间条件生存概率）。
    - 损失: nnet-survival 的离散时间似然。
    """
    model = get_ecg_model(n_classes=n_intervals, last_layer="sigmoid")
    loss_fn = surv_likelihood(n_intervals)
    opt = optimizer or tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss_fn)
    return model

def freeze_backbone(model: tf.keras.Model, train_output_layer_only: bool = False):
    """可选：冻结除最后输出层外的所有层，用于微调场景。"""
    if not train_output_layer_only:
        return
    *body, head = model.layers[:-1], model.layers[-1]
    for layer in body:
        layer.trainable = False
    head.trainable = True

__all__ = ["build_ecg_survival_model", "freeze_backbone"]
