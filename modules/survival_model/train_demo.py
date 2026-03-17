# -*- coding: utf-8 -*-
"""ECG+生存分析示例训练脚本（dry-run）。
- 随机生成假数据，仅用于验证网络/损失 wiring 是否正常。
- 真实训练时，替换成真实 ECG 波形与生存标签即可。"""
import argparse
import numpy as np
import tensorflow as tf

from ecg_survival.model_def import build_ecg_survival_model, freeze_backbone
from ecg_survival.data_utils import SurvivalBreaks, demo_fake_targets


def parse_args():
    parser = argparse.ArgumentParser(description="ECG+生存分析示例训练/dry-run")
    parser.add_argument("--n_intervals", type=int, default=8, help="离散时间区间数量")
    parser.add_argument("--max_time", type=float, default=365.0, help="最大随访时间（与断点上限一致）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch", type=int, default=4, help="示例 batch 大小（仅 dry-run）")
    parser.add_argument("--inspect", action="store_true", help="仅打印模型结构")
    parser.add_argument("--freeze_head_only", action="store_true", help="冻结除输出层外的 backbone")
    return parser.parse_args()


def main():
    args = parse_args()
    breaks = SurvivalBreaks.from_uniform(args.max_time, args.n_intervals)
    model = build_ecg_survival_model(args.n_intervals, lr=args.lr)

    if args.freeze_head_only:
        freeze_backbone(model, train_output_layer_only=True)
        model.compile(optimizer=model.optimizer, loss=model.loss)

    if args.inspect:
        model.summary()
        return

    x = np.random.randn(args.batch, 4096, 12).astype("float32")
    y = demo_fake_targets(args.batch, breaks).astype("float32")

    hist = model.fit(x, y, epochs=1, batch_size=args.batch, verbose=1)
    print("训练完成（dry-run），loss=", hist.history["loss"][-1])


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
