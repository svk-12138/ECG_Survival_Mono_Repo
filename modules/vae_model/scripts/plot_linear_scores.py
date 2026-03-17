#!/usr/bin/env python3
"""Plot ROC and Precision-Recall curves from linear T-score outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def load_scores(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "prob" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} 缺少 prob/label 列")
    return df["label"].to_numpy(dtype=float), df["prob"].to_numpy(dtype=float)


def plot_curves(split: str, labels: np.ndarray, probs: np.ndarray, out_dir: Path) -> dict:
    roc_fpr, roc_tpr, _ = roc_curve(labels, probs)
    pr_prec, pr_recall, _ = precision_recall_curve(labels, probs)
    roc_auc = auc(roc_fpr, roc_tpr)
    pr_auc = auc(pr_recall, pr_prec)

    plt.figure(figsize=(6, 5))
    plt.plot(roc_fpr, roc_tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{split}_roc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(pr_recall, pr_prec, label=f"PR (AUPRC={pr_auc:.3f})", color="tomato")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{split}_pr.png", dpi=200)
    plt.close()

    np.savez(out_dir / f"{split}_curves.npz", roc_fpr=roc_fpr, roc_tpr=roc_tpr, pr_recall=pr_recall, pr_prec=pr_prec)
    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC/PR curves from linear score CSVs.")
    parser.add_argument("--scores-dir", required=True, help="目录，含 train_scores.csv / val_scores.csv / test_scores.csv")
    parser.add_argument("--output", help="输出目录，默认 scores-dir/plots")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.output or (scores_dir / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for split in ("train", "val", "test"):
        csv_path = scores_dir / f"{split}_scores.csv"
        if not csv_path.exists():
            continue
        labels, probs = load_scores(csv_path)
        metrics[split] = plot_curves(split, labels, probs, out_dir)
        print(f"[plot] {split} 曲线已保存至 {out_dir}")

    if metrics:
        pd.DataFrame(metrics).T.to_csv(out_dir / "curve_metrics.csv")
        print(f"[plot] 曲线指标写入 {out_dir / 'curve_metrics.csv'}")
    else:
        print("[plot] 未找到 score CSV，未生成曲线。")


if __name__ == "__main__":
    main()
