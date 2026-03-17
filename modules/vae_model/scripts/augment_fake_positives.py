#!/usr/bin/env python3
"""
Utility to duplicate positive ECG samples so we can run experiments with a
larger proportion of终点(event=1)病例.

Example:
  python scripts/augment_fake_positives.py ^
      --dataset-dir /home/admin123/use/Program/ECG_Survival_Mono_Repo/data ^
      --labels-csv /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/labels.csv ^
      --target-positive-share 0.35 ^
      --output-labels /home/admin123/use/Program/ECG_Survival_Mono_Repo/data/labels_augmented.csv
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _resolve_csv_path(row: pd.Series, csv_col: str, csv_dir: Path, id_col: str) -> Path:
    value = str(row[csv_col])
    candidate = Path(value)
    if candidate.exists():
        return candidate
    fallback = csv_dir / f"{row[id_col]}.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"源文件不存在: {value} 或 {fallback}")


def _next_sample_id(base: str, taken: set[str], counters: Dict[str, int]) -> str:
    counters[base] += 1
    while True:
        new_id = f"{base}_fake{counters[base]:03d}"
        if new_id not in taken:
            taken.add(new_id)
            return new_id
        counters[base] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="复制 event=1 的病例，生成假的正样本以平衡数据。")
    parser.add_argument("--dataset-dir", required=True, help="数据根目录（包含 csv/ 文件夹和 labels.csv）")
    parser.add_argument("--labels-csv", required=True, help="原始 labels.csv 路径")
    parser.add_argument("--output-labels", required=True, help="写入含增广记录的 CSV 路径")
    parser.add_argument("--target-positive-share", type=float, default=0.35, help="希望的正样本占比 (0-1)")
    parser.add_argument("--id-column", default="sample_id", help="labels.csv 中的样本 ID 列")
    parser.add_argument("--label-column", default="event", help="正负标签列")
    parser.add_argument("--csv-path-column", default="csv_path", help="CSV 路径列名")
    parser.add_argument("--seed", type=int, default=1265, help="采样随机种子")
    args = parser.parse_args()

    if not (0 < args.target_positive_share < 1):
        raise ValueError("target_positive_share 必须在 (0,1) 范围内。")

    dataset_dir = Path(args.dataset_dir)
    csv_dir = dataset_dir / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"{csv_dir} 不存在。")

    labels = pd.read_csv(args.labels_csv)
    id_col = args.id_column
    label_col = args.label_column
    csv_col = args.csv_path_column

    if label_col not in labels.columns:
        raise KeyError(f"labels.csv 缺少列 {label_col}")

    positives = labels[labels[label_col] == 1].copy()
    negatives = labels[labels[label_col] == 0].copy()
    pos_count = len(positives)
    neg_count = len(negatives)
    if pos_count == 0:
        raise ValueError("没有 event=1 的样本可供复制。")

    target_total_pos = math.ceil((args.target_positive_share * neg_count) / (1 - args.target_positive_share))
    extras_needed = max(0, target_total_pos - pos_count)
    if extras_needed == 0:
        print("[augment] 当前正样本数量已达到目标占比，无需复制。")
        labels.to_csv(args.output_labels, index=False)
        return

    rng = np.random.default_rng(args.seed)
    sampled_idx = rng.choice(positives.index, size=extras_needed, replace=True)
    extras: List[pd.Series] = []
    taken_ids = set(labels[id_col].astype(str))
    counters: Dict[str, int] = defaultdict(int)

    for idx in sampled_idx:
        row = positives.loc[idx].copy()
        base_id = str(row[id_col])
        new_id = _next_sample_id(base_id, taken_ids, counters)
        source = _resolve_csv_path(row, csv_col, csv_dir, id_col)
        target = csv_dir / f"{new_id}.csv"
        shutil.copy2(source, target)
        row[id_col] = new_id
        row[csv_col] = str(target)
        extras.append(row)

    augmented = pd.concat([labels, pd.DataFrame(extras)], ignore_index=True)
    augmented = augmented.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    augmented.to_csv(args.output_labels, index=False)

    print(f"[augment] 原始正样本 {pos_count}, 负样本 {neg_count}")
    print(f"[augment] 复制生成 {len(extras)} 条假正样本 -> 新文件写入 {args.output_labels}")
    new_pos = (augmented[label_col] == 1).sum()
    print(f"[augment] 新占比：positives {new_pos}/{len(augmented)} = {new_pos / len(augmented):.3f}")


if __name__ == "__main__":
    main()
