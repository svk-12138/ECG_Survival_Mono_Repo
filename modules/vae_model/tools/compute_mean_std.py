#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute per-lead mean and std over a folder of median ECG CSVs.
Usage:
  python modules/vae_model/tools/compute_mean_std.py \
    --data-path "/home/admin123/use/Program/ECG_Survival_Mono_Repo/data/median_sorted/keep" \
    --pattern "*.csv" \
    --output stats_mean_std.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-lead mean/std for ECG CSV dataset.")
    parser.add_argument("--data-path", type=Path, required=True, help="Root folder of CSV files.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for CSV files.")
    parser.add_argument(
        "--leads",
        nargs="+",
        default=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        help="Lead names to compute stats for.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of files for quick sanity run.")
    parser.add_argument("--output", type=Path, default=Path("stats_mean_std.json"), help="Output JSON path.")
    return parser.parse_args()


def welford_update(count: int, mean: float, m2: float, new_values: np.ndarray) -> tuple[int, float, float]:
    # Incremental mean/variance update for a batch of values.
    if new_values.size == 0:
        return count, mean, m2
    batch_count = new_values.size
    batch_mean = float(np.mean(new_values))
    batch_m2 = float(np.sum((new_values - batch_mean) ** 2))
    delta = batch_mean - mean
    new_count = count + batch_count
    new_mean = mean + delta * batch_count / new_count
    new_m2 = m2 + batch_m2 + delta * delta * count * batch_count / new_count
    return new_count, new_mean, new_m2


def main() -> None:
    args = parse_args()
    files = sorted(Path(args.data_path).glob(args.pattern))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No CSV files matched {args.pattern} under {args.data_path}")

    stats: Dict[str, Dict[str, float]] = {}
    accum: Dict[str, Dict[str, float]] = {}
    # initialize accum
    for lead in args.leads:
        accum[lead] = {"count": 0, "mean": 0.0, "m2": 0.0, "missing": 0}

    for idx, f in enumerate(files, 1):
        df = pd.read_csv(f)
        for lead in args.leads:
            if lead not in df.columns:
                accum[lead]["missing"] += 1
                continue
            arr = df[lead].to_numpy()
            arr = arr[np.isfinite(arr)]
            c, m, m2 = accum[lead]["count"], accum[lead]["mean"], accum[lead]["m2"]
            c, m, m2 = welford_update(c, m, m2, arr)
            accum[lead].update({"count": c, "mean": m, "m2": m2})
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(files)} files...")

    for lead, a in accum.items():
        if a["count"] > 1:
            var = a["m2"] / (a["count"] - 1)
            std = float(np.sqrt(var))
        else:
            var, std = 0.0, 0.0
        stats[lead] = {
            "mean": a["mean"],
            "std": std,
            "count": a["count"],
            "missing_files": a["missing"],
        }

    args.output.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved stats to {args.output}")


if __name__ == "__main__":
    main()
