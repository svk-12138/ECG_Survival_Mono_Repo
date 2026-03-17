#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal survival evaluation without external cohort.

输入：
  - manifest JSON：包含 patient_id/time/event
  - risk_scores CSV：包含 sample_id,risk_score（可选完整生存曲线列）

输出：
  - metrics.json：C-index、AUROC、事件率、各风险四分位事件率与相对风险
  - merged.csv：按 sample_id 合并的评估表
  - sample_cindex.csv：可选，每个样本参与的 C-index 贡献
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def concordance_index(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    """简单 C-index 计算（不处理平分、ties 时按0.5处理）。"""
    n = len(time)
    assert n == len(event) == len(score)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 0 and event[j] == 0:
                continue
            if time[i] == time[j]:
                continue
            den += 1
            if time[i] < time[j]:
                ci, cj = i, j
            else:
                ci, cj = j, i
            if score[ci] > score[cj]:
                num += 1
            elif score[ci] == score[cj]:
                num += 0.5
    return num / den if den > 0 else float("nan")


def per_sample_cindex(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按样本统计 C-index 贡献（与整体 C-index 一致的配对规则）。"""

    n = len(time)
    counts = np.zeros(n, dtype=np.int64)
    concordant = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 0 and event[j] == 0:
                continue
            if time[i] == time[j]:
                continue
            counts[i] += 1
            counts[j] += 1
            if time[i] < time[j]:
                ci, cj = i, j
            else:
                ci, cj = j, i
            if score[ci] > score[cj]:
                concordant[i] += 1
                concordant[j] += 1
            elif score[ci] == score[cj]:
                concordant[i] += 0.5
                concordant[j] += 0.5
    sample_cindex = np.full(n, float("nan"), dtype=np.float64)
    valid = counts > 0
    sample_cindex[valid] = concordant[valid] / counts[valid]
    return counts, concordant, sample_cindex


def quartile_stats(df: pd.DataFrame) -> List[dict]:
    qs = []
    for q in range(4):
        part = df[df["risk_quartile"] == q]
        if part.empty:
            qs.append({"quartile": q, "n": 0})
            continue
        ev = part["event"].sum()
        n = len(part)
        qs.append(
            {
                "quartile": q,
                "n": int(n),
                "event": int(ev),
                "event_rate": float(ev / n),
                "time_mean": float(part["time"].mean()),
                "time_median": float(part["time"].median()),
            }
        )
    return qs


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate survival risk scores on internal cohort.")
    p.add_argument("--manifest", required=True, help="JSON manifest with patient_id/time/event")
    p.add_argument("--scores", required=True, help="CSV with sample_id,risk_score")
    p.add_argument("--id-column", default="patient_id", help="ID column in manifest (default patient_id)")
    p.add_argument("--score-id-column", default="sample_id", help="ID column in scores file")
    p.add_argument("--time-column", default="time", help="Time column in manifest")
    p.add_argument("--event-column", default="event", help="Event column in manifest (1/0)")
    p.add_argument("--output-dir", default="outputs/analysis/survival_eval", help="Directory to save outputs")
    p.add_argument("--per-sample-output", default=None, help="输出每样本 C-index 贡献 CSV（可选）")
    p.add_argument("--per-sample-top", type=int, default=0, help="仅保留前 N 个样本（0=全部）")
    args = p.parse_args()

    manifest = pd.read_json(args.manifest)
    scores = pd.read_csv(args.scores)

    if args.id_column not in manifest.columns:
        raise ValueError(f"manifest 缺少 ID 列: {args.id_column}")
    if args.score_id_column not in scores.columns:
        raise ValueError(f"scores 缺少 ID 列: {args.score_id_column}")
    if "risk_score" not in scores.columns:
        raise ValueError("scores 缺少 risk_score 列")

    merged = pd.merge(
        manifest[[args.id_column, args.time_column, args.event_column]],
        scores[[args.score_id_column, "risk_score"]],
        left_on=args.id_column,
        right_on=args.score_id_column,
        how="inner",
    )
    merged.rename(
        columns={
            args.id_column: "sample_id",
            args.time_column: "time",
            args.event_column: "event",
        },
        inplace=True,
    )
    if merged.empty:
        raise ValueError("合并后样本数为 0，请检查 ID 对齐。")

    merged["risk_quartile"] = pd.qcut(merged["risk_score"], 4, labels=False, duplicates="drop")

    time = merged["time"].to_numpy()
    event = merged["event"].to_numpy()
    score = merged["risk_score"].to_numpy()

    cidx = concordance_index(time, event, score)
    auroc = roc_auc_score(event, score) if len(np.unique(event)) > 1 else float("nan")

    qs = quartile_stats(merged)
    rr = None
    if len(qs) == 4 and qs[0].get("event_rate") is not None and qs[-1].get("event_rate") is not None:
        low = qs[0]["event_rate"]
        high = qs[-1]["event_rate"]
        rr = float(high / low) if low > 0 else float("inf")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "merged.csv", index=False)
    metrics = {
        "n": int(len(merged)),
        "events": int(merged["event"].sum()),
        "event_rate": float(merged["event"].mean()),
        "c_index": float(cidx),
        "auroc": float(auroc),
        "quartiles": qs,
        "risk_ratio_q4_over_q1": rr,
    }
    if args.per_sample_output or args.per_sample_top:
        counts, concordant, sample_cidx = per_sample_cindex(time, event, score)
        merged["pair_count"] = counts
        merged["concordant_sum"] = concordant
        merged["sample_cindex"] = sample_cidx
        per_sample = merged.sort_values("sample_cindex", ascending=False, na_position="last")
        if args.per_sample_top and args.per_sample_top > 0:
            per_sample = per_sample.head(args.per_sample_top)
        per_path = Path(args.per_sample_output) if args.per_sample_output else (out_dir / "sample_cindex.csv")
        per_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample.to_csv(per_path, index=False)
        metrics["sample_cindex_output"] = str(per_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[eval] 完成，样本={metrics['n']}，事件={metrics['events']}，"
        f"C-index={metrics['c_index']:.3f}, AUROC={metrics['auroc']:.3f}"
    )


if __name__ == "__main__":
    main()
