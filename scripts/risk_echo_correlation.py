#!/usr/bin/env python3
"""
Compute correlations between AI-ECG 风险分数与超声参数，用于复现论文中的影像关联分析。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats

    HAS_SCIPY = True
except Exception:  # noqa: BLE001
    HAS_SCIPY = False


def _load_scores(paths: List[Path], id_col: str, score_col: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if id_col not in df.columns or score_col not in df.columns:
            raise KeyError(f"{path} 缺少 {id_col}/{score_col}")
        frames.append(df[[id_col, score_col]].rename(columns={score_col: "risk_score"}))
    merged = pd.concat(frames, ignore_index=True).dropna(subset=[id_col, "risk_score"])
    return merged


def _prepare_matrix(
    df: pd.DataFrame,
    target_col: str,
    covariates: List[str],
    min_samples: int,
) -> Optional[tuple[np.ndarray, np.ndarray, List[str]]]:
    subset = df[[target_col, "risk_score", *covariates]].dropna()
    if len(subset) < min_samples:
        return None
    y = subset[target_col].to_numpy(dtype=np.float64)
    matrix_cols = ["intercept", "risk_score", *covariates]
    X_cols = [np.ones(len(subset)), subset["risk_score"].to_numpy(dtype=np.float64)]
    for cov in covariates:
        X_cols.append(subset[cov].to_numpy(dtype=np.float64))
    X = np.column_stack(X_cols)
    return X, y, matrix_cols


def _fit_linear(X: np.ndarray, y: np.ndarray) -> dict:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    n, p = X.shape
    rss = float((resid**2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    df = max(n - p, 1)
    sigma2 = rss / df
    xtx_inv = np.linalg.pinv(X.T @ X)
    cov_beta = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se
    if HAS_SCIPY:
        pvals = 2 * stats.t.sf(np.abs(t_stats), df)
    else:
        pvals = np.full_like(t_stats, np.nan, dtype=float)
    return {
        "beta": beta,
        "se": se,
        "t": t_stats,
        "p": pvals,
        "r2": r2,
        "n": n,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm = x - x.mean()
    ym = y - y.mean()
    numerator = float((xm * ym).sum())
    denom = float(np.sqrt((xm**2).sum() * (ym**2).sum()))
    r = numerator / denom if denom > 0 else float("nan")
    if HAS_SCIPY:
        _, p = stats.pearsonr(x, y)
    else:
        p = float("nan")
    return r, p


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-ECG risk vs echocardiography correlation")
    parser.add_argument("--scores-csv", nargs="+", required=True, help="风险得分 CSV（train_linear_scores 输出）")
    parser.add_argument("--score-column", default="prob", help="scores CSV 中的风险列 (prob/prediction)")
    parser.add_argument("--echo-csv", required=True, help="包含 echocardiography 指标的表格")
    parser.add_argument("--id-column", default="sample_id", help="scores/echo 共有的 ID 列名")
    parser.add_argument("--target-columns", nargs="+", required=True, help="需要分析的 echocardiography 字段")
    parser.add_argument("--covariates", nargs="*", default=["age", "sex"], help="可选协变量")
    parser.add_argument("--min-samples", type=int, default=200, help="每个指标最少样本数")
    parser.add_argument("--output-dir", required=True, help="输出结果目录")
    args = parser.parse_args()

    scores = _load_scores([Path(p) for p in args.scores_csv], args.id_column, args.score_column)
    echo_df = pd.read_csv(args.echo_csv)
    if args.id_column not in echo_df.columns:
        raise KeyError(f"{args.echo_csv} 缺少 {args.id_column}")

    merged = pd.merge(echo_df, scores, on=args.id_column, how="inner")
    if merged.empty:
        raise RuntimeError("scores 与 echo 数据没有重叠样本")

    results = []
    for target in args.target_columns:
        if target not in merged.columns:
            print(f"[warn] {target} 不存在，跳过")
            continue
        valid = merged.dropna(subset=[target, "risk_score"])
        if len(valid) < args.min_samples:
            print(f"[warn] {target} 有效样本 {len(valid)} < {args.min_samples}")
            continue
        pearson_r, pearson_p = _pearson(valid["risk_score"].to_numpy(float), valid[target].to_numpy(float))
        covariates = [c for c in args.covariates if c in merged.columns]
        matrix = _prepare_matrix(merged, target, covariates, args.min_samples)
        if matrix is None:
            print(f"[warn] {target} 协变量后样本不足，跳过回归")
            continue
        X, y, matrix_cols = matrix
        fit = _fit_linear(X, y)
        score_idx = matrix_cols.index("risk_score")
        results.append(
            {
                "target": target,
                "samples": int(fit["n"]),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "coef_risk": float(fit["beta"][score_idx]),
                "coef_p": float(fit["p"][score_idx]),
                "coef_se": float(fit["se"][score_idx]),
                "r2": float(fit["r2"]),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "risk_echo_correlation.csv", index=False)
        (out_dir / "risk_echo_correlation.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[corr] 结果写入 {out_dir}")
    else:
        print("[corr] 未生成任何结果")


if __name__ == "__main__":
    main()
