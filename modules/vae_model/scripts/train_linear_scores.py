#!/usr/bin/env python3
"""
Train linear models on top of exported VAE latent features for either
binary classification (原流程) or连续风险预测（使用生存模型风险分数作为目标）。

Usage:
  python scripts/train_linear_scores.py \
      --latents-dir logs/MedianBeatVAE/version_0/latents \
      --labels-csv labels.csv \
      --id-column sample_id \
      --label-column outcome \
      --target-type classification
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    r2_score,
)


def load_latents(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["ids"], data["latents"]


def align_features(ids: np.ndarray, features: np.ndarray, label_map: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_idx = [i for i, sample_id in enumerate(ids) if sample_id in label_map]
    if not keep_idx:
        return np.empty((0, features.shape[1])), np.empty((0,)), np.empty((0,), dtype=object)
    keep_idx = np.array(keep_idx)
    return features[keep_idx], np.array([label_map[ids[i]] for i in keep_idx]), ids[keep_idx]


def drop_nan_rows(X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, y, ids
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask], ids[mask]


def summarize_labels(name: str, y: np.ndarray, target_type: str) -> None:
    if y.size == 0:
        print(f"[linear] {name}: 0 samples")
        return
    if target_type == "classification":
        unique, counts = np.unique(y, return_counts=True)
        summary = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
        print(f"[linear] {name} label distribution -> {summary}")
    else:
        summary = f"mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}"
        print(f"[linear] {name} regression targets -> {summary}")


def evaluate_split(
    name: str, estimator, X: np.ndarray, y: np.ndarray, target_type: str
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    if X.size == 0:
        return {"samples": 0}, {}

    metrics: Dict[str, float] = {"samples": int(len(y))}
    extras: Dict[str, np.ndarray] = {}

    if target_type == "classification":
        probs = estimator.predict_proba(X)[:, 1]
        logits = (
            estimator.decision_function(X)
            if hasattr(estimator, "decision_function")
            else np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1 - 1e-6))
        )
        metrics.update(
            {
                "auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
                "auprc": float(average_precision_score(y, probs)),
            }
        )
        extras["prob"] = probs
        extras["logit"] = logits
    else:
        preds = estimator.predict(X).astype(float)
        mse = mean_squared_error(y, preds)
        metrics.update(
            {
                "r2": float(r2_score(y, preds)) if len(y) > 1 else float("nan"),
                "mae": float(mean_absolute_error(y, preds)),
                "rmse": float(np.sqrt(mse)),
                "pearson": float(np.corrcoef(y, preds)[0, 1]) if len(y) > 1 else float("nan"),
            }
        )
        extras["prediction"] = preds
        extras["residual"] = preds - y

    return metrics, extras


def compute_ols_tvalues(
    X: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float, fit_intercept: bool = True
) -> Dict[str, np.ndarray]:
    """计算 OLS 的标准误与 t 值（基于训练集残差）。"""
    y = y.astype(float).ravel()
    n_samples, n_features = X.shape
    if fit_intercept:
        X_design = np.column_stack([np.ones(n_samples), X])
        beta = np.concatenate([[intercept], coef])
    else:
        X_design = X
        beta = coef
    y_hat = X_design @ beta
    residual = y - y_hat
    dof = n_samples - X_design.shape[1]
    if dof <= 0:
        return {
            "coef_se": np.full_like(coef, np.nan, dtype=float),
            "t_values": np.full_like(coef, np.nan, dtype=float),
            "intercept_se": np.nan,
            "intercept_t": np.nan,
            "df_resid": np.array([dof], dtype=float),
        }
    sigma2 = float((residual**2).sum() / dof)
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    se_all = np.sqrt(np.diag(sigma2 * xtx_inv))
    if fit_intercept:
        intercept_se = float(se_all[0])
        coef_se = se_all[1:]
        intercept_t = float(beta[0] / intercept_se) if intercept_se > 0 else np.nan
        t_values = coef / coef_se
    else:
        intercept_se = np.nan
        intercept_t = np.nan
        coef_se = se_all
        t_values = coef / coef_se
    return {
        "coef_se": coef_se,
        "t_values": t_values,
        "intercept_se": np.array([intercept_se], dtype=float),
        "intercept_t": np.array([intercept_t], dtype=float),
        "df_resid": np.array([dof], dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear regression T-score using VAE latents.")
    parser.add_argument("--latents-dir", required=True, help="目录，包含 train_latents.npz / val_latents.npz / test_latents.npz")
    parser.add_argument("--labels-csv", required=True, help="CSV，包含 sample_id 及标签")
    parser.add_argument("--id-column", default="sample_id", help="CSV 中 sample id 列名")
    parser.add_argument("--id-suffix", default="", help="可选：为 labels 中的 id 追加后缀以匹配潜变量文件名，例如 '_median12L'")
    parser.add_argument("--label-column", default="label", help="CSV 中标签列名（0/1）")
    parser.add_argument("--output", help="结果输出目录，默认为 latents_dir 旁")
    parser.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        default="classification",
        help="classification=基于0/1事件的逻辑回归；regression=拟合连续风险/得分",
    )
    parser.add_argument("--top-k", type=int, default=5, help="显示权重绝对值最大的潜在因子个数")
    args = parser.parse_args()

    latents_dir = Path(args.latents_dir)
    output_dir = Path(args.output or (latents_dir / "linear_scores"))
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels_csv)
    ids_col = labels_df[args.id_column].astype(str)
    if args.id_suffix:
        ids_col = ids_col + args.id_suffix
    label_map = dict(zip(ids_col, labels_df[args.label_column].astype(float)))

    train_ids, train_lat = load_latents(latents_dir / "train_latents.npz")
    val_ids, val_lat = load_latents(latents_dir / "val_latents.npz")
    test_ids, test_lat = load_latents(latents_dir / "test_latents.npz")

    X_train, y_train, train_ids = align_features(train_ids, train_lat, label_map)
    X_val, y_val, val_ids = align_features(val_ids, val_lat, label_map)
    X_test, y_test, test_ids = align_features(test_ids, test_lat, label_map)

    X_train, y_train, train_ids = drop_nan_rows(X_train, y_train, train_ids)
    X_val, y_val, val_ids = drop_nan_rows(X_val, y_val, val_ids)
    X_test, y_test, test_ids = drop_nan_rows(X_test, y_test, test_ids)

    # 提前提示对齐情况
    total_latents = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"[linear] 对齐后的样本总数: {total_latents}")
    if total_latents == 0:
        # 给出少量示例便于排查
        labels_ids_set = set(label_map.keys())
        try:
            train_ids_raw = np.load(Path(args.latents_dir) / "train_latents.npz", allow_pickle=True)["ids"].astype(str)
        except Exception:
            train_ids_raw = np.array([], dtype=str)
        miss_labels = list(labels_ids_set)[:5]
        miss_latents = list(train_ids_raw[:5])
        raise ValueError(
            "训练集中缺少有效样本，请检查 ID 是否匹配。\n"
            f"示例 labels id: {miss_labels}\n"
            f"示例 latents id: {miss_latents}\n"
            "如需追加统一后缀，请使用 --id-suffix 例如 '_median12L'"
        )

    summarize_labels("train", y_train, args.target_type)
    summarize_labels("val", y_val, args.target_type)
    summarize_labels("test", y_test, args.target_type)

    if X_train.size == 0:
        raise ValueError("训练集中缺少有效样本，无法训练线性模型。")
    if args.target_type == "classification" and len(np.unique(y_train)) < 2:
        raise ValueError("训练集中缺少有效标签或仅单一类别，无法训练分类模型。")

    if args.target_type == "classification":
        estimator: Union[LogisticRegression, LinearRegression] = LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs"
        )
    else:
        estimator = LinearRegression()

    estimator.fit(X_train, y_train)

    metrics = {}
    for split_name, X, y, ids in [
        ("train", X_train, y_train, train_ids),
        ("val", X_val, y_val, val_ids),
        ("test", X_test, y_test, test_ids),
    ]:
        if X.size == 0:
            metrics[split_name] = {"samples": 0}
            continue
        split_metrics, extras = evaluate_split(split_name, estimator, X, y, args.target_type)
        metrics[split_name] = split_metrics
        if args.target_type == "classification":
            df = pd.DataFrame(
                {
                    "sample_id": ids,
                    "label": y,
                    "logit": extras.get("logit"),
                    "prob": extras.get("prob"),
                    "split": split_name,
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "sample_id": ids,
                    "target": y,
                    "prediction": extras.get("prediction"),
                    "residual": extras.get("residual"),
                    "split": split_name,
                }
            )
        df.to_csv(output_dir / f"{split_name}_scores.csv", index=False)

    if args.target_type == "classification":
        coef_vector = estimator.coef_[0]
        intercept = estimator.intercept_.tolist()
    else:
        coef_vector = estimator.coef_.ravel()
        intercept_val = estimator.intercept_
        intercept = float(intercept_val) if np.ndim(intercept_val) == 0 else intercept_val.tolist()

    t_stats = {}
    if args.target_type == "regression":
        t_stats = compute_ols_tvalues(
            X_train, y_train, coef_vector.astype(float), float(intercept), fit_intercept=True
        )

    coef_data = {
        "target_type": args.target_type,
        "intercept": intercept,
        "coefficients": coef_vector.tolist(),
        "coef_se": t_stats.get("coef_se", np.array([])).tolist() if t_stats else [],
        "t_values": t_stats.get("t_values", np.array([])).tolist() if t_stats else [],
        "intercept_se": float(t_stats["intercept_se"][0]) if t_stats else None,
        "intercept_t": float(t_stats["intercept_t"][0]) if t_stats else None,
        "df_resid": int(t_stats["df_resid"][0]) if t_stats else None,
        "top_factors": [],
    }
    # 按 t 值筛选最具影响力的潜在因子（回归模式），分类模式仍按系数绝对值
    if args.target_type == "regression" and t_stats:
        t_vals = t_stats.get("t_values", np.array([]))
        top_sorted = sorted(
            [
                {"latent_dim": idx, "weight": float(coef_vector[idx]), "t_value": float(t_vals[idx])}
                for idx in range(len(coef_vector))
            ],
            key=lambda x: abs(x["t_value"]),
            reverse=True,
        )
        coef_data["top_factors"] = top_sorted[: args.top_k]
    else:
        coef_data["top_factors"] = sorted(
            [{"latent_dim": idx, "weight": float(weight)} for idx, weight in enumerate(coef_vector)],
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )[: args.top_k]
    with open(output_dir / "linear_model.json", "w", encoding="utf-8") as fw:
        json.dump(coef_data, fw, ensure_ascii=False, indent=2)

    metrics_payload = dict(metrics)
    metrics_payload["_meta"] = {"target_type": args.target_type}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fw:
        json.dump(metrics_payload, fw, ensure_ascii=False, indent=2)
    print(f"[linear] 结果写入 {output_dir}")


if __name__ == "__main__":
    main()
