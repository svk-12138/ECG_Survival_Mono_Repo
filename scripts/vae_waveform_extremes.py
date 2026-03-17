#!/usr/bin/env python3
"""
Aggregate high/low AI-ECG风险波形均值，用于复现 Figure 7 风格的可解释性。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False


def _read_scores(paths: Iterable[Path], id_col: str, score_col: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if id_col not in df.columns or score_col not in df.columns:
            raise KeyError(f"{path} 缺少 {id_col}/{score_col}")
        frames.append(df[[id_col, score_col]].rename(columns={score_col: "score"}))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=[id_col, "score"])
    return merged


def _build_median_index(root: Path, glob_pattern: str, extension: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in root.glob(glob_pattern):
        if not path.is_file():
            continue
        if extension and path.suffix.lower() != extension.lower():
            continue
        index[path.stem] = path
    if not index:
        raise FileNotFoundError(f"在 {root} 未找到匹配 {glob_pattern} 的文件")
    return index


def _load_waveform(path: Path, lead_order: List[str], normalize: str) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [str(c).strip() for c in df.columns]
    if cols[0].lower() in {"", "index", "sample", "time", "t"} or cols[0].lower().startswith("unnamed"):
        values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        cols = cols[1:]
    else:
        values = df.to_numpy(dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    lookup = {name.upper(): idx for idx, name in enumerate(cols)}
    indices = []
    for lead in lead_order:
        key = lead.upper()
        if key not in lookup:
            raise KeyError(f"{path} 缺少导联 {lead}")
        indices.append(lookup[key])
    arr = values[:, indices]
    signal = arr.T
    if normalize == "zscore":
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        signal = (signal - mean) / std
    elif normalize == "minmax":
        min_v = signal.min(axis=1, keepdims=True)
        max_v = signal.max(axis=1, keepdims=True)
        denom = max_v - min_v
        denom[denom == 0] = 1.0
        signal = (signal - min_v) / denom
    return signal.astype(np.float32)


def _collect_waveforms(
    sample_ids: Iterable[str],
    index: Dict[str, Path],
    lead_order: List[str],
    normalize: str,
) -> Tuple[np.ndarray, List[str]]:
    collected = []
    missing = []
    for sid in sample_ids:
        path = index.get(str(sid))
        if not path:
            missing.append(str(sid))
            continue
        try:
            collected.append(_load_waveform(path, lead_order, normalize))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 读取 {path} 失败：{exc}")
            missing.append(str(sid))
    if not collected:
        raise RuntimeError("未能成功读取任何波形")
    return np.stack(collected), missing


def _plot_means(
    out_png: Path,
    lead_order: List[str],
    high_mean: np.ndarray,
    high_std: np.ndarray,
    low_mean: np.ndarray,
    low_std: np.ndarray,
) -> None:
    if not HAS_MPL:
        print("[warn] matplotlib 不可用，跳过绘图")
        return
    leads = high_mean.shape[0]
    cols = 2
    rows = (leads + cols - 1) // cols
    x = np.arange(high_mean.shape[-1])
    plt.figure(figsize=(cols * 4, rows * 2.5))
    for lead in range(leads):
        ax = plt.subplot(rows, cols, lead + 1)
        ax.plot(x, high_mean[lead], color="red", label="高风险均值")
        ax.fill_between(
            x,
            high_mean[lead] - high_std[lead],
            high_mean[lead] + high_std[lead],
            color="red",
            alpha=0.15,
        )
        ax.plot(x, low_mean[lead], color="blue", label="低风险均值")
        ax.fill_between(
            x,
            low_mean[lead] - low_std[lead],
            low_mean[lead] + low_std[lead],
            color="blue",
            alpha=0.15,
        )
        ax.set_title(lead_order[lead] if lead < len(lead_order) else f"Lead {lead+1}")
        ax.grid(True, alpha=0.2)
        if lead == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute high/low AI-ECG waveform averages")
    parser.add_argument("--scores-csv", nargs="+", required=True, help="train_linear_scores.py 导出的 *_scores.csv")
    parser.add_argument("--sample-id-column", default="sample_id", help="scores CSV 中 ID 列名")
    parser.add_argument("--score-column", default="prob", help="用于排序的分数字段（prob/prediction 等）")
    parser.add_argument("--top-n", type=int, default=10000, help="取最高/最低样本数量")
    parser.add_argument("--median-dir", required=True, help="median beat CSV 存放目录")
    parser.add_argument("--median-glob", default="**/*.csv", help="扫描 median CSV 的 glob 模式")
    parser.add_argument("--median-ext", default=".csv", help="过滤扩展名")
    parser.add_argument("--lead-order", nargs="+", default=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"], help="导联顺序")
    parser.add_argument("--normalize", choices=["zscore", "minmax", "none"], default="zscore", help="单条波形归一化方式")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()

    scores = _read_scores([Path(p) for p in args.scores_csv], args.sample_id_column, args.score_column)
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    top_ids = scores.head(args.top_n)[args.sample_id_column].astype(str).tolist()
    low_ids = scores.tail(args.top_n)[args.sample_id_column].astype(str).tolist()
    print(f"[waveform] 最高 {len(top_ids)} / 最低 {len(low_ids)} 个样本")

    index = _build_median_index(Path(args.median_dir), args.median_glob, args.median_ext)
    high_waveforms, high_missing = _collect_waveforms(top_ids, index, args.lead_order, args.normalize)
    low_waveforms, low_missing = _collect_waveforms(low_ids, index, args.lead_order, args.normalize)

    high_mean = high_waveforms.mean(axis=0)
    high_std = high_waveforms.std(axis=0)
    low_mean = low_waveforms.mean(axis=0)
    low_std = low_waveforms.std(axis=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "waveform_extremes.npz",
        lead_order=np.array(args.lead_order),
        top_ids=np.array(top_ids),
        low_ids=np.array(low_ids),
        high_mean=high_mean,
        high_std=high_std,
        low_mean=low_mean,
        low_std=low_std,
    )
    _plot_means(output_dir / "waveform_extremes.png", args.lead_order, high_mean, high_std, low_mean, low_std)

    summary = {
        "top_count": int(len(top_ids)),
        "low_count": int(len(low_ids)),
        "high_missing": high_missing,
        "low_missing": low_missing,
        "score_column": args.score_column,
        "normalize": args.normalize,
    }
    import json

    (output_dir / "waveform_extremes.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[waveform] 结果写入 {output_dir}")


if __name__ == "__main__":
    main()
