import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


LEADS_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _stem_without_suffix(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_median"):
        return stem[: -len("_median")]
    return stem


def build_csv_index(csv_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in csv_dirs:
        for path in root.glob("*.csv"):
            base = _stem_without_suffix(path)
            patient_id = base.split(".")[-1]
            if patient_id not in index:
                index[patient_id] = path
    return index


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_signal(path: Path, leads: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "time_ms" in df.columns:
        time_ms = df["time_ms"].to_numpy(dtype=np.float32)
    else:
        time_ms = np.arange(len(df), dtype=np.float32)
    cols = {c.upper(): c for c in df.columns}
    data = []
    for lead in leads:
        col = cols.get(lead.upper())
        if col is None:
            raise KeyError(f"{path} missing lead {lead}")
        data.append(df[col].to_numpy(dtype=np.float32))
    signal = np.stack(data, axis=1)
    return time_ms, signal


def resample_signal(
    time_ms: np.ndarray, signal: np.ndarray, target_len: int, time_template: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if signal.shape[0] == target_len:
        return time_ms, signal
    resampled = np.empty((target_len, signal.shape[1]), dtype=np.float32)
    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(time_template, time_ms, signal[:, i]).astype(np.float32)
    return time_template, resampled


def most_common_length(paths: list[Path], leads: list[str]) -> int:
    counts = Counter()
    for path in paths:
        try:
            _, signal = load_signal(path, leads)
        except KeyError:
            continue
        counts[signal.shape[0]] += 1
    if not counts:
        raise RuntimeError("No valid CSVs found for length check.")
    return counts.most_common(1)[0][0]


def build_training_arrays(
    samples: list[dict],
    csv_index: dict[str, Path],
    leads: list[str],
    target_len: int,
    time_template: np.ndarray,
    max_time: float,
    latent_dim: int,
    rng: np.random.Generator,
    max_train_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    time_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    time_norm = (time_template - float(time_template.min())) / float(time_template.max() - time_template.min())

    lead_X: dict[str, list[np.ndarray]] = {lead: [] for lead in leads}
    lead_y: dict[str, list[np.ndarray]] = {lead: [] for lead in leads}

    for sample in samples:
        patient_id = sample["patient_id"]
        event = float(sample["event"])
        outcome_time = float(sample["time"])
        time_out_norm = outcome_time / max_time if max_time > 0 else outcome_time
        latent = rng.random(latent_dim, dtype=np.float32)

        path = csv_index[patient_id]
        time_ms, signal = load_signal(path, leads)
        time_ms, signal = resample_signal(time_ms, signal, target_len, time_template)

        base = np.column_stack(
            [
                time_idx,
                time_norm,
                np.full(target_len, event, dtype=np.float32),
                np.full(target_len, time_out_norm, dtype=np.float32),
                np.tile(latent, (target_len, 1)),
            ]
        )
        for i, lead in enumerate(leads):
            lead_X[lead].append(base)
            lead_y[lead].append(signal[:, i])

    X_full = {lead: np.vstack(lead_X[lead]) for lead in leads}
    y_full = {lead: np.concatenate(lead_y[lead]) for lead in leads}
    stats = {
        lead: {
            "min": float(np.min(y_full[lead])),
            "max": float(np.max(y_full[lead])),
            "std": float(np.std(y_full[lead])) if float(np.std(y_full[lead])) > 0 else 1.0,
        }
        for lead in leads
    }
    if max_train_samples and max_train_samples > 0:
        X = {}
        y = {}
        for lead in leads:
            n = X_full[lead].shape[0]
            if n > max_train_samples:
                idx = rng.choice(n, size=max_train_samples, replace=False)
                X[lead] = X_full[lead][idx]
                y[lead] = y_full[lead][idx]
            else:
                X[lead] = X_full[lead]
                y[lead] = y_full[lead]
    else:
        X = X_full
        y = y_full
    return X, y, stats


def train_models(
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    leads: list[str],
    n_estimators: int,
    max_depth: int,
    seed: int,
) -> dict[str, RandomForestRegressor]:
    models: dict[str, RandomForestRegressor] = {}
    for lead in leads:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X[lead], y[lead])
        models[lead] = model
    return models


def synthesize_sample(
    models: dict[str, RandomForestRegressor],
    leads: list[str],
    time_template: np.ndarray,
    target_len: int,
    event: int,
    outcome_time: float,
    max_time: float,
    latent_dim: int,
    stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    time_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    time_norm = (time_template - float(time_template.min())) / float(time_template.max() - time_template.min())
    time_out_norm = outcome_time / max_time if max_time > 0 else outcome_time
    latent = rng.random(latent_dim, dtype=np.float32)

    base = np.column_stack(
        [
            time_idx,
            time_norm,
            np.full(target_len, float(event), dtype=np.float32),
            np.full(target_len, float(time_out_norm), dtype=np.float32),
            np.tile(latent, (target_len, 1)),
        ]
    )

    outputs = []
    for lead in leads:
        pred = models[lead].predict(base).astype(np.float32)
        lead_stats = stats[lead]
        noise = rng.normal(0.0, 0.02 * lead_stats["std"], size=pred.shape).astype(np.float32)
        pred = pred + noise
        pred = np.clip(pred, lead_stats["min"], lead_stats["max"]).astype(np.float32)
        outputs.append(pred)
    return np.stack(outputs, axis=1)


def plot_samples(csv_paths: list[Path], out_dir: Path, leads: list[str]) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in csv_paths:
        df = pd.read_csv(path)
        time_ms = df["time_ms"] if "time_ms" in df.columns else np.arange(len(df))
        fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
        for ax, lead in zip(axes, leads):
            ax.plot(time_ms, df[lead], linewidth=1.0)
            ax.set_title(lead)
        for ax in axes[len(leads) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ECG median-beat CSVs with Random Forests.")
    parser.add_argument("--sorted-cindex", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--plots", type=int, default=12)
    parser.add_argument("--file-prefix", type=str, default="rf")
    parser.add_argument("--file-suffix", type=str, default="_median")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Downsample training rows per lead.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    csv_dirs = [p for p in args.csv_dirs if p.exists()]
    if not csv_dirs:
        raise SystemExit("No CSV directories found.")

    csv_index = build_csv_index(csv_dirs)
    manifest = load_manifest(args.manifest)
    manifest_map = {str(item["patient_id"]): item for item in manifest}

    base_ids = sorted(set(manifest_map) & set(csv_index))
    base_entries = [manifest_map[pid] for pid in base_ids]
    pos_entries = [s for s in base_entries if int(s["event"]) == 1]
    neg_entries = [s for s in base_entries if int(s["event"]) == 0]

    if not base_entries:
        raise SystemExit("No base entries found after intersecting manifest and CSVs.")

    max_time = max(float(item["time"]) for item in base_entries)

    cidx_df = pd.read_csv(args.sorted_cindex)
    if "sample_id" not in cidx_df.columns:
        raise SystemExit("sorted-cindex CSV missing sample_id column.")
    sorted_ids = [str(v) for v in cidx_df["sample_id"].tolist()]

    training_samples = []
    missing = []
    for pid in sorted_ids:
        if pid in csv_index and pid in manifest_map:
            training_samples.append(
                {
                    "patient_id": pid,
                    "event": manifest_map[pid]["event"],
                    "time": manifest_map[pid]["time"],
                }
            )
            if len(training_samples) >= args.top_n:
                break
        else:
            missing.append(pid)

    if not training_samples:
        raise SystemExit("No training samples found from sorted-cindex list.")

    training_paths = [csv_index[s["patient_id"]] for s in training_samples]
    target_len = most_common_length(training_paths, LEADS_8)

    first_time, _ = load_signal(training_paths[0], LEADS_8)
    time_template = np.linspace(float(first_time.min()), float(first_time.max()), target_len, dtype=np.float32)

    X, y, stats = build_training_arrays(
        training_samples,
        csv_index,
        LEADS_8,
        target_len,
        time_template,
        max_time,
        args.latent_dim,
        rng,
        args.max_train_samples,
    )
    models = train_models(X, y, LEADS_8, args.n_estimators, args.max_depth, args.seed)

    base_count = len(base_entries)
    target_total = int(round(base_count * args.multiplier))
    target_total = max(target_total, base_count)
    synth_count = target_total - base_count

    if synth_count == 0:
        raise SystemExit("Multiplier results in zero synthetic samples.")

    pos_synth = int(round(len(pos_entries) * (args.multiplier - 1)))
    pos_synth = max(0, min(pos_synth, synth_count))
    neg_synth = synth_count - pos_synth

    output_dir = args.output_dir
    synth_csv_dir = output_dir / "generated" / "csv"
    synth_csv_dir.mkdir(parents=True, exist_ok=True)

    synthetic_manifest = []
    mapping_rows = []

    def pick_entries(pool: list[dict], count: int) -> list[dict]:
        idx = rng.integers(0, len(pool), size=count)
        return [pool[i] for i in idx]

    pos_picks = pick_entries(pos_entries, pos_synth) if pos_synth > 0 else []
    neg_picks = pick_entries(neg_entries, neg_synth) if neg_synth > 0 else []
    picks = pos_picks + neg_picks
    rng.shuffle(picks)

    start_id = 9_000_000
    for i, src in enumerate(picks, start=0):
        patient_id = str(start_id + i)
        event = int(src["event"])
        outcome_time = float(src["time"])
        signal = synthesize_sample(
            models,
            LEADS_8,
            time_template,
            target_len,
            event,
            outcome_time,
            max_time,
            args.latent_dim,
            stats,
            rng,
        )
        df = pd.DataFrame(signal, columns=LEADS_8)
        df.insert(0, "time_ms", time_template)
        file_name = f"{args.file_prefix}.{patient_id}{args.file_suffix}.csv"
        out_path = synth_csv_dir / file_name
        df.to_csv(out_path, index=False)
        synthetic_manifest.append({"patient_id": patient_id, "time": outcome_time, "event": event})
        mapping_rows.append(
            {
                "synthetic_id": patient_id,
                "source_id": src["patient_id"],
                "event": event,
                "time": outcome_time,
            }
        )

    synthetic_manifest_path = output_dir / "synthetic_manifest.json"
    synthetic_manifest_path.write_text(
        json.dumps(synthetic_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    mapping_path = output_dir / "synthetic_mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    merged_dir = output_dir / "dataset_2x_full"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for pid in base_ids:
        src_path = csv_index[pid]
        (merged_dir / src_path.name).write_bytes(src_path.read_bytes())
    for path in synth_csv_dir.glob("*.csv"):
        (merged_dir / path.name).write_bytes(path.read_bytes())

    merged_manifest = base_entries + synthetic_manifest
    merged_manifest_path = output_dir / "train_manifest_aug_full.json"
    merged_manifest_path.write_text(
        json.dumps(merged_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    report = {
        "base_samples": base_count,
        "synthetic_samples": len(synthetic_manifest),
        "target_total": target_total,
        "pos_base": len(pos_entries),
        "neg_base": len(neg_entries),
        "pos_synth": pos_synth,
        "neg_synth": neg_synth,
        "top_n": len(training_samples),
        "missing_from_sorted": len(missing),
        "target_len": target_len,
        "leads": LEADS_8,
    }
    report_path = output_dir / "seed_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    if args.plots > 0:
        synth_paths = list(synth_csv_dir.glob("*.csv"))
        if synth_paths:
            pick_n = min(args.plots, len(synth_paths))
            picks = rng.choice(synth_paths, size=pick_n, replace=False)
            plot_samples(list(picks), output_dir / "visuals", LEADS_8)

    print(f"[OK] base={base_count} synth={len(synthetic_manifest)} total={len(merged_manifest)}")
    print(f"[OK] synthetic CSVs: {synth_csv_dir}")
    print(f"[OK] merged dataset: {merged_dir}")
    print(f"[OK] manifest: {merged_manifest_path}")


if __name__ == "__main__":
    main()
