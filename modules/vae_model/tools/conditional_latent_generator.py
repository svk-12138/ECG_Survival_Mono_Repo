#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic ECG median-beat CSVs by sampling VAE latents conditioned on
event/time buckets. This is a lightweight conditional generator in latent space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import VAEXperiment  # noqa: E402
from models import vae_models  # noqa: E402


@dataclass
class GroupStats:
    mean: np.ndarray
    std: np.ndarray
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional latent sampler (event/time).")
    parser.add_argument("--latents", type=Path, required=True, help="NPZ file from extract_latent_features.py")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON with patient_id/time/event")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for synthetic data")
    parser.add_argument("--config", type=Path, required=True, help="VAE config YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VAE checkpoint .ckpt")
    parser.add_argument("--id-column", default="patient_id", help="ID column name in manifest")
    parser.add_argument("--time-column", default="time", help="Time column name in manifest")
    parser.add_argument("--event-column", default="event", help="Event column name in manifest")
    parser.add_argument("--id-strip-suffix", default="", help="Optional suffix to strip from latent IDs")
    parser.add_argument("--time-bins", default="", help="Comma-separated time bin edges (e.g., 0,1,3,5,10)")
    parser.add_argument("--time-quantiles", type=int, default=0, help="Use N quantile bins if > 0")
    parser.add_argument("--min-group", type=int, default=10, help="Min samples per group before fallback")
    parser.add_argument("--multiplier", type=float, default=0.0, help="New samples = count * multiplier")
    parser.add_argument("--per-group", type=int, default=0, help="Fixed number of new samples per group")
    parser.add_argument("--max-total", type=int, default=0, help="Cap total synthetic samples if > 0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=64, help="Decode batch size")
    parser.add_argument("--latents-only", action="store_true", help="Only generate latent NPZ and manifest")
    return parser.parse_args()


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_model(config: dict, checkpoint: Path, device: torch.device) -> VAEXperiment:
    model = vae_models[config["model_params"]["name"]](**config["model_params"])
    experiment = VAEXperiment.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        vae_model=model,
        params=config["exp_params"],
        map_location=device,
    )
    experiment.eval()
    experiment.to(device)
    return experiment


def _parse_bins(text: str) -> List[float]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def _bin_times(times: np.ndarray, bins: List[float]) -> np.ndarray:
    if not bins:
        return np.zeros_like(times, dtype=np.int64)
    edges = np.array(sorted(bins), dtype=np.float32)
    return np.digitize(times, edges, right=False).astype(np.int64)


def _quantile_bins(times: np.ndarray, n_bins: int) -> List[float]:
    if n_bins <= 1:
        return []
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(times, qs).astype(np.float32)
    edges = np.unique(edges)
    return edges.tolist()


def _group_stats(latents: np.ndarray, indices: np.ndarray) -> GroupStats:
    subset = latents[indices]
    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std[std == 0] = 1e-6
    return GroupStats(mean=mean, std=std, count=int(len(indices)))


def _build_index_map(manifest: List[dict], id_col: str) -> Dict[str, dict]:
    return {str(row[id_col]): row for row in manifest if id_col in row}


def _write_csv(path: Path, signal: np.ndarray, lead_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("idx," + ",".join(lead_names) + "\n")
        for i in range(signal.shape[1]):
            row = [str(i)] + [f"{float(signal[ch, i]):.6f}" for ch in range(signal.shape[0])]
            f.write(",".join(row) + "\n")


def main() -> None:
    args = parse_args()
    if args.multiplier <= 0 and args.per_group <= 0:
        raise ValueError("Provide --multiplier or --per-group to generate samples.")

    npz = np.load(args.latents, allow_pickle=True)
    ids = np.array(npz["ids"]).astype(str)
    latents = np.array(npz["latents"], dtype=np.float32)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    id_map = _build_index_map(manifest, args.id_column)

    matched_ids: List[str] = []
    matched_latents: List[np.ndarray] = []
    times: List[float] = []
    events: List[int] = []
    skipped = 0
    for _id, z in zip(ids, latents):
        clean_id = _id
        if args.id_strip_suffix and clean_id.endswith(args.id_strip_suffix):
            clean_id = clean_id[: -len(args.id_strip_suffix)]
        row = id_map.get(clean_id)
        if row is None:
            skipped += 1
            continue
        matched_ids.append(clean_id)
        matched_latents.append(z)
        times.append(float(row[args.time_column]))
        events.append(int(row[args.event_column]))

    if not matched_latents:
        raise RuntimeError("No latents matched manifest IDs. Check id mapping.")

    latents = np.stack(matched_latents, axis=0)
    times_arr = np.array(times, dtype=np.float32)
    events_arr = np.array(events, dtype=np.int64)

    bins = _parse_bins(args.time_bins)
    if args.time_quantiles > 0:
        bins = _quantile_bins(times_arr, args.time_quantiles)
    bin_ids = _bin_times(times_arr, bins)

    rng = np.random.default_rng(args.seed)
    groups: Dict[Tuple[int, int], np.ndarray] = {}
    for idx, (ev, b) in enumerate(zip(events_arr, bin_ids)):
        groups.setdefault((int(ev), int(b)), []).append(idx)
    groups = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}

    global_stats = _group_stats(latents, np.arange(len(latents)))
    event_stats: Dict[int, GroupStats] = {}
    for ev in np.unique(events_arr):
        event_stats[int(ev)] = _group_stats(latents, np.where(events_arr == ev)[0])

    samples: List[Tuple[str, int, float, int, np.ndarray]] = []
    for (ev, b), idxs in groups.items():
        n_base = len(idxs)
        if args.per_group > 0:
            n_new = args.per_group
        else:
            n_new = int(round(n_base * args.multiplier))
        if n_new <= 0:
            continue

        stats = _group_stats(latents, idxs)
        if stats.count < args.min_group:
            stats = event_stats.get(ev, global_stats)
            if stats.count < args.min_group:
                stats = global_stats

        group_times = times_arr[idxs]
        for i in range(n_new):
            z = stats.mean + stats.std * rng.standard_normal(stats.mean.shape) * args.temperature
            sample_id = f"syn_e{ev}_b{b}_{i:04d}"
            time_value = float(rng.choice(group_times))
            samples.append((sample_id, ev, time_value, b, z.astype(np.float32)))

    if args.max_total and len(samples) > args.max_total:
        samples = samples[: args.max_total]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_ids = np.array([s[0] for s in samples])
    syn_latents = np.stack([s[4] for s in samples], axis=0) if samples else np.empty((0, latents.shape[1]))
    np.savez(out_dir / "synthetic_latents.npz", ids=syn_ids, latents=syn_latents)

    syn_manifest = []
    for sample_id, ev, time_value, b, _ in samples:
        syn_manifest.append(
            {
                "patient_id": sample_id,
                "time": time_value,
                "event": int(ev),
                "meta": {"synthetic": True, "time_bin": int(b)},
            }
        )
    (out_dir / "synthetic_manifest.json").write_text(
        json.dumps(syn_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "matched": len(matched_ids),
        "skipped": skipped,
        "bins": bins,
        "groups": {f"{k[0]}_{k[1]}": int(len(v)) for k, v in groups.items()},
        "generated": len(samples),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.latents_only or not samples:
        print(f"[DONE] Wrote synthetic latents to {out_dir}")
        return

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = init_model(cfg, args.checkpoint, device)
    lead_order = cfg["data_params"].get("lead_order") or [f"Lead{i+1}" for i in range(cfg["model_params"]["in_channels"])]

    csv_dir = out_dir / "csv"
    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            z = torch.from_numpy(np.stack([s[4] for s in chunk])).to(device)
            recon = experiment.model.decode(z).detach().cpu().numpy()
            for (sample_id, _, _, _, _), signal in zip(chunk, recon):
                _write_csv(csv_dir / f"{sample_id}.csv", signal, lead_order)

    print(f"[DONE] Wrote synthetic CSVs to {csv_dir}")


if __name__ == "__main__":
    main()
