#!/usr/bin/env python3
"""Quickly inspect latent vectors stored in *_latents.npz."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="View latent vectors saved by --export-latents.")
    parser.add_argument("--latents", required=True, help="Path to train_latents.npz/val_latents.npz etc.")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to print to console.")
    parser.add_argument("--output", help="Optional CSV path to save the full dataframe.")
    args = parser.parse_args()

    latents_path = Path(args.latents)
    data = np.load(latents_path, allow_pickle=True)
    df = pd.DataFrame(data["latents"], index=data["ids"])
    df.index.name = "sample_id"

    print(df.head(args.rows))
    if args.output:
        df.to_csv(args.output)
        print(f"Saved full latent table to {args.output}")


if __name__ == "__main__":
    main()
