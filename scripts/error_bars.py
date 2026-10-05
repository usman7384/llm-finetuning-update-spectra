#!/usr/bin/env python
"""95% bootstrap intervals for the headline spectral numbers.

Every number in the report is a mean over weight matrices. Matrices in the same
layer aren't independent, so this resamples whole layers (block bootstrap)
rather than single matrices, which gives wider and more honest intervals.

For "how many times below noise" the ratio is bootstrapped too: the same
resampled layers are used for the run and for its noise baseline.

Only reads CSVs that other scripts already wrote, so it takes seconds.

    uv run python scripts/error_bars.py

Writes results/error_bars.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 5000
# (a, b) pairs reported as the paired difference b - a
PAIRS = [
    ("wikitext LM (vs base)", "sft (vs base)"),
    ("sft (vs base)", "python code LM (vs base)"),
    ("wikitext LM (vs base)", "python code LM (vs base)"),
    ("sft (vs base)", "sft seed 43 (vs base)"),
    ("wikitext LM (vs base)", "wikitext LM seed 43"),
    ("python code LM (vs base)", "python code LM seed 43"),
]
rng = np.random.default_rng(0)


def layer_boot(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """N_BOOT x len(cols) array of means over resampled layers."""
    df = df[df["layer_idx"].notna()]
    by_layer = df.groupby("layer_idx")[cols]
    sums = by_layer.sum().to_numpy()
    counts = by_layer.count().to_numpy()
    n = len(sums)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def summarize(label: str, metric: str, df: pd.DataFrame, col: str, noise_col: str | None):
    cols = [col] + ([noise_col] if noise_col else [])
    boot = layer_boot(df, cols)
    row = {
        "source": label,
        "metric": metric,
        "mean": df[col].mean(),
        "lo": np.percentile(boot[:, 0], 2.5),
        "hi": np.percentile(boot[:, 0], 97.5),
    }
    if noise_col:
        ratio = boot[:, 1] / boot[:, 0]
        row.update(
            noise_over_real=df[noise_col].mean() / df[col].mean(),
            ratio_lo=np.percentile(ratio, 2.5),
            ratio_hi=np.percentile(ratio, 97.5),
        )
    return row


def with_noise(runs: pd.DataFrame, noise: pd.DataFrame, value: str) -> pd.DataFrame:
    """Attach the random-init pair's value for the same matrix as a column."""
    n = noise[["key", value]].rename(columns={value: f"noise_{value}"})
    return runs.merge(n, on="key")


def main() -> int:
    rows = []

    families = [("results/stable_rank.csv", "qwen"), ("results/stable_rank_smollm2.csv", "smollm2")]
    for csv, family in families:
        if not Path(csv).exists():
            continue
        df = pd.read_csv(csv)
        noise = df[df["source"] == "random init pair"]
        for source, sub in df.groupby("source", sort=False):
            if source == "random init pair":
                label = f"{family}: {source}"
                rows.append(summarize(label, "stable_rank_frac", sub, "stable_rank_frac", None))
                continue
            sub = with_noise(sub, noise, "stable_rank_frac")
            rows.append(summarize(f"{family}: {source}", "stable_rank_frac", sub,
                                  "stable_rank_frac", "noise_stable_rank_frac"))  # fmt: skip
            rows.append(summarize(f"{family}: {source}", "energy_top1", sub, "energy_top1", None))

        # paired: same matrices in both runs, so layer-to-layer spread cancels
        sr = df.pivot_table(index=["key", "layer_idx"], columns="source", values="stable_rank_frac")
        sr = sr.reset_index()
        for a, b in PAIRS:
            if a in sr and b in sr:
                diff = pd.DataFrame({"layer_idx": sr["layer_idx"], "d": sr[b] - sr[a]})
                rows.append(
                    summarize(f"{family}: {b} - {a}", "stable_rank_frac diff", diff, "d", None)
                )

    for path in sorted(Path("results/released_pairs").glob("*.csv")):
        df = pd.read_csv(path)
        rows.append(summarize(f"released: {path.stem}", "stable_rank_frac", df,
                              "stable_rank_frac", "noise_stable_rank_frac"))  # fmt: skip

    for path in sorted(Path("results/timber_check").glob("*.csv")):
        df = pd.read_csv(path)
        rows.append(summarize(f"timber: {path.stem}", "delta_erank_frac", df,
                              "delta_erank_frac", "noise_erank_frac"))  # fmt: skip

    out = pd.DataFrame(rows)
    out.to_csv("results/error_bars.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print("\nwrote results/error_bars.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
