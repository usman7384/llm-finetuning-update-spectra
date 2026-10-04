#!/usr/bin/env python
"""How concentrated is each weight delta, measured without a threshold?

The first pass counted singular values above 0.01 * sigma_1 and got ~99% of
max rank for everything, real deltas and random noise alike. That count can't
tell how the energy is spread among the values it keeps, so these two measures
are used instead:

    stable rank = sum(sigma_i^2) / sigma_1^2
    energy@k    = share of sum(sigma_i^2) sitting in the top k values

Both ignore overall scale, which matters: the random-init pair differs by a
Frobenius norm ~126x bigger than the SFT delta.

Only reads the spectra svd_deltas.py already saved, so no GPU needed.

    uv run python scripts/stable_rank.py

Writes results/stable_rank.csv and results/stable_rank.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import parse_layer_idx, parse_matrix_type  # noqa: E402

RUNS = {
    "sft (vs base)": "results/svd_sft/spectra",
    "sft seed 43 (vs base)": "results/svd_sft_seed43/spectra",  # run-to-run noise
    "dpo (vs sft)": "results/svd_dpo/spectra",
    "wikitext LM (vs base)": "results/svd_control_lm/spectra",  # no alignment content
    "python code LM (vs base)": "results/svd_control_code/spectra",  # ditto, different data
    "random init pair": "results/svd_control_random/spectra",  # noise floor
}
TOP_K = [1, 2, 4, 8, 16, 32, 64, 128]
ENERGY_TARGETS = [0.5, 0.9, 0.99]


def main() -> int:
    rows = []
    for run, spectra_dir in RUNS.items():
        paths = sorted(Path(spectra_dir).glob("*.pt"))
        if not paths:
            print(f"nothing in {spectra_dir}, skipping")
            continue

        for path in paths:
            sv = torch.load(path, weights_only=True).float()
            if sv.numel() == 0 or sv[0] <= 0:
                continue
            energy = sv**2
            total = float(energy.sum())
            max_rank = sv.numel()
            cumulative = torch.cumsum(energy, dim=0) / total

            row = {
                "source": run,
                "key": path.stem,  # filenames are the state-dict keys
                "layer_idx": parse_layer_idx(path.stem),
                "matrix_type": parse_matrix_type(path.stem),
                "max_rank": max_rank,
                "stable_rank": total / float(energy[0]),
            }
            row["stable_rank_frac"] = row["stable_rank"] / max_rank
            for k in TOP_K:
                row[f"energy_top{k}"] = float(cumulative[min(k, max_rank) - 1])
            for target in ENERGY_TARGETS:
                k = min(int(torch.searchsorted(cumulative, torch.tensor(target))) + 1, max_rank)
                row[f"k_for_{int(target * 100)}pct"] = k
                row[f"k_for_{int(target * 100)}pct_frac"] = k / max_rank
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("results/stable_rank.csv", index=False)

    cols = ["stable_rank_frac", "energy_top1", "energy_top8", "energy_top32", "k_for_50pct_frac"]
    print(df.groupby("source")[cols].mean().to_string())
    print()
    print(df.pivot_table(index="matrix_type", columns="source", values="stable_rank_frac"))

    runs = [r for r in RUNS if (df["source"] == r).any()]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for run in runs:
        sub = df[df["source"] == run]
        ax.plot(TOP_K, [sub[f"energy_top{k}"].mean() for k in TOP_K], "o-", label=run)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TOP_K)
    ax.set_xticklabels(TOP_K)
    ax.set_xlabel("k")
    ax.set_ylabel("share of energy in the top k singular values")
    ax.set_title("Energy in the top k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(runs)), [df[df["source"] == r]["stable_rank_frac"].mean() for r in runs],
           color=[f"C{i}" for i in range(len(runs))])  # fmt: skip
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("stable rank / max rank (lower = more concentrated)")
    ax.set_title("Stable rank")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    for run in runs:
        sub = df[(df["source"] == run) & df["layer_idx"].notna()]
        by_layer = sub.groupby("layer_idx")["stable_rank_frac"].mean()
        ax.plot(by_layer.index, by_layer.values, "o-", label=run, markersize=3)
    ax.set_xlabel("layer")
    ax.set_ylabel("stable rank / max rank")
    ax.set_title("By layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/stable_rank.png", dpi=150)
    plt.close(fig)
    print("\nwrote results/stable_rank.csv and results/stable_rank.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
