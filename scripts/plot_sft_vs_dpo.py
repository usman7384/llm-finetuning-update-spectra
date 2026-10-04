#!/usr/bin/env python
"""SFT delta vs DPO delta, side by side: spectrum shape, effective rank by
layer, and size.

Short version of what it shows: same shape, DPO is about 15x smaller.

    uv run python scripts/plot_sft_vs_dpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def main() -> int:
    runs = [
        ("sft (vs base)", "results/svd_sft", "C0"),
        ("dpo (vs sft)", "results/svd_dpo", "C1"),
    ]
    summaries = {tag: pd.read_csv(f"{d}/matrix_summary.csv") for tag, d, _ in runs}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for tag, d, color in runs:
        first = True
        for pt in Path(d, "spectra").glob("*.pt"):
            sv = torch.load(pt, weights_only=True)
            if sv.numel() == 0 or sv[0] == 0:
                continue
            ax.plot((sv / sv[0]).numpy(), color=color, alpha=0.15, linewidth=0.7,
                    label=tag if first else None)  # fmt: skip
            first = False
    ax.set_xlabel("singular value index")
    ax.set_ylabel("sigma_i / sigma_1")
    ax.set_yscale("log")
    ax.set_title("Spectrum shape")
    ax.legend(fontsize=9)

    ax = axes[1]
    for tag, _, color in runs:
        by_layer = summaries[tag].groupby("layer_idx")["effective_rank_0.01"].mean()
        ax.plot(by_layer.index, by_layer.values, "o-", color=color, label=tag)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean effective rank (threshold 0.01)")
    ax.set_title("Effective rank by layer")
    ax.legend(fontsize=9)

    # this panel is where they actually differ
    ax = axes[2]
    norms = {tag: summaries[tag].groupby("matrix_type")["fro_norm"].mean() for tag, _, _ in runs}
    types = sorted(norms[runs[0][0]].index)
    width = 0.35
    for i, (tag, _, color) in enumerate(runs):
        offset = (i - 0.5) * width
        ax.bar([j + offset for j in range(len(types))], [norms[tag][t] for t in types],
               width, label=tag, color=color)  # fmt: skip
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.set_ylabel("mean Frobenius norm of delta")
    ax.set_title("Size of the change")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("results/sft_vs_dpo.png", dpi=150)
    plt.close(fig)
    print("wrote results/sft_vs_dpo.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
