#!/usr/bin/env python
"""Measure released base -> instruct deltas the same way as our own runs.

Timber (2025) looked at exactly these kinds of pairs and concluded post-training
barely changes effective rank. Our own runs say a thresholded rank can't tell a
real delta from noise (Timber's entropy rank is checked separately in
timber_check.py). So: take public base/instruct pairs,
compute dW = W_instruct - W_base for every attention/MLP matrix, and report
both measures next to a random-matrix baseline of the same shapes.

No training. The SVDs use the GPU if there is one. The time is mostly
downloads: ~17GB for the first five pairs, ~35GB more with the Qwen3 ones
(cached by Hugging Face, so a rerun is quick). Weights are read one matrix at a
time, so RAM isn't the limit, disk is. Point HF_HUB_CACHE at a drive with room.

    uv run python scripts/released_pairs.py
    uv run python scripts/released_pairs.py --pairs qwen2.5-0.5b smollm2-135m

Writes results/released_pairs/<pair>.csv (per matrix), results/released_pairs.csv
(one row per pair) and results/released_pairs.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    LazyWeights,
    energy_top_k,
    is_analyzable_matrix,
    parse_layer_idx,
    parse_matrix_type,
    stable_rank,
)
from align_geometry.hub import weights_dir  # noqa: E402

# all ungated. instruct versions were post-trained from these exact bases
PAIRS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    # both in Timber's table 1. the 8B pair is ~33GB of downloads
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"),
    "qwen3-8b": ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_noise_cache: dict[tuple[int, int], tuple[float, float]] = {}


def noise(shape: tuple[int, int]) -> tuple[float, float]:
    """(thresholded rank frac, stable rank frac) of a Gaussian matrix this shape.

    Two random inits of a model differ by exactly this kind of matrix, and both
    measures ignore scale, so one draw per shape stands in for the noise floor.
    (For Qwen2.5-0.5B's shapes this should come out around 0.436 stable rank,
    same as the random-init pair in docs/report.md.)
    """
    if shape not in _noise_cache:
        g = torch.Generator().manual_seed(0)
        sv = torch.linalg.svdvals(torch.randn(*shape, generator=g).to(DEVICE)).cpu()
        max_rank = min(shape)
        threshold = float((sv >= 0.01 * sv[0]).sum()) / max_rank
        _noise_cache[shape] = (threshold, stable_rank(sv) / max_rank)
    return _noise_cache[shape]


def measure(name: str, base_repo: str, instruct_repo: str) -> pd.DataFrame:
    print(f"\n{name}: fetching {base_repo} and {instruct_repo}")
    # one matrix at a time, so an 8B pair fits in RAM
    base = LazyWeights(weights_dir(base_repo))
    instruct = LazyWeights(weights_dir(instruct_repo))

    rows = []
    for key in base:
        if key not in instruct or not is_analyzable_matrix(key):
            continue
        delta = instruct[key].float() - base[key].float()
        if delta.ndim != 2 or delta.abs().max() == 0:
            continue
        sv = torch.linalg.svdvals(delta.to(DEVICE)).cpu()
        max_rank = min(delta.shape)
        noise_threshold, noise_stable = noise(tuple(delta.shape))
        rows.append(
            {
                "pair": name,
                "key": key,
                "layer_idx": parse_layer_idx(key),
                "matrix_type": parse_matrix_type(key),
                "max_rank": max_rank,
                "threshold_rank_frac": float((sv >= 0.01 * sv[0]).sum()) / max_rank,
                "stable_rank_frac": stable_rank(sv) / max_rank,
                "energy_top1": energy_top_k(sv, 1),
                "fro_norm": float(torch.linalg.norm(delta)),
                "noise_threshold_rank_frac": noise_threshold,
                "noise_stable_rank_frac": noise_stable,
            }
        )
    del base, instruct
    print(f"{name}: {len(rows)} matrices")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=list(PAIRS))
    parser.add_argument("--force", action="store_true", help="redo pairs that already have a csv")
    args = parser.parse_args()

    out_dir = Path("results/released_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in args.pairs:
        # each pair is saved as soon as it's done, so an interrupted run picks up here
        done = out_dir / f"{name}.csv"
        if done.exists() and not args.force:
            print(f"{name}: already done, using {done}")
            df = pd.read_csv(done)
        else:
            df = measure(name, *PAIRS[name])
            df.to_csv(done, index=False)
        summary.append(
            {
                "pair": name,
                "matrices": len(df),
                "threshold_rank_frac": df["threshold_rank_frac"].mean(),
                "noise_threshold_rank_frac": df["noise_threshold_rank_frac"].mean(),
                "stable_rank_frac": df["stable_rank_frac"].mean(),
                "noise_stable_rank_frac": df["noise_stable_rank_frac"].mean(),
                "energy_top1": df["energy_top1"].mean(),
                "mean_fro_norm": df["fro_norm"].mean(),
            }
        )

    s = pd.DataFrame(summary)
    s["noise_over_real"] = s["noise_stable_rank_frac"] / s["stable_rank_frac"]
    s.to_csv("results/released_pairs.csv", index=False)
    print()
    print(s.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = range(len(s))
    for ax, real, ref, title in [
        (axes[0], "threshold_rank_frac", "noise_threshold_rank_frac", "Thresholded rank (0.01)"),
        (axes[1], "stable_rank_frac", "noise_stable_rank_frac", "Stable rank"),
    ]:
        ax.bar([i - 0.2 for i in x], s[real], 0.4, label="base -> instruct", color="C0")
        ax.bar([i + 0.2 for i in x], s[ref], 0.4, label="random matrix", color="C7")
        ax.set_xticks(list(x))
        ax.set_xticklabels(s["pair"], rotation=20, ha="right")
        ax.set_ylabel("fraction of max rank")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/released_pairs.png", dpi=150)
    plt.close(fig)
    print("\nwrote results/released_pairs.csv and results/released_pairs.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
