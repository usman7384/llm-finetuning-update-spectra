#!/usr/bin/env python
"""Singular values of the pretrained weights themselves, as a reference.

The noise baseline answers "is this update more concentrated than a matrix
with no structure?". A reviewer will also ask "more concentrated than the
weights it's added to?", so this saves the spectrum of every attention/MLP
matrix of a base checkpoint, in the same layout svd_deltas.py uses.

    uv run python scripts/weight_spectra.py \
        --checkpoint checkpoints/base --out results/weights_base
    uv run python scripts/weight_spectra.py \
        --checkpoint checkpoints/smollm2_base --out results/weights_smollm2_base

Writes <out>/spectra/<key>.pt and prints mean stable rank and eRank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    LazyWeights,
    entropy_rank,
    is_analyzable_matrix,
    stable_rank,
)
from align_geometry.hub import weights_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="local dir or Hub repo")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) / "spectra"
    out.mkdir(parents=True, exist_ok=True)
    weights = LazyWeights(weights_dir(args.checkpoint))

    stable, erank = [], []
    for key in weights:
        if not is_analyzable_matrix(key):
            continue
        w = weights[key].float()
        if w.ndim != 2:
            continue
        sv = torch.linalg.svdvals(w.to(device)).cpu()
        torch.save(sv, out / f"{key}.pt")
        stable.append(stable_rank(sv) / sv.numel())
        erank.append(entropy_rank(sv) / sv.numel())

    if not stable:
        print(f"no attention/MLP matrices found in {args.checkpoint}")
        return 1
    print(
        f"{args.checkpoint}: {len(stable)} matrices, "
        f"stable rank {sum(stable) / len(stable):.3f}, eRank {sum(erank) / len(erank):.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
