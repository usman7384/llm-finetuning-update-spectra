#!/usr/bin/env python
"""How much post-training changed the parameters the rank analysis leaves out.

The random-update check edits only attention/MLP matrices, so "base + real
update" keeps the base model's embeddings and norms. This records how far
post-training moved those: |W_after - W_base|_F / |W_base|_F per tensor group.

    uv run python scripts/embedding_change.py --pairs qwen3-0.6b

Writes results/embedding_change.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from released_spectra import PAIRS  # noqa: E402

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix  # noqa: E402
from align_geometry.hub import weights_dir  # noqa: E402


def group(key: str) -> str:
    if "embed_tokens" in key:
        return "embeddings"
    if "lm_head" in key:
        return "output head"
    if "norm" in key:
        return "norms"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=["qwen3-0.6b"])
    args = parser.parse_args()
    rows = []
    for pair in args.pairs:
        base, inst = (LazyWeights(weights_dir(m)) for m in PAIRS[pair])
        sums: dict[str, list[float]] = {}
        for k in base:
            if is_analyzable_matrix(k) or k not in inst:
                continue
            b, a = base[k].double(), inst[k].double()
            s = sums.setdefault(group(k), [0.0, 0.0])
            s[0] += float(((a - b) ** 2).sum())
            s[1] += float((b**2).sum())
        for g, (d2, w2) in sums.items():
            rows.append({"pair": pair, "group": g, "relative_change": (d2 / w2) ** 0.5})
    df = pd.DataFrame(rows)
    df.to_csv("results/embedding_change.csv", index=False)
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
