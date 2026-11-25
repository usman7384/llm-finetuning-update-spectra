#!/usr/bin/env python
"""Do two updates agree in sign as well as in direction?

The top-16 subspace overlap (update_overlap.py) is the same for an update and
its negative, and ignores how left and right directions pair up. Here we take
the plain cosine between two updates instead,

    cos = <dA, dB> / (||dA|| ||dB||)

per matrix, averaged over matrices with a layer-bootstrap interval, plus one
cosine over all matrices together. For runs on the same dataset we also
count the training rows they share, since shared rows alone can make two
updates agree.

    uv run python scripts/signed_cosine.py            # CPU is fine, needs the weights

Writes results/v2/signed_cosine.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from update_overlap import FAMILIES, PAIRS  # noqa: E402

from align_geometry.analysis.subspace_overlap import signed_cosine  # noqa: E402
from align_geometry.analysis.svd_delta import (  # noqa: E402
    LazyWeights,
    is_analyzable_matrix,
    parse_layer_idx,
)


def indices(run: str) -> dict:
    return json.loads((Path("data/processed") / f"v2_{run}_indices.json").read_text())


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for fam, (base_dir, runs) in FAMILIES.items():
        pairs = [p for p in PAIRS if p[0] in runs and p[1] in runs]
        if "DPO" in fam:
            pairs.append(("DPO (seed 2)", "DPO swapped"))
        for a, b in pairs:
            # only three checkpoints open at a time: mapping every run runs out of memory
            base, after_a, after_b = (
                LazyWeights(p)
                for p in (base_dir, *(Path("checkpoints/v2") / runs[x] for x in (a, b)))
            )
            per = []
            for key in base:
                if not is_analyzable_matrix(key):
                    continue
                w = base[key].double()
                da, db = after_a[key].double() - w, after_b[key].double() - w
                per.append({
                    "layer": parse_layer_idx(key),
                    "cos": signed_cosine(da, db),
                    "dot": float((da * db).sum()),
                    "aa": float(da.pow(2).sum()),
                    "bb": float(db.pow(2).sum()),
                })  # fmt: skip
            df = pd.DataFrame(per)
            g = df.groupby("layer")["cos"].mean().to_numpy()
            boot = g[rng.integers(0, len(g), (5000, len(g)))].mean(1)
            row = {
                "model": fam,
                "a": a,
                "b": b,
                "cos": df["cos"].mean(),
                "cos_lo": np.percentile(boot, 2.5),
                "cos_hi": np.percentile(boot, 97.5),
                "cos_all_matrices": df["dot"].sum() / np.sqrt(df["aa"].sum() * df["bb"].sum()),
            }
            ia, ib = indices(runs[a]), indices(runs[b])
            if ia["dataset_name"] == ib["dataset_name"]:
                row["n_train"] = len(ia["train_indices"])
                row["n_shared"] = len(set(ia["train_indices"]) & set(ib["train_indices"]))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("results/v2/signed_cosine.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
