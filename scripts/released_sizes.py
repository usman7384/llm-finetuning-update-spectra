#!/usr/bin/env python
"""Size of each released update, and how much of it bf16 storage zeroed out.

The saved spectra give |dW|_F but not |W|_F, so they cannot say whether an
update is large or small for its model. This reads both checkpoints once more
(CPU, no SVD) and records, summed over the analysed matrices:

- |dW|_F and |W_base|_F, and their ratio;
- the share of entries where dW is exactly zero, a direct check on rounding:
  in bf16, a weight that moved by less than half a unit is stored unchanged;
- the cosine between W_base and W_after, a check on lineage: a fine-tune of
  W_base stays close to 1, an unrelated checkpoint is near 0.

    uv run python scripts/released_sizes.py --pairs qwen3-0.6b

Writes results/released_sizes.csv, adding to the rows already there (one
process per pair keeps memory low).
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=list(PAIRS))
    args = parser.parse_args()
    out = Path("results/released_sizes.csv")
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["pair"])
    rows = []
    for pair in args.pairs:
        base_id, inst_id = PAIRS[pair]
        base, inst = LazyWeights(weights_dir(base_id)), LazyWeights(weights_dir(inst_id))
        keys = [k for k in base if is_analyzable_matrix(k) and k in inst]
        d2 = w2 = a2 = dot = zeros = total = 0.0
        dtype = None
        for k in keys:
            wb, wi = base[k], inst[k]
            dtype = dtype or str(wi.dtype).replace("torch.", "")
            d = wi.float() - wb.float()
            d2 += float((d.double() ** 2).sum())
            w2 += float((wb.double() ** 2).sum())
            a2 += float((wi.double() ** 2).sum())
            dot += float((wb.double() * wi.double()).sum())
            zeros += float((d == 0).sum())
            total += d.numel()
        rows.append({"pair": pair, "matrices": len(keys), "stored_dtype": dtype,
                     "delta_fro": d2**0.5, "base_fro": w2**0.5,
                     "relative_size": (d2 / w2) ** 0.5, "zero_fraction": zeros / total,
                     "base_cosine": dot / (w2 * a2) ** 0.5})  # fmt: skip
        print(rows[-1], flush=True)
    df = pd.concat([old[~old["pair"].isin(args.pairs)], pd.DataFrame(rows)])
    df = df.set_index("pair").loc[[p for p in PAIRS if p in set(df["pair"])]].reset_index()
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
