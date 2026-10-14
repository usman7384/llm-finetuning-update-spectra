#!/usr/bin/env python
"""Gap to noise along the Renyi family, on a grid of orders alpha.

Only three of the five measures in the paper are Renyi ranks of the energy
distribution p (orders 0, 2 and infinity). This fills in the family: for each
alpha, 1 - mean R_alpha(update) / mean R_alpha(noise), with a layer-bootstrap
interval, for every run of v2_analysis.py. alpha=0 is skipped (every update is
exactly full rank).

    uv run python scripts/renyi_grid.py

Writes results/v2/renyi_grid.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v2_analysis import FAMILIES, boot_ratio  # noqa: E402

from align_geometry.analysis.svd_delta import renyi_rank  # noqa: E402

ALPHAS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 16, float("inf")]
RUNS = ["SFT", "wikitext", "code", "shuffled words"]


def per_matrix(result_dir: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(f"results/{result_dir}/spectra").glob("*.pt")):
        sv = torch.load(p, weights_only=True).double()
        row = {"key": p.stem, "layer": int(p.stem.split("layers.")[1].split(".")[0])}
        row.update({f"a{a}": renyi_rank(sv, a) / sv.numel() for a in ALPHAS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("key")


def main() -> int:
    rows = []
    for fam, runs in FAMILIES.items():
        noise = per_matrix(runs["noise"])
        for name in RUNS:
            df = per_matrix(runs[name])
            for a in ALPHAS:
                g, lo, hi = boot_ratio(df, noise, f"a{a}")
                rows.append({"model": fam, "run": name, "alpha": a, "update": df[f"a{a}"].mean(),
                             "noise": noise[f"a{a}"].mean(), "gap": g, "lo": lo, "hi": hi})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv("results/v2/renyi_grid.csv", index=False)
    print(out.pivot_table(index=["model", "run"], columns="alpha", values="gap").round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
