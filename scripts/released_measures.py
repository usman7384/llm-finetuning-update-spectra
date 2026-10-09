#!/usr/bin/env python
"""Every rank measure for the released pairs, from saved spectra, with two floors.

Reads results/released_spectra/<pair>/{delta,rounding}/ (from
released_spectra.py), so no weights are needed. For each pair and measure it
reports the update, a Gaussian matrix of the same shape, and how much of the
threshold count the bf16 rounding floor alone accounts for.

    uv run python scripts/released_measures.py

Writes results/released_measures.csv (one row per pair) and
results/released_measures/<pair>.csv (one row per matrix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    count_above_noise_edge,
    entropy_rank,
    renyi_rank,
    stable_rank_without_top,
)

ROOT = Path("results/released_spectra")
ORDER = ["smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen3-0.6b", "qwen2.5-1.5b",
         "smollm2-1.7b", "qwen3-8b",
         # alignment stages with open data, then large non-alignment updates
         "olmo2-1b-sft", "olmo2-1b-dpo", "olmo2-1b-rl", "olmo2-1b-instruct",
         "qwen2.5-math-1.5b", "qwen2.5-coder-0.5b", "qwen2.5-coder-1.5b"]  # fmt: skip
_gauss: dict[tuple[int, int], torch.Tensor] = {}


def gaussian_spectrum(m: int, n: int) -> torch.Tensor:
    if (m, n) not in _gauss:
        g = torch.Generator().manual_seed(0)
        _gauss[(m, n)] = torch.linalg.svdvals(torch.randn(m, n, generator=g)).double()
    return _gauss[(m, n)]


def measures(sv: torch.Tensor, m: int, n: int) -> dict:
    sv = sv.double()
    r = sv.numel()
    e = sv**2
    cum = torch.cumsum(e, 0) / e.sum()
    return {
        "threshold": float((sv >= 0.01 * sv[0]).sum()) / r,
        "k90": (int(torch.searchsorted(cum, torch.tensor(0.9, dtype=cum.dtype))) + 1) / r,
        "erank": entropy_rank(sv) / r,
        "renyi1": renyi_rank(sv, 1) / r,
        "participation": renyi_rank(sv, 2) / r,
        "stable": renyi_rank(sv, float("inf")) / r,
        "stable_wo_top": stable_rank_without_top(sv) / r,
        "above_edge": count_above_noise_edge(sv, m, n),
        "top1": float(cum[0]),
        "top8": float(cum[min(7, r - 1)]),
    }


def main() -> int:
    rows = []
    out_dir = Path("results/released_measures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair in ORDER:
        d_dir, r_dir = ROOT / pair / "delta", ROOT / pair / "rounding"
        paths = sorted(d_dir.glob("*.pt"))
        if not paths:
            print(f"{pair}: no spectra yet, skipping")
            continue
        per = []
        shapes = json.loads((ROOT / pair / "shapes.json").read_text())
        for p in paths:
            key = p.stem
            sv = torch.load(p, weights_only=True)
            r_path = r_dir / p.name
            # no rounding floor for checkpoints that weren't stored in bf16
            rnd = torch.load(r_path, weights_only=True).double() if r_path.exists() else None
            r = sv.numel()
            m, n = shapes[key]
            row = {"pair": pair, "key": key, "r": r}
            row.update({f"real_{k}": v for k, v in measures(sv, m, n).items()})
            row.update(
                {f"noise_{k}": v for k, v in measures(gaussian_spectrum(m, n), m, n).items()}
            )
            # how much of the threshold count could bf16 rounding alone produce?
            if rnd is not None:
                row["rounding_over_threshold"] = float((rnd >= 0.01 * sv[0]).sum()) / r
                row["above_rounding_floor"] = float((sv > rnd[0]).sum()) / r
            per.append(row)
        df = pd.DataFrame(per)
        df.to_csv(out_dir / f"{pair}.csv", index=False)
        summary = {"pair": pair, "matrices": len(df)}
        summary.update(df.drop(columns=["pair", "key"]).mean(numeric_only=True).to_dict())
        rows.append(summary)
        print(f"{pair}: {len(df)} matrices")

    if not rows:
        return 1
    s = pd.DataFrame(rows)
    s.to_csv("results/released_measures.csv", index=False)
    cols = ["pair", "real_threshold", "noise_threshold", "rounding_over_threshold",
            "above_rounding_floor", "real_stable", "noise_stable", "real_stable_wo_top",
            "noise_stable_wo_top", "real_participation", "noise_participation",
            "real_above_edge"]  # fmt: skip
    pd.set_option("display.width", 250)
    print(s.reindex(columns=cols).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
