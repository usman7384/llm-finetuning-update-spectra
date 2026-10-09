#!/usr/bin/env python
"""All the numbers the paper quotes for our own (fp32, v2) fine-tunes.

Reads the saved spectra, so it needs no weights and no GPU. For every run and
measure: the mean over matrices, the gap to the noise baseline with a 95%
layer-bootstrap interval, the same for the pretrained weights, paired seed and
task differences, and stable rank by layer group.

    uv run python scripts/v2_analysis.py

Writes results/v2/{measures,gaps,paired,layer_groups}.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import entropy_rank, renyi_rank  # noqa: E402

FAMILIES = {
    "Qwen2.5-0.5B": {
        "SFT": "v2/svd_sft",
        "wikitext": "v2/svd_control_lm",
        "code": "v2/svd_control_code",
        "shuffled words": "v2/svd_control_shuffled",
        "SFT (seed 2)": "v2/svd_sft_seed43",
        "wikitext (seed 2)": "v2/svd_control_lm_seed43",
        "code (seed 2)": "v2/svd_control_code_seed43",
        "noise": "svd_control_random",
        "pretrained W": "weights_base",
    },
    "SmolLM2-360M": {
        "SFT": "v2/svd_smollm2_sft",
        "wikitext": "v2/svd_smollm2_control_lm",
        "code": "v2/svd_smollm2_control_code",
        "shuffled words": "v2/svd_smollm2_control_shuffled",
        "noise": "svd_smollm2_random",
        "pretrained W": "weights_smollm2_base",
    },
}
MEASURES = ["threshold", "k90", "erank", "participation", "stable", "top1"]
PAIRS = [("wikitext", "SFT"), ("code", "SFT"), ("wikitext", "code"), ("SFT", "shuffled words"),
         ("SFT", "SFT (seed 2)"), ("wikitext", "wikitext (seed 2)"),
         ("code", "code (seed 2)")]  # fmt: skip
N_BOOT = 5000
rng = np.random.default_rng(0)


def per_matrix(result_dir: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(f"results/{result_dir}/spectra").glob("*.pt")):
        sv = torch.load(p, weights_only=True).double()
        r = sv.numel()
        e = sv**2
        cum = torch.cumsum(e, 0) / e.sum()
        rows.append(
            {
                "key": p.stem,
                "layer": int(p.stem.split("layers.")[1].split(".")[0]),
                "threshold": float((sv >= 0.01 * sv[0]).sum()) / r,
                "k90": (int(torch.searchsorted(cum, torch.tensor(0.9, dtype=cum.dtype))) + 1) / r,
                "erank": entropy_rank(sv) / r,
                "participation": renyi_rank(sv, 2) / r,
                "stable": renyi_rank(sv, float("inf")) / r,
                "top1": float(cum[0]),
            }
        )
    return pd.DataFrame(rows).set_index("key")


def boot_ratio(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """1 - mean(a)/mean(b), resampling whole layers, paired by matrix."""
    j = a[[col, "layer"]].join(b[[col]], rsuffix="_b")
    g = j.groupby("layer")
    sa, sb = g[col].sum().to_numpy(), g[col + "_b"].sum().to_numpy()
    idx = rng.integers(0, len(sa), (N_BOOT, len(sa)))
    boot = 1 - sa[idx].sum(1) / sb[idx].sum(1)
    return (
        1 - sa.sum() / sb.sum(),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """mean(b) - mean(a) over matrices, resampling layers, paired by matrix."""
    j = a[[col, "layer"]].join(b[[col]], rsuffix="_b")
    j["d"] = j[col + "_b"] - j[col]
    g = j.groupby("layer")["d"]
    s, n = g.sum().to_numpy(), g.count().to_numpy()
    idx = rng.integers(0, len(s), (N_BOOT, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return float(j["d"].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    out = Path("results/v2")
    measures, gaps, paired, groups = [], [], [], []
    for fam, runs in FAMILIES.items():
        data = {name: per_matrix(d) for name, d in runs.items()}
        for name, df in data.items():
            measures.append({"model": fam, "run": name, **df[MEASURES].mean().to_dict()})
            for ref in ("noise", "pretrained W"):
                if name in ("noise", "pretrained W"):
                    continue
                for col in ["threshold", "k90", "erank", "participation", "stable"]:
                    g, lo, hi = boot_ratio(df, data[ref], col)
                    gaps.append({"model": fam, "run": name, "vs": ref, "measure": col,
                                 "gap": g, "lo": lo, "hi": hi})  # fmt: skip
        for a, b in PAIRS:
            if a in data and b in data:
                d, lo, hi = boot_diff(data[a], data[b], "stable")
                paired.append({"model": fam, "a": a, "b": b, "stable_b_minus_a": d,
                               "lo": lo, "hi": hi})  # fmt: skip
        for name in ["SFT", "wikitext", "code", "shuffled words"]:
            df = data[name]
            bins = pd.cut(
                df["layer"], 4, labels=["first quarter", "second", "third", "last quarter"]
            )
            for grp, val in df.groupby(bins, observed=True)["stable"].mean().items():
                groups.append({"model": fam, "run": name, "layers": grp, "stable": val})

    pd.set_option("display.width", 220)
    for name, rows in [("measures", measures), ("gaps", gaps), ("paired", paired),
                       ("layer_groups", groups)]:  # fmt: skip
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name}")
        print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
