#!/usr/bin/env python
"""Do updates with the same spectral shape point the same way?

Stable rank only describes the shape of a spectrum. Two updates can share it
and still use different directions. For each pair of runs on the same base we
take the top-k left and right singular vectors of each update, matrix by
matrix, and measure their overlap,

    overlap = ||U_A^T U_B||_F^2 / k      (1 = same subspace, k/m = random)

averaged over matrices, with a layer-bootstrap interval. Second seeds give the
ceiling (same data source, different sample and seed); random subspaces give
the floor.

    uv run python scripts/update_overlap.py            # CPU is fine, ~1 h

Writes results/v2/update_overlap.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    LazyWeights,
    is_analyzable_matrix,
    parse_layer_idx,
)

K = 16
FAMILIES = {
    "Qwen2.5-0.5B": ("checkpoints/base", {
        "SFT": "sft", "SFT (seed 2)": "sft_seed43", "wikitext": "control_lm",
        "wikitext (seed 2)": "control_lm_seed43", "code": "control_code",
        "code (seed 2)": "control_code_seed43", "shuffled words": "control_shuffled",
        "code (steps matched)": "control_code_steps"}),
    "SmolLM2-360M": ("checkpoints/smollm2_base", {
        "SFT": "smollm2_sft", "SFT (seed 2)": "smollm2_sft_seed43",
        "wikitext": "smollm2_control_lm", "wikitext (seed 2)": "smollm2_control_lm_seed43",
        "code": "smollm2_control_code", "code (seed 2)": "smollm2_control_code_seed43",
        "shuffled words": "smollm2_control_shuffled",
        "code (steps matched)": "smollm2_control_code_steps"}),
    # DPO updates are measured from the SFT model they start from
    "Qwen2.5-0.5B DPO": ("checkpoints/v2/sft", {
        "DPO": "dpo", "DPO (seed 2)": "dpo_seed43", "DPO swapped": "dpo_swapped"}),
}  # fmt: skip
PAIRS = [("SFT", "SFT (seed 2)"), ("wikitext", "wikitext (seed 2)"), ("code", "code (seed 2)"),
         ("SFT", "wikitext"), ("SFT", "code"), ("wikitext", "code"), ("SFT", "shuffled words"),
         ("wikitext", "shuffled words"), ("code", "code (steps matched)"),
         ("DPO", "DPO (seed 2)"), ("DPO", "DPO swapped")]  # fmt: skip


def top_subspaces(base: LazyWeights, run: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    after = LazyWeights(run)
    out = {}
    for key in base.keys():
        if not is_analyzable_matrix(key) or key not in after:
            continue
        d = after[key].double() - base[key].double()
        u, _, vh = torch.linalg.svd(d, full_matrices=False)
        out[key] = (u[:, :K].float(), vh[:K].T.float())
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for fam, (base_dir, runs) in FAMILIES.items():
        base = LazyWeights(base_dir)
        have = {name: Path("checkpoints/v2") / d for name, d in runs.items()
                if (Path("checkpoints/v2") / d / "model.safetensors").exists()}  # fmt: skip
        subs = {}
        for name, path in have.items():
            print(f"{fam}: SVD of {name}", flush=True)
            subs[name] = top_subspaces(base, path)
        for a, b in PAIRS + [("SFT", "random"), ("DPO", "random")]:
            if a not in subs or (b not in subs and b != "random"):
                continue
            per = []
            for key, (ua, va) in subs[a].items():
                if b == "random":
                    m, n = ua.shape[0], va.shape[0]
                    ub = torch.linalg.qr(torch.randn(m, K))[0]
                    vb = torch.linalg.qr(torch.randn(n, K))[0]
                else:
                    ub, vb = subs[b][key]
                per.append({"layer": parse_layer_idx(key),
                            "left": float((ua.T @ ub).pow(2).sum()) / K,
                            "right": float((va.T @ vb).pow(2).sum()) / K})  # fmt: skip
            df = pd.DataFrame(per)
            g = df.groupby("layer")[["left", "right"]].mean()
            idx = rng.integers(0, len(g), (5000, len(g)))
            row = {"model": fam, "a": a, "b": b}
            for side in ("left", "right"):
                vals = g[side].to_numpy()
                boot = vals[idx].mean(1)
                row.update({side: df[side].mean(), f"{side}_lo": np.percentile(boot, 2.5),
                            f"{side}_hi": np.percentile(boot, 97.5)})  # fmt: skip
            rows.append(row)
            print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/v2/update_overlap.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
