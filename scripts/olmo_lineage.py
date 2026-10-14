#!/usr/bin/env python
"""Which OLMo 2 1B base checkpoint is the released SFT model trained from?

The released SFT checkpoint has cosine ~0 with the `main` revision of
allenai/OLMo-2-0425-1B (results/released_sizes.csv), so it is not a fine-tune
of it. The base repo also publishes its mid-training checkpoints as branches.
This compares the SFT weights with the final checkpoint of each mid-training
run, using only the second safetensors shard (~1 GB) of each candidate.

    uv run python scripts/olmo_lineage.py

Writes results/olmo_lineage.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix  # noqa: E402
from align_geometry.hub import weights_dir  # noqa: E402

BASE = "allenai/OLMo-2-0425-1B"
SFT = "allenai/OLMo-2-0425-1B-SFT"
CANDIDATES = ["main", "stage2-ingredient1-step23852-tokens51B",
              "stage2-ingredient2-step23852-tokens51B",
              "stage2-ingredient3-step23852-tokens51B"]  # fmt: skip


def main() -> int:
    sft = LazyWeights(weights_dir(SFT))
    rows = []
    for rev in CANDIDATES:
        path = hf_hub_download(BASE, "model-00002-of-00002.safetensors", revision=rev)
        dot = nb = ns = 0.0
        n = 0
        with safe_open(path, "pt") as f:
            for key in f.keys():  # noqa: SIM118  (safe_open handles are not iterable)
                if not is_analyzable_matrix(key) or key not in sft:
                    continue
                wb, ws = f.get_tensor(key).double(), sft[key].double()
                dot += float((wb * ws).sum())
                nb += float((wb**2).sum())
                ns += float((ws**2).sum())
                n += 1
        rows.append({"revision": rev, "matrices": n, "cosine_with_sft": dot / (nb * ns) ** 0.5,
                     "relative_delta": ((nb + ns - 2 * dot) / nb) ** 0.5})  # fmt: skip
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/olmo_lineage.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
