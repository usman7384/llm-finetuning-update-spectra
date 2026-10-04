#!/usr/bin/env python
"""Does a LoRA adapter learn the same directions as full fine-tuning, or just
something the same size?

For every matrix and rank, compare the top-r singular vectors of
W_sft - W_base against that rank's (unmerged) adapter: U against B on the
output side, V against A on the input side. Score is the mean cosine of the
principal angles. A random-subspace baseline at the same size is computed
alongside, since the raw number means nothing without it.

    uv run python scripts/lora_overlap.py

Writes results/lora_overlap.csv and results/lora_overlap.png.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.subspace_overlap import (  # noqa: E402
    mean_cosine_similarity,
    orthonormal_basis,
    random_subspace_baseline,
    top_singular_directions,
)
from align_geometry.analysis.svd_delta import is_analyzable_matrix, load_state_dict  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
# PEFT prefixes everything with base_model.model.
ADAPTER_KEY = re.compile(r"^base_model\.model\.(.+)\.lora_[AB]\.weight$")


def base_key(adapter_key: str) -> str:
    m = ADAPTER_KEY.match(adapter_key)
    if not m:
        raise ValueError(f"unexpected adapter key: {adapter_key}")
    return f"{m.group(1)}.weight"


def main() -> int:
    base = load_state_dict("checkpoints/base")
    sft = load_state_dict("checkpoints/sft")
    keys = [k for k in base if k in sft and is_analyzable_matrix(k)]

    baselines: dict[tuple[int, int], float] = {}

    def baseline(dim: int, rank: int) -> float:
        if (dim, rank) not in baselines:
            baselines[(dim, rank)] = random_subspace_baseline(dim, rank, n_trials=20, seed=0)
        return baselines[(dim, rank)]

    rows = []
    for rank in RANKS:
        # lr 1e-4 won at every rank in the sweep, so that's the one to compare against
        path = Path(f"checkpoints/lora_sweep/r{rank}_lr1e-04/adapter/adapter_model.safetensors")
        if not path.exists():
            print(f"skipping rank {rank}, no adapter at {path}")
            continue

        adapters: dict[str, dict] = {}
        for k, tensor in load_file(path).items():
            side = "A" if k.endswith(".lora_A.weight") else "B"
            adapters.setdefault(base_key(k), {})[side] = tensor

        for key in keys:
            if key not in adapters:
                continue
            a = adapters[key]["A"].float()  # (r, in)
            b = adapters[key]["B"].float()  # (out, r)
            delta = sft[key].float() - base[key].float()

            u, v = top_singular_directions(delta, rank)
            rows.append(
                {
                    "rank": rank,
                    "key": key,
                    "cos_output_side": mean_cosine_similarity(u, orthonormal_basis(b)),
                    "cos_input_side": mean_cosine_similarity(v, orthonormal_basis(a.T)),
                    "baseline_output": baseline(delta.shape[0], rank),
                    "baseline_input": baseline(delta.shape[1], rank),
                }
            )
        print(f"rank {rank} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/lora_overlap.csv", index=False)
    summary = df.groupby("rank")[
        ["cos_output_side", "cos_input_side", "baseline_output", "baseline_input"]
    ].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        summary.index, summary["cos_output_side"], "o-", color="C0", label="output side (U vs B)"
    )
    ax.plot(summary.index, summary["cos_input_side"], "s-", color="C1", label="input side (V vs A)")
    ax.plot(summary.index, summary["baseline_output"], "o:", color="C0", alpha=0.5,
            label="random baseline, output dim")  # fmt: skip
    ax.plot(summary.index, summary["baseline_input"], "s:", color="C1", alpha=0.5,
            label="random baseline, input dim")  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("mean cosine of principal angles")
    ax.set_title("LoRA subspace vs. full fine-tuning's top directions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/lora_overlap.png", dpi=150)
    plt.close(fig)

    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
