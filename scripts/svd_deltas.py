#!/usr/bin/env python
"""SVD every attention/MLP weight delta between two checkpoints.

    uv run python scripts/svd_deltas.py --before checkpoints/base \\
        --after checkpoints/sft --tag sft --out results/svd_sft

Writes, under --out:
    matrix_summary.csv           one row per weight matrix
    spectra/<key>.pt             full singular value spectrum per matrix
    spectra.png                  every matrix's normalized decay curve
    effective_rank_by_depth.png

Works on any two safetensors checkpoints. The random-init noise floor came
from pointing it at two freshly initialized models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    compute_deltas,
    deltas_to_records,
    load_state_dict,
)


def make_plots(df: pd.DataFrame, spectra: dict[str, torch.Tensor], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for matrix_type, group in df.groupby("matrix_type"):
        group = group.sort_values("layer_idx")
        ax.plot(group["layer_idx"], group["effective_rank_0.01"], marker="o", label=matrix_type)
    ax.set_xlabel("layer")
    ax.set_ylabel("effective rank (sigma >= 0.01 * sigma_1)")
    ax.set_title("Effective rank of the delta, by layer")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_rank_by_depth.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for sv in spectra.values():
        if sv.numel() and sv[0] != 0:
            ax.plot((sv / sv[0]).numpy(), alpha=0.3, linewidth=0.8)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("sigma_i / sigma_1")
    ax.set_yscale("log")
    ax.set_title(f"Singular value decay, all {len(spectra)} matrices")
    fig.tight_layout()
    fig.savefig(out_dir / "spectra.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--before", required=True, help="checkpoint dir for W_before")
    parser.add_argument("--after", required=True, help="checkpoint dir for W_after")
    parser.add_argument("--tag", required=True, help="short label, e.g. sft or dpo")
    parser.add_argument("--out", required=True, help="output dir")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "spectra").mkdir(parents=True, exist_ok=True)

    print(f"loading {args.before}")
    before = load_state_dict(args.before)
    print(f"loading {args.after}")
    after = load_state_dict(args.after)

    only_before = set(before) - set(after)
    only_after = set(after) - set(before)
    if only_before or only_after:
        # Qwen ties lm_head to the embeddings, so a mismatch here usually means
        # something else is going on. Worth a look before trusting the output.
        print(f"warning: {len(only_before)} keys only in before, {len(only_after)} only in after")

    deltas = compute_deltas(before, after)
    print(f"SVD'd {len(deltas)} matrices ({args.tag})")

    df = pd.DataFrame.from_records(deltas_to_records(deltas))
    df.to_csv(out_dir / "matrix_summary.csv", index=False)

    spectra = {}
    for d in deltas:
        torch.save(d.singular_values, out_dir / "spectra" / f"{d.key.replace('/', '_')}.pt")
        spectra[d.key] = d.singular_values

    make_plots(df, spectra, out_dir)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
