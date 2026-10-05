#!/usr/bin/env python
"""Timber's measurement, done their way, with the controls it didn't have.

Timber (Wu et al. 2025, arXiv:2509.23595) uses Roy & Vetterli's entropy
effective rank (eRank, gamma=1), not a threshold count. Their evidence that
post-training is superficial is that eRank(W_base) and eRank(W_instruct) are
nearly the same for every layer. They also report eRank(dW) at ~85% of full
rank and use it as the cutoff for trimming dW.

Two questions:

1. Would eRank(W) barely move for *any* small change? For each matrix we also
   add a Gaussian perturbation with the same Frobenius norm as dW and see how
   much eRank moves for that. If it moves the same amount, "eRank unchanged"
   just reflects ||dW|| << ||W|| and says nothing about superficiality.
2. Does eRank(dW) tell a real delta from noise? Compared with a Gaussian
   matrix of the same shape, next to stable rank.

Runs on our own checkpoints (where the controls are) and on released pairs,
including Qwen3-0.6B, one of the pairs Timber reports.

    uv run python scripts/timber_check.py
    uv run python scripts/timber_check.py --runs qwen3-0.6b sft control_lm

Writes results/timber_check/<run>.csv, results/timber_check.csv and .png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    entropy_rank,
    is_analyzable_matrix,
    load_state_dict,
    parse_layer_idx,
    parse_matrix_type,
    stable_rank,
)

# name -> (before, after). local paths are our runs, the rest are hub repos
RUNS = {
    # ours, Qwen2.5-0.5B
    "sft": ("checkpoints/base", "checkpoints/sft"),
    "dpo": ("checkpoints/sft", "checkpoints/dpo"),
    "control_lm": ("checkpoints/base", "checkpoints/control_lm"),
    "control_code": ("checkpoints/base", "checkpoints/control_code"),
    # ours, SmolLM2-360M
    "smollm2_sft": ("checkpoints/smollm2_base", "checkpoints/smollm2_sft"),
    "smollm2_control_lm": ("checkpoints/smollm2_base", "checkpoints/smollm2_control_lm"),
    "smollm2_control_code": ("checkpoints/smollm2_base", "checkpoints/smollm2_control_code"),
    # released. qwen3-0.6b is in Timber's table 1
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
}


def resolve(src: str) -> Path:
    if Path(src).exists():
        return Path(src)
    return Path(snapshot_download(src, allow_patterns=["*.safetensors", "*.json"]))


def svdvals(m: torch.Tensor, device: str) -> torch.Tensor:
    return torch.linalg.svdvals(m.to(device)).cpu()


def measure(name: str, before: str, after: str, device: str) -> pd.DataFrame:
    print(f"\n{name}: {before} -> {after}")
    w_b = load_state_dict(resolve(before))
    w_a = load_state_dict(resolve(after))
    g = torch.Generator().manual_seed(0)

    rows = []
    for key in w_b:
        if key not in w_a or not is_analyzable_matrix(key):
            continue
        base = w_b[key].float()
        delta = w_a[key].float() - base
        if delta.ndim != 2 or delta.abs().max() == 0:
            continue
        r = min(delta.shape)
        noise = torch.randn(*delta.shape, generator=g)
        # same size of change as the real delta, no structure at all
        bumped = base + noise * (torch.linalg.norm(delta) / torch.linalg.norm(noise))

        er_base = entropy_rank(svdvals(base, device))
        er_after = entropy_rank(svdvals(base + delta, device))
        er_bumped = entropy_rank(svdvals(bumped, device))
        sv_d = svdvals(delta, device)
        sv_n = svdvals(noise, device)
        rows.append(
            {
                "run": name,
                "key": key,
                "layer_idx": parse_layer_idx(key),
                "matrix_type": parse_matrix_type(key),
                "max_rank": r,
                "rel_delta_norm": float(torch.linalg.norm(delta) / torch.linalg.norm(base)),
                # question 1: eRank of the whole weight matrix
                "erank_base_frac": er_base / r,
                "erank_after_frac": er_after / r,
                "erank_change_real": abs(er_after - er_base) / er_base,
                "erank_change_random": abs(er_bumped - er_base) / er_base,
                # question 2: eRank of the delta itself
                "delta_erank_frac": entropy_rank(sv_d) / r,
                "noise_erank_frac": entropy_rank(sv_n) / r,
                "delta_stable_frac": stable_rank(sv_d) / r,
                "noise_stable_frac": stable_rank(sv_n) / r,
            }
        )
    del w_b, w_a
    print(f"{name}: {len(rows)} matrices")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--force", action="store_true", help="redo runs that already have a csv")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path("results/timber_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in args.runs:
        done = out_dir / f"{name}.csv"
        if done.exists() and not args.force:
            print(f"{name}: already done, using {done}")
            frames.append(pd.read_csv(done))
            continue
        before, after = RUNS[name]
        if not Path(before).exists() and before.startswith("checkpoints"):
            print(f"{name}: {before} missing, skipping")
            continue
        df = measure(name, before, after, device)
        df.to_csv(done, index=False)
        frames.append(df)

    df = pd.concat(frames)
    cols = [
        "rel_delta_norm", "erank_base_frac", "erank_after_frac",
        "erank_change_real", "erank_change_random",
        "delta_erank_frac", "noise_erank_frac", "delta_stable_frac", "noise_stable_frac",
    ]  # fmt: skip
    s = df.groupby("run", sort=False)[cols].mean().reset_index()
    s.to_csv("results/timber_check.csv", index=False)
    pd.set_option("display.width", 200)
    print()
    print(s.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = range(len(s))
    ax = axes[0]
    ax.bar([i - 0.2 for i in x], s["erank_change_real"] * 100, 0.4, label="real change", color="C0")
    ax.bar([i + 0.2 for i in x], s["erank_change_random"] * 100, 0.4,
           label="random change, same norm", color="C7")  # fmt: skip
    ax.set_ylabel("|eRank(W_after) - eRank(W_before)| / eRank(W_before), %")
    ax.set_title("eRank of the whole weight matrix")
    ax = axes[1]
    ax.bar([i - 0.3 for i in x], s["delta_erank_frac"], 0.2, label="eRank(dW)", color="C0")
    ax.bar([i - 0.1 for i in x], s["noise_erank_frac"], 0.2, label="eRank(noise)", color="C7")
    ax.bar([i + 0.1 for i in x], s["delta_stable_frac"], 0.2, label="stable rank(dW)", color="C1")
    ax.bar([i + 0.3 for i in x], s["noise_stable_frac"], 0.2, label="stable rank(noise)",
           color="C8")  # fmt: skip
    ax.set_ylabel("fraction of max rank")
    ax.set_title("Rank of the delta itself")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(s["run"], rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/timber_check.png", dpi=150)
    plt.close(fig)
    print("\nwrote results/timber_check.csv and results/timber_check.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
