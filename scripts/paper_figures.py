#!/usr/bin/env python
"""Figures for the paper, as vector PDFs, straight from the result CSVs.

Nothing here recomputes anything expensive; it only reads what the other
scripts saved, so the numbers in the figures can't drift from the results.

    uv run python scripts/paper_figures.py

Writes paper/figures/*.pdf.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.eval.gsm8k import pass_at_k  # noqa: E402

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,  # embed TrueType, journals complain about Type 3
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)
OUT = Path("paper/figures")
NOISE = "0.6"  # grey for every noise baseline

QWEN = [
    ("SFT", "sft (vs base)", "svd_sft", "sft"),
    ("DPO", "dpo (vs sft)", "svd_dpo", "dpo"),
    ("wikitext", "wikitext LM (vs base)", "svd_control_lm", "control_lm"),
    ("code", "python code LM (vs base)", "svd_control_code", "control_code"),
]
SMOL = [
    ("SFT", "sft (vs base)", "svd_smollm2_sft", "smollm2_sft"),
    ("wikitext", "wikitext LM (vs base)", "svd_smollm2_control_lm", "smollm2_control_lm"),
    ("code", "python code LM (vs base)", "svd_smollm2_control_code", "smollm2_control_code"),
]
PAIR_NAMES = {
    "smollm2-135m": "SmolLM2-135M",
    "smollm2-360m": "SmolLM2-360M",
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "smollm2-1.7b": "SmolLM2-1.7B",
    "qwen3-8b": "Qwen3-8B",
}


def threshold_frac(svd_dir: str) -> float:
    df = pd.read_csv(f"results/{svd_dir}/matrix_summary.csv")
    return float((df["effective_rank_0.01"] / df["max_rank"]).mean())


def family_rows(runs, stable_csv: str, noise_dir: str, noise_erank_from: str) -> pd.DataFrame:
    """The four rank measures for each run plus its noise baseline."""
    sr = pd.read_csv(stable_csv)
    rows = []
    for label, source, svd_dir, tc in runs:
        sub = sr[sr["source"] == source]
        rows.append(
            {
                "run": label,
                "threshold": threshold_frac(svd_dir),
                "k90": sub["k_for_90pct_frac"].mean(),
                "erank": pd.read_csv(f"results/timber_check/{tc}.csv")["delta_erank_frac"].mean(),
                "stable": sub["stable_rank_frac"].mean(),
            }
        )
    noise = sr[sr["source"] == "random init pair"]
    rows.append(
        {
            "run": "noise",
            "threshold": threshold_frac(noise_dir),
            "k90": noise["k_for_90pct_frac"].mean(),
            # same as the random-init pair's own spectra (0.928 / 0.908)
            "erank": pd.read_csv(f"results/timber_check/{noise_erank_from}.csv")[
                "noise_erank_frac"
            ].mean(),
            "stable": noise["stable_rank_frac"].mean(),
        }
    )
    return pd.DataFrame(rows)


def fig_metrics() -> None:
    """Four rank measures, real updates vs noise, both families."""
    fams = [
        ("Qwen2.5-0.5B", family_rows(QWEN, "results/stable_rank.csv", "svd_control_random", "sft")),
        (
            "SmolLM2-360M",
            family_rows(
                SMOL, "results/stable_rank_smollm2.csv", "svd_smollm2_random", "smollm2_sft"
            ),
        ),
    ]
    metrics = [
        ("threshold", "threshold count\n($\\sigma_i \\geq 0.01\\,\\sigma_1$)"),
        ("k90", "rank for 90%\nof energy"),
        ("erank", "entropy eRank\n($\\gamma=1$)"),
        ("stable", "stable rank\n$\\|\\Delta W\\|_F^2 / \\|\\Delta W\\|_2^2$"),
    ]
    colors = {"SFT": "C0", "DPO": "C1", "wikitext": "C2", "code": "C3", "noise": NOISE}
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.1), sharey=True)
    for ax, (col, title) in zip(axes, metrics, strict=True):
        x = 0.0
        ticks, labels = [], []
        for fam, df in fams:
            start = x
            for _, r in df.iterrows():
                ax.bar(x, r[col], 0.8, color=colors[r["run"]])
                x += 1
            ticks.append((start + x - 1) / 2)
            labels.append(fam.split("-")[0])
            x += 0.8
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("fraction of max rank")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, list(colors), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "metrics_vs_noise.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_released() -> None:
    """Threshold count and stable rank on public base/instruct pairs."""
    df = pd.read_csv("results/released_pairs.csv").set_index("pair")
    order = [p for p in PAIR_NAMES if p in df.index]
    df = df.loc[order]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
    for ax, real, noise, title in [
        (axes[0], "threshold_rank_frac", "noise_threshold_rank_frac", "threshold count (0.01)"),
        (axes[1], "stable_rank_frac", "noise_stable_rank_frac", "stable rank"),
    ]:
        ax.bar(x - 0.2, df[real], 0.4, color="C0", label="base $\\to$ instruct")
        ax.bar(x + 0.2, df[noise], 0.4, color=NOISE, label="Gaussian, same shape")
        ax.set_xticks(x)
        ax.set_xticklabels([PAIR_NAMES[p] for p in order], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("fraction of max rank")
    for i, ratio in enumerate(df["noise_over_real"]):
        axes[1].text(i, df["noise_stable_rank_frac"].iloc[i] + 0.01, f"{ratio:.1f}x",
                     ha="center", fontsize=6)  # fmt: skip
    axes[0].set_ylim(0.9, 1.0)
    axes[1].set_ylim(0, 0.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "released_pairs.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_erank_perturbation() -> None:
    """How much eRank(W) moves for the real update vs a random one of the same size."""
    names = {
        "dpo": "DPO (ours)",
        "control_code": "code LM (ours)",
        "sft": "SFT (ours)",
        "control_lm": "wikitext LM (ours)",
        "smollm2_control_code": "SmolLM2 code (ours)",
        "smollm2_control_lm": "SmolLM2 wikitext (ours)",
        "smollm2_sft": "SmolLM2 SFT (ours)",
        "qwen2.5-1.5b": "Qwen2.5-1.5B",
        "qwen3-8b": "Qwen3-8B",
        "smollm2-1.7b": "SmolLM2-1.7B",
        "qwen3-0.6b": "Qwen3-0.6B",
        "qwen2.5-0.5b": "Qwen2.5-0.5B",
        "smollm2-360m": "SmolLM2-360M",
        "smollm2-135m": "SmolLM2-135M",
    }
    rows = []
    for key, label in names.items():
        df = pd.read_csv(f"results/timber_check/{key}.csv")
        rows.append(
            (label, df["erank_change_real"].mean() * 100, df["erank_change_random"].mean() * 100)
        )
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.barh(y + 0.2, [r[1] for r in rows], 0.4, color="C0", label="real update")
    ax.barh(y - 0.2, [r[2] for r in rows], 0.4, color=NOISE, label="random, same norm")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("relative change in eRank($W$), %")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "erank_perturbation.pdf", bbox_inches="tight")
    plt.close(fig)


def pass_curve(variant: str, limit: int) -> list[float]:
    text = Path(f"results/timber_cutoff/{variant}.jsonl").read_text(encoding="utf-8")
    recs = [r for r in map(json.loads, text.splitlines()) if r["idx"] < limit]
    n = len(recs[0]["correct"])
    return [100 * np.mean([pass_at_k(n, sum(r["correct"]), k) for r in recs]) for k in (1, 2, 4, 8)]


def fig_cutoff() -> None:
    """pass@k for Timber's edit with each cutoff."""
    ks = [1, 2, 4, 8]
    styles = {
        "instruct": ("no edit", NOISE, "-"),
        "erank_l0.5": ("eRank cutoff, $\\lambda$=0.5 (Timber)", "C0", "-"),
        "stable_l0.5": ("stable-rank cutoff, $\\lambda$=0.5", "C2", "-"),
        "erank_l0.0": ("eRank cutoff, $\\lambda$=0", "C0", ":"),
        "stable_l0.0": ("stable-rank cutoff, $\\lambda$=0", "C3", ":"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
    for ax, limit, variants, title in [
        (axes[0], 300, list(styles), "first 300 problems"),
        (axes[1], 1319, ["instruct", "erank_l0.5", "stable_l0.5"], "all 1,319 problems"),
    ]:
        for v in variants:
            label, color, ls = styles[v]
            ax.plot(ks, pass_curve(v, limit), marker="o", ms=3, color=color, ls=ls, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_xlabel("$k$")
        ax.set_title(title)
    axes[0].set_ylabel("GSM8K pass@$k$, %")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_passk.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [p for p in ["results/stable_rank.csv", "results/released_pairs.csv"]
               if not os.path.exists(p)]  # fmt: skip
    if missing or not glob.glob("results/timber_cutoff/*.jsonl"):
        print("results missing, run the analysis scripts first:", missing)
        return 1
    fig_metrics()
    fig_released()
    fig_erank_perturbation()
    fig_cutoff()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
