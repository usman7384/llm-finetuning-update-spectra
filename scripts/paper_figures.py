#!/usr/bin/env python
"""Figures for the paper, as vector PDFs, straight from the result files.

Nothing here recomputes anything expensive; it only reads what the other
scripts saved (v2_analysis.py, released_measures.py, timber_check.py,
random_update_check.py, timber_cutoff.py), so the figures can't drift from
the results.

    uv run python scripts/paper_figures.py

Writes paper/figures/*.pdf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
# Okabe-Ito: readable for the common kinds of colour blindness
RUN_COLORS = {
    "SFT": "#0072B2",
    "wikitext": "#009E73",
    "code": "#E69F00",
    "shuffled words": "#CC79A7",
    "pretrained W": "#000000",
    "noise": NOISE,
}
MEASURE_LABELS = {
    "threshold": "threshold\ncount",
    "erank": "entropy\neRank",
    "k90": "90%\nenergy",
    "participation": "participation\nratio",
    "stable": "stable\nrank",
}


def fig_gaps() -> None:
    """How far below noise each measure puts each update, both models."""
    gaps = pd.read_csv("results/v2/gaps.csv")
    gaps = gaps[gaps["vs"] == "noise"]
    order = list(MEASURE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True)
    for ax, model in zip(axes, ["Qwen2.5-0.5B", "SmolLM2-360M"], strict=True):
        sub = gaps[gaps["model"] == model]
        for run in ["SFT", "wikitext", "code", "shuffled words"]:
            r = sub[sub["run"] == run].set_index("measure").loc[order]
            x = np.arange(len(order))
            yerr = [100 * (r["gap"] - r["lo"]), 100 * (r["hi"] - r["gap"])]
            ax.errorbar(x, 100 * r["gap"], yerr=yerr, marker="o", ms=3, capsize=2,
                        color=RUN_COLORS[run], label=run)  # fmt: skip
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([MEASURE_LABELS[m] for m in order], fontsize=7)
        ax.set_title(model)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("how far below noise, %")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "gaps_by_measure.pdf", bbox_inches="tight")
    plt.close(fig)


def load_spectra(result_dir: str) -> list[torch.Tensor]:
    return [
        torch.load(p, weights_only=True).double()
        for p in sorted(Path(f"results/{result_dir}/spectra").glob("*.pt"))
    ]


def mean_curves(result_dir: str, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative energy share and sigma_i / sigma_1 against i / r.

    Each matrix's curve is interpolated onto a common grid of i / r (matrices
    have different ranks), then averaged: arithmetic mean for the energy
    share, geometric mean for the decay. Below 1/r a curve is held at its
    first value, which is why small matrices flatten out at the far left.
    """
    cum, decay = [], []
    for sv in load_spectra(result_dir):
        sv = sv.numpy()
        x = np.arange(1, len(sv) + 1) / len(sv)
        energy = sv**2
        cum.append(np.interp(grid, x, np.cumsum(energy) / energy.sum()))
        decay.append(np.interp(grid, x, sv / sv[0]))
    return np.mean(cum, axis=0), np.exp(np.mean(np.log(np.maximum(decay, 1e-12)), axis=0))


def fig_spectrum() -> None:
    """Where the energy sits: a few strong directions and a long tail (Qwen2.5-0.5B)."""
    # start at 1/128, the smallest rank among the matrices, so no curve is extrapolated
    grid = np.logspace(np.log10(1 / 128), 0, 300)
    # SFT, wikitext and code nearly coincide: SFT goes on top, thick, the other two dashed
    lines = [
        ("wikitext", "v2/svd_control_lm", (0, (4, 2)), 1.0),
        ("code", "v2/svd_control_code", (0, (1, 1.5)), 1.2),
        ("shuffled words", "v2/svd_control_shuffled", "-", 1.0),
        ("noise", "svd_control_random", "-", 1.0),
        ("pretrained W", "weights_base", "--", 1.0),
        ("SFT", "v2/svd_sft", "-", 2.2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
    for label, d, ls, lw in lines:
        cum, decay = mean_curves(d, grid)
        style = {"color": RUN_COLORS[label], "ls": ls, "lw": lw, "label": label,
                 "zorder": 3 if label == "SFT" else 2, "alpha": 0.7 if label == "SFT" else 1}  # fmt: skip
        axes[0].plot(grid, 100 * cum, **style)
        axes[1].plot(grid, decay, **style)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("top fraction of singular directions, $i/r$")
    axes[0].set_ylabel("share of energy, %")
    axes[0].set_title("cumulative energy")
    axes[0].legend(fontsize=6, loc="upper left")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$i/r$")
    axes[1].set_ylabel("$\\sigma_i / \\sigma_1$ (geometric mean)")
    axes[1].set_title("singular value decay")
    fig.tight_layout()
    fig.savefig(OUT / "spectrum_head_tail.pdf", bbox_inches="tight")
    plt.close(fig)


RELEASED = [
    # (key, label, group)
    ("smollm2-135m", "SmolLM2-135M", "instruct release"),
    ("smollm2-360m", "SmolLM2-360M", "instruct release"),
    ("qwen2.5-0.5b", "Qwen2.5-0.5B", "instruct release"),
    ("qwen3-0.6b", "Qwen3-0.6B", "instruct release"),
    ("qwen2.5-1.5b", "Qwen2.5-1.5B", "instruct release"),
    ("smollm2-1.7b", "SmolLM2-1.7B", "instruct release"),
    ("qwen3-8b", "Qwen3-8B", "instruct release"),
    # no olmo2-1b-sft: the released SFT checkpoint is not a fine-tune of the
    # released base (cosine of the weights ~0, results/released_sizes.csv)
    ("olmo2-1b-dpo", "OLMo2-1B DPO", "OLMo 2 stage"),
    ("olmo2-1b-rl", "OLMo2-1B RL", "OLMo 2 stage"),
    ("qwen2.5-coder-0.5b", "Coder-0.5B", "code / math"),
    ("qwen2.5-math-1.5b", "Math-1.5B", "code / math"),
    ("qwen2.5-coder-1.5b", "Coder-1.5B", "code / math"),
]


def fig_released() -> None:
    """Threshold count and stable rank on released updates, against noise."""
    df = pd.read_csv("results/released_measures.csv").set_index("pair")
    rows = [r for r in RELEASED if r[0] in df.index]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, real, noise, title in [
        (axes[0], "real_threshold", "noise_threshold", "threshold count (0.01)"),
        (axes[1], "real_stable", "noise_stable", "stable rank"),
    ]:
        ax.bar(x - 0.2, df.loc[[r[0] for r in rows], real], 0.4, color="#0072B2",
               label="released update")  # fmt: skip
        ax.bar(x + 0.2, df.loc[[r[0] for r in rows], noise], 0.4, color=NOISE,
               label="Gaussian, same shape")  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels([r[1] for r in rows], rotation=90, fontsize=6)
        ax.set_title(title)
        ax.set_ylabel("fraction of max rank")
        # separators between the three groups
        for i in range(1, len(rows)):
            if rows[i][2] != rows[i - 1][2]:
                ax.axvline(i - 0.5, color="0.85", lw=0.8)
    axes[0].set_ylim(0.85, 1.0)
    axes[1].set_ylim(0, 0.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "released_pairs.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_random_update() -> None:
    """eRank(W) barely moves under real or random updates; the model's predictions do."""
    names = {
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
        d = pd.read_csv(f"results/timber_check/{key}.csv")
        rows.append((label, 100 * d.erank_change_real.mean(), 100 * d.erank_change_random.mean()))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(rows))
    axes[0].barh(y + 0.2, [r[1] for r in rows], 0.4, color="#0072B2", label="real update")
    axes[0].barh(y - 0.2, [r[2] for r in rows], 0.4, color=NOISE, label="random, same size")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    axes[0].set_xlabel("change in eRank($W$), %")
    axes[0].set_title("effective rank of $W$ barely moves")
    axes[0].legend(fontsize=6, loc="lower right")

    p = pd.read_csv("results/v2/random_update_check.csv").set_index("update")
    order = [("none (base)", "base"), ("gaussian", "random,\nGaussian"),
             ("spectrum", "random,\nsame\nspectrum"), ("real", "real\nupdate")]  # fmt: skip
    vals = [p.loc[k, "perplexity"] for k, _ in order]
    bars = axes[1].bar(range(len(order)), vals, color=["#000000", NOISE, NOISE, "#0072B2"])
    for b, (k, _) in zip(bars, order, strict=True):
        if k != "none (base)":
            note = f"eRank\n{p.loc[k, 'erank_change_pct']:+.2f}%"
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, note,
                         ha="center", fontsize=6)  # fmt: skip
    axes[1].set_ylim(0, 1.2 * max(vals))
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([lbl for _, lbl in order], fontsize=7)
    axes[1].set_ylabel("wikitext perplexity")
    axes[1].set_title("predictions change (Qwen3-0.6B base)")
    fig.tight_layout()
    fig.savefig(OUT / "random_update.pdf", bbox_inches="tight")
    plt.close(fig)


def pass_curve(variant: str, limit: int) -> list[float] | None:
    path = Path(f"results/timber_cutoff/{variant}.jsonl")
    recs = [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r["idx"] < limit]  # fmt: skip
    if len(recs) < limit:
        return None  # not finished on this many problems
    n = len(recs[0]["correct"])
    return [100 * np.mean([pass_at_k(n, sum(r["correct"]), k) for r in recs]) for k in (1, 2, 4, 8)]


def fig_cutoff() -> None:
    """pass@k for the edit variants."""
    ks = [1, 2, 4, 8]
    styles = {
        "instruct": ("no edit", NOISE, "-"),
        "erank_l0.2": ("Timber, $\\lambda=0.2$", "#56B4E9", "--"),
        "erank_l0.5": ("Timber, $\\lambda=0.5$", "#0072B2", "-"),
        "stable_l0.5": ("tail edit, stable-rank cutoff", "#009E73", "-"),
        "uniform_stable_l0.5": ("uniform scaling, same size", "#CC79A7", "-"),
        "stable_l0.0": ("tail removed, stable-rank cutoff", "#D55E00", ":"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
    for ax, limit, variants, title in [
        (axes[0], 300, list(styles), "first 300 problems"),
        (axes[1], 1319, ["instruct", "erank_l0.2", "erank_l0.5", "stable_l0.5",
                         "uniform_stable_l0.5"],
         "all 1,319 problems"),
    ]:  # fmt: skip
        for v in variants:
            curve = pass_curve(v, limit)
            if curve is None:
                continue
            label, color, ls = styles[v]
            ax.plot(ks, curve, marker="o", ms=3, color=color, ls=ls, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_xlabel("$k$")
        ax.set_title(title)
    axes[0].set_ylabel("GSM8K pass@$k$, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6,
               bbox_to_anchor=(0.5, -0.12))  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "cutoff_passk.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_gaps()
    fig_spectrum()
    fig_released()
    fig_random_update()
    fig_cutoff()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
