#!/usr/bin/env python
"""Eval results across the LoRA ranks, style vs capability.

IFEval is shown as the fraction of SFT's gain each rank gets back. GSM8K and
ARC are shown as raw scores, because that ratio doesn't work for them: SFT made
GSM8K worse, so there's no gain to get back, and ARC's change is smaller than
its own stderr. XSTest is raw for the same reason.

    uv run python scripts/plot_lora_sweep.py

Writes results/lora_sweep.csv and results/lora_sweep.png.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]

STYLE = {
    "ifeval__prompt_level_strict_acc__none": "IFEval prompt-level strict",
    "ifeval__inst_level_loose_acc__none": "IFEval instruction-level loose",
}
CAPABILITY = {
    "gsm8k__exact_match__strict-match": "GSM8K exact match",
    "arc_challenge__acc_norm__none": "ARC-Challenge acc_norm",
}
REFUSAL = {
    "xstest__refusal_rate_safe": "XSTest refusals, safe prompts",
    "xstest__refusal_rate_unsafe": "XSTest refusals, unsafe prompts",
}
GSM8K = "gsm8k__exact_match__strict-match"


def log_rank_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("LoRA rank")


def main() -> int:
    df = pd.read_csv("results/eval_table.csv").set_index("label")
    base, sft = df.loc["base"], df.loc["sft"]

    rows = []
    for rank in RANKS:
        label = f"lora_r{rank}"
        if label not in df.index:
            print(f"no eval row for {label} yet, skipping")
            continue
        lora = df.loc[label]
        row = {"rank": rank}
        for col in STYLE:
            gain = sft[col] - base[col]
            row[f"{col}__fraction_recovered"] = (lora[col] - base[col]) / gain
            row[f"{col}__raw"] = lora[col]
        for col in [*CAPABILITY, *REFUSAL]:
            row[f"{col}__raw"] = lora[col]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("results/lora_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for col, name in STYLE.items():
        ax.plot(out["rank"], out[f"{col}__fraction_recovered"], "o-", label=name)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="same as full SFT")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.5)
    log_rank_axis(ax)
    ax.set_ylabel("fraction of SFT's gain recovered")
    ax.set_title("Style (IFEval)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for col, name in CAPABILITY.items():
        ax.plot(out["rank"], out[f"{col}__raw"], "s--", label=name)
    ax.axhline(base[GSM8K], color="C0", linestyle=":", linewidth=1, label="GSM8K, base model")
    ax.axhline(sft[GSM8K], color="C0", linestyle="-.", linewidth=1, label="GSM8K, full SFT")
    log_rank_axis(ax)
    ax.set_ylabel("score")
    ax.set_title("Capability (GSM8K, ARC)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for col, name in REFUSAL.items():
        ax.plot(out["rank"], out[f"{col}__raw"], "^-", label=name)
    log_rank_axis(ax)
    ax.set_ylabel("refusal rate")
    ax.set_title("Refusals (XSTest)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("results/lora_sweep.png", dpi=150)
    plt.close(fig)
    print("wrote results/lora_sweep.csv and results/lora_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
