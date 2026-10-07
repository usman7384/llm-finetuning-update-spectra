#!/usr/bin/env python
"""Make sure every file docs/report.md quotes numbers from is still around.

The report itself is written by hand. This just catches a CSV or plot going
missing after something gets regenerated or renamed.

    uv run python scripts/check_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

NEEDED = [
    "docs/report.md",
    "results/eval_table.csv",
    "results/lora_sweep.csv",
    "results/lora_sweep.png",
    "results/sft_vs_dpo.png",
    "results/lora_overlap.csv",
    "results/lora_overlap.png",
    "results/stable_rank.csv",
    "results/stable_rank.png",
    "results/svd_sft/matrix_summary.csv",
    "results/svd_sft/spectra.png",
    "results/svd_sft/effective_rank_by_depth.png",
    "results/svd_sft_seed43/matrix_summary.csv",
    "results/svd_dpo/matrix_summary.csv",
    "results/svd_control_lm/matrix_summary.csv",
    "results/svd_control_code/matrix_summary.csv",
    "results/svd_control_random/matrix_summary.csv",
    "checkpoints/sft/manifest.json",
    "checkpoints/sft_seed43/manifest.json",
    "checkpoints/dpo/manifest.json",
    "checkpoints/control_lm/manifest.json",
    "checkpoints/control_code/manifest.json",
    "checkpoints/lora_sweep/sweep_summary.csv",
    # follow-ups: second family, seeds, public pairs, Timber, error bars
    "results/stable_rank_smollm2.csv",
    "results/stable_rank_smollm2.png",
    "results/svd_control_lm_seed43/matrix_summary.csv",
    "results/svd_control_code_seed43/matrix_summary.csv",
    "results/released_pairs.csv",
    "results/released_pairs.png",
    "results/released_pairs/qwen3-8b.csv",
    "results/timber_check.csv",
    "results/timber_check.png",
    "results/timber_check/qwen3-8b.csv",
    "results/timber_cutoff.csv",
    "results/timber_cutoff_full.csv",
    "results/error_bars.csv",
    # pretrained weights, the paper's second reference besides noise
    "results/weights_base/spectra",
    "results/weights_smollm2_base/spectra",
]


def main() -> int:
    missing = [p for p in NEEDED if not Path(p).exists()]
    for p in missing:
        print(f"missing: {p}")
    if not missing:
        print(f"all {len(NEEDED)} files present")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
