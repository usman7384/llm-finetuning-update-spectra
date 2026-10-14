#!/usr/bin/env python
"""How much of the first-round (pure bf16) fine-tunes never moved.

The first round of runs kept the weights in bf16 with no fp32 master copy, so
any step smaller than half a bf16 unit was rounded away. This counts, for each
of those checkpoints, the share of attention/MLP weights still exactly equal
to the base model. The paper quotes it to explain why those runs were redone.

    uv run python scripts/v1_unchanged.py

Writes results/v1_unchanged.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix  # noqa: E402
from align_geometry.hub import weights_dir  # noqa: E402

QWEN, SMOL = "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M"
RUNS = {
    "sft": QWEN,
    "sft_seed43": QWEN,
    "dpo": "checkpoints/sft",  # DPO started from the SFT run
    "control_lm": QWEN,
    "control_lm_seed43": QWEN,
    "control_code": QWEN,
    "control_code_seed43": QWEN,
    "smollm2_sft": SMOL,
    "smollm2_control_lm": SMOL,
    "smollm2_control_code": SMOL,
}


def main() -> int:
    rows = []
    for run, base_id in RUNS.items():
        path = Path("checkpoints") / run
        if not path.exists():
            continue
        base, after = LazyWeights(weights_dir(base_id)), LazyWeights(path)
        same = total = 0
        for k in base:
            if not is_analyzable_matrix(k) or k not in after:
                continue
            a, b = after[k], base[k]
            same += int((a.float() == b.float()).sum())
            total += a.numel()
        rows.append({"run": run, "stored_dtype": str(after[k].dtype).replace("torch.", ""),
                     "unchanged_fraction": same / total})  # fmt: skip
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/v1_unchanged.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
