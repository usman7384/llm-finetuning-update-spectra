#!/usr/bin/env python
"""Save the reference weights the SVD comparisons are measured against.

    uv run python scripts/save_reference_models.py            # base model only
    uv run python scripts/save_reference_models.py --random   # plus the noise floor

checkpoints/base is the untouched base model as bf16 safetensors, so every
delta is taken against the same file rather than whatever the HF cache holds.

--random also saves two freshly initialized copies of the architecture
(seeds 1 and 2). Running svd_deltas.py on that pair gives the noise floor.
They're ~1GB each and only needed for that one comparison.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--random", action="store_true", help="also save two random inits")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.save_pretrained("checkpoints/base")
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained("checkpoints/base")
    print("saved checkpoints/base")

    if args.random:
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        for seed, name in [(1, "random_init_a"), (2, "random_init_b")]:
            torch.manual_seed(seed)
            AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(
                f"checkpoints/{name}"
            )
            print(f"saved checkpoints/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
