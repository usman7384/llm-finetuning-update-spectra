#!/usr/bin/env python
"""Save the reference weights the SVD comparisons are measured against.

    uv run python scripts/save_reference_models.py --random
    uv run python scripts/save_reference_models.py --model HuggingFaceTB/SmolLM2-360M \\
        --prefix smollm2_ --random

Saves the untouched base model as bf16 safetensors to checkpoints/<prefix>base,
so every delta is taken against the same file rather than whatever the HF cache
holds.

--random also saves two freshly initialized copies of the architecture (seeds 1
and 2) to checkpoints/<prefix>random_init_a and _b. Running svd_deltas.py on
that pair gives the noise floor. Only needed for that one comparison, so they
can be deleted afterwards.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prefix", default="", help="e.g. smollm2_ (empty for the Qwen runs)")
    parser.add_argument("--random", action="store_true", help="also save two random inits")
    args = parser.parse_args()

    base_dir = f"checkpoints/{args.prefix}base"
    AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).save_pretrained(base_dir)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(base_dir)
    print(f"saved {base_dir}")

    if args.random:
        cfg = AutoConfig.from_pretrained(args.model)
        for seed, suffix in [(1, "a"), (2, "b")]:
            out = f"checkpoints/{args.prefix}random_init_{suffix}"
            torch.manual_seed(seed)
            AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(out)
            print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
