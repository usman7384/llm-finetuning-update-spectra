#!/usr/bin/env python
"""Run this first. Checks imports, that torch can see the GPU, and how much
VRAM you've got. All the memory settings assume a ~6GB card, so you want to
know early if your machine is different.

    uv run python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import sys


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module_name)
        return True, getattr(mod, "__version__", "unknown")
    except ImportError as e:
        return False, str(e)


def main() -> int:
    ok = True

    print("== core ==")
    for name in ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "safetensors"]:
        found, version = check_import(name)
        print(f"  {name:15s} {'OK' if found else 'MISSING':8s} {version}")
        ok = ok and found

    found, version = check_import("align_geometry")
    print(f"  align_geometry  {'OK' if found else 'MISSING':8s} {version}")
    ok = ok and found

    print("\n== gpu ==")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
                if mem_gb < 8:
                    print("  under 8GB: full fine-tuning needs the 8-bit optimizer and")
                    print("  gradient checkpointing from configs/model_qwen2.5_0.5b.yaml")
        else:
            print("  no CUDA device visible. Training on CPU isn't realistic here.")
            ok = False
    except ImportError:
        print("  torch isn't installed, run `uv sync` first")
        ok = False

    print("\n== extras ==")
    for name, extra in [("lm_eval", "eval"), ("bitsandbytes", "quant")]:
        found, _ = check_import(name)
        print(f"  {name:15s} {'OK' if found else f'missing (uv sync --extra {extra})'}")

    print("\nlooks good" if ok else "\nsomething's missing, see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
