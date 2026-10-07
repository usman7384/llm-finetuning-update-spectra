#!/usr/bin/env python
"""Save full spectra for released base/instruct pairs, plus a bf16 rounding floor.

released_pairs.py and timber_check.py only kept summary numbers, so every new
measure meant downloading the weights again. This saves the singular values
of every update dW = W_instruct - W_base, so later analysis needs no weights.

Released checkpoints are stored in bf16, which rounds every weight to the
nearest bf16 value. That rounding adds a small noise matrix to dW, and it
lands in the tail of the spectrum, which is exactly what a threshold count
counts. For each matrix we also save the spectrum of one simulated rounding
error: independent uniform noise in +-half a bf16 unit at each weight's
magnitude, for both checkpoints, which is what storing them in bf16 adds.

    uv run python scripts/released_spectra.py --pairs qwen3-0.6b smollm2-135m
    uv run python scripts/released_spectra.py --device cuda     # when the GPU is free

Writes results/released_spectra/<pair>/{delta,rounding}/<key>.pt. Resumable
per matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix  # noqa: E402
from align_geometry.hub import weights_dir  # noqa: E402

PAIRS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    "qwen3-8b": ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
}


def bf16_half_ulp(w: torch.Tensor) -> torch.Tensor:
    """Half the spacing between adjacent bf16 values at each entry's magnitude.

    bf16 keeps 8 significant bits, so the spacing at |w| in [2^e, 2^(e+1)) is
    2^(e-7). Zeros get the spacing of the smallest normal number.
    """
    a = w.float().abs().clamp(min=torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(a)) - 7) / 2


def rounding_noise(w_base: torch.Tensor, w_inst: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """One draw of the error that bf16 storage adds to dW: e_inst - e_base."""
    u1 = torch.rand(w_inst.shape, generator=g) * 2 - 1
    u2 = torch.rand(w_base.shape, generator=g) * 2 - 1
    return u1 * bf16_half_ulp(w_inst) - u2 * bf16_half_ulp(w_base)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=list(PAIRS))
    parser.add_argument("--device", default="cpu", help="cpu while a training run uses the GPU")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    for name in args.pairs:
        base_repo, inst_repo = PAIRS[name]
        out = Path("results/released_spectra") / name
        (out / "delta").mkdir(parents=True, exist_ok=True)
        (out / "rounding").mkdir(parents=True, exist_ok=True)
        base = LazyWeights(weights_dir(base_repo))
        inst = LazyWeights(weights_dir(inst_repo))
        keys = [k for k in base if k in inst and is_analyzable_matrix(k)]
        g = torch.Generator().manual_seed(0)
        done = 0
        for i, key in enumerate(keys, 1):
            # draw the noise even when skipping, so a resumed run matches a full one
            w_b, w_i = base[key], inst[key]
            noise = rounding_noise(w_b, w_i, g)
            d_path, r_path = out / "delta" / f"{key}.pt", out / "rounding" / f"{key}.pt"
            if d_path.exists() and r_path.exists():
                continue
            delta = w_i.float() - w_b.float()
            if delta.ndim != 2 or delta.abs().max() == 0:
                continue
            torch.save(torch.linalg.svdvals(delta.to(args.device)).cpu(), d_path)
            torch.save(torch.linalg.svdvals(noise.to(args.device)).cpu(), r_path)
            done += 1
            if i % 25 == 0 or i == len(keys):
                print(f"{name}: {i}/{len(keys)} matrices", flush=True)
        print(f"{name}: {done} new spectra")
    return 0


if __name__ == "__main__":
    sys.exit(main())
