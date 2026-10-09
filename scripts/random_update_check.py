#!/usr/bin/env python
"""Same-size random updates: does eRank(W) notice, does the model?

Timber reads an unchanged effective rank of W, before vs after post-training,
as evidence that post-training is superficial. If a random update of the same
size also leaves eRank(W) unchanged but wrecks the model, an unchanged eRank
says nothing about how much a change matters.

On Qwen3-0.6B (one of Timber's pairs), for every attention/MLP matrix we build
three updates of the same Frobenius norm:

  real      W_instruct - W_base
  gaussian  iid Gaussian, rescaled to the real update's norm
  spectrum  the real update's singular values on random singular vectors

and report the change in eRank(W) and the perplexity of each edited model on
held-out wikitext. Run it when the GPU is free (a few minutes).

    uv run python scripts/random_update_check.py

Writes results/v2/random_update_check.csv.
"""

from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    LazyWeights,
    entropy_rank,
    is_analyzable_matrix,
)
from align_geometry.hub import weights_dir  # noqa: E402

BASE, INSTRUCT = "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"
KINDS = ["real", "gaussian", "spectrum"]


def random_orthonormal(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, k, generator=g))
    return q * torch.sign(torch.diagonal(r))  # Haar-distributed columns


def make_update(kind: str, delta: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    if kind == "real":
        return delta
    if kind == "gaussian":
        noise = torch.randn(delta.shape, generator=g)
        return noise * (torch.linalg.norm(delta) / torch.linalg.norm(noise))
    # same singular values, random directions
    s = torch.linalg.svdvals(delta)
    m, n = delta.shape
    return (random_orthonormal(m, s.numel(), g) * s) @ random_orthonormal(n, s.numel(), g).T


@torch.no_grad()
def perplexity(model, tok, texts: list[str], device: str) -> float:
    nll = n = 0.0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        loss = model(input_ids=ids, labels=ids).loss
        nll += float(loss) * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return math.exp(nll / n)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = LazyWeights(weights_dir(BASE))
    inst = LazyWeights(weights_dir(INSTRUCT))
    keys = [k for k in base if k in inst and is_analyzable_matrix(k)]
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    val = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in val["text"] if len(t) > 500][:200]

    # base model's own perplexity, for reference
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(device).eval()
    rows = [{"update": "none (base)", "perplexity": perplexity(model, tok, texts, device)}]

    for kind in KINDS:
        params = dict(model.named_parameters())
        changes = []
        for key in keys:
            w_b = base[key].float()
            delta = inst[key].float() - w_b
            g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
            upd = make_update(kind, delta, g)
            er_b = entropy_rank(torch.linalg.svdvals(w_b.to(device)).cpu())
            er_a = entropy_rank(torch.linalg.svdvals((w_b + upd).to(device)).cpu())
            changes.append(abs(er_a - er_b) / er_b)
            params[key].data.copy_((w_b + upd).to(device))
        ppl = perplexity(model, tok, texts, device)
        rows.append(
            {
                "update": kind,
                "erank_change_pct": 100 * sum(changes) / len(changes),
                "perplexity": ppl,
            }
        )
        print(rows[-1], flush=True)
        # back to the base weights before the next kind
        for key in keys:
            params[key].data.copy_(base[key].float().to(device))

    df = pd.DataFrame(rows)
    out = Path("results/v2/random_update_check.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
