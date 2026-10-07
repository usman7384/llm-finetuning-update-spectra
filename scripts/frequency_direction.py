#!/usr/bin/env python
"""Does an update's top direction push the output toward frequent tokens?

The shuffled-word control (same words as wikitext, no order) gives the most
concentrated update of all our runs. A simple explanation: scrambled text can
only teach the model its word frequencies, and shifting the output toward
frequent tokens takes very few directions.

Test: for each layer's down_proj (it writes straight into the residual
stream), take the top left singular vector u of the update, map it to the
vocabulary through the final norm and the (tied) unembedding, and correlate
that logit shift with log token frequency in the run's own training data.
The sign of u is arbitrary, so we report |correlation|, against the same
statistic for random directions.

CPU only, so it can run next to a training job.

    uv run python scripts/frequency_direction.py

Writes results/v2/frequency_direction.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import LazyWeights  # noqa: E402

BASE = "checkpoints/base"
RUNS = {
    "SFT": ("checkpoints/v2/sft", "data/processed/v2_sft_indices.json"),
    "wikitext": ("checkpoints/v2/control_lm", "data/processed/v2_control_lm_indices.json"),
    "code": ("checkpoints/v2/control_code", "data/processed/v2_control_code_indices.json"),
    "shuffled words": (
        "checkpoints/v2/control_shuffled",
        "data/processed/v2_control_shuffled_indices.json",
    ),
}


def token_counts(indices_file: str, tokenizer, vocab: int) -> np.ndarray:
    """Token counts over the run's training rows. Word shuffling doesn't change
    which words occur, so the shuffled run uses its rows' original text."""
    meta = json.loads(Path(indices_file).read_text())
    if meta["dataset_name"] == "yahma/alpaca-cleaned":
        ds = load_dataset(meta["dataset_name"], split="train")
        rows = ds.select(meta["train_indices"])
        texts = [f"{r['instruction']}\n{r['input']}\n{r['output']}" for r in rows]
    else:
        ds = (
            load_dataset(meta["dataset_name"], meta["config_name"], split=meta["split"])
            if meta.get("config_name")
            else load_dataset(meta["dataset_name"], split=meta["split"])
        )
        texts = ds.select(meta["train_indices"])[meta["text_field"]]
    counts = np.zeros(vocab)
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        np.add.at(counts, ids, 1)
    return counts


def abs_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y).abs() / (x.norm() * y.norm()))


def main() -> int:
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    base = LazyWeights(BASE)
    unembed = base["model.embed_tokens.weight"].float()  # tied with lm_head
    gain = base["model.norm.weight"].float()
    vocab = unembed.shape[0]
    keys = [k for k in base if k.endswith("mlp.down_proj.weight")]
    g = torch.Generator().manual_seed(0)
    rows = []
    for run, (ckpt, idx) in RUNS.items():
        counts = token_counts(idx, tok, vocab)
        seen = torch.from_numpy(counts > 0)
        logfreq = torch.from_numpy(np.log(counts[counts > 0]))
        after = LazyWeights(ckpt)
        for key in keys:
            delta = after[key].float() - base[key].float()
            u = torch.linalg.svd(delta, full_matrices=False)[0][:, 0]
            shift = unembed @ (gain * u)
            rand = torch.randn(u.numel(), 50, generator=g)
            rand_corr = [abs_corr((unembed @ (gain * r))[seen], logfreq) for r in rand.T]
            rows.append(
                {
                    "run": run,
                    "layer": int(key.split("layers.")[1].split(".")[0]),
                    "top_direction_corr": abs_corr(shift[seen], logfreq),
                    "random_corr_mean": float(np.mean(rand_corr)),
                    "random_corr_p95": float(np.percentile(rand_corr, 95)),
                }
            )
        print(f"{run}: done", flush=True)

    df = pd.DataFrame(rows)
    out = Path("results/v2/frequency_direction.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.groupby("run", sort=False)[
        ["top_direction_corr", "random_corr_mean", "random_corr_p95"]
    ].mean()
    summary["layers_above_random_p95"] = (
        df.assign(a=df.top_direction_corr > df.random_corr_p95)
        .groupby("run", sort=False)["a"]
        .sum()
    )
    print(summary.round(3).to_string())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
