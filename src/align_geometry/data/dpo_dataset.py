"""UltraFeedback into TRL's DPO format.

The source stores chosen/rejected as full [user, assistant] message lists, so
we pull out just the assistant turn and keep the prompt separate. Same split
as sft_dataset uses, which keeps the chat template consistent with the SFT run
this starts from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset


def _to_dpo_format(example: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [example["chosen"][-1]],
        "rejected": [example["rejected"][-1]],
    }


def load_dpo_dataset(
    dataset_name: str,
    split: str,
    n_examples: int,
    subsample_seed: int,
    val_fraction: float,
    indices_out: str | Path | None = None,
) -> tuple[Dataset, Dataset]:
    """Returns (train, val). Same indices-persisting convention as
    load_sft_dataset."""
    full = load_dataset(dataset_name, split=split)

    rng = np.random.default_rng(subsample_seed)
    perm = rng.permutation(len(full))
    n = min(n_examples, len(full))
    subset_idx = perm[:n]

    n_val = max(1, int(n * val_fraction))
    val_idx = subset_idx[:n_val]
    train_idx = subset_idx[n_val:]

    if indices_out is not None:
        indices_out = Path(indices_out)
        indices_out.parent.mkdir(parents=True, exist_ok=True)
        indices_out.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "subsample_seed": subsample_seed,
                    "n_source_rows": len(full),
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                },
                indent=2,
            )
        )

    train = full.select(train_idx.tolist()).map(_to_dpo_format, remove_columns=full.column_names)
    val = full.select(val_idx.tolist()).map(_to_dpo_format, remove_columns=full.column_names)
    return train, val
