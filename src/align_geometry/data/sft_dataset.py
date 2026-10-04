"""Alpaca-shaped data (instruction/input/output) into TRL's conversational
prompt-completion format, so SFTTrainer applies the chat template itself and
masks the prompt out of the loss. Beats hand-rolling template strings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset


@dataclass
class SFTExample:
    instruction: str
    input: str
    output: str


def _to_prompt_completion(example: dict) -> dict:
    user_content = example["instruction"]
    if example["input"]:
        user_content = f"{example['instruction']}\n\n{example['input']}"
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": example["output"]}],
    }


def load_sft_dataset(
    dataset_name: str,
    n_examples: int,
    subsample_seed: int,
    val_fraction: float,
    indices_out: str | Path | None = None,
) -> tuple[Dataset, Dataset]:
    """Returns (train, val).

    Subsampling goes through our own numpy RNG rather than Dataset.shuffle so
    the actual row indices can be written to `indices_out`. That way the
    subsample is reconstructable from the manifest, not just from "rerun the
    same code with the same library versions".
    """
    full = load_dataset(dataset_name, split="train")

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
                    "subsample_seed": subsample_seed,
                    "n_source_rows": len(full),
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                },
                indent=2,
            )
        )

    train = full.select(train_idx.tolist()).map(
        _to_prompt_completion, remove_columns=full.column_names
    )
    val = full.select(val_idx.tolist()).map(_to_prompt_completion, remove_columns=full.column_names)
    return train, val
