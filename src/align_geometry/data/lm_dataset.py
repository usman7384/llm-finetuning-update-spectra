"""Plain text loading for the non-alignment controls.

The random-init control only shows the SFT delta is more concentrated than
noise. It can't show that's an alignment thing. Fine-tuning the same model on
raw prose or source code, at the same token budget and hyperparameters,
isolates that: if those deltas look the same, concentration is just what
fine-tuning does.

Selection is matched on tokens rather than example count, since document
lengths vary a lot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset


def load_lm_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    target_tokens: int,
    tokenizer,
    subsample_seed: int,
    val_fraction: float,
    min_chars: int = 200,
    text_field: str = "text",
    max_tokens_per_example: int | None = None,
    indices_out: str | Path | None = None,
) -> tuple[Dataset, Dataset]:
    """Take documents until `target_tokens` is reached, then split (train, val).

    min_chars drops wikitext's empty lines and section headers, which are ~56%
    of rows and carry no signal.

    Set max_tokens_per_example to the trainer's max_length for any corpus with
    documents longer than that. The trainer truncates, so counting a
    document's full length would credit the budget for tokens it never sees.
    Doesn't matter for wikitext (nothing selected went over 1024 tokens) but
    code files blow past it constantly.
    """
    full = (
        load_dataset(dataset_name, config_name, split=split)
        if config_name
        else load_dataset(dataset_name, split=split)
    )

    rng = np.random.default_rng(subsample_seed)
    perm = rng.permutation(len(full))

    texts = full[text_field]
    selected_idx: list[int] = []
    selected_texts: list[str] = []
    total_tokens = 0

    for i in perm:
        text = texts[int(i)]
        if len(text) < min_chars:
            continue
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if max_tokens_per_example is not None:
            # Only count what the trainer will actually see.
            n_tokens = min(n_tokens, max_tokens_per_example)
        selected_idx.append(int(i))
        selected_texts.append(text)
        total_tokens += n_tokens
        if total_tokens >= target_tokens:
            break

    n_val = max(1, int(len(selected_texts) * val_fraction))
    val_texts, train_texts = selected_texts[:n_val], selected_texts[n_val:]
    val_idx, train_idx = selected_idx[:n_val], selected_idx[n_val:]

    if indices_out is not None:
        indices_out = Path(indices_out)
        indices_out.parent.mkdir(parents=True, exist_ok=True)
        indices_out.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "split": split,
                    "subsample_seed": subsample_seed,
                    "min_chars": min_chars,
                    "text_field": text_field,
                    "max_tokens_per_example": max_tokens_per_example,
                    "target_tokens": target_tokens,
                    "achieved_tokens": total_tokens,
                    "n_source_rows": len(full),
                    "train_indices": train_idx,
                    "val_indices": val_idx,
                },
                indent=2,
            )
        )

    return Dataset.from_dict({"text": train_texts}), Dataset.from_dict({"text": val_texts})
