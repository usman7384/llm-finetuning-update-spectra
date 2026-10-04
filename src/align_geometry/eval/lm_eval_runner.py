"""Thin wrapper over lm-eval's Python API.

Both the style and capability metrics go through here so they load models and
sanitize metric keys the same way. Keeps result differences from being an
artifact of two slightly different wrappers.
"""

from __future__ import annotations

from pathlib import Path


def run_lm_eval_tasks(
    checkpoint_path: str | Path,
    tasks: list[str],
    batch_size: int | str = 8,
    limit: int | float | None = None,
) -> dict[str, float]:
    """Returns a flat {"<task>__<metric>": value} dict.

    Keeps every numeric metric rather than picking one, since which one counts
    as the headline is an analysis decision, not this function's call.
    """
    import lm_eval
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = {
        "pretrained": str(checkpoint_path),
        "dtype": "bfloat16" if device == "cuda" else "float32",
        "attn_implementation": "sdpa",
    }
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=False,
        confirm_run_unsafe_code=False,
        verbosity="WARNING",
    )
    if out is None:
        raise RuntimeError(f"lm_eval.simple_evaluate returned no results for tasks={tasks}")

    flat: dict[str, float] = {}
    for task_name, metrics in out["results"].items():
        for metric_key, value in metrics.items():
            if metric_key == "alias" or not isinstance(value, (int, float)):
                continue
            clean_key = metric_key.replace(",", "__")
            flat[f"{task_name}__{clean_key}"] = float(value)
    return flat
