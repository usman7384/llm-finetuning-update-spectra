"""Capability eval (GSM8K, ARC-Challenge) through lm-eval.

Takes a local checkpoint dir or a hub id, returns {task_metric: score}.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from align_geometry.eval.lm_eval_runner import run_lm_eval_tasks


def run_capability_eval(
    checkpoint_path: str | Path,
    config_path: str | Path,
    limit: int | float | None = None,
) -> dict[str, float]:
    """Run everything under `capability_metrics` in configs/eval.yaml.

    `limit` is for smoke-testing the pipeline. Don't report a limited run.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tasks = [
        entry["task"] for entry in cfg["capability_metrics"] if entry["source"] == "lm_eval_task"
    ]
    batch_size = cfg.get("generation", {}).get("batch_size", 8)
    return run_lm_eval_tasks(checkpoint_path, tasks, batch_size=batch_size, limit=limit)
