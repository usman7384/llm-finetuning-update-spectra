"""Runs the style + capability suites and merges them into one record, which
becomes a row of results/eval_table.csv.

get_eval_task_runners() breaks the same suite into individual tasks. The CLI
uses that instead of run_full_eval_suite so an interrupted run doesn't lose
tasks that already finished. IFEval and GSM8K each take over an hour, so this
matters.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from align_geometry.eval.capability_metrics import run_capability_eval
from align_geometry.eval.lm_eval_runner import run_lm_eval_tasks
from align_geometry.eval.style_metrics import run_style_eval, run_xstest_eval


def run_full_eval_suite(
    checkpoint_path: str | Path,
    config_path: str | Path = "configs/eval.yaml",
    limit: int | float | None = None,
) -> dict[str, float]:
    """Whole suite in one shot, no per-task resume. Fine for smoke tests; the
    CLI uses get_eval_task_runners for real runs."""
    results: dict[str, float] = {}
    results.update(run_style_eval(checkpoint_path, config_path, limit=limit))
    results.update(run_capability_eval(checkpoint_path, config_path, limit=limit))
    return results


def get_eval_task_runners(
    config_path: str | Path = "configs/eval.yaml",
) -> list[tuple[str, str, Callable[[str, int | float | None], dict[str, float]]]]:
    """Ordered (task_name, metric_key_prefix, runner) triples from configs/eval.yaml.

    The prefix is what the CLI looks for in an existing eval_table.csv row to
    decide a task is already done, so it has to match what the runners actually
    prefix their keys with.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen_cfg = cfg.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 8)

    runners: list[tuple[str, str, Callable[[str, int | float | None], dict[str, float]]]] = []

    for entry in cfg["style_metrics"]:
        if entry["source"] == "lm_eval_task":
            task = entry["task"]

            def _run_lm_eval(
                ckpt: str, limit: int | float | None, _task: str = task
            ) -> dict[str, float]:
                return run_lm_eval_tasks(ckpt, [_task], batch_size=batch_size, limit=limit)

            runners.append((task, f"{task}__", _run_lm_eval))
        elif entry["name"] == "xstest_refusal_calibration":

            def _run_xstest(ckpt: str, limit: int | float | None) -> dict[str, float]:
                xstest_limit = int(limit) if isinstance(limit, int) else None
                return run_xstest_eval(
                    ckpt,
                    batch_size=batch_size * 2,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 128),
                    limit=xstest_limit,
                )

            runners.append(("xstest", "xstest__", _run_xstest))
        else:
            raise ValueError(f"Unknown style metric entry: {entry}")

    for entry in cfg["capability_metrics"]:
        task = entry["task"]

        def _run_lm_eval(
            ckpt: str, limit: int | float | None, _task: str = task
        ) -> dict[str, float]:
            return run_lm_eval_tasks(ckpt, [_task], batch_size=batch_size, limit=limit)

        runners.append((task, f"{task}__", _run_lm_eval))

    return runners
