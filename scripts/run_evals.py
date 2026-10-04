#!/usr/bin/env python
"""Run the eval suite on one checkpoint and add/replace its row in
results/eval_table.csv.

    uv run python scripts/run_evals.py --checkpoint Qwen/Qwen2.5-0.5B --label base
    uv run python scripts/run_evals.py --checkpoint checkpoints/sft --label sft

Each task is saved as soon as it finishes, and a rerun with the same --label
skips the ones already there. The full suite takes hours (IFEval alone was over
two on the base model) so this gets used a lot. --force redoes everything.

For a quick check that the pipeline works, add --limit 8. Don't report those
numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read_table(out_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not out_path.exists():
        return [], []
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _upsert_row(out_path: Path, row: dict[str, str | float], label: str) -> None:
    """Replace the row matching `label`, or append. Rewrites the whole file
    each time, which is fine at this size and means nothing is lost if the
    next task gets interrupted."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, rows = _read_table(out_path)

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    replaced = False
    for i, r in enumerate(rows):
        if r.get("label") == label:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="local checkpoint dir or hub id")
    parser.add_argument("--label", required=True, help="row name, e.g. base, sft, lora_r8")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--out", default="results/eval_table.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run N examples per task, for testing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo tasks that are already in the csv",
    )
    args = parser.parse_args()

    from align_geometry.eval.harness import get_eval_task_runners

    out_path = Path(args.out)
    _, existing_rows = _read_table(out_path)
    existing_row = next((r for r in existing_rows if r.get("label") == args.label), None)

    row: dict[str, str | float] = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "timestamp_utc": (existing_row or {}).get("timestamp_utc") or datetime.now(UTC).isoformat(),
        "limit": args.limit if args.limit is not None else "",
    }
    if existing_row:
        row.update(existing_row)
        row["checkpoint"] = str(args.checkpoint)
        row["limit"] = args.limit if args.limit is not None else existing_row.get("limit", "")

    for task_name, prefix, runner in get_eval_task_runners(args.config):
        already_done = not args.force and any(
            k.startswith(prefix) and v not in (None, "") for k, v in row.items()
        )
        if already_done:
            print(f"[{args.label}] skip {task_name} (already in {args.out}; --force to redo)")
            continue

        print(f"[{args.label}] running {task_name} ...")
        metrics = runner(args.checkpoint, args.limit)
        row.update(metrics)
        row["timestamp_utc"] = datetime.now(UTC).isoformat()
        _upsert_row(out_path, row, args.label)
        print(f"[{args.label}] {task_name} done, saved to {args.out}:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
