#!/usr/bin/env python
"""Train every (rank, learning rate) combination in configs/lora_sweep.yaml,
then pick the best learning rate per rank by validation loss.

    uv run python scripts/lora_sweep.py --config configs/lora_sweep.yaml

That's 24 runs and roughly 37 hours on a 3060 laptop. Each point is skipped if
it already finished, so stopping and restarting is fine.

Writes checkpoints/lora_sweep/sweep_summary.csv with every point, not just the
winners.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="retrain points that already finished")
    args = parser.parse_args()

    from align_geometry.train.lora import train_lora

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_dir = Path(cfg["train"]["save_dir"])
    rows: list[dict] = []

    for rank in cfg["sweep"]["ranks"]:
        points = []
        for lr in cfg["sweep"]["learning_rate_candidates"]:
            print(f"=== rank {rank}, lr {lr} ===")
            run_dir = train_lora(args.config, rank, lr, force=args.force)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            points.append((lr, run_dir, manifest))

        best_lr, best_dir, best = min(points, key=lambda p: p[2]["observed_eval_loss"])
        print(f"rank {rank}: best lr {best_lr} (eval loss {best['observed_eval_loss']:.4f})")

        for lr, run_dir, manifest in points:
            rows.append(
                {
                    "rank": rank,
                    "learning_rate": lr,
                    "eval_loss": manifest["observed_eval_loss"],
                    "train_loss": manifest["observed_train_loss"],
                    "expected_train_tokens": manifest["expected_train_tokens"],
                    "checkpoint": str(run_dir),
                    "is_best_lr_for_rank": lr == best_lr,
                }
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "sweep_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
