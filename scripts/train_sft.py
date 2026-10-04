#!/usr/bin/env python
"""Full fine-tune from a config. Used for the main SFT run, the seed-43 repeat
and both plain-LM controls.

    uv run python scripts/train_sft.py --config configs/sft.yaml

Ctrl+C whenever. Running it again picks up from the last checkpoint, and if
the run already finished it just says so and exits. --force starts over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="retrain even if already finished")
    args = parser.parse_args()

    from align_geometry.train.sft import train_sft

    print(f"saved to {train_sft(args.config, force=args.force)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
