#!/usr/bin/env python
"""DPO on top of the SFT checkpoint.

    uv run python scripts/train_dpo.py --config configs/dpo.yaml

Took about 12 hours on the 3060. Same resume behaviour as train_sft.py.
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

    from align_geometry.train.dpo import train_dpo

    print(f"saved to {train_dpo(args.config, force=args.force)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
