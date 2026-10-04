#!/usr/bin/env python
"""Pull and subsample the SFT data without training anything, so you can look
at it first. Optional: train_sft.py does the same thing itself.

    uv run python scripts/prepare_data.py --config configs/sft.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg["data"].get("format") == "text" or "split" in cfg["data"]:
        sys.exit("this only handles SFT-style configs (sft.yaml, sft_seed43.yaml, lora_sweep.yaml)")

    from align_geometry.data.sft_dataset import load_sft_dataset

    indices_path = Path("data/processed") / f"{config_path.stem}_indices.json"
    train_ds, val_ds = load_sft_dataset(
        dataset_name=cfg["data"]["dataset"],
        n_examples=cfg["data"]["n_examples"],
        subsample_seed=cfg["data"]["subsample_seed"],
        val_fraction=cfg["data"]["val_fraction"],
        indices_out=indices_path,
    )
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")
    print(f"indices written to {indices_path}")
    print("first example:", train_ds[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
