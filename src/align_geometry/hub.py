"""Find model weights, whether they're a local checkpoint or a Hub repo."""

from __future__ import annotations

from pathlib import Path

# config + weights is all the analysis needs, skip tokenizers and the rest
WEIGHT_FILES = ["*.safetensors", "*.json"]


def weights_dir(src: str | Path) -> Path:
    """A local directory as-is, otherwise download the repo (cached by the Hub).

    Set HF_HUB_CACHE to put big downloads on another drive.
    """
    path = Path(src)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(src), allow_patterns=WEIGHT_FILES))
