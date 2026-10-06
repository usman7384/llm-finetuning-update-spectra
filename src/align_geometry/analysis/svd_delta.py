"""SVD of a weight delta (W_after - W_before).

Pure linear algebra over two checkpoints' state dicts, no dependency on the
training code, so it works on any pair of safetensors checkpoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

# The Qwen2.5 (Llama-style) projections we look at. Norms, embeddings and
# lm_head are skipped: 1-D or vocab-shaped, not part of the depth story.
MATRIX_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_MATRIX_TYPE_RE = re.compile(r"\.(" + "|".join(MATRIX_TYPES) + r")\.weight$")


@dataclass
class MatrixDelta:
    """SVD result for one weight matrix's delta."""

    key: str
    layer_idx: int | None
    matrix_type: str | None
    shape: tuple[int, int]
    singular_values: torch.Tensor  # 1-D, descending
    fro_norm: float

    def effective_rank(self, threshold: float = 0.01) -> int:
        """Count of singular values >= threshold * sigma_1.

        The 0.01 threshold is a common convention (Timber uses entropy_rank
        below instead, not this). Worth knowing this measure is
        weak: it says nothing about how energy is spread among the values it
        counts, and it reports ~99% of max rank for basically any dense matrix.
        See docs/report.md; stable rank is the better instrument.
        """
        if self.singular_values.numel() == 0:
            return 0
        sigma_1 = self.singular_values[0].item()
        if sigma_1 == 0:
            return 0
        return int((self.singular_values >= threshold * sigma_1).sum().item())

    def normalized_spectrum(self) -> torch.Tensor:
        """Singular values divided by sigma_1, for cross-matrix comparison."""
        if self.singular_values.numel() == 0:
            return self.singular_values
        return self.singular_values / self.singular_values[0]


def stable_rank(singular_values: torch.Tensor) -> float:
    """sum(sigma_i^2) / sigma_1^2. Roughly how many directions the energy is
    spread over. Unlike effective_rank it has no threshold and ignores scale."""
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        return 0.0
    energy = singular_values.float() ** 2
    return float(energy.sum() / energy[0])


def entropy_rank(singular_values: torch.Tensor, gamma: float = 1.0) -> float:
    """Roy & Vetterli's effective rank: exp of the entropy of sigma^gamma / sum.

    This is what Timber calls eRank (they use gamma=1). Between 1 and the
    number of singular values, and like stable rank it ignores scale.
    """
    p = singular_values.double().clamp(min=0) ** gamma
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(torch.exp(-(p * p.log()).sum()))


def energy_top_k(singular_values: torch.Tensor, k: int) -> float:
    """Share of sum(sigma_i^2) sitting in the k largest singular values."""
    energy = singular_values.float() ** 2
    total = float(energy.sum())
    return float(energy[:k].sum()) / total if total > 0 else 0.0


def is_analyzable_matrix(key: str) -> bool:
    """True for the 2-D attention/MLP projections. Excludes embeddings,
    lm_head and anything 1-D."""
    return bool(_MATRIX_TYPE_RE.search(key))


def parse_layer_idx(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def parse_matrix_type(key: str) -> str | None:
    m = _MATRIX_TYPE_RE.search(key)
    return m.group(1) if m else None


def load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a model's weights as a flat state dict.

    Accepts either a directory containing `model.safetensors` (or sharded
    `model-*-of-*.safetensors`) or a direct path to a single safetensors file.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        single = path / "model.safetensors"
        if single.exists():
            return load_file(single)
        shards = sorted(path.glob("model-*-of-*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"No model.safetensors or sharded safetensors files found under {path}"
            )
        state_dict: dict[str, torch.Tensor] = {}
        for shard in shards:
            state_dict.update(load_file(shard))
        return state_dict
    return load_file(path)


class LazyWeights:
    """Reads one tensor at a time from a checkpoint's safetensors files.

    load_state_dict holds the whole model in RAM, which is fine up to ~2B but
    not for an 8B pair on a 16GB machine. Same keys, fetched on demand.
    """

    def __init__(self, checkpoint_path: str | Path):
        from safetensors import safe_open

        path = Path(checkpoint_path)
        files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors files under {path}")
        self._handles = [safe_open(str(f), framework="pt") for f in files]
        self._where = {k: h for h in self._handles for k in h.keys()}

    def keys(self):
        return self._where.keys()

    def __iter__(self):
        return iter(self._where)

    def __contains__(self, key: str) -> bool:
        return key in self._where

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._where[key].get_tensor(key)


def compute_deltas(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    keys: list[str] | None = None,
) -> list[MatrixDelta]:
    """SVD each matching matrix's delta.

    Silently skips keys missing from either side or that aren't analyzable;
    scripts/svd_deltas.py does the coverage warning.
    """
    if keys is None:
        keys = [k for k in before if k in after and is_analyzable_matrix(k)]

    results: list[MatrixDelta] = []
    for key in keys:
        w_before = before[key]
        w_after = after[key]
        if w_before.shape != w_after.shape or w_before.ndim != 2:
            continue

        delta = w_after.float() - w_before.float()
        # svdvals skips U/V, which we don't need for the spectrum.
        singular_values = torch.linalg.svdvals(delta)

        results.append(
            MatrixDelta(
                key=key,
                layer_idx=parse_layer_idx(key),
                matrix_type=parse_matrix_type(key),
                shape=tuple(delta.shape),
                singular_values=singular_values,
                fro_norm=float(torch.linalg.norm(delta)),
            )
        )
    return results


def deltas_to_records(
    deltas: list[MatrixDelta], thresholds: tuple[float, ...] = (0.01, 0.05, 0.1)
) -> list[dict]:
    """One row per matrix, ready for a DataFrame.

    Full spectra get saved separately as .pt files, since max rank varies by
    matrix shape and won't flatten into a single table.
    """
    records = []
    for d in deltas:
        record = {
            "key": d.key,
            "layer_idx": d.layer_idx,
            "matrix_type": d.matrix_type,
            "rows": d.shape[0],
            "cols": d.shape[1],
            "max_rank": min(d.shape),
            "sigma_1": float(d.singular_values[0]) if d.singular_values.numel() else 0.0,
            "fro_norm": d.fro_norm,
        }
        for t in thresholds:
            record[f"effective_rank_{t}"] = d.effective_rank(threshold=t)
        records.append(record)
    return records
