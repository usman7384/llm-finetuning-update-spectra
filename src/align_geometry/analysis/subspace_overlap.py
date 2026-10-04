"""Do two low-rank subspaces point the same way?

Used to compare the top singular directions of a full fine-tune's delta
against what a LoRA adapter found at the same rank.
"""

from __future__ import annotations

import torch


def orthonormal_basis(vectors: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (via QR) spanning the same subspace as `vectors`'s
    columns. Input shape (d, r); output shape (d, r)."""
    q, _ = torch.linalg.qr(vectors.float())
    return q


def principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> torch.Tensor:
    """Principal angles in radians, ascending. Inputs need orthonormal columns.

    SVD of basis_a.T @ basis_b gives the cosines directly; clamped before acos
    because fp error can push them just past 1.
    """
    m = basis_a.T @ basis_b
    cosines = torch.linalg.svdvals(m)
    cosines = torch.clamp(cosines, -1.0, 1.0)
    return torch.acos(cosines)


def mean_cosine_similarity(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    """Mean cosine of the principal angles. 1.0 is identical, 0.0 orthogonal."""
    angles = principal_angles(basis_a, basis_b)
    return float(torch.cos(angles).mean())


def random_subspace_baseline(dim: int, rank: int, n_trials: int = 20, seed: int = 0) -> float:
    """What chance alone gives you at this dimension and rank.

    Without it a raw "0.3 cosine similarity" is uninterpretable, since the
    baseline climbs steeply as rank approaches dim.
    """
    generator = torch.Generator().manual_seed(seed)
    similarities = []
    for _ in range(n_trials):
        a = orthonormal_basis(torch.randn(dim, rank, generator=generator))
        b = orthonormal_basis(torch.randn(dim, rank, generator=generator))
        similarities.append(mean_cosine_similarity(a, b))
    return sum(similarities) / len(similarities)


def top_singular_directions(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`rank` left and right singular vectors, both orthonormal.

    Compare U_r against a LoRA adapter's B (output side) and V_r against its
    A (input side).
    """
    u, _, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    return u[:, :rank], vh[:rank, :].T
