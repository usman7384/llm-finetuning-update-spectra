import torch

from align_geometry.analysis.subspace_overlap import (
    mean_cosine_similarity,
    orthonormal_basis,
    principal_angles,
    random_subspace_baseline,
    top_singular_directions,
)


def test_identical_subspace_has_zero_angle():
    torch.manual_seed(0)
    basis = orthonormal_basis(torch.randn(20, 3))
    angles = principal_angles(basis, basis)
    # acos'(x) blows up near x=1, so fp32 error in the cosine (~1e-7) can show
    # up as an angle error of a few 1e-3 radians. atol reflects that, not a
    # loose test.
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-2)
    assert mean_cosine_similarity(basis, basis) > 0.999


def test_orthogonal_complement_has_right_angle():
    d = 8
    identity = torch.eye(d)
    basis_a = identity[:, :4]
    basis_b = identity[:, 4:]
    angles = principal_angles(basis_a, basis_b)
    assert torch.allclose(angles, torch.full_like(angles, torch.pi / 2), atol=1e-4)
    assert abs(mean_cosine_similarity(basis_a, basis_b)) < 1e-4


def test_random_subspace_baseline_is_near_zero_in_high_dimension():
    # In high ambient dimension, two random low-rank subspaces are nearly orthogonal.
    baseline = random_subspace_baseline(dim=256, rank=4, n_trials=10, seed=1)
    assert 0.0 <= baseline < 0.3


def test_top_singular_directions_recovers_planted_rank():
    torch.manual_seed(0)
    d = 32
    u_true = orthonormal_basis(torch.randn(d, 3))
    v_true = orthonormal_basis(torch.randn(d, 3))
    delta = u_true @ v_true.T  # rank-3 by construction

    u_hat, v_hat = top_singular_directions(delta, rank=3)
    # Recovered left/right subspaces should closely match the planted ones.
    assert mean_cosine_similarity(u_hat, u_true) > 0.99
    assert mean_cosine_similarity(v_hat, v_true) > 0.99
