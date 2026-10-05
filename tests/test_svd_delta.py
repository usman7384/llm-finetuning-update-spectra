import torch

from align_geometry.analysis.svd_delta import (
    compute_deltas,
    deltas_to_records,
    entropy_rank,
    is_analyzable_matrix,
    parse_layer_idx,
    parse_matrix_type,
)


def test_is_analyzable_matrix():
    assert is_analyzable_matrix("model.layers.3.self_attn.q_proj.weight")
    assert is_analyzable_matrix("model.layers.11.mlp.down_proj.weight")
    assert not is_analyzable_matrix("model.layers.3.input_layernorm.weight")
    assert not is_analyzable_matrix("model.embed_tokens.weight")
    assert not is_analyzable_matrix("lm_head.weight")


def test_parse_layer_idx_and_matrix_type():
    key = "model.layers.7.self_attn.o_proj.weight"
    assert parse_layer_idx(key) == 7
    assert parse_matrix_type(key) == "o_proj"


def test_parse_layer_idx_missing_returns_none():
    assert parse_layer_idx("lm_head.weight") is None
    assert parse_matrix_type("lm_head.weight") is None


def test_compute_deltas_low_rank_delta_has_low_effective_rank():
    torch.manual_seed(0)
    d, _r = 64, 32
    before = {"model.layers.0.self_attn.q_proj.weight": torch.randn(d, d)}

    # Construct an exactly rank-4 perturbation and confirm SVD recovers that.
    u = torch.randn(d, 4)
    v = torch.randn(d, 4)
    low_rank_delta = u @ v.T
    after = {
        "model.layers.0.self_attn.q_proj.weight": before["model.layers.0.self_attn.q_proj.weight"]
        + low_rank_delta
    }

    deltas = compute_deltas(before, after)
    assert len(deltas) == 1
    d0 = deltas[0]
    assert d0.layer_idx == 0
    assert d0.matrix_type == "q_proj"
    # Effective rank should be (numerically) close to the true rank of 4, not d.
    assert d0.effective_rank(threshold=0.01) <= 6


def test_compute_deltas_full_rank_delta_has_high_effective_rank():
    torch.manual_seed(0)
    d = 32
    before = {"model.layers.0.mlp.gate_proj.weight": torch.zeros(d, d)}
    after = {"model.layers.0.mlp.gate_proj.weight": torch.randn(d, d)}

    deltas = compute_deltas(before, after)
    d0 = deltas[0]
    # A random full-rank matrix should have most singular values above threshold.
    assert d0.effective_rank(threshold=0.01) >= d - 2


def test_compute_deltas_skips_mismatched_shapes():
    before = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
    after = {"model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4)}
    assert compute_deltas(before, after) == []


def test_deltas_to_records_shape():
    torch.manual_seed(0)
    before = {"model.layers.0.self_attn.v_proj.weight": torch.zeros(16, 16)}
    after = {"model.layers.0.self_attn.v_proj.weight": torch.randn(16, 16)}
    deltas = compute_deltas(before, after)
    records = deltas_to_records(deltas, thresholds=(0.01, 0.1))
    assert len(records) == 1
    record = records[0]
    assert "effective_rank_0.01" in record
    assert "effective_rank_0.1" in record
    assert record["max_rank"] == 16


def test_stable_rank_counts_where_the_energy_is():
    from align_geometry.analysis.svd_delta import energy_top_k, stable_rank

    # the example from the report: one big direction plus 99 small ones.
    # every value clears a 1% threshold, but stable rank sees ~2 directions
    concentrated = torch.tensor([10.0] + [1.0] * 99)
    flat = torch.ones(100)

    assert abs(stable_rank(concentrated) - 1.99) < 1e-6
    assert stable_rank(flat) == 100.0
    assert abs(energy_top_k(concentrated, 1) - 100 / 199) < 1e-6


def test_stable_rank_ignores_scale():
    from align_geometry.analysis.svd_delta import stable_rank

    sv = torch.tensor([3.0, 2.0, 1.0])
    assert abs(stable_rank(sv) - stable_rank(sv * 1000)) < 1e-4


def test_entropy_rank_matches_the_easy_cases():
    # Timber's own examples: one nonzero value gives 1, a flat spectrum gives r
    assert abs(entropy_rank(torch.tensor([1.0, 0.0, 0.0, 0.0])) - 1.0) < 1e-9
    assert abs(entropy_rank(torch.ones(8)) - 8.0) < 1e-9
    sv = torch.tensor([3.0, 2.0, 1.0])
    assert abs(entropy_rank(sv) - entropy_rank(sv * 50)) < 1e-9
