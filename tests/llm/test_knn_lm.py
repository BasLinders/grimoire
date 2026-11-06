"""Tests for the scoped kNN-LM core (docs/KNN_LM.md, PR 1).

Gate criteria:
- ``knn_distribution`` is a proper distribution, aggregates by value token,
  and gives missing neighbors (index -1) zero weight.
- ``fuse`` reduces to the LM at λ=0 and always yields a normalised
  distribution.
- ``ScopedStore`` search matches brute force, clamps k to the store size,
  and ``add_transitions`` pairs hidden[t] with ids[t+1] over the span.
- ``LambdaGate`` stays inside (0, lam_max), starts at the documented
  constant, and ``calibrate_temperature`` sets T to the median distance.
- ``fused_logprobs`` is a strict no-op without a store, and gradients reach
  the gate's parameters (needed for gate fitting).
- Gate persistence round-trips and refuses a mismatched checkpoint.
"""

import math

import pytest
import torch

from grimoire_ai.llm.inference.knn_lm import (
    KNNConfig,
    LambdaGate,
    ScopedStore,
    checkpoint_tag,
    fuse,
    fused_logprobs,
    knn_distribution,
    load_gate,
    save_gate,
)

V = 11
DIM = 8


# --------------------------------------------------------- knn_distribution


def test_knn_distribution_sums_to_one_and_respects_values() -> None:
    D = torch.tensor([[0.1, 0.5, 2.0]])
    I = torch.tensor([[0, 1, 2]])
    values = torch.tensor([3, 3, 7])
    p = knn_distribution(D, I, values, V, temperature=1.0)
    assert torch.allclose(p.sum(-1), torch.ones(1), atol=1e-5)
    assert p[0, 3] > p[0, 7] > 0  # two neighbors vote for token 3
    assert p[0, 0] == 0


def test_knn_distribution_masks_missing_neighbors() -> None:
    D = torch.tensor([[0.1, 1e30]])
    I = torch.tensor([[0, -1]])
    p = knn_distribution(D, I, torch.tensor([4]), V, temperature=1.0)
    assert torch.allclose(p[0, 4], torch.tensor(1.0))


def test_knn_distribution_temperature_flattens() -> None:
    D = torch.tensor([[1.0, 4.0]])
    I = torch.tensor([[0, 1]])
    values = torch.tensor([1, 2])
    sharp = knn_distribution(D, I, values, V, temperature=0.1)
    flat = knn_distribution(D, I, values, V, temperature=100.0)
    assert sharp[0, 1] > flat[0, 1] > 0.5


# --------------------------------------------------------------------- fuse


def test_fuse_lam_zero_matches_lm() -> None:
    logits = torch.randn(2, V)
    p = torch.softmax(torch.randn(2, V), -1)
    out = fuse(logits, p, torch.zeros(2))
    assert torch.allclose(out, logits.log_softmax(-1), atol=1e-4)


def test_fuse_is_a_distribution() -> None:
    logits = torch.randn(3, V)
    p = torch.softmax(torch.randn(3, V), -1)
    out = fuse(logits, p, torch.tensor([0.1, 0.5, 0.8]))
    assert torch.allclose(out.exp().sum(-1), torch.ones(3), atol=1e-4)


def test_fuse_accepts_half_precision_logits() -> None:
    logits = torch.randn(2, V).half()
    p = torch.softmax(torch.randn(2, V), -1)
    out = fuse(logits, p, torch.tensor([0.3, 0.3]))
    assert out.dtype == torch.float32
    assert torch.allclose(out.exp().sum(-1), torch.ones(2), atol=1e-3)


# ------------------------------------------------------------- ScopedStore


def test_store_search_matches_brute_force() -> None:
    torch.manual_seed(0)
    store = ScopedStore(DIM)
    keys = torch.randn(20, DIM)
    store.add(keys, torch.arange(20) % V)
    q = torch.randn(3, DIM)

    D, I = store.search(q, k=5)

    brute = ((q[:, None, :] - keys[None, :, :]) ** 2).sum(-1)
    exp_D, exp_I = brute.topk(5, dim=-1, largest=False)
    assert torch.equal(I, exp_I)
    assert torch.allclose(D, exp_D, atol=1e-4)
    assert torch.all(D[:, :-1] <= D[:, 1:])  # ascending


def test_store_search_clamps_k_and_rejects_empty() -> None:
    store = ScopedStore(DIM)
    with pytest.raises(ValueError):
        store.search(torch.randn(1, DIM), k=4)
    store.add(torch.randn(3, DIM), torch.tensor([1, 2, 3]))
    D, I = store.search(torch.randn(2, DIM), k=16)
    assert D.shape == I.shape == (2, 3)


def test_store_add_validates_shapes() -> None:
    store = ScopedStore(DIM)
    with pytest.raises(ValueError):
        store.add(torch.randn(3, DIM + 1), torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        store.add(torch.randn(3, DIM), torch.tensor([1, 2]))


def test_add_transitions_pairs_hidden_t_with_next_token() -> None:
    T = 10
    hidden = torch.arange(T, dtype=torch.float32)[:, None].repeat(1, DIM)
    ids = list(range(100, 100 + T))
    store = ScopedStore(DIM)

    store.add_transitions(hidden, ids, start=3, end=7)

    assert store.size == 4
    assert store.values.tolist() == ids[3:7]           # values are the span itself
    assert store.keys[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]  # keys start one earlier


def test_add_transitions_rejects_bad_spans() -> None:
    hidden = torch.zeros(5, DIM)
    store = ScopedStore(DIM)
    with pytest.raises(ValueError):
        store.add_transitions(hidden, [1] * 5, start=0, end=3)  # no hidden before 0
    with pytest.raises(ValueError):
        store.add_transitions(hidden, [1] * 5, start=2, end=6)  # past the end
    with pytest.raises(ValueError):
        store.add_transitions(hidden, [1] * 4, start=1, end=2)  # length mismatch
    store.add_transitions(hidden, [1] * 5, start=2, end=2)       # empty span is fine
    assert store.size == 0


# -------------------------------------------------------------- LambdaGate


def test_gate_initial_lambda_is_documented_constant() -> None:
    gate = LambdaGate(lam_max=0.8)
    D = torch.rand(4, 5) * 100
    lm_logp = torch.randn(4, V).log_softmax(-1)
    lam = gate(D, lm_logp)
    expected = 0.8 / (1 + math.e)  # lam_max * sigmoid(-1)
    assert torch.allclose(lam, torch.full((4,), expected), atol=1e-6)


def test_gate_output_stays_in_range() -> None:
    gate = LambdaGate(lam_max=0.5)
    with torch.no_grad():
        gate.lin.weight.fill_(50.0)
        gate.lin.bias.fill_(50.0)
    lam = gate(torch.rand(3, 4), torch.randn(3, V).log_softmax(-1))
    assert torch.all(lam > 0) and torch.all(lam <= 0.5)


def test_gate_rejects_bad_construction() -> None:
    with pytest.raises(ValueError):
        LambdaGate(lam_max=1.0)
    with pytest.raises(ValueError):
        LambdaGate(init_temperature=0.0)


def test_calibrate_temperature_uses_median_positive_distance() -> None:
    gate = LambdaGate()
    D = torch.tensor([[0.0, 100.0, 300.0], [200.0, float("inf"), 400.0]])
    t = gate.calibrate_temperature(D)
    assert t == pytest.approx(200.0)  # median of {100, 200, 300, 400} (lower middle)
    assert gate.temperature.item() == pytest.approx(200.0, rel=1e-5)
    with pytest.raises(ValueError):
        gate.calibrate_temperature(torch.zeros(2, 2))


# ---------------------------------------------------------- fused_logprobs


def test_fused_logprobs_without_store_is_plain_lm() -> None:
    logits = torch.randn(2, V)
    hidden = torch.randn(2, DIM)
    gate, cfg = LambdaGate(), KNNConfig()
    expected = logits.log_softmax(-1)
    assert torch.allclose(fused_logprobs(logits, hidden, None, gate, cfg), expected)
    assert torch.allclose(
        fused_logprobs(logits, hidden, ScopedStore(DIM), gate, cfg), expected
    )


def test_fused_logprobs_boosts_stored_token_and_is_normalised() -> None:
    torch.manual_seed(0)
    store = ScopedStore(DIM)
    key = torch.randn(1, DIM)
    store.add(key, torch.tensor([7]))
    logits = torch.zeros(1, V)  # uniform LM
    out = fused_logprobs(logits, key, store, LambdaGate(), KNNConfig())

    assert out.argmax(-1).item() == 7
    assert torch.allclose(out.exp().sum(-1), torch.ones(1), atol=1e-4)


def test_gradients_reach_gate_parameters() -> None:
    """Gate fitting (PR 3) minimises NLL of fused log-probs w.r.t. the gate."""
    torch.manual_seed(0)
    store = ScopedStore(DIM)
    store.add(torch.randn(6, DIM), torch.tensor([1, 2, 3, 1, 2, 3]))
    gate = LambdaGate()
    out = fused_logprobs(torch.randn(4, V), torch.randn(4, DIM), store, gate, KNNConfig())
    loss = torch.nn.functional.nll_loss(out, torch.tensor([1, 2, 3, 1]))
    loss.backward()
    assert gate.lin.weight.grad is not None and gate.lin.weight.grad.abs().sum() > 0
    assert gate.log_temp.grad is not None and gate.log_temp.grad.abs() > 0


# ------------------------------------------------------------- persistence


def test_gate_save_load_round_trip(tmp_path) -> None:
    gate = LambdaGate(lam_max=0.6)
    with torch.no_grad():
        gate.lin.weight.copy_(torch.tensor([[0.1, -0.2, 0.3]]))
        gate.log_temp.fill_(math.log(123.0))
    cfg = KNNConfig(k=8, lam_max=0.6)
    path = tmp_path / "gate.pt"

    save_gate(path, gate, cfg, checkpoint_tag="abc")
    loaded, loaded_cfg, tag = load_gate(path, expected_checkpoint_tag="abc")

    assert tag == "abc"
    assert loaded_cfg == cfg
    assert loaded.lam_max == 0.6
    assert not loaded.training
    for name, val in gate.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], val)


def test_load_gate_rejects_checkpoint_mismatch(tmp_path) -> None:
    path = tmp_path / "gate.pt"
    save_gate(path, LambdaGate(), KNNConfig(), checkpoint_tag="old")
    with pytest.raises(ValueError, match="different checkpoint"):
        load_gate(path, expected_checkpoint_tag="new")
    load_gate(path)  # no expectation given: loads


def test_checkpoint_tag_tracks_content(tmp_path) -> None:
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"weights-v1")
    b.write_bytes(b"weights-v1")
    assert checkpoint_tag(a) == checkpoint_tag(b)
    b.write_bytes(b"weights-v2")
    assert checkpoint_tag(a) != checkpoint_tag(b)
