"""Unit tests for self-supervised contrastive embedding training.

Coverage:
    contrastive_loss — perfect alignment collapses to ~0, uniform similarity
                        gives log(batch) (no signal), gradients flow.
    EmbedTuner       — loss decreases on a fixed batch, batch retrieval
                        accuracy converges, batch_size=1 is rejected.
"""

import pytest
import torch
import torch.nn.functional as F

from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.training.embed_tune import EmbedTuner, contrastive_loss


def _small_config(dropout: float = 0.5) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# contrastive_loss
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_perfect_alignment_is_near_zero(self):
        """Identical, perfectly-separated rows in both views -> loss ~ 0."""
        emb = torch.eye(4)
        loss = contrastive_loss(emb, emb.clone(), temperature=0.05)
        assert loss.item() < 1e-3

    def test_uniform_similarity_gives_no_signal(self):
        """All rows identical -> softmax is uniform -> loss == log(batch)."""
        emb_a = torch.ones(4, 8)
        emb_b = torch.ones(4, 8)
        loss = contrastive_loss(emb_a, emb_b, temperature=0.05)
        assert loss.item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-4)

    def test_gradients_flow_to_both_views(self):
        emb_a = torch.randn(4, 8, requires_grad=True)
        emb_b = torch.randn(4, 8, requires_grad=True)
        loss = contrastive_loss(emb_a, emb_b)
        loss.backward()
        assert emb_a.grad is not None and emb_a.grad.abs().sum() > 0
        assert emb_b.grad is not None and emb_b.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# EmbedTuner
# ---------------------------------------------------------------------------

class TestEmbedTuner:
    def test_loss_decreases_on_fixed_batch(self):
        torch.manual_seed(0)
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, lr=3e-3, device="cpu")
        input_ids = torch.randint(1, 256, (4, 6))

        losses = [tuner.train_step(input_ids) for _ in range(20)]

        assert losses[-1] < losses[0]

    def test_training_improves_batch_retrieval_accuracy(self):
        """After training on a fixed batch, each row's same-passage view
        must rank above every other row's view in cosine similarity — the
        exact signal downstream retrieval needs.
        """
        torch.manual_seed(0)
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, lr=3e-3, device="cpu")
        input_ids = torch.randint(1, 256, (6, 6))

        for _ in range(80):
            tuner.train_step(input_ids)

        model.train()
        with torch.no_grad():
            emb_a = F.normalize(model._embed_pooled(input_ids), dim=-1)
            emb_b = F.normalize(model._embed_pooled(input_ids), dim=-1)
            sim = emb_a @ emb_b.T

        assert (sim.argmax(dim=1) == torch.arange(6)).all()

    def test_train_step_rejects_batch_of_one(self):
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        with pytest.raises(ValueError, match="batch_size"):
            tuner.train_step(torch.randint(1, 256, (1, 6)))

    def test_train_step_returns_python_float(self):
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        loss = tuner.train_step(torch.randint(1, 256, (3, 6)))
        assert isinstance(loss, float)
