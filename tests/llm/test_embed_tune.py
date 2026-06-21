"""Unit tests for self-supervised contrastive embedding training.

Coverage:
    contrastive_loss   — perfect alignment collapses to ~0, uniform
                          similarity gives log(batch) (no signal), gradients
                          flow.
    PassageDataset      — BOS prefix, truncation to max_seq_len.
    collate_passages    — right-padding, attention mask correctness.
    EmbedTuner          — loss decreases on a fixed batch, batch retrieval
                           accuracy converges, batch_size=1 is rejected,
                           train() cycles a short loader, LoRA-only training
                           leaves base weights untouched.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.lora import LoRALinear
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import BOS_ID, PAD_ID
from grimoire_ai.llm.training.embed_tune import (
    EmbedTuner,
    PassageDataset,
    collate_passages,
    contrastive_loss,
)


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

    def test_train_cycles_short_loader_to_reach_total_steps(self):
        """train() must keep going past one pass over a short loader."""
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        # 2 batches per epoch, but ask for more steps than that.
        batches = [torch.randint(1, 256, (3, 6)) for _ in range(2)]
        loader = [(b, None) for b in batches]

        log_calls = []
        tuner.train(loader, total_steps=7, log_every=7, on_log=lambda s, l: log_calls.append(s))

        assert log_calls == [7]

    def test_train_runs_exact_step_count(self):
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        loader = [(torch.randint(1, 256, (3, 6)), None)]

        steps_logged = []
        tuner.train(loader, total_steps=10, log_every=2, on_log=lambda s, l: steps_logged.append(s))

        assert steps_logged == [2, 4, 6, 8, 10]


# ---------------------------------------------------------------------------
# PassageDataset / collate_passages
# ---------------------------------------------------------------------------

class TestPassageDataset:
    def _tokenizer(self) -> BytePairEncoder:
        enc = BytePairEncoder()
        enc.train(["hello world, the grappled creature has speed zero. " * 20], vocab_size=280)
        return enc

    def test_every_sequence_starts_with_bos(self):
        tok = self._tokenizer()
        ds = PassageDataset(["hello world", "the grappled creature"], tok, max_seq_len=16)
        for i in range(len(ds)):
            assert ds[i][0].item() == BOS_ID

    def test_truncates_to_max_seq_len(self):
        tok = self._tokenizer()
        long_text = "hello world, the grappled creature has speed zero. " * 10
        ds = PassageDataset([long_text], tok, max_seq_len=8)
        assert ds[0].shape[0] == 8

    def test_length_matches_input(self):
        tok = self._tokenizer()
        passages = ["hello world", "the grappled creature", "speed zero"]
        ds = PassageDataset(passages, tok, max_seq_len=16)
        assert len(ds) == len(passages)


class TestCollatePassages:
    def test_pads_to_longest_in_batch(self):
        batch = [torch.tensor([1, 2, 3]), torch.tensor([1, 2])]
        input_ids, attention_mask = collate_passages(batch)
        assert input_ids.shape == (2, 3)
        assert attention_mask.shape == (2, 3)

    def test_padding_uses_pad_id_and_zero_mask(self):
        batch = [torch.tensor([1, 2, 3]), torch.tensor([1, 2])]
        input_ids, attention_mask = collate_passages(batch)
        assert input_ids[1, 2].item() == PAD_ID
        assert attention_mask[1, 2].item() == 0
        assert torch.equal(attention_mask[0], torch.ones(3, dtype=torch.long))

    def test_real_tokens_preserved_in_order(self):
        batch = [torch.tensor([5, 6, 7]), torch.tensor([9, 10])]
        input_ids, _ = collate_passages(batch)
        assert torch.equal(input_ids[0], torch.tensor([5, 6, 7]))
        assert torch.equal(input_ids[1, :2], torch.tensor([9, 10]))

    def test_loader_round_trip_yields_batchable_tensors(self):
        tok = BytePairEncoder()
        tok.train(["hello world, the grappled creature has speed zero. " * 20], vocab_size=280)
        ds = PassageDataset(["hello world", "the grappled creature", "speed zero"], tok, max_seq_len=16)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_passages, drop_last=True)
        input_ids, attention_mask = next(iter(loader))
        assert input_ids.shape == attention_mask.shape
        assert input_ids.shape[0] == 2


# ---------------------------------------------------------------------------
# LoRA-only training
# ---------------------------------------------------------------------------

class TestEmbedTunerWithLora:
    def test_base_weights_unchanged_after_lora_training(self):
        """add_lora_adapters() before constructing EmbedTuner must leave
        every non-adapter weight bit-identical after training.
        """
        torch.manual_seed(0)
        model = GrimoireTransformer(_small_config())
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])

        base_weights_before = {
            f"{name}.base_weight": mod.base_weight.clone()
            for name, mod in model.named_modules()
            if isinstance(mod, LoRALinear)
        }
        non_lora_before = {
            name: p.clone()
            for name, p in model.named_parameters()
            if "lora_A" not in name and "lora_B" not in name
        }

        tuner = EmbedTuner(model, lr=3e-3, device="cpu")
        input_ids = torch.randint(1, 256, (4, 6))
        for _ in range(10):
            tuner.train_step(input_ids)

        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                assert torch.equal(mod.base_weight, base_weights_before[f"{name}.base_weight"])
        for name, p in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                assert torch.equal(p, non_lora_before[name]), f"{name} changed but is not a LoRA param"

    def test_lora_params_change_after_training(self):
        torch.manual_seed(0)
        model = GrimoireTransformer(_small_config())
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        lora_b_before = [p.clone() for n, p in model.named_parameters() if "lora_B" in n]

        tuner = EmbedTuner(model, lr=3e-3, device="cpu")
        input_ids = torch.randint(1, 256, (4, 6))
        for _ in range(10):
            tuner.train_step(input_ids)

        lora_b_after = [p for n, p in model.named_parameters() if "lora_B" in n]
        assert any(
            not torch.equal(before, after)
            for before, after in zip(lora_b_before, lora_b_after)
        )

    def test_lora_only_training_still_converges(self):
        """The contrastive signal must still reach the adapter: batch
        retrieval accuracy should improve even though the trunk is frozen.
        """
        torch.manual_seed(0)
        model = GrimoireTransformer(_small_config())
        model.add_lora_adapters(rank=4, alpha=8.0, targets=["q_proj", "v_proj"])
        tuner = EmbedTuner(model, lr=1e-2, device="cpu")
        input_ids = torch.randint(1, 256, (6, 6))

        for _ in range(200):
            tuner.train_step(input_ids)

        model.train()
        with torch.no_grad():
            emb_a = F.normalize(model._embed_pooled(input_ids), dim=-1)
            emb_b = F.normalize(model._embed_pooled(input_ids), dim=-1)
            sim = emb_a @ emb_b.T

        assert (sim.argmax(dim=1) == torch.arange(6)).all()
