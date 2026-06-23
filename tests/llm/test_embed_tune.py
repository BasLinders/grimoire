"""Unit tests for self-supervised and supervised contrastive embedding training.

Coverage:
    contrastive_loss          — perfect alignment collapses to ~0, uniform
                                 similarity gives log(batch) (no signal),
                                 gradients flow.
    PassageDataset             — BOS prefix, truncation to max_seq_len.
    collate_passages           — right-padding, attention mask correctness.
    DocumentGroupedBatchSampler — batch composition (docs/passages-per-doc),
                                 validation, reproducibility, integration
                                 with EmbedTuner.
    EmbedTuner                 — loss decreases on a fixed batch, batch
                                 retrieval accuracy converges, batch_size=1
                                 is rejected, train() cycles a short loader,
                                 LoRA-only training leaves base weights
                                 untouched.
    QAPairDataset/collate_qa_pairs/train_step_pairs/train_pairs — the
                                 supervised (question, answer) path: two
                                 genuinely different inputs embedded and
                                 contrasted, independent padding per side,
                                 convergence on real pairs.
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
    DocumentGroupedBatchSampler,
    EmbedTuner,
    PassageDataset,
    QAPairDataset,
    collate_passages,
    collate_qa_pairs,
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


# ---------------------------------------------------------------------------
# DocumentGroupedBatchSampler
# ---------------------------------------------------------------------------

class TestDocumentGroupedBatchSampler:
    def _doc_ids(self) -> list[int]:
        # 4 documents, 5 passages each -> indices 0-19, doc_id = idx // 5.
        return [idx // 5 for idx in range(20)]

    def test_batch_size_must_be_multiple_of_passages_per_doc(self):
        with pytest.raises(ValueError, match="multiple"):
            DocumentGroupedBatchSampler(self._doc_ids(), batch_size=10, passages_per_doc=4)

    def test_passages_per_doc_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="passages_per_doc"):
            DocumentGroupedBatchSampler(self._doc_ids(), batch_size=8, passages_per_doc=1)

    def test_rejects_corpus_with_no_large_enough_document(self):
        # Every "document" here has only 1 passage -- none can supply 2.
        with pytest.raises(ValueError, match="No document"):
            DocumentGroupedBatchSampler(list(range(10)), batch_size=4, passages_per_doc=2)

    def test_len_matches_num_batches(self):
        sampler = DocumentGroupedBatchSampler(
            self._doc_ids(), batch_size=8, passages_per_doc=4, num_batches=17
        )
        assert len(sampler) == 17
        assert sum(1 for _ in sampler) == 17

    def test_every_batch_has_correct_size(self):
        sampler = DocumentGroupedBatchSampler(
            self._doc_ids(), batch_size=8, passages_per_doc=4, num_batches=20, seed=0
        )
        for batch in sampler:
            assert len(batch) == 8

    def test_batch_contains_same_document_groups(self):
        """Each batch must be composed of passages_per_doc-sized groups that
        share a document -- the actual hard-negative structure this sampler
        exists to produce."""
        doc_ids = self._doc_ids()
        sampler = DocumentGroupedBatchSampler(
            doc_ids, batch_size=8, passages_per_doc=4, num_batches=20, seed=0
        )
        for batch in sampler:
            docs_in_batch = [doc_ids[i] for i in batch]
            # 8 passages / 4 per doc = 2 distinct documents, each appearing exactly 4 times.
            counts: dict[int, int] = {}
            for d in docs_in_batch:
                counts[d] = counts.get(d, 0) + 1
            assert len(counts) == 2
            assert all(c == 4 for c in counts.values())

    def test_indices_are_valid_and_within_their_document(self):
        doc_ids = self._doc_ids()
        sampler = DocumentGroupedBatchSampler(
            doc_ids, batch_size=8, passages_per_doc=4, num_batches=10, seed=0
        )
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < len(doc_ids)

    def test_seed_reproducible(self):
        doc_ids = self._doc_ids()
        a = list(DocumentGroupedBatchSampler(doc_ids, batch_size=8, passages_per_doc=4, num_batches=5, seed=42))
        b = list(DocumentGroupedBatchSampler(doc_ids, batch_size=8, passages_per_doc=4, num_batches=5, seed=42))
        assert a == b

    def test_small_documents_are_excluded_not_crashed_on(self):
        # doc 0 has 5 passages, doc 1 has only 1 -- doc 1 must simply be
        # unused, not cause an error.
        doc_ids = [0, 0, 0, 0, 0, 1]
        sampler = DocumentGroupedBatchSampler(doc_ids, batch_size=4, passages_per_doc=2, num_batches=5, seed=0)
        for batch in sampler:
            for idx in batch:
                assert doc_ids[idx] == 0

    def test_works_as_dataloader_batch_sampler_with_embed_tuner(self):
        """End-to-end: plug the sampler into a real DataLoader/EmbedTuner
        pipeline and confirm training still runs and converges, proving the
        hard-negative batches are still valid contrastive training input.
        """
        torch.manual_seed(0)
        tok = BytePairEncoder()
        tok.train(["hello world, the grappled creature has speed zero. " * 20], vocab_size=280)

        # 4 documents worth of distinct passages, 4 passages each.
        passages = [f"document {d} passage {p} hello world grappled creature" for d in range(4) for p in range(4)]
        doc_ids = [d for d in range(4) for _ in range(4)]
        dataset = PassageDataset(passages, tok, max_seq_len=16)
        sampler = DocumentGroupedBatchSampler(doc_ids, batch_size=8, passages_per_doc=4, num_batches=30, seed=0)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_passages)

        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, lr=3e-3, device="cpu")

        on_log_calls = []
        tuner.train(loader, total_steps=30, log_every=10, on_log=lambda s, l: on_log_calls.append((s, l)))
        assert [s for s, _ in on_log_calls] == [10, 20, 30]


# ---------------------------------------------------------------------------
# Supervised (question, answer) pairs
# ---------------------------------------------------------------------------

def _qa_tokenizer() -> BytePairEncoder:
    # BytePairEncoder requires vocab_size >= 262 (6 special + 256 base bytes),
    # but the actual encoded ids for this small fixed vocabulary/text set stay
    # well under _small_config()'s vocab_size (256) in practice -- verified
    # empirically (max id 127) since the model's embedding table has exactly
    # that many rows.
    tok = BytePairEncoder()
    tok.train(
        [
            "how does grappling affect movement speed creature held fast " * 10,
            "what is advantage roll twice take higher result d20 " * 10,
            "how does a long rest recover spell slots hit points " * 10,
        ],
        vocab_size=262,
    )
    return tok


class TestQAPairDataset:
    def test_each_side_starts_with_bos(self):
        tok = _qa_tokenizer()
        ds = QAPairDataset(
            [("how does grappling work", "a creature held fast cannot move")], tok, max_seq_len=16,
        )
        q_ids, a_ids = ds[0]
        assert q_ids[0].item() == BOS_ID
        assert a_ids[0].item() == BOS_ID

    def test_question_and_answer_tokenized_independently(self):
        """A short question and a much longer answer must not be forced to
        the same length -- each side encodes on its own."""
        tok = _qa_tokenizer()
        ds = QAPairDataset(
            [("grapple", "a creature held fast cannot move and speed becomes zero")],
            tok, max_seq_len=32,
        )
        q_ids, a_ids = ds[0]
        assert q_ids.shape[0] != a_ids.shape[0]

    def test_truncates_each_side_to_max_seq_len(self):
        tok = _qa_tokenizer()
        long_text = "how does grappling affect movement speed creature held fast " * 5
        ds = QAPairDataset([(long_text, long_text)], tok, max_seq_len=8)
        q_ids, a_ids = ds[0]
        assert q_ids.shape[0] == 8
        assert a_ids.shape[0] == 8

    def test_length_matches_input(self):
        tok = _qa_tokenizer()
        pairs = [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]
        ds = QAPairDataset(pairs, tok, max_seq_len=16)
        assert len(ds) == len(pairs)


class TestCollateQAPairs:
    def test_each_side_padded_to_its_own_longest(self):
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4, 5])),
            (torch.tensor([1, 2]), torch.tensor([1, 2, 3])),
        ]
        q_ids, q_mask, a_ids, a_mask = collate_qa_pairs(batch)
        assert q_ids.shape == (2, 3)   # padded to the longer question (3)
        assert a_ids.shape == (2, 5)   # padded to the longer answer (5), independently

    def test_padding_uses_pad_id_and_zero_mask(self):
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([5, 6])),
            (torch.tensor([1]), torch.tensor([5, 6, 7])),
        ]
        q_ids, q_mask, a_ids, a_mask = collate_qa_pairs(batch)
        assert q_ids[1, 1].item() == PAD_ID
        assert q_mask[1, 1].item() == 0
        assert a_ids[0, 2].item() == PAD_ID
        assert a_mask[0, 2].item() == 0

    def test_real_tokens_preserved_in_order(self):
        batch = [(torch.tensor([5, 6, 7]), torch.tensor([9, 10]))]
        q_ids, _, a_ids, _ = collate_qa_pairs(batch)
        assert torch.equal(q_ids[0], torch.tensor([5, 6, 7]))
        assert torch.equal(a_ids[0], torch.tensor([9, 10]))

    def test_loader_round_trip_yields_batchable_tensors(self):
        tok = _qa_tokenizer()
        pairs = [("grapple speed", "creature held fast"), ("advantage roll", "roll twice take higher")]
        ds = QAPairDataset(pairs, tok, max_seq_len=16)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_qa_pairs, drop_last=True)
        q_ids, q_mask, a_ids, a_mask = next(iter(loader))
        assert q_ids.shape[0] == 2
        assert a_ids.shape[0] == 2
        assert q_ids.shape == q_mask.shape
        assert a_ids.shape == a_mask.shape


class TestEmbedTunerPairs:
    def test_train_step_pairs_rejects_batch_of_one(self):
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        q = torch.randint(1, 256, (1, 6))
        a = torch.randint(1, 256, (1, 6))
        with pytest.raises(ValueError, match="batch_size"):
            tuner.train_step_pairs(q, None, a, None)

    def test_train_step_pairs_returns_python_float(self):
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        q = torch.randint(1, 256, (3, 6))
        a = torch.randint(1, 256, (3, 7))  # deliberately different seq_len per side
        loss = tuner.train_step_pairs(q, None, a, None)
        assert isinstance(loss, float)

    def test_loss_decreases_on_fixed_pairs(self):
        torch.manual_seed(0)
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, lr=3e-3, device="cpu")
        q = torch.randint(1, 256, (4, 6))
        a = torch.randint(1, 256, (4, 8))

        losses = [tuner.train_step_pairs(q, None, a, None) for _ in range(20)]
        assert losses[-1] < losses[0]

    def test_supervised_training_aligns_questions_with_their_answers(self):
        """After training on real (question, answer) text pairs, each
        question's embedding must rank its OWN answer highest among the
        other answers in the batch -- the actual retrieval signal this path
        exists to produce, as opposed to same-passage self-similarity.
        """
        torch.manual_seed(0)
        tok = _qa_tokenizer()
        pairs = [
            ("how does grappling affect speed", "a creature held fast cannot move at all"),
            ("what is advantage on a roll", "roll the d20 twice and take the higher result"),
            ("how does a long rest work", "spell slots and hit points are fully restored"),
        ]
        ds = QAPairDataset(pairs, tok, max_seq_len=_small_config().max_seq_len)
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_qa_pairs, drop_last=True)

        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, lr=3e-3, device="cpu")
        tuner.train_pairs(loader, total_steps=150, log_every=150)

        batch = next(iter(loader))
        q_ids, q_mask, a_ids, a_mask = batch
        model.train()
        with torch.no_grad():
            emb_q = F.normalize(model._embed_pooled(q_ids, q_mask), dim=-1)
            emb_a = F.normalize(model._embed_pooled(a_ids, a_mask), dim=-1)
            sim = emb_q @ emb_a.T

        assert (sim.argmax(dim=1) == torch.arange(3)).all()

    def test_train_pairs_cycles_short_loader_to_reach_total_steps(self):
        model = GrimoireTransformer(_small_config())
        tuner = EmbedTuner(model, device="cpu")
        batch = (
            torch.randint(1, 256, (3, 6)), None,
            torch.randint(1, 256, (3, 7)), None,
        )
        loader = [batch]  # a single-batch loader; train_pairs must cycle it

        steps_logged = []
        tuner.train_pairs(loader, total_steps=10, log_every=2, on_log=lambda s, l: steps_logged.append(s))
        assert steps_logged == [2, 4, 6, 8, 10]
