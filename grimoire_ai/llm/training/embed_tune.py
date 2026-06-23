"""Self-supervised contrastive training for sentence embeddings.

The model is only ever trained with next-token-prediction loss, which gives
no pressure to place semantically similar passages near each other in the
pooled-embedding space ``GrimoireTransformer._embed_pooled`` produces. This
module adds that missing signal via in-batch contrastive learning, using the
SimCSE trick: running the *same* passage through the model twice in
``train()`` mode gives two distinct vectors purely from dropout noise — that
pair is the positive; every other passage in the batch is a free negative.
This requires no labels, no pair-mining, and no domain knowledge, so the
same recipe applies to any corpus.

Loss
----
``contrastive_loss`` is in-batch InfoNCE (a.k.a. multiple-negatives-ranking
loss): cosine-similarity logits between the two views, cross-entropy against
the identity permutation (row ``i`` should match column ``i``).

LoRA
----
``EmbedTuner`` always optimises ``model.parameters()`` filtered to
``requires_grad=True``. Calling ``model.add_lora_adapters(...)`` before
constructing the tuner freezes the base weights and leaves only the
adapter trainable — no other change needed here. The resulting ``.lora``
file is meant to be loaded into a *separate* model instance used only for
embedding (e.g. as ``SemanticRetriever``'s ``embed_fn``), not into the
instance used for chat generation: the adapter reroutes the same
``q_proj``/``v_proj`` weights used by ``forward()``, so applying it to the
generation model would change generation output too, defeating the point
of using LoRA here.

Hard negatives
---------------
Plain random batching makes every negative "easy": two passages picked from
anywhere across a large, topically diverse corpus are almost always about
completely different things, so the loss saturates by learning only coarse
topic separation and never has to discriminate between passages that are
genuinely close (e.g. "advantage" vs "disadvantage", "long rest" vs "short
rest") — confirmed empirically on the real Saga corpus: the trained adapter
beat the untrained baseline's retrieval hit-rate but produced *worse* quiz
answers, because its retrieved context was frequently a confusable
near-miss rather than actually wrong. ``DocumentGroupedBatchSampler`` fixes
this by building every batch from a handful of documents with several
passages each, so in-batch negatives include same-document near-misses
(hard) alongside cross-document negatives (easy) — still fully
self-supervised, still no labels or domain knowledge, just a different way
of grouping the same passages into batches.

Scope
-----
``PassageDataset``/``collate_passages`` tokenize corpus passages the same
way ``InferenceEngine.embed`` does (``BOS`` prefix, truncate, pad with
``PAD_ID``) so a checkpoint trained here behaves identically when consumed
through the normal inference path. Corpus *loading* (reading files, calling
``chunk_text``) stays in the CLI script — this module only tokenizes
whatever passage strings it's given, so it has no opinion on where they
came from.
"""

import random
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import BOS_ID, PAD_ID


def contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """In-batch InfoNCE loss between two views of the same batch of passages.

    Args:
        emb_a: First view, shape ``(batch, d_model)``. Need not be
            pre-normalised.
        emb_b: Second view, shape ``(batch, d_model)``, aligned row-for-row
            with ``emb_a`` (row ``i`` in both is the same passage).
        temperature: Softmax temperature applied to the cosine-similarity
            logits. Lower values sharpen the distribution, making the loss
            penalise near-miss negatives more harshly.

    Returns:
        Scalar cross-entropy loss. Minimised when each row's similarity to
        its matching row in the other view is the highest in that row.
    """
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    sim = emb_a @ emb_b.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


class PassageDataset(Dataset):
    """Corpus passages, tokenized on access for contrastive training batches.

    Tokenization mirrors ``InferenceEngine.embed`` exactly (``BOS`` prefix,
    truncate to ``max_seq_len``) so a LoRA adapter trained here sees the same
    token layout it will see at inference time through that method.

    Tokenization happens lazily in ``__getitem__``, not up front: a
    real-sized corpus chunks into millions of passages (e.g. ~2M for the
    455 MB Saga corpus), and a fixed number of training steps only ever
    touches a small, randomly-sampled fraction of them. Eagerly tokenizing
    every passage before training starts would mean paying that cost for
    passages the run never uses.
    """

    def __init__(
        self,
        passages: list[str],
        tokenizer: BytePairEncoder,
        max_seq_len: int,
    ) -> None:
        """Store the passages and tokenizer for on-access encoding.

        Args:
            passages: Passage strings (e.g. from ``chunk_text``). Order is
                preserved; no domain knowledge about their content is used.
            tokenizer: A loaded ``BytePairEncoder``.
            max_seq_len: Sequences longer than this (including the ``BOS``
                token) are truncated.
        """
        self._passages = passages
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._passages)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = [BOS_ID] + self._tokenizer.encode(self._passages[idx])[: self._max_seq_len - 1]
        return torch.tensor(ids, dtype=torch.long)


def collate_passages(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch of token-id sequences with ``PAD_ID``.

    Right-padding (not the left-padding ``PaddingCollator`` uses for causal
    LM training) is fine here: ``_embed_pooled`` masks padded positions out
    of the mean pool rather than feeding them through autoregressive
    attention, so there is no "padding mid-context" hazard to avoid.

    Args:
        batch: A list of variable-length 1-D token-id tensors, as returned
            by ``PassageDataset.__getitem__``.

    Returns:
        ``(input_ids, attention_mask)``, both of shape
        ``(batch_size, max_len)``, where ``max_len`` is the longest sequence
        in this batch.
    """
    width = max(seq.size(0) for seq in batch)
    input_ids = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, seq in enumerate(batch):
        input_ids[row, : seq.size(0)] = seq
        attention_mask[row, : seq.size(0)] = 1
    return input_ids, attention_mask


class DocumentGroupedBatchSampler:
    """Yields batches with several passages per source document.

    Each batch is built from ``docs_per_batch = batch_size // passages_per_doc``
    documents, sampling ``passages_per_doc`` passages from each (with
    replacement, since a fixed number of training steps only ever touches a
    small fraction of a real corpus anyway — see ``PassageDataset``). The
    result: every batch's negatives are a mix of same-document near-misses
    (hard — close in topic and vocabulary, but a different fact) and
    cross-document negatives (easy — unrelated content), rather than being
    uniformly easy the way pure random batching is.

    Meant to be passed as a ``DataLoader``'s ``batch_sampler``, e.g.
    ``DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_passages)``.
    """

    def __init__(
        self,
        doc_ids: list[int],
        batch_size: int,
        passages_per_doc: int = 4,
        num_batches: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """Group passage indices by document and set up batch sampling.

        Args:
            doc_ids: One document id per passage, aligned with the dataset
                this sampler will be used against (e.g. the index of the
                source file each passage was chunked from).
            batch_size: Passages per batch. Must be a multiple of
                ``passages_per_doc``.
            passages_per_doc: Passages sampled from each chosen document per
                batch. Must be at least 2 -- with 1, a document contributes
                no same-document negative at all.
            num_batches: Length of one pass over this sampler (it samples
                with replacement, so this is just where ``EmbedTuner.train``'s
                cycling logic restarts iteration, not a hard corpus limit).
            seed: Optional seed for reproducible batch composition.

        Raises:
            ValueError: If ``batch_size`` isn't a multiple of
                ``passages_per_doc``, if ``passages_per_doc < 2``, or if no
                document has at least ``passages_per_doc`` passages.
        """
        if passages_per_doc < 2:
            raise ValueError(
                f"passages_per_doc must be >= 2 for same-document negatives "
                f"to exist, got {passages_per_doc}."
            )
        if batch_size % passages_per_doc != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of "
                f"passages_per_doc ({passages_per_doc})."
            )

        by_doc: dict[int, list[int]] = {}
        for idx, doc_id in enumerate(doc_ids):
            by_doc.setdefault(doc_id, []).append(idx)
        self._docs = [idxs for idxs in by_doc.values() if len(idxs) >= passages_per_doc]
        if not self._docs:
            raise ValueError(
                f"No document has >= passages_per_doc ({passages_per_doc}) passages; "
                f"cannot form same-document batches."
            )

        self._passages_per_doc = passages_per_doc
        self._docs_per_batch = batch_size // passages_per_doc
        self._num_batches = num_batches
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterable[list[int]]:
        for _ in range(self._num_batches):
            chosen_docs = self._rng.sample(
                self._docs, k=min(self._docs_per_batch, len(self._docs))
            )
            batch: list[int] = []
            for doc_idxs in chosen_docs:
                batch.extend(self._rng.sample(doc_idxs, k=self._passages_per_doc))
            yield batch

    def __len__(self) -> int:
        return self._num_batches


class EmbedTuner:
    """Runs contrastive training steps against a ``GrimoireTransformer``.

    Attributes:
        model: The model being tuned, moved to ``device``.
        device: ``"cuda"`` or ``"cpu"``.
        temperature: Forwarded to ``contrastive_loss`` on every step.
        optimizer: ``AdamW`` over the model's trainable parameters — every
            parameter with ``requires_grad=True`` at construction time. Call
            ``model.add_lora_adapters(...)`` before constructing this tuner
            to train only a LoRA adapter instead of the full model; nothing
            else needs to change.
    """

    def __init__(
        self,
        model: GrimoireTransformer,
        lr: float = 1e-4,
        temperature: float = 0.05,
        device: Optional[str] = None,
    ) -> None:
        """Set up the tuner and its optimizer.

        Args:
            model: A ``GrimoireTransformer`` to train in place.
            lr: AdamW learning rate.
            temperature: Forwarded to ``contrastive_loss``.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=lr
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Run one contrastive training step on a batch of passages.

        Embeds ``input_ids`` twice (two independent dropout masks, since the
        model is put into ``train()`` mode), computes the in-batch InfoNCE
        loss between the two views, and takes one AdamW step.

        Args:
            input_ids: Token ids, shape ``(batch, seq_len)``. ``batch`` must
                be at least 2 — with a single passage there are no negatives
                and the loss is vacuously zero, silently wasting compute.
            attention_mask: Optional padding mask, shape ``(batch, seq_len)``.

        Returns:
            The scalar loss value for this step (Python float).

        Raises:
            ValueError: If ``input_ids`` has fewer than 2 rows.
        """
        if input_ids.size(0) < 2:
            raise ValueError(
                f"train_step requires batch_size >= 2 for in-batch negatives "
                f"to exist, got {input_ids.size(0)}."
            )

        self.model.train()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        emb_a = self.model._embed_pooled(input_ids, attention_mask)
        emb_b = self.model._embed_pooled(input_ids, attention_mask)
        loss = contrastive_loss(emb_a, emb_b, temperature=self.temperature)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def train(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        total_steps: int,
        log_every: int = 50,
        on_log: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """Run ``train_step`` repeatedly until ``total_steps`` is reached.

        Cycles ``loader`` indefinitely so training can run for more steps
        than there are batches in the dataset, mirroring ``Trainer.train()``.

        Args:
            loader: Iterable of ``(input_ids, attention_mask)`` batches, e.g.
                a ``DataLoader`` over ``PassageDataset`` with
                ``collate_fn=collate_passages``. Every batch must have at
                least 2 rows (see ``train_step``).
            total_steps: Number of ``train_step`` calls to run.
            log_every: Print (and invoke ``on_log`` with) the mean loss over
                the last ``log_every`` steps, every ``log_every`` steps.
            on_log: Optional callback invoked as ``(step, avg_loss)`` at each
                ``log_every`` boundary, in addition to the printed line.
        """
        step = 0
        running_loss = 0.0
        data_iter = iter(loader)
        while step < total_steps:
            try:
                input_ids, attention_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, attention_mask = next(data_iter)

            running_loss += self.train_step(input_ids, attention_mask)
            step += 1

            if step % log_every == 0:
                avg_loss = running_loss / log_every
                print(f"  embed-tune step {step:>6} / {total_steps} | loss {avg_loss:.4f}")
                if on_log is not None:
                    on_log(step, avg_loss)
                running_loss = 0.0
