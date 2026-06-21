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

Scope
-----
This is the minimal loop needed to validate the loss against a real model:
no checkpointing, no LoRA, no corpus loading. Those land in later phases —
this module trains all of ``model.parameters()`` directly, which the LoRA
phase will narrow to adapter-only parameters.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from grimoire_ai.llm.model.transformer import GrimoireTransformer


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


class EmbedTuner:
    """Runs contrastive training steps against a ``GrimoireTransformer``.

    Attributes:
        model: The model being tuned, moved to ``device``.
        device: ``"cuda"`` or ``"cpu"``.
        temperature: Forwarded to ``contrastive_loss`` on every step.
        optimizer: ``AdamW`` over ``model.parameters()``.
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
