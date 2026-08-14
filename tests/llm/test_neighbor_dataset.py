"""Tests for the RETRO training-data pipeline: NeighborAugmentedDataset,
collate_with_neighbors, and Trainer's neighbor_ids wiring.

Completes item #3 from docs/architecture_optimization.md (see
tests/llm/test_chunked_cross_attention.py for the model/attention side):
this is the "feed real retrieved neighbors into a real training step" half.
scripts/build_retrieval_neighbors.py (the offline precomputation script) is
not covered here — it depends on a live SemanticRetriever/InferenceEngine
and is exercised manually rather than unit tested, consistent with how
other scripts in this repo that need a real checkpoint are handled.

The bottom section covers RETRO combined with Multi-Token Prediction
(item #2, PR #180) in the same Trainer.train() run — the one place their
independently-developed code paths actually intersect (Trainer's single
self._forward_model(...) call, which must pass neighbor_ids
unconditionally while only conditionally requesting return_mtp_logits).
Neither feature's own test suite exercised this combination, since each
was developed and tested against main before the other existed.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.data.neighbor_dataset import NeighborAugmentedDataset, collate_with_neighbors
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID
from grimoire_ai.llm.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(retro_layers=None, n_predict=0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        d_ff=64, max_seq_len=16, dropout=0.0, retro_layers=retro_layers,
        n_predict=n_predict,
    )


def _tiny_bin_dataset(tmp_dir: str, cfg: TransformerConfig, stride: int = None) -> TokenizedDataset:
    corpus = np.arange(cfg.vocab_size * 4, dtype=np.int32) % cfg.vocab_size
    path = str(Path(tmp_dir) / "corpus.bin")
    corpus.astype(np.int32).tofile(path)
    return TokenizedDataset(corpus_path=path, seq_len=cfg.max_seq_len, stride=stride)


# ---------------------------------------------------------------------------
# NeighborAugmentedDataset
# ---------------------------------------------------------------------------

def test_neighbor_dataset_len_matches_base(tmp_path) -> None:
    cfg = _tiny_config()
    base = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.zeros((len(base), 2, 8), dtype=np.int32)
    wrapped = NeighborAugmentedDataset(base, neighbor_ids)
    assert len(wrapped) == len(base)


def test_neighbor_dataset_getitem_shapes(tmp_path) -> None:
    cfg = _tiny_config()
    base = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.random.randint(0, cfg.vocab_size, size=(len(base), 2, 8)).astype(np.int32)
    wrapped = NeighborAugmentedDataset(base, neighbor_ids)

    input_ids, target_ids, item_neighbors = wrapped[0]
    assert input_ids.shape == (cfg.max_seq_len,)
    assert target_ids.shape == (cfg.max_seq_len,)
    assert item_neighbors.shape == (2, 8)
    assert item_neighbors.dtype == torch.int64
    assert torch.equal(item_neighbors, torch.from_numpy(neighbor_ids[0]).long())


def test_neighbor_dataset_rejects_length_mismatch(tmp_path) -> None:
    cfg = _tiny_config()
    base = _tiny_bin_dataset(str(tmp_path), cfg)
    wrong_length_neighbors = np.zeros((len(base) + 1, 2, 8), dtype=np.int32)
    with pytest.raises(ValueError, match="neighbor_ids"):
        NeighborAugmentedDataset(base, wrong_length_neighbors)


# ---------------------------------------------------------------------------
# collate_with_neighbors
# ---------------------------------------------------------------------------

def test_collate_with_neighbors_shapes(tmp_path) -> None:
    cfg = _tiny_config()
    base = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.random.randint(0, cfg.vocab_size, size=(len(base), 2, 8)).astype(np.int32)
    wrapped = NeighborAugmentedDataset(base, neighbor_ids)

    batch = [wrapped[0], wrapped[1], wrapped[2]]
    input_ids, target_ids, attention_mask, batch_neighbors = collate_with_neighbors(batch)

    assert input_ids.shape == (3, cfg.max_seq_len)
    assert target_ids.shape == (3, cfg.max_seq_len)
    assert attention_mask.shape == (3, cfg.max_seq_len)
    assert batch_neighbors.shape == (3, 2, 8)


def test_collate_with_neighbors_matches_plain_collator_on_ids(tmp_path) -> None:
    """input_ids/target_ids/attention_mask must be identical to what
    PaddingCollator alone would produce -- collate_with_neighbors reuses it
    internally rather than reimplementing padding."""
    from grimoire_ai.llm.data.collator import PaddingCollator

    cfg = _tiny_config()
    base = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.zeros((len(base), 2, 8), dtype=np.int32)
    wrapped = NeighborAugmentedDataset(base, neighbor_ids)

    batch = [wrapped[0], wrapped[1]]
    plain_pairs = [(inp, tgt) for inp, tgt, _ in batch]

    input_ids, target_ids, attention_mask, _ = collate_with_neighbors(batch)
    ref_input_ids, ref_target_ids, ref_mask = PaddingCollator(pad_id=PAD_ID)(plain_pairs)

    assert torch.equal(input_ids, ref_input_ids)
    assert torch.equal(target_ids, ref_target_ids)
    assert torch.equal(attention_mask, ref_mask)


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------

def test_trainer_rejects_neighbor_ids_without_retro_layers(tmp_path) -> None:
    cfg = _tiny_config(retro_layers=None)
    model = GrimoireTransformer(cfg)
    dataset = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.zeros((len(dataset), 2, 8), dtype=np.int32)

    with pytest.raises(ValueError, match="retro_layers"):
        Trainer(
            model=model,
            train_dataset=dataset,
            neighbor_ids=neighbor_ids,
            device="cpu",
        )


def test_training_run_with_retro_neighbors_enabled(tmp_path) -> None:
    """A short training run with retro_layers + neighbor_ids must complete
    without error, produce a finite loss, and persist the CCA sublayer
    weights in the checkpoint (proving they were actually part of the
    trained model, not silently dropped)."""
    cfg = _tiny_config(retro_layers=[0])
    model = GrimoireTransformer(cfg)
    dataset = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.random.randint(
        1, cfg.vocab_size, size=(len(dataset), 2, 8)
    ).astype(np.int32)
    logs: list[float] = []

    Trainer(
        model=model,
        train_dataset=dataset,
        neighbor_ids=neighbor_ids,
        total_steps=3,
        batch_size=2,
        accumulate_steps=1,
        log_every=1,
        save_every=3,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        on_log=lambda step, loss, lr, elapsed: logs.append(loss),
    ).train()

    assert len(logs) == 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in logs)

    ckpt = torch.load(str(tmp_path / "step_0000003.pt"), map_location="cpu", weights_only=True)
    state_dict = ckpt["model"]
    assert any(key.startswith("blocks.0.cca.") for key in state_dict)


def test_training_run_without_neighbor_ids_unaffected(tmp_path) -> None:
    """Omitting neighbor_ids (the default) must train exactly as before --
    even on a retro_layers-configured model, since CCA sublayers no-op
    without neighbor_emb (see test_chunked_cross_attention.py)."""
    cfg = _tiny_config(retro_layers=[0])
    model = GrimoireTransformer(cfg)
    dataset = _tiny_bin_dataset(str(tmp_path), cfg)
    logs: list[float] = []

    Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=2,
        batch_size=2,
        accumulate_steps=1,
        log_every=1,
        save_every=2,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        on_log=lambda step, loss, lr, elapsed: logs.append(loss),
    ).train()

    assert len(logs) == 2


# ---------------------------------------------------------------------------
# RETRO + Multi-Token Prediction combined (PR #180 x PR #181 intersection)
# ---------------------------------------------------------------------------

def test_training_run_with_retro_and_mtp_both_enabled(tmp_path) -> None:
    """A single training run with retro_layers set, neighbor_ids supplied,
    AND n_predict > 0 must complete without error, produce a finite loss,
    and persist both features' weights in the checkpoint. This is the exact
    combination the two PRs' merge conflict resolution had to get right:
    Trainer's forward call must pass neighbor_ids unconditionally while
    only conditionally requesting return_mtp_logits.
    """
    cfg = _tiny_config(retro_layers=[0], n_predict=2)
    model = GrimoireTransformer(cfg)
    dataset = _tiny_bin_dataset(str(tmp_path), cfg)
    neighbor_ids = np.random.randint(
        1, cfg.vocab_size, size=(len(dataset), 2, 8)
    ).astype(np.int32)
    logs: list[float] = []

    Trainer(
        model=model,
        train_dataset=dataset,
        neighbor_ids=neighbor_ids,
        mtp_loss_weight=0.3,
        total_steps=3,
        batch_size=2,
        accumulate_steps=1,
        log_every=1,
        save_every=3,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        on_log=lambda step, loss, lr, elapsed: logs.append(loss),
    ).train()

    assert len(logs) == 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in logs)

    ckpt = torch.load(str(tmp_path / "step_0000003.pt"), map_location="cpu", weights_only=True)
    state_dict = ckpt["model"]
    assert any(key.startswith("blocks.0.cca.") for key in state_dict), (
        "CCA weights missing from checkpoint with both features enabled."
    )
    assert any(key.startswith("mtp_transforms.") for key in state_dict), (
        "MTP head weights missing from checkpoint with both features enabled."
    )
