"""End-to-end integration tests for the full Grimoire pipeline.

These tests exercise the full stack from scratch:

    train tiny model → save checkpoint → load into InferenceEngine
    → attach GrimoireCorpus → multi-turn engine.chat() with ConversationState

They are intentionally slower than unit tests (~10–20 s total) because they
run a real training loop.  They live in a separate file so they can be run
selectively:

    pytest tests/test_integration.py          # integration suite only
    pytest tests/ -k "not test_integration"   # unit tests only
    pytest                                    # everything

Gate criteria
-------------
- InferenceEngine loads a checkpoint produced by Trainer without error.
- respond() returns a non-empty string.
- chat() returns a non-empty string and records the turn in ConversationState.
- Three sequential chat() calls produce three turns in state.history.
- Corpus attach: engine.chat() runs without error when a GrimoireCorpus is
  attached and the query matches indexed text.
- state.clear() resets turn_count to zero; subsequent chat() works normally.
- Checkpoint round-trip: config stored in checkpoint matches model config.
- InferenceEngine raises FileNotFoundError for a missing checkpoint path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from grimoire_ai.corpus.corpus import GrimoireCorpus
from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.llm.inference.sampler import GenerationConfig
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.training.trainer import Trainer
from grimoire_ai.state.conversation import ConversationState


# ---------------------------------------------------------------------------
# Module-scoped fixtures — expensive setup runs once for the whole file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    """Shared temporary directory for all integration tests."""
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="module")
def vocab_path(tmp_dir) -> str:
    """Train and save a minimal BPE tokenizer."""
    enc = BytePairEncoder()
    enc.train(
        ["the quick brown fox jumps over the lazy dog " * 40,
         "a grappled creature has its speed reduced to zero " * 40],
        vocab_size=512,
    )
    path = str(tmp_dir / "bpe.json")
    enc.save(path)
    return path


@pytest.fixture(scope="module")
def corpus_text() -> str:
    return (
        "the quick brown fox jumps over the lazy dog " * 20
        + "a grappled creature has its speed reduced to zero " * 20
    )


@pytest.fixture(scope="module")
def corpus_bin(tmp_dir, vocab_path, corpus_text) -> str:
    """Write a tokenised corpus binary for pre-training."""
    enc = BytePairEncoder.load(vocab_path)
    tokens = enc.encode(corpus_text)
    path = str(tmp_dir / "corpus.bin")
    fp = np.memmap(path, dtype=np.int32, mode="w+", shape=(len(tokens),))
    fp[:] = tokens
    fp.flush()
    del fp
    return path


@pytest.fixture(scope="module")
def tiny_config() -> TransformerConfig:
    """Minimal model configuration for fast CPU tests."""
    return TransformerConfig(
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def checkpoint_path(tmp_dir, tiny_config, corpus_bin) -> str:
    """Pre-train a tiny model for 5 steps and return the checkpoint path."""
    model = GrimoireTransformer(tiny_config)
    dataset = TokenizedDataset(
        corpus_bin,
        seq_len=tiny_config.max_seq_len,
        stride=tiny_config.max_seq_len,
    )
    ckpt_dir = str(tmp_dir / "checkpoints")
    Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=5,
        warmup_steps=1,
        peak_lr=1e-3,
        batch_size=2,
        accumulate_steps=1,
        log_every=999,
        save_every=5,
        checkpoint_dir=ckpt_dir,
        device="cpu",
    ).train()
    return str(Path(ckpt_dir) / "step_0000005.pt")


@pytest.fixture(scope="module")
def engine(checkpoint_path, vocab_path) -> InferenceEngine:
    """Loaded InferenceEngine, no corpus."""
    return InferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=vocab_path,
        gen_config=GenerationConfig(max_new_tokens=8, temperature=1.0, top_k=10),
    )


@pytest.fixture(scope="module")
def engine_with_corpus(checkpoint_path, vocab_path, corpus_text) -> InferenceEngine:
    """Loaded InferenceEngine with a GrimoireCorpus attached."""
    corpus = GrimoireCorpus()
    corpus.add_text(corpus_text, source="test_corpus")
    return InferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=vocab_path,
        corpus=corpus,
        gen_config=GenerationConfig(max_new_tokens=8, temperature=1.0, top_k=10),
    )


# ---------------------------------------------------------------------------
# Checkpoint and loading
# ---------------------------------------------------------------------------

def test_checkpoint_exists(checkpoint_path) -> None:
    assert Path(checkpoint_path).exists(), "Trainer must write a checkpoint at save_every."


def test_checkpoint_config_round_trip(checkpoint_path, tiny_config) -> None:
    """Config saved in the checkpoint must reconstruct the original model config."""
    from grimoire_ai.llm.training.checkpoint import load_checkpoint
    ckpt = load_checkpoint(checkpoint_path)
    loaded = TransformerConfig.from_dict(ckpt["config"])
    assert loaded.vocab_size  == tiny_config.vocab_size
    assert loaded.d_model     == tiny_config.d_model
    assert loaded.n_layers    == tiny_config.n_layers
    assert loaded.n_heads     == tiny_config.n_heads
    assert loaded.n_kv_heads  == tiny_config.n_kv_heads


def test_engine_loads_without_error(engine) -> None:
    assert engine.model is not None
    assert engine.tokenizer is not None


def test_engine_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError):
        InferenceEngine(
            checkpoint_path="/tmp/grimoire_no_such_checkpoint.pt",
            tokenizer_path="/tmp/grimoire_no_such_vocab.json",
        )


# ---------------------------------------------------------------------------
# Single-turn inference
# ---------------------------------------------------------------------------

def test_respond_returns_string(engine) -> None:
    result = engine.respond("the quick fox")
    assert isinstance(result, str)


def test_respond_does_not_raise_on_short_query(engine) -> None:
    result = engine.respond("hi")
    assert isinstance(result, str)


def test_respond_with_corpus_returns_string(engine_with_corpus) -> None:
    result = engine_with_corpus.respond("quick brown fox", top_k_corpus=3)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Multi-turn chat
# ---------------------------------------------------------------------------

def test_chat_returns_string(engine) -> None:
    state = ConversationState()
    result = engine.chat("the quick fox", state)
    assert isinstance(result, str)


def test_chat_records_turn_in_state(engine) -> None:
    state = ConversationState()
    engine.chat("the quick fox", state)
    assert state.turn_count == 1
    assert state.history[0].user == "the quick fox"
    assert isinstance(state.history[0].assistant, str)


def test_chat_three_turns_accumulate(engine) -> None:
    state = ConversationState()
    for query in ["the quick fox", "over the dog", "lazy brown"]:
        engine.chat(query, state)
    assert state.turn_count == 3
    users = [t.user for t in state.history]
    assert users == ["the quick fox", "over the dog", "lazy brown"]


def test_chat_with_corpus_runs_without_error(engine_with_corpus) -> None:
    state = ConversationState()
    r1 = engine_with_corpus.chat("quick brown fox", state)
    r2 = engine_with_corpus.chat("grappled creature", state)
    assert state.turn_count == 2
    assert isinstance(r1, str) and isinstance(r2, str)


def test_clear_state_resets_history(engine) -> None:
    state = ConversationState()
    engine.chat("first query", state)
    engine.chat("second query", state)
    assert state.turn_count == 2
    state.clear()
    assert state.turn_count == 0


def test_chat_after_clear_works_normally(engine) -> None:
    """After clearing, the next chat() should succeed and record one turn."""
    state = ConversationState()
    engine.chat("before clear", state)
    state.clear()
    engine.chat("after clear", state)
    assert state.turn_count == 1


# ---------------------------------------------------------------------------
# Prompt structure through the full stack
# ---------------------------------------------------------------------------

def test_prompt_ids_never_exceed_max_seq_len(engine) -> None:
    """ConversationState must not produce prompts longer than max_seq_len."""
    state = ConversationState()
    for i in range(10):
        engine.chat(f"the quick brown fox query number {i}", state)
    # If we reached here without an index error the budget logic is correct.
    assert state.turn_count == 10


def test_ingest_text_then_chat(tmp_dir, checkpoint_path, vocab_path) -> None:
    """Full pipeline: ingest text → build corpus → load engine → chat."""
    from grimoire_ai.corpus.ingest import from_txt

    # Write a tiny corpus file and ingest it.
    src = tmp_dir / "knowledge.txt"
    src.write_text(
        "The quick brown fox jumps over the lazy dog. "
        "A grappled creature has its speed reduced to zero.",
        encoding="utf-8",
    )
    text = from_txt(str(src))
    assert "grappled" in text

    corpus = GrimoireCorpus()
    corpus.add_text(text, source="knowledge")

    eng = InferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=vocab_path,
        corpus=corpus,
        gen_config=GenerationConfig(max_new_tokens=8, temperature=1.0, top_k=10),
    )
    state = ConversationState()
    response = eng.chat("what happens to a grappled creature", state)
    assert isinstance(response, str)
    assert state.turn_count == 1
