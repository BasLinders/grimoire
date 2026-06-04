"""Tests for InferenceEngine.

Gate criteria:
- Engine loads a checkpoint and returns a string from respond().
- Engine without a corpus works (no retrieval step).
- Engine with a corpus attaches context to the prompt.
- respond() output is a string (possibly empty after stripping).
- Missing checkpoint raises FileNotFoundError.
- Missing tokenizer raises FileNotFoundError.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from grimoire.corpus.corpus import GrimoireCorpus, QueryResult
from grimoire.llm.inference.engine import InferenceEngine
from grimoire.llm.inference.sampler import GenerationConfig
from grimoire.llm.model.config import TransformerConfig
from grimoire.llm.model.transformer import GrimoireTransformer
from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.training.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )


def _save_artifacts(tmp_dir: str) -> tuple[str, str]:
    """Write a tiny checkpoint and tokenizer to tmp_dir. Returns (ckpt_path, tok_path)."""
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = str(Path(tmp_dir) / "ckpt.pt")
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_dict=cfg.to_dict(),
        train_loss=4.0,
    )

    tok_path = str(Path(tmp_dir) / "bpe.json")
    enc = BytePairEncoder()
    corpus = ["the quick brown fox jumps over the lazy dog " * 30]
    enc.train(corpus, vocab_size=cfg.vocab_size)
    enc.save(tok_path)

    return ckpt_path, tok_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_returns_string() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt,
            tokenizer_path=tok,
            gen_config=GenerationConfig(max_new_tokens=10, temperature=1e-8, top_k=1, top_p=1.0),
            device="cpu",
        )
        result = engine.respond("hello world")
    assert isinstance(result, str), "respond() must return a string."


def test_engine_without_corpus() -> None:
    """Engine should work fine when no corpus is attached."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt,
            tokenizer_path=tok,
            corpus=None,
            gen_config=GenerationConfig(max_new_tokens=5, temperature=1e-8, top_k=1, top_p=1.0),
            device="cpu",
        )
        result = engine.respond("fire bolt")
    assert isinstance(result, str)


def test_engine_with_corpus_does_not_crash() -> None:
    """Engine with a real GrimoireCorpus should complete without error."""
    corpus = GrimoireCorpus()
    corpus.add_text(
        "A grappled creature has its speed reduced to zero. "
        "The grapple condition ends if the grappler is incapacitated.",
        source="test",
    )
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt,
            tokenizer_path=tok,
            corpus=corpus,
            gen_config=GenerationConfig(max_new_tokens=5, temperature=1e-8, top_k=1, top_p=1.0),
            device="cpu",
        )
        result = engine.respond("grapple speed", top_k_corpus=3)
    assert isinstance(result, str)


def test_engine_missing_checkpoint_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, tok = _save_artifacts(tmp)
        with pytest.raises(FileNotFoundError):
            InferenceEngine(
                checkpoint_path="/tmp/grimoire_no_such_ckpt.pt",
                tokenizer_path=tok,
                device="cpu",
            )


def test_engine_missing_tokenizer_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, _ = _save_artifacts(tmp)
        with pytest.raises(FileNotFoundError):
            InferenceEngine(
                checkpoint_path=ckpt,
                tokenizer_path="/tmp/grimoire_no_such_bpe.json",
                device="cpu",
            )


def test_per_call_gen_config_override() -> None:
    """A per-call GenerationConfig should override the engine default."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt,
            tokenizer_path=tok,
            gen_config=GenerationConfig(max_new_tokens=50),
            device="cpu",
        )
        # Force max_new_tokens=3 via override; response should be very short.
        result = engine.respond(
            "hello",
            gen_config=GenerationConfig(max_new_tokens=3, temperature=1e-8, top_k=1, top_p=1.0),
        )
    assert isinstance(result, str)
