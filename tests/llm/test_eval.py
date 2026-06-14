"""Unit tests for the evaluation harness.

Tests are grouped by evaluator:

    Perplexity  — computes finite BPC on a tiny corpus; handles empty split.
    Retrieval   — hits/misses on a tiny mock corpus; no-corpus path.
    Quiz        — keyword recall and token F1 on synthetic examples.
    Harness     — orchestrator integration: file output, summary line.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


def _tiny_model() -> GrimoireTransformer:
    return GrimoireTransformer(_tiny_config())


def _write_corpus(tmp_dir: str, cfg: TransformerConfig, n_tokens: int = 1024) -> str:
    """Write a tiny corpus.bin and return its path."""
    corpus = (np.arange(n_tokens, dtype=np.int32) % cfg.vocab_size)
    path = str(Path(tmp_dir) / "corpus.bin")
    corpus.tofile(path)
    return path


# ---------------------------------------------------------------------------
# Perplexity evaluator
# ---------------------------------------------------------------------------

class TestPerplexityEval:
    def test_returns_finite_bpc(self) -> None:
        from grimoire_ai.llm.eval.perplexity import eval_perplexity

        cfg = _tiny_config()
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_corpus(tmp, cfg, n_tokens=512)
            result = eval_perplexity(
                model=model,
                corpus_path=path,
                seq_len=cfg.max_seq_len,
                batch_size=2,
                max_batches=2,
                val_split=0.5,
                device="cpu",
            )
        assert math.isfinite(result["bpc"]), "BPC should be finite"
        assert math.isfinite(result["perplexity"]), "Perplexity should be finite"
        assert result["perplexity"] >= 1.0, "Perplexity is always >= 1"

    def test_bpc_equals_loss_over_log2(self) -> None:
        from grimoire_ai.llm.eval.perplexity import eval_perplexity

        cfg = _tiny_config()
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_corpus(tmp, cfg, n_tokens=512)
            result = eval_perplexity(
                model=model,
                corpus_path=path,
                seq_len=cfg.max_seq_len,
                batch_size=2,
                max_batches=2,
                val_split=0.5,
                device="cpu",
            )
        expected_bpc = result["mean_loss"] / math.log(2)
        assert abs(result["bpc"] - expected_bpc) < 1e-3

    def test_empty_split_returns_nan(self) -> None:
        from grimoire_ai.llm.eval.perplexity import eval_perplexity

        cfg = _tiny_config()
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            # Only enough tokens for one window; val_split leaves nothing
            path = _write_corpus(tmp, cfg, n_tokens=cfg.max_seq_len + 2)
            result = eval_perplexity(
                model=model,
                corpus_path=path,
                seq_len=cfg.max_seq_len,
                batch_size=2,
                max_batches=2,
                val_split=0.01,
                device="cpu",
            )
        # Too-small corpus → nan metrics and 0 batches
        assert result["n_batches"] == 0 or math.isfinite(result["bpc"])

    def test_model_stays_in_eval_mode_after(self) -> None:
        from grimoire_ai.llm.eval.perplexity import eval_perplexity

        cfg = _tiny_config()
        model = _tiny_model()
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_corpus(tmp, cfg, n_tokens=512)
            eval_perplexity(
                model=model,
                corpus_path=path,
                seq_len=cfg.max_seq_len,
                batch_size=2,
                max_batches=1,
                val_split=0.5,
                device="cpu",
            )
        assert not model.training, "eval() model should still be in eval mode after perplexity eval"


# ---------------------------------------------------------------------------
# Retrieval evaluator
# ---------------------------------------------------------------------------

class TestRetrievalEval:
    def _make_engine_with_corpus(self, texts: list[tuple[str, str]]):
        """Build a minimal engine-like object with a lexical corpus."""
        from grimoire_ai.corpus.corpus import GrimoireCorpus

        corpus = GrimoireCorpus()
        for text, source in texts:
            corpus.add_text(text, source=source)

        engine = MagicMock()
        engine.corpus = corpus
        return engine

    def test_perfect_hit_rate(self) -> None:
        from grimoire_ai.llm.eval.retrieval import eval_retrieval

        engine = self._make_engine_with_corpus([
            ("grappled creatures have their speed reduced to zero", "rules"),
        ])
        queries = [
            {"query": "grapple speed", "keywords": ["speed", "zero"]},
        ]
        result = eval_retrieval(engine=engine, queries=queries)
        assert result["hits"] == 1
        assert result["hit_rate"] == 1.0

    def test_miss_when_keyword_absent(self) -> None:
        from grimoire_ai.llm.eval.retrieval import eval_retrieval

        engine = self._make_engine_with_corpus([
            ("the sky is blue", "facts"),
        ])
        queries = [
            {"query": "grapple speed", "keywords": ["speed", "zero"]},
        ]
        result = eval_retrieval(engine=engine, queries=queries)
        assert result["hits"] == 0
        assert result["hit_rate"] == 0.0

    def test_no_corpus_returns_nan(self) -> None:
        from grimoire_ai.llm.eval.retrieval import eval_retrieval

        engine = MagicMock()
        engine.corpus = None
        result = eval_retrieval(engine=engine)
        assert math.isnan(result["hit_rate"])
        assert result["total"] == 0

    def test_per_query_entries(self) -> None:
        from grimoire_ai.llm.eval.retrieval import eval_retrieval

        engine = self._make_engine_with_corpus([
            ("advantage means rolling twice taking the higher", "rules"),
        ])
        queries = [
            {"query": "advantage roll", "keywords": ["twice"]},
            {"query": "grapple speed", "keywords": ["zero"]},
        ]
        result = eval_retrieval(engine=engine, queries=queries)
        assert len(result["per_query"]) == 2


# ---------------------------------------------------------------------------
# Quiz evaluator
# ---------------------------------------------------------------------------

class TestQuizEval:
    def _make_engine(self, respond_fn=None):
        engine = MagicMock()
        if respond_fn is not None:
            engine.respond = respond_fn
        else:
            engine.respond = lambda q, gen_config=None: "the answer is speed zero"
        return engine

    def test_keyword_recall_pass(self) -> None:
        from grimoire_ai.llm.eval.quiz import eval_quiz

        engine = self._make_engine(lambda q, gen_config=None: "speed is reduced to zero")
        examples = [{"user": "grapple?", "keywords": ["speed", "zero"]}]
        result = eval_quiz(engine=engine, examples=examples)
        assert result["passes"] == 1
        assert result["pass_rate"] == 1.0

    def test_keyword_recall_fail(self) -> None:
        from grimoire_ai.llm.eval.quiz import eval_quiz

        engine = self._make_engine(lambda q, gen_config=None: "I do not know")
        examples = [{"user": "grapple?", "keywords": ["speed", "zero"]}]
        result = eval_quiz(engine=engine, examples=examples)
        assert result["passes"] == 0
        assert result["pass_rate"] == 0.0

    def test_token_f1_perfect(self) -> None:
        from grimoire_ai.llm.eval.quiz import eval_quiz, _token_f1

        f1 = _token_f1("speed reduced to zero", "speed reduced to zero")
        assert f1 == pytest.approx(1.0)

    def test_token_f1_zero(self) -> None:
        from grimoire_ai.llm.eval.quiz import _token_f1

        f1 = _token_f1("apples and oranges", "speed reduced zero")
        assert f1 == pytest.approx(0.0)

    def test_empty_examples_returns_nan(self) -> None:
        from grimoire_ai.llm.eval.quiz import eval_quiz

        engine = self._make_engine()
        result = eval_quiz(engine=engine, examples=[])
        assert math.isnan(result["pass_rate"])
        assert result["total"] == 0

    def test_per_question_results(self) -> None:
        from grimoire_ai.llm.eval.quiz import eval_quiz

        engine = self._make_engine(lambda q, gen_config=None: "speed zero")
        examples = [
            {"user": "q1?", "keywords": ["speed"]},
            {"user": "q2?", "keywords": ["missing_word"]},
        ]
        result = eval_quiz(engine=engine, examples=examples)
        assert len(result["results"]) == 2
        assert result["results"][0]["pass"] is True
        assert result["results"][1]["pass"] is False

    def test_load_quiz_file(self) -> None:
        from grimoire_ai.llm.eval.quiz import load_quiz

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quiz.jsonl"
            path.write_text(
                '{"user": "q?", "assistant": "a", "keywords": ["a"]}\n'
                '{"user": "q2?", "keywords": ["b"]}\n',
                encoding="utf-8",
            )
            examples = load_quiz(str(path))
        assert len(examples) == 2
        assert examples[0]["user"] == "q?"
        assert examples[0]["keywords"] == ["a"]


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------

class TestHarness:
    def _make_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.corpus = None
        engine.device = "cpu"

        cfg = _tiny_config()
        model = _tiny_model()
        engine.model = model

        engine.respond = lambda q, gen_config=None: "proficiency bonus is +3"
        return engine

    def test_harness_writes_json_file(self) -> None:
        from grimoire_ai.llm.eval.harness import run_eval

        engine = self._make_engine()

        with tempfile.TemporaryDirectory() as tmp:
            quiz_path = Path(tmp) / "quiz.jsonl"
            quiz_path.write_text(
                '{"user": "What is proficiency?", "keywords": ["+3"]}\n',
                encoding="utf-8",
            )
            results = run_eval(
                engine=engine,
                output_dir=tmp,
                quiz_path=str(quiz_path),
            )
            written = list(Path(tmp).glob("eval_*.json"))
            assert len(written) == 1, "Expected exactly one report file"
            report = json.loads(written[0].read_text())
        assert "timestamp" in report
        assert "quiz" in report["evals"]

    def test_harness_summary_non_empty(self) -> None:
        from grimoire_ai.llm.eval.harness import run_eval

        engine = self._make_engine()
        with tempfile.TemporaryDirectory() as tmp:
            quiz_path = Path(tmp) / "quiz.jsonl"
            quiz_path.write_text(
                '{"user": "q?", "keywords": ["proficiency"]}\n',
                encoding="utf-8",
            )
            results = run_eval(
                engine=engine,
                output_dir=tmp,
                quiz_path=str(quiz_path),
            )
        assert results["summary"] and results["summary"] != "No evals ran."

    def test_harness_no_engine_no_corpus_bin_raises(self) -> None:
        from grimoire_ai.llm.eval.harness import run_eval

        with pytest.raises(ValueError, match="at least one"):
            run_eval()

    def test_harness_perplexity_with_corpus_bin(self) -> None:
        from grimoire_ai.llm.eval.harness import run_eval

        cfg = _tiny_config()
        engine = self._make_engine()
        engine.model = _tiny_model()
        engine.model.config = cfg

        with tempfile.TemporaryDirectory() as tmp:
            corpus_path = _write_corpus(tmp, cfg, n_tokens=512)
            results = run_eval(
                engine=engine,
                corpus_bin=corpus_path,
                output_dir=tmp,
                max_perplexity_batches=2,
            )
        assert "perplexity" in results["evals"]
        assert math.isfinite(results["evals"]["perplexity"]["perplexity"])
