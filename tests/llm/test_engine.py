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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from grimoire_ai.corpus.corpus import GrimoireCorpus, QueryResult
from grimoire_ai.llm.inference.crag import CragFilter
from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.llm.inference.reranker import Reranker
from grimoire_ai.llm.inference.sampler import GenerationConfig
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import SEP_ID, USR_ID
from grimoire_ai.llm.training.checkpoint import save_checkpoint


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


def test_chat_stream_yields_match_final_response() -> None:
    """chat_stream's incremental yields must be monotonically growing
    prefixes of the final response, and the last yield must equal both its
    non-streaming sibling chat()'s output (same seed, same
    ConversationState-based prompt building) and the response ultimately
    recorded in ConversationState -- this is the property the incremental
    decoder (docs/inference_optimization.md item #5) has to preserve
    relative to the old full-redecode-every-token implementation."""
    from grimoire_ai.state.conversation import ConversationState

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        gen_config = GenerationConfig(max_new_tokens=15, temperature=0.9, top_k=20, top_p=0.95)

        torch.manual_seed(0)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, gen_config=gen_config, device="cpu",
        )
        stream_state = ConversationState()
        yields = list(engine.chat_stream("hello world", stream_state))

        torch.manual_seed(0)
        engine2 = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, gen_config=gen_config, device="cpu",
        )
        non_streamed = engine2.chat("hello world", ConversationState())

    assert len(yields) >= 1
    for earlier, later in zip(yields, yields[1:]):
        assert later.startswith(earlier), (
            f"Streaming yield {later!r} is not an extension of {earlier!r} -- "
            "incremental decode must never revise already-emitted text."
        )
    assert yields[-1] == non_streamed
    assert stream_state.history[-1].assistant == yields[-1]


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


def test_corpus_context_reaches_prompt() -> None:
    """The corpus results must actually be injected into the prompt.

    This is the integration assertion the smoke tests above do not make:
    we patch ``generate`` to capture the prompt_ids the engine builds and
    confirm the corpus context block (SEP…SEP) is present.
    """
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
            device="cpu",
        )
        # Sanity check: the corpus genuinely returns results for this query,
        # otherwise the assertion below would be vacuous.
        assert engine.corpus.query("grapple speed", top_k=3), (
            "Corpus returned no results; test would be vacuous."
        )

        captured: dict[str, list[int]] = {}

        def _fake_generate(model, prompt_ids, config=None, device="cpu", **_kwargs):
            captured["prompt_ids"] = prompt_ids
            return []

        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate):
            engine.respond("grapple speed", top_k_corpus=3)

    assert "prompt_ids" in captured, "generate() was never called."
    assert SEP_ID in captured["prompt_ids"], (
        "Corpus context block (SEP) did not reach the prompt — "
        "the corpus→PromptBuilder wiring is broken."
    )


def test_no_corpus_means_no_context_block() -> None:
    """Without a corpus the prompt must be the query-only layout (no SEP)."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt,
            tokenizer_path=tok,
            corpus=None,
            device="cpu",
        )

        captured: dict[str, list[int]] = {}

        def _fake_generate(model, prompt_ids, config=None, device="cpu", **_kwargs):
            captured["prompt_ids"] = prompt_ids
            return []

        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate):
            engine.respond("fire bolt")

    assert SEP_ID not in captured["prompt_ids"], "No corpus should mean no SEP block."
    assert USR_ID in captured["prompt_ids"], "Query-only prompt must still mark the user turn."


def test_top_k_corpus_forwarded_to_query() -> None:
    """respond(top_k_corpus=N) must pass N through to corpus.query."""
    corpus = MagicMock()
    corpus.query.return_value = []   # empty → query-only prompt, still valid

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt,
            tokenizer_path=tok,
            corpus=corpus,
            gen_config=GenerationConfig(max_new_tokens=2, temperature=1e-8, top_k=1, top_p=1.0),
            device="cpu",
        )
        engine.respond("anything", top_k_corpus=7)

    corpus.query.assert_called_once_with("anything", top_k=7)


# ---------------------------------------------------------------------------
# Reranker wiring (docs/architecture_optimization.md item #7)
# ---------------------------------------------------------------------------

def _reverse_score_fn(query: str, passages: list[str]) -> list[float]:
    """Fake cross-encoder: reverses whatever order it's given (last passage
    gets the highest score) -- makes rerank's effect on ordering unambiguous
    in tests, independent of any real ranking signal."""
    n = len(passages)
    return [float(i) for i in range(n)]


def test_reranker_none_is_noop() -> None:
    """With no reranker attached, _retrieve() must query for exactly top_k,
    not a widened pool -- regression guard for the default (unreranked) path."""
    corpus = MagicMock()
    corpus.query.return_value = []

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus, device="cpu",
        )
        assert engine.reranker is None
        engine._retrieve("anything", top_k=3)

    corpus.query.assert_called_once_with("anything", top_k=3)


def test_reranker_widens_query_then_truncates() -> None:
    """With a reranker attached, _retrieve() must fetch rerank_candidates
    (not just top_k) from the first-stage retriever, then truncate the
    rescored result back down to top_k."""
    candidates = [
        QueryResult(multi_token=(), next_token=None, score=0.0, source=None, excerpt=f"passage {i}")
        for i in range(10)
    ]
    corpus = MagicMock()
    corpus.query.return_value = candidates

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus, device="cpu",
            reranker=Reranker(_reverse_score_fn), rerank_candidates=10,
        )
        results = engine._retrieve("anything", top_k=3)

    corpus.query.assert_called_once_with("anything", top_k=10)
    assert len(results) == 3
    # _reverse_score_fn scores the last-seen candidate highest, so the
    # top 3 after rerank should be passages 9, 8, 7 in that order.
    assert [r.excerpt for r in results] == ["passage 9", "passage 8", "passage 7"]


def test_reranker_changes_prompt_order() -> None:
    """The reranked order must actually reach the assembled prompt, not just
    get reordered internally and ignored -- PromptBuilder joins context in
    list order and trims excess from the right, so reversing the order with
    a tight token budget must flip which excerpt survives truncation."""
    corpus = MagicMock()
    corpus.query.return_value = [
        QueryResult(multi_token=(), next_token=None, score=0.0, source=None, excerpt="zzzalpha zzzalpha zzzalpha"),
        QueryResult(multi_token=(), next_token=None, score=0.0, source=None, excerpt="zzzbeta zzzbeta zzzbeta"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus,
            max_context_tokens=20,  # tight budget: only one excerpt's tokens fit
            reranker=Reranker(_reverse_score_fn), rerank_candidates=2,
            device="cpu",
        )

        captured: dict[str, list[int]] = {}

        def _fake_generate(model, prompt_ids, config=None, device="cpu", **_kwargs):
            captured["prompt_ids"] = prompt_ids
            return []

        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate):
            engine.respond("query", top_k_corpus=2)

    decoded = engine.tokenizer.decode(captured["prompt_ids"])
    # _reverse_score_fn ranks the second (zzzbeta) candidate first, so after
    # rerank it's listed first and should survive the right-side trim.
    assert "zzzbeta" in decoded
    assert "zzzalpha" not in decoded


def test_retrieval_threshold_uses_reranked_score() -> None:
    """The top-1 retrieval_threshold gate must read the post-rerank score,
    not the first-stage retriever's original score."""
    corpus = MagicMock()
    # First-stage top result scores 0.0 (would fail a threshold of 0.5);
    # _reverse_score_fn promotes the last candidate to the top with score 1.0.
    corpus.query.return_value = [
        QueryResult(multi_token=(), next_token=None, score=0.0, source=None, excerpt="first"),
        QueryResult(multi_token=(), next_token=None, score=0.0, source=None, excerpt="second"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus,
            retrieval_threshold=0.5,
            reranker=Reranker(_reverse_score_fn), rerank_candidates=2,
            device="cpu",
        )
        results = engine._retrieve("anything", top_k=2)

    # Post-rerank top score is 1.0 (>= 0.5 threshold), so results must NOT
    # be dropped -- if the gate were still reading the pre-rerank score
    # (0.0 for both), this would incorrectly return [].
    assert len(results) == 2


# ---------------------------------------------------------------------------
# CRAG per-passage filter wiring (docs/architecture_optimization.md item #8)
# ---------------------------------------------------------------------------

def test_crag_filter_applied_before_top1_gate() -> None:
    """CRAG must drop individually low-scoring passages even when the top-1
    result alone would pass the existing retrieval_threshold gate -- proving
    per-passage filtering actually happens, not just the all-or-nothing
    top-1 check."""
    corpus = MagicMock()
    corpus.query.return_value = [
        QueryResult(multi_token=(), next_token=None, score=0.9, source=None, excerpt="zzzkeep zzzkeep"),
        QueryResult(multi_token=(), next_token=None, score=0.05, source=None, excerpt="zzzdrop1 zzzdrop1"),
        QueryResult(multi_token=(), next_token=None, score=0.05, source=None, excerpt="zzzdrop2 zzzdrop2"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus,
            retrieval_threshold=0.5,
            # Equal thresholds collapse the Ambiguous zone -- a plain
            # below/above cutoff, same as this test's own intent.
            crag_filter=CragFilter(lower_threshold=0.3, upper_threshold=0.3),
            device="cpu",
        )

        captured: dict[str, list[int]] = {}

        def _fake_generate(model, prompt_ids, config=None, device="cpu", **_kwargs):
            captured["prompt_ids"] = prompt_ids
            return []

        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate):
            engine.respond("query", top_k_corpus=3)

    decoded = engine.tokenizer.decode(captured["prompt_ids"])
    assert "zzzkeep" in decoded
    assert "zzzdrop1" not in decoded
    assert "zzzdrop2" not in decoded


def test_crag_demotes_ambiguous_passage_behind_correct_one() -> None:
    """An Ambiguous passage must be kept (not dropped) but demoted behind
    a Correct one -- end-to-end proof that PromptBuilder's existing
    right-side budget trim is what "demotion" actually relies on: with a
    tight budget, the demoted passage is the one that gets cut."""
    corpus = MagicMock()
    corpus.query.return_value = [
        # Ambiguous passage listed FIRST by the retriever...
        QueryResult(multi_token=(), next_token=None, score=0.5, source=None, excerpt="zzzambiguous zzzambiguous zzzambiguous"),
        # ...but the Correct one must still end up first after CragFilter,
        # and therefore be the one that survives a tight token budget.
        QueryResult(multi_token=(), next_token=None, score=0.9, source=None, excerpt="zzzcorrect zzzcorrect zzzcorrect"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus,
            max_context_tokens=20,  # tight budget: only one excerpt's tokens fit
            crag_filter=CragFilter(lower_threshold=0.3, upper_threshold=0.7),
            device="cpu",
        )

        captured: dict[str, list[int]] = {}

        def _fake_generate(model, prompt_ids, config=None, device="cpu", **_kwargs):
            captured["prompt_ids"] = prompt_ids
            return []

        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate):
            engine.respond("query", top_k_corpus=2)

    decoded = engine.tokenizer.decode(captured["prompt_ids"])
    assert "zzzcorrect" in decoded
    assert "zzzambiguous" not in decoded


def test_crag_empties_to_pure_chat_fallback() -> None:
    """When CRAG drops every passage, the engine must fall back to the
    pure-chat prompt shape (no SEP context block) -- even though the top-1
    score alone would have passed retrieval_threshold."""
    corpus = MagicMock()
    corpus.query.return_value = [
        QueryResult(multi_token=(), next_token=None, score=0.6, source=None, excerpt="a"),
        QueryResult(multi_token=(), next_token=None, score=0.55, source=None, excerpt="b"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus,
            retrieval_threshold=0.5,               # top-1 (0.6) alone would pass
            crag_filter=CragFilter(lower_threshold=0.9, upper_threshold=0.9),  # but both fail this
            device="cpu",
        )

        captured: dict[str, list[int]] = {}

        def _fake_generate(model, prompt_ids, config=None, device="cpu", **_kwargs):
            captured["prompt_ids"] = prompt_ids
            return []

        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate):
            engine.respond("query", top_k_corpus=2)

    assert SEP_ID not in captured["prompt_ids"], (
        "CRAG emptying the retrieved set must fall back to the no-context prompt."
    )


def test_crag_reads_reranked_score_when_reranker_present() -> None:
    """CragFilter must read the post-rerank score when a reranker is also
    attached, not the first-stage retriever's original score."""
    corpus = MagicMock()
    corpus.query.return_value = [
        QueryResult(multi_token=(), next_token=None, score=0.9, source=None, excerpt="first"),
        QueryResult(multi_token=(), next_token=None, score=0.9, source=None, excerpt="second"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, corpus=corpus,
            reranker=Reranker(_reverse_score_fn), rerank_candidates=2,
            crag_filter=CragFilter(lower_threshold=0.5, upper_threshold=0.5),
            device="cpu",
        )
        results = engine._retrieve("anything", top_k=2)

    # _reverse_score_fn gives "first" a post-rerank score of 0.0 (dropped by
    # the 0.5 CRAG threshold) and "second" a score of 1.0 (kept) -- if
    # CragFilter were still reading the pre-rerank score (0.9 for both),
    # both would survive instead.
    assert len(results) == 1
    assert results[0].excerpt == "second"


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
