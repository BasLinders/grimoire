"""Tests for the kNN-LM generation wiring (docs/KNN_LM.md, PR 2).

Runs against a tiny real ``GrimoireTransformer`` (not a mock), so interface
mismatches surface here.

Gate criteria:
- ``context_span`` finds exactly the ``<BOS><SEP>{ctx}<SEP>`` block, and
  nothing else.
- The store built at prefill holds one entry per context token, with keys
  equal to the ``final_norm`` output one position earlier.
- Prompts without a context block are untouched: seeded sampling output is
  identical with and without ``knn_lm``.
- With λ ≈ 0, greedy output on a context prompt is identical to plain
  ``generate``; rescored logits stay on the raw-logit scale.
- With λ at its cap and a tiny temperature, greedy output only uses context
  tokens (the store measurably steers generation).
- ``generate_stream`` matches ``generate`` under kNN.
- The ``final_norm`` hook never outlives a forward call.
- ``InferenceEngine`` forwards ``knn_lm`` to every generation path.
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.llm.inference.knn_lm import (
    KNNLM,
    KNNConfig,
    LambdaGate,
    context_span,
)
from grimoire_ai.llm.inference.sampler import GenerationConfig, generate, generate_stream
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import AST_ID, BOS_ID, SEP_ID, USR_ID
from grimoire_ai.llm.training.checkpoint import save_checkpoint
from grimoire_ai.state.conversation import ConversationState

CONTEXT = list(range(10, 22))
QUERY = [40, 41]
CTX_PROMPT = [BOS_ID, SEP_ID, *CONTEXT, SEP_ID, USR_ID, *QUERY, AST_ID]
PLAIN_PROMPT = [BOS_ID, USR_ID, *QUERY, AST_ID]
GREEDY = GenerationConfig(max_new_tokens=8, temperature=1.0, top_k=1, top_p=1.0)


def _tiny_model() -> GrimoireTransformer:
    torch.manual_seed(0)
    cfg = TransformerConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        d_ff=64, max_seq_len=32, dropout=0.0,
    )
    return GrimoireTransformer(cfg).eval()


def _knn(bias: float, temperature: float = 10.0, lam_max: float = 0.8) -> KNNLM:
    gate = LambdaGate(lam_max=lam_max, init_temperature=temperature)
    with torch.no_grad():
        gate.lin.bias.fill_(bias)
    return KNNLM(gate, KNNConfig(k=4, lam_max=lam_max))


class _RecordingKNNLM(KNNLM):
    """KNNLM that keeps every session it opens, for inspection."""

    def __init__(self, base: KNNLM) -> None:
        super().__init__(base.gate, base.cfg)
        self.sessions = []

    def session(self):
        s = super().session()
        self.sessions.append(s)
        return s


# ------------------------------------------------------------- context_span


def test_context_span_finds_sep_block() -> None:
    assert context_span(CTX_PROMPT) == (2, 2 + len(CONTEXT))
    assert CTX_PROMPT[slice(*context_span(CTX_PROMPT))] == CONTEXT


def test_context_span_none_without_context() -> None:
    assert context_span(PLAIN_PROMPT) is None
    assert context_span([BOS_ID, SEP_ID, SEP_ID, USR_ID, AST_ID]) is None  # empty block
    assert context_span([BOS_ID, SEP_ID, 10, 11]) is None                 # unclosed
    assert context_span([SEP_ID, *CONTEXT, SEP_ID]) is None              # no BOS
    assert context_span([]) is None


def test_context_span_matches_conversation_prompt_layout() -> None:
    """chat()/chat_stream() build prompts via ConversationState, not PromptBuilder."""
    enc = BytePairEncoder()
    enc.train(["the quick brown fox jumps over the lazy dog " * 30], vocab_size=300)
    state = ConversationState()
    state.add_turn("earlier question", "earlier answer")
    ctx = enc.encode("the lazy dog")
    ids = state.build_prompt_ids(
        query="what jumps", tokenizer=enc, context_ids=ctx, max_seq_len=256
    )
    assert ids[slice(*context_span(ids))] == ctx


# ------------------------------------------------------------ store at prefill


def test_prefill_store_matches_final_norm_states() -> None:
    model = _tiny_model()
    session = _knn(bias=-1.0).session()
    session.prefill(model, torch.tensor([CTX_PROMPT]))

    captured = {}
    handle = model.final_norm.register_forward_hook(
        lambda _m, _i, out: captured.setdefault("h", out.detach())
    )
    with torch.no_grad():
        model(torch.tensor([CTX_PROMPT]))
    handle.remove()

    start, end = context_span(CTX_PROMPT)
    assert session.store.size == len(CONTEXT)
    assert session.store.values.tolist() == CONTEXT
    assert torch.allclose(session.store.keys, captured["h"][0, start - 1 : end - 1], atol=1e-5)


def test_hook_never_outlives_a_forward_call() -> None:
    model = _tiny_model()
    generate(model, CTX_PROMPT, GREEDY, knn_lm=_knn(bias=-1.0))
    assert len(model.final_norm._forward_hooks) == 0


# ------------------------------------------------------------- no-op paths


def test_no_context_prompt_is_identical_to_plain_sampling() -> None:
    model = _tiny_model()
    cfg = GenerationConfig(max_new_tokens=10, temperature=1.0, top_k=0, top_p=1.0)
    knn = _RecordingKNNLM(_knn(bias=5.0))

    torch.manual_seed(123)
    plain = generate(model, PLAIN_PROMPT, cfg)
    torch.manual_seed(123)
    with_knn = generate(model, PLAIN_PROMPT, cfg, knn_lm=knn)

    assert with_knn == plain
    assert knn.sessions[0].store is None
    assert knn.sessions[0].lambdas == []


def test_left_truncated_prompt_loses_its_store() -> None:
    """generate() keeps only the last max_seq_len ids; a prompt cut past its
    <BOS><SEP> header has no recognisable context block."""
    model = _tiny_model()
    long_prompt = [BOS_ID, SEP_ID] + [10] * 40 + [SEP_ID, USR_ID, *QUERY, AST_ID]
    knn = _RecordingKNNLM(_knn(bias=5.0))
    generate(model, long_prompt, GREEDY, knn_lm=knn)
    assert knn.sessions[0].store is None


def test_near_zero_lambda_matches_plain_greedy() -> None:
    model = _tiny_model()
    knn = _RecordingKNNLM(_knn(bias=-50.0))

    plain = generate(model, CTX_PROMPT, GREEDY)
    with_knn = generate(model, CTX_PROMPT, GREEDY, knn_lm=knn)

    assert with_knn == plain
    # One λ per rescored forward: the prefill plus every step. The sampler
    # also runs a forward after the final token (its logits go unused), so
    # there's at least one λ per generated token.
    assert len(knn.sessions[0].lambdas) >= len(with_knn)
    assert max(knn.sessions[0].lambdas) < 1e-5


def test_rescored_logits_stay_on_raw_logit_scale() -> None:
    """The logsumexp shift keeps rescored values comparable to raw logits,
    so repetition penalty (sign-dependent) behaves as it does without kNN."""
    model = _tiny_model()
    prompt = torch.tensor([CTX_PROMPT])
    with torch.no_grad():
        raw, _ = model(prompt, use_cache=True)
    rescored, _ = _knn(bias=-50.0).session().prefill(model, prompt)
    assert torch.allclose(rescored, raw[0, -1].float(), atol=1e-4)


# ------------------------------------------------------------ kNN steers


def test_strong_knn_steers_greedy_output_into_context_tokens() -> None:
    model = _tiny_model()
    knn = _RecordingKNNLM(_knn(bias=50.0, temperature=1e-4))

    out = generate(model, CTX_PROMPT, GREEDY, knn_lm=knn)

    assert out, "expected at least one generated token"
    assert set(out) <= set(CONTEXT)
    assert all(math.isclose(lam, 0.8, rel_tol=1e-4) for lam in knn.sessions[0].lambdas)


def test_stream_matches_generate_under_knn() -> None:
    model = _tiny_model()
    knn = _knn(bias=0.5, temperature=50.0)
    assert list(generate_stream(model, CTX_PROMPT, GREEDY, knn_lm=knn)) == generate(
        model, CTX_PROMPT, GREEDY, knn_lm=knn
    )


# ------------------------------------------------------------------ engine


def _engine(tmp: str) -> InferenceEngine:
    cfg = TransformerConfig(
        vocab_size=512, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        d_ff=64, max_seq_len=32, dropout=0.0,
    )
    model = GrimoireTransformer(cfg)
    ckpt = str(Path(tmp) / "ckpt.pt")
    save_checkpoint(
        path=ckpt, model=model, optimizer=torch.optim.AdamW(model.parameters()),
        step=1, config_dict=cfg.to_dict(), train_loss=4.0,
    )
    tok = str(Path(tmp) / "bpe.json")
    enc = BytePairEncoder()
    enc.train(["the quick brown fox jumps over the lazy dog " * 30], vocab_size=512)
    enc.save(tok)
    return InferenceEngine(ckpt, tok, device="cpu", knn_lm=_knn(bias=-1.0))


def test_engine_forwards_knn_lm_on_every_path() -> None:
    seen = []

    def _fake_generate(model, prompt_ids, config=None, device="cpu", **kwargs):
        seen.append(kwargs.get("knn_lm"))
        return []

    def _fake_stream(model, prompt_ids, config=None, device="cpu", **kwargs):
        seen.append(kwargs.get("knn_lm"))
        return iter(())

    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(tmp)
        with patch("grimoire_ai.llm.inference.engine.generate", _fake_generate), \
             patch("grimoire_ai.llm.inference.engine.generate_stream", _fake_stream):
            engine.respond("fox")
            engine.chat("fox", ConversationState())
            list(engine.chat_stream("fox", ConversationState()))

    assert len(seen) == 3
    assert all(k is engine.knn_lm for k in seen)


def test_engine_defaults_to_no_knn() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(tmp)
        engine_default = InferenceEngine(
            engine.checkpoint_path, str(Path(tmp) / "bpe.json"), device="cpu"
        )
        assert engine_default.knn_lm is None
