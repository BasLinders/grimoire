"""Tests for the autoregressive sampler.

Gate criteria:
- Greedy decoding (temperature → 0) always picks the argmax token.
- Generation stops at EOS before reaching max_new_tokens.
- Output length never exceeds max_new_tokens even without EOS.
- Top-k=1 is equivalent to greedy (deterministic).
- Empty prompt raises ValueError.
- Repetition penalty reduces the probability of repeated tokens.
"""

import pytest
import torch
import torch.nn as nn

from grimoire.llm.inference.sampler import GenerationConfig, generate
from grimoire.llm.model.config import TransformerConfig
from grimoire.llm.model.transformer import GrimoireTransformer
from grimoire.llm.tokenizer.special_tokens import EOS_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )


def _tiny_model() -> GrimoireTransformer:
    return GrimoireTransformer(_tiny_config())


class _EOSModel(nn.Module):
    """Stub model that always outputs EOS as the highest-logit token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, max_seq_len=32)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False,
                past_kvs=None, **_kwargs):
        batch, seq = input_ids.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        logits = torch.full((batch, seq, self.config.vocab_size), -1e9)
        logits[:, :, EOS_ID] = 1e9
        if use_cache:
            full = past_len + seq
            kv = (torch.zeros(batch, 1, full, 1), torch.zeros(batch, 1, full, 1))
            return logits, [kv]
        return logits


class _ConstantModel(nn.Module):
    """Stub model that always outputs token 10 as the highest-logit."""

    def __init__(self, vocab_size: int, chosen_token: int = 10) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, max_seq_len=32)
        self._dummy = nn.Parameter(torch.zeros(1))
        self.chosen_token = chosen_token

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False,
                past_kvs=None, **_kwargs):
        batch, seq = input_ids.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        logits = torch.full((batch, seq, self.config.vocab_size), -1e9)
        logits[:, :, self.chosen_token] = 1e9
        if use_cache:
            full = past_len + seq
            kv = (torch.zeros(batch, 1, full, 1), torch.zeros(batch, 1, full, 1))
            return logits, [kv]
        return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_prompt_raises() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError):
        generate(model, [], config=GenerationConfig(max_new_tokens=5))


def test_stops_at_eos() -> None:
    """Model that always outputs EOS should produce an empty result."""
    model = _EOSModel(vocab_size=64)
    result = generate(
        model,
        prompt_ids=[1, 4, 6, 7],   # BOS USR two tokens
        config=GenerationConfig(max_new_tokens=20, temperature=1e-8, top_k=0, top_p=1.0),
    )
    assert result == [], f"Expected empty list when model emits EOS immediately, got {result}."


def test_max_new_tokens_respected() -> None:
    """Without EOS the output must be exactly max_new_tokens long."""
    model = _ConstantModel(vocab_size=64, chosen_token=10)
    limit = 7
    result = generate(
        model,
        prompt_ids=[1, 4, 6],
        config=GenerationConfig(max_new_tokens=limit, temperature=1e-8, top_k=0, top_p=1.0),
    )
    assert len(result) == limit, f"Expected {limit} tokens, got {len(result)}."


def test_greedy_picks_argmax() -> None:
    """With temperature → 0 and top_k=1 the sampler must pick the argmax."""
    model = _ConstantModel(vocab_size=64, chosen_token=10)
    result = generate(
        model,
        prompt_ids=[1, 4, 6],
        config=GenerationConfig(max_new_tokens=5, temperature=1e-8, top_k=1, top_p=1.0),
    )
    assert all(t == 10 for t in result), f"Greedy should always pick token 10, got {result}."


def test_top_k_one_is_deterministic() -> None:
    """top_k=1 is greedy: two calls with the same seed must agree."""
    model = _tiny_model()
    model.eval()
    prompt = [1, 4, 6, 7, 8]
    cfg = GenerationConfig(max_new_tokens=10, temperature=1.0, top_k=1, top_p=1.0)

    torch.manual_seed(0)
    r1 = generate(model, prompt, cfg)
    torch.manual_seed(0)
    r2 = generate(model, prompt, cfg)
    assert r1 == r2, "top_k=1 should be deterministic."


def test_output_is_list_of_ints() -> None:
    model = _tiny_model()
    result = generate(
        model,
        prompt_ids=[1, 4, 6],
        config=GenerationConfig(max_new_tokens=4, temperature=1e-8, top_k=1, top_p=1.0),
    )
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


def test_repetition_penalty_reduces_repeats() -> None:
    """With a high repetition penalty the same token should appear less often."""
    model = _tiny_model()
    model.eval()
    prompt = [1, 4, 10, 10, 10]   # lots of token 10 already in prompt context

    torch.manual_seed(42)
    no_penalty = generate(
        model, prompt,
        config=GenerationConfig(max_new_tokens=20, temperature=1.0, top_k=0, top_p=1.0,
                                repetition_penalty=1.0),
    )
    torch.manual_seed(42)
    with_penalty = generate(
        model, prompt,
        config=GenerationConfig(max_new_tokens=20, temperature=1.0, top_k=0, top_p=1.0,
                                repetition_penalty=2.0),
    )
    # With penalty, token 10 should appear less (or equal) in the output.
    count_no_penalty   = no_penalty.count(10)
    count_with_penalty = with_penalty.count(10)
    assert count_with_penalty <= count_no_penalty, (
        f"Repetition penalty should reduce token 10 count: "
        f"no_penalty={count_no_penalty}, with_penalty={count_with_penalty}."
    )
