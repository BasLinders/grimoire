"""Autoregressive token sampler for the GrimoireTransformer.

Generation algorithm
--------------------
Given a prompt (list of token ids), the sampler runs an autoregressive loop:

    1. Feed the current sequence to the model → logits of shape (1, seq, vocab).
    2. Take the logits at the *last* position only.
    3. Apply ``repetition_penalty`` to discount tokens that already appear in
       the generated portion.
    4. Divide by ``temperature`` (sharpens the distribution when < 1.0,
       flattens it when > 1.0).
    5. Apply ``top_k`` masking: zero out all logits except the top-k.
    6. Apply ``top_p`` (nucleus) masking: zero out tokens whose cumulative
       probability exceeds ``top_p`` when the vocabulary is sorted by
       descending probability.
    7. Sample one token from the resulting distribution.
    8. Append the sampled token to the sequence and repeat.

Stop conditions
---------------
Generation stops when the model emits ``<EOS>`` (id 2) or when
``max_new_tokens`` new tokens have been produced, whichever comes first.

KV-cache note
-------------
The prompt is run through the model once (the "prefill" pass) to populate a
per-layer KV cache; each subsequent step feeds only the single newly
sampled token and reuses the cache instead of re-running the full forward
pass, giving O(n) total cost in sequence length rather than O(n²).
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from grimoire_ai.llm.inference.constrained_decoding import RepetitionLoopGuard
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import EOS_ID

if TYPE_CHECKING:
    from grimoire_ai.llm.inference.constrained_decoding import StatBlockConstraint
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder


@dataclass
class GenerationConfig:
    """Hyperparameters that control the sampling strategy.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate beyond the prompt.
            Generation may stop earlier if ``<EOS>`` is emitted.
        temperature: Softmax temperature.  Values below 1.0 make the
            distribution sharper (more greedy); values above 1.0 make it
            flatter (more random).  Set to a very small value (e.g. 1e-8)
            for deterministic greedy decoding.  Ignored when
            ``adaptive_temperature`` is ``True``.
        top_k: If positive, only the ``top_k`` highest-probability tokens are
            kept before sampling.  Set to 0 to disable.
        top_p: Nucleus sampling threshold.  The smallest set of tokens whose
            cumulative probability exceeds ``top_p`` is kept; the rest are
            discarded.  Set to 1.0 to disable.
        repetition_penalty: Multiplicative penalty applied to the logits of
            tokens that already appear in the generated (not prompt) portion of
            the sequence.  Values > 1.0 reduce repetition; 1.0 disables.
        adaptive_temperature: When ``True`` the temperature is recomputed at
            every step from the model's own confidence (the normalised Shannon
            entropy of the next-token distribution) instead of using the fixed
            ``temperature`` value.  A confident, peaked distribution (low
            entropy) is given a *higher* temperature to add diversity without
            risking incoherence; an uncertain, flat distribution (high entropy)
            is given a *lower* temperature so the model commits to its most
            plausible continuations rather than sampling noise.  This is a
            statistically grounded alternative to a single hand-tuned
            temperature and to the blunt ``repetition_penalty``.
        adaptive_temp_floor: Lowest temperature the adaptive schedule may emit
            (applied when the model is maximally uncertain).
        adaptive_temp_ceiling: Highest temperature the adaptive schedule may
            emit (applied when the model is maximally confident).
        loop_guard_max_repeats: When > 0, enables a ``RepetitionLoopGuard``
            (see ``constrained_decoding.py``) that hard-bans the token which
            would extend an already-established repeating loop past this
            many consecutive copies — a structural backstop for
            ``repetition_penalty``, which only discounts repeats rather
            than forbidding them. ``0`` (default) disables it, matching
            this codebase's existing "0 = off" convention.
        loop_guard_max_period: Longest repeating block length (in tokens)
            the loop guard checks. Only consulted when
            ``loop_guard_max_repeats > 0``.
    """

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    adaptive_temperature: bool = False
    adaptive_temp_floor: float = 0.5
    adaptive_temp_ceiling: float = 1.3
    loop_guard_max_repeats: int = 0
    loop_guard_max_period: int = 4


def adaptive_temperature(
    logits: torch.Tensor,
    floor: float = 0.5,
    ceiling: float = 1.3,
) -> float:
    """Derive a sampling temperature from a distribution's normalised entropy.

    The next-token distribution is obtained from ``logits`` via softmax and its
    Shannon entropy ``H = -Σ p log p`` is normalised by ``log(vocab)`` so it
    lands in ``[0, 1]``: 0 means a one-hot (perfectly confident) prediction and
    1 means a uniform (maximally uncertain) one.

    The returned temperature interpolates *inversely* with uncertainty::

        temperature = floor + (ceiling - floor) * (1 - H_norm)

    so a confident model (``H_norm → 0``) samples near ``ceiling`` (more
    diverse) and an uncertain model (``H_norm → 1``) samples near ``floor``
    (more conservative).

    Args:
        logits: A 1-D logits tensor of shape ``(vocab,)``.
        floor: Minimum temperature (uncertain case).
        ceiling: Maximum temperature (confident case).

    Returns:
        A float temperature in ``[floor, ceiling]``.
    """
    vocab = logits.numel()
    if vocab <= 1:
        return floor
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    h_norm = float((entropy / math.log(vocab)).clamp(0.0, 1.0))
    return floor + (ceiling - floor) * (1.0 - h_norm)


def _resolve_loop_guard(config: GenerationConfig) -> Optional[RepetitionLoopGuard]:
    """Build a ``RepetitionLoopGuard`` from config, or ``None`` if disabled."""
    if config.loop_guard_max_repeats <= 0:
        return None
    return RepetitionLoopGuard(config.loop_guard_max_repeats, config.loop_guard_max_period)


def _apply_repetition_penalty(
    next_logits: torch.Tensor,
    generated: list[int],
    penalty: float,
) -> None:
    """Discount logits of tokens already present in ``generated``, in place.

    Vectorised equivalent of looping over ``set(generated)`` and scaling each
    token's logit individually: that loop does one Python-level scalar
    tensor read + write per unique generated token, every decode step (O(n)
    per step, O(n^2) over a full generation), which is especially costly on
    CUDA/MPS where each scalar access is effectively its own tiny kernel
    launch. This does the same discount as a single gather, one vectorised
    ``torch.where``, and one scatter, regardless of how many unique tokens
    have been generated so far.
    """
    penalty_ids = torch.unique(
        torch.tensor(generated, dtype=torch.long, device=next_logits.device)
    )
    penalty_logits = next_logits[penalty_ids]
    next_logits[penalty_ids] = torch.where(
        penalty_logits > 0,
        penalty_logits / penalty,
        penalty_logits * penalty,
    )


def _require_tokenizer_for_stat_block_constraint(
    stat_block_constraint: Optional["StatBlockConstraint"],
    tokenizer: Optional["BytePairEncoder"],
) -> None:
    if stat_block_constraint is not None and tokenizer is None:
        raise ValueError(
            "tokenizer is required when stat_block_constraint is set — "
            "StatBlockConstraint.mask() needs it to decode the generated "
            "text so far."
        )


def generate_stream(
    model: GrimoireTransformer,
    prompt_ids: list[int],
    config: GenerationConfig | None = None,
    device: str = "cpu",
    stat_block_constraint: Optional["StatBlockConstraint"] = None,
    tokenizer: Optional["BytePairEncoder"] = None,
):
    """Like ``generate`` but yields one token id at a time as it is sampled.

    Useful for streaming UIs that want to display tokens as they arrive rather
    than waiting for the full response.

    Args:
        model: A trained ``GrimoireTransformer`` in eval mode.
        prompt_ids: Token ids for the prompt.
        config: Sampling configuration.
        device: PyTorch device string.
        stat_block_constraint: Optional ``StatBlockConstraint`` (see
            ``constrained_decoding.py``) restricting stat-block field
            values to well-formed continuations. Requires ``tokenizer``.
        tokenizer: Required when ``stat_block_constraint`` is set — used to
            decode the generated-so-far text at each step.

    Raises:
        ValueError: If ``prompt_ids`` is empty, or if
            ``stat_block_constraint`` is given without ``tokenizer``.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")
    _require_tokenizer_for_stat_block_constraint(stat_block_constraint, tokenizer)

    if config is None:
        config = GenerationConfig()
    loop_guard = _resolve_loop_guard(config)

    model.eval()
    max_seq = model.config.max_seq_len

    ids = list(prompt_ids)
    generated: list[int] = []

    prompt_context = ids[-max_seq:]
    prompt_tensor = torch.tensor([prompt_context], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, past_kvs = model(prompt_tensor, use_cache=True)

    next_logits = logits[0, -1, :].float().clone()

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            if config.repetition_penalty != 1.0 and generated:
                _apply_repetition_penalty(next_logits, generated, config.repetition_penalty)

            # Structural constraints — hard-mask before the probability-shaping
            # steps below. Masking to -inf commutes with temperature scaling
            # and with top-k/top-p (both only ever remove further candidates),
            # so applying these first or last produces the same distribution;
            # grouped here with the other pre-sampling adjustments for clarity.
            if loop_guard is not None:
                next_logits = loop_guard.mask(next_logits, generated)
            if stat_block_constraint is not None:
                next_logits = stat_block_constraint.mask(next_logits, generated, tokenizer)

            # Temperature scaling — adaptive (entropy-derived) or fixed.
            if config.adaptive_temperature:
                temp = adaptive_temperature(
                    next_logits,
                    config.adaptive_temp_floor,
                    config.adaptive_temp_ceiling,
                )
            else:
                temp = config.temperature
            if temp != 1.0:
                next_logits = next_logits / max(temp, 1e-8)

            if config.top_k > 0:
                k = min(config.top_k, next_logits.size(-1))
                top_k_values = torch.topk(next_logits, k).values
                threshold = top_k_values[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                # Compute softmax once and reuse to avoid fragile double-call.
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
                # Shift by one so the token that pushes us over the threshold is kept.
                remove_mask = cumulative_probs - probs_sorted > config.top_p
                sorted_logits[remove_mask] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter_(
                    0, sorted_indices, sorted_logits
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

            if next_token == EOS_ID:
                break

            generated.append(next_token)
            yield next_token

            past_len = past_kvs[0][0].shape[2]
            if past_len >= max_seq:
                past_kvs = [
                    (kv[0][:, :, 1:, :], kv[1][:, :, 1:, :])
                    for kv in past_kvs
                ]

            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, past_kvs = model(token_tensor, past_kvs=past_kvs, use_cache=True)
            next_logits = logits[0, -1, :].float().clone()


def generate(
    model: GrimoireTransformer,
    prompt_ids: list[int],
    config: GenerationConfig | None = None,
    device: str = "cpu",
    stat_block_constraint: Optional["StatBlockConstraint"] = None,
    tokenizer: Optional["BytePairEncoder"] = None,
) -> list[int]:
    """Generate tokens autoregressively from a prompt.

    Args:
        model: A trained ``GrimoireTransformer``.  Should be in eval mode
            before calling (dropout is disabled automatically via
            ``torch.no_grad``).
        prompt_ids: Token ids for the prompt, as produced by
            ``PromptBuilder.build``.
        config: Sampling configuration.  Defaults to ``GenerationConfig()``
            if ``None``.
        device: PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
        stat_block_constraint: Optional ``StatBlockConstraint`` (see
            ``constrained_decoding.py``) restricting stat-block field
            values to well-formed continuations. Requires ``tokenizer``.
        tokenizer: Required when ``stat_block_constraint`` is set — used to
            decode the generated-so-far text at each step.

    Returns:
        A list of *newly generated* token ids (not including the prompt).
        The caller can decode this list with ``BytePairEncoder.decode``.

    Raises:
        ValueError: If ``prompt_ids`` is empty, or if
            ``stat_block_constraint`` is given without ``tokenizer``.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")
    _require_tokenizer_for_stat_block_constraint(stat_block_constraint, tokenizer)

    if config is None:
        config = GenerationConfig()
    loop_guard = _resolve_loop_guard(config)

    model.eval()
    max_seq = model.config.max_seq_len

    ids = list(prompt_ids)
    generated: list[int] = []

    # --- Prompt pass (full sequence, cache populated) -------------------
    # Truncate from the left if the prompt itself exceeds max_seq_len.
    prompt_context = ids[-max_seq:]
    prompt_tensor = torch.tensor([prompt_context], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, past_kvs = model(prompt_tensor, use_cache=True)

    next_logits = logits[0, -1, :].float().clone()

    # --- Autoregressive generation loop (one token at a time) -----------
    # After the prompt pass we feed only the single newly sampled token on
    # each step; the KV cache holds the full context so the model sees the
    # whole history at O(1) cost per step instead of O(n).
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            # Apply sampling filters to the logits from the previous step.

            # Repetition penalty on the generated portion only.
            if config.repetition_penalty != 1.0 and generated:
                _apply_repetition_penalty(next_logits, generated, config.repetition_penalty)

            # Structural constraints — hard-mask before the probability-shaping
            # steps below (see the matching comment in generate_stream for why
            # the ordering relative to temperature/top-k/top-p doesn't matter).
            if loop_guard is not None:
                next_logits = loop_guard.mask(next_logits, generated)
            if stat_block_constraint is not None:
                next_logits = stat_block_constraint.mask(next_logits, generated, tokenizer)

            # Temperature scaling — adaptive (entropy-derived) or fixed.
            if config.adaptive_temperature:
                temp = adaptive_temperature(
                    next_logits,
                    config.adaptive_temp_floor,
                    config.adaptive_temp_ceiling,
                )
            else:
                temp = config.temperature
            if temp != 1.0:
                next_logits = next_logits / max(temp, 1e-8)

            # Top-k masking.
            if config.top_k > 0:
                k = min(config.top_k, next_logits.size(-1))
                top_k_values = torch.topk(next_logits, k).values
                threshold = top_k_values[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            # Top-p (nucleus) masking.
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                # Compute softmax once and reuse to avoid fragile double-call.
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
                # Shift by one so the token that pushes us over the threshold is kept.
                remove_mask = cumulative_probs - probs_sorted > config.top_p
                sorted_logits[remove_mask] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter_(
                    0, sorted_indices, sorted_logits
                )

            # Sample from the filtered distribution.
            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

            if next_token == EOS_ID:
                break

            generated.append(next_token)

            # Truncate the cache when it has reached max_seq_len so RoPE
            # position indices and the causal mask never go out of bounds.
            # Drop the oldest token from every layer's K and V.
            past_len = past_kvs[0][0].shape[2]
            if past_len >= max_seq:
                past_kvs = [
                    (kv[0][:, :, 1:, :], kv[1][:, :, 1:, :])
                    for kv in past_kvs
                ]

            # Feed the single new token with the updated cache.
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, past_kvs = model(token_tensor, past_kvs=past_kvs, use_cache=True)
            next_logits = logits[0, -1, :].float().clone()

    return generated
