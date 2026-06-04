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
This implementation re-runs the full forward pass on every step — correct
but O(n²) in sequence length.  A KV-cache (caching K/V projections from
previous steps) would reduce this to O(n) per step.  That optimisation is
deferred to Phase 5 as it requires non-trivial changes to
``GroupedQueryAttention``.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from grimoire.llm.model.transformer import GrimoireTransformer
from grimoire.llm.tokenizer.special_tokens import EOS_ID


@dataclass
class GenerationConfig:
    """Hyperparameters that control the sampling strategy.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate beyond the prompt.
            Generation may stop earlier if ``<EOS>`` is emitted.
        temperature: Softmax temperature.  Values below 1.0 make the
            distribution sharper (more greedy); values above 1.0 make it
            flatter (more random).  Set to a very small value (e.g. 1e-8)
            for deterministic greedy decoding.
        top_k: If positive, only the ``top_k`` highest-probability tokens are
            kept before sampling.  Set to 0 to disable.
        top_p: Nucleus sampling threshold.  The smallest set of tokens whose
            cumulative probability exceeds ``top_p`` is kept; the rest are
            discarded.  Set to 1.0 to disable.
        repetition_penalty: Multiplicative penalty applied to the logits of
            tokens that already appear in the generated (not prompt) portion of
            the sequence.  Values > 1.0 reduce repetition; 1.0 disables.
    """

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0


def generate(
    model: GrimoireTransformer,
    prompt_ids: list[int],
    config: GenerationConfig | None = None,
    device: str = "cpu",
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

    Returns:
        A list of *newly generated* token ids (not including the prompt).
        The caller can decode this list with ``BytePairEncoder.decode``.

    Raises:
        ValueError: If ``prompt_ids`` is empty.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")

    if config is None:
        config = GenerationConfig()

    model.eval()
    max_seq = model.config.max_seq_len

    ids = list(prompt_ids)
    generated: list[int] = []

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            # Truncate from the left if the sequence exceeds max_seq_len.
            context = ids[-max_seq:]
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            logits = model(input_tensor)          # (1, seq, vocab)
            next_logits = logits[0, -1, :].float()  # (vocab,)

            # Repetition penalty on the generated portion only.
            if config.repetition_penalty != 1.0 and generated:
                unique_generated = set(generated)
                for token_id in unique_generated:
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= config.repetition_penalty
                    else:
                        next_logits[token_id] *= config.repetition_penalty

            # Temperature scaling.
            if config.temperature != 1.0:
                next_logits = next_logits / max(config.temperature, 1e-8)

            # Top-k masking.
            if config.top_k > 0:
                k = min(config.top_k, next_logits.size(-1))
                top_k_values = torch.topk(next_logits, k).values
                threshold = top_k_values[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            # Top-p (nucleus) masking.
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative probability exceeds top_p.
                # Shift by one so the token that pushes us over the threshold is kept.
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > config.top_p
                sorted_logits[remove_mask] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter_(
                    0, sorted_indices, sorted_logits
                )

            # Sample from the filtered distribution.
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == EOS_ID:
                break

            ids.append(next_token)
            generated.append(next_token)

    return generated
