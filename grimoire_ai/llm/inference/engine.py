"""InferenceEngine: end-to-end query → response pipeline.

This is the public object that agents (Saga, planning assistant, etc.) use.
It owns all components — a loaded model, the BPE tokenizer, an optional
GrimoireCorpus, a PromptBuilder, and a GenerationConfig — and exposes a
single ``respond`` method.

Typical usage
-------------
    from grimoire_ai.llm.inference.engine import InferenceEngine

    engine = InferenceEngine(
        checkpoint_path="data/checkpoints/step_5000.pt",
        tokenizer_path="data/tokenizer/bpe.json",
    )
    print(engine.respond("What happens when a creature is grappled?"))

With corpus grounding
---------------------
    from grimoire_ai.corpus.corpus import GrimoireCorpus

    corpus = GrimoireCorpus()
    corpus.add_text(open("data/raw/dnd_srd.txt").read(), source="dnd_srd")

    engine = InferenceEngine(
        checkpoint_path="data/checkpoints/step_5000.pt",
        tokenizer_path="data/tokenizer/bpe.json",
        corpus=corpus,
    )
    print(engine.respond("grapple speed movement", top_k_corpus=5))
"""

from typing import Optional

import torch

from grimoire_ai.corpus.corpus import GrimoireCorpus
from grimoire_ai.llm.inference.prompt import PromptBuilder
from grimoire_ai.llm.inference.sampler import GenerationConfig, generate
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import AST_ID, EOS_ID
from grimoire_ai.llm.training.checkpoint import load_checkpoint


class InferenceEngine:
    """End-to-end inference pipeline for GrimoireTransformer.

    Loads a trained checkpoint, wires up all components, and exposes a
    single ``respond`` method that accepts a plain-text query and returns a
    plain-text response.

    Attributes:
        model: The loaded ``GrimoireTransformer`` in eval mode.
        tokenizer: The ``BytePairEncoder`` used to encode prompts and decode
            responses.
        corpus: Optional ``GrimoireCorpus`` for retrieval-augmented generation.
        prompt_builder: ``PromptBuilder`` configured with the tokenizer and
            context budget.
        gen_config: Default ``GenerationConfig`` used when ``respond`` is
            called without an explicit config override.
        device: PyTorch device string.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        corpus: Optional[GrimoireCorpus] = None,
        gen_config: Optional[GenerationConfig] = None,
        max_context_tokens: int = 512,
        device: Optional[str] = None,
    ) -> None:
        """Load model and tokenizer from disk and prepare the engine.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint written by
                ``save_checkpoint``.
            tokenizer_path: Path to a BPE vocabulary JSON written by
                ``BytePairEncoder.save``.
            corpus: Optional ``GrimoireCorpus`` instance.  When provided,
                ``respond`` will query it and inject the results as context.
            gen_config: Default generation hyperparameters.  Defaults to
                ``GenerationConfig()`` if ``None``.
            max_context_tokens: Token budget passed to ``PromptBuilder``.
                Must be at most ``model.config.max_seq_len``.
            device: PyTorch device (``"cpu"``, ``"cuda"``, etc.).  Auto-
                detected (CUDA if available, otherwise CPU) when ``None``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load checkpoint and reconstruct the model.
        ckpt = load_checkpoint(checkpoint_path)
        config = TransformerConfig.from_dict(ckpt["config"])
        self.model = GrimoireTransformer(config)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)
        self.model.eval()

        # Load tokenizer.
        self.tokenizer = BytePairEncoder.load(tokenizer_path)

        self.corpus = corpus
        # A prompt can never usefully exceed the model's context window; clamp
        # so PromptBuilder never emits a prompt that generate() would silently
        # left-truncate (which could drop the <USR>/context framing).
        effective_context = min(max_context_tokens, config.max_seq_len)
        self.prompt_builder = PromptBuilder(
            tokenizer=self.tokenizer,
            max_context_tokens=effective_context,
        )
        self.gen_config = gen_config if gen_config is not None else GenerationConfig()

    def respond(
        self,
        query: str,
        top_k_corpus: int = 5,
        gen_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a response to a user query.

        Pipeline:
        1. Query the corpus (if one was provided) for ``top_k_corpus`` results.
        2. Build the prompt token-id sequence via ``PromptBuilder``.
        3. Run autoregressive generation with ``generate()``.
        4. Decode the new tokens back to a string with the BPE tokenizer.

        Args:
            query: Plain-text user query.
            top_k_corpus: Number of corpus results to retrieve and inject as
                context.  Ignored when no corpus is attached.
            gen_config: Per-call generation config override.  Falls back to
                ``self.gen_config`` when ``None``.

        Returns:
            The generated response as a plain-text string, with leading and
            trailing whitespace stripped.
        """
        cfg = gen_config if gen_config is not None else self.gen_config

        results = []
        if self.corpus is not None:
            results = self.corpus.query(query, top_k=top_k_corpus)

        prompt_ids = self.prompt_builder.build(query, results)
        new_token_ids = generate(
            model=self.model,
            prompt_ids=prompt_ids,
            config=cfg,
            device=self.device,
        )

        return self.tokenizer.decode(new_token_ids).strip()

    def chat(
        self,
        query: str,
        state: "ConversationState",  # noqa: F821 — imported below to avoid circularity
        top_k_corpus: int = 5,
        gen_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a response and record the turn in ``state``.

        Unlike ``respond()``, this method maintains conversational continuity:
        the full turn history stored in ``state`` is injected into every
        prompt, so the model can reference prior exchanges.

        Pipeline:
        1. Query the corpus (if attached) for ``top_k_corpus`` results.
        2. Encode the corpus results into context token ids.
        3. Call ``state.build_prompt_ids()`` to assemble the multi-turn prompt,
           fitting history and context within the model's sequence length.
        4. Run autoregressive generation.
        5. Decode the response and add the turn to ``state``.

        Args:
            query: Plain-text user query.
            state: ``ConversationState`` for the current session.  Modified
                in-place — the new turn is appended after generation.
            top_k_corpus: Number of corpus results to retrieve.  Ignored when
                no corpus is attached.
            gen_config: Per-call generation config override.

        Returns:
            The generated response as a plain-text string.
        """
        from grimoire_ai.state.conversation import ConversationState  # local import avoids circular dep

        cfg = gen_config if gen_config is not None else self.gen_config

        # Retrieve corpus context and encode it to token ids.
        context_ids: list[int] = []
        if self.corpus is not None:
            results = self.corpus.query(query, top_k=top_k_corpus)
            context_ids = self.prompt_builder._encode_context(results)

        prompt_ids = state.build_prompt_ids(
            query=query,
            tokenizer=self.tokenizer,
            context_ids=context_ids or None,
            max_seq_len=self.model.config.max_seq_len,
        )

        new_token_ids = generate(
            model=self.model,
            prompt_ids=prompt_ids,
            config=cfg,
            device=self.device,
        )

        response = self.tokenizer.decode(new_token_ids).strip()
        state.add_turn(query, response)
        return response
