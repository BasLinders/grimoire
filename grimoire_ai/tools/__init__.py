"""Math tool — safe arithmetic evaluation for Grimoire.

Provides ``MathTool``, which detects arithmetic expressions in user queries
and evaluates them without calling ``eval()`` or spawning a shell.  The
evaluator is a pure ``ast``-based visitor that only allows numeric literals,
the six arithmetic operators, parentheses, and a small whitelist of math
functions.

Query-side use
--------------
Attach a ``MathTool`` to an ``InferenceEngine`` to pre-compute any arithmetic
detected in the user's query and inject the result as corpus context before
generation.  No model fine-tuning is required for this path.

Response-side use (post fine-tuning)
-------------------------------------
After the model is fine-tuned on ``scripts/finetune_data/tool_call_examples.jsonl``,
it learns to emit ``<TOOL:python>expression</TOOL>`` tags in its response.
``MathTool.process_response()`` finds those tags, evaluates each expression,
and substitutes the result in-place — turning
``<TOOL:python>1234*56</TOOL>`` into ``1234 * 56 = 69104``.
"""
