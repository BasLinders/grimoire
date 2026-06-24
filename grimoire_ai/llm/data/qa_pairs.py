"""Parse StackExchange Q&A corpus files into (query, passage) training pairs.

``scripts/scrape_stackexchange_rpg.py`` writes the rpg.stackexchange.com data
dump as ``.txt`` files in a fixed block format (one question with its answers
per block, blocks separated by ``---``):

    # <question title>
    Score: <N>

    <question body…>

    ## Answer (accepted)  (score: <N>)

    <answer body…>

    ## Answer  (score: <N>)

    <answer body…>

    ---

These are genuine question→answer relevance pairs — exactly the supervised
signal a retrieval embedding model needs (and the signal self-supervised
SimCSE-style training never provides). This module extracts them so embedding
training can optimise the *actual* retrieval task ("does this passage answer
this query") rather than a same-passage proxy.

The parser is deliberately tolerant: it keys off the ``#``/``##``/``---``
structural markers and tolerates missing scores, missing bodies, and blocks
with no answers (which are simply skipped) rather than assuming every block is
well-formed.

Instruction-tuning conversion
------------------------------
The same pairs double as instruction-tuning data for the *generator*: a
StackExchange question maps directly onto ``ConversationDataset``'s
``"user"`` field. The mismatch is the answer — community answers run
discursive and long (median ~1200 characters across this corpus, some over
20,000), where the model needs to learn short, direct replies (the existing
hand-authored ``scripts/finetune_data/saga_v1.jsonl`` averages ~320
characters). ``qa_pairs_to_finetune_examples`` bridges this by extracting two
different-length slices of the *same* answer: a longer one as ``"context"``
(simulating what a retriever would hand the model) and a shorter one as
``"assistant"`` (the concise reply to learn to produce from it). There's no
separately-authored "ideal short answer" to draw on, so this derives one from
the community answer itself rather than inventing one — a known
simplification, not a substitute for curated examples.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

# A question block opens with a level-1 ATX heading: "# <title>". We match at
# line start and capture the title text. (Answers use "## ", which this does
# not match because of the required single "#" followed by a space and a
# non-"#" character is not enforced here — we disambiguate by checking the
# answer pattern first when scanning.)
_QUESTION_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$")

# An answer header: "## Answer (accepted)  (score: N)" or "## Answer  (score: N)".
# The "(accepted)" group is optional; the score may be negative.
_ANSWER_RE = re.compile(
    r"^##\s+Answer\s*(?P<accepted>\(accepted\))?\s*\(score:\s*(?P<score>-?\d+)\)\s*$"
)

# The per-question "Score: N" line that follows the title.
_QSCORE_RE = re.compile(r"^Score:\s*(?P<score>-?\d+)\s*$")

# Block separator.
_SEPARATOR = "---"


@dataclass(frozen=True)
class QAPair:
    """A single question→answer relevance pair extracted from the dump.

    Attributes:
        question: The question title text (the natural search-query form).
        answer: The answer body text (the passage that should rank highly for
            ``question``).
        answer_score: The community score of the answer. Higher is better;
            used for quality filtering. ``None`` when the header carried no
            parseable score.
        accepted: Whether this was the question asker's accepted answer.
    """

    question: str
    answer: str
    answer_score: Optional[int]
    accepted: bool


def _flush_answer(
    title: str,
    score: Optional[int],
    accepted: bool,
    body_lines: list[str],
) -> Optional[QAPair]:
    """Build a ``QAPair`` from accumulated answer-body lines, or ``None``.

    Returns ``None`` when either the question title or the answer body is
    empty after stripping — a pair with no text on either side carries no
    training signal.
    """
    answer = "\n".join(body_lines).strip()
    question = title.strip()
    if not question or not answer:
        return None
    return QAPair(question=question, answer=answer, answer_score=score, accepted=accepted)


def parse_qa_text(text: str) -> Iterator[QAPair]:
    """Yield every ``QAPair`` found in one scraped Q&A document.

    Walks the document line by line, tracking the current question title and
    the answer currently being accumulated. A new ``## Answer`` header, a new
    ``# `` question, or a ``---`` separator each flush the answer in progress.

    Args:
        text: The full text of one ``rpg_se_*.txt`` file.

    Yields:
        One ``QAPair`` per (question, answer) pairing, in document order. A
        question with three answers yields three pairs sharing the same
        ``question`` text.
    """
    title: Optional[str] = None
    in_answer = False
    ans_score: Optional[int] = None
    ans_accepted = False
    ans_lines: list[str] = []

    def flush() -> Optional[QAPair]:
        nonlocal in_answer, ans_lines
        if not in_answer or title is None:
            in_answer = False
            ans_lines = []
            return None
        pair = _flush_answer(title, ans_score, ans_accepted, ans_lines)
        in_answer = False
        ans_lines = []
        return pair

    for line in text.splitlines():
        answer_match = _ANSWER_RE.match(line)
        if answer_match:
            pair = flush()
            if pair is not None:
                yield pair
            in_answer = True
            ans_accepted = answer_match.group("accepted") is not None
            score_str = answer_match.group("score")
            ans_score = int(score_str) if score_str is not None else None
            continue

        # A question heading also closes any answer in progress. Check the
        # answer pattern first (above) so "## Answer" is never mistaken for a
        # question — "##" does not match _QUESTION_RE's single-"#" anchor
        # anyway, but ordering keeps the intent explicit.
        question_match = _QUESTION_RE.match(line)
        if question_match and not line.startswith("##"):
            pair = flush()
            if pair is not None:
                yield pair
            title = question_match.group("title")
            continue

        if line.strip() == _SEPARATOR:
            pair = flush()
            if pair is not None:
                yield pair
            continue

        # The per-question "Score:" line is metadata, not answer body.
        if not in_answer and _QSCORE_RE.match(line):
            continue

        if in_answer:
            ans_lines.append(line)

    # Flush a trailing answer with no closing separator.
    pair = flush()
    if pair is not None:
        yield pair


def load_qa_pairs(
    corpus_dir: str | Path,
    pattern: str = "rpg_se_*.txt",
    min_score: int = 1,
    accepted_only: bool = False,
) -> list[QAPair]:
    """Load and filter Q&A pairs from a directory of scraped files.

    Args:
        corpus_dir: Directory containing the scraped ``.txt`` files.
        pattern: Glob selecting the Q&A files (default matches the
            StackExchange scraper's ``rpg_se_*.txt`` output).
        min_score: Keep only answers whose score is at least this value.
            Answers with no parseable score are treated as score 0 and so are
            dropped whenever ``min_score > 0``. Filtering on score is the
            cheapest quality lever — low/negative-scored answers are often
            wrong or off-topic, which is noise for a relevance objective.
        accepted_only: When ``True``, keep only the asker's accepted answer
            for each question, ignoring ``min_score``.

    Returns:
        A list of ``QAPair`` in file-then-document order. Empty when no files
        match the pattern.

    Raises:
        FileNotFoundError: If ``corpus_dir`` does not exist.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    pairs: list[QAPair] = []
    for path in sorted(corpus_path.glob(pattern)):
        text = path.read_text(encoding="utf-8")
        for pair in parse_qa_text(text):
            if accepted_only:
                if not pair.accepted:
                    continue
            else:
                score = pair.answer_score if pair.answer_score is not None else 0
                if score < min_score:
                    continue
            pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Instruction-tuning conversion
# ---------------------------------------------------------------------------

# Default budgets, chosen to land near scripts/finetune_data/saga_v1.jsonl's
# existing style (assistant answers there run ~185-530 characters; context
# there is the bare source rule a sentence or two shorter than the answer).
_DEFAULT_CONTEXT_MAX_CHARS = 600
_DEFAULT_ANSWER_MAX_CHARS = 350

# Reused for both context and answer extraction below -- same sentence
# boundary rule as grimoire_ai.llm.inference.semantic.chunk_text's packing,
# kept local here since that pattern is private to that module.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Line boundary, checked before sentence-splitting: StackExchange answers
# often format lists as one item per line with little or no terminal
# punctuation (e.g. a table of abbreviations), which _SENTENCE_SPLIT alone
# would never break on -- without this, such a list is indistinguishable
# from one giant unpunctuated "sentence" and blows straight through any
# character budget.
_LINE_SPLIT = re.compile(r"\n+")


@dataclass(frozen=True)
class FinetuneExample:
    """A (question, concise answer, context) triple, field-compatible with
    ``grimoire_ai.llm.data.conversation.ConversationDataset``'s JSONL format.
    """

    user: str
    assistant: str
    context: str

    def to_json_line(self) -> str:
        """Serialize to one JSONL line (no trailing newline)."""
        return json.dumps({"user": self.user, "assistant": self.assistant, "context": self.context})


def _take_sentences_within_budget(text: str, max_chars: int) -> str:
    """Take whole sentences/lines from the start of ``text`` up to ``max_chars``.

    Splits on line breaks first, then sentences within each line, before
    packing -- a list-formatted answer (one item per line, little or no
    terminal punctuation) would otherwise look like a single giant
    "sentence" to a sentence-only splitter and blow straight through the
    budget (observed on the real corpus: list-of-abbreviations answers with
    no period for thousands of characters).

    Packs units greedily, stopping *before* one that would push the running
    length over budget. Always returns at least the first unit, even if it
    alone exceeds ``max_chars`` -- truncating mid-unit would risk cutting off
    the actual answer rather than just trailing detail.

    Args:
        text: Source text to extract a leading slice of.
        max_chars: Target maximum length in characters.

    Returns:
        The extracted slice, stripped. Empty only if ``text`` is blank.
    """
    units: list[str] = []
    for line in _LINE_SPLIT.split(text.strip()):
        line = line.strip()
        if not line:
            continue
        units.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    if not units:
        return ""
    taken = [units[0]]
    total = len(units[0])
    for unit in units[1:]:
        if total + 1 + len(unit) > max_chars:
            break
        taken.append(unit)
        total += 1 + len(unit)
    return " ".join(taken)


def qa_pair_to_finetune_example(
    pair: QAPair,
    context_max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
    answer_max_chars: int = _DEFAULT_ANSWER_MAX_CHARS,
) -> Optional[FinetuneExample]:
    """Convert one ``QAPair`` into a ``FinetuneExample``, or ``None``.

    Both ``context`` and ``assistant`` are leading slices of the *same*
    ``pair.answer`` text, extracted independently with ``answer_max_chars <=
    context_max_chars`` so the assistant target is consistent with (a prefix
    of, in practice) what the context shows -- the model is being taught to
    state the direct answer the context already contains, not to introduce
    information absent from it.

    Args:
        pair: A parsed question/answer pair.
        context_max_chars: Budget for the ``context`` slice.
        answer_max_chars: Budget for the ``assistant`` slice. Should not
            exceed ``context_max_chars``.

    Returns:
        ``None`` when the answer is too short to form a meaningful pair
        (e.g. a one-word "Yes." with no elaboration) -- there is nothing
        useful to extract twice.
    """
    context = _take_sentences_within_budget(pair.answer, context_max_chars)
    assistant = _take_sentences_within_budget(pair.answer, answer_max_chars)
    if len(context) < 20 or len(assistant) < 10:
        return None
    return FinetuneExample(user=pair.question, assistant=assistant, context=context)


def qa_pairs_to_finetune_examples(
    pairs: Iterable[QAPair],
    context_max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
    answer_max_chars: int = _DEFAULT_ANSWER_MAX_CHARS,
) -> list[FinetuneExample]:
    """Convert many ``QAPair``s, dropping any too short to split meaningfully.

    Args:
        pairs: Parsed question/answer pairs, e.g. from ``load_qa_pairs``.
        context_max_chars: Forwarded to ``qa_pair_to_finetune_example``.
        answer_max_chars: Forwarded to ``qa_pair_to_finetune_example``.

    Returns:
        A list of ``FinetuneExample``, shorter than ``pairs`` by however many
        were dropped as too short.
    """
    examples = (
        qa_pair_to_finetune_example(p, context_max_chars, answer_max_chars) for p in pairs
    )
    return [ex for ex in examples if ex is not None]
