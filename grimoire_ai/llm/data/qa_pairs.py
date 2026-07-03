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

# Sentence-level "fluff" markers: forum-answer conventions (hedging, quoting
# other users, addressing the asker directly, rhetorical bounce-back
# questions) that carry no factual content but, left in, teach the generator
# to imitate discussion-board voice instead of stating the answer. Checked
# against a single sentence/line unit at a time so the surrounding factual
# sentences in the same answer are unaffected.
_FLUFF_SENTENCE_RE = re.compile(
    r"""
    \b(
        i\s+(think|believe|guess) |
        in\s+my\s+opinion |
        as\s+far\s+as\s+i\s+know |
        correct\s+me\s+if | i'?m\s+not\s+sure |
        edit\s*: | update\s*: | related\s*: | see\s+also | possible\s+duplicate |
        welcome\s+to | thanks?\s+for |
        [a-z][a-z0-9_.-]{2,20}\s+(mentioned|said|wrote|noted|pointed\s+out|answered) |
        as\s+\S+\s+(mentioned|noted|said) |
        \b(the|this|your)\s+question\b | you\s+(asked|mean)\b | \bOP\b
    )
    """,
    re.I | re.X,
)


def _is_fluff_sentence(unit: str) -> bool:
    """Return True for a sentence/line that is forum framing, not fact.

    Covers two patterns: sentences matching a hedge/quote/meta-reference
    phrase anywhere in the text, and sentences that are themselves a
    rhetorical question directed at the reader rather than a statement (the
    actual answer to such a question is virtually always a *separate*
    sentence, so dropping the question loses no signal).
    """
    if _FLUFF_SENTENCE_RE.search(unit):
        return True
    return unit.strip().endswith("?")


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


def _split_units(text: str) -> list[str]:
    """Split text into line-then-sentence units (see ``_take_sentences_within_budget``)."""
    units: list[str] = []
    for line in _LINE_SPLIT.split(text.strip()):
        line = line.strip()
        if not line:
            continue
        units.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    return units


def _pack_units(units: list[str], max_chars: int) -> str:
    """Greedily pack ``units`` up to ``max_chars``, joined with spaces.

    Always keeps the first unit even if it alone exceeds ``max_chars`` --
    truncating mid-unit would risk cutting off the actual answer rather
    than just trailing detail. Callers that pre-filter ``units`` (e.g. to
    drop fluff sentences) get "" for free when filtering removes
    everything, since an empty list short-circuits below.

    Args:
        units: Ordered candidate units (already filtered, if desired).
        max_chars: Target maximum length in characters.

    Returns:
        The packed slice, stripped. Empty only if ``units`` is empty.
    """
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
    return _pack_units(_split_units(text), max_chars)


def _take_clean_sentences_within_budget(text: str, max_chars: int) -> str:
    """Like ``_take_sentences_within_budget``, but skips forum-voice fluff.

    Drops hedges, quotes of other users, meta-references to "the question",
    and rhetorical questions directed at the reader (see
    ``_is_fluff_sentence``) before packing -- these are the sentences most
    real StackExchange answers open with, so packing the *raw* leading
    slice (as ``_take_sentences_within_budget`` does) systematically front-
    loads discussion-board framing ahead of the actual fact. Like the raw
    packer, the first surviving unit is always kept even if it alone
    exceeds ``max_chars``. Only returns "" if filtering removes every
    unit -- an answer that is fluff from end to end, which the caller
    treats the same as "too short to use".

    Args:
        text: Source text (typically a full community answer).
        max_chars: Target maximum length in characters.

    Returns:
        The packed, fluff-filtered slice, stripped. Empty only if every
        unit was flagged as fluff.
    """
    clean_units = [u for u in _split_units(text) if not _is_fluff_sentence(u)]
    return _pack_units(clean_units, max_chars)


def qa_pair_to_finetune_example(
    pair: QAPair,
    context_max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
    answer_max_chars: int = _DEFAULT_ANSWER_MAX_CHARS,
    context_pair: Optional["QAPair"] = None,
) -> Optional[FinetuneExample]:
    """Convert one ``QAPair`` into a ``FinetuneExample``, or ``None``.

    ``context`` is a raw leading slice of ``context_pair.answer`` (defaults
    to ``pair.answer`` itself when no ``context_pair`` is given) --
    deliberately unfiltered, since it stands in for what a retriever would
    actually hand the model (including whatever hedging or framing the real
    passage opens with). ``assistant`` is a *fluff-filtered* slice of
    ``pair.answer`` (see ``_take_clean_sentences_within_budget``):
    forum-voice hedges, quotes of other users, and rhetorical questions are
    dropped before packing, so the model is taught to state the fact the
    community answer contains, not to reproduce discussion-board framing
    verbatim.

    Passing a *different* answer to the same question as ``context_pair``
    (see ``qa_pairs_to_finetune_examples``) breaks the context/assistant
    overlap that the default same-source behaviour otherwise has -- without
    it, ``context`` is always a superset-prefix of ``assistant``, which in
    practice taught the generator to echo/loop through whatever passage it's
    given at inference instead of extracting the answer from it.

    Args:
        pair: The question/answer pair supplying ``user`` and ``assistant``.
        context_max_chars: Budget for the ``context`` slice.
        answer_max_chars: Budget for the ``assistant`` slice. Should not
            exceed ``context_max_chars``.
        context_pair: Optional different answer to the same question, used
            as the source for ``context`` instead of ``pair`` itself.

    Returns:
        ``None`` when the answer is too short (or too fluff-heavy) to form
        a meaningful pair -- there is nothing useful to extract twice.
    """
    context_source = context_pair if context_pair is not None else pair
    context = _take_sentences_within_budget(context_source.answer, context_max_chars)
    assistant = _take_clean_sentences_within_budget(pair.answer, answer_max_chars)
    if len(context) < 20 or len(assistant) < 10:
        return None
    return FinetuneExample(user=pair.question, assistant=assistant, context=context)


def _pick_context_pair(others: list[QAPair]) -> Optional[QAPair]:
    """Pick the best *different* answer to use as ``context`` for a pair.

    Prefers the asker's accepted answer among the others; falls back to the
    highest-scored one, then to whichever came first -- the same "best
    answer wins" heuristic ``load_qa_pairs``' ``min_score``/``accepted_only``
    filtering already uses elsewhere in this module.

    Args:
        others: The other answers to the same question (``pair`` excluded).

    Returns:
        The chosen ``QAPair``, or ``None`` if ``others`` is empty.
    """
    if not others:
        return None
    accepted = [o for o in others if o.accepted]
    if accepted:
        return accepted[0]
    scored = [o for o in others if o.answer_score is not None]
    if scored:
        return max(scored, key=lambda o: o.answer_score)
    return others[0]


def qa_pairs_to_finetune_examples(
    pairs: Iterable[QAPair],
    context_max_chars: int = _DEFAULT_CONTEXT_MAX_CHARS,
    answer_max_chars: int = _DEFAULT_ANSWER_MAX_CHARS,
) -> list[FinetuneExample]:
    """Convert many ``QAPair``s, dropping any too short to split meaningfully.

    Groups pairs by question first. For a question with multiple community
    answers, each example's ``context`` is drawn from a *different* answer
    than its ``assistant`` (see ``_pick_context_pair``), so the model
    practices extracting the fact from an independent passage instead of
    echoing a near-duplicate of its own target. Questions with only one
    answer fall back to the same-source behaviour (there is no independent
    text to draw ``context`` from).

    Args:
        pairs: Parsed question/answer pairs, e.g. from ``load_qa_pairs``.
        context_max_chars: Forwarded to ``qa_pair_to_finetune_example``.
        answer_max_chars: Forwarded to ``qa_pair_to_finetune_example``.

    Returns:
        A list of ``FinetuneExample``, shorter than ``pairs`` by however many
        were dropped as too short.
    """
    by_question: dict[str, list[QAPair]] = {}
    for p in pairs:
        by_question.setdefault(p.question, []).append(p)

    examples: list[FinetuneExample] = []
    for group in by_question.values():
        for i, pair in enumerate(group):
            others = group[:i] + group[i + 1 :]
            context_pair = _pick_context_pair(others)
            ex = qa_pair_to_finetune_example(
                pair, context_max_chars, answer_max_chars, context_pair=context_pair
            )
            if ex is not None:
                examples.append(ex)
    return examples
