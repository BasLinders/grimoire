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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

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
