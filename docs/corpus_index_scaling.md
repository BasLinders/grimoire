# Corpus index memory scaling

Not started. Scoped 2026-08-16 after live-debugging a stuck
`scripts/evaluate.py --corpus-dir data/corpus/saga/` run; parked here for
whoever picks it up next rather than solved inline.

## The problem

`CorpusIndex` (`grimoire_ai/corpus/index.py`) holds its entire lexical
index — `_store` and `_word_postings` — as plain Python `dict`/`set`
structures, in-process, with no ceiling and no ability to spill to disk.
`GrimoireCorpus.query()` needs the whole index resident to score Jaccard
candidates, so nothing about the current design can discard data between
files: the peak memory footprint is the *entire* corpus, always, no matter
how the ingestion loop is chunked.

**Confirmed by direct reproduction, not just code inspection.** Indexing
`data/corpus/saga/` (1469 `.txt` files) via `evaluate.py --corpus-dir`
grew to ~10 GB resident by file 600/1469, on a machine with 15.7 GB total
RAM and only 0.7 GB free at the time. That's severe OS-level paging:
effective CPU utilization was ~20% (930s of CPU time over ~76 minutes of
wall clock) — the process was alive and doing real work, not deadlocked,
but spending ~80% of wall-clock time blocked on page faults/disk I/O
instead of computing. From the outside (PID/CPU-time inspection alone)
this is indistinguishable from a genuine hang, which is what triggered
this investigation in the first place.

Per-entry cost breakdown, for context on where the memory actually goes:
every unique 4-gram gets one `IndexEntry` dataclass (source label,
next-token string, and up to a 200-char excerpt — `_EXCERPT_WINDOW` in
`grimoire_ai/corpus/corpus.py`) in `_store`, plus a reference in each of
up to 4 `_word_postings[word]` sets (one per word in the n-gram). Python's
per-object overhead on dataclasses/sets/dicts multiplies this fast; a
corpus with millions of unique 4-grams (plausible once generated content
like EntiGraph output is included) adds up to multiple GB in ways that
don't show up as a bug in code review — the algorithm is correct, just
unbounded.

## Why this isn't a one-off Saga problem

`CorpusIndex`/`GrimoireCorpus` are fully agent-agnostic — no D&D-specific
logic anywhere (the same finding came up scoping the token-F1 quiz-eval
fix earlier this session: this eval infrastructure is shared, not
Saga-specific). `AgentRegistry`/`AgentRouter` already support more than
one agent, and per-agent corpora are expected to keep growing as more
agents are added (confirmed by the user, 2026-08-16) — meaning whatever
ships here needs to hold up against corpora that are bigger than today's
Saga corpus and not D&D-shaped, not just patch this one instance.

## Immediate mitigation (already shipped, separate from this doc)

`scripts/evaluate.py --corpus-limit N` samples down to N files (fixed
seed, reproducible) before reading/indexing, so an eval run doesn't need
the full corpus resident. This is a workaround scoped to *evaluation*
specifically — eval only needs *some* retrieval-grounding context in the
quiz prompt, not full corpus coverage. It does not touch `CorpusIndex`
itself and does nothing for production chat-time retrieval, which
presumably still wants the whole per-agent corpus indexed.

## Options for the real fix

1. **Shrink per-entry memory footprint.** Drop or shrink the excerpt
   window in `IndexEntry`; reconsider whether `_word_postings` needs a
   live Python `set` per word. Low effort, but doesn't remove the
   ceiling — just moves the failure point further out. A corpus that's
   large enough will still exhaust RAM eventually.

2. **Cap corpus size per agent at index-build time, in production too.**
   Same idea as `--corpus-limit`, applied where chat-time retrieval
   builds its index, not just eval. This is more a product decision than
   an engineering one — what's an acceptable retrieval-coverage tradeoff
   per agent, and does that answer differ by agent?

3. **Move `CorpusIndex` off in-RAM Python dict/set onto a disk-backed or
   memory-mapped structure.** The actual fix if corpus growth continues
   as expected. Candidates: SQLite (FTS5 for the inverted-index/posting-
   list side) or a purpose-built on-disk inverted index. Index size then
   scales with disk, not RAM, and multiple agents' corpora stop stacking
   their memory footprints when loaded together in the same process.
   Two things worth deciding as part of scoping this properly, not after:
   - **Rebuild-every-run vs. build-once-and-reuse.** The current design
     rebuilds the whole index from scratch on every run. A disk-backed
     index makes persistent, incrementally-updated storage a much more
     natural fit than the current model — worth designing for that
     directly rather than just adding persistence to the existing
     rebuild-every-time flow.
   - **Preserving the word-sharing narrowing `candidates_for_words`
     already does.** `CorpusIndex.candidates_for_words` (index.py) scores
     only multi-tokens sharing a word with the query, instead of scanning
     every entry — an on-disk version needs to keep that narrowing (e.g.
     via an indexed lookup), not regress to a full table scan per query.

## Recommendation

Option 3 is the real fix given confirmed continued corpus growth across
agents; options 1–2 are reasonable stopgaps that buy time without
removing the structural ceiling. Scoping option 3 to an implementable
plan (on-disk data model, migration path off the current rebuild-every-
run model, and validation that query narrowing still holds) is a
task-sized unit of work on its own — deliberately not started here.
