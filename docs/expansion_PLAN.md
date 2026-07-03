# Corpus Expansion Plan

Context and next steps for growing Grimoire's training corpus, written up after a
session of fine-tuning experiments that converged on a single conclusion: the
model's failure modes (context-copying, question-echoing, factual hallucination)
look like data-scarcity symptoms, not architecture-too-small symptoms.

## Why this matters

- Chinchilla-optimal training is ~20 tokens/param. The current model (`small-25M`)
  has ~90M tokens available (measured with the actual BPE tokenizer, 3.51
  chars/token — not the ~4 chars/token rule of thumb) against a ~500M-token
  compute-optimal target. That's 18% of compute-optimal for the model *already
  in production*, before any consideration of scaling up.
- `medium-85M` would need ~1.7B tokens to hit the same ratio; the project's own
  preset comments already gate that preset behind "~100M tokens," a threshold
  the corpus doesn't yet clear.
- Conclusion: growing the corpus is a higher-leverage, lower-risk move than
  scaling parameters right now. Param scaling should follow corpus growth, sized
  to whatever token count is actually achieved (20:1 ratio), not precede it.

## Sources ruled out or deferred

- **Other TTRPG systems (Pathfinder, GURPS, etc.):** deferred. The model
  currently can't reliably distinguish grounded context from hallucinated
  content within a *single* ruleset; mixing rules-similar-but-different systems
  (e.g. D&D 5e's flat proficiency bonus vs. Pathfinder 2e's proficiency ranks)
  would add a harder discrimination problem on top of one that isn't solved yet.
  Revisit once single-system grounding is reliable — and if revisited, use
  separate `corpus_dirs`/agents per system rather than one shared pool.
- **Bulk Claude-generated adventures (from nothing):** ruled out as a primary
  strategy. Three problems: (1) volume needed (400M+ tokens) is impractical to
  generate at usable quality, (2) rules content generated without verification
  risks baking hallucinated mechanics into training data as if they were ground
  truth — precisely the failure mode this session spent hours diagnosing, (3)
  heavy reliance on one model's output narrows corpus diversity (see: model
  collapse literature, e.g. Shumailov et al.).

## Sources to pursue

- [x] **Project Gutenberg fantasy/mythology expansion.** Ran
      `scrape_gutenberg.py` + `scrape_gutenberg_extended.py` (curated lists
      already existed in the repo, just hadn't been executed) — went from 17
      to 103 files, adding ~14.3M tokens (measured with the real BPE
      tokenizer). Corpus is now ~92.8M tokens total. 3 book IDs had no working
      plain-text mirror (Parzival #9934, Symbolic Logic #38986, An
      Introduction to Mathematics #21076) and were skipped; not investigated
      further. Fixed a Windows console `UnicodeEncodeError` in
      `scrape_gutenberg_extended.py`'s final summary print (non-cp1252 arrow
      character) — cosmetic, didn't affect the downloaded files.
- [ ] **Math/stats textbook-style sources** (Wikibooks full books, OpenStax
      CC-BY texts) as a lower-priority complement to the existing `wp_math_*`
      stub articles, for the "data-science assistant" side of the agent scope.
      Note: the extended Gutenberg scrape incidentally added 6 math/logic
      titles (Hardy, Russell, Venn, Dudeney, Abbott, Couturat) as a side
      effect — this item is still about the Wikibooks/OpenStax sources
      specifically, not covered by that.

## Derived-adventure pipeline (Gutenberg → structured D&D content)

Not raw scraping, and not unconstrained generation — using real, diverse prose
as inspiration material that Claude restructures into adventure form, with any
mechanical content (stat blocks, DCs, XP budgets) verified against the existing
corpus's ground-truth rules data (5etools bestiary/SRD) rather than invented or
recalled from model memory.

- [x] **Resolve licensing scope for source material** before scraping — public
      domain / clearly open-licensed only. No bulk scraping of
      copyright-ambiguous web fiction. (Both scrapers pull only curated,
      known-public-domain Gutenberg IDs — no change needed here.)
- [x] **Source-level dedup on the raw scrape:**
  - [x] Strip Gutenberg boilerplate license header/footer (byte-identical
        across every book — the crudest possible duplication if left in).
        Already handled by both scrapers' `_clean()` regex; spot-checked a
        scraped file's head/tail, boilerplate is gone.
  - [x] Avoid pulling multiple translations/editions of the same underlying
        work unless deliberately wanted for stylistic variety. Checked the
        curated lists — no duplicate editions, only one incidental double
        listing (Yellow Fairy Book, same Gutenberg ID in both scripts, so it
        just downloads once).
  - [x] Run near-duplicate detection (MinHash/shingling) across the new scrape
        *and* against the existing corpus before merging. Wrote
        `scripts/dedup_corpus.py` (word-shingle MinHash, no new heavy
        dependency) and validated it against synthetic exact/partial-copy
        cases before trusting the result. Ran it: 0 pairs found at
        `--threshold 0.3` across all 103 Gutenberg files vs. the full
        1169-file corpus — no near-duplicates.
- [ ] **Generation-level diversity constraints on derived adventures:**
  - [ ] Cap adventures derived per source text (e.g. one per book) so the
        output pool isn't a rehash of a small number of idea seeds.
  - [ ] Force structural variety explicitly per generation — adventure type
        (dungeon crawl / political intrigue / survival / investigation),
        level range, environment — rather than leaving it to chance.
  - [ ] Deduplicate generated adventures against *each other*, not just against
        source material.
  - [ ] Verify all mechanical content (stat blocks, DCs, damage, XP budgets)
        against existing verified corpus data before inclusion.
- [ ] **Provenance tagging:** keep scraped-raw and derived-synthetic content in
      separate subdirectories/naming conventions, so ingestion scripts (e.g.
      `build_finetune_data_from_qa.py` and future equivalents) can weight,
      filter, or exclude synthetic content independently.

## Open decisions for next session

- [ ] Once corpus token count is known post-expansion, decide target model
      size using the 20:1 ratio rather than jumping straight to `medium-85M`.
- [ ] Re-measure the MiniLM retrieval baseline at
      `--quiz-repetition-penalty 1.3` (currently only measured at 1.0 — not a
      fair comparison against everything else in this session's results).
- [ ] Phase 5 (pre-existing gap, unrelated to this plan): wire semantic/LoRA
      retrieval into the live UI — it currently only supports lexical search
      via `corpus_dirs`.
