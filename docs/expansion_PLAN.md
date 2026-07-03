# Corpus Expansion Plan

Context and next steps for growing Grimoire's training corpus, written up after a
session of fine-tuning experiments that converged on a single conclusion: the
model's failure modes (context-copying, question-echoing, factual hallucination)
look like data-scarcity symptoms, not architecture-too-small symptoms.

## Why this matters

- Chinchilla-optimal training is ~20 tokens/param. The current model (`small-25M`)
  has 129,343,636 tokens available (measured 2026-07-03 with the actual BPE
  tokenizer via `grimoire-preprocess`, 3.51 chars/token — not the ~4
  chars/token rule of thumb) against a ~500M-token compute-optimal target.
  That's ~26% of compute-optimal for the model *already in production*,
  before any consideration of scaling up.
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
- [x] **Project Gutenberg catalog-based bulk expansion.** Follow-up to the
      hand-curated pass above. Gutenberg's search-result *pages* explicitly
      warn against scraping them ("you'll only get your IP blocked"), so
      `scripts/scrape_gutenberg_catalog.py` instead downloads the official
      bulk catalog CSV (`pg_catalog.csv`, cached under `data/catalogs/`) and
      filters it locally by subject keyword + language — no hand-guessed IDs.
      The current keyword set matches ~3,400 candidate English texts; ~300
      downloaded so far, taking the corpus from 103 to 403 `gutenberg_*`
      files. Corpus is now 129,343,636 tokens total (measured 2026-07-03 with
      `grimoire-preprocess`), ~26% of the 500M Chinchilla-optimal target for
      `small-25M`. The catalog scraper, not more hand-curated lists, is the
      path to further volume.
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
- [x] **Generation-level diversity constraints on derived adventures** —
      satisfied by a small manually-written pilot, growing in batches (5 per
      batch through batch 6, 10 per batch from batch 7 onward — see the
      running batch log below for the current count and per-batch size;
      40 adventures across 7 batches as of the last update, 40,248 tokens,
      written directly rather than via a scripted API pipeline — see note
      below):
  - [x] Cap adventures derived per source text (one per book): Beowulf
        (#16328), Prose/Poetic Edda (#4785/#23265), The Odyssey (#1727),
        Grimms'/Yellow Fairy Book (#2591/#7154), The Worm Ouroboros (#39058).
  - [x] Force structural variety explicitly: adventure types (dungeon crawl,
        political intrigue, survival, investigation, dungeon crawl again at a
        deliberately different tier/environment), level ranges spanning
        tiers 1-2 (1-4, 3-6, 5-8, 7-10), environments (mere lair, mead-hall
        court, archipelago, village, ruined fortress).
  - [x] Deduplicated against each other and against the full existing corpus
        via `scripts/dedup_corpus.py` — 0 near-duplicate pairs at a loose
        threshold (0.2-0.25, smaller shingle size than the Gutenberg check,
        since these adventures deliberately reuse some monster vocabulary).
  - [x] Verified all mechanical content (monster names + CR + XP) against
        `srd_monsters.txt` by reading the actual stat blocks before citing
        them (Sea Hag CR2/450XP, Green Hag CR3/700XP, Night Hag CR5/1800XP,
        Troll CR5/1800XP, Wight CR3/700XP, Ghoul CR1/200XP, Ogre CR2/450XP,
        Kobold CR1/8/25XP, Griffon CR2/450XP, Skeleton/Zombie CR1/4/50XP) —
        no invented stat blocks; adventures reference monsters by name/CR
        rather than reprinting full stat blocks, matching real published-
        adventure convention.
- [x] **Provenance tagging** — output written to `data/corpus/saga_derived/`,
      a directory separate from the raw-scraped `data/corpus/saga/`, with a
      standard header block per file (source title + Gutenberg ID, adventure
      type, level range, environment, explicit "Derived-synthetic" marker) so
      future ingestion scripts can filter/weight this content independently.
      Not yet added to the `saga` agent's `corpus_dirs` in `agents.json` —
      that's a separate decision once there's more than a 5-file pilot batch.

**How these are generated:** not via the scripted `generate_derived_adventures.py`
+ Anthropic API pipeline originally envisioned above — the user opted to avoid
the additional API billing (separate from their existing Claude subscription)
and instead has Claude write these directly in-session, in batches (5 per
batch through batch 6, increased to 10 per batch from batch 7 onward at the
user's request), each verified against corpus ground-truth before inclusion
the same way as the pilot batch. This is a deliberately slow, steady growth
path rather than a corpus-scale solution — revisit the scripted API pipeline
if/when more volume is wanted and the user is ready to spend on it.

## Derived-adventure batch log

Running record of every batch, so a future session can pick up the count and
avoid reusing source books. Token counts measured with the real BPE tokenizer
(`data/tokenizer/bpe.json`), not a chars/4 estimate.

| Batch | Files | Source books used (Gutenberg ID) | Types / level ranges / environments | Tokens (batch) | Tokens (cumulative) |
|---|---|---|---|---|---|
| 1 | `adventure_001`-`005` | Beowulf (#16328), Prose Edda (#4785) + Poetic Edda (#23265), The Odyssey (#1727), Grimms' Fairy Tales (#2591) + Yellow Fairy Book (#7154), The Worm Ouroboros (#39058) | dungeon crawl (1-4, mere lair), political intrigue (5-8, mead-hall court), survival (3-6, archipelago), investigation (1-3, village), dungeon crawl (7-10, ruined fortress) | 5,861 | 5,861 |
| 2 | `adventure_006`-`010` | Le Morte d'Arthur (#1251/#1252), The Mabinogion (#4486), One Thousand and One Nights (#128) + Arabian Nights Entertainments (#558), Myths and Legends of Ancient Greece and Rome (#2680), The Nibelungenlied (#557) | heist (5-8, tournament keep), survival (3-6, fey borderland), investigation (4-6, desert trade city), dungeon crawl (5-8, buried labyrinth), heist (11-14, dragon hoard) | 5,066 | 10,927 |
| 3 | `adventure_011`-`015` | The Iliad (#6130), Metamorphoses (#348), The Divine Comedy / Inferno (#20), The Witch-cult in Western Europe (#2021), Peter Pan (#16) | political intrigue (6-9, siege camp), investigation (4-6, riverside village), dungeon crawl (13-16, planar rift monastery), political intrigue/investigation (5-7, rural county), survival (3-5, drifting island) | 5,314 | 16,241 |
| 4 | `adventure_016`-`020` | The Aeneid (#228), Hero Tales and Legends of the Rhine (#7882), The Mythology of the British Islands (#9880), The Golden Bough (#22696), The Lesser Key of Solomon (#46849) | survival (3-5, coastline/isles), dungeon crawl (6-8, sunken forge-hall), investigation (2-4, moorland barrows), political intrigue (4-6, harvest-kingdom court), heist (8-9, sealed tower-library) | 4,776 | 21,017 |
| 5 | `adventure_021`-`025` | The Book of Wonder (#7132), Phantastes (#325), Heroic Romances of Ireland Vol 1 (#10329), The Mahabharata Book 1 (#3972), Irish Fairy Tales (#21451) | investigation (3-5, dream-border city), survival (3-5, shifting enchanted forest), political intrigue (6-8, Irish high-king's court), heist (13-16, rival kingdom's treasury), dungeon crawl (2-4, fairy rath) | 4,916 | 25,933 |
| 6 | `adventure_026`-`030` | The Ramayana (#7864), The Princess and the Goblin (#947), The Gods of Pegana (#8432), The Wood Beyond the World (#7143), Hawaiian Folk Tales (#17034) | survival (5-7, exile forest), dungeon crawl (2-4, goblin tunnels), political intrigue (6-8, temple-city), heist (6-8, witch-queen's manor), investigation (5-7, volcanic islands) | 4,947 | 30,880 |
| 7 (batch size increased to 10) | `adventure_031`-`040` | The Once and Future King excerpt (#831), The Canterbury Tales (#583), Theogony and Works and Days (#22381), Paradise Lost (#8789), The Faerie Queene Book 1 (#3071), Household Tales by Brothers Grimm (#5314), Fairy Tales by Hans Christian Andersen (#1597), Treasure Island (#113), Frankenstein (#84), Journey to the Centre of the Earth (#1268) | political intrigue (5-7, royal court), investigation (3-5, pilgrimage road), dungeon crawl (9-11, primordial vault), heist (17-20, fallen-celestial ruin), political intrigue (6-8, illusion-shrouded holding), survival (2-4, unnatural winter), investigation (2-4, fishing town), heist (6-8, treasure island), investigation (7-9, alchemist's tower), dungeon crawl (8-10, deep caverns) | 9,368 | 40,248 |

**Source books used so far (do not reuse without deliberate reason):** Beowulf
#16328, Prose Edda #4785, Poetic Edda #23265, The Odyssey #1727, Grimms' Fairy
Tales #2591, Yellow Fairy Book #7154, The Worm Ouroboros #39058, Le Morte
d'Arthur #1251/#1252, The Mabinogion #4486, One Thousand and One Nights #128,
Arabian Nights Entertainments #558, Myths and Legends of Ancient Greece and
Rome #2680, The Nibelungenlied #557, The Iliad #6130, Metamorphoses #348,
The Divine Comedy (Inferno) #20, The Witch-cult in Western Europe #2021,
Peter Pan #16, The Aeneid #228, Hero Tales and Legends of the Rhine #7882,
The Mythology of the British Islands #9880, The Golden Bough #22696, The
Lesser Key of Solomon #46849, The Book of Wonder #7132, Phantastes #325,
Heroic Romances of Ireland Vol 1 #10329, The Mahabharata Book 1 #3972,
Irish Fairy Tales #21451, The Ramayana #7864, The Princess and the Goblin
#947, The Gods of Pegana #8432, The Wood Beyond the World #7143, Hawaiian
Folk Tales #17034, The Once and Future King excerpt #831, The Canterbury
Tales #583, Theogony and Works and Days #22381, Paradise Lost #8789, The
Faerie Queene Book 1 #3071, Household Tales by Brothers Grimm #5314, Fairy
Tales by Hans Christian Andersen #1597, Treasure Island #113, Frankenstein
#84, Journey to the Centre of the Earth #1268.

**Ground-truth monster CR/XP verified so far (reusable across future
batches without re-checking):** Kobold CR1/8 (25 XP), Merfolk CR1/8 (25 XP),
Ghoul CR1 (200 XP), Harpy CR1 (200 XP), Dryad CR1 (200 XP), Specter CR1
(200 XP), Skeleton/Zombie/Goblin CR1/4 (50 XP each), Hobgoblin CR1/2
(100 XP), Sea Hag CR2 (450 XP), Ogre CR2 (450 XP), Griffon CR2 (450 XP),
Bandit Captain CR2 (450 XP, open5e_monsters.txt), Green Hag CR3 (700 XP),
Wight CR3 (700 XP), Minotaur CR3 (700 XP), Manticore CR3 (700 XP), Werewolf
CR3 (700 XP), Basilisk CR3 (700 XP), Doppelganger CR3 (700 XP), Lamia CR4
(1,100 XP), Ettin CR4 (1,100 XP), Troll CR5 (1,800 XP), Night Hag CR5
(1,800 XP), Hill Giant CR5 (1,800 XP), Fire Elemental CR5 (1,800 XP,
Salamander is the same CR/XP if a fire-themed alternative is wanted),
Otyugh CR5 (1,800 XP), Flesh Golem CR5 (1,800 XP), Medusa CR6 (2,300 XP),
Stone Giant CR7 (2,900 XP), Chain Devil CR8 (3,900 XP), Young Green Dragon
CR8 (3,900 XP), Erinyes CR12 (8,400 XP), Rakshasa CR13 (10,000 XP), Pit
Fiend CR20 (25,000 XP, deliberately overwhelming — used only as an
avoid-not-fight guardian, not a straight combat encounter), Djinni
CR11 (7,200 XP, referenced for lore only — not intended as a fought
encounter at the level ranges used so far). Source:
`data/corpus/saga/srd_monsters.txt` unless noted otherwise. Verify a
monster's CR/XP here before reusing it, and add any newly-verified creature
to this list.

## Source-based sample weighting

The mechanism (`--weight-pattern` on `grimoire-preprocess` → per-document
weight sidecars → `scripts/build_source_weights.py` → `sample_weights.npy` →
`Trainer`'s `sample_weights_path`) existed before this decision but had never
been given real values. Byte-share breakdown of `data/corpus/saga/` (403
`gutenberg_*` files, 1,469 total) showed D&D-specific content is already the
majority by volume (~61%) — the dilution risk is prospective, from the
catalog scraper's ~3,400-book candidate pool, not already acute. One finding
shaped the scheme: ~110 official WotC rulebooks/adventures/oneshots (7.2% of
corpus bytes) have no shared filename prefix, so they can't be targeted by a
specific glob — handled with a trailing `*:WEIGHT` catch-all rule instead of
a rename, since `--weight-pattern` rules are matched in order with first-
match-wins (`grimoire_ai/llm/data/preprocessing.py`'s `_resolve_weight`).

- [x] **Decide weight values.** Down-weight raw bulk literature; leave
      already-D&D-specific categories at baseline; upweight the highest-
      authority structured/official content via the catch-all:
      ```
      gutenberg_*:0.5
      wp_fantasy_*:0.5
      wp_math_*:1
      wp_dnd_*:1
      rpg_se_*:1
      fr_wiki_*:1
      dragon_*:1
      dnd_*:1
      synth_*:1
      *:1.75
      ```
      The catch-all lands on `srd_*`, `5etools_*`, `open5e_*`, and the
      untaggable official-book long tail. Effective (weight × byte-share,
      renormalized) result: `gutenberg_*` drops from 37.8% raw share to
      ~21.4%; the official-book long tail nearly doubles from 7.2% to ~14.2%;
      `open5e_*` goes from 2.0% to ~4.1%. `wp_math_*` is left untouched —
      separate data-science-assistant scope, not diluting anything.
- [x] **Tag the corpus.** Re-ran `grimoire-preprocess` with the rules above
      against the full corpus (2026-07-03) — 129,343,636 tokens (unchanged;
      weighting only adds sidecars, doesn't retokenize differently), wrote
      `corpus.bin.doc_end_offsets.npy` / `corpus.bin.doc_weights.npy` for all
      1,469 documents.
- [ ] **Build `sample_weights.npy` and use it in a real training run.** Use
      the Pre-train tab's "Build sample weights from tags" button (not a raw
      `build_source_weights.py` CLI call — it needs to match whatever
      `seq_len`/`stride`/`val_split` the actual run uses, which the button
      handles automatically and a hand-rolled call does not), then set
      `sample_weights_path` before starting pre-training. Still open: no
      weighted training run has happened yet, so there's no signal on
      whether these specific multipliers help — revisit with finer-grained
      tiers (e.g. splitting `rpg_se_*` from official books) only after a
      baseline weighted-vs-unweighted comparison, not before.

## Open decisions for next session

- [ ] Once corpus token count is known post-expansion, decide target model
      size using the 20:1 ratio rather than jumping straight to `medium-85M`.
- [ ] Re-measure the MiniLM retrieval baseline at
      `--quiz-repetition-penalty 1.3` (currently only measured at 1.0 — not a
      fair comparison against everything else in this session's results).
- [ ] Phase 5 (pre-existing gap, unrelated to this plan): wire semantic/LoRA
      retrieval into the live UI — it currently only supports lexical search
      via `corpus_dirs`.
