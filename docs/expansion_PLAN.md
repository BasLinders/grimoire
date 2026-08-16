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
- [x] **Build `sample_weights.npy` and use it in a real training run.** Built
      via `scripts/build_source_weights.py` (`seq_len=1024`, `stride=512`,
      `val_split=0.0` — matching `Trainer`'s own defaults; the file assumes
      no held-out region since the run below used `val_split=0`): 252,623
      windows total, 41.6% at weight 0.5, 48.5% at weight 1.0, 9.9% at
      weight 1.75.
- [x] **Paired baseline-vs-weighted comparison, `small-25M`, 15,258 steps
      each (2026-07-03).** Same corpus, same hyperparameters, only
      `sample_weights_path` differed:
      | | Baseline | Weighted |
      |---|---|---|
      | Val loss @ step 9,156/9,156 | 2.9548 | (not recorded) |
      | Val loss @ step 10,682/10,682 | 2.9046 | (not recorded) |
      | Val loss @ step 13,734 | 2.8524 | **2.6944** |
      | Wall-clock | 24,143.5s | 24,140.7s |
      **Result: weighting helps.** 0.158 absolute / ~5.5% relative
      reduction in validation loss at the matching checkpoint, at
      effectively zero wall-clock cost. Train loss also tracked lower
      throughout (~2.97–2.99 vs ~3.08–3.12 in the same step range), and val
      loss improving *more* than train loss argues against this being
      overfitting to the reweighted mix.
- [x] **Stack Exchange markup cleanup.** Qualitative generation check on the
      `weighted` checkpoint above surfaced a literal `## Answer (score: 4)`
      fragment bleeding into output — raw StackExchange dump scaffolding
      (`Score: N`, `Tags:`, `## Answer (accepted) (score: N)`, `---`
      separators) hadn't been stripped from `rpg_se_*` files.
      `scripts/clean_stackexchange_markup.py` strips it in place across all
      217 files (originals backed up to `data/corpus/saga_se_qa_source/`
      first, since `data/` is gitignored — later renamed from
      `saga_backup_pre_se_cleanup/` once it turned out to serve an ongoing
      purpose, not just a revert point; see below). Corpus dropped from
      129,343,636 to 128,096,453 tokens
      (~1.0% — scaffolding removal, not content loss). Re-tagged with the
      same `--weight-pattern` rules and rebuilt `sample_weights.npy`.
- [x] **Found and fixed a real bug in the validation split.** Retraining
      from scratch on the cleaned+re-tagged corpus (`weighted_clean`,
      `small-25M`, 15,258 steps, 2026-07-07) gave a val loss (3.5059 @ step
      13,734) that looked like a severe regression against the earlier
      2.6944. Root cause: `_build_datasets`' `val_split` held out "the
      final N% of the corpus" by raw token position, and since files
      concatenate in alphabetically-sorted order, that tail was almost
      entirely short Wikipedia/Wikibooks stub articles — not a
      representative sample, and not comparable to a properly representative
      one. This was true of *every* prior run, not something the cleanup
      introduced. Fixed in `train.py`'s `_split_blocks`: partitions the
      corpus into 500 scattered blocks and randomly assigns ~`val_split`
      fraction to validation, so held-out data is a real cross-section of
      the corpus. `TokenizedDataset` gained a `regions` parameter
      (multiple `(start, end)` pairs) to support this.
- [x] **Per-tier val loss breakdown — the actual answer on whether weighting
      helps.** A single aggregate val-loss number can't distinguish
      "weighting made things worse" from "weighting is working exactly as
      intended, and this metric isn't the right one" — a weighted model
      is deliberately trying to minimize loss on the *weighted*
      distribution, not the corpus's natural one, so a representative
      validation set will always show it doing worse on down-weighted
      content it saw less of. Broke the `weighted_clean` checkpoint's loss
      down by tier instead:
      | Tier | Held-out val loss | Train-inclusive sample* |
      |---|---|---|
      | 0.5 (down-weighted) | 3.5822 | 3.2628 |
      | 1.0 (baseline) | 3.1202 | 3.0170 |
      | 1.75 (up-weighted) | — (none in this held-out sample) | **2.1863** |

      *sampled from the full corpus since the held-out set (only 5 scattered
      blocks) happened to miss the up-weighted tier entirely by chance — see
      the stratified-split fix below. Train-inclusion gives an optimistic
      bias, but the other two tiers only shifted 0.10–0.32 nats between
      held-out and train-inclusive measurement, so it isn't wildly inflated.

      **Result: weighting is working as designed.** Monotonic ordering
      exactly matching the intended prioritization (down-weighted > baseline
      > up-weighted), with a large 0.83-nat gap to the up-weighted tier —
      the official rulebooks/SRD/5etools/open5e content this whole effort
      is about is fit substantially better than everything else. The
      elevated aggregate number was misleading on its own.
- [x] **Stratified validation split**, so future comparisons don't depend on
      luck to cover every tier. `train.py` gained `_split_by_tier`: within
      each `--weight-pattern` tier separately (using the document-level tag
      sidecars), shuffles that tier's documents and holds out `val_split`
      fraction of *its own* tokens — guarantees every tier present in the
      corpus gets proportional validation coverage, whole documents only
      (never splits one mid-file). Exposed as:
      - `--val-stratified` on `grimoire-train` (config key
        `"val_stratified": true`) and on `scripts/build_source_weights.py`
        (must match between the two, same as `--val-split` already had to)
      - A "Stratify validation by weight tags" checkbox in the Pre-train
        tab, next to Validation split
      Requires the corpus to have `--weight-pattern` tag sidecars; errors
      clearly if they're missing rather than silently falling back.
- [x] **Qualitative generation check on `weighted_clean` (2026-07-08).** Same
      5 prompts as the earlier `baseline`-vs-`weighted` check, same sampling
      config, three-way comparison across all three checkpoints.
      - **The specific bug this cleanup targeted is confirmed fixed.** The
        `weighted` checkpoint's Fireball-prompt output contained a literal
        `## Answer (score: 4)` StackExchange markup fragment (the finding
        that motivated the cleanup). `weighted_clean`'s output on the same
        prompt has no such artifact.
      - **Not a uniform win.** The dragon's-lair prompt was clearly the best
        output of any checkpoint tested (`weighted_clean` produced
        consistent scenario description with numbered dungeon areas, e.g.
        "area 16c" — a real published-adventure convention). But a new
        pattern also appeared: on the narrative (rogue/crypt) prompt,
        `weighted_clean` derailed into a structured wiki-infobox-style item
        card (`Rarity: Common, Daggerford`, `Type: Weapon... CR: 8 HP: 1`,
        a plausible-looking URL) — a different flavor of topic drift, likely
        reflecting more exposure to structured wiki/database corpus content.
      - **A real regression on one specific fact, investigated below (CR/XP
        recall).**
      - No checkpoint reliably states correct facts overall — expected,
        unchanged from the earlier check; still raw pretraining at ~26% of
        Chinchilla-optimal tokens, no fine-tuning or retrieval grounding.
      - **Verdict: cleared to move forward.** No new regression that
        outweighs the confirmed fix and the per-tier loss evidence above;
        remaining incoherence is the same pre-existing limitation of a
        sub-optimal-data small pretrained model.
- [x] **Dug into the CR5/XP-recall regression.** Single-seed sampling on the
      "Challenge Rating... grants" prompt showed `weighted_clean` producing
      more rambling, less grounded output than `weighted`. Repeated with 5
      seeds each to check it wasn't noise:
      | | XP mentioned | Correct XP (1,800)? | Register |
      |---|---|---|---|
      | `weighted` | 2/5 seeds | Yes, exactly (seed 3) | Stayed in stat-block/CR-table register throughout, even when wrong |
      | `weighted_clean` | 1/5 seeds | No (nonsensical "50 XP... for every 1 hour") | 3/5 seeds rambled into confused meta-discussion ("Factors to Monsters") |

      **Real and reproducible, not single-seed noise** — but hard to
      attribute cleanly to any one change, because `weighted` and
      `weighted_clean` differ in three ways at once: the cleaned corpus, the
      validation-split *method* (the old buggy contiguous-tail split vs. the
      new scattered-block split — meaning a **different ~1% of the corpus
      was excluded from training** in each run, so CR/XP-table-adjacent
      content may simply have landed in one run's held-out blocks and not
      the other's by chance), and ordinary run-to-run training noise
      (different init/shuffle order, no replicate run exists to rule this
      out). A narrow, 10-sample finding from one prompt family — doesn't
      overturn the per-tier loss evidence (thousands of windows) that
      up-weighted content is fit substantially better overall; consistent
      with an aggregate metric hiding one specific pocket of weakness.
      **Not disqualifying**: the production system relies on retrieval, not
      raw pretrain memorization, to inject verified CR/XP facts — exactly
      the gap this weakness represents is what retrieval grounding exists to
      cover.
- [x] **Found and fixed a real regression the SE cleanup caused in a
      different pipeline.** Preparing to actually run
      `scripts/build_finetune_data_from_qa.py` against `weighted_clean`
      surfaced it: `grimoire_ai.llm.data.qa_pairs.load_qa_pairs` (used by
      that script and by `embed_tune.py`'s `--qa-corpus-dir`) parses Q&A
      structure by keying off the exact markers
      `clean_stackexchange_markup.py` strips (`# title`, `Score: N`,
      `## Answer (score: N)`, `---`). Pointed at the live, cleaned
      `data/corpus/saga/`, it now silently returns **zero** pairs
      (confirmed: 0 vs. 77,797 from the pre-cleanup copy, 31,202
      accepted-only) — the cleanup fixed pretraining-corpus quality at the
      cost of breaking this downstream consumer, undetected until actually
      needed. Fix: the pre-cleanup backup already had exactly what this
      needs, so it was renamed from `saga_backup_pre_se_cleanup/` (implies
      "revert point only") to `saga_se_qa_source/` (its real, ongoing
      role) rather than left as an accidental dependency on a directory
      named like a one-off safety net. `build_finetune_data_from_qa.py`,
      `embed_tune.py`, and `clean_stackexchange_markup.py`'s own default
      backup path all updated to point at/use the new name, and both
      scripts' empty-pairs errors now name this exact gotcha instead of a
      generic "no pairs found" message.
- [x] **Fine-tuned `weighted_clean` — found and fixed a second regression
      from the same cleanup, in the fine-tune data itself.** First attempt
      used `--accepted-only` (following `build_finetune_data_from_qa.py`'s
      own docstring example) — 31,294 examples, `small-25M`, 3,000 steps
      (~1.5 epochs at effective batch 16). Training loss collapsed to
      ~0.09, suspiciously low; qualitative check confirmed why:
      **severely degenerate**, every response echoing the question then
      collapsing into token loops (`encounter encounter encounter...`,
      `is is is is...`), even with the correct `repetition_penalty=1.3` —
      not a sampling-config artifact, a genuinely broken checkpoint.

      Root cause: `--accepted-only` keeps exactly one (the accepted)
      answer per question, so **0 of 31,305 questions had a second answer
      available**. `qa_pairs_to_finetune_examples`'s `_pick_context_pair`
      mechanism — which exists specifically to draw `context` from a
      *different* answer than `assistant` — never had anything to pick
      from, so 100% of examples fell back to the same-source path. That
      fallback is the module's own documented risk: *"context is always a
      superset-prefix of assistant, which in practice taught the generator
      to echo/loop through whatever passage it's given."* `--accepted-only`
      structurally guaranteed the worst case of exactly that, for the
      entire dataset.

      Rebuilt with `--min-score 1` instead (77,740 examples; 23,401 of
      43,202 questions have multiple answers, so the context/assistant
      separation actually functions for most of the data), 7,288 steps
      (~1.5 epochs at the larger size). Loss settled around 2.7–2.8 instead
      of collapsing. Qualitative check: coherent, on-topic, real
      terminology (Player's Handbook, CR, HP, DMG, advantage/disadvantage,
      spell slots), no repetition loops, no question-echoing. Facts are
      still sometimes wrong or muddled — expected, same
      ~26%-of-Chinchilla-optimal pretraining limitation as ever, not a
      fine-tuning failure. Checkpoint:
      `checkpoints/finetune/saga-se-qa-weighted-clean-v2/step_0007288.pt`.
- [x] **Side-by-side comparison against the actual production checkpoint —
      found the currently-live checkpoint is itself severely degenerate.**
      7 prompts (the 5 above plus 2 more: grapple, cantrip), same
      `repetition_penalty=1.3` as `agents.json`. `saga-se-qa-clean-v2`
      (what was live) failed on **every single prompt** with the same
      question-echo-then-repetition-loop collapse diagnosed above
      (`does does does...`, `the the the...`, `I I I I...`, `one one
      one...`) — not a marginal quality gap, unusable output across the
      board. `saga-se-qa-weighted-clean-v2` stayed coherent on all 7,
      facts sometimes wrong/muddled but no collapse. Strongly suggests the
      live checkpoint was built the same `--accepted-only` way and shipped
      without this qualitative check ever having been run against it.
- [x] **Updated `agents.json`** — `saga`'s checkpoint now points at
      `saga-se-qa-weighted-clean-v2/step_0007288.pt`, replacing the
      degenerate `saga-se-qa-clean-v2`.
- [x] **Ran the formal evaluation harness on both checkpoints (2026-07-08)** —
      quantitative confirmation of the qualitative finding above, not just
      a "seems better" impression:
      | Metric | Old (`saga-se-qa-clean-v2`) | New (`saga-se-qa-weighted-clean-v2`) |
      |---|---|---|
      | Perplexity | 23,133.19 | 195.95 |
      | BPC | 14.4977 | 7.6143 |
      | Retrieval hit-rate | 5.0% (1/20) | 5.0% (1/20) |
      | Quiz pass-rate | 2.0% (1/49) | 20.4% (10/49) |
      | Quiz kw-recall | 1.36% | 11.56% |
      | Quiz token-F1 | 0.0675 | 0.1768 |

      Both run with `--quiz-repetition-penalty 1.3`, matching
      `agents.json`'s actual generation setting, on the same
      `data/corpus/saga/` + `data/processed/corpus.bin`.

      **Perplexity 23,133 is worse than random guessing** — with a
      16,384-token vocabulary, uniform-random guessing gives perplexity
      ≈16,384. As unambiguous a confirmation of "genuinely broken" as a
      number can give, matching the repetition-loop qualitative finding
      exactly.

      **Retrieval hit-rate is identical (5.0%, exactly 1/20) on both** —
      exactly as expected, since the lexical encoder never touches the
      model's embeddings. Confirms it's a separate, pre-existing gap (see
      the per-query inspection below), not something either checkpoint's
      fine-tuning affects.

      **Quiz pass-rate is a full order of magnitude better** (2.0% →
      20.4%), directly validating the fine-tune fix on the metric that
      actually reflects usefulness.

      **The retrieval gap itself is worth a closer look separately** (not
      blocking the checkpoint swap): inspected the saved report's
      per-query detail — retrieved top passages are consistently
      irrelevant to the query (e.g. "grapple speed movement" surfaces a
      random adventure-module encounter description, "frightened condition
      disadvantage" surfaces illithid lore). The lexical (stemmed 4-gram
      Jaccard) engine was likely tuned against a much smaller, rules-dense
      corpus; at 1,469 files now dominated by adventure modules and bulk
      fiction, simple word-overlap has a much harder time surfacing the
      right SRD passage among far more volume. Matches the project's own
      stated design (semantic retrieval, not lexical, was always meant to
      be the primary path) — a pre-existing structural gap, not a
      regression from anything in this session. Candidate for its own
      follow-up (semantic/LoRA retrieval wiring is already tracked
      separately in `PLAN.md`'s Phase 5 gap).
- [ ] Revisit finer-grained weight tiers (e.g. splitting `rpg_se_*` Q&A
      prose from official-book prose, both currently lumped at/near
      baseline) — now with a much clearer per-tier signal to design against
      (via `--val-stratified`) instead of guessing blind.

## Open decisions for next session

- [x] Once corpus token count is known post-expansion, decide target model
      size using the 20:1 ratio rather than jumping straight to `medium-85M`.
      **Stayed at `small-25M`.** Post-expansion corpus is 124,851,189 tokens
      — barely different from the ~129M this reminder was originally written
      against (the session's changes were about *quality*, not volume; the
      quality filter and the `wotc-srd` re-scrape net *removed* more
      duplicate/junk content than EntiGraph added). At the real 20:1 ratio,
      124.85M/500M (~25%) against `small-25M`'s optimum is still better than
      124.85M/1.7B (~7.3%) against `medium-85M`'s.
- [ ] Re-measure the MiniLM retrieval baseline at
      `--quiz-repetition-penalty 1.3` (currently only measured at 1.0 — not a
      fair comparison against everything else in this session's results).
- [ ] Phase 5 (pre-existing gap, unrelated to this plan): wire semantic/LoRA
      retrieval into the live UI — it currently only supports lexical search
      via `corpus_dirs`.
- [x] EntiGraph-generated passages (`scripts/generate_open5e_entigraph.py`,
      output in `data/corpus/saga_derived/entigraph_*.txt`) needed a real
      preprocess+retrain pass to actually reach the model. Wired via
      repeatable `grimoire-preprocess --input`, weighted `entigraph_*:1`,
      included in the `weighted_clean_v2` pretrain run below.
- [x] `open5e_spells.txt`/`open5e_monsters.txt` in the existing corpus were
      43%/48% duplicate-name entries blending Open5e's official `wotc-srd`
      document with unrelated third-party rulesets (`a5e`, `kp`), yet both
      sat in the corpus's highest weight tier (`*:1.75` catch-all) as if
      uniformly official. Re-scraped via `scripts/scrape_open5e.py
      --endpoints spells monsters --document-slug wotc-srd` (322/319 records,
      down from 3207/1435, zero duplicates, zero third-party leakage
      afterward) and folded into the `weighted_clean_v2` preprocess+retrain
      pass below.
- [x] **Pretrained `weighted_clean_v2`** (`small-25M`, 15,259 steps,
      `--val-stratified`, 2026-08-15) — first run on the corpus after the
      quality filter, the `wotc-srd` re-scrape, and the EntiGraph additions
      above. 124,851,189 tokens (1512 input files, 21 dropped by
      `--quality-filter` — 2 genuine junk documents, 1 known/deferred
      `mean_word_length` false positive on `Monster Manual (2025).txt`, 18
      MathML-noise-heavy `wp_math_*` pages). 20,198.1s wall-clock (faster
      than the July 3 `weighted_clean` run's 24,143.5s for the same step
      count). Checkpoint: `checkpoints/pretrain/weighted_clean_v2/step_0015259.pt`.
      Config: `train_config_weighted_clean_v2.json`.
- [x] **Per-tier validation loss on `weighted_clean_v2`** (new reusable tool:
      `scripts/eval_per_tier.py`, reproduces `train.py`'s own
      `--val-stratified` split so results are directly comparable to what
      training itself held out): `0.5` (down-weighted) 3.5468, `1.0`
      (baseline) 3.2673, `1.75` (up-weighted) 2.3677 — monotonic ordering
      exactly matching the intended prioritization, same as the July 3
      finding. Notably *more* trustworthy than that earlier result: July
      3's up-weighted number (2.1863) was a train-inclusive estimate
      because the old scattered-block validation split happened to miss
      that tier entirely by chance, while this run's `--val-stratified`
      split guarantees proportional coverage (189/18,871 windows, ~1% as
      intended) — and the gap to baseline held up under this stricter,
      genuinely held-out measurement (0.8996 nats vs. July 3's 0.83-nat
      train-inclusive gap).
- [x] **Qualitative completion check on `weighted_clean_v2`**
      (`scripts/qualitative_check.py`, 2026-08-15). Coherent grammar, real
      D&D terminology, no repetition loops or degenerate collapse on any
      of the 6 prompts — matches the quality bar of prior checks at this
      training stage (facts sometimes wrong/muddled, expected at ~25% of
      Chinchilla-optimal). One finding: the `condition` prompt derailed
      mid-generation into a **verbatim, memorized third-party product URL**
      (`Rarity: rare` / `Document  url: https://koboldpress.com/kpstore/
      product/vault-of-magic-for-5th-edition/`) — traced to a real bug, not
      just topic drift (see below).
- [x] **Found and fixed `_fmt_generic`'s `document__url` leak** (surfaced by
      the finding above). `_fmt_generic` (the fallback formatter for every
      Open5e endpoint except monsters/spells, which have dedicated
      formatters) excluded `document__slug`/`document__title` but not
      `document__url`/`document__license_url`, so every entry in those
      other endpoints printed a raw source URL into corpus text. Checked
      the actual corpus: **100% of entries leak this** in
      `armor`/`backgrounds`/`conditions`/`feats`/`magic_items`/`races`/
      `weapons`, 45/52 in `sections`. Fixed by excluding the whole
      `document__*` prefix.
- [x] **Checked the same document-blending problem (found in
      `open5e_spells.txt`/`open5e_monsters.txt`, see above) across the rest
      of the `open5e_*` files**, via the URLs the leak above incidentally
      exposed (no API access needed — the leaked URL names each entry's
      source document directly). `conditions`/`sections` are already 100%
      `wotc-srd`, no action needed. The rest are contaminated, `magic_items`
      and `backgrounds`/`feats` severely so:
      | File | wotc-srd share |
      |---|---|
      | `armor` | 18/23 (78%) |
      | `weapons` | 37/68 (54%) |
      | `races` | 9/20 (45%) |
      | `magic_items` | 237/1,618 (15%) |
      | `backgrounds` | 1/42 (2%) |
      | `feats` | 1/74 (1%) |

      `backgrounds` and `feats` are almost entirely Kobold Press/A5E/Green
      Ronin homebrew presented as core D&D content — the opposite of what
      this corpus's weight-pattern scheme assumes for this tier (currently
      the *highest*-weighted, `*:1.75`).
- [x] **Re-scraped the 6 contaminated endpoints**, filtered to `wotc-srd`
      (2026-08-15): `armor` 23→18, `weapons` 68→37, `races` 20→9,
      `magic_items` 1,618→237 (28.2% duplicate-name rate before, 0% after),
      `backgrounds` 42→1 (just Acolyte — the real 5e SRD's only licensed
      background), `feats` 74→1 (just Grappler). Confirmed correct against
      known 5e SRD licensing scope, not a scraper bug — both single-entry
      files were spot-checked and are legitimate, complete, clean SRD text.
      Zero `document__url` leaks remained in any of the six afterward.
- [x] **Pretrained `weighted_clean_v3`** (`small-25M`, 15,259 steps,
      `--val-stratified`, 2026-08-15) on the corpus after the endpoint
      re-scrape above — 124,412,387 tokens (down ~439K from `v2`, consistent
      with the six files shrinking), same 21 quality-filter drops as `v2`
      (identical indices/reasons — none of the re-scraped files were among
      them). 20,272.8s wall-clock, essentially identical to `v2`'s 20,198.1s
      (expected: runtime tracks step count, not corpus size). Checkpoint:
      `checkpoints/pretrain/weighted_clean_v3/step_0015259.pt`. Config:
      `train_config_weighted_clean_v3.json`.
- [x] **Per-tier validation loss on `weighted_clean_v3`** vs. `v2`: `0.5`
      3.5468→3.5462, `1.0` 3.2673→3.2667, `1.75` 2.3677→2.3739 — every
      delta within run-to-run noise, monotonic tier ordering still holds.
      Aggregate training-log val loss was ~0.005 nats higher than `v2` at
      each matching step, which looked like a possible regression at first,
      but the per-tier breakdown showed it wasn't real. Expected either
      way: this fix was about *correctness* (removing memorized third-party
      content and a URL leak), not volume or coverage, so it was never
      going to show up as a loss improvement.
- [x] **Qualitative completion check on `weighted_clean_v3`**
      (2026-08-15) — confirms the `document__url` leak fix actually
      worked: the `condition` prompt (which derailed into a verbatim
      Kobold Press product URL on `v2`) no longer does so on `v3`. Still
      factually wrong in the ordinary, expected way (says grappled
      prevents actions/reactions, which is actually closer to
      restrained/incapacitated) — a muddled-facts error, not a
      memorization artifact. Overall quality matches `v2`: coherent, no
      repetition loops, real terminology, a correct adventure-module
      reference (`Lost Mine of Phandelver`). Same recurring "forum Q&A
      register" drift on narrative prompts as `v2` — pre-existing,
      unrelated to this fix, not a regression.
- [ ] Curate more Q&A pairs for fine-tuning `weighted_clean_v3`, the way
      `weighted_clean` was fine-tuned into `saga-se-qa-weighted-clean-v2`.
