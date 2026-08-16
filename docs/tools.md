# Tools — cross-agent developer utilities

A planned `tools/` directory (top-level, sibling to `scripts/`) for
developer/data-curation utilities that are useful across *any* agent,
not just Saga — as opposed to `scripts/`, which has grown almost
entirely Saga/D&D-corpus-specific (Open5e scrapers, corpus generators,
preprocessing CLIs tied to the Saga corpus's own file-naming
conventions like `open5e_*`/`entigraph_*`/`*_se_*`).

**Not to be confused with `grimoire_ai/tools/`** (`MathTool` and
friends) — that package is *runtime* agent tool-calling, invoked during
chat inference when a model response requests a tool (e.g.
`<TOOL:python>` arithmetic). This directory is for *developer-time*
utilities run by hand between training runs, never imported by the
running application.

Motivation: `AgentRegistry`/`AgentRouter`/`MultiAgentEngine` already
support multiple agents (`docs/PLAN.md`'s Phase 6, done, 18-test suite)
— Saga is just the first agent actually built out on top of that
infrastructure. Any data-curation utility that isn't inherently
Saga-specific will need a home that doesn't imply otherwise, so the
next agent's maintainer finds it instead of re-writing it.

## Planned steps

- [ ] Create the `tools/` directory.
- [ ] Move `scripts/downsample_jsonl.py` there. Already written generic
      on purpose (works on any JSONL file, no Saga/fine-tune-specific
      assumptions) — built when combining Saga's fine-tune sources
      turned up a real need (three general StackExchange sites
      outweighing the entire D&D-specific dataset; see
      `docs/training_PLAN.md`), and structured generically since a
      second agent hitting the same
      general-content-outweighs-domain-content problem is exactly the
      scenario this directory exists for. Update its own usage
      examples and `docs/training_PLAN.md`'s reference to the new
      path when this happens. Hold off on the actual move until a
      second real call site shows up (e.g. a second agent's fine-tune
      mix) — the point of `tools/` is demonstrated reuse, and moving a
      file after one use site doesn't yet prove that; it's flagged
      here so the decision isn't lost, not so it happens immediately.
- [ ] As other genuinely cross-agent utilities come up, apply the same
      test before placing them here: does the script assume anything
      Saga/D&D-specific (filenames, corpus layout, `--weight-pattern`
      tags, `document__slug` conventions), or would it work unmodified
      for a hypothetical second agent's data? Only the latter belongs
      in `tools/` — anything that still bakes in a Saga assumption
      stays in `scripts/` until it's generalized.
