# scripts/

Standalone developer-time CLIs for the Saga data/training pipeline, organized
by pipeline stage. Not an installed package — invoke each script directly
with `python scripts/<subdir>/<name>.py`.

Not to be confused with the planned `tools/` directory described in
[docs/tools.md](../docs/tools.md), which is for cross-agent utilities —
everything here is Saga/D&D-corpus-specific.

```
scrape/       Per-source corpus scrapers (Wikipedia, Wikibooks, arXiv, Gutenberg,
              D&D Wiki, GitHub D&D repos, Fandom wikis, Open5e, 5etools, Internet
              Archive magazines, Stack Exchange) + rename_wotc_sourcebooks.py and
              the shared _stackexchange_common.py helper.

corpus/       Corpus assembly and cleanup: build_saga_corpus.py (seed corpus),
              clean_stackexchange_markup.py, dedup_corpus.py, score_corpus_quality.py,
              downsample_jsonl.py.

finetune/     Fine-tune data generation and prep: generate_lore.py,
              generate_open5e_qa.py, generate_open5e_entigraph.py,
              build_finetune_data_from_qa.py, dehedge_finetune_data.py,
              validate_finetune_data.py, score_difficulty.py, build_source_weights.py.
  data/       JSONL fine-tune datasets (Saga Q&A, math tool calls, general chat).

retrieval/    build_retrieval_neighbors.py, embed_tune.py — RETRO-style neighbor
              precompute and contrastive LoRA fine-tuning for retrieval embeddings.

train/        finetune_saga.py, lr_range_test.py, compare_checkpoints.py.

eval/         evaluate.py, eval_per_tier.py, qualitative_check.py.
  data/       saga_quiz.jsonl — the Q&A quiz used by evaluate.py.

export/       export_gguf.py — export a GQA checkpoint to GGUF for llama.cpp.

references/   Hand-authored math/probability reference .txt files, copied into
              the corpus by corpus/build_saga_corpus.py.
```

Each script's own docstring has full usage details and `--help` output.
