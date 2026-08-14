import pytest
from grimoire_ai.corpus.corpus import GrimoireCorpus
from grimoire_ai.corpus.index import CorpusIndex
from grimoire_ai.corpus.stemmer import GrimoireStemmer
from grimoire_ai.corpus.tokenizer import GrimoireTokenizer

FIXTURE_TEXT = (
    "A grappled creature has its speed reduced to zero. "
    "The grapple condition also prevents the creature from moving."
)


class TestGrimoireStemmer:
    def setup_method(self):
        self.stemmer = GrimoireStemmer()

    def test_stem_consistency(self):
        assert self.stemmer.stem("grapple") == self.stemmer.stem("grappling")
        assert self.stemmer.stem("grapple") == self.stemmer.stem("grappled")

    def test_acronym_preserved_lowercase(self):
        assert self.stemmer.stem("D&D") == "d&d"
        assert self.stemmer.stem("HP") == "hp"
        assert self.stemmer.stem("AC") == "ac"
        assert self.stemmer.stem("NPC") == "npc"

    def test_tokenize_and_stem_extracts_roots(self):
        tokens = self.stemmer.tokenize_and_stem(FIXTURE_TEXT)
        assert "grappl" in tokens
        assert "creatur" in tokens
        assert "speed" in tokens

    def test_single_char_tokens_excluded(self):
        tokens = self.stemmer.tokenize_and_stem("A big ox ran.")
        assert "a" not in tokens


class TestGrimoireTokenizer:
    def setup_method(self):
        self.tokenizer = GrimoireTokenizer(n=4)

    def test_builds_correct_ngrams(self):
        tokens = ["grappl", "creatur", "speed", "reduc", "zero"]
        result = self.tokenizer.build(tokens)
        assert result[0] == ("grappl", "creatur", "speed", "reduc")
        assert result[1] == ("creatur", "speed", "reduc", "zero")

    def test_count_is_len_minus_n_plus_one(self):
        tokens = ["a", "b", "c", "d", "e"]
        result = self.tokenizer.build(tokens)
        assert len(result) == 2

    def test_shorter_than_n_returns_empty(self):
        assert self.tokenizer.build(["one", "two", "three"]) == []

    def test_exact_length_returns_one(self):
        assert len(self.tokenizer.build(["a", "b", "c", "d"])) == 1


class TestCorpusIndex:
    def test_add_and_retrieve(self):
        index = CorpusIndex()
        mt = ("grappl", "creatur", "speed", "reduc")
        index.add(mt, next_token="zero", source="dnd")
        entry = index.get(mt)
        assert entry.next_token == "zero"
        assert entry.source == "dnd"
        assert entry.frequency == 1

    def test_duplicate_increments_frequency(self):
        index = CorpusIndex()
        mt = ("grappl", "creatur", "speed", "reduc")
        index.add(mt)
        index.add(mt)
        assert index.get(mt).frequency == 2

    def test_missing_key_returns_none(self):
        index = CorpusIndex()
        assert index.get(("x", "y", "z", "w")) is None

    def test_len(self):
        index = CorpusIndex()
        index.add(("a", "b", "c", "d"))
        index.add(("e", "f", "g", "h"))
        assert len(index) == 2

    def test_candidates_for_words_finds_sharing_multi_tokens(self):
        index = CorpusIndex()
        index.add(("grappl", "creatur", "speed", "reduc"))
        index.add(("fire", "burn", "everyth", "nearbi"))
        candidates = index.candidates_for_words({"speed"})
        assert candidates == {("grappl", "creatur", "speed", "reduc")}

    def test_candidates_for_words_unions_across_multiple_words(self):
        index = CorpusIndex()
        mt_a = ("grappl", "creatur", "speed", "reduc")
        mt_b = ("fire", "burn", "everyth", "nearbi")
        index.add(mt_a)
        index.add(mt_b)
        candidates = index.candidates_for_words({"speed", "fire"})
        assert candidates == {mt_a, mt_b}

    def test_candidates_for_words_empty_for_unknown_word(self):
        index = CorpusIndex()
        index.add(("grappl", "creatur", "speed", "reduc"))
        assert index.candidates_for_words({"nonexist"}) == set()

    def test_candidates_for_words_empty_for_empty_query(self):
        index = CorpusIndex()
        index.add(("grappl", "creatur", "speed", "reduc"))
        assert index.candidates_for_words(set()) == set()

    def test_candidates_for_words_does_not_duplicate_across_shared_words(self):
        """A multi-token sharing more than one word with the query must still
        appear exactly once (set union, not concatenation)."""
        index = CorpusIndex()
        mt = ("grappl", "creatur", "speed", "reduc")
        index.add(mt)
        candidates = index.candidates_for_words({"grappl", "speed", "creatur"})
        assert candidates == {mt}


class TestGrimoireCorpus:
    def setup_method(self):
        self.corpus = GrimoireCorpus(n=4)
        self.corpus.add_text(FIXTURE_TEXT, source="dnd_srd")

    def test_corpus_populated(self):
        assert self.corpus.size > 0

    def test_query_returns_results(self):
        results = self.corpus.query("grapple speed movement", top_k=3)
        assert len(results) > 0

    def test_query_result_fields(self):
        results = self.corpus.query("grapple speed", top_k=1)
        r = results[0]
        assert r.source == "dnd_srd"
        assert isinstance(r.score, float)
        assert isinstance(r.multi_token, tuple)

    def test_query_relevance(self):
        results = self.corpus.query("grappled creature speed", top_k=5)
        all_tokens = {t for r in results for t in r.multi_token}
        assert "grappl" in all_tokens or "creatur" in all_tokens

    def test_unknown_query_returns_empty(self):
        results = self.corpus.query("quantum entanglement photon", top_k=5)
        assert results == []

    def test_add_text_returns_count(self):
        corpus = GrimoireCorpus(n=4)
        count = corpus.add_text("The wizard cast a powerful fireball spell at the enemy.")
        assert count > 0

    def test_query_matches_brute_force_scan_over_random_corpus(self):
        """The inverted-index-backed query() (docs/inference_optimization.md
        item #8) must return exactly the same (multi_token, score) pairs a
        full linear scan over every indexed multi-token would -- narrowing
        the candidate set is a pure speedup, never a change in results."""
        import random

        rng = random.Random(0)
        words_pool = [
            "grapple", "fire", "stealth", "wizard", "dragon", "sword",
            "shield", "magic", "speed", "creature", "attack", "damage",
        ]
        corpus = GrimoireCorpus(n=4)
        for i in range(50):
            text = " ".join(rng.choices(words_pool, k=rng.randint(5, 20)))
            corpus.add_text(text, source=f"doc{i}")

        def brute_force_scores(query_text: str) -> dict:
            query_tokens = corpus._stemmer.tokenize_and_stem(query_text)
            query_mts = corpus._tokenizer.build(query_tokens)
            query_set = (
                {t for mt in query_mts for t in mt} if query_mts else set(query_tokens)
            )
            scores = {}
            for mt, entry in corpus._index.all_entries().items():
                overlap = len(set(mt) & query_set)
                if overlap > 0:
                    union = len(set(mt) | query_set)
                    scores[mt] = (overlap / union, entry.frequency)
            return scores

        for query in [
            "grapple speed creature",
            "fire damage wizard",
            "stealth attack dragon sword",
            "nonexistent gibberish words",
        ]:
            # top_k large enough to return every scoring candidate, so this
            # compares the full result set, not just a truncated head.
            results = corpus.query(query, top_k=10_000)
            actual = {(r.multi_token, r.score) for r in results}
            expected = {
                (mt, jaccard) for mt, (jaccard, _freq) in brute_force_scores(query).items()
            }
            assert actual == expected, f"Mismatch for query {query!r}"


class TestExcerpts:
    def setup_method(self):
        self.corpus = GrimoireCorpus(n=4)
        self.corpus.add_text(
            "A grappled creature has its speed reduced to zero.",
            source="dnd_srd",
        )

    def test_query_results_have_excerpts(self):
        results = self.corpus.query("grapple speed", top_k=3)
        assert all(r.excerpt is not None for r in results), (
            "All results should carry an excerpt when add_text is used."
        )

    def test_excerpt_contains_original_words(self):
        results = self.corpus.query("grapple speed", top_k=1)
        excerpt = results[0].excerpt
        # The excerpt must contain recognisable words from the original text
        # (unstemmed), not the stemmed index tokens.
        assert any(w in excerpt for w in ("grappled", "creature", "speed", "zero")), (
            f"Excerpt does not look like original text: {excerpt!r}"
        )

    def test_excerpt_within_window(self):
        from grimoire_ai.corpus.corpus import _EXCERPT_WINDOW
        results = self.corpus.query("grapple speed", top_k=3)
        for r in results:
            assert len(r.excerpt) <= _EXCERPT_WINDOW + 50, (
                f"Excerpt suspiciously long: {len(r.excerpt)} chars"
            )

    def test_index_add_without_excerpt_gives_none(self):
        from grimoire_ai.corpus.index import CorpusIndex
        idx = CorpusIndex()
        idx.add(("a", "b", "c", "d"), next_token="next")
        assert idx.get(("a", "b", "c", "d")).excerpt is None
