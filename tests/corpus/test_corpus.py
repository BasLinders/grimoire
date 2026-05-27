import pytest
from grimoire.corpus.corpus import GrimoireCorpus
from grimoire.corpus.index import CorpusIndex
from grimoire.corpus.stemmer import GrimoireStemmer
from grimoire.corpus.tokenizer import GrimoireTokenizer

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
