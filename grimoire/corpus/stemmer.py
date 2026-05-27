import re
from nltk.stem import PorterStemmer


class GrimoireStemmer:
    # Uppercase sequences up to 6 chars treated as acronyms and kept as-is (lowercased)
    _ACRONYM_RE = re.compile(r'^[A-Z][A-Z0-9&\-]{0,5}$')
    _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&\-']*")

    def __init__(self):
        self._stemmer = PorterStemmer()

    def stem(self, word: str) -> str:
        if self._ACRONYM_RE.match(word):
            return word.lower()
        return self._stemmer.stem(word.lower())

    def tokenize_and_stem(self, text: str) -> list[str]:
        tokens = self._TOKEN_RE.findall(text)
        return [self.stem(t) for t in tokens if len(t) > 1]
