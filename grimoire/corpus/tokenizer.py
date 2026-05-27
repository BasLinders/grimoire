class GrimoireTokenizer:
    def __init__(self, n: int = 4):
        self.n = n

    def build(self, stemmed_tokens: list[str]) -> list[tuple[str, ...]]:
        if len(stemmed_tokens) < self.n:
            return []
        return [
            tuple(stemmed_tokens[i : i + self.n])
            for i in range(len(stemmed_tokens) - self.n + 1)
        ]
