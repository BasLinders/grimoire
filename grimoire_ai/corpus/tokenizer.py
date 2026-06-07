"""N-gram builder for the Grimoire corpus pipeline.

An n-gram (or "multi-token" in Granville's terminology) is a contiguous
sequence of n stemmed words. For example, with n=4 the sentence:

    ["grappl", "creatur", "speed", "reduc", "zero"]

produces two 4-grams:

    ("grappl", "creatur", "speed", "reduc")
    ("creatur", "speed", "reduc", "zero")

These multi-tokens serve as the atomic units stored in the corpus index and
used as keys in the RBF interpolator. Using n=4 matches the setting from
Granville's NVIDIA case study, which achieved 96 % next-token accuracy.
"""


class GrimoireTokenizer:
    """Converts a list of stemmed tokens into overlapping n-gram tuples.

    Each tuple is called a "multi-token" and represents a fixed-length
    window of consecutive stemmed words. Overlapping windows ensure that
    every word appears in multiple multi-tokens, preserving local context.

    Attributes:
        n: The window size (number of stemmed words per multi-token).
            Granville's default is 4.
    """

    def __init__(self, n: int = 4) -> None:
        """Initialise the tokenizer with a given n-gram window size.

        Args:
            n: Number of stemmed tokens per multi-token. Must be a positive
                integer. Defaults to 4, matching Granville's case study.
        """
        self.n = n

    def build(self, stemmed_tokens: list[str]) -> list[tuple[str, ...]]:
        """Build all overlapping n-gram tuples from a stemmed token list.

        Slides a window of width ``n`` across ``stemmed_tokens`` one position
        at a time, yielding ``len(stemmed_tokens) - n + 1`` tuples in total.
        Returns an empty list when there are fewer tokens than ``n``.

        Args:
            stemmed_tokens: An ordered list of stemmed words, as produced
                by ``GrimoireStemmer.tokenize_and_stem``.

        Returns:
            A list of tuples, each containing exactly ``n`` stemmed strings.
            The list is empty if ``len(stemmed_tokens) < n``.

        Examples:
            >>> t = GrimoireTokenizer(n=4)
            >>> t.build(["grappl", "creatur", "speed", "reduc", "zero"])
            [('grappl', 'creatur', 'speed', 'reduc'),
             ('creatur', 'speed', 'reduc', 'zero')]
        """
        if len(stemmed_tokens) < self.n:
            return []
        return [
            tuple(stemmed_tokens[i : i + self.n])
            for i in range(len(stemmed_tokens) - self.n + 1)
        ]
