"""Byte-Pair Encoding (BPE) tokenizer for the Grimoire LLM.

BPE builds a vocabulary by iteratively merging the most frequent pair of
adjacent symbols in a corpus.  Starting from a base alphabet of individual
UTF-8 bytes (represented as single characters after a byte-to-char mapping),
each merge step replaces the most common adjacent pair with a new symbol.
After ``vocab_size - n_special`` merges the vocabulary is complete.

Using byte-level pre-tokenisation makes the encoder lossless: every possible
byte sequence can be represented exactly, with no unknown tokens.  This is
important for domain text that contains dice notation (``2d6+3``), math
symbols, and D&D shorthand (``DC 15``, ``+2 longsword``).

Vocabulary layout
-----------------
ids 0 – 5        Reserved for special tokens (see ``special_tokens.py``).
ids 6 – 261      The 256 base byte symbols (one per possible byte value).
ids 262 – …      Learned BPE merge symbols, in the order they were created.

Typical usage
-------------
    encoder = BytePairEncoder()
    encoder.train(corpus_texts, vocab_size=8192)
    encoder.save("data/tokenizer/bpe.json")

    encoder2 = BytePairEncoder.load("data/tokenizer/bpe.json")
    ids = encoder2.encode("A grappled creature loses its speed.")
    text = encoder2.decode(ids)
    assert text == "A grappled creature loses its speed."
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from grimoire.llm.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    SPECIAL_TOKEN_TO_ID,
)

# Regex that splits text on whitespace boundaries while keeping each word
# together with any leading space (GPT-2 style).  This means "hello world"
# becomes [" hello", " world"] rather than ["hello", " ", "world"], which
# preserves spacing information through the encode/decode round-trip.
_WORD_SPLIT_RE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s""")

# Number of special-token ids reserved at the top of the vocabulary.
_N_SPECIAL = len(ALL_SPECIAL_TOKENS)  # 6

# 256 base byte symbols start immediately after the special tokens.
_BASE_OFFSET = _N_SPECIAL  # first base-byte id = 6


def _bytes_to_chars(b: bytes) -> list[str]:
    """Convert a byte sequence to a list of single-character base symbols.

    Each byte value 0–255 is mapped to a unique printable Unicode character
    so it can be stored in a plain string and manipulated as a character.
    The mapping used here is the same one GPT-2 uses: bytes in the printable
    ASCII range map to themselves; the remaining 94 bytes map to the
    Unicode block starting at U+0100.

    Args:
        b: Raw bytes to convert.

    Returns:
        A list where each element is a one-character string representing
        one byte from the input.
    """
    return [_BYTE_TO_CHAR[byte] for byte in b]


def _build_byte_char_maps() -> tuple[dict[int, str], dict[str, int]]:
    """Build bidirectional maps between byte values (0-255) and characters.

    Printable ASCII characters (33–126 and 161–172, 174–255) map to
    themselves. The remaining bytes map sequentially to characters
    starting at U+0100 so every byte has a unique, unambiguous
    character representation.

    Returns:
        A tuple ``(byte_to_char, char_to_byte)`` where each is a dict
        covering all 256 possible byte values.
    """
    bs: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs: list[int] = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_to_char = {b: chr(c) for b, c in zip(bs, cs)}
    char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    return byte_to_char, char_to_byte


_BYTE_TO_CHAR, _CHAR_TO_BYTE = _build_byte_char_maps()


class BytePairEncoder:
    """Byte-level BPE tokenizer that can be trained, saved, and loaded.

    After training, the encoder maps arbitrary text to lists of integer ids
    and back. It is safe to use across Python sessions as long as the
    vocabulary file is the same.

    Attributes:
        vocab_size: The total vocabulary size including special tokens and
            base byte symbols.  Set after ``train`` is called or after
            ``load`` is called.
        _encoder: Mapping from string symbol to integer id.
        _decoder: Mapping from integer id to string symbol.
        _merges: Ordered list of ``(a, b)`` string pairs representing BPE
            merge rules, in the order they were learned.  During encoding
            rules are applied in this order (highest priority first).
        _merge_ranks: Dict mapping ``(a, b)`` pair to its rank (index in
            ``_merges``).  Used for O(1) priority lookup during encoding.
        _trained: Whether the encoder has a vocabulary loaded.
    """

    def __init__(self) -> None:
        """Initialise an untrained encoder."""
        self.vocab_size: int = 0
        self._encoder: dict[str, int] = {}
        self._decoder: dict[int, str] = {}
        self._merges: list[tuple[str, str]] = []
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus_texts: list[str], vocab_size: int = 8192) -> None:
        """Learn BPE merge rules from a list of raw text strings.

        The training procedure is:
        1. Pre-tokenise each text into words (using the GPT-2 word-split
           pattern) and convert each word to a sequence of base byte
           characters.
        2. Count the frequency of every word form across the corpus.
        3. Iteratively find the most frequent adjacent symbol pair,
           record a new merge rule, and apply it to all word forms.
        4. Repeat until the vocabulary reaches ``vocab_size``.

        The resulting vocabulary is stored internally and can be persisted
        with ``save``.

        Args:
            corpus_texts: Raw text documents to learn the vocabulary from.
                Larger and more representative corpora produce better
                subword splits for the target domain.
            vocab_size: Target vocabulary size including the 6 special
                tokens and the 256 base byte symbols.  Must be at least
                262 (6 special + 256 base).  Defaults to 8192.

        Raises:
            ValueError: If ``vocab_size`` is less than 262.
        """
        if vocab_size < _N_SPECIAL + 256:
            raise ValueError(
                f"vocab_size must be at least {_N_SPECIAL + 256} "
                f"({_N_SPECIAL} special + 256 base bytes); got {vocab_size}."
            )

        # --- Step 1: build word frequency map ----------------------------
        # Each word is stored as a tuple of single-char base symbols so we
        # can efficiently replace pairs during merge steps.
        word_freqs: dict[tuple[str, ...], int] = defaultdict(int)
        for text in corpus_texts:
            for word in re.findall(_WORD_SPLIT_RE, text):
                chars = _bytes_to_chars(word.encode("utf-8"))
                word_freqs[tuple(chars)] += 1

        # --- Step 2: initialise vocabulary -------------------------------
        # Special tokens first, then the 256 base byte characters.
        vocab: dict[str, int] = {}
        for tok in ALL_SPECIAL_TOKENS:
            vocab[tok] = SPECIAL_TOKEN_TO_ID[tok]

        for byte_val in range(256):
            char = _BYTE_TO_CHAR[byte_val]
            vocab[char] = _BASE_OFFSET + byte_val

        merges: list[tuple[str, str]] = []
        n_merges_needed = vocab_size - len(vocab)

        # --- Step 3: iterative BPE merges --------------------------------
        for _ in range(n_merges_needed):
            pair_freqs = self._count_pairs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            merged_symbol = best_pair[0] + best_pair[1]
            vocab[merged_symbol] = len(vocab)
            merges.append(best_pair)
            word_freqs = self._apply_merge(word_freqs, best_pair, merged_symbol)

        # --- Step 4: store results ---------------------------------------
        self.vocab_size = len(vocab)
        self._encoder = vocab
        self._decoder = {v: k for k, v in vocab.items()}
        self._merges = merges
        self._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._trained = True

    @staticmethod
    def _count_pairs(
        word_freqs: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, str], int]:
        """Count the frequency of every adjacent symbol pair across all words.

        For a word ``("a", "b", "c")`` with frequency 3, this contributes
        3 to the count of pair ``("a", "b")`` and 3 to ``("b", "c")``.

        Args:
            word_freqs: Mapping from word (as a tuple of symbols) to its
                corpus frequency.

        Returns:
            A dict mapping each adjacent pair to its total weighted frequency.
        """
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        return pair_freqs

    @staticmethod
    def _apply_merge(
        word_freqs: dict[tuple[str, ...], int],
        pair: tuple[str, str],
        merged: str,
    ) -> dict[tuple[str, ...], int]:
        """Apply one merge rule to every word in the frequency map.

        Replaces every occurrence of ``pair`` within word tuples with the
        single ``merged`` symbol, preserving word frequencies.

        Args:
            word_freqs: Current word-frequency map (tuple of symbols → count).
            pair: The adjacent symbol pair to merge, e.g. ``("h", "e")``.
            merged: The new symbol to replace the pair with, e.g. ``"he"``.

        Returns:
            A new word-frequency map with the merge applied everywhere.
        """
        new_word_freqs: dict[tuple[str, ...], int] = {}
        a, b = pair
        for word, freq in word_freqs.items():
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of integer token ids.

        The text is first split into words using the same pattern used
        during training. Each word is converted to base byte characters
        and then BPE merges are applied greedily in the order they were
        learned (lowest rank = highest priority).

        Special tokens embedded literally in ``text`` (e.g. ``"<BOS>"``)
        are detected before byte-conversion and inserted as their reserved
        ids directly.

        Args:
            text: Raw input string.  May contain special tokens in their
                literal string form (e.g. ``"<SEP>"``).

        Returns:
            A list of integer ids representing the tokenized text.

        Raises:
            RuntimeError: If the encoder has not been trained or loaded.
        """
        self._require_trained()
        ids: list[int] = []

        # Split on special tokens first so they pass through unchanged.
        parts = self._split_on_special_tokens(text)
        for part in parts:
            if part in SPECIAL_TOKEN_TO_ID:
                ids.append(SPECIAL_TOKEN_TO_ID[part])
            else:
                for word in re.findall(_WORD_SPLIT_RE, part):
                    ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single pre-tokenised word to a list of BPE ids.

        Converts the word to base byte characters, then applies all merge
        rules in priority order using a priority-queue approach that
        processes the highest-priority (lowest-rank) applicable merge first.

        Args:
            word: A single word string (may include a leading space, as
                produced by the GPT-2 split pattern).

        Returns:
            A list of integer ids for this word's BPE encoding.
        """
        import heapq

        chars = _bytes_to_chars(word.encode("utf-8"))
        if len(chars) == 1:
            return [self._encoder[chars[0]]]

        # Represent the word as a doubly-linked list via parallel prev/next
        # arrays for O(1) neighbour updates during merges.
        # For simplicity at this scale, we use a list and rebuild pairs.
        symbols = list(chars)

        while len(symbols) > 1:
            # Find the merge with the lowest rank among all adjacent pairs.
            best_rank = len(self._merges)  # sentinel: no valid merge found
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_ranks.get(pair, len(self._merges))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break # no applicable merge remains

            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]

        return [self._encoder[sym] for sym in symbols]

    @staticmethod
    def _split_on_special_tokens(text: str) -> list[str]:
        """Split text into segments, isolating any literal special tokens.

        For example, ``"hello <SEP> world"`` becomes
        ``["hello ", "<SEP>", " world"]``.

        Args:
            text: Raw input that may contain special token strings.

        Returns:
            A list of string segments. Special tokens appear as isolated
            elements; all other text appears as plain segments between them.
        """
        pattern = "(" + "|".join(re.escape(t) for t in ALL_SPECIAL_TOKENS) + ")"
        return [part for part in re.split(pattern, text) if part]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string.

        Each id is converted back to its string symbol via the decoder
        dict. Special token symbols are passed through as their literal
        strings (e.g. id 3 → ``"<SEP>"``).  All other symbols are
        byte-character sequences that are converted back to UTF-8 bytes
        and then decoded to a string.

        Args:
            ids: A list of integer token ids as returned by ``encode``.

        Returns:
            The decoded string. Guaranteed to round-trip losslessly for
            any input that does not contain literal special token strings.

        Raises:
            RuntimeError: If the encoder has not been trained or loaded.
        """
        self._require_trained()
        text_parts: list[str] = []
        byte_buf: list[int] = []

        def _flush_bytes() -> None:
            if byte_buf:
                text_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for token_id in ids:
            symbol = self._decoder.get(token_id)
            if symbol is None:
                continue
            if symbol in SPECIAL_TOKEN_TO_ID:
                _flush_bytes()
                text_parts.append(symbol)
            else:
                for char in symbol:
                    byte_buf.append(_CHAR_TO_BYTE[char])

        _flush_bytes()
        return "".join(text_parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the trained vocabulary and merge rules to a JSON file.

        The file contains three keys:
        - ``"vocab"``: a dict mapping symbol string to integer id.
        - ``"merges"``: a list of ``[a, b]`` pairs in training order.
        - ``"vocab_size"``: the total vocabulary size as an integer.

        Args:
            path: File path to write. Parent directories are created if
                they do not exist.

        Raises:
            RuntimeError: If the encoder has not been trained or loaded.
        """
        self._require_trained()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self._encoder,
            "merges": list(self._merges),
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BytePairEncoder":
        """Load a previously saved encoder from a JSON file.

        Args:
            path: File path to read. Must be a file written by ``save``.

        Returns:
            A fully initialised ``BytePairEncoder`` ready to encode and
            decode text.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls()
        enc.vocab_size = data["vocab_size"]
        enc._encoder = {k: int(v) for k, v in data["vocab"].items()}
        enc._decoder = {int(v): k for k, v in data["vocab"].items()}
        enc._merges = [tuple(pair) for pair in data["merges"]]
        enc._merge_ranks = {
            tuple(pair): rank for rank, pair in enumerate(enc._merges)
        }
        enc._trained = True
        return enc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        """Raise RuntimeError if the encoder has no vocabulary loaded.

        Args: None.

        Raises:
            RuntimeError: If neither ``train`` nor ``load`` has been called.
        """
        if not self._trained:
            raise RuntimeError(
                "BytePairEncoder has no vocabulary.  Call train() or load() first."
            )

    def __len__(self) -> int:
        """Return the vocabulary size.

        Returns:
            The number of symbols in the vocabulary, including special tokens
            and base byte symbols.
        """
        return self.vocab_size
