from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexEntry:
    next_token: Optional[str]
    frequency: int = 1
    source: Optional[str] = None

    def increment(self):
        self.frequency += 1


class CorpusIndex:
    def __init__(self):
        self._store: dict[tuple[str, ...], IndexEntry] = {}

    def add(
        self,
        multi_token: tuple[str, ...],
        next_token: Optional[str] = None,
        source: Optional[str] = None,
    ):
        if multi_token in self._store:
            self._store[multi_token].increment()
        else:
            self._store[multi_token] = IndexEntry(next_token=next_token, source=source)

    def get(self, multi_token: tuple[str, ...]) -> Optional[IndexEntry]:
        return self._store.get(multi_token)

    def all_entries(self) -> dict[tuple[str, ...], IndexEntry]:
        return self._store

    def __len__(self) -> int:
        return len(self._store)
