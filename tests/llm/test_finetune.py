"""Tests for the fine-tuning train/validation split helper.

Gate criteria:
- val_split <= 0 returns the original dataset and no validation set.
- A positive split partitions examples into disjoint train/val subsets.
- The split is reproducible (seeded) and always keeps >= 1 example per side.
- Datasets too small to split are returned whole with no validation set.
"""

from torch.utils.data import Dataset

from grimoire_ai.llm.training.finetune import split_dataset


class _RangeDataset(Dataset):
    """Minimal indexable dataset of integers 0..n-1 for split tests."""

    def __init__(self, n: int) -> None:
        self._items = list(range(n))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> int:
        return self._items[idx]


def _values(subset) -> set[int]:
    """Collect the underlying integer values reachable through a subset."""
    return {subset[i] for i in range(len(subset))}


def test_no_split_when_val_split_zero() -> None:
    ds = _RangeDataset(10)
    train, val = split_dataset(ds, val_split=0.0)
    assert train is ds
    assert val is None


def test_split_partitions_disjointly() -> None:
    ds = _RangeDataset(10)
    train, val = split_dataset(ds, val_split=0.2)
    assert len(train) == 8
    assert len(val) == 2
    train_vals, val_vals = _values(train), _values(val)
    assert train_vals.isdisjoint(val_vals)
    assert train_vals | val_vals == set(range(10))


def test_split_is_reproducible() -> None:
    ds = _RangeDataset(20)
    _, val_a = split_dataset(ds, val_split=0.25, seed=7)
    _, val_b = split_dataset(ds, val_split=0.25, seed=7)
    assert _values(val_a) == _values(val_b)


def test_split_keeps_at_least_one_each_side() -> None:
    # A tiny fraction that rounds to zero must still hold out one example.
    ds = _RangeDataset(5)
    train, val = split_dataset(ds, val_split=0.01)
    assert len(val) == 1
    assert len(train) == 4


def test_split_clamps_large_fraction() -> None:
    # A fraction that would consume the whole set must leave one for training.
    ds = _RangeDataset(4)
    train, val = split_dataset(ds, val_split=1.0)
    assert len(train) == 1
    assert len(val) == 3


def test_single_example_dataset_is_not_split() -> None:
    ds = _RangeDataset(1)
    train, val = split_dataset(ds, val_split=0.5)
    assert train is ds
    assert val is None
