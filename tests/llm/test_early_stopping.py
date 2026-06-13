"""Tests for bootstrap-based early stopping.

Gate criteria:
- CI half-width is 0 for <2 samples and for identical samples, positive for spread.
- EarlyStopper stops after `patience` flat evaluations.
- EarlyStopper keeps going while genuine (beyond-noise) improvement continues.
"""

import numpy as np

from grimoire_ai.llm.training.early_stopping import (
    EarlyStopper,
    bootstrap_ci_halfwidth,
)


def test_halfwidth_zero_for_insufficient_or_constant_samples() -> None:
    assert bootstrap_ci_halfwidth([]) == 0.0
    assert bootstrap_ci_halfwidth([1.5]) == 0.0
    # Identical values have no spread → all bootstrap means equal → 0 width.
    assert bootstrap_ci_halfwidth([2.0, 2.0, 2.0, 2.0]) == 0.0


def test_halfwidth_positive_and_shrinks_with_n() -> None:
    rng = np.random.default_rng(0)
    small = rng.normal(0.0, 1.0, size=20).tolist()
    large = rng.normal(0.0, 1.0, size=2000).tolist()
    hw_small = bootstrap_ci_halfwidth(small, seed=1)
    hw_large = bootstrap_ci_halfwidth(large, seed=1)
    assert hw_small > 0.0
    # More data → tighter CI for the mean.
    assert hw_large < hw_small


def test_stopper_triggers_after_patience_flat_evals() -> None:
    stopper = EarlyStopper(patience=2, n_boot=200)
    flat = [1.0, 1.0, 1.0, 1.0]   # zero spread → halfwidth 0
    # First eval establishes the best (improvement vs inf).
    assert stopper.update(flat) is False
    # Second eval: no improvement → bad #1.
    assert stopper.update(flat) is False
    # Third eval: no improvement → bad #2 == patience → stop.
    assert stopper.update(flat) is True


def test_stopper_resets_on_real_improvement() -> None:
    stopper = EarlyStopper(patience=2, n_boot=200)
    # Steadily, clearly decreasing means (each well below the previous).
    assert stopper.update([5.0, 5.0, 5.0, 5.0]) is False
    assert stopper.update([4.0, 4.0, 4.0, 4.0]) is False
    assert stopper.update([3.0, 3.0, 3.0, 3.0]) is False
    assert stopper.update([2.0, 2.0, 2.0, 2.0]) is False
    assert stopper.best_loss == 2.0
    assert stopper.num_bad_evals == 0
