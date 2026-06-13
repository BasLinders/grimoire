"""Statistically grounded early stopping for training.

Naive early stopping ("stop when validation loss stops improving") is fooled
by noise: a validation loss estimated from a finite held-out set is itself a
random quantity, so a small dip or bump between two evaluation points may be
pure sampling noise rather than real progress.

This module quantifies that noise with a non-parametric bootstrap.  At each
evaluation we have the per-batch validation losses; resampling them with
replacement many times and taking the spread of the resampled means gives a
confidence interval for the *true* mean validation loss.  We then treat an
apparent improvement as real only when it exceeds the half-width of that
interval — i.e. when it is larger than the noise we would expect from the
finite validation set.  Training stops once that condition fails for
``patience`` consecutive evaluations.

The approach needs no SciPy: ``numpy`` alone provides the resampling and the
empirical percentiles.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci_halfwidth(
    losses: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """Estimate the half-width of a bootstrap CI for the mean of ``losses``.

    Resamples ``losses`` with replacement ``n_boot`` times, computes the mean of
    each resample, and returns half the width of the central ``1 - alpha``
    percentile interval of those means.  This is a noise scale: an observed
    change in mean validation loss smaller than this half-width is statistically
    indistinguishable from sampling noise.

    Args:
        losses: Per-batch validation losses from a single evaluation pass.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level; the interval covers ``1 - alpha`` of the
            bootstrap means (e.g. ``alpha=0.05`` → a 95% interval).
        seed: RNG seed for reproducibility.

    Returns:
        The CI half-width as a non-negative float.  Returns ``0.0`` when fewer
        than two losses are supplied (no spread can be estimated).
    """
    arr = np.asarray(list(losses), dtype=np.float64)
    if arr.size < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    # (n_boot, n) index matrix → resampled means in one vectorised pass.
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)

    lo = np.percentile(boot_means, 100.0 * (alpha / 2.0))
    hi = np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))
    return float((hi - lo) / 2.0)


class EarlyStopper:
    """Decide when validation loss has stopped improving beyond the noise.

    Feed each evaluation's per-batch losses to :meth:`update`.  It tracks the
    best mean loss seen so far and counts consecutive evaluations whose
    improvement over that best is within the bootstrap noise band.  Once that
    count reaches ``patience`` the run should stop.

    Attributes:
        patience: Number of consecutive non-improving evaluations tolerated
            before signalling a stop.
        best_loss: Lowest mean validation loss observed so far.
        num_bad_evals: Consecutive evaluations without a significant improvement.
    """

    def __init__(
        self,
        patience: int = 3,
        n_boot: int = 1000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> None:
        """Initialise the stopper.

        Args:
            patience: Consecutive non-improving evaluations tolerated before
                :meth:`update` returns ``True``.
            n_boot: Bootstrap resamples per evaluation.
            alpha: Significance level for the bootstrap CI.
            seed: Base RNG seed; each evaluation perturbs it so successive
                bootstraps are not identically correlated.
        """
        self.patience = patience
        self.n_boot = n_boot
        self.alpha = alpha
        self.seed = seed
        self.best_loss = float("inf")
        self.num_bad_evals = 0
        self._eval_count = 0

    def update(self, batch_losses: Sequence[float]) -> bool:
        """Record one evaluation and report whether training should stop.

        Args:
            batch_losses: Per-batch validation losses from this evaluation pass.

        Returns:
            ``True`` if validation loss has failed to improve beyond the noise
            band for ``patience`` consecutive evaluations, else ``False``.
        """
        arr = np.asarray(list(batch_losses), dtype=np.float64)
        if arr.size == 0:
            return False

        mean_loss = float(arr.mean())
        halfwidth = bootstrap_ci_halfwidth(
            arr, n_boot=self.n_boot, alpha=self.alpha, seed=self.seed + self._eval_count
        )
        self._eval_count += 1

        # An improvement counts only if it clears the noise band.
        if mean_loss < self.best_loss - halfwidth:
            self.best_loss = mean_loss
            self.num_bad_evals = 0
        else:
            # Still track the running best even on a "noisy" improvement so the
            # threshold tightens as the model genuinely gets better.
            self.best_loss = min(self.best_loss, mean_loss)
            self.num_bad_evals += 1

        return self.num_bad_evals >= self.patience
