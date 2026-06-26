"""Purged + embargoed cross-validation utilities (Lopez de Prado, Chapter 7).

This module isolates the canonical purge+embargo algorithm from
:mod:`src.model` so it can be reused by the standard CV loop and by the
combinatorial purged cross-validation (CPCV) backtester
(:mod:`src.cpcv_backtest`).

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
Chapter 7 (Cross-Validation in Finance) — purged k-fold, embargo.
Chapter 12 (Backtesting through Cross-Validation) — combinatorial purged CV.
"""

from __future__ import annotations

import itertools
import logging
from typing import Generator, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared purging helper
# ─────────────────────────────────────────────────────────────────────────────

def _purged_train_indices(
    n_samples: int,
    test_idx: np.ndarray,
    t1: pd.Series,
    embargo: int,
) -> np.ndarray:
    """Compute the purged + embargoed training index for a single test fold.

    Implements Snippet 7.3 from Lopez de Prado (2018):

    * **Purge.** Drop training observations whose label end time ``t1[train]``
      falls inside the test fold's time span ``[test_t0, test_t1_max]``.
    * **Embargo.** Drop the next ``embargo`` observations immediately after
      the last test observation, since their labels (computed from features
      that overlap the test region) leak post-test information back.

    Args:
        n_samples: Total number of observations.
        test_idx: Positional indices of the current test fold.
        t1: Series aligned 1-1 with the rows of ``X`` (positional, not
            label-based).  Values are the dates by which each label is
            fully observed.
        embargo: Number of observations to embargo after each test fold.

    Returns:
        Sorted 1-D numpy array of training-set positional indices.
    """
    if len(test_idx) == 0:
        return np.arange(n_samples)

    test_start_pos = int(test_idx.min())
    test_end_pos = int(test_idx.max())

    # Time-domain boundaries of the test fold.
    test_t0 = t1.index[test_start_pos]      # first feature-date inside test
    test_t1_max = t1.iloc[test_end_pos]      # last label-end inside test

    # All positional indices of the full sample.
    all_idx = np.arange(n_samples)

    # 1) Drop the test rows themselves.
    candidate_mask = np.ones(n_samples, dtype=bool)
    candidate_mask[test_idx] = False

    # 2) Embargo: drop the `embargo` rows immediately after the test fold.
    embargo_end = min(n_samples, test_end_pos + 1 + max(0, embargo))
    candidate_mask[test_end_pos + 1 : embargo_end] = False

    # 3) Purge: among the remaining candidates, keep only rows whose label
    #    end time is *strictly before* the test fold starts, OR whose feature
    #    row sits *after* the embargo region.  Rows whose label window
    #    overlaps the test span leak target information into training.
    train_idx_candidates = all_idx[candidate_mask]

    # Vectorised purge: a row at position p is safe iff
    #   t1[p] < test_t0           (label fully observed before test)
    #   OR  p >= embargo_end      (sits after the test+embargo region)
    t1_vals = t1.values[train_idx_candidates]
    pos = train_idx_candidates
    safe_mask = (t1_vals < test_t0) | (pos >= embargo_end)
    train_idx = train_idx_candidates[safe_mask]

    return train_idx


# ─────────────────────────────────────────────────────────────────────────────
# PurgedKFold (Chapter 7)
# ─────────────────────────────────────────────────────────────────────────────

class PurgedKFold(_BaseKFold):
    """K-fold CV with Lopez de Prado purging and bilateral embargo.

    Unlike the standard ``KFold``/``TimeSeriesSplit`` this class is *label-
    aware*: each row carries an end time ``t1`` (the date by which its label
    is fully observed).  Training rows whose label window overlaps the test
    fold are purged, and rows immediately after the test fold are embargoed.

    The folds themselves are contiguous and ordered (no shuffling), matching
    the time-series ordering of the data.

    Args:
        n_splits: Number of CV folds.
        t1: Series indexed by the *feature-row* timestamp (positional 1-1
            with ``X``) whose values are the label end times.  Must be
            passed at construction time so the same purging logic applies
            across every call to :meth:`split`.
        pct_embargo: Fraction of total observations to embargo after each
            test fold (e.g. 0.01 = 1%).  Set to 0 to disable the embargo.

    Notes:
        This class extends ``_BaseKFold`` so it is a drop-in replacement
        anywhere a scikit-learn CV splitter is expected.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.0,
    ) -> None:
        if t1 is None:
            raise ValueError(
                "PurgedKFold requires `t1` (label end times). "
                "Build it from `build_feature_matrix(...)`."
            )
        if not isinstance(t1, pd.Series):
            raise TypeError(f"t1 must be a pandas Series, got {type(t1).__name__}.")
        if pct_embargo < 0.0:
            raise ValueError(f"pct_embargo must be >= 0, got {pct_embargo}.")

        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = float(pct_embargo)

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X) if not isinstance(X, pd.DataFrame) else X.shape[0]
        if n_samples != len(self.t1):
            raise ValueError(
                f"X has {n_samples} rows but t1 has {len(self.t1)}. "
                "They must be aligned 1-1 (positional)."
            )

        embargo = int(n_samples * self.pct_embargo)

        # Contiguous folds, no shuffle — preserves time ordering.
        indices = np.arange(n_samples)
        fold_bounds = np.array_split(indices, self.n_splits)

        for fold_num, test_idx in enumerate(fold_bounds, start=1):
            if len(test_idx) == 0:
                logger.warning(f"PurgedKFold fold {fold_num}: empty test set, skipping.")
                continue

            train_idx = _purged_train_indices(n_samples, test_idx, self.t1, embargo)

            if len(train_idx) == 0:
                raise RuntimeError(
                    f"PurgedKFold fold {fold_num}: training set is empty after "
                    f"purge+embargo (embargo={embargo}). Reduce pct_embargo, "
                    "lower n_splits, or shorten label horizons."
                )

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# CombinatorialPurgedCV (Chapter 12)
# ─────────────────────────────────────────────────────────────────────────────

class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (CPCV) — Section 12.4.

    Partitions the sample into ``n_splits`` contiguous groups and forms all
    ``C(n_splits, n_test_splits)`` combinations where ``n_test_splits``
    groups serve as the test set.  Each combination receives the same
    purge+embargo treatment as :class:`PurgedKFold`.

    From the prediction grid one can reconstruct
    ``phi(n_splits, n_test_splits) = n_test_splits * C(n_splits, n_test_splits) / n_splits``
    distinct backtest paths (Figure 12.2 of the book), each of which is
    an independent series of out-of-sample predictions covering the full
    sample.

    Args:
        n_splits: Total number of contiguous groups (``N`` in the book).
        n_test_splits: Number of groups assigned to the test set per
            combination (``k`` in the book).  Typically 2.
        t1: Series of label end times, indexed by feature-row timestamp.
        pct_embargo: Fraction of total observations to embargo per test
            group block (default 0.01 = 1%).
    """

    def __init__(
        self,
        n_splits: int,
        n_test_splits: int,
        t1: pd.Series,
        pct_embargo: float = 0.01,
    ) -> None:
        if n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})."
            )
        if n_test_splits < 1:
            raise ValueError(f"n_test_splits must be >= 1, got {n_test_splits}.")
        if not isinstance(t1, pd.Series):
            raise TypeError(f"t1 must be a pandas Series, got {type(t1).__name__}.")
        if pct_embargo < 0.0:
            raise ValueError(f"pct_embargo must be >= 0, got {pct_embargo}.")

        self.n_splits = int(n_splits)
        self.n_test_splits = int(n_test_splits)
        self.t1 = t1
        self.pct_embargo = float(pct_embargo)

    # ── Internals ───────────────────────────────────────────────────────────

    def _group_bounds(self, n_samples: int) -> list[np.ndarray]:
        """Contiguous positional partitions of [0, n_samples)."""
        return [np.asarray(g) for g in np.array_split(np.arange(n_samples), self.n_splits)]

    def _combinations(self) -> list[tuple[int, ...]]:
        """All ``C(N, k)`` group-id combinations used as test sets."""
        return list(itertools.combinations(range(self.n_splits), self.n_test_splits))

    # ── Public API ──────────────────────────────────────────────────────────

    def get_n_combinations(self) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def get_n_paths(self) -> int:
        """Number of distinct backtest paths produced by the path-assembly."""
        return self.n_test_splits * self.get_n_combinations() // self.n_splits

    def split(
        self, X: np.ndarray | pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """Yield ``(train_idx, test_idx, combo_id)`` for every combination."""
        n_samples = len(X) if not isinstance(X, pd.DataFrame) else X.shape[0]
        if n_samples != len(self.t1):
            raise ValueError(
                f"X has {n_samples} rows but t1 has {len(self.t1)}."
            )
        embargo = int(n_samples * self.pct_embargo)
        groups = self._group_bounds(n_samples)

        for combo_id, combo in enumerate(self._combinations()):
            test_idx = np.concatenate([groups[g] for g in combo])
            test_idx.sort()

            train_idx = _purged_train_indices(n_samples, test_idx, self.t1, embargo)
            if len(train_idx) == 0:
                logger.warning(
                    f"CPCV combination {combo_id} (groups={combo}): empty train set "
                    "after purge+embargo — skipping."
                )
                continue
            yield train_idx, test_idx, combo_id

    def get_paths(self, n_samples: int) -> np.ndarray:
        """Map each (group, combination) cell to a backtest-path id.

        Returns:
            Array of shape ``(n_paths, n_splits)`` whose element ``[p, g]``
            is the *combination id* whose prediction on group ``g`` belongs
            to path ``p`` — or -1 when group ``g`` is not tested in any
            combination on path ``p``.  Each path covers all ``n_splits``
            groups (one out-of-sample prediction per group, taken from a
            single combination at a time).

        Notes:
            Implements the path-assembly rule from Figure 12.2: cycle
            through the combinations and, for each group, assign the
            combination's test-prediction to the next available path slot.
        """
        combos = self._combinations()
        n_paths = self.get_n_paths()
        paths = np.full((n_paths, self.n_splits), -1, dtype=np.int64)

        # For each group g, list the combinations that test it (in combo order).
        for g in range(self.n_splits):
            testing_combos = [ci for ci, c in enumerate(combos) if g in c]
            # Each group is tested in exactly k * C(N-1, k-1) combinations;
            # they must be evenly split across n_paths slots.
            if len(testing_combos) != n_paths:
                # Sanity: by construction k*C(N,k)/N = C(N-1, k-1) per group.
                raise AssertionError(
                    f"Group {g} tested in {len(testing_combos)} combinations, "
                    f"expected {n_paths} (paths)."
                )
            for path_id, ci in enumerate(testing_combos):
                paths[path_id, g] = ci

        return paths
