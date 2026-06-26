"""Tests for Combinatorial Purged CV (`src.cv.CombinatorialPurgedCV`) and the
CPCV backtester (`src.cpcv_backtest.run_cpcv_backtest`)."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from src.cpcv_backtest import run_cpcv_backtest
from src.cv import CombinatorialPurgedCV


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_xy_t1(
    n: int = 240,
    horizon: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Build a small synthetic dataset compatible with the CPCV backtester.

    Returns (X, y, t1, target_dates, anchor, y_level).  ``y`` is a level-space
    target (so y_level == y), which lets the direction-strategy reduce to
    sign(y_pred - y_anchor).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    feature_dates = dates[: n - horizon] if horizon > 0 else dates
    label_dates = dates[horizon:] if horizon > 0 else dates

    # Random-walk-ish price series
    log_returns = rng.normal(0.0, 0.02, size=n).cumsum()
    price = 50.0 * np.exp(log_returns)
    X = rng.standard_normal((len(feature_dates), 3))
    anchor = price[: n - horizon] if horizon > 0 else price
    y_level = price[horizon:] if horizon > 0 else price
    y = y_level.copy()                                  # level-space target
    t1 = pd.Series(label_dates, index=feature_dates, name="t1")
    return X, y, t1, label_dates, anchor, y_level


# ─────────────────────────────────────────────────────────────────────────────
# CombinatorialPurgedCV — combinatorics
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinatorialPurgedCVCombinatorics:
    def test_n6_k2_yields_15_combinations_and_5_paths(self):
        """Book Section 12.4 example: N=6, k=2 ⇒ C(6,2)=15 combos and
        phi(6,2) = 2*15/6 = 5 backtest paths."""
        _X, _y, t1, *_ = _synthetic_xy_t1(n=120, horizon=1)
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, t1=t1, pct_embargo=0.0,
        )
        assert cv.get_n_combinations() == comb(6, 2) == 15
        assert cv.get_n_paths() == 5

    def test_path_assignment_covers_every_group_exactly_once_per_path(self):
        """Each backtest path must cover all N groups (one test fold per
        group), and the assignment grid must contain no -1 cells for k≥1."""
        _X, _y, t1, *_ = _synthetic_xy_t1(n=120, horizon=1)
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, t1=t1, pct_embargo=0.0,
        )
        paths = cv.get_paths(n_samples=120)
        assert paths.shape == (5, 6)
        assert (paths >= 0).all(), "Every (path, group) cell must be assigned."

    def test_path_assignment_each_combination_appears_correct_count(self):
        """Combination i covers k of the N groups; across all n_paths rows
        each combination index should appear exactly k times."""
        _X, _y, t1, *_ = _synthetic_xy_t1(n=120, horizon=1)
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_splits=2, t1=t1, pct_embargo=0.0,
        )
        paths = cv.get_paths(n_samples=120)
        # Each combination contributes its k test groups across the paths grid
        for combo_id in range(cv.get_n_combinations()):
            assert (paths == combo_id).sum() == cv.n_test_splits


# ─────────────────────────────────────────────────────────────────────────────
# Purging guarantee under multi-period labels
# ─────────────────────────────────────────────────────────────────────────────

class TestCPCVPurging:
    def test_no_train_row_overlaps_test_label_window(self):
        """With horizon=5 and pct_embargo=0, every train row's label end
        time must be strictly before the test span begins, OR sit after the
        whole test region (the embargo zone is degenerate here)."""
        X, _y, t1, *_ = _synthetic_xy_t1(n=300, horizon=5)
        cv = CombinatorialPurgedCV(
            n_splits=5, n_test_splits=2, t1=t1, pct_embargo=0.0,
        )
        for train_idx, test_idx, _combo_id in cv.split(X):
            test_t0 = t1.index[int(test_idx.min())]
            test_t1_max = t1.iloc[int(test_idx.max())]
            train_t1_vals = t1.iloc[train_idx].values
            # Vectorised purge guarantee: either label fully resolved before
            # test, or feature row sits after the test window.
            safe = (train_t1_vals < test_t0) | (train_idx > int(test_idx.max()))
            assert safe.all(), "CPCV leaked label-overlapping training rows."


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end backtester smoke test
# ─────────────────────────────────────────────────────────────────────────────

class _DummyDirectionModel:
    """Trivial model that predicts y_train.mean() for every test row.

    Keeps the CPCV smoke test independent of pyGAM's installation/state and
    fast enough to run in CI.  Predictions are constant per fit, which is
    fine — the test only checks plumbing, not statistical accuracy.
    """

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean, dtype=np.float64)


def _dummy_factory():
    return _DummyDirectionModel()


class TestCPCVBacktesterSmoke:
    def test_smoke_produces_path_distribution_and_finite_pbo(self):
        X, y, t1, target_dates, anchor, y_level = _synthetic_xy_t1(
            n=240, horizon=1, seed=1,
        )
        result = run_cpcv_backtest(
            X, y, t1, _dummy_factory,
            anchor=anchor,
            y_level=y_level,
            target_dates=target_dates,
            n_splits=4, n_test_splits=2,
            pct_embargo=0.0,
            n_jobs=1,
        )
        # 4 groups choose 2 → 6 combinations → phi = 2*6/4 = 3 paths
        assert result.summary["n_combinations"] == 6
        assert result.summary["n_paths"] == 3
        assert len(result.path_metrics) == 3

        # PBO is a probability in [0, 1] (or NaN if no paths — should not be here)
        assert 0.0 <= result.summary["pbo"] <= 1.0

        # Sharpe summary is finite
        assert np.isfinite(result.summary["sharpe_mean"])
        assert np.isfinite(result.summary["sharpe_std"])

        # Returns dataframe shaped (n_obs, n_paths) with the right index
        assert result.path_returns.shape == (len(target_dates), 3)
        assert (result.path_returns.index == target_dates).all()

    def test_smoke_writes_artifacts(self, tmp_path):
        X, y, t1, target_dates, anchor, y_level = _synthetic_xy_t1(
            n=180, horizon=1, seed=2,
        )
        out = tmp_path / "cpcv"
        result = run_cpcv_backtest(
            X, y, t1, _dummy_factory,
            anchor=anchor, y_level=y_level, target_dates=target_dates,
            n_splits=4, n_test_splits=2, pct_embargo=0.0,
            output_dir=out, n_jobs=1,
        )
        assert (out / "summary.json").exists()
        assert (out / "paths_metrics.parquet").exists()
        assert (out / "paths_returns.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Input-validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCPCVValidation:
    def test_raises_when_k_geq_n(self):
        _X, _y, t1, *_ = _synthetic_xy_t1(n=60, horizon=1)
        with pytest.raises(ValueError, match="n_test_splits"):
            CombinatorialPurgedCV(n_splits=4, n_test_splits=4, t1=t1)

    def test_raises_when_t1_not_series(self):
        with pytest.raises(TypeError, match="pandas Series"):
            CombinatorialPurgedCV(
                n_splits=4, n_test_splits=2, t1=np.arange(10),
            )
