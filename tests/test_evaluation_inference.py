"""Tests for Phase 0 benchmark-relative inference (src.evaluation).

Covers Diebold-Mariano (HLN), Pesaran-Timmermann, MSPE ratio, and the
Berkowitz density-calibration LR test.  All use synthetic data with known
properties so the expected verdict is unambiguous.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.cpcv_backtest import deflated_sharpe, expected_max_sharpe, pbo_cscv
from src.evaluation import (
    berkowitz_lr,
    compute_benchmark_tests,
    diebold_mariano,
    model_confidence_set,
    mspe_ratio,
    pesaran_timmermann,
    spa_test,
)


class TestDieboldMariano:
    def test_identical_forecasts_not_significant(self):
        rng = np.random.default_rng(0)
        loss = rng.normal(1.0, 0.3, 500)
        stat, p = diebold_mariano(loss, loss.copy())
        # Zero loss differential ⇒ degenerate variance ⇒ NaN, OR ~0 stat.
        assert np.isnan(stat) or abs(stat) < 1e-6

    def test_model_better_is_negative_and_significant(self):
        rng = np.random.default_rng(1)
        loss_model = rng.normal(1.0, 0.2, 600)
        # Benchmark loss is systematically larger (model is better).
        loss_naive = loss_model + np.abs(rng.normal(0.3, 0.05, 600))
        stat, p = diebold_mariano(loss_model, loss_naive)
        assert stat < 0.0           # convention: A (model) better ⇒ negative
        assert p < 0.05

    def test_too_few_obs_returns_nan(self):
        stat, p = diebold_mariano(np.array([1.0, 2.0]), np.array([1.5, 1.0]))
        assert np.isnan(stat) and np.isnan(p)

    def test_horizon_widens_variance(self):
        rng = np.random.default_rng(2)
        d = rng.normal(0.05, 1.0, 800)
        loss_a = np.zeros_like(d)
        loss_b = -d  # so loss_a - loss_b = d
        stat_h1, _ = diebold_mariano(loss_a, loss_b, h=1)
        stat_h5, _ = diebold_mariano(loss_a, loss_b, h=5)
        # Both finite; the statistic magnitude generally shrinks as h grows
        # because the long-run variance accumulates autocovariance terms.
        assert np.isfinite(stat_h1) and np.isfinite(stat_h5)


class TestPesaranTimmermann:
    def test_directional_edge_is_significant(self):
        rng = np.random.default_rng(3)
        n = 800
        actual = rng.normal(0, 1, n)
        pred = 0.6 * actual + rng.normal(0, 0.8, n)  # genuine sign skill
        stat, p = pesaran_timmermann(actual, pred)
        assert stat > 0
        assert p < 0.05

    def test_no_edge_not_significant(self):
        rng = np.random.default_rng(4)
        n = 800
        actual = rng.normal(0, 1, n)
        pred = rng.normal(0, 1, n)  # independent of actual
        stat, p = pesaran_timmermann(actual, pred)
        assert p > 0.05


class TestMspeRatio:
    def test_equal_forecasts_ratio_one(self):
        rng = np.random.default_rng(5)
        y = rng.normal(70, 5, 300)
        pred = y + rng.normal(0, 1, 300)
        assert abs(mspe_ratio(y, pred, pred) - 1.0) < 1e-12

    def test_better_model_below_one(self):
        rng = np.random.default_rng(6)
        y = rng.normal(70, 5, 300)
        good = y + rng.normal(0, 0.5, 300)
        naive = y + rng.normal(0, 2.0, 300)
        assert mspe_ratio(y, good, naive) < 1.0


class TestBerkowitz:
    def test_calibrated_pit_not_rejected(self):
        rng = np.random.default_rng(7)
        pit = rng.uniform(0.0, 1.0, 1000)
        lr, p = berkowitz_lr(pit)
        assert p > 0.05

    def test_overdispersed_pit_rejected(self):
        rng = np.random.default_rng(8)
        # True innovations have variance 4 but the forecaster assumed N(0,1):
        # PITs computed under the wrong (too-narrow) scale are U-shaped.
        z_true = rng.normal(0, 2.0, 1000)
        pit_bad = stats.norm.cdf(z_true)  # treats variance-4 draws as N(0,1)
        lr, p = berkowitz_lr(pit_bad)
        assert p < 0.05

    def test_too_few_pits_returns_nan(self):
        lr, p = berkowitz_lr(np.array([0.1, 0.5, 0.9]))
        assert np.isnan(lr) and np.isnan(p)


class TestComputeBenchmarkTests:
    def test_bundle_keys_present(self):
        rng = np.random.default_rng(9)
        n = 400
        anchor = np.full(n, 100.0)
        actual = 100.0 + rng.normal(0, 1, n)
        pred = 100.0 + 0.5 * (actual - 100.0) + rng.normal(0, 0.8, n)
        naive = anchor
        pit = rng.uniform(0, 1, n)
        out = compute_benchmark_tests(actual, pred, naive, anchor=anchor, pit=pit)
        for key in (
            "mspe_ratio", "dm_stat", "dm_pvalue",
            "pt_stat", "pt_pvalue", "berkowitz_lr", "berkowitz_pvalue",
        ):
            assert key in out

    def test_bundle_skips_optional_when_absent(self):
        rng = np.random.default_rng(10)
        y = rng.normal(70, 5, 200)
        pred = y + rng.normal(0, 1, 200)
        naive = np.roll(y, 1)
        out = compute_benchmark_tests(y[1:], pred[1:], naive[1:])
        assert "mspe_ratio" in out and "dm_stat" in out
        assert "pt_stat" not in out and "berkowitz_lr" not in out


class TestPBOCSCV:
    def test_all_noise_strategies_high_pbo(self):
        # With no genuine signal, picking the IS-best of N pure-noise strategies
        # is maximal overfitting: the favourable IS noise reverts OOS, so the
        # selected strategy lands below the OOS median more often than not ⇒
        # PBO > 0.5 (the canonical overfit signature).
        rng = np.random.default_rng(11)
        perf = rng.normal(0, 1, size=(480, 8))
        out = pbo_cscv(perf, n_partitions=10)
        assert out["pbo"] > 0.5
        assert out["n_strategies"] == 8

    def test_one_skilled_strategy_low_pbo(self):
        # One column has a genuine positive mean; it is selected IS and ranks
        # high OOS ⇒ PBO near 0.  Must be well below the all-noise case.
        rng = np.random.default_rng(12)
        perf = rng.normal(0, 1, size=(480, 8))
        perf[:, 3] += 0.5  # persistent edge
        out = pbo_cscv(perf, n_partitions=10)
        assert out["pbo"] < 0.20

    def test_single_strategy_returns_nan(self):
        out = pbo_cscv(np.random.default_rng(0).normal(0, 1, (100, 1)))
        assert np.isnan(out["pbo"]) and out["n_strategies"] == 1


class TestDeflatedSharpe:
    def test_high_sr_few_trials_is_significant(self):
        # Strong per-period Sharpe, few trials, long sample ⇒ DSR → 1.
        dsr = deflated_sharpe(sr_hat=0.20, sr_std_trials=0.03, n_trials=3,
                              skew=0.0, kurtosis=3.0, n_obs=1000)
        assert dsr > 0.95

    def test_modest_sr_many_trials_not_significant(self):
        # Same Sharpe but selected from many noisy trials ⇒ deflated away.
        dsr = deflated_sharpe(sr_hat=0.05, sr_std_trials=0.08, n_trials=500,
                              skew=0.0, kurtosis=3.0, n_obs=500)
        assert dsr < 0.5

    def test_expected_max_grows_with_trials(self):
        assert expected_max_sharpe(0.05, 100) > expected_max_sharpe(0.05, 5)


class TestModelConfidenceSet:
    def test_best_model_included_dominated_excluded(self):
        rng = np.random.default_rng(13)
        n = 400
        losses = pd.DataFrame({
            "good": np.abs(rng.normal(0.1, 0.05, n)),
            "bad_1": np.abs(rng.normal(1.0, 0.05, n)),
            "bad_2": np.abs(rng.normal(1.2, 0.05, n)),
        })
        out = model_confidence_set(losses, alpha=0.10, n_bootstrap=200, seed=0)
        assert "good" in out["included"]
        assert "bad_1" in out["excluded"] or "bad_2" in out["excluded"]

    def test_degenerate_single_model(self):
        out = model_confidence_set(pd.DataFrame({"only": [1.0, 2.0, 3.0]}))
        assert out["included"] == ["only"] and "note" in out


class TestSPA:
    def test_detects_model_beating_benchmark(self):
        rng = np.random.default_rng(14)
        n = 400
        benchmark = np.abs(rng.normal(1.0, 0.1, n))
        models = pd.DataFrame({
            "challenger": np.abs(rng.normal(0.5, 0.1, n)),  # clearly better
            "similar": np.abs(rng.normal(1.0, 0.1, n)),
        })
        out = spa_test(benchmark, models, n_bootstrap=200, seed=0)
        assert out["pvalue_consistent"] < 0.10
