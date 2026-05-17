"""Calibration tests for predictive-distribution back-transformation.

The 2026-05-16 daily report showed empirical price-space coverage of ~5% on
a nominal 50% band — a 10x mis-calibration that pointed at a space-mixing
bug between log-return and price spaces in the backtest plumbing.  These
tests synthesise data where the truth is known so any regression in the
back-transform shows up immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_engineering import (
    FRED_PUBLICATION_LAG_DAYS,
    merge_datasets,
)
from src.evaluation import (
    PredictiveDistribution,
    empirical_coverage,
    fit_residual_distribution,
)


# ── Calibration: model-space → price-space round-trip ────────────────────

@pytest.mark.parametrize("width", [0.50, 0.80, 0.90, 0.95])
def test_log_return_pi_back_transforms_with_correct_coverage(width):
    """Bands built as anchor*exp(return-quantile) must cover the true price.

    Mirrors the per-fold band construction in src/backtest.py:248-265.  The
    earlier (buggy) call paths produced empirical coverage ~width/10; the
    Johnson SU back-transform should produce coverage within a few pp of
    nominal under Monte-Carlo error.
    """
    rng = np.random.default_rng(0)
    n = 5000
    sigma_r = 0.01  # ~1% daily log-return vol — realistic for WTI
    mu_r = 0.0

    anchor = 100.0 + rng.standard_normal(n).cumsum() * 0.1
    anchor = np.clip(anchor, 50.0, 200.0)
    returns_true = rng.normal(mu_r, sigma_r, size=n)
    prices_true = anchor * np.exp(returns_true)

    y_pred_model = np.full(n, mu_r)
    sigma_model = np.full(n, sigma_r)

    # Fit Johnson SU on a held-out half (no in-sample contamination).
    # tau_max=inf disables the fat-tail cap; we want a faithful fit to the
    # synthetic Normal residuals, not the production fat-tail prior.
    half = n // 2
    dist = fit_residual_distribution(
        returns_true[:half], y_pred_model[:half], sigma_model[:half],
        tau_max=np.inf,
    )

    q_lo, q_hi = (1 - width) / 2, (1 + width) / 2
    r_lo = dist.ppf_array(q_lo, y_pred_model[half:], sigma_model[half:])
    r_hi = dist.ppf_array(q_hi, y_pred_model[half:], sigma_model[half:])

    price_lo = anchor[half:] * np.exp(r_lo)
    price_hi = anchor[half:] * np.exp(r_hi)
    coverage = empirical_coverage(prices_true[half:], price_lo, price_hi)

    # Allow ~3pp tolerance for Monte-Carlo + Johnson SU vs true-Gaussian
    assert abs(coverage - width) < 0.03, (
        f"width={width:.2f}: empirical coverage {coverage:.3f} "
        f"deviates >3pp from nominal"
    )


def test_dist_global_fit_no_space_mixing():
    """fit_residual_distribution must be called with all args in one space.

    The pre-fix backtest summed a price-space residual (y_true - y_pred)
    into a model-space mean (y_pred_model), which is mathematically wrong.
    This test reproduces the model-space inputs the fixed backtest uses
    and confirms the resulting dist round-trips coverage correctly.
    """
    rng = np.random.default_rng(1)
    n = 4000
    anchor = 100.0 + rng.standard_normal(n).cumsum() * 0.05
    sigma_r = 0.012
    returns_true = rng.normal(0.0, sigma_r, size=n)
    prices_true = anchor * np.exp(returns_true)

    y_pred_model = np.zeros(n)
    sigma_model = np.full(n, sigma_r)

    # Correct path: y_true_model = log(price/anchor) — model-space target
    y_true_model = np.log(prices_true / anchor)
    dist = fit_residual_distribution(
        y_true_model, y_pred_model, sigma_model, tau_max=np.inf,
    )

    width = 0.80
    q_lo, q_hi = (1 - width) / 2, (1 + width) / 2
    r_lo = dist.ppf_array(q_lo, y_pred_model, sigma_model)
    r_hi = dist.ppf_array(q_hi, y_pred_model, sigma_model)
    cov_model = empirical_coverage(y_true_model, r_lo, r_hi)
    cov_price = empirical_coverage(prices_true, anchor * np.exp(r_lo), anchor * np.exp(r_hi))

    # Both spaces should agree to within Monte-Carlo error
    assert abs(cov_model - width) < 0.03
    assert abs(cov_price - width) < 0.03
    assert abs(cov_price - cov_model) < 0.02


def test_predictive_distribution_self_calibrates_on_gaussian():
    """Sanity check: PredictiveDistribution fitted on Normal residuals
    reproduces Normal coverage."""
    rng = np.random.default_rng(2)
    n = 5000
    sigma = 1.5
    mu_pred = np.zeros(n)
    sigma_pred = np.full(n, sigma)
    y_true = rng.normal(0.0, sigma, size=n)
    dist = fit_residual_distribution(y_true, mu_pred, sigma_pred, tau_max=np.inf)
    for width in (0.50, 0.90):
        q_lo, q_hi = (1 - width) / 2, (1 + width) / 2
        lo = dist.ppf_array(q_lo, mu_pred, sigma_pred)
        hi = dist.ppf_array(q_hi, mu_pred, sigma_pred)
        cov = empirical_coverage(y_true, lo, hi)
        assert abs(cov - width) < 0.03


# ── FRED publication-lag shifting (Step 1) ───────────────────────────────

def test_fred_publication_lag_shifts_monthly_series_forward():
    """`merge_datasets` must shift FRED monthly columns by their publication
    lag so the model on May 31 does not yet see May's unemployment value
    (released ~June 5 in reality)."""
    oil_dates = pd.date_range("2026-04-01", "2026-07-01", freq="B")
    oil_df = pd.DataFrame(
        {
            "CL=F_close": np.linspace(80, 90, len(oil_dates)),
        },
        index=oil_dates,
    )

    fred_dates = pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01"])
    fred_df = pd.DataFrame(
        {"unemployment": [3.9, 4.0, 4.1], "fed_funds": [5.25, 5.25, 5.50]},
        index=fred_dates,
    )

    merged = merge_datasets(oil_df, fred_df, eia_dfs={})

    # May's unemployment value (4.0) should NOT be present on 2026-05-15
    # (publication-lagged by 38 days → first visible ~2026-06-08).
    may15 = merged.loc["2026-05-15", "unemployment"]
    assert may15 == pytest.approx(3.9), (
        "May 15 should still see April's print (3.9), not May's (4.0)"
    )

    # By mid-June, May's value (4.0) should be visible.
    jun19 = merged.loc["2026-06-19", "unemployment"]
    assert jun19 == pytest.approx(4.0), (
        "Jun 19 should see May's print (4.0), publication-shifted from May 1"
    )

    # Daily series (fed_funds) is not shifted: 5.25 on May 15.
    assert merged.loc["2026-05-15", "fed_funds"] == pytest.approx(5.25)


def test_fred_publication_lag_can_be_disabled():
    """Passing an empty dict to `fred_publication_lags` disables shifting
    (lets tests exercise the legacy join path explicitly)."""
    oil_dates = pd.date_range("2026-04-01", "2026-07-01", freq="B")
    oil_df = pd.DataFrame(
        {"CL=F_close": np.ones(len(oil_dates))},
        index=oil_dates,
    )
    fred_df = pd.DataFrame(
        {"unemployment": [3.9, 4.0, 4.1]},
        index=pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01"]),
    )
    merged_default = merge_datasets(oil_df, fred_df, eia_dfs={})
    merged_disabled = merge_datasets(
        oil_df, fred_df, eia_dfs={}, fred_publication_lags={}
    )
    # With shifting on, May 15 sees April's value (3.9); with shifting off,
    # May 15 sees May 1's value (4.0).
    assert merged_default.loc["2026-05-15", "unemployment"] == pytest.approx(3.9)
    assert merged_disabled.loc["2026-05-15", "unemployment"] == pytest.approx(4.0)


def test_fred_lag_map_covers_known_monthly_series():
    """Regression guard: any series the report flagged as a likely leakage
    artefact (unemployment, cpi, pmi_mfg, etc.) must have a positive lag."""
    must_be_shifted = ["unemployment", "cpi", "indpro", "pmi_mfg", "cap_util"]
    for k in must_be_shifted:
        assert FRED_PUBLICATION_LAG_DAYS.get(k, 0) > 0, (
            f"FRED series '{k}' is monthly and must have a publication lag > 0"
        )
