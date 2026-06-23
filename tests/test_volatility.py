"""Tests for Phase 1 volatility: range estimators + GARCH/EWMA/HAR scale."""

import numpy as np
import pandas as pd
import pytest

from src.data_engineering import add_range_volatility
from src.volatility import (
    ewma_conditional_sigma,
    fit_har,
    garch_conditional_sigma,
    har_predict,
)


# ─────────────────────────────────────────────
# Range-based volatility estimators
# ─────────────────────────────────────────────

def _ohlc(n, high, low, open_, close, seed=0):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "CL=F_high": np.full(n, float(high)),
            "CL=F_low": np.full(n, float(low)),
            "CL=F_open": np.full(n, float(open_)),
            "CL=F_close": np.full(n, float(close)),
        },
        index=idx,
    )


class TestRangeVolatility:
    def test_parkinson_matches_closed_form(self):
        df = _ohlc(60, high=72.0, low=68.0, open_=70.0, close=71.0)
        out = add_range_volatility(
            df, "CL=F_high", "CL=F_low", "CL=F_open", "CL=F_close", windows=[5]
        )
        expected = np.sqrt((np.log(72.0 / 68.0) ** 2) / (4.0 * np.log(2.0)))
        got = out["CL=F_close_parkinson_5"].dropna().iloc[-1]
        assert abs(got - expected) < 1e-9

    def test_flat_ohlc_gives_zero_vol(self):
        df = _ohlc(40, high=70.0, low=70.0, open_=70.0, close=70.0)
        out = add_range_volatility(
            df, "CL=F_high", "CL=F_low", "CL=F_open", "CL=F_close", windows=[5]
        )
        for est in ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
            col = out[f"CL=F_close_{est}_5"].dropna()
            assert np.allclose(col.values, 0.0, atol=1e-12)

    def test_all_columns_created_and_nonnegative(self):
        rng = np.random.default_rng(0)
        n = 120
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 70 + rng.standard_normal(n).cumsum()
        df = pd.DataFrame(
            {
                "CL=F_open": close + rng.normal(0, 0.2, n),
                "CL=F_high": close + np.abs(rng.normal(0.5, 0.2, n)),
                "CL=F_low": close - np.abs(rng.normal(0.5, 0.2, n)),
                "CL=F_close": close,
            },
            index=idx,
        )
        out = add_range_volatility(
            df, "CL=F_high", "CL=F_low", "CL=F_open", "CL=F_close", windows=[5, 20]
        )
        for w in (5, 20):
            for est in ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
                col = out[f"CL=F_close_{est}_{w}"].dropna()
                assert (col.values >= 0).all()
                assert len(col) > 0

    def test_missing_column_skips_gracefully(self):
        df = pd.DataFrame({"CL=F_close": [70.0, 71.0]})
        out = add_range_volatility(
            df, "CL=F_high", "CL=F_low", "CL=F_open", "CL=F_close"
        )
        assert "CL=F_close_parkinson_5" not in out.columns


# ─────────────────────────────────────────────
# EWMA / GARCH conditional volatility
# ─────────────────────────────────────────────

class TestEwma:
    def test_is_causal(self):
        # σ_t must depend only on residuals strictly before t: changing the
        # last residual cannot change any earlier σ.
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 300)
        s1, _ = ewma_conditional_sigma(r)
        r2 = r.copy()
        r2[-1] += 5.0
        s2, _ = ewma_conditional_sigma(r2)
        np.testing.assert_allclose(s1[:-1], s2[:-1], rtol=1e-12)

    def test_tracks_regime_shift(self):
        # Calm regime then turbulent regime ⇒ EWMA σ rises in the second half.
        rng = np.random.default_rng(2)
        calm = rng.normal(0, 0.005, 300)
        wild = rng.normal(0, 0.05, 300)
        r = np.concatenate([calm, wild])
        sigma, _ = ewma_conditional_sigma(r)
        assert sigma[500:].mean() > 3 * sigma[:250].mean()


class TestGarch:
    def test_recovers_clustering(self):
        # Simulate a GARCH(1,1) process; the filtered conditional σ should
        # correlate with the true latent σ.
        rng = np.random.default_rng(3)
        n = 1500
        omega, alpha, beta = 0.05, 0.10, 0.85
        sig2 = np.empty(n)
        r = np.empty(n)
        sig2[0] = omega / (1 - alpha - beta)
        for t in range(n):
            if t > 0:
                sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
            r[t] = np.sqrt(sig2[t]) * rng.standard_normal()
        r = r / 100.0  # bring into log-return scale
        cond = garch_conditional_sigma(r)
        assert cond.kind in ("garch", "ewma")
        assert np.isfinite(cond.forecast) and cond.forecast > 0
        corr = np.corrcoef(cond.sigma, np.sqrt(sig2) / 100.0)[0, 1]
        assert corr > 0.4

    def test_short_series_falls_back_to_ewma(self):
        rng = np.random.default_rng(4)
        r = rng.normal(0, 0.01, 50)  # < 100 ⇒ GARCH skipped
        cond = garch_conditional_sigma(r)
        assert cond.kind == "ewma"
        assert cond.sigma.size == 50


class TestHar:
    def test_fit_and_predict_finite(self):
        rng = np.random.default_rng(5)
        # Persistent realised-variance series.
        rv = np.empty(400)
        rv[0] = 1.0
        for t in range(1, 400):
            rv[t] = max(0.02 + 0.9 * rv[t - 1] + rng.normal(0, 0.05), 1e-4)
        model = fit_har(rv)
        assert model["coef"].shape == (4,)
        pred = har_predict(model, rv)
        assert np.isfinite(pred) and pred >= 0
