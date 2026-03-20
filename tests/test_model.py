"""Tests for GAM model functions."""

import numpy as np
import pandas as pd
import pytest

from src.config_loader import resolve_model_config
from src.model import (
    _classify_feature,
    build_feature_matrix,
    compute_bic,
    compute_rolling_nu_tau,
    cross_validate,
    define_gam_terms,
    fit_distributional_gam,
    fit_gam,
    fit_sigma_gam,
    select_sigma_features,
    select_nu_tau_features,
    stepwise_aic_selection,
)


@pytest.fixture
def sample_feature_df():
    """Create a synthetic feature DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2023-01-02", periods=n, freq="B")

    price = 70 + rng.standard_normal(n).cumsum()
    df = pd.DataFrame(
        {
            "CL=F_close": price,
            "CL=F_open": price + rng.standard_normal(n) * 0.5,
            "CL=F_volume": rng.integers(1000, 5000, n),
            "usd_index": 100 + rng.standard_normal(n).cumsum() * 0.3,
            "fed_funds": 5.0 + rng.standard_normal(n) * 0.1,
            "CL=F_close_lag_1": np.roll(price, 1),
            "CL=F_close_lag_7": np.roll(price, 7),
            "CL=F_close_sma_14": pd.Series(price).rolling(14).mean().values,
            "CL=F_close_std_14": pd.Series(price).rolling(14).std().values,
            "CL=F_close_pct_change": pd.Series(price).pct_change().values,
            "day_of_week": [d.dayofweek for d in dates],
            "month": [d.month for d in dates],
            "crude_stocks": 400_000 + rng.standard_normal(n).cumsum() * 500,
        },
        index=dates,
    )
    return df.dropna()


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "features": {"target": "CL=F_close"},
        "model": {"n_splines": 10, "cv_splits": 3, "lam_search": [0.1, 1, 10]},
    }


class TestClassifyFeature:
    def test_macro_features(self):
        assert _classify_feature("usd_index") == "linear"
        assert _classify_feature("cpi") == "linear"
        assert _classify_feature("fed_funds") == "linear"
        assert _classify_feature("t10y2y") == "linear"

    def test_cyclic_features(self):
        assert _classify_feature("day_of_week") == "cyclic"
        assert _classify_feature("month") == "cyclic"
        assert _classify_feature("day_of_year") == "cyclic"

    def test_rolling_features(self):
        assert _classify_feature("CL=F_close_sma_14") == "rolling"
        assert _classify_feature("CL=F_close_std_50") == "rolling"

    def test_lag_features(self):
        assert _classify_feature("CL=F_close_lag_1") == "lag"
        assert _classify_feature("CL=F_close_lag_30") == "lag"

    def test_return_features(self):
        assert _classify_feature("CL=F_close_pct_change") == "return"
        assert _classify_feature("CL=F_close_log_return") == "return"

    def test_default_spline(self):
        assert _classify_feature("crude_stocks") == "spline"
        assert _classify_feature("some_unknown_feature") == "spline"


class TestBuildFeatureMatrix:
    def test_excludes_target(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert "CL=F_close" not in names

    def test_excludes_raw_ohlcv(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert "CL=F_open" not in names
        assert "CL=F_volume" not in names

    def test_output_shapes(self, sample_feature_df, sample_config):
        X, y, names, target_dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        # Default horizon=1 drops the last row (target unknown)
        assert X.shape[0] == len(sample_feature_df) - 1
        assert X.shape[1] == len(names)
        assert y.shape[0] == len(sample_feature_df) - 1
        assert len(target_dates) == X.shape[0]

    def test_output_shapes_horizon_zero(self, sample_feature_df, sample_config):
        X, y, names, target_dates = build_feature_matrix(
            sample_feature_df, "CL=F_close", forecast_horizon=0
        )
        assert X.shape[0] == len(sample_feature_df)
        assert y.shape[0] == len(sample_feature_df)
        assert len(target_dates) == len(sample_feature_df)

    def test_target_dates_shifted(self, sample_feature_df, sample_config):
        X, y, names, target_dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        # target_dates should be the actual next trading day from the index
        assert target_dates[0] == sample_feature_df.index[1]
        assert target_dates[-1] == sample_feature_df.index[-1]

    def test_feature_names_list(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


class TestDefineGamTerms:
    def test_returns_terms_object(self, sample_feature_df, sample_config):
        _, _, names, _ = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        assert terms is not None


class TestFitGam:
    def test_fit_on_synthetic_data(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[0.1, 1])
        predictions = gam.predict(X)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))


class TestCrossValidate:
    def test_cv_returns_correct_structure(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        results = cross_validate(X, y, terms, n_splits=3, embargo_days=5, lam_values=[0.1, 1])

        assert "fold_metrics" in results
        assert "mean_mae" in results
        assert "mean_rmse" in results
        assert "mean_mape" in results
        assert len(results["fold_metrics"]) == 3
        assert results["mean_mae"] > 0
        assert results["mean_rmse"] > 0


class TestComputeBic:
    def test_bic_greater_than_aic_for_large_n(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[0.1, 1])
        bic = compute_bic(gam, len(y))
        aic = gam.statistics_["AIC"]
        # BIC = AIC + edof * (log(n) - 2); for n > ~7.4, log(n) > 2 so BIC > AIC
        assert bic >= aic

    def test_bic_returns_float(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[1])
        assert isinstance(compute_bic(gam, len(y)), float)


class TestStepwiseBIC:
    def test_bic_stepwise_reduces_features(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        n_orig = len(names)
        X_sel, sel_idx, sel_names, dropped, gam = stepwise_aic_selection(
            X, y, names, sample_config,
            lam_values=[0.1, 1],
            criterion="bic",
            target_max_terms=4,
        )
        assert len(sel_names) <= 4
        assert len(sel_names) + len(dropped) == n_orig
        assert X_sel.shape[1] == len(sel_names)

    def test_aic_stepwise_still_works(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        X_sel, sel_idx, sel_names, dropped, gam = stepwise_aic_selection(
            X, y, names, sample_config,
            lam_values=[0.1, 1],
            criterion="aic",
        )
        assert len(sel_names) >= 1
        assert X_sel.shape[1] == len(sel_names)


class TestSigmaModel:
    def test_select_sigma_features(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        X_sig, sig_names, indices = select_sigma_features(names, X)
        # At least _std_ should match CL=F_close_std_14
        assert len(sig_names) >= 1
        assert X_sig.shape[1] == len(sig_names)
        assert all(n in names for n in sig_names)

    def test_fit_sigma_gam(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[0.1, 1])
        residuals = y - gam.predict(X)

        X_sig, sig_names, _ = select_sigma_features(names, X)
        sigma_gam, sel_names, sel_idx = fit_sigma_gam(
            X_sig, residuals, sig_names, sample_config, lam_values=[0.1, 1]
        )
        # sigma_gam must predict same n rows as X_sig[:, sel_idx]
        preds = sigma_gam.predict(X_sig[:, sel_idx])
        assert len(preds) == len(y)
        sigma_pred = np.exp(preds)
        assert np.all(sigma_pred > 0)


class TestNuTauModels:
    def test_compute_rolling_nu_tau(self):
        rng = np.random.default_rng(0)
        std_res = rng.standard_normal(200)
        valid_mask, nu_vals, tau_vals = compute_rolling_nu_tau(std_res, window=30)
        assert valid_mask.sum() == len(nu_vals)
        assert valid_mask.sum() == len(tau_vals)
        assert valid_mask.sum() == 200 - 30 + 1
        assert np.all(tau_vals > 0)

    def test_select_nu_tau_features_fallback(self, sample_feature_df, sample_config):
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        # sample_feature_df has no ovx/vix/hy_spread — should fallback gracefully
        X_nt, nt_names, indices = select_nu_tau_features(names, X)
        assert X_nt.shape[0] == X.shape[0]
        assert len(nt_names) >= 1

    def test_fit_distributional_gam(self, sample_feature_df, sample_config):
        rng = np.random.default_rng(1)
        X, y, names, _dates = build_feature_matrix(sample_feature_df, "CL=F_close")
        X_nt, nt_names, _ = select_nu_tau_features(names, X)
        std_res = rng.standard_normal(len(y))
        valid_mask, rolling_nu, _ = compute_rolling_nu_tau(std_res, window=30)
        X_valid = X_nt[valid_mask]
        nu_gam, sel_names, sel_idx = fit_distributional_gam(
            X_valid, rolling_nu, nt_names, sample_config,
            lam_values=[0.1, 1], max_terms=3, param_name="nu",
        )
        preds = nu_gam.predict(X_valid[:, sel_idx])
        assert len(preds) == X_valid.shape[0]


class TestResolveModelConfig:
    """Tests for the preset-based model config resolution."""

    def test_legacy_flat_config(self):
        """Old-style config without active_mode returns as-is."""
        config = {
            "model": {
                "n_splines": 20,
                "cv_splits": 3,
                "lam_search": [0.1, 1],
            }
        }
        resolved = resolve_model_config(config)
        assert resolved["n_splines"] == 20
        assert resolved["cv_splits"] == 3
        assert "_active_mode" not in resolved

    def test_fast_mode(self):
        """Fast preset resolves its values correctly."""
        config = {
            "model": {
                "active_mode": "fast",
                "fast": {
                    "stepwise_criterion": "aic",
                    "stepwise_max_steps": 15,
                    "target_max_terms": None,
                    "lam_search": [0.01, 1, 100],
                    "cv_splits": 3,
                },
                "thorough": {
                    "stepwise_criterion": "bic",
                    "cv_splits": 5,
                },
                "n_splines": 25,
            }
        }
        resolved = resolve_model_config(config)
        assert resolved["_active_mode"] == "fast"
        assert resolved["stepwise_criterion"] == "aic"
        assert resolved["stepwise_max_steps"] == 15
        assert resolved["target_max_terms"] is None
        assert resolved["lam_search"] == [0.01, 1, 100]
        assert resolved["cv_splits"] == 3
        assert resolved["n_splines"] == 25

    def test_thorough_mode(self):
        """Thorough preset resolves its values correctly."""
        config = {
            "model": {
                "active_mode": "thorough",
                "fast": {"cv_splits": 3},
                "thorough": {
                    "stepwise_criterion": "bic",
                    "stepwise_max_steps": 50,
                    "target_max_terms": 25,
                    "cv_splits": 5,
                },
            }
        }
        resolved = resolve_model_config(config)
        assert resolved["_active_mode"] == "thorough"
        assert resolved["stepwise_criterion"] == "bic"
        assert resolved["cv_splits"] == 5
        assert resolved["target_max_terms"] == 25

    def test_shared_keys_applied(self):
        """Shared keys outside preset blocks are picked up."""
        config = {
            "model": {
                "active_mode": "fast",
                "fast": {"cv_splits": 3},
                "thorough": {},
                "n_splines": 30,
                "embargo_days": 150,
            }
        }
        resolved = resolve_model_config(config)
        assert resolved["n_splines"] == 30
        assert resolved["embargo_days"] == 150

    def test_preset_overrides_shared(self):
        """Preset block values win over shared keys."""
        config = {
            "model": {
                "active_mode": "fast",
                "fast": {"n_splines": 15},
                "thorough": {},
                "n_splines": 30,
            }
        }
        resolved = resolve_model_config(config)
        assert resolved["n_splines"] == 15

    def test_invalid_mode_raises(self):
        """Unknown active_mode raises ValueError."""
        config = {"model": {"active_mode": "turbo"}}
        with pytest.raises(ValueError, match="Unknown model.active_mode"):
            resolve_model_config(config)

    def test_defaults_fill_missing_keys(self):
        """Hard-coded defaults fill in any keys absent from both shared and preset."""
        config = {
            "model": {
                "active_mode": "fast",
                "fast": {},
                "thorough": {},
            }
        }
        resolved = resolve_model_config(config)
        # Should fall back to _MODEL_DEFAULTS
        assert resolved["n_splines"] == 25
        assert resolved["embargo_days"] == 200
        assert resolved["sigma_max_terms"] == 12
