"""Tests for GAM model functions."""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    _classify_feature,
    build_feature_matrix,
    cross_validate,
    define_gam_terms,
    fit_gam,
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
        X, y, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert "CL=F_close" not in names

    def test_excludes_raw_ohlcv(self, sample_feature_df, sample_config):
        X, y, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert "CL=F_open" not in names
        assert "CL=F_volume" not in names

    def test_output_shapes(self, sample_feature_df, sample_config):
        X, y, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert X.shape[0] == len(sample_feature_df)
        assert X.shape[1] == len(names)
        assert y.shape[0] == len(sample_feature_df)

    def test_feature_names_list(self, sample_feature_df, sample_config):
        X, y, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


class TestDefineGamTerms:
    def test_returns_terms_object(self, sample_feature_df, sample_config):
        _, _, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        assert terms is not None


class TestFitGam:
    def test_fit_on_synthetic_data(self, sample_feature_df, sample_config):
        X, y, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[0.1, 1])
        predictions = gam.predict(X)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))


class TestCrossValidate:
    def test_cv_returns_correct_structure(self, sample_feature_df, sample_config):
        X, y, names = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        results = cross_validate(X, y, terms, n_splits=3, lam_values=[0.1, 1])

        assert "fold_metrics" in results
        assert "mean_mae" in results
        assert "mean_rmse" in results
        assert "mean_mape" in results
        assert len(results["fold_metrics"]) == 3
        assert results["mean_mae"] > 0
        assert results["mean_rmse"] > 0
