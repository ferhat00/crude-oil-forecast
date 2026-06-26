"""Tests for GAM model functions."""

import numpy as np
import pandas as pd
import pytest

from src.config_loader import resolve_model_config
from src.model import (
    SkewedFeatureTransformer,
    _classify_feature,
    _get_monotonic_constraint,
    _is_skewed_feature,
    build_feature_matrix,
    compute_anchor_prices,
    compute_bic,
    compute_rolling_nu_tau,
    conformal_residual_quantile,
    cross_validate,
    define_gam_terms,
    drop_correlated_features,
    fit_distributional_gam,
    fit_gam,
    fit_sigma_gam,
    get_target_y_level,
    reconstruct_price_from_returns,
    select_sigma_features,
    select_nu_tau_features,
    stationary_bootstrap_indices,
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
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert "CL=F_close" not in names

    def test_excludes_raw_ohlcv(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert "CL=F_open" not in names
        assert "CL=F_volume" not in names

    def test_output_shapes(self, sample_feature_df, sample_config):
        X, y, names, target_dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        # Default horizon=1 drops the last row (target unknown)
        assert X.shape[0] == len(sample_feature_df) - 1
        assert X.shape[1] == len(names)
        assert y.shape[0] == len(sample_feature_df) - 1
        assert len(target_dates) == X.shape[0]

    def test_output_shapes_horizon_zero(self, sample_feature_df, sample_config):
        X, y, names, target_dates, _t1 = build_feature_matrix(
            sample_feature_df, "CL=F_close", forecast_horizon=0
        )
        assert X.shape[0] == len(sample_feature_df)
        assert y.shape[0] == len(sample_feature_df)
        assert len(target_dates) == len(sample_feature_df)

    def test_target_dates_shifted(self, sample_feature_df, sample_config):
        X, y, names, target_dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        # target_dates should be the actual next trading day from the index
        assert target_dates[0] == sample_feature_df.index[1]
        assert target_dates[-1] == sample_feature_df.index[-1]

    def test_feature_names_list(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


class TestDefineGamTerms:
    def test_returns_terms_object(self, sample_feature_df, sample_config):
        _, _, names, _, _ = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        assert terms is not None


class TestFitGam:
    def test_fit_on_synthetic_data(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[0.1, 1])
        predictions = gam.predict(X)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))


class TestCrossValidate:
    def test_cv_returns_correct_structure(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
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
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[0.1, 1])
        bic = compute_bic(gam, len(y))
        aic = gam.statistics_["AIC"]
        # BIC = AIC + edof * (log(n) - 2); for n > ~7.4, log(n) > 2 so BIC > AIC
        assert bic >= aic

    def test_bic_returns_float(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        terms = define_gam_terms(names, sample_config)
        gam = fit_gam(X, y, terms, lam_values=[1])
        assert isinstance(compute_bic(gam, len(y)), float)


class TestStepwiseBIC:
    def test_bic_stepwise_reduces_features(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
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
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        X_sel, sel_idx, sel_names, dropped, gam = stepwise_aic_selection(
            X, y, names, sample_config,
            lam_values=[0.1, 1],
            criterion="aic",
        )
        assert len(sel_names) >= 1
        assert X_sel.shape[1] == len(sel_names)


class TestSigmaModel:
    def test_select_sigma_features(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        X_sig, sig_names, indices = select_sigma_features(names, X)
        # At least _std_ should match CL=F_close_std_14
        assert len(sig_names) >= 1
        assert X_sig.shape[1] == len(sig_names)
        assert all(n in names for n in sig_names)

    def test_fit_sigma_gam(self, sample_feature_df, sample_config):
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
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
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
        # sample_feature_df has no ovx/vix/hy_spread — should fallback gracefully
        X_nt, nt_names, indices = select_nu_tau_features(names, X)
        assert X_nt.shape[0] == X.shape[0]
        assert len(nt_names) >= 1

    def test_fit_distributional_gam(self, sample_feature_df, sample_config):
        rng = np.random.default_rng(1)
        X, y, names, _dates, _t1 = build_feature_matrix(sample_feature_df, "CL=F_close")
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


class TestLogReturnTarget:
    """Item 1: build_feature_matrix with target_transform='log_return'."""

    def test_log_return_target_y_in_return_space(self, sample_feature_df):
        X, y, names, dates, _t1 = build_feature_matrix(
            sample_feature_df, "CL=F_close", target_transform="log_return",
        )
        # log-returns of a near-random walk should be O(1e-2) — not in the 60-80 range
        assert np.all(np.abs(y) < 1.0)
        # Manually verified relationship
        prices = sample_feature_df["CL=F_close"].values
        expected_y0 = np.log(prices[1] / prices[0])
        np.testing.assert_almost_equal(y[0], expected_y0, decimal=10)

    def test_log_return_drops_level_lag_features(self, sample_feature_df):
        X_lvl, _, names_lvl, _, _ = build_feature_matrix(
            sample_feature_df, "CL=F_close", target_transform="level",
        )
        X_ret, _, names_ret, _, _ = build_feature_matrix(
            sample_feature_df, "CL=F_close", target_transform="log_return",
        )
        # Level-mode keeps the lag features; log-return mode drops them
        assert "CL=F_close_lag_1" in names_lvl
        assert "CL=F_close_lag_1" not in names_ret
        assert "CL=F_close_lag_7" in names_lvl
        assert "CL=F_close_lag_7" not in names_ret

    def test_anchor_and_y_level_alignment(self, sample_feature_df):
        anchor = compute_anchor_prices(sample_feature_df, "CL=F_close", forecast_horizon=1)
        y_level = get_target_y_level(sample_feature_df, "CL=F_close", forecast_horizon=1)
        prices = sample_feature_df["CL=F_close"].values
        # anchor[i] = price[i]; y_level[i] = price[i+1]
        np.testing.assert_array_equal(anchor, prices[:-1])
        np.testing.assert_array_equal(y_level, prices[1:])

    def test_reconstruct_price_round_trip(self, sample_feature_df):
        _, y_ret, _, _, _ = build_feature_matrix(
            sample_feature_df, "CL=F_close", target_transform="log_return",
        )
        anchor = compute_anchor_prices(sample_feature_df, "CL=F_close", forecast_horizon=1)
        y_level = get_target_y_level(sample_feature_df, "CL=F_close", forecast_horizon=1)
        reconstructed = reconstruct_price_from_returns(y_ret, anchor)
        np.testing.assert_allclose(reconstructed, y_level, rtol=1e-10)

    def test_unknown_transform_raises(self, sample_feature_df):
        with pytest.raises(ValueError, match="Unknown target_transform"):
            build_feature_matrix(
                sample_feature_df, "CL=F_close", target_transform="not_a_transform",
            )


class TestSkewedFeaturePatterns:
    def test_returns_are_skewed(self):
        assert _is_skewed_feature("CL=F_close_log_return")
        assert _is_skewed_feature("CL=F_close_pct_change")

    def test_oscillators_are_skewed(self):
        assert _is_skewed_feature("CL=F_close_rsi_14")
        assert _is_skewed_feature("CL=F_close_williams_r_14")
        assert _is_skewed_feature("CL=F_close_stoch_k_14")

    def test_macro_not_skewed(self):
        assert not _is_skewed_feature("usd_index")
        assert not _is_skewed_feature("fed_funds")


class TestSkewedFeatureTransformer:
    def test_transform_round_shape(self, sample_feature_df):
        X, _, names, _, _ = build_feature_matrix(sample_feature_df, "CL=F_close")
        t = SkewedFeatureTransformer(names).fit(X)
        Xt = t.transform(X)
        assert Xt.shape == X.shape

    def test_skewed_columns_become_uniform(self, sample_feature_df):
        X, _, names, _, _ = build_feature_matrix(sample_feature_df, "CL=F_close")
        t = SkewedFeatureTransformer(names).fit(X)
        Xt = t.transform(X)
        # Pick a skewed column (pct_change in our fixture)
        if "CL=F_close_pct_change" in names:
            j = names.index("CL=F_close_pct_change")
            # Transformed column should be roughly uniform on [0, 1]
            col = Xt[:, j]
            assert col.min() >= 0.0
            assert col.max() <= 1.0
            # Mean of uniform(0,1) is 0.5; allow generous tolerance for small n
            assert 0.3 < float(np.mean(col)) < 0.7


class TestMonotonicLookup:
    def test_inventory_decreases_price(self):
        assert _get_monotonic_constraint("crude_stocks") == "monotonic_dec"
        assert _get_monotonic_constraint("cushing_stocks") == "monotonic_dec"

    def test_dxy_decreases_price(self):
        assert _get_monotonic_constraint("usd_index") == "monotonic_dec"

    def test_refinery_increases_price(self):
        assert _get_monotonic_constraint("refinery_utilization") == "monotonic_inc"

    def test_unknown_returns_none(self):
        assert _get_monotonic_constraint("vix_close") is None
        assert _get_monotonic_constraint("CL=F_close_rsi_14") is None


class TestCollinearityPruning:
    def test_drops_perfectly_correlated_duplicate(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.standard_normal(n)
        X = np.column_stack([x, x + 1e-9, rng.standard_normal(n)])
        names = ["a", "a_dup", "b"]
        keep, kept_names, dropped = drop_correlated_features(X, names, threshold=0.97)
        assert "a" in kept_names
        assert "b" in kept_names
        assert "a_dup" in dropped

    def test_no_drops_when_uncorrelated(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 4))
        names = list("abcd")
        keep, kept_names, dropped = drop_correlated_features(X, names, threshold=0.97)
        assert dropped == []
        assert kept_names == names


class TestStationaryBootstrap:
    def test_indices_have_correct_length(self):
        idx = stationary_bootstrap_indices(100, mean_block_length=10,
                                           rng=np.random.default_rng(0))
        assert len(idx) == 100

    def test_indices_in_range(self):
        idx = stationary_bootstrap_indices(50, mean_block_length=5,
                                           rng=np.random.default_rng(0))
        assert idx.min() >= 0
        assert idx.max() < 50


class TestConformalQuantile:
    def test_perfect_predictions_yield_zero_quantile(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        q = conformal_residual_quantile(y_true, y_pred, alpha=0.1)
        assert q == 0.0

    def test_quantile_increases_with_errors(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(500)
        small_err = y + rng.standard_normal(500) * 0.1
        big_err = y + rng.standard_normal(500) * 1.0
        q_small = conformal_residual_quantile(y, small_err, alpha=0.1)
        q_big = conformal_residual_quantile(y, big_err, alpha=0.1)
        assert q_big > q_small

    def test_alpha_changes_quantile(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(500)
        y_pred = y + rng.standard_normal(500) * 0.5
        q_90 = conformal_residual_quantile(y, y_pred, alpha=0.10)
        q_99 = conformal_residual_quantile(y, y_pred, alpha=0.01)
        # Tighter alpha → larger nonconformity quantile
        assert q_99 > q_90


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


# ─────────────────────────────────────────────────────────────────────────────
# Purged + embargoed CV (Chapter 7) — src.cv.PurgedKFold
# ─────────────────────────────────────────────────────────────────────────────


def _synthetic_t1(n: int, horizon: int) -> tuple[np.ndarray, pd.Series]:
    """Build an (X, t1) pair where row i's label resolves at row i+horizon."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    feature_dates = dates[: n - horizon] if horizon > 0 else dates
    label_dates = dates[horizon:] if horizon > 0 else dates
    rng = np.random.default_rng(0)
    X = rng.standard_normal((len(feature_dates), 2))
    t1 = pd.Series(label_dates, index=feature_dates, name="t1")
    return X, t1


class TestPurgedKFold:
    def test_h1_no_embargo_matches_uniform_partition(self):
        """For horizon=1 and pct_embargo=0, the test folds are contiguous and
        cover the full sample exactly once — the basic KFold property the
        legacy splitter only approximated after embargo."""
        from src.cv import PurgedKFold

        X, t1 = _synthetic_t1(n=100, horizon=1)
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
        test_chunks = [te for _tr, te in cv.split(X)]
        assert len(test_chunks) == 5
        combined = np.concatenate(test_chunks)
        np.testing.assert_array_equal(combined, np.arange(len(X)))

    def test_h5_purges_overlapping_labels(self):
        """With h=5, training rows whose t1 lies inside the test span must be
        purged.  The first test row at position p has feature-date t1.index[p];
        any train row with t1 >= that date must be dropped."""
        from src.cv import PurgedKFold

        X, t1 = _synthetic_t1(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
        for train_idx, test_idx in cv.split(X):
            test_t0 = t1.index[int(test_idx.min())]
            train_t1 = t1.iloc[train_idx]
            # Either label fully resolved before test starts, OR row sits
            # after the test region entirely.
            assert (
                (train_t1 < test_t0) | (train_idx >= int(test_idx.max()) + 1)
            ).all(), "PurgedKFold leaked label-overlapping training rows."

    def test_bilateral_embargo_drops_post_test_rows(self):
        """Rows immediately after each test fold (within the embargo window)
        must be excluded from training — the embargo is *bilateral* in the
        canonical algorithm."""
        from src.cv import PurgedKFold

        X, t1 = _synthetic_t1(n=100, horizon=1)
        cv = PurgedKFold(n_splits=4, t1=t1, pct_embargo=0.05)  # 5 obs embargo
        embargo = int(len(X) * 0.05)
        for train_idx, test_idx in cv.split(X):
            test_end = int(test_idx.max())
            # No training rows in (test_end, test_end + embargo].
            forbidden = set(range(test_end + 1, test_end + 1 + embargo))
            assert not (set(train_idx.tolist()) & forbidden), (
                "Embargo zone not respected: train_idx overlaps post-test embargo."
            )

    def test_raises_on_misaligned_t1(self):
        from src.cv import PurgedKFold

        X, t1 = _synthetic_t1(n=100, horizon=1)
        cv = PurgedKFold(n_splits=3, t1=t1.iloc[:-5], pct_embargo=0.0)
        with pytest.raises(ValueError, match="aligned"):
            list(cv.split(X))

    def test_raises_when_t1_missing(self):
        from src.cv import PurgedKFold

        with pytest.raises(ValueError, match="t1"):
            PurgedKFold(n_splits=3, t1=None)


class TestEmbargoedTimeSeriesSplitT1Wrapper:
    def test_delegates_to_purged_when_t1_passed(self):
        """The wrapper must use the new purged algorithm when given t1, so
        existing tests will exercise the canonical code path once t1 is
        threaded through."""
        from src.model import EmbargoedTimeSeriesSplit

        X, t1 = _synthetic_t1(n=200, horizon=1)
        splitter = EmbargoedTimeSeriesSplit(n_splits=4, embargo_days=10, t1=t1)
        folds = list(splitter.split(X))
        # PurgedKFold yields n_splits contiguous test folds covering the sample.
        assert len(folds) == 4
        combined = np.concatenate([te for _tr, te in folds])
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(X)))

    def test_legacy_mode_when_t1_absent(self):
        """Without t1, the wrapper must reproduce the original one-sided
        embargo behaviour for backward compatibility."""
        from src.model import EmbargoedTimeSeriesSplit

        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        splitter = EmbargoedTimeSeriesSplit(n_splits=4, embargo_days=5)
        folds = list(splitter.split(X))
        # Each test fold should start strictly after train_end + embargo_days.
        for train_idx, test_idx in folds:
            assert test_idx.min() > train_idx.max() + 5
