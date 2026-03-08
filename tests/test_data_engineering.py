"""Tests for data engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.data_engineering import (
    add_atr_features,
    add_calendar_features,
    add_cmf,
    add_lag_features,
    add_momentum_features,
    add_momentum_windows,
    add_return_features,
    add_rolling_correlations,
    add_rolling_features,
    add_seasonal_flags,
    add_stochastic,
    add_williams_r,
    merge_datasets,
)


@pytest.fixture
def sample_oil_df():
    """Create a simple daily oil price DataFrame."""
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CL=F_close": 70 + rng.standard_normal(100).cumsum(),
            "CL=F_volume": rng.integers(1000, 5000, 100),
        },
        index=dates,
    )


@pytest.fixture
def sample_ohlcv_df():
    """Create a daily OHLCV DataFrame for indicator tests."""
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    rng = np.random.default_rng(7)
    close = 70 + rng.standard_normal(100).cumsum()
    high = close + rng.uniform(0.5, 2.0, 100)
    low = close - rng.uniform(0.5, 2.0, 100)
    volume = rng.integers(1000, 5000, 100).astype(float)
    return pd.DataFrame(
        {
            "CL=F_close": close,
            "CL=F_high": high,
            "CL=F_low": low,
            "CL=F_volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def flat_ohlcv_df():
    """OHLCV DataFrame with identical high and low (degenerate range)."""
    dates = pd.date_range("2023-01-02", periods=50, freq="B")
    price = np.full(50, 70.0)
    return pd.DataFrame(
        {
            "CL=F_close": price,
            "CL=F_high": price,
            "CL=F_low": price,
            "CL=F_volume": np.zeros(50),
        },
        index=dates,
    )


@pytest.fixture
def sample_fred_df():
    """Create a simple FRED macro DataFrame."""
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "usd_index": 100 + rng.standard_normal(100).cumsum() * 0.5,
            "fed_funds": 5.0 + rng.standard_normal(100) * 0.1,
        },
        index=dates,
    )


@pytest.fixture
def sample_eia_dfs():
    """Create EIA-like weekly data."""
    dates = pd.date_range("2023-01-06", periods=20, freq="W-FRI")
    rng = np.random.default_rng(42)
    return {
        "crude_stocks": pd.DataFrame(
            {"value": 400_000 + rng.standard_normal(20).cumsum() * 1000},
            index=dates,
        )
    }


@pytest.fixture
def sample_cot_df():
    """Create CFTC COT-like weekly data."""
    dates = pd.date_range("2023-01-06", periods=20, freq="W-FRI")
    rng = np.random.default_rng(42)
    mm_long = 200_000 + rng.standard_normal(20).cumsum() * 5000
    mm_short = 100_000 + rng.standard_normal(20).cumsum() * 5000
    return pd.DataFrame(
        {
            "cot_mm_long": mm_long,
            "cot_mm_short": mm_short,
            "cot_mm_net": mm_long - mm_short,
            "cot_open_interest": 400_000 + rng.standard_normal(20).cumsum() * 10000,
        },
        index=dates,
    )


@pytest.fixture
def sample_alpha_vantage_df():
    """Create Alpha Vantage-like monthly economic indicator data."""
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "av_consumer_sentiment": 70 + rng.standard_normal(12) * 5,
            "av_nonfarm_payroll": 200 + rng.standard_normal(12) * 50,
        },
        index=dates,
    )


class TestMergeDatasets:
    def test_merge_produces_daily_index(self, sample_oil_df, sample_fred_df, sample_eia_dfs):
        merged = merge_datasets(sample_oil_df, sample_fred_df, sample_eia_dfs)
        assert isinstance(merged.index, pd.DatetimeIndex)
        assert len(merged) > 0

    def test_merge_contains_all_columns(self, sample_oil_df, sample_fred_df, sample_eia_dfs):
        merged = merge_datasets(sample_oil_df, sample_fred_df, sample_eia_dfs)
        assert "CL=F_close" in merged.columns
        assert "usd_index" in merged.columns
        assert "crude_stocks" in merged.columns

    def test_merge_no_nans(self, sample_oil_df, sample_fred_df, sample_eia_dfs):
        merged = merge_datasets(sample_oil_df, sample_fred_df, sample_eia_dfs)
        assert not merged.isna().any().any()

    def test_merge_includes_cot_columns(
        self, sample_oil_df, sample_fred_df, sample_eia_dfs, sample_cot_df
    ):
        merged = merge_datasets(
            sample_oil_df, sample_fred_df, sample_eia_dfs, cot_df=sample_cot_df
        )
        assert "cot_mm_long" in merged.columns
        assert "cot_mm_net" in merged.columns

    def test_merge_includes_alpha_vantage_columns(
        self, sample_oil_df, sample_fred_df, sample_eia_dfs, sample_alpha_vantage_df
    ):
        merged = merge_datasets(
            sample_oil_df, sample_fred_df, sample_eia_dfs,
            alpha_vantage_df=sample_alpha_vantage_df,
        )
        assert "av_consumer_sentiment" in merged.columns
        assert "av_nonfarm_payroll" in merged.columns

    def test_merge_without_optional_sources(self, sample_oil_df, sample_fred_df, sample_eia_dfs):
        """Backward compatibility: all optional args default to None."""
        merged = merge_datasets(sample_oil_df, sample_fred_df, sample_eia_dfs)
        cot_cols = [c for c in merged.columns if c.startswith("cot_")]
        av_cols = [c for c in merged.columns if c.startswith("av_")]
        assert len(cot_cols) == 0
        assert len(av_cols) == 0


class TestLagFeatures:
    def test_lag_creates_correct_columns(self, sample_oil_df):
        df = add_lag_features(sample_oil_df.copy(), "CL=F_close", [1, 7])
        assert "CL=F_close_lag_1" in df.columns
        assert "CL=F_close_lag_7" in df.columns

    def test_lag_values_are_shifted(self, sample_oil_df):
        df = add_lag_features(sample_oil_df.copy(), "CL=F_close", [1])
        # lag_1[i] should equal close[i-1]
        np.testing.assert_array_equal(
            df["CL=F_close_lag_1"].iloc[1:].values,
            df["CL=F_close"].iloc[:-1].values,
        )

    def test_lag_first_values_are_nan(self, sample_oil_df):
        df = add_lag_features(sample_oil_df.copy(), "CL=F_close", [7])
        assert df["CL=F_close_lag_7"].iloc[:7].isna().all()


class TestRollingFeatures:
    def test_rolling_creates_correct_columns(self, sample_oil_df):
        df = add_rolling_features(sample_oil_df.copy(), "CL=F_close", [14])
        assert "CL=F_close_sma_14" in df.columns
        assert "CL=F_close_std_14" in df.columns

    def test_rolling_mean_is_correct(self, sample_oil_df):
        df = add_rolling_features(sample_oil_df.copy(), "CL=F_close", [14])
        # Manually compute rolling mean at position 20
        expected = df["CL=F_close"].iloc[7:21].mean()
        actual = df["CL=F_close_sma_14"].iloc[20]
        np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_rolling_warmup_is_nan(self, sample_oil_df):
        df = add_rolling_features(sample_oil_df.copy(), "CL=F_close", [50])
        assert df["CL=F_close_sma_50"].iloc[:49].isna().all()
        assert not np.isnan(df["CL=F_close_sma_50"].iloc[49])


class TestCalendarFeatures:
    def test_calendar_creates_columns(self, sample_oil_df):
        df = add_calendar_features(sample_oil_df.copy())
        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "day_of_year" in df.columns

    def test_day_of_week_range(self, sample_oil_df):
        df = add_calendar_features(sample_oil_df.copy())
        assert df["day_of_week"].min() >= 0
        assert df["day_of_week"].max() <= 6

    def test_month_range(self, sample_oil_df):
        df = add_calendar_features(sample_oil_df.copy())
        assert df["month"].min() >= 1
        assert df["month"].max() <= 12


class TestReturnFeatures:
    def test_return_creates_columns(self, sample_oil_df):
        df = add_return_features(sample_oil_df.copy(), ["CL=F_close"])
        assert "CL=F_close_pct_change" in df.columns
        assert "CL=F_close_log_return" in df.columns

    def test_pct_change_values(self, sample_oil_df):
        df = add_return_features(sample_oil_df.copy(), ["CL=F_close"])
        # pct_change[1] should be (price[1] - price[0]) / price[0]
        expected = (
            sample_oil_df["CL=F_close"].iloc[1] - sample_oil_df["CL=F_close"].iloc[0]
        ) / sample_oil_df["CL=F_close"].iloc[0]
        actual = df["CL=F_close_pct_change"].iloc[1]
        np.testing.assert_almost_equal(actual, expected, decimal=10)


class TestMomentumFeatures:
    def test_creates_rsi_and_macd_columns(self, sample_ohlcv_df):
        df = add_momentum_features(sample_ohlcv_df.copy(), "CL=F_close")
        assert "CL=F_close_rsi_14" in df.columns
        assert "CL=F_close_macd" in df.columns
        assert "CL=F_close_macd_signal" in df.columns
        assert "CL=F_close_macd_hist" in df.columns

    def test_rsi_range(self, sample_ohlcv_df):
        df = add_momentum_features(sample_ohlcv_df.copy(), "CL=F_close")
        valid = df["CL=F_close_rsi_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_hist_is_line_minus_signal(self, sample_ohlcv_df):
        df = add_momentum_features(sample_ohlcv_df.copy(), "CL=F_close")
        expected = df["CL=F_close_macd"] - df["CL=F_close_macd_signal"]
        pd.testing.assert_series_equal(
            df["CL=F_close_macd_hist"], expected, check_names=False
        )

    def test_custom_macd_params(self, sample_ohlcv_df):
        df = add_momentum_features(
            sample_ohlcv_df.copy(), "CL=F_close",
            rsi_period=10, macd_fast=5, macd_slow=10, macd_signal=3,
        )
        assert "CL=F_close_rsi_10" in df.columns
        assert "CL=F_close_macd" in df.columns


class TestATRFeatures:
    def test_creates_atr_columns(self, sample_ohlcv_df):
        df = add_atr_features(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close"
        )
        assert "CL=F_close_atr_14" in df.columns
        assert "CL=F_close_atr_pct_14" in df.columns

    def test_atr_is_non_negative(self, sample_ohlcv_df):
        df = add_atr_features(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close"
        )
        valid = df["CL=F_close_atr_14"].dropna()
        assert (valid >= 0).all()

    def test_atr_custom_period(self, sample_ohlcv_df):
        df = add_atr_features(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close", period=7
        )
        assert "CL=F_close_atr_7" in df.columns

    def test_flat_range_atr_is_zero(self, flat_ohlcv_df):
        """When high == low and price is constant, ATR converges to zero."""
        df = add_atr_features(
            flat_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close", period=5
        )
        valid = df["CL=F_close_atr_5"].dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)


class TestWilliamsR:
    def test_creates_williams_r_column(self, sample_ohlcv_df):
        df = add_williams_r(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close"
        )
        assert "CL=F_close_williams_r_14" in df.columns

    def test_williams_r_range(self, sample_ohlcv_df):
        df = add_williams_r(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close"
        )
        valid = df["CL=F_close_williams_r_14"].dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_flat_range_williams_r_is_nan(self, flat_ohlcv_df):
        """When high == low, Williams %R is undefined (NaN)."""
        df = add_williams_r(
            flat_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close", period=5
        )
        valid_mask = ~flat_ohlcv_df["CL=F_high"].rolling(5, min_periods=5).max().isna()
        assert df.loc[valid_mask, "CL=F_close_williams_r_5"].isna().all()

    def test_custom_period(self, sample_ohlcv_df):
        df = add_williams_r(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close", period=7
        )
        assert "CL=F_close_williams_r_7" in df.columns


class TestStochastic:
    def test_creates_stoch_columns(self, sample_ohlcv_df):
        df = add_stochastic(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close"
        )
        assert "CL=F_close_stoch_k_14" in df.columns
        assert "CL=F_close_stoch_d_14" in df.columns

    def test_stoch_k_range(self, sample_ohlcv_df):
        df = add_stochastic(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close"
        )
        valid = df["CL=F_close_stoch_k_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_flat_range_stoch_is_nan(self, flat_ohlcv_df):
        """When high == low, %K is undefined (NaN)."""
        df = add_stochastic(
            flat_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close",
            k_period=5, d_period=3,
        )
        valid_mask = ~flat_ohlcv_df["CL=F_high"].rolling(5, min_periods=5).max().isna()
        assert df.loc[valid_mask, "CL=F_close_stoch_k_5"].isna().all()

    def test_stoch_d_is_sma_of_k(self, sample_ohlcv_df):
        df = add_stochastic(
            sample_ohlcv_df.copy(), "CL=F_high", "CL=F_low", "CL=F_close",
            k_period=14, d_period=3,
        )
        expected_d = df["CL=F_close_stoch_k_14"].rolling(3, min_periods=3).mean()
        pd.testing.assert_series_equal(
            df["CL=F_close_stoch_d_14"], expected_d, check_names=False
        )


class TestCMF:
    def test_creates_cmf_column(self, sample_ohlcv_df):
        df = add_cmf(
            sample_ohlcv_df.copy(),
            "CL=F_high", "CL=F_low", "CL=F_close", "CL=F_volume",
        )
        assert "CL=F_close_cmf_20" in df.columns

    def test_cmf_range(self, sample_ohlcv_df):
        df = add_cmf(
            sample_ohlcv_df.copy(),
            "CL=F_high", "CL=F_low", "CL=F_close", "CL=F_volume",
        )
        valid = df["CL=F_close_cmf_20"].dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_zero_volume_produces_nan(self, flat_ohlcv_df):
        """When all volume is zero, CMF sum/sum is NaN (0/0)."""
        df = add_cmf(
            flat_ohlcv_df.copy(),
            "CL=F_high", "CL=F_low", "CL=F_close", "CL=F_volume", period=5,
        )
        valid_mask = ~flat_ohlcv_df["CL=F_volume"].rolling(5, min_periods=5).sum().isna()
        assert df.loc[valid_mask, "CL=F_close_cmf_5"].isna().all()

    def test_flat_high_low_produces_nan(self):
        """When high == low, MFM is NaN because the range is zero."""
        dates = pd.date_range("2023-01-02", periods=50, freq="B")
        price = np.full(50, 70.0)
        df = pd.DataFrame(
            {
                "CL=F_close": price,
                "CL=F_high": price,
                "CL=F_low": price,
                "CL=F_volume": np.ones(50) * 1000,
            },
            index=dates,
        )
        result = add_cmf(
            df, "CL=F_high", "CL=F_low", "CL=F_close", "CL=F_volume", period=5
        )
        valid_mask = ~df["CL=F_volume"].rolling(5, min_periods=5).sum().isna()
        assert result.loc[valid_mask, "CL=F_close_cmf_5"].isna().all()


class TestRollingCorrelations:
    def test_creates_correlation_columns(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        rng = np.random.default_rng(9)
        df["gold_close"] = 1800 + rng.standard_normal(len(df)).cumsum()
        result = add_rolling_correlations(df, "CL=F_close", ["gold"], windows=[20])
        assert "corr_gold_20" in result.columns

    def test_correlation_range(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        rng = np.random.default_rng(9)
        df["gold_close"] = 1800 + rng.standard_normal(len(df)).cumsum()
        result = add_rolling_correlations(df, "CL=F_close", ["gold"], windows=[20])
        valid = result["corr_gold_20"].dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_empty_pair_names_no_new_columns(self, sample_ohlcv_df):
        """Empty correlation_pairs should leave the DataFrame unchanged."""
        df = sample_ohlcv_df.copy()
        before_cols = set(df.columns)
        result = add_rolling_correlations(df, "CL=F_close", [], windows=[20])
        assert set(result.columns) == before_cols

    def test_missing_pair_column_is_skipped(self, sample_ohlcv_df, caplog):
        """A pair whose _close column doesn't exist should be silently skipped."""
        import logging
        df = sample_ohlcv_df.copy()
        before_cols = set(df.columns)
        with caplog.at_level(logging.WARNING, logger="src.data_engineering"):
            result = add_rolling_correlations(
                df, "CL=F_close", ["nonexistent"], windows=[20]
            )
        assert set(result.columns) == before_cols

    def test_multiple_windows(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        rng = np.random.default_rng(9)
        df["gold_close"] = 1800 + rng.standard_normal(len(df)).cumsum()
        result = add_rolling_correlations(df, "CL=F_close", ["gold"], windows=[10, 30])
        assert "corr_gold_10" in result.columns
        assert "corr_gold_30" in result.columns


class TestMomentumWindows:
    def test_creates_momentum_columns(self, sample_ohlcv_df):
        df = add_momentum_windows(sample_ohlcv_df.copy(), "CL=F_close", [5, 21])
        assert "CL=F_close_momentum_5" in df.columns
        assert "CL=F_close_momentum_21" in df.columns

    def test_momentum_values_correct(self, sample_ohlcv_df):
        df = add_momentum_windows(sample_ohlcv_df.copy(), "CL=F_close", [5])
        price = sample_ohlcv_df["CL=F_close"]
        expected = (price - price.shift(5)) / price.shift(5)
        pd.testing.assert_series_equal(
            df["CL=F_close_momentum_5"], expected, check_names=False
        )

    def test_warmup_rows_are_nan(self, sample_ohlcv_df):
        df = add_momentum_windows(sample_ohlcv_df.copy(), "CL=F_close", [10])
        assert df["CL=F_close_momentum_10"].iloc[:10].isna().all()
        assert not np.isnan(df["CL=F_close_momentum_10"].iloc[10])


class TestSeasonalFlags:
    def test_driving_season_flag_created(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"driving_season": True})
        assert "driving_season" in result.columns

    def test_driving_season_binary(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"driving_season": True})
        assert set(result["driving_season"].unique()).issubset({0, 1})

    def test_heating_season_flag_created(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"heating_season": True})
        assert "heating_season" in result.columns

    def test_heating_season_correct_months(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"heating_season": True})
        for month, flag in zip(dates.month, result["heating_season"]):
            if month in (10, 11, 12, 1, 2, 3):
                assert flag == 1
            else:
                assert flag == 0

    def test_us_holiday_flag_created(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"us_holidays": True})
        assert "us_holiday" in result.columns

    def test_us_holiday_is_binary(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"us_holidays": True})
        assert set(result["us_holiday"].unique()).issubset({0, 1})

    def test_us_holiday_known_date(self):
        """Independence Day (July 4) should be flagged as a holiday."""
        dates = pd.date_range("2023-07-03", "2023-07-05", freq="D")
        df = pd.DataFrame({"price": 70.0}, index=dates)
        result = add_seasonal_flags(df.copy(), {"us_holidays": True})
        assert result.loc["2023-07-04", "us_holiday"] == 1

    def test_empty_config_adds_no_columns(self, sample_oil_df):
        before_cols = set(sample_oil_df.columns)
        result = add_seasonal_flags(sample_oil_df.copy(), {})
        assert set(result.columns) == before_cols

    def test_disabled_flags_add_no_columns(self, sample_oil_df):
        before_cols = set(sample_oil_df.columns)
        result = add_seasonal_flags(
            sample_oil_df.copy(),
            {"driving_season": False, "heating_season": False, "us_holidays": False},
        )
        assert set(result.columns) == before_cols
