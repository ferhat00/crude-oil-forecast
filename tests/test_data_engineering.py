"""Tests for data engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.data_engineering import (
    add_calendar_features,
    add_lag_features,
    add_return_features,
    add_rolling_features,
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
