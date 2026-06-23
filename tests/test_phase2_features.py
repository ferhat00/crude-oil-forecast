"""Tests for Phase 2 predictors: inventory seasonal/surprise + GPR/GECON."""

import numpy as np
import pandas as pd

from src.data_acquisition import fetch_gecon_data, fetch_gpr_data
from src.data_engineering import (
    add_gpr_features,
    add_inventory_seasonal_features,
)


def _seasonal_stock_df(years: int = 8) -> pd.DataFrame:
    """Daily crude_stocks that depend only on ISO week-of-year (identical years)."""
    idx = pd.date_range("2016-01-04", periods=years * 252, freq="B")
    woy = idx.isocalendar()["week"].astype(int).to_numpy()
    val = 400.0 + 20.0 * np.sin(2 * np.pi * woy / 52.0)
    return pd.DataFrame({"crude_stocks": val}, index=idx)


class TestInventorySeasonal:
    def test_columns_created(self):
        out = add_inventory_seasonal_features(_seasonal_stock_df())
        for suffix in ("seasonal_dev", "seasonal_z", "seasonal_z_pos",
                       "seasonal_z_neg", "surprise"):
            assert f"crude_stocks_{suffix}" in out.columns

    def test_identical_years_have_near_zero_deviation(self):
        # When every year repeats the same seasonal pattern, the deviation from
        # the prior-years same-week norm must be ~0 once a few years exist.
        out = add_inventory_seasonal_features(_seasonal_stock_df(years=8))
        last_year = out.loc["2022":]
        assert last_year["crude_stocks_seasonal_dev"].abs().median() < 1.0

    def test_anomaly_lifts_deviation_and_surprise(self):
        df = _seasonal_stock_df(years=8)
        df.iloc[-3:, df.columns.get_loc("crude_stocks")] += 60.0  # late-sample build
        out = add_inventory_seasonal_features(df)
        # The anomalous week's deviation is large and positive.
        assert out["crude_stocks_seasonal_dev"].iloc[-1] > 30.0
        # The surprise spikes on its release day.
        assert out["crude_stocks_surprise"].abs().max() > 20.0

    def test_surprise_zero_off_release_days(self):
        out = add_inventory_seasonal_features(_seasonal_stock_df())
        iso = out.index.isocalendar()
        yw = iso["year"].astype(int) * 100 + iso["week"].astype(int)
        is_release = ~pd.Series(yw.to_numpy(), index=out.index).duplicated()
        off = out.loc[~is_release.to_numpy(), "crude_stocks_surprise"]
        assert np.allclose(off.to_numpy(), 0.0)

    def test_pos_neg_split_signs(self):
        out = add_inventory_seasonal_features(_seasonal_stock_df())
        z_pos = out["crude_stocks_seasonal_z_pos"].dropna()
        z_neg = out["crude_stocks_seasonal_z_neg"].dropna()
        assert (z_pos >= 0).all()
        assert (z_neg <= 0).all()

    def test_missing_column_no_op(self):
        df = pd.DataFrame({"other": [1.0, 2.0]},
                          index=pd.date_range("2020-01-01", periods=2))
        out = add_inventory_seasonal_features(df)
        assert "crude_stocks_seasonal_z" not in out.columns


class TestGprFeatures:
    def test_columns_and_binary_spike(self):
        idx = pd.date_range("2018-01-01", periods=400, freq="B")
        rng = np.random.default_rng(0)
        gpr = np.abs(rng.normal(100, 30, 400))
        gpr[200] = 500.0  # a spike
        df = pd.DataFrame({"gpr": gpr}, index=idx)
        out = add_gpr_features(df)
        assert {"gpr_log", "gpr_trend_30", "gpr_spike"} <= set(out.columns)
        assert set(np.unique(out["gpr_spike"].dropna())) <= {0.0, 1.0}
        # The injected spike day should be flagged.
        assert out["gpr_spike"].iloc[200] == 1.0


class TestFetchers:
    def test_fetch_gpr_parses_csv(self, tmp_path):
        p = tmp_path / "gpr.csv"
        pd.DataFrame(
            {"date": ["2020-01-01", "2020-01-02"], "GPRD": [100.0, 150.0],
             "event": ["x", "y"]}
        ).to_csv(p, index=False)
        out = fetch_gpr_data(str(p))
        assert list(out.columns) == ["gpr"]
        assert len(out) == 2
        assert out["gpr"].iloc[1] == 150.0

    def test_fetch_gecon_parses_csv(self, tmp_path):
        p = tmp_path / "gecon.csv"
        pd.DataFrame(
            {"date": ["2020-01-01", "2020-02-01"], "GECON": [0.5, -0.3]}
        ).to_csv(p, index=False)
        out = fetch_gecon_data(str(p))
        assert list(out.columns) == ["gecon"]
        assert out["gecon"].iloc[0] == 0.5

    def test_fetch_gecon_fallback_first_numeric(self, tmp_path):
        # No column literally named GECON ⇒ first numeric non-date column used.
        p = tmp_path / "g.csv"
        pd.DataFrame(
            {"date": ["2020-01-01", "2020-02-01"], "factor": [1.0, 2.0]}
        ).to_csv(p, index=False)
        out = fetch_gecon_data(str(p))
        assert list(out.columns) == ["gecon"]
        assert len(out) == 2
