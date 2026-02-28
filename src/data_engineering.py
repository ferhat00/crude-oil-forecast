"""Data engineering: merging, cleaning, and feature construction."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import get_project_root

logger = logging.getLogger(__name__)


def merge_datasets(
    oil_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    eia_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge all data sources on date index.

    EIA data (weekly/monthly) is resampled to daily via forward-fill before merging.
    All DataFrames are outer-joined on date, then forward-filled to handle
    weekends/holidays across sources.

    Args:
        oil_df: Daily oil price data.
        fred_df: Daily/irregular FRED macro data.
        eia_dfs: Dict of EIA DataFrames (e.g., {"crude_stocks": df, "production": df}).

    Returns:
        Merged DataFrame with daily frequency and no gaps.
    """
    # Start with oil prices as the base (daily trading days)
    merged = oil_df.copy()

    # Join FRED data
    merged = merged.join(fred_df, how="left")

    # Resample EIA data to daily and join
    for name, eia_df in eia_dfs.items():
        # Rename value column to avoid conflicts
        eia_renamed = eia_df.rename(columns={"value": name})
        # Resample to daily frequency and forward-fill
        eia_daily = eia_renamed.resample("D").ffill()
        merged = merged.join(eia_daily, how="left")

    # Forward-fill remaining NaN from holidays/weekends
    merged = merged.ffill()

    # Drop rows before first complete observation
    merged = merged.dropna()

    logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Create lagged versions of the target variable.

    Args:
        df: Input DataFrame.
        target_col: Column to create lags for.
        lags: List of lag periods (e.g., [1, 7, 30]).

    Returns:
        DataFrame with new lag columns added.
    """
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Calculate rolling mean and standard deviation.

    Args:
        df: Input DataFrame.
        target_col: Column to compute rolling stats for.
        windows: List of window sizes (e.g., [14, 50]).

    Returns:
        DataFrame with new rolling columns added.
    """
    for window in windows:
        df[f"{target_col}_sma_{window}"] = (
            df[target_col].rolling(window=window, min_periods=window).mean()
        )
        df[f"{target_col}_std_{window}"] = (
            df[target_col].rolling(window=window, min_periods=window).std()
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the DatetimeIndex.

    Creates: day_of_week (0-6), month (1-12), day_of_year (1-366).

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with calendar columns added.
    """
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    return df


def add_return_features(
    df: pd.DataFrame,
    price_cols: list[str],
) -> pd.DataFrame:
    """Compute daily percentage and log returns.

    Args:
        df: Input DataFrame.
        price_cols: List of price columns to compute returns for.

    Returns:
        DataFrame with return columns added.
    """
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_pct_change"] = df[col].pct_change()
            df[f"{col}_log_return"] = np.log(df[col] / df[col].shift(1))
    return df


def build_features(config: dict) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Loads raw parquet files, merges, creates all features, drops NaN rows,
    and saves the result to data/processed/features.parquet.

    Args:
        config: Project configuration dictionary.

    Returns:
        Feature-engineered DataFrame ready for modeling.
    """
    root = get_project_root(config)
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = config["features"]
    target = feat_cfg["target"]

    # Load raw data
    oil_df = pd.read_parquet(raw_dir / "oil_prices.parquet")
    fred_df = pd.read_parquet(raw_dir / "fred_macro.parquet")

    eia_dfs = {}
    for name in config["data"].get("eia_series", {}).keys():
        eia_path = raw_dir / f"eia_{name}.parquet"
        if eia_path.exists():
            eia_dfs[name] = pd.read_parquet(eia_path)

    logger.info(f"Oil price columns: {list(oil_df.columns)}")

    # Merge all sources
    df = merge_datasets(oil_df, fred_df, eia_dfs)

    logger.info(f"Merged dataframe columns: {list(df.columns)}")

    # Determine target column (case-insensitive search)
    target_col = None
    for col in df.columns:
        if col.lower() == target.lower():
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}"
        )

    logger.info(f"Using target column: {target_col}")

    # Determine price columns for returns
    oil_tickers = config["data"]["oil_tickers"]
    price_cols = []
    for t in oil_tickers:
        for col in df.columns:
            if col.lower() == f"{t}_close".lower():
                price_cols.append(col)
                break

    # Feature engineering
    df = add_lag_features(df, target_col, feat_cfg["lag_days"])
    df = add_rolling_features(df, target_col, feat_cfg["rolling_windows"])
    df = add_calendar_features(df)
    df = add_return_features(df, price_cols)

    # Drop rows with NaN from lag/rolling warm-up
    before = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before - len(df)} rows with NaN (warm-up period)")
    logger.info(f"Final feature set: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Save
    save_path = processed_dir / "features.parquet"
    df.to_parquet(save_path)
    logger.info(f"Saved features to {save_path}")

    return df
