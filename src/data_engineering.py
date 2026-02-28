"""Data engineering: merging, cleaning, and feature construction."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import get_project_root

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────

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
        eia_renamed = eia_df.rename(columns={"value": name})
        eia_daily = eia_renamed.resample("D").ffill()
        merged = merged.join(eia_daily, how="left")

    # Forward-fill remaining NaN from holidays/weekends
    merged = merged.ffill()

    # Drop rows where we still have no data at all
    merged = merged.dropna()

    logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


# ─────────────────────────────────────────────
# Autoregressive Lag Features
# ─────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Create lagged versions of the target (autoregressive lags).

    Args:
        df: Input DataFrame.
        target_col: Column to create lags for.
        lags: List of lag periods (e.g., [1, 3, 7, 30]).

    Returns:
        DataFrame with new lag columns added.
    """
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_exogenous_lags(
    df: pd.DataFrame,
    exog_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Create lagged versions of exogenous variables.

    Macro/fundamental data affects oil prices with a delay.
    Lagging these variables explicitly captures that transmission mechanism.

    Args:
        df: Input DataFrame.
        exog_cols: List of exogenous column names to lag.
        lags: List of lag periods (e.g., [7, 14]).

    Returns:
        DataFrame with lagged exogenous columns added.
    """
    for col in exog_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ─────────────────────────────────────────────
# Rolling Window Statistics
# ─────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Calculate rolling SMA, EMA, and volatility.

    SMA (Simple Moving Average): Equal weight over the window.
    EMA (Exponential Moving Average): More weight on recent days, reacts faster to shocks.
    Std Dev: Proxy for rolling volatility / market uncertainty.

    Args:
        df: Input DataFrame.
        target_col: Column to compute rolling stats for.
        windows: List of window sizes (e.g., [14, 50, 200]).

    Returns:
        DataFrame with SMA, EMA, and std columns added.
    """
    series = df[target_col]

    for window in windows:
        # Simple Moving Average
        df[f"{target_col}_sma_{window}"] = (
            series.rolling(window=window, min_periods=window).mean()
        )
        # Exponential Moving Average
        df[f"{target_col}_ema_{window}"] = (
            series.ewm(span=window, min_periods=window, adjust=False).mean()
        )
        # Rolling Volatility (std dev)
        df[f"{target_col}_std_{window}"] = (
            series.rolling(window=window, min_periods=window).std()
        )

    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    target_col: str,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Band features.

    Bollinger Bands = SMA ± n_std × rolling_std.
    The %B indicator (position within the bands) measures whether price is
    near the upper or lower extreme of recent volatility.

    Args:
        df: Input DataFrame.
        target_col: Price column.
        window: Rolling window in days.
        n_std: Number of standard deviations for the bands.

    Returns:
        DataFrame with upper/lower band and %B added.
    """
    sma = df[target_col].rolling(window=window, min_periods=window).mean()
    std = df[target_col].rolling(window=window, min_periods=window).std()

    upper = sma + n_std * std
    lower = sma - n_std * std

    df[f"{target_col}_bb_upper_{window}"] = upper
    df[f"{target_col}_bb_lower_{window}"] = lower
    # %B: 0 = at lower band, 1 = at upper band, >1 or <0 = outside bands
    df[f"{target_col}_bb_pct_{window}"] = (df[target_col] - lower) / (upper - lower)
    # Band width: proxy for volatility regime
    df[f"{target_col}_bb_width_{window}"] = (upper - lower) / sma

    return df


def add_momentum_features(
    df: pd.DataFrame,
    target_col: str,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Add momentum indicators: RSI and MACD.

    RSI (Relative Strength Index): Measures overbought (>70) / oversold (<30) conditions.
    MACD: Difference of two EMAs; the histogram captures trend acceleration.

    Args:
        df: Input DataFrame.
        target_col: Price column.
        rsi_period: Lookback window for RSI.

    Returns:
        DataFrame with RSI, MACD line, signal line, and histogram added.
    """
    series = df[target_col]

    # ── RSI ──────────────────────────────────────────
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"{target_col}_rsi_{rsi_period}"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────
    ema_12 = series.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()

    df[f"{target_col}_macd"] = macd_line
    df[f"{target_col}_macd_signal"] = signal_line
    df[f"{target_col}_macd_hist"] = macd_line - signal_line

    return df


# ─────────────────────────────────────────────
# Calendar Features
# ─────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the DatetimeIndex.

    Creates cyclic integer features suitable for pyGAM's cyclic splines.

    Returns:
        DataFrame with day_of_week, month, day_of_year, quarter, is_month_end.
    """
    df["day_of_week"] = df.index.dayofweek      # 0=Mon … 6=Sun
    df["month"] = df.index.month                # 1–12
    df["day_of_year"] = df.index.dayofyear      # 1–366
    df["quarter"] = df.index.quarter            # 1–4
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)
    return df


# ─────────────────────────────────────────────
# Financial Transformations
# ─────────────────────────────────────────────

def add_return_features(
    df: pd.DataFrame,
    price_cols: list[str],
) -> pd.DataFrame:
    """Compute daily percentage and log returns.

    Returns stabilise non-stationary price series and are standard
    inputs in financial modelling.

    Args:
        df: Input DataFrame.
        price_cols: List of price columns to compute returns for.

    Returns:
        DataFrame with pct_change and log_return columns added.
    """
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_pct_change"] = df[col].pct_change()
            df[f"{col}_log_return"] = np.log(df[col] / df[col].shift(1))
    return df


def add_spread_features(
    df: pd.DataFrame,
    wti_col: str,
    brent_col: str,
) -> pd.DataFrame:
    """Compute the Brent–WTI price spread.

    The spread reflects regional supply/demand imbalances, infrastructure
    constraints, and geopolitical risk premiums. It widens and narrows
    based on US export policy, pipeline capacity, and OPEC decisions.

    Args:
        df: Input DataFrame.
        wti_col: WTI close price column name.
        brent_col: Brent close price column name.

    Returns:
        DataFrame with spread, spread_pct, and spread_ratio added.
    """
    if wti_col not in df.columns or brent_col not in df.columns:
        logger.warning(
            f"Spread features skipped: '{wti_col}' or '{brent_col}' not in columns."
        )
        return df

    spread = df[brent_col] - df[wti_col]
    df["brent_wti_spread"] = spread
    df["brent_wti_spread_pct"] = spread / df[wti_col]
    df["brent_wti_ratio"] = df[brent_col] / df[wti_col]
    # 14-day rolling mean and std of the spread (is it widening/narrowing?)
    df["brent_wti_spread_sma_14"] = spread.rolling(14, min_periods=14).mean()
    df["brent_wti_spread_std_14"] = spread.rolling(14, min_periods=14).std()

    return df


def add_price_level_features(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Add features derived from price levels relative to recent history.

    These capture whether the market is above or below meaningful reference
    points — important signals for mean-reversion and trend-following.

    Args:
        df: Input DataFrame.
        target_col: Price column.

    Returns:
        DataFrame with distance-from-MA and rolling-high/low features.
    """
    price = df[target_col]

    # Distance of current price from its moving averages
    for window in [14, 50, 200]:
        sma_col = f"{target_col}_sma_{window}"
        if sma_col in df.columns:
            df[f"{target_col}_dist_sma_{window}"] = (price - df[sma_col]) / df[sma_col]

    # 52-week (≈252 trading day) rolling high and low — classic range reference
    df[f"{target_col}_52w_high"] = price.rolling(252, min_periods=252).max()
    df[f"{target_col}_52w_low"] = price.rolling(252, min_periods=252).min()
    df[f"{target_col}_52w_pct_range"] = (
        (price - df[f"{target_col}_52w_low"])
        / (df[f"{target_col}_52w_high"] - df[f"{target_col}_52w_low"])
    )

    return df


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────

def build_features(config: dict) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Loads raw parquet files, merges, creates all features, drops NaN rows,
    and saves the result to data/processed/features.parquet.

    Feature groups created:
    - Autoregressive lags of the target (t-1, t-3, t-7, t-30)
    - Exogenous lags of macro variables (7d, 14d)
    - Rolling SMA, EMA, and volatility (14, 50, 200 day windows)
    - Bollinger Bands and %B (20-day)
    - Momentum indicators: RSI(14) and MACD(12,26,9)
    - Calendar features with cyclic encoding (month, day_of_week, day_of_year)
    - Daily returns and log-returns
    - Brent–WTI spread and ratio
    - Price level features (distance from MAs, 52-week range position)

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

    # ── Load raw data ─────────────────────────────────
    oil_df = pd.read_parquet(raw_dir / "oil_prices.parquet")
    fred_df = pd.read_parquet(raw_dir / "fred_macro.parquet")

    eia_dfs = {}
    for name in config["data"].get("eia_series", {}).keys():
        eia_path = raw_dir / f"eia_{name}.parquet"
        if eia_path.exists():
            eia_dfs[name] = pd.read_parquet(eia_path)

    logger.info(f"Oil price columns: {list(oil_df.columns)}")

    # ── Merge ─────────────────────────────────────────
    df = merge_datasets(oil_df, fred_df, eia_dfs)
    logger.info(f"Merged columns: {list(df.columns)}")

    # ── Resolve target column (case-insensitive) ──────
    target_col = None
    for col in df.columns:
        if col.lower() == target.lower():
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    logger.info(f"Target column: {target_col}")

    # ── Resolve close price columns for all tickers ───
    oil_tickers = config["data"]["oil_tickers"]
    close_cols = {}
    for t in oil_tickers:
        for col in df.columns:
            if col.lower() == f"{t}_close".lower():
                close_cols[t] = col
                break

    # Identify WTI and Brent (if both present)
    wti_col = close_cols.get("CL=F")
    brent_col = close_cols.get("BZ=F")

    # Macro columns for exogenous lags
    macro_cols = [c for c in ["usd_index", "cpi", "fed_funds", "t10y2y"] if c in df.columns]

    # ── Feature Engineering ───────────────────────────
    # 1. Autoregressive lags: t-1, t-3, t-7, t-30
    df = add_lag_features(df, target_col, [1, 3, 7, 30])

    # 2. Exogenous lags (macro impact is delayed)
    df = add_exogenous_lags(df, macro_cols, lags=[7, 14])

    # 3. Rolling SMAs, EMAs, and volatility: 14, 50, 200 days
    df = add_rolling_features(df, target_col, [14, 50, 200])

    # 4. Bollinger Bands (20-day, 2 std)
    df = add_bollinger_bands(df, target_col, window=20, n_std=2.0)

    # 5. Momentum: RSI(14), MACD(12,26,9)
    df = add_momentum_features(df, target_col, rsi_period=14)

    # 6. Price level features (needs MAs to be computed first)
    df = add_price_level_features(df, target_col)

    # 7. Calendar features
    df = add_calendar_features(df)

    # 8. Returns and log-returns for all close prices
    df = add_return_features(df, list(close_cols.values()))

    # 9. Brent–WTI spread
    if wti_col and brent_col:
        df = add_spread_features(df, wti_col, brent_col)

    # ── Drop warm-up NaN rows ─────────────────────────
    before = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before - len(df)} rows (warm-up period)")
    logger.info(f"Final feature set: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Save ──────────────────────────────────────────
    save_path = processed_dir / "features.parquet"
    df.to_parquet(save_path)
    logger.info(f"Saved features to {save_path}")

    return df
