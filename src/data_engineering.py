"""Data engineering: merging, cleaning, and feature construction."""

import logging
from pathlib import Path

import holidays as holidays_lib
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
    market_df: pd.DataFrame | None = None,
    cot_df: pd.DataFrame | None = None,
    alpha_vantage_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all data sources on date index.

    EIA, COT, and Alpha Vantage data (weekly/monthly) are resampled to daily
    via forward-fill before merging. All DataFrames are left-joined on the oil
    price base index, then forward-filled to handle weekends/holidays.

    Args:
        oil_df: Daily oil OHLCV data (WTI + Brent).
        fred_df: Daily/irregular FRED macro data.
        eia_dfs: Dict of EIA DataFrames with weekly/monthly frequency.
        market_df: Optional DataFrame of broad market close prices.
        cot_df: Optional weekly CFTC COT positioning DataFrame.
        alpha_vantage_df: Optional monthly Alpha Vantage economic indicators.

    Returns:
        Merged DataFrame with daily frequency and no gaps in the oil series.
    """
    # Start with oil prices as the base (daily trading days)
    merged = oil_df.copy()

    # Join FRED data (left join — FRED may have gaps on weekends)
    merged = merged.join(fred_df, how="left")

    # Resample EIA data to daily and join
    for name, eia_df in eia_dfs.items():
        eia_renamed = eia_df.rename(columns={"value": name})
        eia_daily = eia_renamed.resample("D").ffill()
        merged = merged.join(eia_daily, how="left")

    # Join broad market data (already daily, but may have NaN on holidays)
    if market_df is not None and not market_df.empty:
        merged = merged.join(market_df, how="left")

    # COT positioning data (weekly → resample to daily via ffill)
    if cot_df is not None and not cot_df.empty:
        cot_daily = cot_df.resample("D").ffill()
        merged = merged.join(cot_daily, how="left")

    # Alpha Vantage economic indicators (monthly → resample to daily via ffill)
    if alpha_vantage_df is not None and not alpha_vantage_df.empty:
        av_daily = alpha_vantage_df.resample("D").ffill()
        merged = merged.join(av_daily, how="left")

    # Forward-fill remaining NaN from weekends / different trading calendars
    merged = merged.ffill()

    # Drop rows where core oil data is still missing
    merged = merged.dropna(subset=oil_df.columns.tolist())

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
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Add momentum indicators: RSI and MACD.

    RSI (Relative Strength Index): Measures overbought (>70) / oversold (<30) conditions.
    MACD: Difference of two EMAs; the histogram captures trend acceleration.

    Args:
        df: Input DataFrame.
        target_col: Price column.
        rsi_period: Lookback window for RSI.
        macd_fast: Fast EMA span for MACD.
        macd_slow: Slow EMA span for MACD.
        macd_signal: Signal line EMA span for MACD.

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
    ema_fast = series.ewm(span=macd_fast, min_periods=macd_fast, adjust=False).mean()
    ema_slow = series.ewm(span=macd_slow, min_periods=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, min_periods=macd_signal, adjust=False).mean()

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


def add_crack_spread_features(
    df: pd.DataFrame,
    wti_col: str,
    gasoline_col: str | None = None,
    heating_oil_col: str | None = None,
) -> pd.DataFrame:
    """Compute refinery crack spreads.

    Crack spreads measure the margin between crude oil input costs and
    refined product prices. When margins are high, refineries ramp up
    crude purchases, creating upward demand pressure on the crude price.

    Gasoline and heating oil futures are quoted in USD/gallon.
    Crude oil is quoted in USD/barrel.  1 barrel = 42 gallons, so:

        Gasoline crack ($/bbl) = gasoline_close × 42 − wti_close
        Heating oil crack ($/bbl) = heating_oil_close × 42 − wti_close

    The 3-2-1 crack spread is the industry benchmark:
        3-2-1 = (2 × gasoline + 1 × heating_oil) × 42 / 3  −  wti_close

    Args:
        df: Input DataFrame.
        wti_col: WTI crude close price column (USD/bbl).
        gasoline_col: RBOB gasoline close price column (USD/gal), or None.
        heating_oil_col: Heating oil close price column (USD/gal), or None.

    Returns:
        DataFrame with crack spread columns added.
    """
    GALLONS_PER_BARREL = 42.0

    if gasoline_col and gasoline_col in df.columns:
        gasoline_bbl = df[gasoline_col] * GALLONS_PER_BARREL
        df["crack_gasoline"] = gasoline_bbl - df[wti_col]
        df["crack_gasoline_sma_14"] = df["crack_gasoline"].rolling(14, min_periods=14).mean()

    if heating_oil_col and heating_oil_col in df.columns:
        ho_bbl = df[heating_oil_col] * GALLONS_PER_BARREL
        df["crack_heating_oil"] = ho_bbl - df[wti_col]
        df["crack_heating_oil_sma_14"] = df["crack_heating_oil"].rolling(14, min_periods=14).mean()

    if (gasoline_col and gasoline_col in df.columns
            and heating_oil_col and heating_oil_col in df.columns):
        gasoline_bbl = df[gasoline_col] * GALLONS_PER_BARREL
        ho_bbl = df[heating_oil_col] * GALLONS_PER_BARREL
        df["crack_3_2_1"] = (2 * gasoline_bbl + ho_bbl) / 3 - df[wti_col]
        df["crack_3_2_1_sma_14"] = df["crack_3_2_1"].rolling(14, min_periods=14).mean()
        logger.info("Computed gasoline, heating-oil, and 3-2-1 crack spreads")

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
# Additional Momentum & Volume Indicators
# ─────────────────────────────────────────────

def add_atr_features(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    period: int = 14,
) -> pd.DataFrame:
    """Add Average True Range (ATR) and normalised ATR.

    ATR is a realised-volatility proxy that measures the average daily price
    range over a rolling window, accounting for overnight gaps.

    Args:
        df: Input DataFrame containing OHLC columns.
        high_col: Column name for daily highs.
        low_col: Column name for daily lows.
        close_col: Column name for close prices (also used as the name prefix).
        period: EWM span for smoothing the True Range.

    Returns:
        DataFrame with ``{close_col}_atr_{period}`` and
        ``{close_col}_atr_pct_{period}`` added.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    df[f"{close_col}_atr_{period}"] = atr
    df[f"{close_col}_atr_pct_{period}"] = atr / close

    return df


def add_williams_r(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    period: int = 14,
) -> pd.DataFrame:
    """Add Williams %R momentum oscillator.

    %R measures where the close sits within the rolling high-low range.
    Values near 0 indicate overbought; values near -100 indicate oversold.

    Args:
        df: Input DataFrame.
        high_col: Daily high column.
        low_col: Daily low column.
        close_col: Close price column (also used as name prefix).
        period: Rolling lookback window.

    Returns:
        DataFrame with ``{close_col}_williams_r_{period}`` added.
    """
    rolling_high = df[high_col].rolling(period, min_periods=period).max()
    rolling_low = df[low_col].rolling(period, min_periods=period).min()
    df[f"{close_col}_williams_r_{period}"] = (
        (rolling_high - df[close_col]) / (rolling_high - rolling_low) * -100
    )
    return df


def add_stochastic(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Add Stochastic %K and %D oscillators.

    %K = position of close within the k_period rolling range (0–100).
    %D = simple moving average of %K over d_period days (signal line).

    Args:
        df: Input DataFrame.
        high_col: Daily high column.
        low_col: Daily low column.
        close_col: Close price column (also used as name prefix).
        k_period: Lookback window for %K.
        d_period: Smoothing window for %D.

    Returns:
        DataFrame with ``{close_col}_stoch_k_{k_period}`` and
        ``{close_col}_stoch_d_{k_period}`` added.
    """
    rolling_high = df[high_col].rolling(k_period, min_periods=k_period).max()
    rolling_low = df[low_col].rolling(k_period, min_periods=k_period).min()
    stoch_k = (df[close_col] - rolling_low) / (rolling_high - rolling_low) * 100
    df[f"{close_col}_stoch_k_{k_period}"] = stoch_k
    df[f"{close_col}_stoch_d_{k_period}"] = stoch_k.rolling(d_period, min_periods=d_period).mean()
    return df


def add_cmf(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    period: int = 20,
) -> pd.DataFrame:
    """Add Chaikin Money Flow (CMF).

    CMF measures volume-weighted buying vs. selling pressure over a rolling
    window. Positive values indicate accumulation; negative values indicate
    distribution. Useful as a leading indicator of crude demand shifts.

    Args:
        df: Input DataFrame.
        high_col: Daily high column.
        low_col: Daily low column.
        close_col: Close price column (also used as name prefix).
        volume_col: Volume column.
        period: Rolling window for CMF calculation.

    Returns:
        DataFrame with ``{close_col}_cmf_{period}`` added.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]

    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume

    df[f"{close_col}_cmf_{period}"] = (
        mfv.rolling(period, min_periods=period).sum()
        / volume.rolling(period, min_periods=period).sum()
    )
    return df


def add_rolling_correlations(
    df: pd.DataFrame,
    target_col: str,
    pair_names: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling correlations of the target with cross-asset price series.

    Rolling correlations capture regime-dependent relationships (e.g., the
    WTI–DXY correlation flips sign across risk-on/risk-off regimes).

    Args:
        df: Input DataFrame. Each pair must have a ``{pair}_close`` column.
        target_col: Target price column (WTI close).
        pair_names: List of descriptive names; looks up ``{name}_close`` in df.
        windows: Rolling window sizes (days).

    Returns:
        DataFrame with ``corr_{pair}_{window}`` columns added.
    """
    target = df[target_col]
    for pair in pair_names:
        pair_col = f"{pair}_close"
        if pair_col not in df.columns:
            logger.warning(f"Rolling correlation skipped: '{pair_col}' not in columns")
            continue
        for window in windows:
            df[f"corr_{pair}_{window}"] = target.rolling(window, min_periods=window).corr(df[pair_col])
    return df


def add_momentum_windows(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Add time-series momentum features (multi-period returns).

    Captures medium-to-long-term price trends: 1-month, 1-quarter, and
    6-month simple returns.  Momentum strategies in commodities have
    historically earned positive risk premia.

    Args:
        df: Input DataFrame.
        target_col: Price column.
        windows: List of lookback periods (e.g., [21, 63, 126]).

    Returns:
        DataFrame with ``{target_col}_momentum_{window}`` columns added.
    """
    price = df[target_col]
    for window in windows:
        df[f"{target_col}_momentum_{window}"] = (price - price.shift(window)) / price.shift(window)
    return df


def add_seasonal_flags(
    df: pd.DataFrame,
    seasonal_cfg: dict,
) -> pd.DataFrame:
    """Add binary seasonality flags.

    Three demand-seasonality signals:
    - driving_season: 1 between US Memorial Day and Labor Day (gasoline peak)
    - heating_season: 1 during Oct–Mar (distillate/heating oil demand peak)
    - us_holiday: 1 on US federal holidays (thin-liquidity, exaggerated moves)

    Args:
        df: Input DataFrame with DatetimeIndex.
        seasonal_cfg: Dict with boolean keys ``driving_season``,
            ``heating_season``, and ``us_holidays``.

    Returns:
        DataFrame with flag columns added for each enabled flag.
    """
    idx = df.index

    if seasonal_cfg.get("driving_season"):
        driving = pd.Series(0, index=idx)
        for year in idx.year.unique():
            # Memorial Day = last Monday of May
            memorial_day = pd.Timestamp(year=year, month=5, day=31) + pd.offsets.Week(weekday=0, n=0) - pd.offsets.Week(weekday=0, n=0)
            # Walk back to find last Monday of May
            md = pd.Timestamp(year=year, month=5, day=31)
            while md.dayofweek != 0:
                md -= pd.Timedelta(days=1)
            # Labor Day = first Monday of September
            ld = pd.Timestamp(year=year, month=9, day=1)
            while ld.dayofweek != 0:
                ld += pd.Timedelta(days=1)
            driving.loc[(idx >= md) & (idx <= ld)] = 1
        df["driving_season"] = driving.astype(int)

    if seasonal_cfg.get("heating_season"):
        df["heating_season"] = idx.month.isin([10, 11, 12, 1, 2, 3]).astype(int)

    if seasonal_cfg.get("us_holidays"):
        years = range(idx.year.min(), idx.year.max() + 1)
        us_holiday_dates = set(holidays_lib.country_holidays("US", years=years).keys())
        df["us_holiday"] = idx.normalize().map(
            lambda d: 1 if d.date() in us_holiday_dates else 0
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
    - Momentum indicators: RSI, MACD (configurable), ATR, Williams %R, Stochastic, CMF
    - Rolling cross-asset correlations (WTI vs DXY, gold, S&P, copper, nat gas)
    - Time-series momentum (1M, 1Q, 6M returns)
    - Seasonality flags (driving season, heating season, US holidays)
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

    # ── Feature config params ──────────────────────────────────────────────
    macd_fast = feat_cfg.get("macd_fast", 12)
    macd_slow = feat_cfg.get("macd_slow", 26)
    macd_signal = feat_cfg.get("macd_signal", 9)
    atr_period = feat_cfg.get("atr_period", 14)
    williams_r_period = feat_cfg.get("williams_r_period", 14)
    stoch_k = feat_cfg.get("stochastic_k_period", 14)
    stoch_d = feat_cfg.get("stochastic_d_period", 3)
    cmf_period = feat_cfg.get("cmf_period", 20)
    corr_windows = feat_cfg.get("correlation_windows", [30, 90])
    corr_pairs = feat_cfg.get("correlation_pairs", [])
    mom_windows = feat_cfg.get("momentum_windows", [21, 63, 126])
    seasonal_cfg = feat_cfg.get("seasonal_flags", {})

    # ── Load raw data ─────────────────────────────────
    oil_df = pd.read_parquet(raw_dir / "oil_prices.parquet")
    fred_df = pd.read_parquet(raw_dir / "fred_macro.parquet")

    eia_dfs = {}
    for name in config["data"].get("eia_series", {}).keys():
        eia_path = raw_dir / f"eia_{name}.parquet"
        if eia_path.exists():
            eia_dfs[name] = pd.read_parquet(eia_path)

    market_df = None
    market_path = raw_dir / "market_tickers.parquet"
    if market_path.exists():
        market_df = pd.read_parquet(market_path)
        logger.info(f"Loaded market data: {list(market_df.columns)}")

    cot_df = None
    cot_path = raw_dir / "cot.parquet"
    if cot_path.exists():
        cot_df = pd.read_parquet(cot_path)
        logger.info(f"Loaded COT data: {len(cot_df)} rows")

    alpha_vantage_df = None
    av_path = raw_dir / "alpha_vantage.parquet"
    if av_path.exists():
        alpha_vantage_df = pd.read_parquet(av_path)
        logger.info(f"Loaded Alpha Vantage data: {list(alpha_vantage_df.columns)}")

    logger.info(f"Oil price columns: {list(oil_df.columns)}")

    # ── Merge ─────────────────────────────────────────
    df = merge_datasets(
        oil_df, fred_df, eia_dfs, market_df,
        cot_df=cot_df,
        alpha_vantage_df=alpha_vantage_df,
    )
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

    # Macro columns for exogenous lags (FRED + yield indices from market data)
    macro_cols = [
        c for c in [
            "usd_index", "cpi", "fed_funds", "t10y2y",
            "t3m_yield_close", "t5y_yield_close",
            "t10y_yield_close", "t30y_yield_close",
            # Financial conditions & systemic risk
            "nfci", "stlfsi", "epu_us", "epu_global",
            # Real economy / demand
            "pmi_mfg", "retail_gas", "miles_driven",
            # Fed balance sheet / liquidity
            "fed_balance", "rrp",
            # Credit / risk appetite
            "ig_spread", "ted_spread",
        ]
        if c in df.columns
    ]

    # All close columns from market tickers (for returns + lags)
    market_close_cols = [c for c in df.columns if c.endswith("_close") and c != target_col]

    # Crack spread constituent columns
    gasoline_col = "gasoline_close" if "gasoline_close" in df.columns else None
    heating_oil_col = "heating_oil_close" if "heating_oil_close" in df.columns else None

    # ── Feature Engineering ───────────────────────────
    # 1. Autoregressive lags: t-1, t-3, t-7, t-30
    df = add_lag_features(df, target_col, [1, 3, 7, 30])

    # 2. Exogenous lags (macro + yield impact is delayed)
    df = add_exogenous_lags(df, macro_cols, lags=[7, 14])

    # 3a. COT positioning lags (positioning is a leading indicator, delayed release)
    cot_cols = [c for c in df.columns if c.startswith("cot_")]
    if cot_cols:
        df = add_exogenous_lags(df, cot_cols, lags=[7, 14])

    # 3b. Alpha Vantage economic indicator lags
    av_cols = [c for c in df.columns if c.startswith("av_")]
    if av_cols:
        df = add_exogenous_lags(df, av_cols, lags=[7, 14])

    # 3c. Key market lags (energy commodities and currencies move together with crude)
    market_lag_cols = [
        c for c in market_close_cols
        if any(k in c for k in [
            "nat_gas", "gasoline", "heating_oil",
            "eur_usd", "usd_cad", "usd_nok", "usd_rub",
            "gold", "copper", "sp500",
        ])
    ]
    df = add_exogenous_lags(df, market_lag_cols, lags=[1, 7])

    # 4. Rolling SMAs, EMAs, and volatility: 14, 50, 200 days
    df = add_rolling_features(df, target_col, [14, 50, 200])

    # 5. Bollinger Bands (20-day, 2 std)
    df = add_bollinger_bands(df, target_col, window=20, n_std=2.0)

    # 6. Momentum: RSI, MACD (configurable params)
    df = add_momentum_features(
        df, target_col,
        rsi_period=feat_cfg.get("rsi_period", 14),
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    )

    # 6b. ATR, Williams %R, Stochastic, CMF (use WTI OHLCV)
    wti_high = "CL=F_high" if "CL=F_high" in df.columns else None
    wti_low = "CL=F_low" if "CL=F_low" in df.columns else None
    wti_vol = "CL=F_volume" if "CL=F_volume" in df.columns else None
    if wti_col and wti_high and wti_low:
        df = add_atr_features(df, wti_high, wti_low, wti_col, atr_period)
        df = add_williams_r(df, wti_high, wti_low, wti_col, williams_r_period)
        df = add_stochastic(df, wti_high, wti_low, wti_col, stoch_k, stoch_d)
        if wti_vol:
            df = add_cmf(df, wti_high, wti_low, wti_col, wti_vol, cmf_period)

    # 6c. Rolling cross-asset correlations
    if corr_pairs:
        df = add_rolling_correlations(df, target_col, corr_pairs, corr_windows)

    # 6d. Time-series momentum (multi-period returns)
    if mom_windows:
        df = add_momentum_windows(df, target_col, mom_windows)

    # 6e. Seasonality flags
    if seasonal_cfg:
        df = add_seasonal_flags(df, seasonal_cfg)

    # 7. Price level features (needs MAs to be computed first)
    df = add_price_level_features(df, target_col)

    # 8. Calendar features
    df = add_calendar_features(df)

    # 9. Returns and log-returns — oil close prices + market close prices
    all_return_cols = list(close_cols.values()) + market_close_cols
    df = add_return_features(df, all_return_cols)

    # 10. Brent–WTI spread
    if wti_col and brent_col:
        df = add_spread_features(df, wti_col, brent_col)

    # 11. Crack spreads (refinery margin proxies for crude demand)
    if wti_col:
        df = add_crack_spread_features(df, wti_col, gasoline_col, heating_oil_col)

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
