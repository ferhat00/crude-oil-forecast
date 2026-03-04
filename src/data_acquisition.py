"""Data acquisition from Yahoo Finance, FRED, and EIA APIs."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

from src.config_loader import (
    get_alpha_vantage_params,
    get_eia_params,
    get_fred_client,
    get_project_root,
)

logger = logging.getLogger(__name__)


def fetch_oil_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data for oil futures from Yahoo Finance.

    Args:
        tickers: List of Yahoo Finance tickers (e.g., ["CL=F", "BZ=F"]).
        start: Start date string (YYYY-MM-DD).
        end: End date string or None for today.
        save_path: Path to save parquet file.

    Returns:
        DataFrame with columns like 'CL=F_close', 'CL=F_volume', etc.
    """
    logger.info(f"Fetching oil prices for {tickers} from {start} to {end or 'today'}")

    # Download all tickers together
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if data.empty:
        raise ValueError("No oil price data could be downloaded")

    # Handle MultiIndex columns — yfinance returns (field, ticker) order
    # Build a lookup so we can restore original ticker casing (yfinance may lowercase)
    ticker_map = {t.lower(): t for t in tickers}
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex: ('Close', 'cl=f') -> 'CL=F_close'
        new_columns = []
        for field, ticker_lower in data.columns:
            original_ticker = ticker_map.get(ticker_lower.lower(), ticker_lower)
            new_columns.append(f"{original_ticker}_{field.lower()}")
        data.columns = new_columns
    else:
        # Single ticker returned flat columns like 'Close', 'Volume'
        if len(tickers) == 1:
            ticker = tickers[0]
            data.columns = [f"{ticker}_{col.lower()}" for col in data.columns]
        else:
            raise ValueError("Expected MultiIndex columns for multiple tickers")

    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info(f"Saved oil prices to {save_path} ({len(df)} rows)")

    return df


def fetch_fred_series(
    fred_client: Fred,
    series_map: dict[str, str],
    start: str,
    end: str | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Download macroeconomic series from FRED.

    Args:
        fred_client: Initialized fredapi.Fred instance.
        series_map: Dict mapping column names to FRED series IDs.
            e.g., {"usd_index": "DTWEXBGS", "fed_funds": "DFF"}
        start: Start date string.
        end: End date string or None.
        save_path: Path to save parquet file.

    Returns:
        DataFrame with columns named by series_map keys.
    """
    logger.info(f"Fetching FRED series: {list(series_map.keys())}")

    frames = {}
    for name, series_id in series_map.items():
        try:
            series = fred_client.get_series(
                series_id, observation_start=start, observation_end=end
            )
            frames[name] = series
            logger.info(f"  {name} ({series_id}): {len(series)} observations")
        except Exception as e:
            logger.warning(f"  Failed to fetch {name} ({series_id}): {e}")

    if not frames:
        raise ValueError("No FRED data could be downloaded")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info(f"Saved FRED data to {save_path} ({len(df)} rows)")

    return df


def fetch_eia_data(
    base_url: str,
    api_key: str,
    series_id: str,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Download energy data from EIA API v2.

    Args:
        base_url: EIA API base URL.
        api_key: EIA API key.
        series_id: EIA series ID (e.g., "PET.WTTSTUS1.W").
        save_path: Path to save parquet file.

    Returns:
        DataFrame with 'date' index and 'value' column.
    """
    logger.info(f"Fetching EIA series: {series_id}")

    url = f"{base_url}/seriesid/{series_id}"
    params = {
        "api_key": api_key,
        "out": "json",
    }

    max_retries = 3
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            if attempt < max_retries:
                wait = 10 * attempt
                logger.warning(
                    f"  EIA request failed (attempt {attempt}/{max_retries}): {exc}, "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            raise
        if response.status_code in (502, 503, 504) and attempt < max_retries:
            wait = 10 * attempt
            logger.warning(
                f"  EIA HTTP {response.status_code} (attempt {attempt}/{max_retries}), "
                f"retrying in {wait}s..."
            )
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    data = response.json()

    # EIA API v2 returns data in response.data
    if "response" in data and "data" in data["response"]:
        records = data["response"]["data"]
    else:
        raise ValueError(f"Unexpected EIA API response structure for {series_id}")

    df = pd.DataFrame(records)

    # Parse the period column as date
    if "period" in df.columns:
        df["date"] = pd.to_datetime(df["period"])
        df = df.set_index("date").sort_index()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    # Keep only the value column
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["value"]]

    logger.info(f"  {series_id}: {len(df)} observations")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info(f"Saved EIA data to {save_path}")

    return df


def fetch_market_tickers(
    ticker_map: dict[str, str],
    start: str,
    end: str | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Download daily close prices for a broad set of market instruments.

    Handles currencies, energy commodities, equities, ETFs, and yield indices
    in a single yfinance batch call.  Only the adjusted Close price is kept —
    OHLCV is not needed for these exogenous signals.

    Args:
        ticker_map: Dict mapping descriptive name → Yahoo Finance ticker.
            e.g., {"nat_gas": "NG=F", "sp500": "^GSPC", "eur_usd": "EURUSD=X"}
        start: Start date string (YYYY-MM-DD).
        end: End date string or None for today.
        save_path: If given, saves result as parquet.

    Returns:
        DataFrame with DatetimeIndex and columns named ``{name}_close``.
        Tickers that fail to download are logged and skipped.
    """
    names = list(ticker_map.keys())
    tickers = list(ticker_map.values())
    logger.info(
        f"Fetching {len(tickers)} market tickers from {start} to {end or 'today'}"
    )

    # Case-insensitive lookup: yfinance-lowercased ticker → original ticker
    ticker_lower_to_orig = {t.lower(): t for t in tickers}
    ticker_to_name = {t: n for n, t in ticker_map.items()}

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if data.empty:
        logger.warning("No market data returned from yfinance — skipping.")
        return pd.DataFrame()

    # Extract Close prices only from the (field, ticker) MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data["Close"].copy()
    else:
        # Single-ticker fallback
        close_df = data[["Close"]].copy()
        close_df.columns = [tickers[0]]

    # Rename columns: yfinance-altered ticker → "{descriptive_name}_close"
    rename_map = {}
    for col in close_df.columns:
        orig_ticker = ticker_lower_to_orig.get(str(col).lower(), str(col))
        desc_name = ticker_to_name.get(orig_ticker, orig_ticker)
        rename_map[col] = f"{desc_name}_close"
    close_df = close_df.rename(columns=rename_map)

    close_df.index = pd.to_datetime(close_df.index)
    close_df.index.name = "date"

    for col in close_df.columns:
        n_valid = close_df[col].notna().sum()
        n_total = len(close_df)
        if n_valid < 0.5 * n_total:
            logger.warning(f"  {col}: sparse — only {n_valid}/{n_total} valid rows")
        else:
            logger.info(f"  {col}: {n_valid} valid rows")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        close_df.to_parquet(save_path)
        logger.info(
            f"Saved market data to {save_path} "
            f"({len(close_df)} rows, {len(close_df.columns)} series)"
        )

    return close_df


def fetch_cot_data(
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch CFTC Commitments of Traders data for WTI crude oil.

    Uses the CFTC public Socrata API (no API key required) to download the
    Disaggregated Futures-only report filtered to CRUDE OIL, LIGHT SWEET on NYMEX.

    Returns:
        DataFrame indexed by weekly report date with positioning columns:
        cot_open_interest, cot_commercial_long/short/net,
        cot_mm_long/short/net, cot_mm_long_pct.
    """
    logger.info("Fetching CFTC COT data for WTI crude oil")

    # CFTC Socrata API — Disaggregated Futures Only report
    base_url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    all_records: list[dict] = []
    limit = 50000
    offset = 0

    _market_name = "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE"

    while True:
        params = {
            "$where": f"market_and_exchange_names='{_market_name}'",
            "$limit": limit,
            "$offset": offset,
            "$order": "report_date_as_yyyy_mm_dd ASC",
        }
        for attempt in range(1, 4):
            try:
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                if attempt < 3:
                    wait = 10 * attempt
                    logger.warning(f"  COT request failed (attempt {attempt}/3): {exc}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        records = response.json()
        if not records:
            break
        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit

    df = pd.DataFrame(all_records)

    col_map = {
        "report_date_as_yyyy_mm_dd": "date",
        "open_interest_all": "cot_open_interest",
        "prod_merc_positions_long": "cot_commercial_long",
        "prod_merc_positions_short": "cot_commercial_short",
        "m_money_positions_long_all": "cot_mm_long",
        "m_money_positions_short_all": "cot_mm_short",
    }
    present = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(present.keys())].rename(columns=present)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived positioning metrics
    if {"cot_mm_long", "cot_mm_short", "cot_open_interest"}.issubset(df.columns):
        df["cot_mm_net"] = df["cot_mm_long"] - df["cot_mm_short"]
        df["cot_mm_long_pct"] = df["cot_mm_long"] / df["cot_open_interest"]
    if {"cot_commercial_long", "cot_commercial_short"}.issubset(df.columns):
        df["cot_commercial_net"] = df["cot_commercial_long"] - df["cot_commercial_short"]

    logger.info(f"  COT data: {len(df)} weekly observations")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info(f"Saved COT data to {save_path}")

    return df


def fetch_alpha_vantage_data(
    base_url: str,
    api_key: str,
    series_map: dict[str, str],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch economic indicator series from Alpha Vantage.

    Args:
        base_url: Alpha Vantage base URL.
        api_key: Alpha Vantage API key.
        series_map: Dict mapping output column name to AV function name
            e.g. {"av_consumer_sentiment": "CONSUMER_SENTIMENT"}.
        save_path: Optional path to save merged parquet.

    Returns:
        DataFrame indexed by date with one column per series.
    """
    frames: list[pd.DataFrame] = []

    for col_name, function_name in series_map.items():
        logger.info(f"Fetching Alpha Vantage series: {function_name}")
        params = {"function": function_name, "apikey": api_key}

        for attempt in range(1, 4):
            try:
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                if attempt < 3:
                    wait = 10 * attempt
                    logger.warning(
                        f"  AV {function_name} failed (attempt {attempt}/3): {exc}, "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        data = response.json()
        records = data.get("data", [])
        if not records:
            logger.warning(f"  Alpha Vantage {function_name}: no data returned")
            continue

        series_df = pd.DataFrame(records)
        series_df["date"] = pd.to_datetime(series_df["date"])
        series_df = series_df.set_index("date").sort_index()
        series_df[col_name] = pd.to_numeric(series_df["value"], errors="coerce")
        frames.append(series_df[[col_name]])
        logger.info(f"  {function_name}: {len(series_df)} observations")

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for other in frames[1:]:
        df = df.join(other, how="outer")
    df = df.sort_index()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info(f"Saved Alpha Vantage data to {save_path}")

    return df


def acquire_all(config: dict) -> dict[str, pd.DataFrame]:
    """Run all data acquisition steps.

    Args:
        config: Project configuration dictionary.

    Returns:
        Dict with keys 'oil', 'fred', 'market', one key per EIA series,
        and optionally 'cot' and 'alpha_vantage'.
    """
    root = get_project_root(config)
    raw_dir = root / "data" / "raw"
    data_cfg = config["data"]
    start = data_cfg["start_date"]
    end = data_cfg.get("end_date")
    result = {}

    # Oil prices (WTI + Brent, full OHLCV)
    result["oil"] = fetch_oil_prices(
        tickers=data_cfg["oil_tickers"],
        start=start,
        end=end,
        save_path=raw_dir / "oil_prices.parquet",
    )

    # FRED macro data
    fred_client = get_fred_client(config)
    result["fred"] = fetch_fred_series(
        fred_client=fred_client,
        series_map=data_cfg["fred_series"],
        start=start,
        end=end,
        save_path=raw_dir / "fred_macro.parquet",
    )

    # CFTC COT positioning data (no API key required)
    if config.get("cot", {}).get("enabled", False):
        cot_path = raw_dir / "cot.parquet"
        try:
            result["cot"] = fetch_cot_data(save_path=cot_path)
        except Exception as exc:
            if cot_path.exists():
                logger.warning(f"COT fetch failed ({exc}); loading cached data from {cot_path}")
                result["cot"] = pd.read_parquet(cot_path)
            else:
                raise

    # Alpha Vantage economic indicators
    av_cfg = config.get("alpha_vantage", {})
    av_key = av_cfg.get("api_key", "")
    if av_key and not av_key.startswith("YOUR_"):
        av_path = raw_dir / "alpha_vantage.parquet"
        try:
            av_base_url, av_api_key, av_series_map = get_alpha_vantage_params(config)
            result["alpha_vantage"] = fetch_alpha_vantage_data(
                base_url=av_base_url,
                api_key=av_api_key,
                series_map=av_series_map,
                save_path=av_path,
            )
        except Exception as exc:
            if av_path.exists():
                logger.warning(
                    f"Alpha Vantage fetch failed ({exc}); loading cached data from {av_path}"
                )
                result["alpha_vantage"] = pd.read_parquet(av_path)
            else:
                logger.warning(f"Alpha Vantage fetch failed and no cache available: {exc}")

    # EIA energy data
    base_url, api_key = get_eia_params(config)
    for name, series_id in data_cfg.get("eia_series", {}).items():
        save_path = raw_dir / f"eia_{name}.parquet"
        try:
            result[f"eia_{name}"] = fetch_eia_data(
                base_url=base_url,
                api_key=api_key,
                series_id=series_id,
                save_path=save_path,
            )
        except Exception as exc:
            if save_path.exists():
                logger.warning(
                    f"EIA fetch failed ({exc}); loading cached data from {save_path}"
                )
                result[f"eia_{name}"] = pd.read_parquet(save_path)
            else:
                raise

    # Broad market tickers (currencies, commodities, equities, rates)
    market_tickers = config.get("market_tickers", {})
    if market_tickers:
        result["market"] = fetch_market_tickers(
            ticker_map=market_tickers,
            start=start,
            end=end,
            save_path=raw_dir / "market_tickers.parquet",
        )

    logger.info(f"Data acquisition complete. Datasets: {list(result.keys())}")
    return result
