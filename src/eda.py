"""Exploratory data analysis: decomposition, ACF/PACF, and visualization."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from src.config_loader import get_project_root

logger = logging.getLogger(__name__)


def plot_price_history(
    df: pd.DataFrame,
    price_cols: list[str],
    save_path: str | Path = "outputs/figures/price_history.png",
) -> None:
    """Plot historical oil price time series."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in price_cols:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=0.8)
    ax.set_title("Crude Oil Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved price history plot to {save_path}")


def plot_decomposition(
    series: pd.Series,
    period: int = 252,
    save_path: str | Path = "outputs/figures/decomposition.png",
) -> None:
    """Decompose time series into trend, seasonal, and residual components."""
    result = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed", linewidth=0.8)
    result.trend.plot(ax=axes[1], title="Trend", linewidth=0.8)
    result.seasonal.plot(ax=axes[2], title="Seasonal", linewidth=0.8)
    result.resid.plot(ax=axes[3], title="Residual", linewidth=0.8)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Time Series Decomposition", y=1.02, fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved decomposition plot to {save_path}")


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 60,
    save_path: str | Path = "outputs/figures/acf_pacf.png",
) -> None:
    """Plot ACF and PACF for the given series."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    plot_acf(series.dropna(), lags=lags, ax=axes[0], title="Autocorrelation (ACF)")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title="Partial Autocorrelation (PACF)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ACF/PACF plot to {save_path}")


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/figures/correlation.png",
) -> None:
    """Plot heatmap of pairwise feature correlations."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Feature Correlation Matrix")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved correlation matrix to {save_path}")


def plot_feature_distributions(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/figures/distributions.png",
) -> None:
    """Plot histograms for all numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=50, edgecolor="black", linewidth=0.3)
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved distribution plots to {save_path}")


def run_stationarity_tests(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/figures/stationarity_report.csv",
    sample_max_rows: int = 5000,
) -> pd.DataFrame:
    """Run ADF and KPSS stationarity tests on every numeric column.

    Two tests, opposite null hypotheses, used together for robustness:

    * **ADF (Augmented Dickey–Fuller)**: H₀ = unit root (non-stationary).
      Reject (p < 0.05) ⇒ evidence of stationarity.
    * **KPSS**: H₀ = stationary around a level (or trend).
      Reject (p < 0.05) ⇒ evidence of non-stationarity.

    Verdicts:
      - ADF reject + KPSS fail-to-reject → **stationary** ✓
      - ADF fail + KPSS reject           → **non-stationary** (I(1) candidate)
      - ADF reject + KPSS reject         → likely difference-stationary
        with structural break or trend non-stationarity
      - ADF fail   + KPSS fail           → inconclusive (low power)

    Args:
        df: DataFrame with a numeric column per feature.
        save_path: CSV destination for the per-feature report.
        sample_max_rows: Cap rows used in the tests for speed; takes the most
            recent observations (most relevant distribution).

    Returns:
        DataFrame indexed by feature name with columns
        ``adf_stat, adf_pvalue, kpss_stat, kpss_pvalue, verdict, n_obs``.
    """
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric) > sample_max_rows:
        numeric = numeric.iloc[-sample_max_rows:]
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 50 or series.nunique() <= 1:
            rows.append({
                "feature": col, "adf_stat": np.nan, "adf_pvalue": np.nan,
                "kpss_stat": np.nan, "kpss_pvalue": np.nan,
                "verdict": "insufficient_data", "n_obs": int(len(series)),
            })
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(series.values, autolag="AIC")
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
        try:
            # statsmodels emits an InterpolationWarning when p is at the table
            # boundary; we deliberately ignore — the verdict logic only cares
            # about the ≤ 0.05 / ≥ 0.05 partition.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_stat, kpss_p, *_ = kpss(series.values, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

        adf_reject = (not np.isnan(adf_p)) and adf_p < 0.05
        kpss_reject = (not np.isnan(kpss_p)) and kpss_p < 0.05
        if adf_reject and not kpss_reject:
            verdict = "stationary"
        elif (not adf_reject) and kpss_reject:
            verdict = "non_stationary_I1"
        elif adf_reject and kpss_reject:
            verdict = "trend_or_break"
        else:
            verdict = "inconclusive"

        rows.append({
            "feature": col, "adf_stat": float(adf_stat) if not np.isnan(adf_stat) else np.nan,
            "adf_pvalue": float(adf_p) if not np.isnan(adf_p) else np.nan,
            "kpss_stat": float(kpss_stat) if not np.isnan(kpss_stat) else np.nan,
            "kpss_pvalue": float(kpss_p) if not np.isnan(kpss_p) else np.nan,
            "verdict": verdict,
            "n_obs": int(len(series)),
        })
    report = pd.DataFrame(rows).set_index("feature")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(save_path)
    logger.info(
        f"Stationarity report saved to {save_path} "
        f"({(report['verdict'] == 'stationary').sum()} stationary / "
        f"{(report['verdict'] == 'non_stationary_I1').sum()} I(1) / "
        f"{len(report)} total)"
    )
    return report


def run_eda(config: dict) -> None:
    """Run all EDA steps: load features and generate all plots."""
    root = get_project_root(config)
    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load processed features
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = config["features"]["target"]

    # Price columns
    oil_tickers = config["data"]["oil_tickers"]
    price_cols = [f"{t}_close" for t in oil_tickers]

    # Generate all plots
    plot_price_history(df, price_cols, fig_dir / "price_history.png")

    if target in df.columns:
        plot_decomposition(df[target], period=252, save_path=fig_dir / "decomposition.png")
        plot_acf_pacf(df[target], lags=60, save_path=fig_dir / "acf_pacf.png")

    plot_correlation_matrix(df, fig_dir / "correlation.png")
    plot_feature_distributions(df, fig_dir / "distributions.png")

    # Stationarity tests on every numeric feature (item 14)
    run_stationarity_tests(df, save_path=fig_dir / "stationarity_report.csv")

    logger.info("EDA complete. All figures saved to outputs/figures/")
