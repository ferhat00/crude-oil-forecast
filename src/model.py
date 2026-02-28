"""GAM model definition, training, cross-validation, and persistence."""

import logging
from pathlib import Path
from typing import Generator

import joblib
import numpy as np
import pandas as pd
from pygam import LinearGAM, l, s
from sklearn.model_selection import TimeSeriesSplit

from src.config_loader import get_project_root

logger = logging.getLogger(__name__)

# Feature name patterns → pyGAM term type
# Linear (l): features with an expected near-linear relationship with crude price
MACRO_PATTERNS = (
    "usd_index", "cpi", "fed_funds", "t10y2y",
    "t3m_yield", "t5y_yield", "t10y_yield", "t30y_yield",
    "eur_usd", "gbp_usd", "usd_cad", "usd_nok",
    "usd_rub", "usd_cny", "aud_usd", "nzd_usd", "usd_chf",
)
# Cyclic spline (s, basis='cp'): calendar features that wrap around
CYCLIC_PATTERNS = ("day_of_week", "month", "day_of_year", "quarter")
# Spline with more knots: rolling/momentum indicators
ROLLING_PATTERNS = ("_sma_", "_ema_", "_std_", "_bb_", "macd", "_rsi_", "crack_")
# Spline: autoregressive and exogenous lags
LAG_PATTERNS = ("_lag_",)
# Spline: returns and log-returns
RETURN_PATTERNS = ("_pct_change", "_log_return")
# Spline: spread, ratio, and price-level features
SPREAD_PATTERNS = ("_spread", "_ratio", "_dist_sma", "_pct_range", "_52w")


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Separate target from features.

    Excludes raw price/volume columns that would leak future information,
    keeping only engineered features and exogenous variables.

    Args:
        df: Feature-engineered DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y, feature_names).
    """
    # Columns to exclude: target itself and raw price/volume OHLCV columns
    # (we keep their engineered derivatives like lags, rolling, returns)
    exclude_patterns = ("_open", "_high", "_low", "_volume")
    exclude_cols = {target_col}
    for col in df.columns:
        if col == target_col:
            continue
        # Exclude raw close columns for other tickers (keep only target's derivatives)
        if col.endswith("_close") and col != target_col:
            exclude_cols.add(col)
        # Exclude raw OHLCV columns
        if any(col.endswith(p) for p in exclude_patterns):
            exclude_cols.add(col)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_names = feature_cols

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    logger.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Features: {feature_names}")

    return X, y, feature_names


def _classify_feature(name: str) -> str:
    """Classify a feature name into a category for GAM term selection."""
    name_lower = name.lower()

    if any(p in name_lower for p in CYCLIC_PATTERNS):
        return "cyclic"
    if any(p in name_lower for p in MACRO_PATTERNS):
        return "linear"
    if any(p in name_lower for p in ROLLING_PATTERNS):
        return "rolling"
    if any(p in name_lower for p in LAG_PATTERNS):
        return "lag"
    if any(p in name_lower for p in RETURN_PATTERNS):
        return "return"
    if any(p in name_lower for p in SPREAD_PATTERNS):
        return "spline"
    return "spline"  # default


def define_gam_terms(
    feature_names: list[str],
    config: dict,
) -> object:
    """Build pyGAM terms formula based on feature names.

    Term assignment strategy:
    - Macro indicators (usd_index, cpi, fed_funds, t10y2y): linear term l()
    - Rolling statistics (sma, std): spline s(n_splines=25)
    - Lag features: spline s(n_splines=20)
    - Calendar features (day_of_week, month, day_of_year): cyclic spline s(basis='cp')
    - Returns/volatility: spline s(n_splines=15)
    - Default: spline s(n_splines=20)

    Args:
        feature_names: List of feature column names.
        config: Project config (for n_splines setting).

    Returns:
        pyGAM terms expression.
    """
    n_splines_default = config.get("model", {}).get("n_splines", 25)
    terms = None

    for i, name in enumerate(feature_names):
        category = _classify_feature(name)

        if category == "linear":
            term = l(i)
        elif category == "cyclic":
            # Cyclic splines for calendar features
            n_sp = 7 if "day_of_week" in name else 12 if "month" in name else 30
            term = s(i, n_splines=n_sp, basis="cp")
        elif category == "rolling":
            term = s(i, n_splines=n_splines_default)
        elif category == "lag":
            term = s(i, n_splines=20)
        elif category == "return":
            term = s(i, n_splines=15)
        else:
            term = s(i, n_splines=20)

        terms = term if terms is None else terms + term
        logger.debug(f"  Feature {i}: {name} -> {category}")

    logger.info(f"Defined {len(feature_names)} GAM terms")
    return terms


class EmbargoedTimeSeriesSplit:
    """Time series cross-validator with embargo gap (Lopez de Prado, 2018).

    Rolling and lagged features computed on the full dataset cause data leakage
    at each train/test boundary: e.g., a 200-day SMA for the first test
    observation contains 199 days from the training period.

    Solution: drop `embargo_days` observations from the **start** of each
    test fold, ensuring the test set only begins after rolling-window memory
    has fully expired.  `embargo_days` should equal the largest rolling window
    used during feature engineering (default 200).

    Args:
        n_splits: Number of CV folds (passed to TimeSeriesSplit).
        embargo_days: Observations to skip at the start of each test fold.
            Set this to the maximum rolling window used in feature engineering.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 200) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(
        self, X: np.ndarray, y=None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            # The embargo cutoff: first index that is truly independent of training
            # Train ends at train_idx[-1]; embargo extends embargo_days beyond that.
            embargo_cutoff = train_idx[-1] + self.embargo_days
            test_idx_clean = test_idx[test_idx > embargo_cutoff]

            n_dropped = len(test_idx) - len(test_idx_clean)
            if n_dropped:
                logger.debug(
                    f"  Fold {fold_num}: dropped {n_dropped} embargoed test obs "
                    f"(embargo_days={self.embargo_days})"
                )

            if len(test_idx_clean) == 0:
                logger.warning(
                    f"  Fold {fold_num}: 0 test observations remain after "
                    f"{self.embargo_days}-day embargo — skipping fold. "
                    "Consider reducing embargo_days or increasing cv_splits."
                )
                continue

            yield train_idx, test_idx_clean

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def fit_gam(
    X: np.ndarray,
    y: np.ndarray,
    terms: object,
    lam_values: list[float] | None = None,
) -> LinearGAM:
    """Fit a LinearGAM with automatic lambda tuning via grid search.

    Args:
        X: Feature matrix.
        y: Target vector.
        terms: pyGAM terms expression from define_gam_terms.
        lam_values: List of lambda values for grid search. If None, uses logspace.

    Returns:
        Fitted LinearGAM model.
    """
    if lam_values is None:
        lam_values = np.logspace(-3, 3, 11).tolist()

    gam = LinearGAM(terms)
    gam.gridsearch(X, y, lam=np.logspace(-3, 3, 11))

    logger.info(f"GAM fitted. GCV score: {gam.statistics_['GCV']:.4f}")
    logger.info(f"Pseudo R-squared: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

    return gam


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    terms: object,
    n_splits: int = 5,
    embargo_days: int = 200,
    lam_values: list[float] | None = None,
) -> dict:
    """Perform embargoed time series cross-validation.

    Uses :class:`EmbargoedTimeSeriesSplit` to ensure that:
    1. We always predict future from past (no temporal leakage).
    2. Rolling/lag features computed on the full dataset do not bleed
       training-period information into test observations at fold boundaries.

    Args:
        X: Feature matrix.
        y: Target vector.
        terms: pyGAM terms expression.
        n_splits: Number of CV folds.
        embargo_days: Observations to strip from the start of each test fold.
            Should equal the largest rolling window used in feature engineering
            (default 200 = the 200-day SMA window).
        lam_values: Lambda values for grid search.

    Returns:
        Dict with fold_metrics, mean_mae, mean_rmse, mean_mape, embargo_days.
    """
    splitter = EmbargoedTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)
    fold_metrics = []

    logger.info(
        f"Starting {n_splits}-fold embargoed CV "
        f"(embargo_days={embargo_days}, ~{embargo_days} obs stripped per fold)"
    )

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(
            f"  Fold {fold}: train={len(train_idx)} obs, "
            f"test={len(test_idx)} obs (after embargo)"
        )

        gam = fit_gam(X_train, y_train, terms, lam_values)
        y_pred = gam.predict(X_test)

        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        fold_metrics.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        })
        logger.info(
            f"  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%"
        )

    if not fold_metrics:
        raise RuntimeError(
            "All CV folds were skipped — no test observations survived the embargo. "
            "Reduce embargo_days in config or increase the dataset size."
        )

    results = {
        "fold_metrics": fold_metrics,
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mean_mape": float(np.mean([m["mape"] for m in fold_metrics])),
        "embargo_days": embargo_days,
    }

    logger.info(
        f"CV Results (embargoed) — MAE: {results['mean_mae']:.2f}, "
        f"RMSE: {results['mean_rmse']:.2f}, "
        f"MAPE: {results['mean_mape']:.2f}%"
    )

    return results


def save_model(gam: LinearGAM, path: str | Path = "outputs/models/gam_model.pkl") -> None:
    """Serialize fitted GAM model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gam, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str | Path = "outputs/models/gam_model.pkl") -> LinearGAM:
    """Load a serialized GAM model."""
    return joblib.load(path)


def save_feature_names(
    feature_names: list[str],
    path: str | Path = "outputs/models/feature_names.pkl",
) -> None:
    """Save feature names list alongside the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_names, path)


def load_feature_names(
    path: str | Path = "outputs/models/feature_names.pkl",
) -> list[str]:
    """Load saved feature names."""
    return joblib.load(path)


def train_and_save(config: dict) -> tuple[LinearGAM, dict]:
    """Full training pipeline: load data, cross-validate, fit, save.

    Args:
        config: Project configuration dictionary.

    Returns:
        Tuple of (fitted_gam, cv_results).
    """
    root = get_project_root(config)
    model_dir = root / "outputs" / "models"

    # Load features
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = config["features"]["target"]

    # Build feature matrix
    X, y, feature_names = build_feature_matrix(df, target)

    # Define GAM terms
    terms = define_gam_terms(feature_names, config)

    # Cross-validate
    model_cfg = config.get("model", {})
    cv_results = cross_validate(
        X, y, terms,
        n_splits=model_cfg.get("cv_splits", 5),
        embargo_days=model_cfg.get("embargo_days", 200),
        lam_values=model_cfg.get("lam_search"),
    )

    # Fit on full dataset
    logger.info("Fitting final model on full dataset...")
    gam = fit_gam(X, y, terms, model_cfg.get("lam_search"))

    # Save model and feature names
    save_model(gam, model_dir / "gam_model.pkl")
    save_feature_names(feature_names, model_dir / "feature_names.pkl")

    return gam, cv_results
