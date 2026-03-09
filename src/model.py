"""GAM model definition, training, cross-validation, and persistence."""

import logging
from pathlib import Path
from typing import Generator

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pygam import LinearGAM, l, s
from sklearn.model_selection import TimeSeriesSplit

from src.config_loader import get_project_root, resolve_model_config

logger = logging.getLogger(__name__)

# Feature name patterns → pyGAM term type
# Linear (l): features with an expected near-linear relationship with crude price
MACRO_PATTERNS = (
    "usd_index", "cpi", "fed_funds", "t10y2y",
    "t3m_yield", "t5y_yield", "t10y_yield", "t30y_yield",
    "eur_usd", "gbp_usd", "usd_cad", "usd_nok",
    "usd_rub", "usd_cny", "aud_usd", "nzd_usd", "usd_chf",
    # Financial conditions & systemic risk
    "nfci", "stlfsi", "epu_us", "epu_global",
    # Real economy / demand
    "pmi_mfg", "retail_gas", "miles_driven",
    # Fed balance sheet / liquidity
    "fed_balance", "rrp",
    # Credit / risk appetite
    "ig_spread", "ted_spread",
    # Binary seasonality flags (0/1 → linear term is appropriate)
    "driving_season", "heating_season", "us_holiday",
)
# Cyclic spline (s, basis='cp'): calendar features that wrap around
CYCLIC_PATTERNS = ("day_of_week", "month", "day_of_year", "quarter")
# Spline with more knots: rolling/momentum indicators
ROLLING_PATTERNS = (
    "_sma_", "_ema_", "_std_", "_bb_", "macd", "_rsi_", "crack_",
    "_atr_", "_williams_r_", "_stoch_", "_cmf_", "corr_", "_momentum_",
)
# Spline: autoregressive and exogenous lags
LAG_PATTERNS = ("_lag_",)
# Spline: returns and log-returns
RETURN_PATTERNS = ("_pct_change", "_log_return")
# Spline: spread, ratio, and price-level features
SPREAD_PATTERNS = ("_spread", "_ratio", "_dist_sma", "_pct_range", "_52w")

# Volatility / risk patterns → candidate features for the sigma (scale) sub-model
SIGMA_FEATURE_PATTERNS = (
    "ovx", "vix", "_std_", "_bb_width", "hy_spread", "_log_return", "t5yie", "t10yie",
    "_atr_", "_atr_pct", "_stoch_", "_williams_r_", "nfci", "stlfsi",
)

# Risk / regime patterns → candidate features for the nu (skewness) and tau (tail) sub-models
NU_TAU_FEATURE_PATTERNS = ("ovx", "vix", "hy_spread", "t5yie", "t10yie", "nfci", "stlfsi", "epu_us")


def compute_bic(gam: LinearGAM, n_samples: int) -> float:
    """Bayesian Information Criterion for a fitted LinearGAM.

    BIC penalises model complexity more aggressively than AIC for large
    datasets (penalty grows with log(n) instead of a constant 2)::

        BIC = AIC + edof × (log(n) − 2)

    Args:
        gam: Fitted LinearGAM with populated .statistics_.
        n_samples: Number of training observations.

    Returns:
        BIC as a float.
    """
    aic = gam.statistics_["AIC"]
    edof = float(gam.statistics_.get("edof", 0.0))
    return float(aic + edof * (np.log(n_samples) - 2.0))


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


def _build_gam_terms_for_indices(
    feature_indices: list[int],
    all_feature_names: list[str],
    config: dict,
) -> object:
    """Build pyGAM terms for a subset of features, re-indexed sequentially.

    When features are dropped during stepwise selection the remaining
    features must be re-numbered 0, 1, 2, … so that pyGAM's column indices
    align with the subsetted ``X[:, feature_indices]`` matrix.

    Args:
        feature_indices: Ordered column positions in the *full* feature matrix.
        all_feature_names: Names for every column of the full feature matrix.
        config: Project configuration (for n_splines).

    Returns:
        pyGAM terms expression.
    """
    n_splines_default = config.get("model", {}).get("n_splines", 25)
    terms = None

    for new_idx, orig_idx in enumerate(feature_indices):
        name = all_feature_names[orig_idx]
        category = _classify_feature(name)

        if category == "linear":
            term = l(new_idx)
        elif category == "cyclic":
            n_sp = 7 if "day_of_week" in name else 12 if "month" in name else 30
            term = s(new_idx, n_splines=n_sp, basis="cp")
        elif category == "rolling":
            term = s(new_idx, n_splines=n_splines_default)
        elif category == "lag":
            term = s(new_idx, n_splines=20)
        elif category == "return":
            term = s(new_idx, n_splines=15)
        else:
            term = s(new_idx, n_splines=20)

        terms = term if terms is None else terms + term
        logger.debug(f"  new={new_idx} orig={orig_idx}: {name} -> {category}")

    return terms


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
    terms = _build_gam_terms_for_indices(
        list(range(len(feature_names))), feature_names, config
    )
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


# ─────────────────────────────────────────────
# Module-level workers (must be picklable by loky)
# ─────────────────────────────────────────────

def _fit_one_lam(
    lam_val: float,
    terms: object,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, "LinearGAM"]:
    """Fit a LinearGAM with a single fixed lambda; return (GCV, gam).

    pyGAM broadcasts a scalar lam to all terms automatically.
    """
    gam = LinearGAM(terms, lam=lam_val)
    gam.fit(X, y)
    return float(gam.statistics_["GCV"]), gam


def _fit_fold_worker(
    fold_num: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    terms: object,
    lam_values: list[float],
) -> dict:
    """Fit one CV fold and return metrics + deferred log lines.

    Calls fit_gam with n_jobs=1 to avoid nested parallelism.
    Log lines are returned as strings so the main process can emit them
    in fold order after all workers complete.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    gam = fit_gam(X_train, y_train, terms, lam_values, n_jobs=1)
    y_pred = gam.predict(X_test)

    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    log_lines = [
        f"  Fold {fold_num}: train={len(train_idx)} obs, test={len(test_idx)} obs (after embargo)",
        f"  Fold {fold_num}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%",
    ]
    return {
        "fold": fold_num,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "log_lines": log_lines,
    }


def fit_gam(
    X: np.ndarray,
    y: np.ndarray,
    terms: object,
    lam_values: list[float] | None = None,
    n_jobs: int = 1,
) -> LinearGAM:
    """Fit a LinearGAM with lambda tuning via parallel grid search.

    Args:
        X: Feature matrix.
        y: Target vector.
        terms: pyGAM terms expression from define_gam_terms.
        lam_values: List of lambda values for grid search. If None, uses logspace.
        n_jobs: Number of parallel jobs for the lambda grid search.
            1 = sequential (default, safe for nested calls).
            -1 = use all logical CPUs.

    Returns:
        Fitted LinearGAM model.
    """
    if lam_values is None:
        lam_values = np.logspace(-3, 3, 11).tolist()

    results: list[tuple[float, LinearGAM]] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_lam)(lam, terms, X, y) for lam in lam_values
    )

    best_gcv, gam = min(results, key=lambda r: r[0])

    logger.info(f"GAM fitted. GCV score: {best_gcv:.4f}")
    logger.info(f"Pseudo R-squared: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

    return gam


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    terms: object,
    n_splits: int = 5,
    embargo_days: int = 200,
    lam_values: list[float] | None = None,
    n_jobs: int = 1,
) -> dict:
    """Perform embargoed time series cross-validation.

    Uses :class:`EmbargoedTimeSeriesSplit` to ensure that:
    1. We always predict future from past (no temporal leakage).
    2. Rolling/lag features computed on the full dataset do not bleed
       training-period information into test observations at fold boundaries.

    Folds are fitted in parallel when ``n_jobs != 1``.  Each fold worker
    calls ``fit_gam`` with ``n_jobs=1`` to avoid nested parallelism.

    Args:
        X: Feature matrix.
        y: Target vector.
        terms: pyGAM terms expression.
        n_splits: Number of CV folds.
        embargo_days: Observations to strip from the start of each test fold.
            Should equal the largest rolling window used in feature engineering
            (default 200 = the 200-day SMA window).
        lam_values: Lambda values for grid search.
        n_jobs: Number of parallel jobs for fold fitting.
            1 = sequential (default).  -1 = all logical CPUs.

    Returns:
        Dict with fold_metrics, mean_mae, mean_rmse, mean_mape, embargo_days.
    """
    if lam_values is None:
        lam_values = np.logspace(-3, 3, 11).tolist()

    splitter = EmbargoedTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)

    logger.info(
        f"Starting {n_splits}-fold embargoed CV "
        f"(embargo_days={embargo_days}, ~{embargo_days} obs stripped per fold, "
        f"n_jobs={n_jobs})"
    )

    # Pre-collect folds so we can dispatch them all at once
    folds = [
        (fold_num, train_idx, test_idx)
        for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X), start=1)
    ]

    if not folds:
        raise RuntimeError(
            "All CV folds were skipped — no test observations survived the embargo. "
            "Reduce embargo_days in config or increase the dataset size."
        )

    raw_results: list[dict] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_fold_worker)(fold_num, train_idx, test_idx, X, y, terms, lam_values)
        for fold_num, train_idx, test_idx in folds
    )

    # Sort by fold number (parallel dispatch order is not guaranteed)
    raw_results.sort(key=lambda r: r["fold"])

    fold_metrics = []
    for result in raw_results:
        for line in result.pop("log_lines"):
            logger.info(line)
        fold_metrics.append(result)

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


def stepwise_aic_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: dict,
    lam_values: list[float] | None = None,
    max_steps: int | None = None,
    n_jobs: int = 1,
    criterion: str = "bic",
    target_max_terms: int | None = None,
) -> tuple[np.ndarray, list[int], list[str], list[str], LinearGAM]:
    """Iterative backward BIC/AIC elimination for GAM feature selection.

    At each step the least-significant term (highest p-value) is the candidate
    for removal.  A trial model is fitted *without* that term using a fixed λ
    (inherited from the previous step) for speed.  The drop is accepted when
    trial score ≤ current score (where score is BIC by default, or AIC).
    Once no single removal improves the score the loop stops.

    A secondary **hard-cap** loop then unconditionally drops the highest
    p-value terms one at a time until ``target_max_terms`` is reached, even
    if removing them would worsen the information criterion.  This keeps the
    final model small enough to avoid overfitting when the dataset is large.

    Both loops use fast fixed-λ trial fits; the final model is refit with a
    full grid-search for a clean λ estimate.

    Args:
        X: Full feature matrix (n_samples × n_features).
        y: Target vector.
        feature_names: Column names aligned with X.
        config: Project configuration.
        lam_values: Lambda values for the initial and final grid-search.
        max_steps: Maximum elimination iterations (default: len(feature_names)).
        criterion: Information criterion for the elimination loop.
            ``"bic"`` (default) penalises complexity more aggressively;
            ``"aic"`` is less conservative.
        target_max_terms: Hard upper bound on the number of retained terms.
            After the criterion-based loop, terms are dropped by descending
            p-value until this limit is reached (ignored if None).

    Returns:
        Tuple of (X_selected, selected_indices, selected_names, dropped_names, gam).
        ``X_selected`` is ``X[:, selected_indices]`` and ``gam`` is the final
        model fitted on the selected features with a full grid-search.
    """
    if max_steps is None:
        max_steps = len(feature_names)

    current_indices = list(range(len(feature_names)))
    dropped_names: list[str] = []

    # ── Initial full model (grid-search) ─────────────────────────────────────
    terms = _build_gam_terms_for_indices(current_indices, feature_names, config)
    gam = fit_gam(X[:, current_indices], y, terms, lam_values, n_jobs=n_jobs)
    n_samples = len(y)
    current_score = compute_bic(gam, n_samples) if criterion == "bic" else gam.statistics_["AIC"]
    current_lam = gam.lam  # reuse for fast trial fits

    logger.info(
        f"Stepwise {criterion.upper()} selection — start: {len(current_indices)} features, "
        f"{criterion.upper()}={current_score:.4f}"
    )

    for step in range(max_steps):
        if len(current_indices) <= 1:
            break

        p_values = list(gam.statistics_.get("p_values", []))
        if not p_values:
            logger.warning("p_values unavailable — stopping stepwise selection.")
            break

        # Align p-values with current feature count (exclude intercept if present)
        term_pvals = np.array(p_values[: len(current_indices)])
        worst_local = int(np.argmax(term_pvals))
        worst_pval = float(term_pvals[worst_local])
        candidate_orig = current_indices[worst_local]
        candidate_name = feature_names[candidate_orig]

        # ── Trial: fit without the worst term (fast, fixed λ) ─────────────────
        trial_indices = [idx for i, idx in enumerate(current_indices) if i != worst_local]
        trial_terms = _build_gam_terms_for_indices(trial_indices, feature_names, config)

        # Build a fixed-λ model (no grid-search) for speed
        trial_lam = [lv for i, lv in enumerate(current_lam) if i != worst_local]
        trial_gam_fast = LinearGAM(trial_terms)
        trial_gam_fast.lam = trial_lam
        trial_gam_fast.fit(X[:, trial_indices], y)
        trial_score = (
            compute_bic(trial_gam_fast, n_samples) if criterion == "bic"
            else trial_gam_fast.statistics_["AIC"]
        )

        if trial_score <= current_score:
            logger.info(
                f"  Step {step + 1}: drop '{candidate_name}' "
                f"(p={worst_pval:.4f}, {criterion.upper()} {current_score:.4f} → {trial_score:.4f} ✓)"
            )
            dropped_names.append(candidate_name)
            current_indices = trial_indices
            gam = trial_gam_fast
            current_score = trial_score
            current_lam = gam.lam
        else:
            logger.info(
                f"  Step {step + 1}: keeping '{candidate_name}' "
                f"(p={worst_pval:.4f}, {criterion.upper()} would worsen "
                f"{current_score:.4f} → {trial_score:.4f} ✗). Stopping."
            )
            break

    # ── Hard cap: unconditional elimination to reach target_max_terms ─────────
    if target_max_terms is not None and len(current_indices) > target_max_terms:
        logger.info(
            f"Applying hard cap: reducing from {len(current_indices)} → "
            f"{target_max_terms} terms by p-value order…"
        )
        while len(current_indices) > target_max_terms and len(current_indices) > 1:
            p_values = list(gam.statistics_.get("p_values", []))
            if not p_values:
                break
            term_pvals = np.array(p_values[: len(current_indices)])
            worst_local = int(np.argmax(term_pvals))
            candidate_name = feature_names[current_indices[worst_local]]
            trial_indices = [idx for i, idx in enumerate(current_indices) if i != worst_local]
            trial_terms = _build_gam_terms_for_indices(trial_indices, feature_names, config)
            trial_lam = [lv for i, lv in enumerate(current_lam) if i != worst_local]
            trial_gam = LinearGAM(trial_terms)
            trial_gam.lam = trial_lam
            trial_gam.fit(X[:, trial_indices], y)
            logger.info(
                f"  Hard-cap drop: '{candidate_name}' "
                f"({len(current_indices)} → {len(trial_indices)} terms)"
            )
            dropped_names.append(candidate_name)
            current_indices = trial_indices
            gam = trial_gam
            current_lam = gam.lam

    selected_names = [feature_names[i] for i in current_indices]
    logger.info(
        f"Stepwise complete: kept {len(selected_names)}/{len(feature_names)} features, "
        f"dropped {len(dropped_names)}: {dropped_names}"
    )

    # ── Final model: full grid-search on selected features ────────────────────
    if dropped_names:
        logger.info("Refitting final model on selected features (full grid-search)…")
        final_terms = _build_gam_terms_for_indices(current_indices, feature_names, config)
        gam = fit_gam(X[:, current_indices], y, final_terms, lam_values, n_jobs=n_jobs)
        logger.info(f"Final {criterion.upper()} after grid-search: {gam.statistics_['AIC']:.4f}")

    return X[:, current_indices], current_indices, selected_names, dropped_names, gam


def compute_rolling_nu_tau(
    std_residuals: np.ndarray,
    window: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling skewness (nu) and tail-weight proxy (tau) from standardised residuals.

    Rolling skewness estimates the time-varying asymmetry of forecast errors.
    Rolling excess kurtosis is mapped to a tau proxy::

        tau = 1 / (1 + max(0, excess_kurtosis))

    This ensures large kurtosis (heavy tails) maps to small tau, consistent
    with Johnson SU where smaller tau → heavier tails.

    Args:
        std_residuals: Standardised residuals ``(y - mu) / sigma``.
        window: Rolling window in observations (default: 60 trading days).

    Returns:
        Tuple of (valid_mask, rolling_nu, rolling_tau) where *valid_mask* is
        a boolean array of length ``len(std_residuals)`` marking non-NaN rows,
        and *rolling_nu* / *rolling_tau* are the corresponding valid values.
    """
    s = pd.Series(std_residuals)
    rolling_nu = s.rolling(window).skew().values
    rolling_kurt = s.rolling(window).kurt().values       # excess kurtosis
    rolling_tau = 1.0 / (1.0 + np.clip(rolling_kurt, 0.0, None))
    valid_mask = ~np.isnan(rolling_nu) & ~np.isnan(rolling_tau)
    return valid_mask, rolling_nu[valid_mask], rolling_tau[valid_mask]


def select_sigma_features(
    feature_names: list[str],
    X: np.ndarray,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Select volatility-relevant features from the mu feature set for the sigma sub-model.

    Matches feature names against :data:`SIGMA_FEATURE_PATTERNS`.  If no
    matches are found all features are returned as fallback.

    Args:
        feature_names: Names of the mu model's selected features.
        X: Feature matrix aligned with *feature_names*.

    Returns:
        Tuple of (X_sigma, sigma_names, col_indices) where *col_indices*
        are positions into *X* / *feature_names*.
    """
    indices = [
        i for i, n in enumerate(feature_names)
        if any(p in n.lower() for p in SIGMA_FEATURE_PATTERNS)
    ]
    if not indices:
        logger.warning("No sigma feature patterns matched — using all mu features as fallback.")
        return X, feature_names, list(range(len(feature_names)))
    sigma_names = [feature_names[i] for i in indices]
    logger.info(f"Sigma feature candidates ({len(sigma_names)}): {sigma_names}")
    return X[:, indices], sigma_names, indices


def select_nu_tau_features(
    feature_names: list[str],
    X: np.ndarray,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Select risk/regime features from the mu feature set for nu/tau sub-models.

    Matches feature names against :data:`NU_TAU_FEATURE_PATTERNS`.  Falls
    back to sigma feature patterns, then all features if no matches found.

    Args:
        feature_names: Names of the mu model's selected features.
        X: Feature matrix aligned with *feature_names*.

    Returns:
        Tuple of (X_nu_tau, nu_tau_names, col_indices).
    """
    indices = [
        i for i, n in enumerate(feature_names)
        if any(p in n.lower() for p in NU_TAU_FEATURE_PATTERNS)
    ]
    if not indices:
        # Fallback: use sigma patterns (also volatility-related)
        indices = [
            i for i, n in enumerate(feature_names)
            if any(p in n.lower() for p in SIGMA_FEATURE_PATTERNS)
        ]
    if not indices:
        logger.warning("No nu/tau feature patterns matched — using all mu features as fallback.")
        return X, feature_names, list(range(len(feature_names)))
    nu_tau_names = [feature_names[i] for i in indices]
    logger.info(f"Nu/tau feature candidates ({len(nu_tau_names)}): {nu_tau_names}")
    return X[:, indices], nu_tau_names, indices


def fit_sigma_gam(
    X_sigma: np.ndarray,
    mu_residuals: np.ndarray,
    sigma_feature_names: list[str],
    config: dict,
    lam_values: list[float] | None = None,
    n_jobs: int = 1,
) -> tuple[LinearGAM, list[str], list[int]]:
    """Fit a GAM on log|residuals| to model conditional heteroscedasticity (σ).

    The response is ``log(|mu_residuals| + 1e-6)``.  Fitting on the
    log-absolute-residual is a standard approach for modelling the scale
    equation in location-scale models:  at prediction time the conditional
    sigma is recovered as ``exp(sigma_gam.predict(X_sigma))``.

    Stepwise BIC selection is applied (controlled by ``model.sigma_stepwise``
    and ``model.sigma_max_terms`` in config) to keep the sigma model compact.

    Args:
        X_sigma: Feature matrix for sigma candidates.
        mu_residuals: In-sample residuals from the mu model ``(y - mu_pred)``.
        sigma_feature_names: Column names aligned with *X_sigma*.
        config: Project configuration.
        lam_values: Lambda grid for pyGAM fitting.
        n_jobs: Parallel workers.

    Returns:
        Tuple of (sigma_gam, selected_names, selected_local_indices) where
        *selected_local_indices* are positions into *X_sigma*.
    """
    model_cfg = resolve_model_config(config)
    sigma_max = model_cfg.get("sigma_max_terms")
    do_stepwise = model_cfg.get("sigma_stepwise", True)
    log_abs_resid = np.log(np.abs(mu_residuals) + 1e-6)

    if do_stepwise and len(sigma_feature_names) > 1:
        _, selected_indices, selected_names, dropped, gam = stepwise_aic_selection(
            X_sigma, log_abs_resid, sigma_feature_names, config,
            lam_values=lam_values,
            max_steps=model_cfg.get("stepwise_max_steps"),
            n_jobs=n_jobs,
            criterion=model_cfg.get("stepwise_criterion", "bic"),
            target_max_terms=sigma_max,
        )
        logger.info(f"Sigma model: {len(selected_names)} terms retained.")
        return gam, selected_names, selected_indices
    else:
        terms = _build_gam_terms_for_indices(
            list(range(len(sigma_feature_names))), sigma_feature_names, config
        )
        gam = fit_gam(X_sigma, log_abs_resid, terms, lam_values, n_jobs=n_jobs)
        return gam, sigma_feature_names, list(range(len(sigma_feature_names)))


def get_sigma_from_sigma_gam(sigma_gam: LinearGAM, X_sigma: np.ndarray) -> np.ndarray:
    """Predict conditional scale σ from the sigma sub-model.

    The sigma model regresses ``log|residuals|``, so the predicted scale is::

        σ_pred = exp(sigma_gam.predict(X_sigma))

    Results are clipped to ``[1e-6, ∞)`` to prevent zero or negative scales.

    Args:
        sigma_gam: Fitted sigma LinearGAM (response was log|residuals|).
        X_sigma: Feature matrix matching the sigma model's training columns.

    Returns:
        Array of shape (n_samples,) with per-observation σ.
    """
    return np.clip(np.exp(sigma_gam.predict(X_sigma)), 1e-6, None)


def fit_distributional_gam(
    X_nu_tau: np.ndarray,
    response: np.ndarray,
    feature_names: list[str],
    config: dict,
    lam_values: list[float] | None = None,
    n_jobs: int = 1,
    max_terms: int | None = None,
    param_name: str = "nu",
) -> tuple[LinearGAM, list[str], list[int]]:
    """Fit a GAM to model one distributional parameter (nu or tau) as a function of features.

    Rolling skewness (nu) and tail-weight proxy (tau) are regressed on
    volatility/risk features.  Stepwise BIC reduces each sub-model to at most
    ``max_terms`` features so the parameters remain interpretable.

    Args:
        X_nu_tau: Feature matrix of candidate predictors.
        response: Rolling nu (skewness) or rolling tau (tail weight) array,
            aligned with the valid rows of X_nu_tau after the rolling window.
        feature_names: Column names aligned with *X_nu_tau*.
        config: Project configuration.
        lam_values: Lambda grid for pyGAM fitting.
        n_jobs: Parallel workers.
        max_terms: Hard cap on retained terms (overrides config value).
        param_name: ``\"nu\"`` or ``\"tau\"`` — used only for logging.

    Returns:
        Tuple of (gam, selected_names, selected_local_indices) where
        *selected_local_indices* are positions into *X_nu_tau*.
    """
    model_cfg = resolve_model_config(config)
    do_stepwise = model_cfg.get("sigma_stepwise", True)  # reuse sigma_stepwise flag

    if do_stepwise and len(feature_names) > 1:
        _, selected_indices, selected_names, dropped, gam = stepwise_aic_selection(
            X_nu_tau, response, feature_names, config,
            lam_values=lam_values,
            max_steps=model_cfg.get("stepwise_max_steps"),
            n_jobs=n_jobs,
            criterion=model_cfg.get("stepwise_criterion", "bic"),
            target_max_terms=max_terms,
        )
        logger.info(f"{param_name} model: {len(selected_names)} terms retained.")
        return gam, selected_names, selected_indices
    else:
        terms = _build_gam_terms_for_indices(
            list(range(len(feature_names))), feature_names, config
        )
        gam = fit_gam(X_nu_tau, response, terms, lam_values, n_jobs=n_jobs)
        return gam, feature_names, list(range(len(feature_names)))


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


def save_all_models(
    models: dict,
    path: str | Path = "outputs/models/gam_models.pkl",
) -> None:
    """Save all distributional sub-models (mu, sigma, nu, tau) as a dict.

    Args:
        models: Dict with keys ``\"mu\"``, ``\"sigma\"``, ``\"nu\"``, ``\"tau\"``
            mapped to fitted LinearGAM instances.
        path: Output path for the serialised dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    logger.info(f"All models saved to {path}")


def load_all_models(
    path: str | Path = "outputs/models/gam_models.pkl",
) -> dict:
    """Load all distributional sub-models from the joint pickle.

    Returns:
        Dict with keys ``\"mu\"``, ``\"sigma\"``, ``\"nu\"``, ``\"tau\"``
        mapped to fitted LinearGAM instances.
    """
    return joblib.load(path)


def train_and_save(config: dict) -> tuple:
    """Full training pipeline: load data, (optionally) select features, CV, fit, save.

    Pipeline order:
    1. Build full feature matrix.
    2. If ``model.stepwise_selection`` is true, run backward AIC elimination
       on the full dataset to identify a parsimonious feature set.
    3. Run embargoed time-series cross-validation on the selected features.
    4. Fit the final model on all data.
    5. Persist model + feature names.

    Args:
        config: Project configuration dictionary.

    Returns:
        Tuple of (fitted_gam, cv_results).
    """
    root = get_project_root(config)
    model_dir = root / "outputs" / "models"
    model_cfg = resolve_model_config(config)
    lam_values = model_cfg.get("lam_search")
    n_jobs = model_cfg.get("n_jobs", 1)

    # Load features
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = config["features"]["target"]

    # Build feature matrix
    X, y, feature_names = build_feature_matrix(df, target)

    # ── Optional stepwise BIC feature selection ───────────────────────────────
    if model_cfg.get("stepwise_selection", False):
        logger.info("Running stepwise feature selection…")
        X, _, feature_names, dropped, _ = stepwise_aic_selection(
            X, y, feature_names, config,
            lam_values=lam_values,
            max_steps=model_cfg.get("stepwise_max_steps"),
            n_jobs=n_jobs,
            criterion=model_cfg.get("stepwise_criterion", "bic"),
            target_max_terms=model_cfg.get("target_max_terms"),
        )
        logger.info(f"Selected {len(feature_names)} features after stepwise selection.")
        if dropped:
            logger.info(f"Dropped features: {dropped}")

    # Build terms over the (possibly reduced) feature set.
    # feature_names is already the selected subset after stepwise_aic_selection,
    # so sequential 0-based indexing via define_gam_terms is correct.
    terms = define_gam_terms(feature_names, config)

    # Cross-validate on selected features
    cv_results = cross_validate(
        X, y, terms,
        n_splits=model_cfg.get("cv_splits", 5),
        embargo_days=model_cfg.get("embargo_days", 200),
        lam_values=lam_values,
        n_jobs=n_jobs,
    )

    # ── Stage 1: Fit final mu (location) model on full dataset ────────────────
    logger.info("Fitting final mu model on full dataset...")
    mu_gam = fit_gam(X, y, terms, lam_values, n_jobs=n_jobs)

    # ── Stage 2: Sigma model (conditional heteroscedasticity) ─────────────────
    logger.info("Fitting sigma (conditional scale) sub-model…")
    mu_pred = mu_gam.predict(X)
    mu_residuals = y - mu_pred
    X_sigma_all, sigma_all_names, _ = select_sigma_features(feature_names, X)
    sigma_gam, sigma_feature_names, sigma_sel_idx = fit_sigma_gam(
        X_sigma_all, mu_residuals, sigma_all_names, config,
        lam_values=lam_values, n_jobs=n_jobs,
    )
    X_sigma = X_sigma_all[:, sigma_sel_idx]
    sigma_pred = get_sigma_from_sigma_gam(sigma_gam, X_sigma)

    # ── Stage 3: Nu and Tau models (rolling moments of standardised residuals) ─
    nu_tau_window = model_cfg.get("nu_tau_window", 60)
    std_residuals = mu_residuals / np.clip(sigma_pred, 1e-8, None)
    valid_mask, rolling_nu, rolling_tau = compute_rolling_nu_tau(
        std_residuals, window=nu_tau_window
    )

    X_nu_tau_all, nu_tau_all_names, _ = select_nu_tau_features(feature_names, X)
    X_nu_tau_valid = X_nu_tau_all[valid_mask]

    logger.info("Fitting nu (skewness) sub-model…")
    nu_gam, nu_feature_names, nu_sel_idx = fit_distributional_gam(
        X_nu_tau_valid, rolling_nu, nu_tau_all_names, config,
        lam_values=lam_values, n_jobs=n_jobs,
        max_terms=model_cfg.get("nu_max_terms", 8), param_name="nu",
    )

    logger.info("Fitting tau (tail-weight) sub-model…")
    tau_gam, tau_feature_names, tau_sel_idx = fit_distributional_gam(
        X_nu_tau_valid, rolling_tau, nu_tau_all_names, config,
        lam_values=lam_values, n_jobs=n_jobs,
        max_terms=model_cfg.get("tau_max_terms", 8), param_name="tau",
    )

    # ── Save all models and feature name lists ─────────────────────────────────
    save_model(mu_gam, model_dir / "gam_model.pkl")        # backward compat
    save_feature_names(feature_names, model_dir / "feature_names.pkl")

    all_models = {"mu": mu_gam, "sigma": sigma_gam, "nu": nu_gam, "tau": tau_gam}
    save_all_models(all_models, model_dir / "gam_models.pkl")

    save_feature_names(sigma_feature_names, model_dir / "sigma_feature_names.pkl")
    save_feature_names(nu_feature_names, model_dir / "nu_feature_names.pkl")
    save_feature_names(tau_feature_names, model_dir / "tau_feature_names.pkl")

    logger.info(
        f"All models saved. "
        f"μ: {len(feature_names)} terms, "
        f"σ: {len(sigma_feature_names)} terms, "
        f"ν: {len(nu_feature_names)} terms, "
        f"τ: {len(tau_feature_names)} terms"
    )

    return mu_gam, sigma_gam, nu_gam, tau_gam, cv_results
