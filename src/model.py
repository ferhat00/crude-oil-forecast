"""GAM model definition, training, cross-validation, and persistence."""

import logging
from pathlib import Path
from typing import Generator

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pygam import ExpectileGAM, LinearGAM, l, s, te
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer

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

# ── Monotonicity: domain-driven shape constraints applied to spline terms when
# config.enable_monotonic is true.  Keyed by feature-name substring; first match
# wins.  Use sparingly — wrong constraints hurt fit more than they help.
MONOTONIC_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    # Inventory build is bearish for oil
    ("crude_stocks", "monotonic_dec"),
    ("cushing_stocks", "monotonic_dec"),
    ("gasoline_stocks", "monotonic_dec"),
    ("distillate_stocks", "monotonic_dec"),
    # Stronger USD → cheaper oil in dollar terms
    ("usd_index", "monotonic_dec"),
    # Refinery utilisation up → crude demand up
    ("refinery_utilization", "monotonic_inc"),
    # Crude exports up → tighter US balance → bullish
    ("crude_exports", "monotonic_inc"),
)

# ── Skewed feature patterns: candidates for quantile-knot pre-transform.  pyGAM
# splines place knots evenly across the feature range, which wastes capacity on
# the tails of skewed distributions.  Pre-applying a quantile (rank) transform
# makes evenly-spaced knots correspond to quantile-spaced knots in the original
# feature space.
SKEWED_FEATURE_PATTERNS = (
    "_pct_change", "_log_return", "_rsi_", "_williams_r_", "_stoch_",
    "_bb_pct_", "_spread", "_momentum_", "_dist_sma_", "_pct_range",
)

# ── Default interaction whitelist (applied only when config.enable_interactions
# is true and BOTH features survive stepwise selection).  Tensor-product surfaces
# capture joint effects pyGAM additive splines miss.  Each entry is a (feature_a,
# feature_b) substring pair that must each match exactly one selected feature.
DEFAULT_INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("usd_index", "crude_stocks"),
    ("vix_close", "ovx"),
    ("CL=F_close_rsi_14", "CL=F_close_atr_14"),
    ("brent_wti_spread", "usd_index"),
    ("wti_m1_m2_spread", "crude_stocks_chg"),
)


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
    forecast_horizon: int = 1,
    target_transform: str = "level",
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex, pd.Series]:
    """Separate target from features with an optional forward target shift.

    When ``forecast_horizon >= 1`` the target is shifted forward so that
    features from day *i* predict the closing price ``forecast_horizon``
    trading days ahead.  The last ``forecast_horizon`` rows are dropped
    (their target is unknown).

    Excludes raw price/volume columns that would leak future information,
    keeping only engineered features and exogenous variables.

    Args:
        df: Feature-engineered DataFrame with DatetimeIndex.
        target_col: Name of the target column.
        forecast_horizon: Number of trading days ahead to forecast.
            0 keeps same-day alignment (useful for extracting the latest
            feature row for live prediction).
        target_transform: ``"level"`` (default) returns the raw price; ``"log_return"``
            returns ``log(P_{t+h} / P_t)``.  In log-return mode the price-level
            autoregressive lag features (matching ``f"{target_col}_lag_*"``) are
            additionally dropped, since they would re-introduce the level into
            the model and dominate the smooths.  Use :func:`compute_anchor_prices`
            and :func:`reconstruct_price_from_returns` to convert predictions back
            to a price scale.

    Returns:
        Tuple of (X, y, feature_names, target_dates, t1) where:

        * ``target_dates`` is the DatetimeIndex of the dates being predicted
          (i.e. ``t + forecast_horizon`` for each kept feature row at ``t``);
        * ``y`` is in the requested transform space;
        * ``t1`` is a pandas Series indexed by the feature-row date (the date
          *at which features are observed*) whose values are the date by
          which label ``y[i]`` is fully known — required by Lopez de Prado
          purged/embargoed CV ([[purged-kfold]]).  For ``forecast_horizon=0``
          ``t1[i] == index[i]`` (no future information needed).
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

    # In log-return mode, drop price-level lags of the target — they share the
    # near-random-walk variance and would dominate the smooths just like the
    # level target itself does.
    if target_transform == "log_return":
        level_lag_prefix = f"{target_col}_lag_"
        for col in list(df.columns):
            if col.startswith(level_lag_prefix):
                exclude_cols.add(col)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_names = feature_cols

    X = df[feature_cols].values.astype(np.float64)
    price_all = df[target_col].values.astype(np.float64)

    if forecast_horizon >= 1:
        # Shift target forward: features from day i predict day i+horizon
        n = len(df)
        n_valid = n - forecast_horizon
        X = X[:n_valid]
        future_price = price_all[forecast_horizon:]
        anchor_price = price_all[:n_valid]
        target_dates = df.index[forecast_horizon:]
        # t1 indexed by the *feature-row* date (df.index[:n_valid]); value is
        # the date at which the corresponding label is fully observed.
        feature_dates = df.index[:n_valid]
        t1 = pd.Series(target_dates, index=feature_dates, name="t1")
    else:
        future_price = price_all
        anchor_price = price_all
        target_dates = df.index
        # horizon=0 (live prediction): label needs no future info.
        t1 = pd.Series(df.index, index=df.index, name="t1")

    if target_transform == "log_return":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(future_price / np.clip(anchor_price, 1e-12, None))
    elif target_transform == "level":
        y = future_price
    else:
        raise ValueError(
            f"Unknown target_transform '{target_transform}'. "
            "Expected 'level' or 'log_return'."
        )

    logger.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features "
                f"(horizon={forecast_horizon}, target_transform={target_transform})")
    logger.debug(f"Features: {feature_names}")

    return X, y, feature_names, target_dates, t1


def compute_anchor_prices(
    df: pd.DataFrame,
    target_col: str,
    forecast_horizon: int = 1,
) -> np.ndarray:
    """Return anchor price ``P_t`` aligned with each forecast in *target_dates*.

    For a row at calendar position *i* whose target is ``df[target_col][i+horizon]``,
    the anchor is ``df[target_col][i]``.  The output length matches the *X*
    returned by :func:`build_feature_matrix` for the same horizon.
    """
    price = df[target_col].values.astype(np.float64)
    if forecast_horizon >= 1:
        return price[: len(price) - forecast_horizon]
    return price


def reconstruct_price_from_returns(
    y_pred_returns: np.ndarray,
    anchor_prices: np.ndarray,
) -> np.ndarray:
    """Convert predicted log-returns back to price levels: ``P_t × exp(r̂_{t+h})``."""
    return np.asarray(anchor_prices, dtype=np.float64) * np.exp(
        np.asarray(y_pred_returns, dtype=np.float64)
    )


def get_target_y_level(
    df: pd.DataFrame,
    target_col: str,
    forecast_horizon: int = 1,
) -> np.ndarray:
    """Return the actual realised price level on each forecast date.

    Mirrors :func:`build_feature_matrix` alignment: for ``forecast_horizon=1``
    the result is ``df[target_col][1:]``.  Used for price-scale evaluation
    metrics regardless of which target transform the model was trained on.
    """
    price = df[target_col].values.astype(np.float64)
    if forecast_horizon >= 1:
        return price[forecast_horizon:]
    return price


# ─────────────────────────────────────────────
# Monotonicity / interaction / skewed-feature helpers
# ─────────────────────────────────────────────

def _get_monotonic_constraint(name: str) -> str | None:
    """Return the monotonicity constraint for *name* or ``None`` if none applies."""
    name_lower = name.lower()
    for pattern, constraint in MONOTONIC_CONSTRAINTS:
        if pattern in name_lower:
            return constraint
    return None


def _is_skewed_feature(name: str) -> bool:
    """True when *name* matches a known skewed-distribution pattern."""
    name_lower = name.lower()
    return any(p in name_lower for p in SKEWED_FEATURE_PATTERNS)


def _resolve_interaction_pairs(
    feature_names: list[str],
    config_pairs: list[list[str]] | None,
) -> list[tuple[int, int]]:
    """Match configured interaction substrings to selected feature indices.

    For each (a_substring, b_substring) pair, find features in *feature_names*
    whose name contains the substring.  When exactly one match is found for
    each side, emit (idx_a, idx_b).  Mismatches are logged and skipped.
    """
    pairs_in = config_pairs if config_pairs else [list(p) for p in DEFAULT_INTERACTION_PAIRS]
    resolved: list[tuple[int, int]] = []
    for pair in pairs_in:
        if len(pair) != 2:
            logger.warning(f"Interaction pair must have exactly 2 entries: {pair} — skipping")
            continue
        a_sub, b_sub = pair[0].lower(), pair[1].lower()
        a_matches = [i for i, n in enumerate(feature_names) if a_sub in n.lower()]
        b_matches = [i for i, n in enumerate(feature_names) if b_sub in n.lower()]
        if len(a_matches) != 1 or len(b_matches) != 1:
            logger.debug(
                f"Interaction skipped (ambiguous or missing): "
                f"{pair[0]}={len(a_matches)} matches, {pair[1]}={len(b_matches)} matches"
            )
            continue
        if a_matches[0] == b_matches[0]:
            continue
        resolved.append((a_matches[0], b_matches[0]))
    if resolved:
        logger.info(
            f"Interaction terms enabled ({len(resolved)} pairs): "
            + ", ".join(f"{feature_names[a]} × {feature_names[b]}" for a, b in resolved)
        )
    return resolved


# ─────────────────────────────────────────────
# Skewed feature transformer (quantile knots via rank transform)
# ─────────────────────────────────────────────

class SkewedFeatureTransformer:
    """Per-column quantile transform applied to a whitelist of skewed features.

    Rationale: pyGAM splines place knots **evenly** across each feature's range.
    For features whose mass is concentrated in one part of the range (returns,
    %B, RSI, momentum, spreads), this wastes basis capacity in the tails and
    starves the body.  Mapping each value to its empirical CDF (rank) makes the
    transformed distribution uniform on [0, 1], so evenly-spaced knots in the
    transformed space correspond to **quantile-spaced knots** in the original
    space.  This is a model-agnostic equivalent to custom knot placement.

    Fit on the training set, then apply identically at prediction.  Holds
    one :class:`sklearn.preprocessing.QuantileTransformer` per skewed column,
    keyed by feature **name** so that downstream column reorderings (from
    stepwise selection, multicollinearity pruning, etc.) do not invalidate
    the fitted transformers — :meth:`subset_to` rebuilds the index map.
    """

    def __init__(self, feature_names: list[str], n_quantiles: int = 200) -> None:
        self.feature_names = list(feature_names)
        self.n_quantiles = n_quantiles
        # transformers keyed by feature name; skewed_idx is a derived view
        self.transformers: dict[str, QuantileTransformer] = {}
        self.skewed_idx: list[int] = [
            i for i, n in enumerate(self.feature_names) if _is_skewed_feature(n)
        ]

    def fit(self, X: np.ndarray) -> "SkewedFeatureTransformer":
        for i in self.skewed_idx:
            name = self.feature_names[i]
            n_q = min(self.n_quantiles, max(10, X.shape[0] // 5))
            qt = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform",
                                     subsample=int(1e6))
            qt.fit(X[:, i].reshape(-1, 1))
            self.transformers[name] = qt
        if self.skewed_idx:
            logger.info(
                f"SkewedFeatureTransformer fitted on {len(self.skewed_idx)} columns: "
                f"{[self.feature_names[i] for i in self.skewed_idx]}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.skewed_idx:
            return X
        X_out = X.copy()
        for i in self.skewed_idx:
            name = self.feature_names[i]
            qt = self.transformers.get(name)
            if qt is not None:
                X_out[:, i] = qt.transform(X[:, i].reshape(-1, 1)).ravel()
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def subset_to(self, new_feature_names: list[str]) -> "SkewedFeatureTransformer":
        """Return a transformer for a subset / reordering of the feature columns.

        Reuses the *fitted* QuantileTransformers from this instance — any new
        feature name not present in the original fit is silently treated as
        non-skewed (no transform applied).  Used after stepwise selection or
        multicollinearity pruning to keep the column-index map aligned.
        """
        out = SkewedFeatureTransformer(new_feature_names, n_quantiles=self.n_quantiles)
        out.transformers = {
            n: self.transformers[n]
            for n in new_feature_names
            if n in self.transformers
        }
        return out


# ─────────────────────────────────────────────
# Bagged GAM ensemble (item 11)
# ─────────────────────────────────────────────

def stationary_bootstrap_indices(
    n: int,
    mean_block_length: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap indices of length *n*.

    Block lengths are geometric with mean *mean_block_length*; block starts
    are uniform on [0, n).  Indices wrap modulo *n*.  Preserves short-range
    autocorrelation in time-series data unlike i.i.d. bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()
    if mean_block_length < 1:
        mean_block_length = 1.0
    p = 1.0 / mean_block_length
    indices = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        L = max(1, int(rng.geometric(p)))
        L = min(L, n - i)
        for k in range(L):
            indices[i + k] = (start + k) % n
        i += L
    return indices


class BaggedLinearGAM:
    """Bagged ensemble of LinearGAMs trained on stationary-bootstrap resamples.

    Cheap variance reduction: predictions average across *n_bags* GAMs each
    fitted on a block-bootstrap of the training data.  Provides ``.predict``
    (mean across bags) and ``.prediction_intervals`` (averaged bag-PIs widened
    by inter-bag variance).
    """

    def __init__(
        self,
        terms_factory,
        lam_values: list[float] | None = None,
        n_bags: int = 10,
        block_length: int = 21,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.terms_factory = terms_factory  # callable: () -> pyGAM terms expression
        self.lam_values = lam_values
        self.n_bags = n_bags
        self.block_length = block_length
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.gams: list[LinearGAM] = []
        # Mirror enough of the LinearGAM API that callers can treat us as one
        self.statistics_: dict = {}
        self.lam: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedLinearGAM":
        rng = np.random.default_rng(self.random_state)
        seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(self.n_bags)]
        block = self.block_length

        def _fit_one(seed: int) -> LinearGAM:
            local_rng = np.random.default_rng(seed)
            idx = stationary_bootstrap_indices(len(y), block, rng=local_rng)
            terms = self.terms_factory()
            return fit_gam(X[idx], y[idx], terms, self.lam_values, n_jobs=1)

        self.gams = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_fit_one)(s) for s in seeds
        )
        # Expose the first bag's statistics so downstream code that introspects
        # ``.statistics_`` / ``.lam`` continues to work.
        self.statistics_ = self.gams[0].statistics_
        self.lam = self.gams[0].lam
        logger.info(f"BaggedLinearGAM fitted ({self.n_bags} bags, block={block}).")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([g.predict(X) for g in self.gams])
        return preds.mean(axis=1)

    def prediction_intervals(self, X: np.ndarray, width: float = 0.95) -> np.ndarray:
        # Stack per-bag PIs and combine: mean lower / mean upper, then widen by
        # 1.96 × std-of-bag-means in each direction to capture model uncertainty
        # across the bootstrap.
        all_lower = np.column_stack([g.prediction_intervals(X, width=width)[:, 0] for g in self.gams])
        all_upper = np.column_stack([g.prediction_intervals(X, width=width)[:, 1] for g in self.gams])
        mean_pred = np.column_stack([g.predict(X) for g in self.gams]).mean(axis=1)
        bag_std = np.column_stack([g.predict(X) for g in self.gams]).std(axis=1, ddof=1)
        # Widen the average band by 1.96 σ_between_bags
        z = 1.96
        lower = all_lower.mean(axis=1) - z * bag_std
        upper = all_upper.mean(axis=1) + z * bag_std
        return np.column_stack([lower, upper])


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

    Optional add-ons (controlled by config flags):
    * **Monotonic constraints**: when ``model.enable_monotonic`` is true and a
      feature name matches :data:`MONOTONIC_CONSTRAINTS`, the spline term is
      built with the corresponding ``constraints=`` argument.
    * **Tensor interactions**: when ``model.enable_interactions`` is true,
      a small whitelist of theoretically-motivated ``te(i, j)`` terms is
      appended (only for pairs where both base features are still present).

    Args:
        feature_indices: Ordered column positions in the *full* feature matrix.
        all_feature_names: Names for every column of the full feature matrix.
        config: Project configuration.

    Returns:
        pyGAM terms expression.
    """
    model_section = config.get("model", {})
    n_splines_default = model_section.get("n_splines", 25)
    enable_monotonic = bool(model_section.get("enable_monotonic", False))
    enable_interactions = bool(model_section.get("enable_interactions", False))
    interaction_pairs_cfg = model_section.get("interactions")
    interaction_n_splines = int(model_section.get("interaction_n_splines", 6))
    interaction_lam = float(model_section.get("interaction_lam", 10.0))

    selected_names = [all_feature_names[i] for i in feature_indices]
    terms = None

    def _spline(new_idx: int, n_splines: int, name: str):
        kwargs: dict = {"n_splines": n_splines}
        if enable_monotonic:
            constraint = _get_monotonic_constraint(name)
            if constraint is not None:
                kwargs["constraints"] = constraint
                logger.debug(f"  applying constraint '{constraint}' to {name}")
        return s(new_idx, **kwargs)

    for new_idx, orig_idx in enumerate(feature_indices):
        name = all_feature_names[orig_idx]
        category = _classify_feature(name)

        if category == "linear":
            term = l(new_idx)
        elif category == "cyclic":
            n_sp = 7 if "day_of_week" in name else 12 if "month" in name else 30
            term = s(new_idx, n_splines=n_sp, basis="cp")
        elif category == "rolling":
            term = _spline(new_idx, n_splines_default, name)
        elif category == "lag":
            term = _spline(new_idx, 20, name)
        elif category == "return":
            term = _spline(new_idx, 15, name)
        else:
            term = _spline(new_idx, 20, name)

        terms = term if terms is None else terms + term
        logger.debug(f"  new={new_idx} orig={orig_idx}: {name} -> {category}")

    # ── Tensor interaction terms (item 5) ────────────────────────────────────
    if enable_interactions:
        pairs = _resolve_interaction_pairs(selected_names, interaction_pairs_cfg)
        for a, b in pairs:
            te_term = te(a, b, n_splines=[interaction_n_splines, interaction_n_splines],
                         lam=interaction_lam)
            terms = te_term if terms is None else terms + te_term

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
    """Time series cross-validator with embargo (Lopez de Prado, 2018).

    Two modes of operation:

    * **t1-aware (preferred).** If ``t1`` is passed, this class delegates to
      :class:`src.cv.PurgedKFold` — the canonical Chapter 7 algorithm.
      Training rows whose label window overlaps the test fold are *purged*
      (not the test rows), and ``pct_embargo = embargo_days / n_samples``
      rows immediately after the test fold are dropped.
    * **Legacy one-sided fallback (no t1).** Reproduces the original
      behaviour of stripping ``embargo_days`` observations from the *start*
      of each test fold.  Kept so existing call sites that have not yet
      threaded ``t1`` continue to work unchanged.

    Args:
        n_splits: Number of CV folds.
        embargo_days: Embargo window size in observations.  In legacy mode,
            this is the number of test rows stripped at each fold boundary;
            in t1-aware mode, it is the number of post-test rows embargoed.
        t1: Optional Series of label end times (see :func:`build_feature_matrix`).
            Required for the canonical purged behaviour.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 200,
        t1: pd.Series | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.t1 = t1

    def split(
        self, X: np.ndarray, y=None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        # ── Preferred path: delegate to PurgedKFold when t1 is available ──
        if self.t1 is not None:
            from src.cv import PurgedKFold

            n_samples = len(X)
            pct_embargo = (self.embargo_days / n_samples) if n_samples else 0.0
            inner = PurgedKFold(
                n_splits=self.n_splits, t1=self.t1, pct_embargo=pct_embargo,
            )
            yield from inner.split(X)
            return

        # ── Legacy fallback: original one-sided embargo on the test side ──
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
    t1: pd.Series | None = None,
) -> dict:
    """Perform embargoed time series cross-validation.

    Uses :class:`EmbargoedTimeSeriesSplit` to ensure that:
    1. We always predict future from past (no temporal leakage).
    2. Rolling/lag features computed on the full dataset do not bleed
       training-period information into test observations at fold boundaries.

    When ``t1`` is supplied the splitter delegates to
    :class:`src.cv.PurgedKFold` (canonical Chapter 7 purge + bilateral
    embargo).  When omitted, the legacy one-sided test-side embargo is
    used — kept for backward compatibility.

    Folds are fitted in parallel when ``n_jobs != 1``.  Each fold worker
    calls ``fit_gam`` with ``n_jobs=1`` to avoid nested parallelism.

    Args:
        X: Feature matrix.
        y: Target vector.
        terms: pyGAM terms expression.
        n_splits: Number of CV folds.
        embargo_days: Embargo window size (observations).  In the t1-aware
            path this becomes ``pct_embargo = embargo_days / n_samples``.
            Should equal the largest rolling window used in feature
            engineering (default 200 = the 200-day SMA window).
        lam_values: Lambda values for grid search.
        n_jobs: Number of parallel jobs for fold fitting.
            1 = sequential (default).  -1 = all logical CPUs.
        t1: Optional label end-times series (see :func:`build_feature_matrix`).

    Returns:
        Dict with fold_metrics, mean_mae, mean_rmse, mean_mape, embargo_days.
    """
    if lam_values is None:
        lam_values = np.logspace(-3, 3, 11).tolist()

    splitter = EmbargoedTimeSeriesSplit(
        n_splits=n_splits, embargo_days=embargo_days, t1=t1,
    )

    cv_mode = "purged-kfold (t1-aware)" if t1 is not None else "legacy one-sided"
    logger.info(
        f"Starting {n_splits}-fold embargoed CV [{cv_mode}] "
        f"(embargo_days={embargo_days}, n_jobs={n_jobs})"
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
    # Scalar λ for trial fits — broadcasts safely across SplineTerm + TensorTerm
    # (per-term list mutation breaks the term/penalty mapping when interactions are on).
    trial_lam_scalar = float(np.median(lam_values))

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
        trial_gam_fast = LinearGAM(trial_terms)
        trial_gam_fast.lam = trial_lam_scalar
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
            trial_gam = LinearGAM(trial_terms)
            trial_gam.lam = trial_lam_scalar
            trial_gam.fit(X[:, trial_indices], y)
            logger.info(
                f"  Hard-cap drop: '{candidate_name}' "
                f"({len(current_indices)} → {len(trial_indices)} terms)"
            )
            dropped_names.append(candidate_name)
            current_indices = trial_indices
            gam = trial_gam

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
    1. Build feature matrix in the configured target space (level or log-return).
    2. Drop near-collinear features via correlation pruning (item 10).
    3. Optionally apply quantile transforms to skewed features (item 8).
    4. If ``model.stepwise_selection`` is true, backward AIC/BIC elimination.
    5. Run embargoed time-series cross-validation on the selected features.
    6. Fit the final mu model.  When ``model.iterative_refits > 0``, perform
       a poor-man's GAMLSS pass: re-fit μ with weights = 1/σ̂² (item 9).
    7. Optionally bag the mu model (item 11).
    8. Fit σ, ν, τ sub-models.
    9. Persist all models, feature names, transformer, and metadata.
    """
    root = get_project_root(config)
    model_dir = root / "outputs" / "models"
    model_cfg = resolve_model_config(config)
    lam_values = model_cfg.get("lam_search")
    n_jobs = model_cfg.get("n_jobs", 1)

    feat_cfg = config.get("features", {})
    target_transform = feat_cfg.get("target_transform", "level")

    # Load features
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = feat_cfg["target"]

    # ── Build feature matrix (in target_transform space) ──────────────────────
    X, y, feature_names, target_dates, t1 = build_feature_matrix(
        df, target, target_transform=target_transform
    )
    logger.info(
        f"train_and_save: target_transform='{target_transform}', "
        f"y range=[{y.min():.4f}, {y.max():.4f}], n_features={len(feature_names)}"
    )

    # ── Item 10: multicollinearity pruning before stepwise BIC ────────────────
    corr_threshold = float(model_cfg.get("collinearity_threshold", 0.97))
    if corr_threshold and corr_threshold < 1.0:
        keep_idx, kept_names, dropped_corr = drop_correlated_features(
            X, feature_names, threshold=corr_threshold,
        )
        if dropped_corr:
            logger.info(
                f"Multicollinearity prune (|ρ|>{corr_threshold}): "
                f"dropped {len(dropped_corr)} features → {dropped_corr}"
            )
            X = X[:, keep_idx]
            feature_names = kept_names

    # ── Item 8: SkewedFeatureTransformer (quantile-knot equivalent) ───────────
    use_quantile_knots = bool(model_cfg.get("quantile_knots", False))
    transformer: SkewedFeatureTransformer | None = None
    if use_quantile_knots:
        transformer = SkewedFeatureTransformer(feature_names).fit(X)
        if transformer.skewed_idx:
            X = transformer.transform(X)

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

        # Stepwise drops columns; reuse the fitted QTs but re-index by name.
        # X has already been transformed in-place above, so we don't re-fit —
        # subset_to just builds a new column-index map over the surviving names.
        if transformer is not None:
            transformer = transformer.subset_to(feature_names)

    # Build terms over the (possibly reduced) feature set.
    terms = define_gam_terms(feature_names, config)

    # Cross-validate on selected features.  Pass `t1` so the splitter follows
    # Chapter 7 purging instead of the legacy one-sided embargo.
    cv_results = cross_validate(
        X, y, terms,
        n_splits=model_cfg.get("cv_splits", 5),
        embargo_days=model_cfg.get("embargo_days", 200),
        lam_values=lam_values,
        n_jobs=n_jobs,
        t1=t1,
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

    # ── Item 9: iterative heteroscedastic refit (poor-man's GAMLSS) ───────────
    n_refits = int(model_cfg.get("iterative_refits", 0))
    for it in range(n_refits):
        weights = 1.0 / np.clip(sigma_pred ** 2, 1e-8, None)
        # Normalise weights to mean 1 so lam stays comparable across iterations
        weights = weights * (len(weights) / weights.sum())
        logger.info(
            f"Iterative refit {it + 1}/{n_refits}: re-fitting mu with sample weights "
            f"(min={weights.min():.4f}, max={weights.max():.4f})"
        )
        mu_gam_iter = LinearGAM(terms, lam=mu_gam.lam)
        mu_gam_iter.fit(X, y, weights=weights)
        mu_gam = mu_gam_iter
        mu_pred = mu_gam.predict(X)
        mu_residuals = y - mu_pred
        # Re-fit sigma on updated residuals (skip stepwise — keep selected cols)
        sigma_gam_iter = LinearGAM(
            _build_gam_terms_for_indices(
                list(range(len(sigma_feature_names))), sigma_feature_names, config
            ),
            lam=sigma_gam.lam,
        )
        sigma_gam_iter.fit(X_sigma, np.log(np.abs(mu_residuals) + 1e-6))
        sigma_gam = sigma_gam_iter
        sigma_pred = get_sigma_from_sigma_gam(sigma_gam, X_sigma)

    # ── Item 11: optional bagged ensemble around mu (replaces mu_gam if enabled)
    bagging_cfg = model_cfg.get("bagging", {}) or {}
    if bagging_cfg.get("enabled", False):
        n_bags = int(bagging_cfg.get("n_bags", 10))
        block_length = int(bagging_cfg.get("block_length", 21))
        logger.info(f"Fitting bagged mu ensemble: {n_bags} bags, block={block_length}")
        feature_names_for_factory = list(feature_names)

        def _terms_factory():
            return define_gam_terms(feature_names_for_factory, config)

        bagged = BaggedLinearGAM(
            _terms_factory,
            lam_values=lam_values,
            n_bags=n_bags,
            block_length=block_length,
            random_state=int(bagging_cfg.get("random_state", 0)),
            n_jobs=int(bagging_cfg.get("n_jobs", 1)),
        )
        bagged.fit(X, y)
        mu_gam = bagged
        mu_pred = mu_gam.predict(X)
        mu_residuals = y - mu_pred
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

    # ── Save all models, feature name lists, and metadata ─────────────────────
    save_model(mu_gam, model_dir / "gam_model.pkl")        # backward compat
    save_feature_names(feature_names, model_dir / "feature_names.pkl")

    all_models = {"mu": mu_gam, "sigma": sigma_gam, "nu": nu_gam, "tau": tau_gam}
    save_all_models(all_models, model_dir / "gam_models.pkl")

    save_feature_names(sigma_feature_names, model_dir / "sigma_feature_names.pkl")
    save_feature_names(nu_feature_names, model_dir / "nu_feature_names.pkl")
    save_feature_names(tau_feature_names, model_dir / "tau_feature_names.pkl")

    if transformer is not None:
        joblib.dump(transformer, model_dir / "feature_transformer.pkl")
        logger.info(f"Saved skewed-feature transformer to {model_dir / 'feature_transformer.pkl'}")

    # Persist target_transform metadata so downstream code knows how to interpret y
    metadata = {
        "target_transform": target_transform,
        "target_col": target,
        "forecast_horizon": 1,
        "selected_features": list(feature_names),
        "uses_transformer": transformer is not None,
        "iterative_refits": n_refits,
        "bagging_enabled": bool(bagging_cfg.get("enabled", False)),
    }
    joblib.dump(metadata, model_dir / "model_metadata.pkl")
    logger.info(f"Saved model metadata: {metadata}")

    logger.info(
        f"All models saved. "
        f"μ: {len(feature_names)} terms, "
        f"σ: {len(sigma_feature_names)} terms, "
        f"ν: {len(nu_feature_names)} terms, "
        f"τ: {len(tau_feature_names)} terms"
    )

    return mu_gam, sigma_gam, nu_gam, tau_gam, cv_results


# ─────────────────────────────────────────────
# Multicollinearity pruning (item 10)
# ─────────────────────────────────────────────

def drop_correlated_features(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.97,
) -> tuple[list[int], list[str], list[str]]:
    """Greedy pruning of features whose pairwise |Pearson ρ| exceeds *threshold*.

    Walks columns left-to-right; when the candidate column is too correlated
    with any already-kept column, the candidate is dropped.  This favours
    earlier-in-the-list features (which by convention in this project are
    shorter-window / more reactive — e.g., 14d SMA before 200d SMA).

    Returns:
        (keep_indices, kept_names, dropped_names) where *keep_indices* index
        into the original *X* / *feature_names*.
    """
    if X.shape[1] != len(feature_names):
        raise ValueError("X.shape[1] must equal len(feature_names)")
    if X.shape[1] <= 1:
        return list(range(X.shape[1])), list(feature_names), []

    # Build rank-correlation-friendly sample (cap to first 5000 rows for speed
    # on large feature matrices)
    n_sample = min(X.shape[0], 5000)
    X_sample = X[-n_sample:]  # use most recent rows — distribution most relevant
    # Pearson ρ via numpy; ignore columns with zero variance
    std = X_sample.std(axis=0)
    nonzero = std > 1e-12

    keep: list[int] = []
    dropped: list[str] = []
    kept_data: list[np.ndarray] = []  # standardised columns we have kept

    for j in range(X.shape[1]):
        if not nonzero[j]:
            dropped.append(feature_names[j])
            continue
        col_std = (X_sample[:, j] - X_sample[:, j].mean()) / std[j]
        too_correlated = False
        for prev in kept_data:
            rho = float(np.dot(col_std, prev) / n_sample)
            if abs(rho) > threshold:
                too_correlated = True
                break
        if too_correlated:
            dropped.append(feature_names[j])
        else:
            keep.append(j)
            kept_data.append(col_std)

    kept_names = [feature_names[i] for i in keep]
    return keep, kept_names, dropped


# ─────────────────────────────────────────────
# Quantile/Expectile GAMs and conformal calibration (item 7)
# ─────────────────────────────────────────────

def fit_expectile_gams(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: dict,
    expectiles: tuple[float, ...] = (0.05, 0.50, 0.95),
    lam_values: list[float] | None = None,
    n_jobs: int = 1,
) -> dict[float, ExpectileGAM]:
    """Fit one :class:`pygam.ExpectileGAM` per requested expectile level.

    Expectiles are an ALS-loss generalisation of OLS:  expectile(0.5) is the
    mean, expectile(0.05) is a lower-tail expectile, expectile(0.95) upper.
    Distinct from quantile regression but shares the asymmetric-loss spirit
    and is what pyGAM ships natively.

    The output dict can be used as the primary PI source — a 90% PI is then
    ``[expectile(0.05), expectile(0.95)]`` from the *training* loss directly,
    rather than from a Gaussian residual assumption.
    """
    out: dict[float, ExpectileGAM] = {}
    for q in expectiles:
        terms = define_gam_terms(feature_names, config)
        gam = ExpectileGAM(terms, expectile=float(q))
        if lam_values is not None and len(lam_values) > 1:
            try:
                gam.gridsearch(X, y, lam=lam_values, progress=False)
            except Exception as exc:
                logger.warning(f"ExpectileGAM gridsearch failed for q={q}: {exc} — falling back to default lam.")
                gam.fit(X, y)
        else:
            gam.fit(X, y)
        out[float(q)] = gam
        logger.info(f"Fitted ExpectileGAM at q={q}.")
    return out


def conformal_residual_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray | None = None,
    alpha: float = 0.10,
) -> float:
    """Split-conformal nonconformity quantile for symmetric prediction intervals.

    Given holdout residuals ``r = y_true - y_pred`` and per-row scale *sigma*
    (defaults to 1), returns the empirical ``ceil((n+1)(1-α)) / n``-quantile
    of ``|r|/σ``.  Used to build a finite-sample-valid PI of the form

        ``[y_pred − q · σ, y_pred + q · σ]``

    that achieves exact ≥(1−α) coverage on exchangeable data — a distribution-
    free guarantee even when the parametric (e.g., Johnson SU) assumption is
    mis-specified.  See Vovk et al. (2005), Lei et al. (2018).
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("conformal_residual_quantile: empty calibration set")
    if sigma is None:
        sigma = np.ones(n)
    sigma = np.clip(sigma, 1e-8, None)
    nonconformity = np.abs(y_true - y_pred) / sigma
    # Conservative ceiling correction → exact finite-sample coverage
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(n - 1, rank))
    return float(np.sort(nonconformity)[rank])
