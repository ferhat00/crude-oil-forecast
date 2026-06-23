"""Conditional-volatility models for the predictive scale σ.

The σ-GAM in :mod:`src.model` regresses ``log|residual|`` on contemporaneous
volatility features.  That captures *level* dependence (σ is larger when OVX /
realised-vol features are high) but **not** the serial *clustering* that
dominates daily financial volatility — a calm day is most likely followed by a
calm day, a shock by more shocks.  This module adds that serial dynamic:

* :func:`ewma_conditional_sigma` — RiskMetrics EWMA; fully causal, zero
  parameters, the robust fallback.
* :func:`garch_conditional_sigma` — GARCH(1,1) via the ``arch`` package (the
  parsimonious daily-vol workhorse), with automatic EWMA fallback when the
  optimiser fails to converge.
* :func:`fit_har` / :func:`har_predict` — Corsi (2009) HAR on a realised-
  variance series for clean multi-horizon variance forecasts (used by the
  multi-horizon fan chart in a later phase).

All inputs are model-space residuals / returns (``O(1e-2)`` for log-returns);
``arch`` is imported lazily so importing this module never requires it.

References
----------
Engle (1982); Bollerslev (1986) — GARCH.
J.P. Morgan RiskMetrics (1996) — EWMA, λ = 0.94 for daily data.
Corsi (2009), *J. Financial Econometrics* — HAR.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)

# RiskMetrics decay for daily data.
EWMA_LAMBDA = 0.94
# Scale residuals up before fitting GARCH (arch is happiest with returns in the
# rough range of percentage points); the conditional vol is scaled back down.
GARCH_SCALE = 100.0


@dataclasses.dataclass
class ConditionalSigma:
    """Output of :func:`garch_conditional_sigma`.

    Attributes:
        sigma: In-sample conditional volatility aligned to the (finite)
            residual series — each ``sigma[t]`` is filtered from information up
            to ``t`` and is safe to use as a per-row scale.
        forecast: One-step-ahead conditional volatility (the scale to apply to
            the next, not-yet-observed, residual).
        kind: ``"garch"`` or ``"ewma"`` — which model produced the estimate.
    """
    sigma: np.ndarray
    forecast: float
    kind: str


def ewma_conditional_sigma(
    residuals: np.ndarray,
    lam: float = EWMA_LAMBDA,
) -> tuple[np.ndarray, float]:
    """Causal RiskMetrics EWMA conditional volatility.

    ``σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}`` — each ``σ_t`` uses only residuals
    strictly before ``t``, so the series is leakage-free for backtesting.

    Returns:
        ``(sigma_series, next_forecast)`` where ``sigma_series`` has the same
        length as the finite residuals and ``next_forecast`` is the one-step-
        ahead σ after the last observation.
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return np.array([]), 1e-3

    seed = np.nanvar(r[: min(n, 20)])
    if not np.isfinite(seed) or seed <= 0:
        seed = float(np.nanvar(r)) or 1e-6
    var = np.empty(n, dtype=np.float64)
    prev = float(seed)
    for t in range(n):
        var[t] = prev                     # σ_t depends on residuals[:t]
        prev = lam * prev + (1.0 - lam) * r[t] ** 2
    sigma = np.sqrt(np.clip(var, 1e-12, None))
    next_forecast = float(np.sqrt(max(prev, 1e-12)))
    return sigma, next_forecast


def fit_garch(
    residuals: np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    scale: float = GARCH_SCALE,
):
    """Fit a zero-mean GARCH(p, q) to residuals via ``arch``; ``None`` on failure.

    The mean is fixed at zero (the GAM already removed the conditional mean).
    Returns the fitted ``arch`` result object, or ``None`` if there are too few
    observations or the optimiser fails (caller falls back to EWMA).
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 100:
        return None
    try:
        from arch import arch_model

        am = arch_model(
            r * scale, mean="Zero", vol="GARCH", p=p, q=q, dist=dist, rescale=False,
        )
        return am.fit(disp="off", show_warning=False)
    except Exception as exc:  # convergence / linalg / import
        logger.debug(f"GARCH fit failed ({exc}); will fall back to EWMA.")
        return None


def garch_conditional_sigma(
    residuals: np.ndarray,
    lam: float = EWMA_LAMBDA,
    scale: float = GARCH_SCALE,
    dist: str = "normal",
) -> ConditionalSigma:
    """Conditional volatility from GARCH(1,1), falling back to EWMA.

    Returns a :class:`ConditionalSigma` with the in-sample conditional-vol
    series and the one-step-ahead forecast, both in the residuals' own units.
    """
    res = fit_garch(residuals, dist=dist, scale=scale)
    if res is not None:
        try:
            sigma = np.asarray(res.conditional_volatility, dtype=np.float64) / scale
            fc_var = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
            forecast = float(np.sqrt(max(fc_var, 1e-12)) / scale)
            if np.isfinite(forecast) and forecast > 0 and np.all(np.isfinite(sigma)):
                return ConditionalSigma(sigma=sigma, forecast=forecast, kind="garch")
        except Exception as exc:
            logger.debug(f"GARCH forecast extraction failed ({exc}); using EWMA.")

    sigma, forecast = ewma_conditional_sigma(residuals, lam=lam)
    return ConditionalSigma(sigma=sigma, forecast=forecast, kind="ewma")


# ─────────────────────────────────────────────
# HAR (Corsi 2009) — multi-horizon realised-variance forecasting
# ─────────────────────────────────────────────

def _har_design(rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the HAR design matrix (daily / weekly / monthly RV averages)."""
    rv = np.asarray(rv, dtype=np.float64)
    n = rv.size
    rows_x, rows_y = [], []
    for t in range(22, n):
        daily = rv[t - 1]
        weekly = rv[t - 5:t].mean()
        monthly = rv[t - 22:t].mean()
        rows_x.append([1.0, daily, weekly, monthly])
        rows_y.append(rv[t])
    return np.asarray(rows_x), np.asarray(rows_y)


def fit_har(rv: np.ndarray) -> dict:
    """Fit Corsi's HAR model ``RV_t = β0 + βd·RV_d + βw·RV_w + βm·RV_m``.

    Args:
        rv: 1-D realised-variance (or realised-vol) series.

    Returns:
        Dict with ``coef`` (length-4 array ``[β0, βd, βw, βm]``) and ``n_obs``.
        Falls back to an intercept-only model (mean) when the series is short.
    """
    rv = np.asarray(rv, dtype=np.float64)
    rv = rv[np.isfinite(rv)]
    if rv.size < 30:
        return {"coef": np.array([float(np.nanmean(rv)) if rv.size else 0.0, 0, 0, 0]),
                "n_obs": int(rv.size)}
    X, y = _har_design(rv)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"coef": coef, "n_obs": int(len(y))}


def har_predict(model: dict, rv_recent: np.ndarray) -> float:
    """One-step-ahead HAR forecast from the most recent ≥ 22 RV observations."""
    rv = np.asarray(rv_recent, dtype=np.float64)
    rv = rv[np.isfinite(rv)]
    coef = model["coef"]
    if rv.size < 22:
        return float(coef[0]) if rv.size == 0 else float(np.mean(rv))
    feats = np.array([1.0, rv[-1], rv[-5:].mean(), rv[-22:].mean()])
    return float(max(np.dot(coef, feats), 0.0))
