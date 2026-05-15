"""Model evaluation: metrics, baselines, and diagnostic + probability plots."""

import logging
from pathlib import Path

import dataclasses

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.config_loader import get_project_root
from src.model import (
    build_feature_matrix,
    compute_anchor_prices,
    get_target_y_level,
    load_feature_names,
    load_model,
    reconstruct_price_from_returns,
)

logger = logging.getLogger(__name__)

# Confidence levels and their matching blue shades (dark → light)
CI_LEVELS = [0.50, 0.80, 0.90, 0.95]
# ColorBrewer Blues: darkest at centre, lightest at the outside
CI_COLORS_MPL = ["#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"]
CI_ALPHAS_MPL = [0.75, 0.55, 0.40, 0.25]


# ─────────────────────────────────────────────
# 4-Parameter Predictive Distribution (Johnson SU)
# ─────────────────────────────────────────────

@dataclasses.dataclass
class PredictiveDistribution:
    """Johnson SU (Sinh-Arcsinh / SHASH) predictive distribution.

    Encapsulates the four parameters estimated from model residuals:

    * **ν (nu)** — skewness shape.  ν = 0 is symmetric; ν > 0 right-skewed.
    * **τ (tau)** — tail-weight shape (must be > 0).  Smaller τ → heavier
      tails than Normal; τ → ∞ approaches Gaussian tails.
    * **loc_std** — mean of the standardised residuals (ideally ≈ 0).
    * **scale_std** — scale of the standardised residuals (ideally ≈ 1).

    For a new prediction with GAM mean *μ* and heteroscedastic scale *σ*
    (derived from the GAM's 95% PI), the realised distribution is::

        johnsonsu(ν, τ, loc = μ + σ·loc_std, scale = σ·scale_std)

    This preserves the per-observation scale heterogeneity from the GAM
    while replacing the Normal assumption with a distribution that can
    capture skewness and excess kurtosis in the residuals.
    """
    nu: float | np.ndarray  # skewness — scalar (global) or per-observation array
    tau: float | np.ndarray  # tail-weight — scalar or per-observation array (smaller = fatter tails)
    loc_std: float   # location offset in standardised space
    scale_std: float # scale in standardised space

    # ── Per-observation distribution helpers ────────────────────────────────────

    def _params(self, mu: float, sigma: float, idx: int | None = None) -> tuple:
        """Return the (a, b, loc, scale) scipy args for one prediction.

        When *nu* or *tau* are per-observation arrays the observation index
        *idx* selects the correct value.  When *idx* is None and the fields
        are arrays their mean is used (suitable for single-date display).
        """
        if isinstance(self.nu, np.ndarray):
            nu_i = float(self.nu[idx]) if idx is not None else float(np.mean(self.nu))
        else:
            nu_i = float(self.nu)
        if isinstance(self.tau, np.ndarray):
            tau_i = float(self.tau[idx]) if idx is not None else float(np.mean(self.tau))
        else:
            tau_i = float(self.tau)
        return (
            nu_i,
            tau_i,
            float(mu) + float(sigma) * self.loc_std,
            float(sigma) * self.scale_std,
        )

    def pdf(self, x: np.ndarray, mu: float, sigma: float, idx: int | None = None) -> np.ndarray:
        """Evaluate the predictive PDF at price grid *x*."""
        a, b, loc, scale = self._params(mu, sigma, idx=idx)
        return stats.johnsonsu.pdf(x, a, b, loc=loc, scale=scale)

    def ppf(self, q: float, mu: float, sigma: float, idx: int | None = None) -> float:
        """Percent-point function (inverse CDF) at quantile *q*."""
        a, b, loc, scale = self._params(mu, sigma, idx=idx)
        return float(stats.johnsonsu.ppf(q, a, b, loc=loc, scale=scale))

    def cdf(self, x: float, mu: float, sigma: float, idx: int | None = None) -> float:
        """CDF evaluated at price *x*."""
        a, b, loc, scale = self._params(mu, sigma, idx=idx)
        return float(stats.johnsonsu.cdf(x, a, b, loc=loc, scale=scale))

    # ── Vectorised over many observations ─────────────────────────────────────

    def ppf_array(self, q: float, y_pred: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Return PPF at quantile *q* for each (μ_i, σ_i) pair.

        When *nu* or *tau* are per-observation arrays each call to
        :meth:`_params` uses the matching observation index *i*.
        """
        return np.array([
            float(stats.johnsonsu.ppf(q, *self._params(m, s, idx=i)))
            for i, (m, s) in enumerate(zip(y_pred, sigma))
        ])

    def get_all_intervals(
        self,
        y_pred: np.ndarray,
        sigma: np.ndarray,
        widths: list[float] | None = None,
    ) -> dict[float, np.ndarray]:
        """Compute prediction intervals at multiple confidence levels.

        Returns:
            Dict mapping width → array of shape (n, 2) with [lower, upper].
        """
        if widths is None:
            widths = CI_LEVELS
        result = {}
        for w in widths:
            q_lo = (1.0 - w) / 2.0
            q_hi = (1.0 + w) / 2.0
            lower = self.ppf_array(q_lo, y_pred, sigma)
            upper = self.ppf_array(q_hi, y_pred, sigma)
            result[w] = np.column_stack([lower, upper])
        return result

    def describe(self) -> str:
        nu_val = float(np.mean(self.nu)) if isinstance(self.nu, np.ndarray) else float(self.nu)
        tau_val = float(np.mean(self.tau)) if isinstance(self.tau, np.ndarray) else float(self.tau)
        nu_type = " (per-obs)" if isinstance(self.nu, np.ndarray) else ""
        tau_type = " (per-obs)" if isinstance(self.tau, np.ndarray) else ""
        tail_desc = (
            "heavier than Normal (leptokurtic)"
            if tau_val < 1.0
            else "similar to or lighter than Normal"
        )
        skew_desc = (
            "right-skewed" if nu_val > 0.05
            else "left-skewed" if nu_val < -0.05
            else "approximately symmetric"
        )
        return (
            f"Johnson SU: ν={nu_val:.4f}{nu_type} ({skew_desc}), "
            f"τ={tau_val:.4f}{tau_type} ({tail_desc}), "
            f"loc_std={self.loc_std:.4f}, scale_std={self.scale_std:.4f}"
        )


def fit_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    tau_max: float = 0.8,
) -> PredictiveDistribution:
    """Fit a Johnson SU distribution to the standardised model residuals.

    Standardising by the per-observation GAM scale *σ* means the fitted
    shape parameters (ν, τ) capture the distributional form of the errors
    independently of their magnitude, while the magnitude is preserved
    through *σ* at prediction time.

    Args:
        y_true: Observed values.
        y_pred: GAM point predictions (aligned with y_true).
        sigma: Per-observation predictive std deviations (from GAM 95% PI).
        tau_max: Upper bound on the fitted τ parameter.  Since smaller τ
            means heavier tails, capping τ prevents MLE from converging to a
            Gaussian-tailed solution when in-sample residuals look Normal.
            Default 0.8 guarantees leptokurtic (heavier-than-Normal) tails.

    Returns:
        Fitted :class:`PredictiveDistribution`.
    """
    residuals = y_true - y_pred
    r_std = residuals / np.clip(sigma, 1e-8, None)   # standardise

    # scipy.stats.johnsonsu.fit returns (a=ν, b=τ, loc, scale)
    nu, tau, loc_std, scale_std = stats.johnsonsu.fit(r_std)
    tau = min(tau, tau_max)  # enforce fat-tail floor (smaller τ → heavier tails)

    dist = PredictiveDistribution(
        nu=nu, tau=tau, loc_std=loc_std, scale_std=scale_std
    )
    logger.info(f"Fitted predictive distribution — {dist.describe()}")
    return dist


def calibrate_sigma_scale(
    gam,
    X: np.ndarray,
    y_true: np.ndarray,
    dist: "PredictiveDistribution",
    target_coverage: float = 0.90,
    val_frac: float = 0.2,
    search_range: tuple[float, float] = (0.5, 5.0),
    n_steps: int = 100,
) -> float:
    """Find the smallest *sigma_scale* that achieves target empirical coverage.

    Uses the most recent ``val_frac`` fraction of the data as a holdout
    validation set (time-order preserved — no leakage).  Searches linearly
    over *search_range* and returns the first candidate where the observed
    proportion of actuals inside the ``target_coverage`` prediction interval
    meets or exceeds the target.

    Args:
        gam: Fitted LinearGAM.
        X: Feature matrix aligned with y_true.
        y_true: Observed prices.
        dist: Fitted :class:`PredictiveDistribution` (carries Johnson SU shape).
        target_coverage: Desired empirical coverage fraction (default: 0.90).
        val_frac: Fraction of data reserved for validation (default: 0.20).
        search_range: (min, max) range of sigma_scale values to search.
        n_steps: Number of candidate values within search_range.

    Returns:
        Calibrated sigma_scale float.  Falls back to ``search_range[1]`` with
        a warning if the target cannot be reached within the search range.
    """
    n_val = max(int(len(y_true) * val_frac), 30)
    X_val = X[-n_val:]
    y_val = y_true[-n_val:]
    y_pred_val = gam.predict(X_val)

    q_lo = (1.0 - target_coverage) / 2.0
    q_hi = (1.0 + target_coverage) / 2.0

    for scale in np.linspace(search_range[0], search_range[1], n_steps):
        sigma_val = get_prediction_sigma(gam, X_val, sigma_scale=scale)
        lower = dist.ppf_array(q_lo, y_pred_val, sigma_val)
        upper = dist.ppf_array(q_hi, y_pred_val, sigma_val)
        coverage = float(np.mean((y_val >= lower) & (y_val <= upper)))
        if coverage >= target_coverage:
            logger.info(
                f"calibrate_sigma_scale: scale={scale:.3f} → "
                f"empirical {target_coverage * 100:.0f}% coverage = {coverage * 100:.1f}%"
            )
            return float(scale)

    logger.warning(
        f"calibrate_sigma_scale: could not reach {target_coverage * 100:.0f}% coverage "
        f"within search_range={search_range}. Returning {search_range[1]:.1f}."
    )
    return float(search_range[1])


# ─────────────────────────────────────────────
# Basic Metrics
# ─────────────────────────────────────────────

def naive_baseline_predictions(y: np.ndarray) -> np.ndarray:
    """Naive forecast: tomorrow = today."""
    pred = np.empty_like(y)
    pred[0] = np.nan
    pred[1:] = y[:-1]
    return pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE."""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compare_with_baseline(
    y_true: np.ndarray,
    y_gam: np.ndarray,
    y_naive: np.ndarray,
) -> pd.DataFrame:
    """Compare GAM and naive baseline metrics side by side."""
    gam_metrics = compute_metrics(y_true, y_gam)
    naive_metrics = compute_metrics(y_true, y_naive)
    comparison = pd.DataFrame(
        [
            {"Model": "GAM", **gam_metrics},
            {"Model": "Naive (tomorrow=today)", **naive_metrics},
        ]
    )
    return comparison.set_index("Model")


# ─────────────────────────────────────────────
# Item 13: probabilistic scoring metrics
# ─────────────────────────────────────────────

def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_anchor: np.ndarray,
) -> float:
    """Fraction of forecasts whose directional sign matches the realised move.

    Compares ``sign(y_pred - y_anchor)`` with ``sign(y_true - y_anchor)`` where
    *y_anchor* is the previous trading day's close.  A naive forecast (tomorrow
    = today) has zero directional information; a GAM with directional accuracy
    > 0.5 has trading value even if its MAE is similar to the naive baseline.
    Ties (zero change) are counted as misses.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_anchor = np.asarray(y_anchor)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_anchor))
    if not mask.any():
        return float("nan")
    pred_sign = np.sign(y_pred[mask] - y_anchor[mask])
    true_sign = np.sign(y_true[mask] - y_anchor[mask])
    return float(np.mean(pred_sign == true_sign))


def pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    q: float,
) -> float:
    """Pinball (quantile) loss at level *q*.

    The proper scoring rule for quantile forecasts: minimised in expectation
    by reporting the true ``q``-quantile.  Use to evaluate ExpectileGAM /
    quantile-regression PIs at multiple levels.
    """
    diff = np.asarray(y_true) - np.asarray(y_pred_quantile)
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def crps_via_samples(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Per-observation CRPS estimated from a sample matrix.

    *samples* has shape ``(n_obs, n_draws)``.  CRPS is computed via the
    expectation form

        ``CRPS_i = E|X − y_i| − ½ E|X − X'|``

    using ``X, X'`` as two independent halves of the sample matrix.
    Returns one scalar per observation.
    """
    samples = np.asarray(samples)
    y_true = np.asarray(y_true).reshape(-1, 1)
    n_draws = samples.shape[1]
    half = n_draws // 2
    if half < 2:
        raise ValueError("crps_via_samples needs at least 4 samples per observation")
    s1 = samples[:, :half]
    s2 = samples[:, half: 2 * half]
    return np.mean(np.abs(s1 - y_true), axis=1) - 0.5 * np.mean(np.abs(s1 - s2), axis=1)


def crps_johnsonsu(
    y_true: np.ndarray,
    y_pred_mu: np.ndarray,
    sigma: np.ndarray,
    dist: "PredictiveDistribution",
    n_samples: int = 500,
    random_state: int = 0,
) -> np.ndarray:
    """Per-observation CRPS for Johnson SU predictive distributions via MC.

    Generates *n_samples* draws from each row's Johnson SU and feeds them to
    :func:`crps_via_samples`.  Returns one CRPS value per observation.
    """
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    samples = np.empty((n, n_samples))
    for i in range(n):
        a, b, loc, scale = dist._params(float(y_pred_mu[i]), float(sigma[i]), idx=i)
        samples[i, :] = stats.johnsonsu.rvs(
            a, b, loc=loc, scale=scale, size=n_samples, random_state=rng
        )
    return crps_via_samples(y_true, samples)


def log_score_johnsonsu(
    y_true: np.ndarray,
    y_pred_mu: np.ndarray,
    sigma: np.ndarray,
    dist: "PredictiveDistribution",
) -> np.ndarray:
    """Negative log predictive density (log-score) for Johnson SU.

    Lower is better.  The log-score is a strictly proper scoring rule.
    Returns one score per observation.
    """
    n = len(y_true)
    out = np.empty(n)
    for i in range(n):
        a, b, loc, scale = dist._params(float(y_pred_mu[i]), float(sigma[i]), idx=i)
        out[i] = -stats.johnsonsu.logpdf(float(y_true[i]), a, b, loc=loc, scale=scale)
    return out


def compute_pit(
    y_true: np.ndarray,
    y_pred_mu: np.ndarray,
    sigma: np.ndarray,
    dist: "PredictiveDistribution",
) -> np.ndarray:
    """Probability Integral Transform values ``F_i(y_i)`` per observation.

    Under correct calibration the PIT values are i.i.d. Uniform(0, 1); a
    histogram with consistent height ≈ 1 indicates a well-calibrated forecast.
    U-shaped → too-narrow PIs; bell-shaped → too-wide PIs; skewed → mean bias.
    """
    return np.array([
        dist.cdf(float(y_true[i]), float(y_pred_mu[i]), float(sigma[i]), idx=i)
        for i in range(len(y_true))
    ])


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of observations falling inside ``[lower, upper]``."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def plot_pit_histogram(
    pit: np.ndarray,
    n_bins: int = 20,
    save_path: str | Path = "outputs/figures/pit_histogram.png",
) -> None:
    """Plot the PIT histogram with the Uniform(0,1) reference line.

    Used for forecast calibration diagnostics.  Bars consistently above 1.0
    in any bin signal that the predictive distribution puts too little mass
    on that quantile; bars below 1.0 signal too much mass.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        pit, bins=n_bins, range=(0, 1), density=True,
        color="#6baed6", edgecolor="black", linewidth=0.3,
    )
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2,
               label="Uniform(0, 1) reference")
    ax.set_xlabel("PIT value  F(y)")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram — calibration diagnostic\n"
                 "(flat ⇒ calibrated; U-shape ⇒ too narrow; bell ⇒ too wide)")
    ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PIT histogram to {save_path}")


def compute_probabilistic_scores(
    y_true: np.ndarray,
    y_pred_mu: np.ndarray,
    sigma: np.ndarray,
    dist: "PredictiveDistribution",
    y_anchor: np.ndarray | None = None,
    quantile_levels: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95),
    n_crps_samples: int = 500,
) -> dict[str, float]:
    """Bundle of probabilistic-forecast scores summarising distributional skill.

    Returns a dict with:
        crps_mean, log_score_mean,
        pinball_q05 / q10 / q50 / q90 / q95,
        coverage_50 / 80 / 90 / 95,
        directional_accuracy (only when *y_anchor* provided),
        pit_uniformity_pvalue (Kolmogorov–Smirnov against Uniform).
    """
    out: dict[str, float] = {}
    crps = crps_johnsonsu(y_true, y_pred_mu, sigma, dist, n_samples=n_crps_samples)
    out["crps_mean"] = float(np.mean(crps))
    out["log_score_mean"] = float(np.mean(log_score_johnsonsu(y_true, y_pred_mu, sigma, dist)))
    for q in quantile_levels:
        q_pred = dist.ppf_array(q, y_pred_mu, sigma)
        out[f"pinball_q{int(q * 100):02d}"] = pinball_loss(y_true, q_pred, q)
    for w in (0.50, 0.80, 0.90, 0.95):
        lo = dist.ppf_array((1 - w) / 2, y_pred_mu, sigma)
        hi = dist.ppf_array((1 + w) / 2, y_pred_mu, sigma)
        out[f"coverage_{int(w * 100)}"] = empirical_coverage(y_true, lo, hi)
    if y_anchor is not None:
        out["directional_accuracy"] = directional_accuracy(y_true, y_pred_mu, y_anchor)
    pit = compute_pit(y_true, y_pred_mu, sigma, dist)
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    out["pit_uniformity_pvalue"] = float(ks_p)
    return out


# ─────────────────────────────────────────────
# Prediction Uncertainty Helpers
# ─────────────────────────────────────────────

def get_prediction_sigma(gam, X: np.ndarray, sigma_scale: float = 1.0) -> np.ndarray:
    """Derive per-observation predictive standard deviation from the GAM.

    pyGAM's prediction intervals assume a Normal predictive distribution:
        PI(width) = mean ± z_{width/2} × σ_pred

    We back out σ_pred from the 95% PI:
        σ_pred = (upper_95 - lower_95) / (2 × 1.96)

    This σ incorporates both parameter uncertainty (model uncertainty)
    and residual variance, giving the full predictive spread.

    Args:
        gam: Fitted LinearGAM.
        X: Feature matrix (n_samples × n_features).
        sigma_scale: Multiplicative inflation factor applied to the raw pyGAM
            sigma.  Values > 1 widen all downstream prediction intervals and
            PDFs; 1.0 (default) preserves the original behaviour.

    Returns:
        Array of shape (n_samples,) with per-observation σ.
    """
    pi_95 = gam.prediction_intervals(X, width=0.95)
    sigma = (pi_95[:, 1] - pi_95[:, 0]) / (2 * 1.96)
    return np.clip(sigma, a_min=1e-6, a_max=None) * sigma_scale


def get_all_prediction_intervals(
    gam,
    X: np.ndarray,
    widths: list[float] | None = None,
    dist: "PredictiveDistribution | None" = None,
    sigma_scale: float = 1.0,
) -> dict[float, np.ndarray]:
    """Compute prediction intervals at multiple confidence levels.

    If a :class:`PredictiveDistribution` is supplied the intervals are based
    on the fitted Johnson SU quantiles (potentially asymmetric, with heavier
    tails).  Otherwise falls back to the Gaussian intervals from pyGAM.

    Args:
        gam: Fitted LinearGAM.
        X: Feature matrix.
        widths: List of PI widths (e.g., [0.50, 0.80, 0.90, 0.95]).
        dist: Optional fitted :class:`PredictiveDistribution`.
        sigma_scale: Multiplicative inflation factor passed to
            :func:`get_prediction_sigma`.  Values > 1 widen intervals.

    Returns:
        Dict mapping width → array of shape (n_samples, 2) with [lower, upper].
    """
    if widths is None:
        widths = CI_LEVELS
    if dist is None:
        return {w: gam.prediction_intervals(X, width=w) for w in widths}

    y_pred = gam.predict(X)
    sigma = get_prediction_sigma(gam, X, sigma_scale=sigma_scale)
    return dist.get_all_intervals(y_pred, sigma, widths)


# ─────────────────────────────────────────────
# Static Matplotlib Plots
# ─────────────────────────────────────────────

def plot_fan_chart(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gam,
    X: np.ndarray,
    widths: list[float] | None = None,
    last_n_days: int = 500,
    save_path: str | Path = "outputs/figures/fan_chart.png",
    dist: "PredictiveDistribution | None" = None,
) -> None:
    """Plot a fan chart: actual line + point forecast + stacked CI bands.

    Each confidence band is a progressively lighter shade of blue.
    The innermost (50%) band is the darkest, giving an intuitive sense
    of where the price is most likely to fall.

    When *dist* is provided the bands are computed from the Johnson SU
    distribution (asymmetric, potentially heavier-tailed than Normal).

    Args:
        dates: DatetimeIndex aligned with y_true / y_pred.
        y_true: Actual prices.
        y_pred: GAM point predictions.
        gam: Fitted LinearGAM (for prediction intervals / sigma).
        X: Feature matrix aligned with dates.
        widths: CI levels to draw.
        last_n_days: Limit chart to this many recent trading days.
        save_path: Where to save the figure.
        dist: Optional :class:`PredictiveDistribution`; uses Johnson SU PIs
            when supplied, Gaussian PIs otherwise.
    """
    if widths is None:
        widths = CI_LEVELS

    # Trim to last N observations for readability
    n = min(last_n_days, len(dates))
    dates_n = dates[-n:]
    y_true_n = y_true[-n:]
    y_pred_n = y_pred[-n:]
    X_n = X[-n:]

    intervals = get_all_prediction_intervals(gam, X_n, widths, dist=dist)

    fig, ax = plt.subplots(figsize=(16, 7))

    # Draw bands from widest (lightest) to narrowest (darkest)
    for width, color, alpha in zip(
        reversed(widths),
        reversed(CI_COLORS_MPL[: len(widths)]),
        reversed(CI_ALPHAS_MPL[: len(widths)]),
    ):
        lower = intervals[width][:, 0]
        upper = intervals[width][:, 1]
        ax.fill_between(
            dates_n,
            lower,
            upper,
            color=color,
            alpha=alpha,
            label=f"{int(width * 100)}% PI",
        )

    # Point forecast and actual on top
    ax.plot(dates_n, y_pred_n, color="#08306b", linewidth=1.2,
            label="Point forecast", zorder=4)
    ax.plot(dates_n, y_true_n, color="black", linewidth=0.8,
            linestyle="--", alpha=0.7, label="Actual", zorder=5)

    ax.set_title("Crude Oil Forecast Fan Chart — Predictive Confidence Intervals",
                 fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved fan chart to {save_path}")


def plot_predictive_density(
    mu: float,
    sigma: float,
    y_actual: float | None = None,
    widths: list[float] | None = None,
    save_path: str | Path = "outputs/figures/predictive_density.png",
    title: str = "Predictive Distribution",
    dist: "PredictiveDistribution | None" = None,
) -> None:
    """Plot the predictive probability density curve for a single forecast.

    When *dist* is supplied the curve is a Johnson SU PDF with four
    parameters (ν, τ, loc, scale), capturing skewness and heavier tails.
    Without *dist* a Gaussian PDF is used as a fallback.

    Layout:
        X-axis → crude oil price (USD)
        Y-axis → probability density
        Bands  → 50 / 80 / 90 / 95% prediction intervals, dark → light blue
        Dashed line → point forecast (μ)
        Red marker  → actual price (if provided)

    Args:
        mu: GAM point forecast.
        sigma: Heteroscedastic predictive std deviation (from GAM 95% PI).
        y_actual: Actual realised price (optional, drawn as a red line).
        widths: Quantile bands to fill.
        save_path: Where to save the figure.
        title: Figure title.
        dist: Optional :class:`PredictiveDistribution`; uses Johnson SU when
            supplied, Normal otherwise.
    """
    if widths is None:
        widths = CI_LEVELS

    use_johnsonsu = dist is not None

    # Build x-axis: use Johnson SU 0.1–99.9th percentile range so we always
    # capture the full density regardless of tail heaviness
    if use_johnsonsu:
        x_lo = dist.ppf(0.001, mu, sigma)
        x_hi = dist.ppf(0.999, mu, sigma)
        # Widen slightly for visual breathing room
        margin = (x_hi - x_lo) * 0.1
        x = np.linspace(x_lo - margin, x_hi + margin, 1000)
        pdf = dist.pdf(x, mu, sigma)
        dist_label = f"Johnson SU (ν={dist.nu:.2f}, τ={dist.tau:.2f})"
    else:
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        dist_label = "Gaussian"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Fill quantile bands widest → narrowest (lightest → darkest)
    for width, color, alpha in zip(
        reversed(widths),
        reversed(CI_COLORS_MPL[: len(widths)]),
        reversed(CI_ALPHAS_MPL[: len(widths)]),
    ):
        if use_johnsonsu:
            lower = dist.ppf((1 - width) / 2, mu, sigma)
            upper = dist.ppf((1 + width) / 2, mu, sigma)
        else:
            z = stats.norm.ppf(0.5 + width / 2)
            lower = mu - z * sigma
            upper = mu + z * sigma
        mask = (x >= lower) & (x <= upper)
        ax.fill_between(
            x[mask], pdf[mask],
            color=color, alpha=alpha + 0.1,
            label=f"{int(width * 100)}% PI"
        )

    # Full PDF curve
    ax.plot(x, pdf, color="#08306b", linewidth=2, label=f"Predictive PDF ({dist_label})")

    # Point forecast line
    ax.axvline(mu, color="#08306b", linestyle="--", linewidth=1.2,
               label=f"Forecast: ${mu:.2f}")

    # Actual price (if known)
    if y_actual is not None:
        ax.axvline(y_actual, color="crimson", linestyle="-", linewidth=1.5,
                   label=f"Actual: ${y_actual:.2f}")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Crude Oil Price (USD)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved predictive density to {save_path}")


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path = "outputs/figures/residuals.png",
) -> None:
    """4-panel residual diagnostic plot."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted")

    axes[0, 1].plot(residuals, linewidth=0.5)
    axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 1].set_xlabel("Observation Index")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals Over Time")

    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 0].hist(residuals, bins=50, edgecolor="black", linewidth=0.3, density=True)
    axes[1, 0].plot(
        x_range,
        stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
        "r-", linewidth=1.5,
    )
    axes[1, 0].set_xlabel("Residual Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Residual Distribution")

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual Diagnostics", fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved residual diagnostics to {save_path}")


def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path = "outputs/figures/actual_vs_pred.png",
) -> None:
    """Plot actual vs predicted prices over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, y_true, label="Actual", linewidth=0.8, alpha=0.8)
    ax.plot(dates, y_pred, label="GAM Predicted", linewidth=0.8, alpha=0.8)
    ax.set_title("Actual vs Predicted Crude Oil Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved actual vs predicted plot to {save_path}")


# ─────────────────────────────────────────────
# Advanced Residual Diagnostics
# ─────────────────────────────────────────────

def test_residual_white_noise(
    residuals: np.ndarray,
    lags: int = 40,
) -> pd.DataFrame:
    """Ljung-Box white-noise test on model residuals.

    Tests H₀: the residuals are independently distributed (white noise).
    A significant p-value (< 0.05) indicates unexplained autocorrelation,
    which may signal missing features or model mis-specification.

    Args:
        residuals: Array of (y_true - y_pred) values.
        lags: Number of lags to test.

    Returns:
        DataFrame with columns ['lb_stat', 'lb_pvalue'] indexed by lag.
    """
    lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    # Standardise column names across statsmodels versions
    lb_result.columns = ["lb_stat", "lb_pvalue"]
    lb_result.index.name = "lag"

    n_sig = (lb_result["lb_pvalue"] < 0.05).sum()
    logger.info(
        f"Ljung-Box test ({lags} lags): {n_sig}/{lags} lags significant at p<0.05"
    )
    if n_sig == 0:
        logger.info("  -> Residuals appear consistent with white noise (good).")
    else:
        logger.warning(
            f"  -> {n_sig} significant lags detected. "
            "The model may be leaving autocorrelated structure in the residuals."
        )

    return lb_result


def plot_ljung_box(
    residuals: np.ndarray,
    lags: int = 40,
    save_path: str | Path = "outputs/figures/ljung_box.png",
) -> None:
    """Bar chart of Ljung-Box p-values by lag.

    Bars are green when p ≥ 0.05 (fail to reject white noise) and red when
    p < 0.05 (significant autocorrelation detected).  The dashed horizontal
    line marks the α = 0.05 significance threshold.

    Args:
        residuals: Array of model residuals.
        lags: Number of lags to display.
        save_path: Output file path.
    """
    lb_result = test_residual_white_noise(residuals, lags=lags)
    pvalues = lb_result["lb_pvalue"].values
    lag_labels = lb_result.index.values

    colors = ["#d62728" if p < 0.05 else "#2ca02c" for p in pvalues]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(lag_labels, pvalues, color=colors, edgecolor="none", width=0.7)
    ax.axhline(y=0.05, color="black", linestyle="--", linewidth=1.0,
               label="α = 0.05 threshold")
    ax.set_xlabel("Lag")
    ax.set_ylabel("p-value")
    ax.set_title("Ljung-Box White-Noise Test — Residual p-values by Lag\n"
                 "(Green = white noise, Red = significant autocorrelation)")
    ax.set_ylim(0, max(pvalues.max() * 1.1, 0.12))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Ljung-Box plot to {save_path}")


def plot_residual_acf_pacf(
    residuals: np.ndarray,
    lags: int = 60,
    save_path: str | Path = "outputs/figures/residual_acf_pacf.png",
) -> None:
    """ACF and PACF of model residuals.

    If residuals are true white noise there should be no significant spikes
    (outside the ±1.96/√n confidence bounds) at any lag.

    Args:
        residuals: Array of model residuals.
        lags: Number of lags to show.
        save_path: Output file path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_acf(residuals, lags=lags, ax=axes[0], zero=False,
             title="ACF of Residuals", alpha=0.05)
    plot_pacf(residuals, lags=lags, ax=axes[1], zero=False,
              title="PACF of Residuals", alpha=0.05, method="ywm")

    for ax in axes:
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Lag")

    fig.suptitle(
        "Residual Autocorrelation — white noise ⟹ no spikes outside shaded band",
        fontsize=12,
    )
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved residual ACF/PACF plot to {save_path}")


# ─────────────────────────────────────────────
# GAM-Specific Diagnostics
# ─────────────────────────────────────────────

def print_gam_statistics(gam) -> None:
    """Log and print the key pyGAM model statistics."""
    stats_dict = gam.statistics_
    lines = [
        "",
        "=== GAM Model Statistics ===",
        f"  GCV score       : {stats_dict.get('GCV', float('nan')):.6f}",
        f"  AIC             : {stats_dict.get('AIC', float('nan')):.4f}",
        f"  AICc            : {stats_dict.get('AICc', float('nan')):.4f}",
        f"  Pseudo R^2      : {stats_dict.get('pseudo_r2', {}).get('explained_deviance', float('nan')):.4f}",
        f"  Scale (sigma^2) : {stats_dict.get('scale', float('nan')):.4f}",
        f"  N (observations): {stats_dict.get('n_samples', 'N/A')}",
        "",
    ]
    output = "\n".join(lines)
    # Use safe encoding for Windows cp1252 console
    try:
        print(output)
    except UnicodeEncodeError:
        print(output.encode("ascii", errors="replace").decode("ascii"))
    logger.info(output)


def plot_gam_term_diagnostics(
    gam,
    feature_names: list[str],
    save_path: str | Path = "outputs/figures/gam_term_diagnostics.png",
) -> None:
    """Two-panel bar chart: smoothing λ and effective DoF per GAM term.

    **Lambda (λ)**: The smoothing penalty. A large λ forces a near-linear term;
    a small λ allows more flexibility. Comparing λ values across features shows
    which features the GAM smoothed heavily (low flexibility) vs. loosely.

    **Effective Degrees of Freedom (EDoF)**: The degrees of freedom consumed by
    each spline term. EDoF ≈ 1 means the term is essentially linear; higher
    values indicate more curvature.  The sum of all EDoFs plus the intercept
    equals the total model complexity.

    Args:
        gam: Fitted LinearGAM (must have .statistics_ populated by gridsearch).
        feature_names: Feature column names (length must match number of terms).
        save_path: Output file path.
    """
    stats_dict = gam.statistics_
    edof_per_term = stats_dict.get("edof_per_term", [])
    p_values = stats_dict.get("p_values", [])

    # gam.lam is a list of λ values, one per term
    lam_values = np.array(gam.lam).flatten()

    n_terms = len(feature_names)

    # Guard: align lengths (intercept adds an extra term in some pyGAM versions)
    lam_plot = lam_values[:n_terms] if len(lam_values) >= n_terms else lam_values
    edof_plot = (
        np.array(edof_per_term[:n_terms])
        if len(edof_per_term) >= n_terms
        else np.array(edof_per_term)
    )
    pval_plot = (
        np.array(p_values[:n_terms])
        if len(p_values) >= n_terms
        else np.array(p_values)
    )

    x = np.arange(len(feature_names))
    short_names = [
        n if len(n) <= 22 else f"...{n[-19:]}" for n in feature_names
    ]

    fig, axes = plt.subplots(2, 1, figsize=(max(16, n_terms * 0.35), 10))

    # ── Panel 1: Smoothing Lambda ────────────────────────────────────────────
    ax = axes[0]
    bar_colors_lam = ["#2171b5"] * len(lam_plot)
    bars = ax.bar(x[: len(lam_plot)], np.log10(lam_plot + 1e-10),
                  color=bar_colors_lam, edgecolor="none", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("log₁₀(λ)  [higher = smoother]")
    ax.set_title("Smoothing Penalty λ per GAM Term  (larger = less flexible)")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: Effective Degrees of Freedom ────────────────────────────────
    ax2 = axes[1]
    if len(edof_plot) > 0:
        # Colour by significance (p_values): red if significant, blue otherwise
        if len(pval_plot) == len(edof_plot):
            bar_colors_edof = [
                "#d62728" if p < 0.05 else "#2171b5" for p in pval_plot
            ]
        else:
            bar_colors_edof = ["#2171b5"] * len(edof_plot)

        ax2.bar(x[: len(edof_plot)], edof_plot,
                color=bar_colors_edof, edgecolor="none", width=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=75, ha="right", fontsize=7)
        ax2.set_ylabel("Effective Degrees of Freedom")
        ax2.set_title(
            "Effective DoF per Term  "
            "(EDoF≈1 ⟹ near-linear | Red = significant p < 0.05)"
        )
        ax2.axhline(y=1, color="black", linestyle="--", linewidth=0.8,
                    label="EDoF = 1 (linear)")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "EDoF not available (run gridsearch first)",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved GAM term diagnostics to {save_path}")


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────

def run_evaluation(config: dict) -> pd.DataFrame:
    """Run full evaluation: metrics, diagnostics, fan chart, density plot.

    Honours ``features.target_transform``: when "log_return" the model's
    predictions are in log-return space, but all reported metrics, fan-chart
    bands, and probability forecasts are reconstructed back to **price level**
    (USD/bbl) so the output is comparable across modes and intuitive to read.
    """
    import joblib  # local: only needed for transformer load
    root = get_project_root(config)
    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = config.get("features", {})
    target = feat_cfg["target"]
    target_transform = feat_cfg.get("target_transform", "level")

    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    gam = load_model(root / "outputs" / "models" / "gam_model.pkl")

    # Build the full feature matrix, then restrict to the columns the model
    # was actually trained on (stepwise selection may have dropped features).
    X_full, y_model, all_feature_names, target_dates = build_feature_matrix(
        df, target, target_transform=target_transform,
    )
    saved_names_path = root / "outputs" / "models" / "feature_names.pkl"
    feature_names = (
        load_feature_names(saved_names_path)
        if saved_names_path.exists()
        else all_feature_names
    )
    # Map saved names → column positions in the full matrix
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    try:
        col_idx = [name_to_idx[n] for n in feature_names]
    except KeyError as missing:
        raise ValueError(
            f"Saved feature name {missing} not found in feature matrix. "
            "Re-run the pipeline to regenerate features and model."
        )
    X = X_full[:, col_idx]

    # Apply saved skewed-feature transformer (if model was trained with it)
    transformer_path = root / "outputs" / "models" / "feature_transformer.pkl"
    if transformer_path.exists():
        transformer = joblib.load(transformer_path)
        X = transformer.transform(X)
        logger.info("Applied saved SkewedFeatureTransformer to evaluation features.")

    # Anchor price (P_t) for reconstruction in log-return mode
    anchor = compute_anchor_prices(df, target, forecast_horizon=1)
    y_level = get_target_y_level(df, target, forecast_horizon=1)

    # Model predictions in transform space
    y_pred_model = gam.predict(X)
    if target_transform == "log_return":
        y_pred_level = reconstruct_price_from_returns(y_pred_model, anchor)
    else:
        y_pred_level = y_pred_model

    # Naive baseline always in price-level space
    y_naive_level = naive_baseline_predictions(y_level)
    residuals_level = y_level - y_pred_level
    residuals_model = y_model - y_pred_model

    # ── Metrics comparison (price-scale) ──────────────────────────────────────
    comparison = compare_with_baseline(y_level, y_pred_level, y_naive_level)
    # Add directional accuracy row for both
    dir_gam = directional_accuracy(y_level, y_pred_level, anchor)
    dir_naive = directional_accuracy(y_level, y_naive_level, anchor)
    comparison["directional_accuracy"] = [dir_gam, dir_naive]
    logger.info(f"\nModel Comparison:\n{comparison.to_string()}")
    print("\n=== Model Comparison ===")
    print(comparison.to_string())
    print()

    # ── GAM model-level statistics ────────────────────────────────────────────
    print_gam_statistics(gam)

    # ── Standard plots (price-scale) ──────────────────────────────────────────
    plot_actual_vs_predicted(target_dates, y_level, y_pred_level,
                             fig_dir / "actual_vs_pred.png")
    plot_residual_diagnostics(y_level, y_pred_level, fig_dir / "residuals.png")

    # ── Fit Johnson SU predictive distribution in MODEL SPACE ────────────────
    # (so σ heteroscedasticity is fitted on the actual loss residuals)
    sigma_model = get_prediction_sigma(gam, X)
    dist_model = fit_residual_distribution(y_model, y_pred_model, sigma_model)
    logger.info(f"Predictive distribution (model space): {dist_model.describe()}")

    # ── Probabilistic scores (model space; CRPS is unit-equivariant under
    #     monotonic transform, log-score / coverage are also computed here) ───
    prob_scores = compute_probabilistic_scores(
        y_true=y_model,
        y_pred_mu=y_pred_model,
        sigma=sigma_model,
        dist=dist_model,
        y_anchor=np.zeros_like(y_model) if target_transform == "log_return" else anchor,
    )
    logger.info("Probabilistic scores: " +
                ", ".join(f"{k}={v:.4f}" for k, v in prob_scores.items()))
    print("\n=== Probabilistic Scores ===")
    for k, v in prob_scores.items():
        print(f"  {k:30s} {v:>10.4f}")
    print()

    # PIT histogram for calibration diagnostics
    pit = compute_pit(y_model, y_pred_model, sigma_model, dist_model)
    plot_pit_histogram(pit, save_path=fig_dir / "pit_histogram.png")

    # ── Probability forecast plots ────────────────────────────────────────────
    # Fan chart in price space: convert each quantile band from return space
    # back to price using the row-aligned anchor.
    if target_transform == "log_return":
        # Build per-quantile lower/upper in price space
        widths = CI_LEVELS
        intervals_price = {}
        for w in widths:
            q_lo = (1 - w) / 2
            q_hi = (1 + w) / 2
            r_lo = dist_model.ppf_array(q_lo, y_pred_model, sigma_model)
            r_hi = dist_model.ppf_array(q_hi, y_pred_model, sigma_model)
            intervals_price[w] = np.column_stack([
                anchor * np.exp(r_lo),
                anchor * np.exp(r_hi),
            ])
        _plot_fan_chart_from_intervals(
            target_dates, y_level, y_pred_level, intervals_price,
            save_path=fig_dir / "fan_chart.png",
            title="Crude Oil Forecast Fan Chart — Johnson SU on log-returns "
                  "(reconstructed to price)",
        )
    else:
        plot_fan_chart(target_dates, y_level, y_pred_level, gam, X,
                       save_path=fig_dir / "fan_chart.png",
                       dist=dist_model)

    # Genuine T+1 forecast: predict tomorrow using today's (last available) features
    X_full_0, _, _, _ = build_feature_matrix(
        df, target, forecast_horizon=0, target_transform=target_transform,
    )
    X_latest = X_full_0[-1:, col_idx]
    if transformer_path.exists():
        X_latest = transformer.transform(X_latest)
    mu_tomorrow_model = float(gam.predict(X_latest)[0])
    sigma_tomorrow_model = float(get_prediction_sigma(gam, X_latest)[0])
    last_close = float(df[target].iloc[-1])
    tomorrow_date = df.index[-1] + pd.offsets.BDay(1)
    if target_transform == "log_return":
        mu_tomorrow_price = last_close * np.exp(mu_tomorrow_model)
        # Approximate price-space sigma via the delta method: σ_p ≈ P · σ_r
        sigma_tomorrow_price = last_close * sigma_tomorrow_model
    else:
        mu_tomorrow_price = mu_tomorrow_model
        sigma_tomorrow_price = sigma_tomorrow_model

    plot_predictive_density(
        mu=mu_tomorrow_price,
        sigma=sigma_tomorrow_price,
        y_actual=None,
        save_path=fig_dir / "predictive_density.png",
        title=f"Predictive Distribution — {tomorrow_date.date()}",
        dist=dist_model if target_transform != "log_return" else None,
    )

    # ── Advanced residual diagnostics on model-space residuals ───────────────
    lb_results = test_residual_white_noise(residuals_model, lags=40)
    plot_ljung_box(residuals_model, lags=40, save_path=fig_dir / "ljung_box.png")
    plot_residual_acf_pacf(residuals_model, lags=60,
                           save_path=fig_dir / "residual_acf_pacf.png")

    # ── GAM term-level diagnostics ────────────────────────────────────────────
    plot_gam_term_diagnostics(gam, feature_names,
                              save_path=fig_dir / "gam_term_diagnostics.png")

    return comparison


def _plot_fan_chart_from_intervals(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intervals: dict[float, np.ndarray],
    save_path: str | Path,
    last_n_days: int = 500,
    title: str = "Crude Oil Forecast Fan Chart",
) -> None:
    """Variant of :func:`plot_fan_chart` that takes pre-computed intervals.

    Used when the intervals are constructed in price space from a model that
    was actually fitted on a transformed target (e.g. log-returns reconstructed
    to USD/bbl).
    """
    n = min(last_n_days, len(dates))
    dates_n = dates[-n:]
    y_true_n = y_true[-n:]
    y_pred_n = y_pred[-n:]
    widths = sorted(intervals.keys(), reverse=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    for width, color, alpha in zip(
        widths,
        list(reversed(CI_COLORS_MPL[: len(widths)])),
        list(reversed(CI_ALPHAS_MPL[: len(widths)])),
    ):
        band = intervals[width][-n:]
        ax.fill_between(dates_n, band[:, 0], band[:, 1],
                        color=color, alpha=alpha,
                        label=f"{int(width * 100)}% PI")

    ax.plot(dates_n, y_pred_n, color="#08306b", linewidth=1.2,
            label="Point forecast", zorder=4)
    ax.plot(dates_n, y_true_n, color="black", linewidth=0.8,
            linestyle="--", alpha=0.7, label="Actual", zorder=5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved fan chart to {save_path}")
