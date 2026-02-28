"""Model evaluation: metrics, baselines, and diagnostic + probability plots."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config_loader import get_project_root
from src.model import build_feature_matrix, load_model

logger = logging.getLogger(__name__)

# Confidence levels and their matching blue shades (dark → light)
CI_LEVELS = [0.50, 0.80, 0.90, 0.95]
# ColorBrewer Blues: darkest at centre, lightest at the outside
CI_COLORS_MPL = ["#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"]
CI_ALPHAS_MPL = [0.75, 0.55, 0.40, 0.25]


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
# Prediction Uncertainty Helpers
# ─────────────────────────────────────────────

def get_prediction_sigma(gam, X: np.ndarray) -> np.ndarray:
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

    Returns:
        Array of shape (n_samples,) with per-observation σ.
    """
    pi_95 = gam.prediction_intervals(X, width=0.95)
    sigma = (pi_95[:, 1] - pi_95[:, 0]) / (2 * 1.96)
    return np.clip(sigma, a_min=1e-6, a_max=None)


def get_all_prediction_intervals(
    gam,
    X: np.ndarray,
    widths: list[float] | None = None,
) -> dict[float, np.ndarray]:
    """Compute prediction intervals at multiple confidence levels.

    Args:
        gam: Fitted LinearGAM.
        X: Feature matrix.
        widths: List of PI widths (e.g., [0.50, 0.80, 0.90, 0.95]).

    Returns:
        Dict mapping width → array of shape (n_samples, 2) with [lower, upper].
    """
    if widths is None:
        widths = CI_LEVELS
    return {w: gam.prediction_intervals(X, width=w) for w in widths}


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
) -> None:
    """Plot a fan chart: actual line + point forecast + stacked CI bands.

    Each confidence band is a progressively lighter shade of blue.
    The innermost (50%) band is the darkest, giving an intuitive sense
    of where the price is most likely to fall.

    Args:
        dates: DatetimeIndex aligned with y_true / y_pred.
        y_true: Actual prices.
        y_pred: GAM point predictions.
        gam: Fitted LinearGAM (for prediction intervals).
        X: Feature matrix aligned with dates.
        widths: CI levels to draw.
        last_n_days: Limit chart to this many recent trading days.
        save_path: Where to save the figure.
    """
    if widths is None:
        widths = CI_LEVELS

    # Trim to last N observations for readability
    n = min(last_n_days, len(dates))
    dates_n = dates[-n:]
    y_true_n = y_true[-n:]
    y_pred_n = y_pred[-n:]
    X_n = X[-n:]

    intervals = get_all_prediction_intervals(gam, X_n, widths)

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
) -> None:
    """Plot the predictive probability density curve for a single forecast.

    Draws a Normal PDF centred at the point forecast μ with spread σ.
    Quantile bands are filled in progressively lighter blues so the eye
    is immediately drawn to the most probable price region.

    Layout:
        X-axis → crude oil price (USD)
        Y-axis → probability density
        Bands  → 50 / 80 / 90 / 95% prediction intervals, dark → light blue
        Dashed line → point forecast (μ)
        Red marker  → actual price (if provided)

    Args:
        mu: Point forecast (mean of the predictive distribution).
        sigma: Predictive std deviation (from get_prediction_sigma).
        y_actual: Actual realised price (optional, drawn as a red line).
        widths: Quantile bands to fill.
        save_path: Where to save the figure.
        title: Figure title.
    """
    if widths is None:
        widths = CI_LEVELS

    # Build x-axis spanning ±4σ around the forecast
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Fill quantile bands widest → narrowest (lightest → darkest)
    for width, color, alpha in zip(
        reversed(widths),
        reversed(CI_COLORS_MPL[: len(widths)]),
        reversed(CI_ALPHAS_MPL[: len(widths)]),
    ):
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
    ax.plot(x, pdf, color="#08306b", linewidth=2, label="Predictive PDF")

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
# Pipeline Orchestrator
# ─────────────────────────────────────────────

def run_evaluation(config: dict) -> pd.DataFrame:
    """Run full evaluation: metrics, diagnostics, fan chart, density plot."""
    root = get_project_root(config)
    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = config["features"]["target"]
    gam = load_model(root / "outputs" / "models" / "gam_model.pkl")

    X, y, _ = build_feature_matrix(df, target)

    y_gam = gam.predict(X)
    y_naive = naive_baseline_predictions(y)

    comparison = compare_with_baseline(y, y_gam, y_naive)
    logger.info(f"\nModel Comparison:\n{comparison.to_string()}")
    print("\n=== Model Comparison ===")
    print(comparison.to_string())
    print()

    # Standard plots
    plot_actual_vs_predicted(df.index, y, y_gam, fig_dir / "actual_vs_pred.png")
    plot_residual_diagnostics(y, y_gam, fig_dir / "residuals.png")

    # Fan chart (last 500 trading days)
    plot_fan_chart(df.index, y, y_gam, gam, X,
                   save_path=fig_dir / "fan_chart.png")

    # Predictive density for the most recent observation
    sigma = get_prediction_sigma(gam, X)
    plot_predictive_density(
        mu=float(y_gam[-1]),
        sigma=float(sigma[-1]),
        y_actual=float(y[-1]),
        save_path=fig_dir / "predictive_density.png",
        title="Predictive Distribution — Most Recent Observation",
    )

    return comparison
