"""Model evaluation: metrics, baselines, and diagnostic plots."""

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


def naive_baseline_predictions(y: np.ndarray) -> np.ndarray:
    """Generate naive forecast: predict tomorrow = today.

    Args:
        y: Actual target values.

    Returns:
        Array where pred[i] = y[i-1]. First element is NaN.
    """
    pred = np.empty_like(y)
    pred[0] = np.nan
    pred[1:] = y[:-1]
    return pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict with keys 'mae', 'rmse', 'mape'.
    """
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
    """Compare GAM and naive baseline metrics side by side.

    Args:
        y_true: Actual values.
        y_gam: GAM predictions.
        y_naive: Naive baseline predictions.

    Returns:
        DataFrame with model comparison.
    """
    gam_metrics = compute_metrics(y_true, y_gam)
    naive_metrics = compute_metrics(y_true, y_naive)

    comparison = pd.DataFrame(
        [
            {"Model": "GAM", **gam_metrics},
            {"Model": "Naive (tomorrow=today)", **naive_metrics},
        ]
    )
    comparison = comparison.set_index("Model")
    return comparison


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path = "outputs/figures/residuals.png",
) -> None:
    """Generate 4-panel residual diagnostic plot.

    Panels: residuals vs fitted, residuals over time, histogram, Q-Q plot.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals vs fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted")

    # Residuals over time
    axes[0, 1].plot(residuals, linewidth=0.5)
    axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 1].set_xlabel("Observation Index")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals Over Time")

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, edgecolor="black", linewidth=0.3, density=True)
    # Overlay normal distribution
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 0].plot(
        x_range,
        stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
        "r-",
        linewidth=1.5,
    )
    axes[1, 0].set_xlabel("Residual Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Residual Distribution")

    # Q-Q plot
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


def run_evaluation(config: dict) -> pd.DataFrame:
    """Run full evaluation: load model, compute metrics, generate plots.

    Args:
        config: Project configuration dictionary.

    Returns:
        Comparison DataFrame (GAM vs naive baseline).
    """
    root = get_project_root(config)
    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data and model
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = config["features"]["target"]
    gam = load_model(root / "outputs" / "models" / "gam_model.pkl")

    # Build feature matrix
    X, y, _ = build_feature_matrix(df, target)

    # Generate predictions
    y_gam = gam.predict(X)
    y_naive = naive_baseline_predictions(y)

    # Compare metrics
    comparison = compare_with_baseline(y, y_gam, y_naive)
    logger.info(f"\nModel Comparison:\n{comparison.to_string()}")
    print("\n=== Model Comparison ===")
    print(comparison.to_string())
    print()

    # Generate plots
    plot_actual_vs_predicted(df.index, y, y_gam, fig_dir / "actual_vs_pred.png")
    plot_residual_diagnostics(y, y_gam, fig_dir / "residuals.png")

    return comparison
