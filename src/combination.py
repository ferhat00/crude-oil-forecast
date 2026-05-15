"""Forecast combination: Bates-Granger optimal weight against the naive baseline.

The naive forecast (``ŷ = y_t``) is famously hard to beat for daily oil prices
because the price series is near-random-walk.  But combinations of imperfect
forecasts often outperform either alone — the classical result of Bates &
Granger (1969).  This module finds the convex weight ``w ∈ [0, 1]`` that
minimises walk-forward squared loss for the combination

    ``ŷ_combo = w · ŷ_GAM + (1 − w) · y_t``

and exposes a small API for applying that weight to live forecasts.

Inputs come from the walk-forward path produced by :mod:`src.backtest`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import get_project_root

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CombinationWeights:
    """Optimal combination weights and the diagnostic loss surface.

    Attributes:
        w_gam: Convex weight on the GAM forecast (0 = use only naive,
            1 = use only GAM).  ``1 - w_gam`` is the naive weight.
        rmse_combo: RMSE of the optimal combined forecast.
        rmse_gam: RMSE of the GAM-alone forecast.
        rmse_naive: RMSE of the naive-alone forecast.
        loss_grid: DataFrame indexed by candidate ``w_gam`` values with the
            corresponding combination RMSE — useful for plotting.
    """
    w_gam: float
    rmse_combo: float
    rmse_gam: float
    rmse_naive: float
    loss_grid: pd.DataFrame


def fit_bates_granger_weight(
    y_true: np.ndarray,
    y_gam: np.ndarray,
    y_naive: np.ndarray,
    grid_steps: int = 101,
) -> CombinationWeights:
    """Find the convex ``w_gam`` that minimises combination RMSE.

    The classical closed-form Bates-Granger weight uses the inverse-variance
    rule and assumes uncorrelated errors.  For real forecasts the GAM and
    naive errors are highly correlated, so we instead grid-search the convex
    combination — robust and only marginally slower.

    Args:
        y_true: Realised target values.
        y_gam: GAM point forecasts (aligned with y_true).
        y_naive: Naive point forecasts (typically the prior period's actual).
        grid_steps: Number of candidate weights in [0, 1].

    Returns:
        :class:`CombinationWeights`.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_gam = np.asarray(y_gam, dtype=np.float64)
    y_naive = np.asarray(y_naive, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_gam) | np.isnan(y_naive))
    yt, yg, yn = y_true[mask], y_gam[mask], y_naive[mask]

    weights = np.linspace(0.0, 1.0, grid_steps)
    rmses = np.empty_like(weights)
    for i, w in enumerate(weights):
        combo = w * yg + (1 - w) * yn
        rmses[i] = float(np.sqrt(np.mean((yt - combo) ** 2)))

    best_idx = int(np.argmin(rmses))
    w_star = float(weights[best_idx])

    rmse_gam = float(np.sqrt(np.mean((yt - yg) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((yt - yn) ** 2)))

    grid = pd.DataFrame({"w_gam": weights, "rmse": rmses}).set_index("w_gam")
    logger.info(
        f"Bates-Granger combination: w*={w_star:.3f} → RMSE={rmses[best_idx]:.4f} "
        f"(GAM-alone={rmse_gam:.4f}, naive-alone={rmse_naive:.4f})"
    )
    return CombinationWeights(
        w_gam=w_star,
        rmse_combo=float(rmses[best_idx]),
        rmse_gam=rmse_gam,
        rmse_naive=rmse_naive,
        loss_grid=grid,
    )


def combine_forecasts(
    y_gam: np.ndarray,
    y_naive: np.ndarray,
    w_gam: float,
) -> np.ndarray:
    """Apply the convex combination ``w · y_gam + (1 − w) · y_naive``."""
    return float(w_gam) * np.asarray(y_gam, dtype=np.float64) + (
        1.0 - float(w_gam)
    ) * np.asarray(y_naive, dtype=np.float64)


def fit_from_backtest(
    backtest_path: pd.DataFrame,
) -> CombinationWeights:
    """Convenience wrapper: read y_true / y_pred / y_anchor from a backtest path."""
    return fit_bates_granger_weight(
        y_true=backtest_path["y_true"].values,
        y_gam=backtest_path["y_pred"].values,
        y_naive=backtest_path["y_anchor"].values,
    )


def save_weights(
    weights: CombinationWeights,
    config: dict,
    save_path: str | Path | None = None,
) -> Path:
    """Persist the optimal weight + diagnostics so live code can reuse them."""
    root = get_project_root(config)
    if save_path is None:
        save_path = root / "outputs" / "combination_weights.json"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w_gam": weights.w_gam,
        "w_naive": 1.0 - weights.w_gam,
        "rmse_combo": weights.rmse_combo,
        "rmse_gam": weights.rmse_gam,
        "rmse_naive": weights.rmse_naive,
        "skill_vs_gam": 1.0 - weights.rmse_combo / weights.rmse_gam if weights.rmse_gam else None,
        "skill_vs_naive": 1.0 - weights.rmse_combo / weights.rmse_naive if weights.rmse_naive else None,
    }
    save_path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info(f"Saved Bates-Granger weights to {save_path}")
    return save_path


def load_weights(
    config: dict,
    save_path: str | Path | None = None,
) -> dict | None:
    """Load saved combination weights; returns ``None`` if not yet computed."""
    root = get_project_root(config)
    if save_path is None:
        save_path = root / "outputs" / "combination_weights.json"
    save_path = Path(save_path)
    if not save_path.exists():
        return None
    return json.loads(save_path.read_text())
