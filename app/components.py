"""Reusable Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from pygam import LinearGAM

from src.evaluation import PredictiveDistribution

# ── Shared colour palette ────────────────────────────────────────────────────
# CI bands in shades of blue, innermost (50%) darkest, outermost (95%) lightest
_CI_LEVELS = [0.50, 0.80, 0.90, 0.95]
_CI_FILL_RGBA = [
    "rgba(8,  48,  107, 0.45)",   # 50%  — dark blue
    "rgba(33, 113, 181, 0.30)",   # 80%  — medium blue
    "rgba(107, 174, 214, 0.20)",  # 90%  — light blue
    "rgba(198, 219, 239, 0.15)",  # 95%  — very light blue
]
_CI_LINE_RGBA = ["rgba(0,0,0,0)"] * 4   # invisible borders between bands


def _get_sigma_from_gam(
    gam: LinearGAM,
    X: np.ndarray,
    sigma_scale: float = 1.0,
    sigma_gam: LinearGAM | None = None,
    X_sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Per-observation predictive σ from sigma_gam (preferred) or PI width (fallback).

    When *sigma_gam* and *X_sigma* are provided the sigma sub-model is used::

        σ = exp(sigma_gam.predict(X_sigma)) × sigma_scale

    Otherwise falls back to backing out σ from the GAM 95% PI::

        σ = (upper_95 − lower_95) / (2 × 1.96) × sigma_scale
    """
    if sigma_gam is not None and X_sigma is not None:
        return np.clip(np.exp(sigma_gam.predict(X_sigma)), 1e-6, None) * sigma_scale
    pi_95 = gam.prediction_intervals(X, width=0.95)
    return np.clip((pi_95[:, 1] - pi_95[:, 0]) / (2 * 1.96), 1e-6, None) * sigma_scale


def _get_intervals(
    gam: LinearGAM,
    X: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    widths: list[float],
    dist: PredictiveDistribution | None,
) -> dict[float, np.ndarray]:
    """Compute PI arrays using Johnson SU (if dist) or Gaussian (fallback)."""
    if dist is not None:
        return dist.get_all_intervals(y_pred, sigma, widths)
    return {w: gam.prediction_intervals(X, width=w) for w in widths}


# ── Price overview ───────────────────────────────────────────────────────────

def plot_price_chart(df: pd.DataFrame, price_cols: list[str]) -> go.Figure:
    """Interactive line chart of oil prices with range slider."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, col in enumerate(price_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines", name=col,
                line=dict(width=1, color=colors[i % len(colors)]),
            ))
    fig.update_layout(
        title="Crude Oil Prices",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=500,
    )
    return fig


# ── Forecast fan chart ───────────────────────────────────────────────────────

def plot_fan_chart_plotly(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gam: LinearGAM,
    X: np.ndarray,
    widths: list[float] | None = None,
    last_n_days: int = 500,
    dist: PredictiveDistribution | None = None,
    sigma_scale: float = 1.0,
) -> go.Figure:
    """Interactive fan chart with layered confidence bands in shades of blue.

    When *dist* is provided the CI bands are computed from the Johnson SU
    distribution (asymmetric, potentially heavier-tailed) rather than the
    Gaussian PIs from pyGAM.

    Args:
        dates: DatetimeIndex aligned with y_true / y_pred / X.
        y_true: Actual prices.
        y_pred: GAM point predictions.
        gam: Fitted LinearGAM.
        X: Feature matrix aligned with dates.
        widths: CI levels to draw (default: 50/80/90/95%).
        last_n_days: Show only the most recent N trading days.
        dist: Optional :class:`~src.evaluation.PredictiveDistribution`.
        sigma_scale: Multiplicative inflation factor for pyGAM's raw σ.

    Returns:
        Plotly Figure.
    """
    if widths is None:
        widths = _CI_LEVELS

    n = min(last_n_days, len(dates))
    dates_n = dates[-n:]
    y_true_n = y_true[-n:]
    y_pred_n = y_pred[-n:]
    X_n = X[-n:]

    sigma_n = _get_sigma_from_gam(gam, X_n, sigma_scale=sigma_scale)
    intervals = _get_intervals(gam, X_n, y_pred_n, sigma_n, widths, dist)

    dist_tag = " (Johnson SU)" if dist is not None else " (Gaussian)"
    fig = go.Figure()

    # Draw bands widest → narrowest so narrower bands sit on top
    for width, fill, line_col in zip(
        reversed(widths),
        reversed(_CI_FILL_RGBA[: len(widths)]),
        reversed(_CI_LINE_RGBA[: len(widths)]),
    ):
        lower = intervals[width][:, 0]
        upper = intervals[width][:, 1]
        label = f"{int(width * 100)}% PI{dist_tag}"

        fig.add_trace(go.Scatter(
            x=dates_n, y=upper,
            mode="lines",
            line=dict(width=0, color=line_col),
            name=label,
            legendgroup=label,
            showlegend=True,
            hovertemplate=f"{label} upper: $%{{y:.2f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=dates_n, y=lower,
            mode="lines",
            line=dict(width=0, color=line_col),
            fillcolor=fill,
            fill="tonexty",
            name=label,
            legendgroup=label,
            showlegend=False,
            hovertemplate=f"{label} lower: $%{{y:.2f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=dates_n, y=y_pred_n,
        mode="lines", name="Point forecast",
        line=dict(color="#08306b", width=1.5),
        hovertemplate="Forecast: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates_n, y=y_true_n,
        mode="lines", name="Actual",
        line=dict(color="black", width=1, dash="dot"),
        hovertemplate="Actual: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Crude Oil Forecast — Fan Chart{dist_tag}",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=550,
    )
    return fig


# ── Fan chart in price space (works for log-return targets) ─────────────────

def plot_fan_chart_price_space_plotly(
    dates: pd.DatetimeIndex,
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    y_pred_model: np.ndarray,
    sigma_model: np.ndarray,
    anchor: np.ndarray,
    dist: PredictiveDistribution,
    last_n_days: int = 500,
    widths: list[float] | None = None,
    target_transform: str = "log_return",
    sigma_scale: float = 1.0,
) -> go.Figure:
    """Fan chart whose bands are constructed in MODEL space then mapped to price.

    Required when the GAM was trained on log-returns: the Johnson SU bands live
    in return space, so each band's lower/upper is exp-transformed against the
    aligned anchor price ``P_t``.  When ``target_transform == "level"`` this
    reduces to the standard Johnson SU intervals.

    Args:
        dates: Forecast dates (one per row).
        y_true_price: Realised price at each date.
        y_pred_price: Point forecast in price space.
        y_pred_model: Raw GAM output (before transform).
        sigma_model: Per-row predictive σ in model space.
        anchor: Anchor price ``P_t`` aligned with each forecast.
        dist: Fitted Johnson SU on model-space residuals.
        last_n_days: Trim the chart to the most recent N rows.
        widths: PI levels.
        target_transform: ``"log_return"`` or ``"level"``.
        sigma_scale: Multiplicative inflation factor on σ.
    """
    if widths is None:
        widths = _CI_LEVELS

    n = min(last_n_days, len(dates))
    dates_n = dates[-n:]
    y_true_n = y_true_price[-n:]
    y_pred_n = y_pred_price[-n:]
    y_pred_model_n = y_pred_model[-n:]
    sigma_n = sigma_model[-n:] * sigma_scale
    anchor_n = anchor[-n:]

    fig = go.Figure()
    space_tag = " (Johnson SU on log-returns → price)" if target_transform == "log_return" else " (Johnson SU)"

    for width, fill in zip(reversed(widths), reversed(_CI_FILL_RGBA[: len(widths)])):
        q_lo, q_hi = (1 - width) / 2, (1 + width) / 2
        r_lo = dist.ppf_array(q_lo, y_pred_model_n, sigma_n)
        r_hi = dist.ppf_array(q_hi, y_pred_model_n, sigma_n)
        if target_transform == "log_return":
            lower = anchor_n * np.exp(r_lo)
            upper = anchor_n * np.exp(r_hi)
        else:
            lower, upper = r_lo, r_hi
        label = f"{int(width * 100)}% PI{space_tag}"

        fig.add_trace(go.Scatter(
            x=dates_n, y=upper, mode="lines",
            line=dict(width=0),
            name=label, legendgroup=label, showlegend=True,
            hovertemplate=f"{label} upper: $%{{y:.2f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=dates_n, y=lower, mode="lines",
            line=dict(width=0), fillcolor=fill, fill="tonexty",
            name=label, legendgroup=label, showlegend=False,
            hovertemplate=f"{label} lower: $%{{y:.2f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=dates_n, y=y_pred_n, mode="lines", name="Point forecast",
        line=dict(color="#08306b", width=1.5),
        hovertemplate="Forecast: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates_n, y=y_true_n, mode="lines", name="Actual",
        line=dict(color="black", width=1, dash="dot"),
        hovertemplate="Actual: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Crude Oil Forecast Fan Chart{space_tag}",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=550,
    )
    return fig


def plot_predictive_density_price_space_plotly(
    mu_model: float,
    sigma_model: float,
    anchor: float,
    dist: PredictiveDistribution,
    y_actual_price: float | None = None,
    title: str = "Predictive Distribution (price space)",
    target_transform: str = "log_return",
    widths: list[float] | None = None,
    obs_idx: int | None = None,
) -> go.Figure:
    """Single-date predictive PDF in PRICE space.

    For log-return targets, builds a fine x-grid in **return** space, computes
    Johnson SU densities there, then change-of-variables transforms to price
    via ``p = anchor · exp(r)`` and Jacobian ``1 / p``.  This preserves the
    skew + heavy tails of the Johnson SU.

    For level targets, falls back to the standard Johnson SU PDF in price space.
    """
    if widths is None:
        widths = _CI_LEVELS

    if target_transform == "log_return":
        r_lo = float(dist.ppf(0.001, mu_model, sigma_model, idx=obs_idx))
        r_hi = float(dist.ppf(0.999, mu_model, sigma_model, idx=obs_idx))
        margin = (r_hi - r_lo) * 0.1
        r_grid = np.linspace(r_lo - margin, r_hi + margin, 1000)
        pdf_r = dist.pdf(r_grid, mu_model, sigma_model, idx=obs_idx)
        # Change of variables: density on price = density on return / (anchor·exp(r))
        x_grid = anchor * np.exp(r_grid)
        pdf_x = pdf_r / np.clip(x_grid, 1e-6, None)
    else:
        x_lo = float(dist.ppf(0.001, mu_model, sigma_model, idx=obs_idx))
        x_hi = float(dist.ppf(0.999, mu_model, sigma_model, idx=obs_idx))
        margin = (x_hi - x_lo) * 0.1
        x_grid = np.linspace(x_lo - margin, x_hi + margin, 1000)
        pdf_x = dist.pdf(x_grid, mu_model, sigma_model, idx=obs_idx)

    fig = go.Figure()

    for width, fill in zip(reversed(widths), reversed(_CI_FILL_RGBA[: len(widths)])):
        q_lo, q_hi = (1 - width) / 2, (1 + width) / 2
        if target_transform == "log_return":
            lo = anchor * np.exp(float(dist.ppf(q_lo, mu_model, sigma_model, idx=obs_idx)))
            hi = anchor * np.exp(float(dist.ppf(q_hi, mu_model, sigma_model, idx=obs_idx)))
        else:
            lo = float(dist.ppf(q_lo, mu_model, sigma_model, idx=obs_idx))
            hi = float(dist.ppf(q_hi, mu_model, sigma_model, idx=obs_idx))
        mask = (x_grid >= lo) & (x_grid <= hi)
        x_band = np.concatenate([[lo], x_grid[mask], [hi]])
        # Interpolate band edges onto pdf for cleaner fills
        y_lo_pdf = float(np.interp(lo, x_grid, pdf_x))
        y_hi_pdf = float(np.interp(hi, x_grid, pdf_x))
        y_band = np.concatenate([[y_lo_pdf], pdf_x[mask], [y_hi_pdf]])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band, mode="lines", fill="tozeroy",
            fillcolor=fill, line=dict(width=0),
            name=f"{int(width * 100)}% PI  [${lo:.1f} – ${hi:.1f}]",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=x_grid, y=pdf_x, mode="lines",
        line=dict(color="#08306b", width=2.5),
        name="Predictive PDF",
        hovertemplate="Price: $%{x:.2f}<br>Density: %{y:.5f}<extra></extra>",
    ))

    if target_transform == "log_return":
        mu_price = anchor * np.exp(mu_model)
    else:
        mu_price = mu_model
    fig.add_vline(
        x=mu_price, line=dict(color="#08306b", dash="dash", width=1.5),
        annotation_text=f"Forecast<br>${mu_price:.2f}", annotation_position="top right",
        annotation_font_color="#08306b",
    )
    if y_actual_price is not None:
        fig.add_vline(
            x=y_actual_price, line=dict(color="crimson", dash="solid", width=1.5),
            annotation_text=f"Actual<br>${y_actual_price:.2f}", annotation_position="top left",
            annotation_font_color="crimson",
        )

    fig.update_layout(
        title=title, xaxis_title="Crude Oil Price (USD)", yaxis_title="Probability Density",
        template="plotly_white", height=450, showlegend=True,
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    return fig


def plot_backtest_path(
    path: pd.DataFrame,
    last_n_days: int = 500,
    show_pi_widths: tuple[int, ...] = (95,),
) -> go.Figure:
    """Plot the walk-forward backtest path with selected PI bands."""
    df = path.sort_index().tail(last_n_days)
    fig = go.Figure()
    for w in show_pi_widths:
        lo_col, hi_col = f"lower_{w}", f"upper_{w}"
        if lo_col in df.columns and hi_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[hi_col], mode="lines", line=dict(width=0),
                name=f"{w}% PI", legendgroup=f"PI{w}", showlegend=True,
                hovertemplate=f"{w}% PI upper: $%{{y:.2f}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df[lo_col], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(107, 174, 214, 0.20)",
                name=f"{w}% PI", legendgroup=f"PI{w}", showlegend=False,
                hovertemplate=f"{w}% PI lower: $%{{y:.2f}}<extra></extra>",
            ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["y_pred"], mode="lines",
        line=dict(color="#ff7f0e", width=1.4), name="GAM forecast (walk-forward)",
        hovertemplate="GAM: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["y_anchor"], mode="lines",
        line=dict(color="#d62728", width=0.9, dash="dash"), name="Naive (today)",
        hovertemplate="Naive: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["y_true"], mode="lines",
        line=dict(color="black", width=1.0), name="Actual",
        hovertemplate="Actual: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Walk-Forward Backtest Path (last {last_n_days} days)",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=520,
    )
    return fig


def plot_rolling_mae_plotly(
    path: pd.DataFrame,
    window: int = 63,
) -> go.Figure:
    """Plot rolling MAE for GAM vs naive on the walk-forward path."""
    df = path.sort_index()
    abs_err_gam = (df["y_true"] - df["y_pred"]).abs()
    abs_err_naive = (df["y_true"] - df["y_anchor"]).abs()
    roll_gam = abs_err_gam.rolling(window, min_periods=window).mean()
    roll_naive = abs_err_naive.rolling(window, min_periods=window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=roll_gam, mode="lines",
        name=f"GAM rolling {window}d MAE", line=dict(color="#08306b", width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=roll_naive, mode="lines",
        name=f"Naive rolling {window}d MAE", line=dict(color="#d62728", width=1.0, dash="dash"),
    ))
    if roll_gam.notna().any():
        thresh = float(np.nanpercentile(roll_gam, 95))
        fig.add_hline(
            y=thresh, line=dict(color="orange", dash="dash", width=1),
            annotation_text=f"95th-pct = {thresh:.2f}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=f"Rolling {window}-Day MAE — Concept-Drift Monitor",
        xaxis_title="Date", yaxis_title="MAE (USD)",
        template="plotly_white", hovermode="x unified", height=400,
    )
    return fig


# ── Predictive density ───────────────────────────────────────────────────────

def plot_predictive_density_plotly(
    mu: float,
    sigma: float,
    y_actual: float | None = None,
    widths: list[float] | None = None,
    title: str = "Predictive Distribution",
    dist: PredictiveDistribution | None = None,
) -> go.Figure:
    """Probability density curve for a single forecast.

    When *dist* is provided a Johnson SU PDF (4 parameters: ν, τ, loc, scale)
    is drawn, capturing skewness and heavy tails.  The CI bands and forecast
    percentile are computed from the same distribution.  Without *dist* a
    Gaussian PDF is used as fallback.

    Args:
        mu: GAM point forecast.
        sigma: Per-observation predictive std deviation.
        y_actual: Actual realised price (optional red marker).
        widths: Quantile widths to shade.
        title: Chart title.
        dist: Optional :class:`~src.evaluation.PredictiveDistribution`.

    Returns:
        Plotly Figure.
    """
    if widths is None:
        widths = _CI_LEVELS

    use_johnsonsu = dist is not None

    # ── Build x-grid ──────────────────────────────────────────────────────────
    if use_johnsonsu:
        x_lo = dist.ppf(0.001, mu, sigma)
        x_hi = dist.ppf(0.999, mu, sigma)
        margin = (x_hi - x_lo) * 0.1
        x = np.linspace(x_lo - margin, x_hi + margin, 800)
        pdf = dist.pdf(x, mu, sigma)
        nu_val = float(np.mean(dist.nu)) if isinstance(dist.nu, np.ndarray) else float(dist.nu)
        tau_val = float(np.mean(dist.tau)) if isinstance(dist.tau, np.ndarray) else float(dist.tau)
        pdf_label = f"Johnson SU PDF (ν={nu_val:.2f}, τ={tau_val:.2f})"
    else:
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 800)
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        pdf_label = "Gaussian PDF"

    fig = go.Figure()

    # ── CI bands widest → narrowest ───────────────────────────────────────────
    for width, fill in zip(reversed(widths), reversed(_CI_FILL_RGBA[: len(widths)])):
        if use_johnsonsu:
            lo = dist.ppf((1 - width) / 2, mu, sigma)
            hi = dist.ppf((1 + width) / 2, mu, sigma)
        else:
            z = stats.norm.ppf(0.5 + width / 2)
            lo, hi = mu - z * sigma, mu + z * sigma

        mask = (x >= lo) & (x <= hi)
        label = f"{int(width * 100)}% PI  [${lo:.1f} – ${hi:.1f}]"

        x_band = np.concatenate([[lo], x[mask], [hi]])
        y_band = np.concatenate([[0], pdf[mask], [0]])

        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            mode="lines",
            fill="tozeroy",
            fillcolor=fill,
            line=dict(width=0),
            name=label,
            hoverinfo="skip",
        ))

    # ── Full PDF curve ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=pdf,
        mode="lines",
        line=dict(color="#08306b", width=2.5),
        name=pdf_label,
        hovertemplate="Price: $%{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
    ))

    # ── Point forecast marker ──────────────────────────────────────────────────
    fig.add_vline(
        x=mu,
        line=dict(color="#08306b", dash="dash", width=1.5),
        annotation_text=f"Forecast<br>${mu:.2f}",
        annotation_position="top right",
        annotation_font_color="#08306b",
    )

    # ── Actual price marker ────────────────────────────────────────────────────
    if y_actual is not None:
        fig.add_vline(
            x=y_actual,
            line=dict(color="crimson", dash="solid", width=1.5),
            annotation_text=f"Actual<br>${y_actual:.2f}",
            annotation_position="top left",
            annotation_font_color="crimson",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Crude Oil Price (USD)",
        yaxis_title="Probability Density",
        template="plotly_white",
        height=450,
        showlegend=True,
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    return fig


# ── Partial dependence ───────────────────────────────────────────────────────

def plot_partial_dependence(
    gam: LinearGAM,
    X: np.ndarray,
    feature_idx: int,
    feature_name: str,
) -> go.Figure:
    """GAM partial dependence plot with 95% confidence band."""
    XX = gam.generate_X_grid(term=feature_idx)
    pdep, confi = gam.partial_dependence(term=feature_idx, X=XX, width=0.95)
    x_vals = XX[:, feature_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_vals, x_vals[::-1]]),
        y=np.concatenate([confi[:, 0], confi[:, 1][::-1]]),
        fill="toself", fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=pdep, mode="lines",
        name="Partial Dependence",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(
        title=f"Partial Dependence: {feature_name}",
        xaxis_title=feature_name, yaxis_title="Effect on Price (USD)",
        template="plotly_white", hovermode="x unified", height=450,
    )
    return fig


# ── Forecast vs actual ───────────────────────────────────────────────────────

def plot_forecast_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> go.Figure:
    """Actual vs predicted prices with optional confidence band."""
    fig = go.Figure()
    if y_lower is not None and y_upper is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([y_lower, y_upper[::-1]]),
            fill="toself", fillcolor="rgba(31, 119, 180, 0.1)",
            line=dict(color="rgba(255,255,255,0)"), name="95% CI",
        ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_true, mode="lines", name="Actual",
        line=dict(color="#1f77b4", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred, mode="lines", name="Predicted",
        line=dict(color="#ff7f0e", width=1),
    ))
    fig.update_layout(
        title="Forecast vs Actual Crude Oil Price",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=500,
    )
    return fig


# ── What-if analysis ─────────────────────────────────────────────────────────

def plot_what_if_analysis(
    gam: LinearGAM,
    base_X: np.ndarray,
    vary_feature_idx: int,
    vary_range: np.ndarray,
    feature_name: str,
    dist: PredictiveDistribution | None = None,
    sigma_scale: float = 1.0,
) -> go.Figure:
    """Predicted price as one feature sweeps across a range, with CI band.

    If *dist* is provided the 95% CI uses Johnson SU quantiles.
    """
    predictions, ci_lower, ci_upper = [], [], []
    sigma_base = _get_sigma_from_gam(gam, base_X, sigma_scale=sigma_scale)

    for val in vary_range:
        X_mod = base_X.copy()
        X_mod[0, vary_feature_idx] = val
        mu_val = gam.predict(X_mod)[0]
        predictions.append(mu_val)

        if dist is not None:
            sig_val = float(_get_sigma_from_gam(gam, X_mod, sigma_scale=sigma_scale)[0])
            ci_lower.append(dist.ppf(0.025, mu_val, sig_val))
            ci_upper.append(dist.ppf(0.975, mu_val, sig_val))
        else:
            ci = gam.prediction_intervals(X_mod, width=0.95)
            ci_lower.append(ci[0, 0])
            ci_upper.append(ci[0, 1])

    predictions = np.array(predictions)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    ci_label = "95% CI (Johnson SU)" if dist is not None else "95% CI (Gaussian)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([vary_range, vary_range[::-1]]),
        y=np.concatenate([ci_lower, ci_upper[::-1]]),
        fill="toself", fillcolor="rgba(255, 127, 14, 0.15)",
        line=dict(color="rgba(255,255,255,0)"), name=ci_label,
    ))
    fig.add_trace(go.Scatter(
        x=vary_range, y=predictions, mode="lines",
        name="Predicted Price", line=dict(color="#ff7f0e", width=2),
    ))
    current_val = base_X[0, vary_feature_idx]
    current_pred = gam.predict(base_X)[0]
    fig.add_trace(go.Scatter(
        x=[current_val], y=[current_pred], mode="markers",
        name="Current",
        marker=dict(size=10, color="red", symbol="diamond"),
    ))
    fig.update_layout(
        title=f"What-If Analysis: {feature_name}",
        xaxis_title=feature_name, yaxis_title="Predicted Price (USD)",
        template="plotly_white", hovermode="x unified", height=450,
    )
    return fig
