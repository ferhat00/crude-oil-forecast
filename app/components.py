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


def _get_sigma_from_gam(gam: LinearGAM, X: np.ndarray, sigma_scale: float = 1.0) -> np.ndarray:
    """Back out per-observation predictive σ from the 95% PI."""
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
        pdf_label = f"Johnson SU PDF (ν={dist.nu:.2f}, τ={dist.tau:.2f})"
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
