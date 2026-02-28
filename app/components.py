"""Reusable Plotly chart builders for the Streamlit dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pygam import LinearGAM


def plot_price_chart(
    df: pd.DataFrame,
    price_cols: list[str],
) -> go.Figure:
    """Create an interactive Plotly line chart of oil prices with range slider.

    Args:
        df: DataFrame with DatetimeIndex and price columns.
        price_cols: List of price column names to plot.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, col in enumerate(price_cols):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1, color=colors[i % len(colors)]),
                )
            )

    fig.update_layout(
        title="Crude Oil Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
        ),
        height=500,
    )

    return fig


def plot_partial_dependence(
    gam: LinearGAM,
    X: np.ndarray,
    feature_idx: int,
    feature_name: str,
) -> go.Figure:
    """Generate partial dependence plot for a single feature.

    Shows the smooth function learned by the GAM with 95% confidence band.

    Args:
        gam: Fitted LinearGAM model.
        X: Feature matrix (used for generating grid).
        feature_idx: Index of the feature in X.
        feature_name: Display name for the feature.

    Returns:
        Plotly Figure object.
    """
    XX = gam.generate_X_grid(term=feature_idx)
    pdep, confi = gam.partial_dependence(term=feature_idx, X=XX, width=0.95)

    x_vals = XX[:, feature_idx]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([confi[:, 0], confi[:, 1][::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
        )
    )

    # Partial dependence line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pdep,
            mode="lines",
            name="Partial Dependence",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig.update_layout(
        title=f"Partial Dependence: {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Effect on Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
    )

    return fig


def plot_forecast_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> go.Figure:
    """Plot actual vs predicted prices with optional confidence band.

    Args:
        dates: DatetimeIndex for x-axis.
        y_true: Actual values.
        y_pred: Predicted values.
        y_lower: Lower confidence bound (optional).
        y_upper: Upper confidence bound (optional).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Confidence band
    if y_lower is not None and y_upper is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([y_lower, y_upper[::-1]]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="#ff7f0e", width=1),
        )
    )

    fig.update_layout(
        title="Forecast vs Actual Crude Oil Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=500,
    )

    return fig


def plot_what_if_analysis(
    gam: LinearGAM,
    base_X: np.ndarray,
    vary_feature_idx: int,
    vary_range: np.ndarray,
    feature_name: str,
) -> go.Figure:
    """Show predicted price as a single feature varies.

    Takes the latest observation as a baseline, sweeps one feature
    across vary_range, and plots the resulting predictions.

    Args:
        gam: Fitted LinearGAM model.
        base_X: Single row of features (baseline, e.g., latest observation).
        vary_feature_idx: Index of the feature to vary.
        vary_range: Array of values to sweep across.
        feature_name: Display name for the feature.

    Returns:
        Plotly Figure object.
    """
    predictions = []
    ci_lower = []
    ci_upper = []

    for val in vary_range:
        X_modified = base_X.copy()
        X_modified[0, vary_feature_idx] = val
        pred = gam.predict(X_modified)[0]
        ci = gam.prediction_intervals(X_modified, width=0.95)
        predictions.append(pred)
        ci_lower.append(ci[0, 0])
        ci_upper.append(ci[0, 1])

    predictions = np.array(predictions)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    fig = go.Figure()

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([vary_range, vary_range[::-1]]),
            y=np.concatenate([ci_lower, ci_upper[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    )

    # Predicted line
    fig.add_trace(
        go.Scatter(
            x=vary_range,
            y=predictions,
            mode="lines",
            name="Predicted Price",
            line=dict(color="#ff7f0e", width=2),
        )
    )

    # Mark current value
    current_val = base_X[0, vary_feature_idx]
    current_pred = gam.predict(base_X)[0]
    fig.add_trace(
        go.Scatter(
            x=[current_val],
            y=[current_pred],
            mode="markers",
            name="Current",
            marker=dict(size=10, color="red", symbol="diamond"),
        )
    )

    fig.update_layout(
        title=f"What-If Analysis: {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Predicted Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
    )

    return fig
