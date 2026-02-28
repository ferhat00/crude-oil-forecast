"""Streamlit dashboard for crude oil price forecasting.

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components import (
    plot_forecast_vs_actual,
    plot_partial_dependence,
    plot_price_chart,
    plot_what_if_analysis,
)
from src.config_loader import load_config
from src.evaluation import compute_metrics, naive_baseline_predictions
from src.model import build_feature_matrix, load_feature_names, load_model


@st.cache_data
def load_data(root: str) -> pd.DataFrame:
    """Load the processed feature dataset."""
    return pd.read_parquet(Path(root) / "data" / "processed" / "features.parquet")


@st.cache_resource
def load_gam_model(root: str):
    """Load the fitted GAM model."""
    return load_model(Path(root) / "outputs" / "models" / "gam_model.pkl")


@st.cache_data
def load_features(root: str) -> list[str]:
    """Load saved feature names."""
    return load_feature_names(Path(root) / "outputs" / "models" / "feature_names.pkl")


def main():
    st.set_page_config(
        page_title="Crude Oil Forecast",
        page_icon="",
        layout="wide",
    )

    st.title("Crude Oil Price Forecasting with GAMs")

    # Load config and data
    config = load_config()
    root = config["_project_root"]

    try:
        df = load_data(root)
        gam = load_gam_model(root)
        feature_names = load_features(root)
    except FileNotFoundError as e:
        st.error(
            f"Required files not found: {e}\n\n"
            "Please run the pipeline first:\n"
            "```\npython scripts/run_pipeline.py\n```"
        )
        return

    target = config["features"]["target"]
    X, y, _ = build_feature_matrix(df, target)

    # Sidebar
    st.sidebar.header("Settings")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df.index.min().date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
    )

    # Filter data by date range
    if len(date_range) == 2:
        mask = (df.index >= pd.Timestamp(date_range[0])) & (
            df.index <= pd.Timestamp(date_range[1])
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price Overview", "Model Performance", "Partial Dependence", "What-If Analysis"]
    )

    # Tab 1: Price Overview
    with tab1:
        st.subheader("Historical Crude Oil Prices")

        oil_tickers = config["data"]["oil_tickers"]
        price_cols = [f"{t}_close" for t in oil_tickers if f"{t}_close" in df.columns]
        fig = plot_price_chart(df_filtered, price_cols)
        st.plotly_chart(fig, use_container_width=True)

        # Key statistics
        if target in df_filtered.columns:
            col1, col2, col3, col4 = st.columns(4)
            latest = df_filtered[target].iloc[-1]
            change_30d = (
                latest - df_filtered[target].iloc[-min(30, len(df_filtered))]
            )
            pct_30d = (change_30d / df_filtered[target].iloc[-min(30, len(df_filtered))]) * 100
            vol_30 = df_filtered[target].iloc[-30:].std() if len(df_filtered) >= 30 else 0

            col1.metric("Latest Price", f"${latest:.2f}")
            col2.metric("30-Day Change", f"${change_30d:.2f}", f"{pct_30d:.1f}%")
            col3.metric("30-Day Volatility", f"${vol_30:.2f}")
            col4.metric("Data Points", f"{len(df_filtered):,}")

    # Tab 2: Model Performance
    with tab2:
        st.subheader("Model Performance")

        y_pred = gam.predict(X)
        y_naive = naive_baseline_predictions(y)

        # Prediction intervals
        ci = gam.prediction_intervals(X, width=0.95)

        fig = plot_forecast_vs_actual(
            df.index, y, y_pred, ci[:, 0], ci[:, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics comparison
        st.subheader("Metrics Comparison")
        col1, col2 = st.columns(2)

        gam_metrics = compute_metrics(y, y_pred)
        naive_metrics = compute_metrics(y[1:], y_naive[1:])

        with col1:
            st.markdown("**GAM Model**")
            st.metric("MAE", f"${gam_metrics['mae']:.2f}")
            st.metric("RMSE", f"${gam_metrics['rmse']:.2f}")
            st.metric("MAPE", f"{gam_metrics['mape']:.2f}%")

        with col2:
            st.markdown("**Naive Baseline** (tomorrow = today)")
            st.metric("MAE", f"${naive_metrics['mae']:.2f}")
            st.metric("RMSE", f"${naive_metrics['rmse']:.2f}")
            st.metric("MAPE", f"{naive_metrics['mape']:.2f}%")

        # Residual distribution
        st.subheader("Residual Distribution")
        residuals = y - y_pred
        fig_resid = go.Figure()
        fig_resid.add_trace(
            go.Histogram(x=residuals, nbinsx=80, name="Residuals",
                         marker_color="#1f77b4", opacity=0.7)
        )
        fig_resid.update_layout(
            title="Distribution of Prediction Residuals",
            xaxis_title="Residual (USD)",
            yaxis_title="Count",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    # Tab 3: Partial Dependence
    with tab3:
        st.subheader("Partial Dependence Plots")
        st.markdown(
            "See how each feature influences the predicted oil price, "
            "holding all other features constant."
        )

        selected_feature = st.selectbox(
            "Select Feature",
            options=feature_names,
            index=0,
        )

        if selected_feature:
            feat_idx = feature_names.index(selected_feature)
            fig = plot_partial_dependence(gam, X, feat_idx, selected_feature)
            st.plotly_chart(fig, use_container_width=True)

            # Show feature statistics
            feat_vals = X[:, feat_idx]
            col1, col2, col3 = st.columns(3)
            col1.metric("Min", f"{feat_vals.min():.4f}")
            col2.metric("Mean", f"{feat_vals.mean():.4f}")
            col3.metric("Max", f"{feat_vals.max():.4f}")

    # Tab 4: What-If Analysis
    with tab4:
        st.subheader("What-If Analysis")
        st.markdown(
            "Adjust feature values and see how the predicted oil price changes. "
            "The baseline is the most recent observation."
        )

        # Use the latest observation as baseline
        base_X = X[-1:].copy()
        current_pred = gam.predict(base_X)[0]

        st.metric("Current Predicted Price", f"${current_pred:.2f}")

        # Feature selection for what-if
        whatif_feature = st.selectbox(
            "Select Feature to Vary",
            options=feature_names,
            index=0,
            key="whatif_feature",
        )

        if whatif_feature:
            feat_idx = feature_names.index(whatif_feature)
            feat_vals = X[:, feat_idx]

            current_val = base_X[0, feat_idx]
            feat_min = float(feat_vals.min())
            feat_max = float(feat_vals.max())
            feat_range = feat_max - feat_min

            # Slider to set range
            st.markdown(f"**Current value:** {current_val:.4f}")
            slider_range = st.slider(
                f"Vary {whatif_feature}",
                min_value=feat_min - 0.1 * feat_range,
                max_value=feat_max + 0.1 * feat_range,
                value=(feat_min, feat_max),
                key="whatif_range",
            )

            vary_range = np.linspace(slider_range[0], slider_range[1], 100)
            fig = plot_what_if_analysis(
                gam, base_X, feat_idx, vary_range, whatif_feature
            )
            st.plotly_chart(fig, use_container_width=True)

            # Quick scenario buttons
            st.markdown("**Quick Scenarios:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                pct_change = -0.05
                scenario_val = current_val * (1 + pct_change)
                scenario_X = base_X.copy()
                scenario_X[0, feat_idx] = scenario_val
                scenario_pred = gam.predict(scenario_X)[0]
                st.metric(
                    f"-5% ({scenario_val:.2f})",
                    f"${scenario_pred:.2f}",
                    f"${scenario_pred - current_pred:.2f}",
                )

            with col2:
                st.metric("Current", f"${current_pred:.2f}")

            with col3:
                pct_change = 0.05
                scenario_val = current_val * (1 + pct_change)
                scenario_X = base_X.copy()
                scenario_X[0, feat_idx] = scenario_val
                scenario_pred = gam.predict(scenario_X)[0]
                st.metric(
                    f"+5% ({scenario_val:.2f})",
                    f"${scenario_pred:.2f}",
                    f"${scenario_pred - current_pred:.2f}",
                )


if __name__ == "__main__":
    main()
