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
    plot_fan_chart_plotly,
    plot_forecast_vs_actual,
    plot_partial_dependence,
    plot_predictive_density_plotly,
    plot_price_chart,
    plot_what_if_analysis,
    _get_sigma_from_gam,
)
from src.config_loader import load_config
from src.evaluation import (
    PredictiveDistribution,
    calibrate_sigma_scale,
    compute_metrics,
    fit_residual_distribution,
    get_prediction_sigma,
    naive_baseline_predictions,
)
from src.model import build_feature_matrix, load_feature_names, load_model


@st.cache_data
def load_data(root: str) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "data" / "processed" / "features.parquet")


@st.cache_resource
def load_gam_model(root: str):
    return load_model(Path(root) / "outputs" / "models" / "gam_model.pkl")


@st.cache_data
def load_features(root: str) -> list[str]:
    return load_feature_names(Path(root) / "outputs" / "models" / "feature_names.pkl")


@st.cache_resource
def build_predictive_dist(root: str, _gam) -> PredictiveDistribution:
    """Fit Johnson SU predictive distribution from in-sample residuals.

    The leading underscore on ``_gam`` prevents Streamlit from attempting to
    hash the model object (which would fail); the distribution is keyed only
    on ``root``.  Call this after loading the model so the cache key is stable.
    """
    df = load_data(root)
    cfg = load_config()
    target = cfg["features"]["target"]
    feature_names = load_features(root)
    X_full, y, all_names = build_feature_matrix(df, target)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_idx[n] for n in feature_names]
    X = X_full[:, col_idx]
    y_pred = _gam.predict(X)
    sigma = get_prediction_sigma(_gam, X)
    return fit_residual_distribution(y, y_pred, sigma)


@st.cache_data
def compute_calibrated_sigma_scale(
    root: str,
    _gam,
    _dist: PredictiveDistribution,
    target_coverage: float = 0.90,
) -> float:
    """Auto-calibrate sigma_scale so empirical coverage matches *target_coverage*.

    The leading underscores on ``_gam`` and ``_dist`` prevent Streamlit from
    attempting to hash those objects; the result is keyed only on ``root`` and
    ``target_coverage``.
    """
    df = load_data(root)
    cfg = load_config()
    target = cfg["features"]["target"]
    feature_names = load_features(root)
    X_full, y, all_names = build_feature_matrix(df, target)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_idx[n] for n in feature_names]
    X = X_full[:, col_idx]
    return calibrate_sigma_scale(_gam, X, y, _dist, target_coverage=target_coverage)


def main():
    st.set_page_config(
        page_title="Crude Oil Forecast",
        page_icon="🛢",
        layout="wide",
    )
    st.title("🛢 Crude Oil Price Forecasting with GAMs")

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
    X_full, y, all_feature_names = build_feature_matrix(df, target)

    # Restrict to the columns the model was trained on (stepwise may have
    # dropped features; saved feature_names reflects the final selection).
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    col_idx = [name_to_idx[n] for n in feature_names]
    X = X_full[:, col_idx]

    # Pre-compute predictions and fit Johnson SU distribution once
    y_pred = gam.predict(X)
    dist = build_predictive_dist(root, gam)  # cached Johnson SU fit

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df.index.min().date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
    )
    if len(date_range) == 2:
        mask = (df.index >= pd.Timestamp(date_range[0])) & (
            df.index <= pd.Timestamp(date_range[1])
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    fan_days = st.sidebar.slider(
        "Fan chart — days to show", min_value=60, max_value=2000,
        value=500, step=50,
    )

    # Show fitted distribution parameters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Predictive Distribution**")
    st.sidebar.markdown(
        f"Johnson SU fitted to in-sample residuals  \n"
        f"ν = {dist.nu:.3f} (skewness)  \n"
        f"τ = {dist.tau:.3f} (tail weight)  \n"
        f"{'Heavier tails than Normal' if dist.tau < 1 else 'Similar/lighter tails than Normal'}"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Uncertainty Scaling**")
    sigma_scale_auto = compute_calibrated_sigma_scale(root, gam, dist)
    sigma_scale = st.sidebar.slider(
        "σ scale  (auto-calibrated default)",
        min_value=0.5, max_value=5.0,
        value=round(sigma_scale_auto * 10) / 10,
        step=0.1,
        help=(
            "Multiplies pyGAM's raw prediction-interval width.  "
            "1.0 = raw pyGAM output (narrow).  "
            "Default is auto-calibrated to achieve empirical 90% coverage "
            "on the most recent 20% of in-sample data."
        ),
    )
    st.sidebar.caption(f"Auto-calibrated value: {sigma_scale_auto:.2f}×")

    # Recompute sigma using the user-chosen scale factor
    sigma = _get_sigma_from_gam(gam, X, sigma_scale=sigma_scale)
    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Overview",
        "Model Performance",
        "Probability Forecast",
        "Partial Dependence",
        "What-If Analysis",
    ])

    # ── Tab 1: Price Overview ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Historical Crude Oil Prices")
        oil_tickers = config["data"]["oil_tickers"]
        price_cols = [f"{t}_close" for t in oil_tickers if f"{t}_close" in df.columns]
        st.plotly_chart(plot_price_chart(df_filtered, price_cols), use_container_width=True)

        if target in df_filtered.columns:
            col1, col2, col3, col4 = st.columns(4)
            latest = df_filtered[target].iloc[-1]
            prev30 = df_filtered[target].iloc[-min(30, len(df_filtered))]
            change_30d = latest - prev30
            pct_30d = change_30d / prev30 * 100
            vol_30 = df_filtered[target].iloc[-30:].std() if len(df_filtered) >= 30 else 0

            col1.metric("Latest Price", f"${latest:.2f}")
            col2.metric("30-Day Change", f"${change_30d:.2f}", f"{pct_30d:.1f}%")
            col3.metric("30-Day Volatility", f"${vol_30:.2f}")
            col4.metric("Data Points", f"{len(df_filtered):,}")

    # ── Tab 2: Model Performance ──────────────────────────────────────────────
    with tab2:
        st.subheader("Forecast vs Actual")
        ci_95 = gam.prediction_intervals(X, width=0.95)
        st.plotly_chart(
            plot_forecast_vs_actual(df.index, y, y_pred, ci_95[:, 0], ci_95[:, 1]),
            use_container_width=True,
        )

        st.subheader("Metrics Comparison")
        y_naive = naive_baseline_predictions(y)
        gam_metrics = compute_metrics(y, y_pred)
        naive_metrics = compute_metrics(y[1:], y_naive[1:])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**GAM Model**")
            st.metric("MAE",  f"${gam_metrics['mae']:.2f}")
            st.metric("RMSE", f"${gam_metrics['rmse']:.2f}")
            st.metric("MAPE", f"{gam_metrics['mape']:.2f}%")
        with col2:
            st.markdown("**Naive Baseline** (tomorrow = today)")
            st.metric("MAE",  f"${naive_metrics['mae']:.2f}")
            st.metric("RMSE", f"${naive_metrics['rmse']:.2f}")
            st.metric("MAPE", f"{naive_metrics['mape']:.2f}%")

        st.subheader("Residual Distribution")
        residuals = y - y_pred
        fig_r = go.Figure(go.Histogram(
            x=residuals, nbinsx=80,
            marker_color="#1f77b4", opacity=0.7,
        ))
        fig_r.update_layout(
            title="Prediction Residuals",
            xaxis_title="Residual (USD)", yaxis_title="Count",
            template="plotly_white", height=350,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Tab 3: Probability Forecast ───────────────────────────────────────────
    with tab3:
        st.subheader("Predictive Probability")
        st.markdown(
            "The GAM's forecasts are not just point estimates — they carry a full "
            "**predictive distribution**. The fan chart shows the uncertainty band "
            "over time; the density curve shows the probability of each possible "
            "price outcome for a selected date."
        )

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### Fan Chart")
            st.markdown(
                "Confidence bands from darkest (50% PI) to lightest (95% PI). "
                "The actual price should fall inside the 95% band about 95% of the time."
            )
            st.plotly_chart(
                plot_fan_chart_plotly(
                    df.index, y, y_pred, gam, X,
                    last_n_days=fan_days,
                    dist=dist,
                    sigma_scale=sigma_scale,
                ),
                use_container_width=True,
            )

        with col_right:
            st.markdown("#### Predictive Density")
            st.markdown(
                "Select a date to see the full probability distribution for "
                "that day's forecast."
            )

            # Date picker scoped to available dates
            available_dates = df.index.date
            default_date = available_dates[-1]
            selected_date = st.date_input(
                "Select date",
                value=default_date,
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="density_date",
            )

            # Find the nearest available index
            ts = pd.Timestamp(selected_date)
            idx = df.index.searchsorted(ts)
            idx = min(idx, len(df) - 1)

            mu_sel = float(y_pred[idx])
            sigma_sel = float(sigma[idx])
            actual_sel = float(y[idx])

            st.plotly_chart(
                plot_predictive_density_plotly(
                    mu=mu_sel,
                    sigma=sigma_sel,
                    y_actual=actual_sel,
                    title=f"Predictive Distribution — {df.index[idx].date()}",
                    dist=dist,
                ),
                use_container_width=True,
            )

            # Summary stats — use Johnson SU CDF when available, else Normal
            if dist is not None:
                pctile = float(dist.cdf(actual_sel, mu_sel, sigma_sel) * 100)
                dist_note = f"ν={dist.nu:.2f}, τ={dist.tau:.2f}"
            else:
                pctile = float(stats_norm_cdf(actual_sel, mu_sel, sigma_sel) * 100)
                dist_note = "Normal"
            st.markdown(f"""
| Metric | Value |
|---|---|
| Point forecast | ${mu_sel:.2f} |
| Pred. σ | ${sigma_sel:.2f} |
| Distribution | {dist_note} |
| Actual price | ${actual_sel:.2f} |
| Actual percentile | {pctile:.1f}th |
""")

    # ── Tab 4: Partial Dependence ─────────────────────────────────────────────
    with tab4:
        st.subheader("Partial Dependence Plots")
        st.markdown(
            "See how each feature influences the predicted oil price, "
            "holding all other features constant."
        )
        selected_feature = st.selectbox("Select Feature", options=feature_names, index=0)
        if selected_feature:
            feat_idx = feature_names.index(selected_feature)
            st.plotly_chart(
                plot_partial_dependence(gam, X, feat_idx, selected_feature),
                use_container_width=True,
            )
            feat_vals = X[:, feat_idx]
            col1, col2, col3 = st.columns(3)
            col1.metric("Min",  f"{feat_vals.min():.4f}")
            col2.metric("Mean", f"{feat_vals.mean():.4f}")
            col3.metric("Max",  f"{feat_vals.max():.4f}")

    # ── Tab 5: What-If Analysis ───────────────────────────────────────────────
    with tab5:
        st.subheader("What-If Analysis")
        st.markdown(
            "Adjust a feature value and see how the predicted price and its "
            "**probability distribution** respond. The baseline is the most recent observation."
        )

        base_X = X[-1:].copy()
        current_pred = gam.predict(base_X)[0]
        current_sigma = float(_get_sigma_from_gam(gam, base_X, sigma_scale=sigma_scale)[0])

        st.metric("Current Predicted Price", f"${current_pred:.2f}",
                  help=f"σ = ${current_sigma:.2f}")

        whatif_feature = st.selectbox(
            "Select Feature to Vary", options=feature_names,
            index=0, key="whatif_feature",
        )

        if whatif_feature:
            feat_idx = feature_names.index(whatif_feature)
            feat_vals = X[:, feat_idx]
            current_val = float(base_X[0, feat_idx])
            feat_min, feat_max = float(feat_vals.min()), float(feat_vals.max())
            feat_range = feat_max - feat_min

            st.markdown(f"**Current value:** {current_val:.4f}")
            slider_range = st.slider(
                f"Vary {whatif_feature}",
                min_value=feat_min - 0.1 * feat_range,
                max_value=feat_max + 0.1 * feat_range,
                value=(feat_min, feat_max),
                key="whatif_range",
            )

            vary_range = np.linspace(slider_range[0], slider_range[1], 100)

            col_chart, col_density = st.columns([3, 2])
            with col_chart:
                st.markdown("#### Price vs Feature")
                st.plotly_chart(
                    plot_what_if_analysis(
                        gam, base_X, feat_idx, vary_range, whatif_feature,
                        dist=dist,
                        sigma_scale=sigma_scale,
                    ),
                    use_container_width=True,
                )

            with col_density:
                st.markdown("#### Predictive Density at Current Value")
                st.plotly_chart(
                    plot_predictive_density_plotly(
                        mu=current_pred,
                        sigma=current_sigma,
                        title=f"Forecast Distribution\n({whatif_feature} = {current_val:.3f})",
                        dist=dist,
                    ),
                    use_container_width=True,
                )

            # Quick ±5% scenarios
            st.markdown("**Quick Scenarios:**")
            col1, col2, col3 = st.columns(3)
            for col, pct in zip([col1, col2, col3], [-0.05, 0.0, 0.05]):
                val = current_val * (1 + pct) if pct != 0.0 else current_val
                X_s = base_X.copy()
                X_s[0, feat_idx] = val
                p = float(gam.predict(X_s)[0])
                sig = float(_get_sigma_from_gam(gam, X_s, sigma_scale=sigma_scale)[0])
                if dist is not None:
                    lo95 = dist.ppf(0.025, p, sig)
                    hi95 = dist.ppf(0.975, p, sig)
                    pi_str = f"95% PI: ${lo95:.1f}–${hi95:.1f}"
                else:
                    pi_str = f"σ=${sig:.2f}"
                label = f"{'+' if pct > 0 else ''}{int(pct*100)}%  ({val:.2f})" if pct != 0 else "Current"
                col.metric(label, f"${p:.2f}", pi_str)


# ── Helper imported inline to avoid circular deps ────────────────────────────
from scipy.stats import norm as _norm
def stats_norm_cdf(x, mu, sigma):
    return _norm.cdf(x, loc=mu, scale=sigma)


if __name__ == "__main__":
    main()
