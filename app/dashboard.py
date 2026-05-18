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
import joblib
from src.backtest import load_backtest_path, run_walk_forward_backtest
from src.combination import combine_forecasts, fit_from_backtest, load_weights as load_combo_weights
from src.config_loader import load_config
from src.evaluation import (
    PredictiveDistribution,
    calibrate_sigma_scale,
    compute_metrics,
    compute_probabilistic_scores,
    directional_accuracy,
    fit_residual_distribution,
    get_prediction_sigma,
    naive_baseline_predictions,
)
from src.model import (
    _classify_feature,
    build_feature_matrix,
    compute_anchor_prices,
    get_target_y_level,
    load_all_models,
    load_feature_names,
    load_model,
    get_sigma_from_sigma_gam,
    reconstruct_price_from_returns,
)


@st.cache_data
def load_data(root: str) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "data" / "processed" / "features.parquet")


@st.cache_resource
def load_feature_transformer(root: str):
    """Load SkewedFeatureTransformer if present; else return None."""
    p = Path(root) / "outputs" / "models" / "feature_transformer.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource
def load_model_metadata(root: str) -> dict:
    """Load metadata sidecar (target_transform, etc.); empty dict if missing."""
    p = Path(root) / "outputs" / "models" / "model_metadata.pkl"
    if not p.exists():
        return {}
    return joblib.load(p)


def _to_price_space(
    y_pred_model: np.ndarray,
    anchor: np.ndarray,
    target_transform: str,
) -> np.ndarray:
    """Convert raw model predictions to price-level space."""
    if target_transform == "log_return":
        return np.asarray(anchor) * np.exp(np.asarray(y_pred_model))
    return np.asarray(y_pred_model)


@st.cache_resource
def load_gam_model(root: str):
    """Load the mu GAM (backward-compatible single model)."""
    return load_model(Path(root) / "outputs" / "models" / "gam_model.pkl")


@st.cache_resource
def load_all_gam_models_cached(root: str) -> dict:
    """Load all distributional sub-models (mu, sigma, nu, tau) from gam_models.pkl.

    Falls back to wrapping the legacy gam_model.pkl with key ``\"mu\"`` if the
    joint file does not yet exist (pre-retrain scenario).
    """
    models_path = Path(root) / "outputs" / "models" / "gam_models.pkl"
    if models_path.exists():
        return load_all_models(models_path)
    return {"mu": load_model(Path(root) / "outputs" / "models" / "gam_model.pkl")}


@st.cache_data
def load_features(root: str) -> list[str]:
    return load_feature_names(Path(root) / "outputs" / "models" / "feature_names.pkl")


@st.cache_data
def load_sub_feature_names(root: str) -> dict[str, list[str]]:
    """Load feature name lists for the sigma, nu, and tau sub-models."""
    base = Path(root) / "outputs" / "models"
    result: dict[str, list[str]] = {}
    for param in ("sigma", "nu", "tau"):
        path = base / f"{param}_feature_names.pkl"
        if path.exists():
            result[param] = load_feature_names(path)
    return result


@st.cache_resource
def build_predictive_dist(
    root: str,
    _gam,
    _models: dict | None = None,
) -> PredictiveDistribution:
    """Fit Johnson SU predictive distribution in MODEL space.

    * **Sigma**: If a sigma sub-model is present the per-observation σ is computed
      via ``exp(sigma_gam.predict(X_sigma))`` instead of the GAM 95% PI width.
    * **Nu / Tau**: If nu/tau sub-models are present the shape parameters become
      per-observation arrays from model predictions.  Otherwise globally fitted
      scalars are used (original behaviour).

    Honours ``features.target_transform`` — when "log_return" the residuals are
    fitted in return space.  Callers convert quantiles to price space via the
    helpers in :mod:`app.components`.

    The leading underscore on ``_gam`` / ``_models`` prevents Streamlit from
    attempting to hash these objects; the result is keyed only on ``root``.
    """
    df = load_data(root)
    cfg = load_config()
    target = cfg["features"]["target"]
    target_transform = cfg["features"].get("target_transform", "level")
    feature_names = load_features(root)
    sub_names = load_sub_feature_names(root)

    X_full, y, all_names, _target_dates, _t1 = build_feature_matrix(
        df, target, target_transform=target_transform,
    )
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_idx[n] for n in feature_names if n in name_to_idx]
    X = X_full[:, col_idx]
    transformer = load_feature_transformer(root)
    if transformer is not None:
        X = transformer.transform(X)
    y_pred = _gam.predict(X)

    # ── Sigma ─────────────────────────────────────────────────────────────────
    if _models and "sigma" in _models and "sigma" in sub_names:
        sigma_col = [feature_names.index(n) for n in sub_names["sigma"] if n in feature_names]
        X_sigma = X[:, sigma_col]
        sigma = get_sigma_from_sigma_gam(_models["sigma"], X_sigma)
    else:
        sigma = get_prediction_sigma(_gam, X)

    # ── Fit Johnson SU on standardised residuals ───────────────────────────────
    dist = fit_residual_distribution(y, y_pred, sigma)

    # ── Override nu/tau with per-observation model predictions if available ────
    if _models and "nu" in _models and "nu" in sub_names:
        nu_col = [feature_names.index(n) for n in sub_names["nu"] if n in feature_names]
        X_nu = X[:, nu_col]
        nu_arr = np.clip(_models["nu"].predict(X_nu), -10.0, 10.0)
        dist = PredictiveDistribution(
            nu=nu_arr, tau=dist.tau,
            loc_std=dist.loc_std, scale_std=dist.scale_std,
        )

    if _models and "tau" in _models and "tau" in sub_names:
        tau_col = [feature_names.index(n) for n in sub_names["tau"] if n in feature_names]
        X_tau = X[:, tau_col]
        tau_arr = np.clip(_models["tau"].predict(X_tau), 0.1, 0.9)
        dist = PredictiveDistribution(
            nu=dist.nu, tau=tau_arr,
            loc_std=dist.loc_std, scale_std=dist.scale_std,
        )

    return dist


@st.cache_data
def compute_calibrated_sigma_scale(
    root: str,
    _gam,
    _dist: PredictiveDistribution,
    target_coverage: float = 0.90,
) -> float:
    """Auto-calibrate sigma_scale so empirical coverage matches *target_coverage*.

    Coverage is measured in MODEL space (whatever the GAM was trained on);
    quantile mapping to price space is monotonic so coverage is preserved.

    The leading underscores on ``_gam`` and ``_dist`` prevent Streamlit from
    attempting to hash those objects; the result is keyed only on ``root`` and
    ``target_coverage``.
    """
    df = load_data(root)
    cfg = load_config()
    target = cfg["features"]["target"]
    target_transform = cfg["features"].get("target_transform", "level")
    feature_names = load_features(root)
    X_full, y, all_names, _target_dates, _t1 = build_feature_matrix(
        df, target, target_transform=target_transform,
    )
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_idx[n] for n in feature_names if n in name_to_idx]
    X = X_full[:, col_idx]
    transformer = load_feature_transformer(root)
    if transformer is not None:
        X = transformer.transform(X)
    return calibrate_sigma_scale(_gam, X, y, _dist, target_coverage=target_coverage)


def build_feature_importance_df(gam, feature_names: list[str]) -> pd.DataFrame:
    """Build a feature significance table from pyGAM model statistics.

    Columns: Feature, Type (l / s / cyclic), p-value, EDoF, λ.
    Sorted ascending by p-value so the most significant features appear first.

    Args:
        gam: Fitted LinearGAM with populated ``.statistics_``.
        feature_names: Column names (must match the number of non-intercept terms).

    Returns:
        DataFrame sorted by p-value ascending.
    """
    stats_dict = gam.statistics_
    n = len(feature_names)

    p_values_raw = list(stats_dict.get("p_values", []))
    edof_raw = list(stats_dict.get("edof_per_term", []))
    lam_raw = list(np.array(gam.lam).flatten())

    rows = []
    for i, name in enumerate(feature_names):
        cat = _classify_feature(name)
        type_label = {"linear": "l()", "cyclic": "s(cp)", "rolling": "s()", "lag": "s()",
                      "return": "s()", "spline": "s()"}.get(cat, "s()")
        rows.append({
            "Feature": name,
            "Type": type_label,
            "p-value": float(p_values_raw[i]) if i < len(p_values_raw) else float("nan"),
            "EDoF": float(edof_raw[i]) if i < len(edof_raw) else float("nan"),
            "\u03bb": float(lam_raw[i]) if i < len(lam_raw) else float("nan"),
        })
    df = pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)
    return df


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
    target_transform = config["features"].get("target_transform", "level")
    X_full, y_model, all_feature_names, target_dates, _t1 = build_feature_matrix(
        df, target, target_transform=target_transform,
    )

    # Restrict to the columns the model was trained on (stepwise may have
    # dropped features; saved feature_names reflects the final selection).
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    col_idx = [name_to_idx[n] for n in feature_names if n in name_to_idx]
    X = X_full[:, col_idx]

    # Latest feature row for live forecast (dropped by horizon shift)
    X_full_latest, _, _, _, _ = build_feature_matrix(
        df, target, forecast_horizon=0, target_transform=target_transform,
    )
    X_latest = X_full_latest[-1:, :][:, col_idx]

    # Apply the saved skewed-feature transformer (item 8) to both X paths
    transformer = load_feature_transformer(root)
    if transformer is not None:
        X = transformer.transform(X)
        X_latest = transformer.transform(X_latest)

    # Load distributional sub-models (sigma, nu, tau) if available
    all_models = load_all_gam_models_cached(root)
    sub_names = load_sub_feature_names(root)

    # Build sigma-sub-model feature matrix for use in components
    sigma_gam_model = all_models.get("sigma")
    if sigma_gam_model is not None and "sigma" in sub_names:
        sigma_col_idx = [feature_names.index(n) for n in sub_names["sigma"] if n in feature_names]
        X_sigma = X[:, sigma_col_idx]
    else:
        sigma_gam_model, X_sigma = None, None

    # Anchor and price-level series (always price USD/bbl)
    anchor = compute_anchor_prices(df, target, forecast_horizon=1)
    y_level = get_target_y_level(df, target, forecast_horizon=1)

    # Pre-compute predictions and fit Johnson SU distribution once
    y_pred_model = gam.predict(X)
    y_pred_price = _to_price_space(y_pred_model, anchor, target_transform)
    # Backward-compat alias used throughout the rest of main()
    y = y_level
    y_pred = y_pred_price
    dist = build_predictive_dist(root, gam, _models=all_models)  # cached

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
    nu_val = dist.nu
    tau_val = dist.tau
    if isinstance(nu_val, np.ndarray):
        st.sidebar.markdown(
            f"Johnson SU — **per-observation** ν and τ  \n"
            f"ν range: [{nu_val.min():.3f}, {nu_val.max():.3f}]  (mean {nu_val.mean():.3f})  \n"
            f"τ range: [{tau_val.min():.3f}, {tau_val.max():.3f}]  (mean {tau_val.mean():.3f})  \n"
            f"σ source: {'sigma sub-model' if sigma_gam_model is not None else 'PI width'}"
        )
    else:
        st.sidebar.markdown(
            f"Johnson SU fitted to in-sample residuals  \n"
            f"ν = {float(nu_val):.3f} (skewness)  \n"
            f"τ = {float(tau_val):.3f} (tail weight)  \n"
            f"{'Heavier tails than Normal' if float(tau_val) < 1 else 'Similar/lighter tails than Normal'}"
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
            "Multiplies the predictive σ.  "
            "Default is auto-calibrated to achieve empirical 90% coverage "
            "on the most recent 20% of in-sample data."
        ),
    )
    st.sidebar.caption(f"Auto-calibrated value: {sigma_scale_auto:.2f}×")

    # Recompute sigma using the user-chosen scale factor
    sigma = _get_sigma_from_gam(
        gam, X, sigma_scale=sigma_scale,
        sigma_gam=sigma_gam_model, X_sigma=X_sigma,
    )
    # ── Tabs ─────────────────────────────────────────────────────────────────
    # CPCV tab is only shown when results exist on disk
    # (outputs/backtests/cpcv/summary.json).
    cpcv_summary_path = Path(root) / "outputs" / "backtests" / "cpcv" / "summary.json"
    show_cpcv_tab = cpcv_summary_path.exists()
    tab_labels = [
        "Price Overview",
        "Model Performance",
        "Probability Forecast",
        "Partial Dependence",
        "What-If Analysis",
        "Model Terms",
        "Backtest",
    ]
    if show_cpcv_tab:
        tab_labels.append("CPCV Distribution")
    _tabs = st.tabs(tab_labels)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = _tabs[:7]
    tab8 = _tabs[7] if show_cpcv_tab else None

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

        # Next-day forecast using the latest available features
        st.markdown("---")
        st.subheader("Next-Day Forecast")
        forecast_mu_model = float(gam.predict(X_latest)[0])
        X_sigma_latest = X_latest[:, sigma_col_idx] if sigma_gam_model is not None and "sigma" in sub_names else None
        forecast_sigma_model = float(_get_sigma_from_gam(
            gam, X_latest, sigma_scale=sigma_scale,
            sigma_gam=sigma_gam_model, X_sigma=X_sigma_latest,
        )[0])
        # Reconstruct to price space when needed
        last_close = float(df[target].iloc[-1])
        if target_transform == "log_return":
            forecast_mu_price = last_close * float(np.exp(forecast_mu_model))
            pi_lo = last_close * float(np.exp(forecast_mu_model - 1.96 * forecast_sigma_model))
            pi_hi = last_close * float(np.exp(forecast_mu_model + 1.96 * forecast_sigma_model))
        else:
            forecast_mu_price = forecast_mu_model
            pi_lo = forecast_mu_price - 1.96 * forecast_sigma_model
            pi_hi = forecast_mu_price + 1.96 * forecast_sigma_model
        forecast_date = df.index[-1] + pd.offsets.BDay(1)
        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Forecast Date", f"{forecast_date.date()}")
        fcol2.metric("Point Forecast", f"${forecast_mu_price:.2f}",
                     delta=f"${forecast_mu_price - latest:.2f} vs last close",
                     delta_color="normal")
        fcol3.metric("95% Prediction Interval", f"${pi_lo:.2f} — ${pi_hi:.2f}")
        if target_transform == "log_return":
            st.caption(
                f"Target space: log-return; predicted r̂ = {forecast_mu_model:+.4f} "
                f"(σ̂ = {forecast_sigma_model:.4f}).  "
                "Price reconstructed via P̂ = P_t · exp(r̂)."
            )

    # ── Tab 2: Model Performance ──────────────────────────────────────────────
    with tab2:
        st.subheader("Forecast vs Actual (1-Day-Ahead)")
        # Build 95% PI in price space directly from Johnson SU quantiles
        sigma_for_ci = _get_sigma_from_gam(
            gam, X, sigma_scale=sigma_scale,
            sigma_gam=sigma_gam_model, X_sigma=X_sigma,
        )
        if target_transform == "log_return":
            r_lo = dist.ppf_array(0.025, y_pred_model, sigma_for_ci)
            r_hi = dist.ppf_array(0.975, y_pred_model, sigma_for_ci)
            ci_lo_price = anchor * np.exp(r_lo)
            ci_hi_price = anchor * np.exp(r_hi)
        else:
            ci_lo_price = dist.ppf_array(0.025, y_pred_model, sigma_for_ci)
            ci_hi_price = dist.ppf_array(0.975, y_pred_model, sigma_for_ci)
        st.plotly_chart(
            plot_forecast_vs_actual(target_dates, y, y_pred, ci_lo_price, ci_hi_price),
            use_container_width=True,
        )

        st.subheader("Metrics Comparison (price scale)")
        y_naive = naive_baseline_predictions(y)
        gam_metrics = compute_metrics(y, y_pred)
        naive_metrics = compute_metrics(y[1:], y_naive[1:])
        dir_gam = directional_accuracy(y, y_pred, anchor)
        dir_naive = directional_accuracy(y[1:], y_naive[1:], anchor[1:])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**GAM Model**")
            st.metric("MAE",  f"${gam_metrics['mae']:.2f}")
            st.metric("RMSE", f"${gam_metrics['rmse']:.2f}")
            st.metric("MAPE", f"{gam_metrics['mape']:.2f}%")
            st.metric("Directional accuracy", f"{dir_gam * 100:.1f}%")
        with col2:
            st.markdown("**Naive Baseline** (tomorrow = today)")
            st.metric("MAE",  f"${naive_metrics['mae']:.2f}")
            st.metric("RMSE", f"${naive_metrics['rmse']:.2f}")
            st.metric("MAPE", f"{naive_metrics['mape']:.2f}%")
            st.metric("Directional accuracy", f"{dir_naive * 100:.1f}%")

        # ── Probabilistic scores (item 13) ────────────────────────────────────
        with st.expander("Probabilistic scores (CRPS / pinball / log-score / coverage)"):
            with st.spinner("Computing probabilistic scores…"):
                prob = compute_probabilistic_scores(
                    y_true=y_model,
                    y_pred_mu=y_pred_model,
                    sigma=sigma_for_ci,
                    dist=dist,
                    y_anchor=np.zeros_like(y_model) if target_transform == "log_return" else anchor,
                    n_crps_samples=200,
                )
            prob_df = pd.DataFrame(
                [(k, v) for k, v in prob.items()],
                columns=["metric", "value"],
            )
            st.dataframe(prob_df.style.format({"value": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.caption(
                "All probabilistic scores are computed in MODEL space "
                f"({'log-return' if target_transform == 'log_return' else 'price'}); "
                "lower CRPS / pinball / log-score = better.  "
                "Coverage values should be close to their nominal level."
            )

        st.subheader("Residual Distribution (price scale)")
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
            from app.components import plot_fan_chart_price_space_plotly
            st.plotly_chart(
                plot_fan_chart_price_space_plotly(
                    dates=target_dates,
                    y_true_price=y_level,
                    y_pred_price=y_pred_price,
                    y_pred_model=y_pred_model,
                    sigma_model=sigma,
                    anchor=anchor,
                    dist=dist,
                    last_n_days=fan_days,
                    target_transform=target_transform,
                    sigma_scale=1.0,  # sigma already scaled above
                ),
                use_container_width=True,
            )

        with col_right:
            st.markdown("#### Predictive Density")
            st.markdown(
                "Select a date to see the full probability distribution for "
                "that day's forecast."
            )

            # Date picker scoped to target dates (the dates being predicted)
            available_dates = target_dates.date
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
            idx = target_dates.searchsorted(ts)
            idx = min(idx, len(target_dates) - 1)

            mu_model_sel = float(y_pred_model[idx])
            sigma_sel = float(sigma[idx])
            actual_price_sel = float(y_level[idx])
            anchor_sel = float(anchor[idx])
            mu_price_sel = float(y_pred_price[idx])

            from app.components import plot_predictive_density_price_space_plotly
            st.plotly_chart(
                plot_predictive_density_price_space_plotly(
                    mu_model=mu_model_sel,
                    sigma_model=sigma_sel,
                    anchor=anchor_sel,
                    dist=dist,
                    y_actual_price=actual_price_sel,
                    title=f"Predictive Distribution — {target_dates[idx].date()}",
                    target_transform=target_transform,
                    obs_idx=idx,
                ),
                use_container_width=True,
            )

            # Summary stats — Johnson SU CDF in MODEL space; map actual to model space
            if target_transform == "log_return":
                actual_model_sel = float(np.log(actual_price_sel / anchor_sel))
            else:
                actual_model_sel = actual_price_sel
            pctile = float(dist.cdf(actual_model_sel, mu_model_sel, sigma_sel, idx=idx) * 100)
            _nu_disp = float(dist.nu[idx]) if isinstance(dist.nu, np.ndarray) else float(dist.nu)
            _tau_disp = float(dist.tau[idx]) if isinstance(dist.tau, np.ndarray) else float(dist.tau)
            dist_note = f"ν={_nu_disp:.2f}, τ={_tau_disp:.2f}"
            st.markdown(f"""
| Metric | Value |
|---|---|
| Point forecast (price) | ${mu_price_sel:.2f} |
| Pred. σ ({target_transform}) | {sigma_sel:.4f} |
| Distribution | Johnson SU ({dist_note}) |
| Actual price | ${actual_price_sel:.2f} |
| Actual percentile | {pctile:.1f}th |
""")

    # ── Tab 4: Partial Dependence ─────────────────────────────────────────────
    with tab4:
        st.subheader("Partial Dependence Plots")
        st.markdown(
            "See how each feature influences the predicted oil price, "
            "holding all other features constant."
        )
        if target_transform == "log_return":
            st.caption(
                "**Note**: y-axis is in **log-return** space (the model's training target). "
                "A partial-dependence value of +0.01 means the feature shifts the predicted "
                "log-return by 0.01, i.e. ≈ 1% change in next-day price."
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
        if target_transform == "log_return":
            st.caption(
                "**Note**: the chart's y-axis is in log-return space.  Use the "
                "metric below for the equivalent reconstructed price."
            )

        base_X = X_latest.copy()
        current_pred_model = float(gam.predict(base_X)[0])
        current_sigma = float(_get_sigma_from_gam(gam, base_X, sigma_scale=sigma_scale)[0])
        if target_transform == "log_return":
            current_pred = last_close * float(np.exp(current_pred_model))
        else:
            current_pred = current_pred_model

        st.metric("Current Predicted Price", f"${current_pred:.2f}",
                  help=f"σ ({target_transform}) = {current_sigma:.4f}")

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
                from app.components import plot_predictive_density_price_space_plotly
                st.plotly_chart(
                    plot_predictive_density_price_space_plotly(
                        mu_model=current_pred_model,
                        sigma_model=current_sigma,
                        anchor=last_close,
                        dist=dist,
                        title=f"Forecast Distribution\n({whatif_feature} = {current_val:.3f})",
                        target_transform=target_transform,
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
                p_model = float(gam.predict(X_s)[0])
                sig_model = float(_get_sigma_from_gam(gam, X_s, sigma_scale=sigma_scale)[0])
                if target_transform == "log_return":
                    p_price = last_close * float(np.exp(p_model))
                    lo95_price = last_close * float(np.exp(dist.ppf(0.025, p_model, sig_model)))
                    hi95_price = last_close * float(np.exp(dist.ppf(0.975, p_model, sig_model)))
                else:
                    p_price = p_model
                    lo95_price = float(dist.ppf(0.025, p_model, sig_model))
                    hi95_price = float(dist.ppf(0.975, p_model, sig_model))
                pi_str = f"95% PI: ${lo95_price:.1f}–${hi95_price:.1f}"
                label = f"{'+' if pct > 0 else ''}{int(pct*100)}%  ({val:.2f})" if pct != 0 else "Current"
                col.metric(label, f"${p_price:.2f}", pi_str)

    # ── Tab 6: Model Terms ────────────────────────────────────────────────────
    with tab6:
        st.subheader("Distributional Model Terms — Feature Significance")
        st.markdown(
            "All four distributional sub-models are shown below, sorted by **p-value** "
            "(low = significant).  Red p-values (< 0.05) indicate terms that contribute "
            "significantly to that distribution parameter.  EDoF ≈ 1 means the term is "
            "nearly linear; higher values indicate non-linear curvature.  "
            "λ is the smoothing penalty (higher = smoother / less flexible)."
        )

        _PARAM_LABELS = {
            "mu":    ("μ — Location (price level)",     "mu",    gam,         feature_names),
            "sigma": ("σ — Scale (conditional volatility)", "sigma", None, None),
            "nu":    ("ν — Skewness",                   "nu",    None, None),
            "tau":   ("τ — Tail weight",                "tau",   None, None),
        }

        # Inject sub-models from all_models into the label table
        for _pk in ("sigma", "nu", "tau"):
            _sub_gam = all_models.get(_pk)
            _sub_names = sub_names.get(_pk)
            _lbl, _key, _, _ = _PARAM_LABELS[_pk]
            _PARAM_LABELS[_pk] = (_lbl, _key, _sub_gam, _sub_names)

        for _param_key in ("mu", "sigma", "nu", "tau"):
            _lbl, _pk, _sub_gam, _sub_names = _PARAM_LABELS[_param_key]
            st.markdown(f"#### {_lbl}")

            if _sub_gam is None or _sub_names is None:
                st.info(
                    f"No {_param_key} sub-model found. Re-run the pipeline to generate it."
                )
                continue

            try:
                _df_imp = build_feature_importance_df(_sub_gam, _sub_names)
                n_terms = len(_df_imp)
                low_p = _df_imp["p-value"].min()
                st.caption(
                    f"{n_terms} term{'s' if n_terms != 1 else ''} retained  |  "
                    f"lowest p-value: {low_p:.4f}"
                )

                # Style: highlight significant rows (p < 0.05) in light red
                def _highlight_sig(row):
                    pv = row.get("p-value", 1.0)
                    color = "background-color: #ffe0e0" if (not pd.isna(pv) and pv < 0.05) else ""
                    return [color] * len(row)

                st.dataframe(
                    _df_imp.style
                        .apply(_highlight_sig, axis=1)
                        .format({"p-value": "{:.4f}", "EDoF": "{:.2f}", "λ": "{:.4f}"}),
                    use_container_width=True,
                    height=min(400, 36 + n_terms * 35),
                )
            except Exception as _e:
                st.warning(f"Could not build {_param_key} terms table: {_e}")


    # ── Tab 7: Backtest (walk-forward + drift) ────────────────────────────────
    with tab7:
        st.subheader("Walk-Forward Backtest — Production Simulation")
        st.markdown(
            "Cross-validation reports summary statistics suitable for hyperparameter "
            "tuning, but does not simulate production: the model is never re-fit on a "
            "sliding window.  This tab replays the entire history with periodic re-fits "
            "and stitches a continuous out-of-sample forecast path.  All metrics here "
            "are honest out-of-sample numbers."
        )

        bt_path = load_backtest_path(config)
        col_a, col_b = st.columns([4, 1])
        with col_b:
            bt_show_days = st.slider(
                "Days to show", min_value=60, max_value=2000, value=500, step=50,
                key="bt_show_days",
            )
            run_now = st.button("Re-run backtest", help="Recomputes from scratch — slow")
        if run_now:
            with st.spinner("Running walk-forward backtest…"):
                bt = run_walk_forward_backtest(config, df=df)
                bt_path = bt.path
                st.success(
                    f"Backtest complete: {len(bt.path)} out-of-sample observations "
                    f"across {bt.config_snapshot['n_refits']} refits."
                )
        if bt_path is None:
            st.info(
                "No saved backtest found.  Run `python scripts/run_pipeline.py` "
                "(without `--skip-backtest`) or click **Re-run backtest** above."
            )
        else:
            from app.components import plot_backtest_path, plot_rolling_mae_plotly
            st.plotly_chart(
                plot_backtest_path(bt_path, last_n_days=bt_show_days, show_pi_widths=(95,)),
                use_container_width=True,
            )

            # Summary scores
            scores_path = (Path(root) / "outputs" / "backtest_path.scores.json")
            if scores_path.exists():
                import json as _json
                scores = _json.loads(scores_path.read_text())
                cols = st.columns(4)
                cols[0].metric("Walk-forward MAE",
                               f"${scores.get('mae_price', float('nan')):.2f}")
                cols[1].metric("RMSE skill vs naive",
                               f"{scores.get('skill_rmse', 0) * 100:+.1f}%",
                               help="Positive ⇒ GAM beats naive on RMSE")
                cols[2].metric("Directional accuracy",
                               f"{scores.get('directional_accuracy', float('nan')) * 100:.1f}%")
                cov95 = scores.get('coverage_95_empirical', float('nan'))
                cols[3].metric("Empirical 95% coverage",
                               f"{cov95 * 100:.1f}%",
                               help="Should be ≈ 95%; far from 95% ⇒ PI mis-calibrated")

            # Concept-drift monitor (item 15)
            st.markdown("---")
            st.subheader("Concept-Drift Monitor")
            st.markdown(
                "Rolling 63-day MAE of the walk-forward path.  When the orange "
                "threshold is exceeded for an extended period, consider an "
                "off-cadence re-fit — performance may have deteriorated past "
                "what the calendar-based refit window will catch."
            )
            st.plotly_chart(
                plot_rolling_mae_plotly(bt_path, window=63),
                use_container_width=True,
            )

            # Bates-Granger combination weight
            st.markdown("---")
            st.subheader("Bates-Granger Combination vs Naive")
            w = load_combo_weights(config)
            if w is None:
                if st.button("Fit combination weight", key="fit_combo"):
                    cw = fit_from_backtest(bt_path)
                    from src.combination import save_weights as _save
                    _save(cw, config)
                    st.success(f"w_gam* = {cw.w_gam:.3f} → RMSE {cw.rmse_combo:.4f}")
                    w = load_combo_weights(config)
            if w is not None:
                ccol1, ccol2, ccol3 = st.columns(3)
                ccol1.metric("Optimal w (GAM)", f"{w['w_gam']:.3f}",
                             help=f"1 − w on naive = {1.0 - w['w_gam']:.3f}")
                ccol2.metric("Combo RMSE", f"${w['rmse_combo']:.2f}")
                if w.get("skill_vs_gam") is not None:
                    ccol3.metric("Skill vs GAM-alone",
                                 f"{w['skill_vs_gam'] * 100:+.2f}%")

    # ── Tab 8: CPCV Distribution (only when results exist on disk) ────────────
    if tab8 is not None:
        with tab8:
            st.subheader("Combinatorial Purged CV — Backtest Path Distribution")
            st.markdown(
                "Lopez de Prado's CPCV (Ch. 12) holds out every combination of "
                "test groups and reassembles many independent backtest paths. "
                "Each path Sharpe below is one of those alternative histories — "
                "the spread tells you how lucky the single walk-forward path was."
            )
            import json as _json
            summary = _json.loads(cpcv_summary_path.read_text())
            metrics_path = cpcv_summary_path.parent / "paths_metrics.parquet"
            try:
                path_metrics = pd.read_parquet(metrics_path)
            except Exception as _e:
                st.error(f"Could not load CPCV metrics: {_e}")
                path_metrics = None

            cols = st.columns(4)
            cols[0].metric("Paths", summary.get("n_paths", 0))
            cols[1].metric("Sharpe mean", f"{summary.get('sharpe_mean', 0):.3f}")
            cols[2].metric(
                "Sharpe std", f"{summary.get('sharpe_std', 0):.3f}",
                help="Lower = more stable across alternative histories.",
            )
            cols[3].metric(
                "PBO proxy", f"{summary.get('pbo', 0):.1%}",
                help=(
                    "Fraction of CPCV paths with non-positive Sharpe. "
                    "Above ~50% suggests the historical Sharpe is unlikely "
                    "to repeat OOS."
                ),
            )

            if path_metrics is not None and len(path_metrics):
                import plotly.express as _px
                fig = _px.histogram(
                    path_metrics, x="sharpe", nbins=20,
                    title="Sharpe distribution across CPCV paths",
                )
                fig.update_layout(
                    xaxis_title="Annualised Sharpe", yaxis_title="Paths",
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Per-path metrics"):
                    st.dataframe(path_metrics.round(4))


# ── Helper imported inline to avoid circular deps ────────────────────────────
from scipy.stats import norm as _norm
def stats_norm_cdf(x, mu, sigma):
    return _norm.cdf(x, loc=mu, scale=sigma)


if __name__ == "__main__":
    main()
