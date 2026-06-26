"""End-to-end pipeline runner for crude oil price forecasting.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-download
    python scripts/run_pipeline.py --skip-eda
    python scripts/run_pipeline.py --skip-download --skip-eda
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import run_walk_forward_backtest
from src.combination import fit_from_backtest, save_weights as save_combination_weights
from src.config_loader import get_project_root, load_config
from src.data_acquisition import acquire_all
from src.data_engineering import build_features
from src.eda import run_eda
from src.evaluation import get_prediction_sigma, run_evaluation
from src.model import (
    build_feature_matrix,
    conformal_residual_quantile,
    load_feature_names,
    load_model,
    reconstruct_price_from_returns,
    train_and_save,
)
import joblib


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_cpcv_step(config: dict, cpcv_cfg: dict, logger: logging.Logger) -> None:
    """Run CPCV backtest on the trained-feature set and persist results.

    Uses the same feature pipeline as the walk-forward backtest but fits a
    fresh GAM per CPCV combination on the *selected* feature columns so
    stepwise drops are honoured.
    """
    from src.cpcv_backtest import make_gam_factory, run_cpcv_backtest
    from src.model import compute_anchor_prices, define_gam_terms, get_target_y_level

    root = get_project_root(config)
    feat_cfg = config["features"]
    target = feat_cfg["target"]
    target_transform = feat_cfg.get("target_transform", "level")

    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    X_full, y_model, all_feature_names, target_dates, t1 = build_feature_matrix(
        df, target, target_transform=target_transform,
    )

    # Restrict to the model's selected features (mirrors evaluation/dashboard).
    saved_names_path = root / "outputs" / "models" / "feature_names.pkl"
    feature_names = (
        load_feature_names(saved_names_path)
        if saved_names_path.exists() else all_feature_names
    )
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    col_idx = [name_to_idx[n] for n in feature_names if n in name_to_idx]
    X = X_full[:, col_idx]

    transformer_path = root / "outputs" / "models" / "feature_transformer.pkl"
    if transformer_path.exists():
        transformer = joblib.load(transformer_path)
        X = transformer.transform(X)

    # Build a fresh terms factory each fit (pyGAM mutates terms during fit).
    feature_names_for_factory = list(feature_names)

    def _terms_factory():
        return define_gam_terms(feature_names_for_factory, config)

    from src.config_loader import resolve_model_config
    model_cfg = resolve_model_config(config)
    model_factory = make_gam_factory(
        _terms_factory, lam_values=model_cfg.get("lam_search"),
    )

    anchor = compute_anchor_prices(df, target, forecast_horizon=1)
    y_level = get_target_y_level(df, target, forecast_horizon=1)

    output_dir = root / "outputs" / "backtests" / "cpcv"
    result = run_cpcv_backtest(
        X, y_model, t1, model_factory,
        anchor=anchor,
        y_level=y_level,
        target_dates=target_dates,
        n_splits=int(cpcv_cfg.get("n_splits", 6)),
        n_test_splits=int(cpcv_cfg.get("n_test_splits", 2)),
        pct_embargo=float(cpcv_cfg.get("pct_embargo", 0.01)),
        output_dir=output_dir,
        n_jobs=int(model_cfg.get("n_jobs", 1)),
    )

    logger.info("CPCV summary:")
    for k, v in result.summary.items():
        logger.info(f"  {k:20s} {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crude oil price forecasting pipeline"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data acquisition (use existing data in data/raw/)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA plot generation",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip walk-forward backtest (slow; defaults to running)",
    )
    parser.add_argument(
        "--cpcv",
        action="store_true",
        help=(
            "Also run the Combinatorial Purged CV backtester (Ch. 12). "
            "Writes outputs/backtests/cpcv/. Can be enabled persistently "
            "via the cpcv: section in config.yaml."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("pipeline")
    config = load_config()

    # Step 1: Data Acquisition
    if args.skip_download:
        logger.info("Skipping data acquisition (--skip-download)")
    else:
        logger.info("=" * 60)
        logger.info("STEP 1: Data Acquisition")
        logger.info("=" * 60)
        acquire_all(config)

    # Step 2: Feature Engineering
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 60)
    build_features(config)

    # Step 3: EDA
    if args.skip_eda:
        logger.info("Skipping EDA (--skip-eda)")
    else:
        logger.info("=" * 60)
        logger.info("STEP 3: Exploratory Data Analysis")
        logger.info("=" * 60)
        run_eda(config)

    # Step 4: Model Training
    logger.info("=" * 60)
    logger.info("STEP 4: Model Training & Cross-Validation")
    logger.info("=" * 60)
    mu_gam, sigma_gam, nu_gam, tau_gam, cv_results = train_and_save(config)

    # Step 5: Evaluation
    logger.info("=" * 60)
    logger.info("STEP 5: Model Evaluation")
    logger.info("=" * 60)
    run_evaluation(config)

    # Step 6: Walk-Forward Backtest + Bates-Granger combination
    if args.skip_backtest:
        logger.info("Skipping walk-forward backtest (--skip-backtest)")
        bt_result = None
    else:
        logger.info("=" * 60)
        logger.info("STEP 6: Walk-Forward Backtest")
        logger.info("=" * 60)
        bt_result = run_walk_forward_backtest(config)
        logger.info("Backtest summary:")
        for k, v in bt_result.scores.items():
            logger.info(f"  {k:30s} {v}")

        # Density-weighted acceptance gate (Phase 0): the headline numbers a
        # change must not worsen.  Lower CRPS/log-score is better; MSPE ratio
        # < 1 beats no-change; small DM/PT p-values mean the edge is real;
        # large Berkowitz p-value means the fan chart is calibrated.
        s = bt_result.scores
        logger.info(
            "GATE | CRPS=%.5f log_score=%.4f | MSPE_ratio=%.4f (DM p=%.3f) | "
            "dir_acc=%.3f (PT p=%.3f) | Berkowitz p=%.3f",
            s.get("crps_mean_model", float("nan")),
            s.get("log_score_mean_model", float("nan")),
            s.get("mspe_ratio", float("nan")),
            s.get("dm_pvalue", float("nan")),
            s.get("directional_accuracy", float("nan")),
            s.get("pt_pvalue", float("nan")),
            s.get("berkowitz_pvalue_model", float("nan")),
        )

        # Bates-Granger optimal naive combination weights
        logger.info("Computing Bates-Granger combination weights against naive baseline…")
        weights = fit_from_backtest(bt_result.path)
        save_combination_weights(weights, config)

    # Step 6.5: Combinatorial Purged CV (Ch. 12) — optional second backtester
    cpcv_cfg = (config.get("cpcv") or {})
    if args.cpcv or cpcv_cfg.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STEP 6.5: Combinatorial Purged CV Backtest")
        logger.info("=" * 60)
        _run_cpcv_step(config, cpcv_cfg, logger)

    # Step 7: Next-Day Forecast
    logger.info("=" * 60)
    logger.info("STEP 7: Next-Day Forecast")
    logger.info("=" * 60)

    root = get_project_root(config)
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    feat_cfg = config["features"]
    target = feat_cfg["target"]
    target_transform = feat_cfg.get("target_transform", "level")

    # Load trained model and its selected feature names
    gam = load_model(root / "outputs" / "models" / "gam_model.pkl")
    saved_names_path = root / "outputs" / "models" / "feature_names.pkl"
    X_full, _, all_feature_names, _, _ = build_feature_matrix(
        df, target, forecast_horizon=0, target_transform=target_transform,
    )
    feature_names = (
        load_feature_names(saved_names_path)
        if saved_names_path.exists()
        else all_feature_names
    )
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    col_idx = [name_to_idx[n] for n in feature_names]

    # Use the most recent row of features to forecast the next trading day
    X_latest = X_full[-1:, :][:, col_idx]

    # Apply saved skewed-feature transformer if model was trained with it
    transformer_path = root / "outputs" / "models" / "feature_transformer.pkl"
    if transformer_path.exists():
        transformer = joblib.load(transformer_path)
        X_latest = transformer.transform(X_latest)

    forecast_mu_model = float(gam.predict(X_latest)[0])
    forecast_sigma_model = float(get_prediction_sigma(gam, X_latest)[0])
    forecast_date = df.index[-1] + pd.offsets.BDay(1)
    last_close = float(df[target].iloc[-1])

    # Split-conformal PI calibration: if a walk-forward backtest is
    # available, use its OOS residuals to compute the empirical 95%
    # nonconformity quantile in MODEL space.  Fall back to the Gaussian
    # 1.96 when no path is available (e.g., --skip-backtest).  This
    # guarantees marginal 95% coverage under exchangeability, even when
    # the Johnson SU shape is mis-specified.
    pi_multiplier = 1.96
    if bt_result is not None and len(bt_result.path) > 50:
        if target_transform == "log_return":
            y_true_model_bt = np.log(
                bt_result.path["y_true"].values
                / np.clip(bt_result.path["y_anchor"].values, 1e-12, None)
            )
        else:
            y_true_model_bt = bt_result.path["y_true"].values
        try:
            pi_multiplier = conformal_residual_quantile(
                y_true_model_bt,
                bt_result.path["y_pred_model"].values,
                sigma=bt_result.path["sigma_model"].values,
                alpha=0.05,
            )
            logger.info(
                f"Conformal PI multiplier (α=0.05): {pi_multiplier:.3f} "
                f"(vs Gaussian 1.96)"
            )
        except Exception as exc:
            logger.warning(f"Conformal calibration failed ({exc}); using Gaussian 1.96")

    if target_transform == "log_return":
        forecast_mu_price = float(reconstruct_price_from_returns(
            np.array([forecast_mu_model]), np.array([last_close]),
        )[0])
        # Delta-method PI: σ_p ≈ P · σ_r (good when r is small)
        forecast_sigma_price = last_close * forecast_sigma_model
        # Asymmetric PI in price space using exp transform
        pi_lower = last_close * np.exp(forecast_mu_model - pi_multiplier * forecast_sigma_model)
        pi_upper = last_close * np.exp(forecast_mu_model + pi_multiplier * forecast_sigma_model)
    else:
        forecast_mu_price = forecast_mu_model
        forecast_sigma_price = forecast_sigma_model
        pi_lower = forecast_mu_price - pi_multiplier * forecast_sigma_price
        pi_upper = forecast_mu_price + pi_multiplier * forecast_sigma_price

    logger.info(f"Last trading day:    {df.index[-1].date()}")
    logger.info(f"Last close:          ${last_close:.2f}")
    logger.info(f"Forecast date:       {forecast_date.date()}")
    logger.info(f"Target space:        {target_transform}")
    logger.info(f"Point forecast:      ${forecast_mu_price:.2f}")
    logger.info(f"95% PI:              ${pi_lower:.2f} — ${pi_upper:.2f}")

    forecast_data = {
        "last_trading_day": str(df.index[-1].date()),
        "last_close": round(last_close, 2),
        "forecast_date": str(forecast_date.date()),
        "target_transform": target_transform,
        "point_forecast": round(forecast_mu_price, 2),
        "sigma": round(forecast_sigma_price, 2),
        "pi_95_lower": round(pi_lower, 2),
        "pi_95_upper": round(pi_upper, 2),
        "pi_multiplier": round(pi_multiplier, 4),
    }
    if target_transform == "log_return":
        forecast_data["forecast_log_return"] = round(forecast_mu_model, 5)

    # If a Bates-Granger weight was computed, also write the combined forecast
    if bt_result is not None:
        from src.combination import load_weights as _load_w, combine_forecasts as _combine
        w = _load_w(config)
        if w is not None:
            combo = _combine(
                np.array([forecast_mu_price]),
                np.array([last_close]),
                w_gam=w["w_gam"],
            )[0]
            forecast_data["combo_weight_gam"] = round(w["w_gam"], 4)
            forecast_data["combo_forecast"] = round(float(combo), 2)
            logger.info(f"Combo forecast (w_gam={w['w_gam']:.2f}):  ${combo:.2f}")

    forecast_dir = root / "outputs"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = forecast_dir / "forecast.json"
    forecast_path.write_text(json.dumps(forecast_data, indent=2))
    logger.info(f"Saved forecast to {forecast_path}")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  - Figures:        outputs/figures/")
    logger.info("  - Models:         outputs/models/gam_models.pkl  (mu, sigma, nu, tau)")
    logger.info("  - Forecast:       outputs/forecast.json")
    if bt_result is not None:
        logger.info("  - Backtest path:  outputs/backtest_path.parquet")
        logger.info("  - Combo weights:  outputs/combination_weights.json")
    if args.cpcv or (config.get("cpcv") or {}).get("enabled", False):
        logger.info("  - CPCV results:   outputs/backtests/cpcv/")
    logger.info("  - Run dashboard:  streamlit run app/dashboard.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
