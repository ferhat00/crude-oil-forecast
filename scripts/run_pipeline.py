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

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import get_project_root, load_config
from src.data_acquisition import acquire_all
from src.data_engineering import build_features
from src.eda import run_eda
from src.evaluation import get_prediction_sigma, run_evaluation
from src.model import (
    build_feature_matrix,
    load_feature_names,
    load_model,
    train_and_save,
)


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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

    # Step 6: Next-Day Forecast
    logger.info("=" * 60)
    logger.info("STEP 6: Next-Day Forecast")
    logger.info("=" * 60)

    root = get_project_root(config)
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    target = config["features"]["target"]

    # Load trained model and its selected feature names
    gam = load_model(root / "outputs" / "models" / "gam_model.pkl")
    saved_names_path = root / "outputs" / "models" / "feature_names.pkl"
    X_full, _, all_feature_names, _ = build_feature_matrix(df, target, forecast_horizon=0)
    feature_names = (
        load_feature_names(saved_names_path)
        if saved_names_path.exists()
        else all_feature_names
    )
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    col_idx = [name_to_idx[n] for n in feature_names]

    # Use the most recent row of features to forecast the next trading day
    X_latest = X_full[-1:, :][:, col_idx]
    forecast_mu = float(gam.predict(X_latest)[0])
    forecast_sigma = float(get_prediction_sigma(gam, X_latest)[0])
    forecast_date = df.index[-1] + pd.offsets.BDay(1)
    last_close = float(df[target].iloc[-1])

    logger.info(f"Last trading day:    {df.index[-1].date()}")
    logger.info(f"Last close:          ${last_close:.2f}")
    logger.info(f"Forecast date:       {forecast_date.date()}")
    logger.info(f"Point forecast:      ${forecast_mu:.2f}")
    logger.info(f"95% PI:              ${forecast_mu - 1.96 * forecast_sigma:.2f}"
                f" — ${forecast_mu + 1.96 * forecast_sigma:.2f}")

    # Save forecast to JSON
    forecast_dir = root / "outputs"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecast_data = {
        "last_trading_day": str(df.index[-1].date()),
        "last_close": round(last_close, 2),
        "forecast_date": str(forecast_date.date()),
        "point_forecast": round(forecast_mu, 2),
        "sigma": round(forecast_sigma, 2),
        "pi_95_lower": round(forecast_mu - 1.96 * forecast_sigma, 2),
        "pi_95_upper": round(forecast_mu + 1.96 * forecast_sigma, 2),
    }
    forecast_path = forecast_dir / "forecast.json"
    forecast_path.write_text(json.dumps(forecast_data, indent=2))
    logger.info(f"Saved forecast to {forecast_path}")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  - Figures:       outputs/figures/")
    logger.info("  - Models:        outputs/models/gam_models.pkl  (mu, sigma, nu, tau)")
    logger.info("  - Forecast:      outputs/forecast.json")
    logger.info("  - Run dashboard: streamlit run app/dashboard.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
