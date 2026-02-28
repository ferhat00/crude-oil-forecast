"""End-to-end pipeline runner for crude oil price forecasting.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-download
    python scripts/run_pipeline.py --skip-eda
    python scripts/run_pipeline.py --skip-download --skip-eda
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.data_acquisition import acquire_all
from src.data_engineering import build_features
from src.eda import run_eda
from src.evaluation import run_evaluation
from src.model import train_and_save


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
    gam, cv_results = train_and_save(config)

    # Step 5: Evaluation
    logger.info("=" * 60)
    logger.info("STEP 5: Model Evaluation")
    logger.info("=" * 60)
    run_evaluation(config)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  - Figures: outputs/figures/")
    logger.info("  - Model:   outputs/models/gam_model.pkl")
    logger.info("  - Run dashboard: streamlit run app/dashboard.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
