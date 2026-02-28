# CLAUDE.md — Project Conventions

## Project Overview
Crude oil price forecasting using Generalized Additive Models (pyGAM).
Predicts WTI crude oil futures (CL=F) closing price.

## Tech Stack
- Python 3.11+
- pyGAM (LinearGAM) for modeling
- yfinance, fredapi, EIA API v2 for data
- Streamlit + Plotly for dashboard
- pandas, numpy, scikit-learn, statsmodels

## Project Structure
- `src/` — core library modules (data_acquisition, data_engineering, eda, model, evaluation)
- `app/` — Streamlit dashboard (run with: `streamlit run app/dashboard.py`)
- `scripts/` — CLI pipeline runner
- `data/raw/` — downloaded data (gitignored)
- `data/processed/` — feature-engineered data (gitignored)
- `outputs/` — figures and serialized models (gitignored)
- `config.yaml` — API keys (gitignored); use `config.example.yaml` as template

## Key Conventions
- All data passes through pandas DataFrames with DatetimeIndex
- Feature matrix X is a numpy array; column-to-index mapping tracked via feature_names list
- GAM term assignment by feature name substring matching (see `src/model.py`)
- Parquet format for data persistence
- TimeSeriesSplit for all validation (never random split)

## Running
1. Copy `config.example.yaml` to `config.yaml` and fill in API keys
2. `python scripts/run_pipeline.py` — full pipeline
3. `streamlit run app/dashboard.py` — dashboard

## Testing
`pytest tests/`
