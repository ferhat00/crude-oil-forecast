# Crude Oil Price Forecasting with GAMs

End-to-end pipeline for forecasting crude oil prices using Generalized Additive Models (pyGAM). Combines multiple data sources (Yahoo Finance, EIA, FRED) with interpretable machine learning to produce price forecasts with interactive visualizations.

## Features

- **Multi-source data ingestion** — WTI/Brent prices (yfinance), US crude inventories & production (EIA API), macroeconomic indicators (FRED API)
- **Automated feature engineering** — lag variables, rolling statistics, calendar features, return metrics
- **Interpretable GAM modeling** — linear terms for macro factors, spline terms for nonlinear effects, cyclic splines for seasonality
- **Time series cross-validation** — proper temporal splitting with MAE/RMSE/MAPE evaluation
- **Interactive Streamlit dashboard** — partial dependence plots, what-if analysis, forecast visualization

## Prerequisites

- Python 3.11+
- [EIA API key](https://www.eia.gov/opendata/register.php) (free)
- [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)

## Setup

```bash
# Clone and enter the project
git clone https://github.com/your-username/crude-oil-forecast.git
cd crude-oil-forecast

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp config.example.yaml config.yaml
# Edit config.yaml with your EIA and FRED API keys
```

## Usage

### Run the full pipeline

```bash
python scripts/run_pipeline.py
```

Options:
- `--skip-download` — skip data acquisition (use existing data in `data/raw/`)
- `--skip-eda` — skip EDA plot generation

### Launch the dashboard

```bash
streamlit run app/dashboard.py
```

### Run tests

```bash
pytest tests/
```

## Project Structure

```
├── src/                    # Core library
│   ├── config_loader.py    # Configuration management
│   ├── data_acquisition.py # Data fetching (yfinance, FRED, EIA)
│   ├── data_engineering.py # Feature engineering pipeline
│   ├── eda.py              # Exploratory data analysis
│   ├── model.py            # GAM definition, training, cross-validation
│   └── evaluation.py       # Metrics, baselines, diagnostics
├── app/                    # Streamlit dashboard
│   ├── dashboard.py        # Main app entry point
│   └── components.py       # Plotly chart builders
├── scripts/
│   └── run_pipeline.py     # End-to-end CLI runner
├── tests/                  # Unit tests
├── data/                   # Raw and processed data (gitignored)
└── outputs/                # Figures and models (gitignored)
```

## License

MIT
