"""Configuration loader for the crude oil forecasting pipeline."""

import os
from pathlib import Path

import yaml
from fredapi import Fred


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (where config.yaml lives)."""
    current = Path(__file__).resolve().parent.parent
    return current


def load_config(path: str | None = None) -> dict:
    """Load YAML config file and return as dictionary.

    Falls back to environment variables EIA_API_KEY and FRED_API_KEY
    if config.yaml is absent or keys are placeholder values.
    """
    if path is None:
        path = _find_project_root() / "config.yaml"
    else:
        path = Path(path)

    if path.exists():
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Minimal config from environment variables
        config = {
            "eia": {
                "api_key": os.environ.get("EIA_API_KEY", ""),
                "base_url": "https://api.eia.gov/v2",
            },
            "fred": {
                "api_key": os.environ.get("FRED_API_KEY", ""),
            },
            "data": {
                "start_date": "2015-01-01",
                "end_date": None,
                "oil_tickers": ["CL=F", "BZ=F"],
                "fred_series": {
                    "usd_index": "DTWEXBGS",
                    "cpi": "CPIAUCSL",
                    "fed_funds": "DFF",
                    "t10y2y": "T10Y2Y",
                },
                "eia_series": {
                    "crude_stocks": "PET.WTTSTUS1.W",
                    "production": "PET.MCRFPUS2.M",
                },
            },
            "features": {
                "lag_days": [1, 7, 30],
                "rolling_windows": [14, 50],
                "target": "CL=F_close",
            },
            "model": {
                "n_splines": 25,
                "lam_search": [0.001, 0.01, 0.1, 1, 10, 100],
                "cv_splits": 5,
            },
        }

    # Override API keys from environment if config has placeholders
    eia_key = config.get("eia", {}).get("api_key", "")
    if not eia_key or eia_key.startswith("YOUR_"):
        env_key = os.environ.get("EIA_API_KEY", "")
        if env_key:
            config.setdefault("eia", {})["api_key"] = env_key

    fred_key = config.get("fred", {}).get("api_key", "")
    if not fred_key or fred_key.startswith("YOUR_"):
        env_key = os.environ.get("FRED_API_KEY", "")
        if env_key:
            config.setdefault("fred", {})["api_key"] = env_key

    # Inject project root for path resolution
    config["_project_root"] = str(_find_project_root())

    return config


def get_project_root(config: dict) -> Path:
    """Return the project root path from config."""
    return Path(config["_project_root"])


def get_fred_client(config: dict) -> Fred:
    """Return an initialized fredapi.Fred instance."""
    api_key = config["fred"]["api_key"]
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "FRED API key not configured. Set it in config.yaml or FRED_API_KEY env var."
        )
    return Fred(api_key=api_key)


def get_eia_params(config: dict) -> tuple[str, str]:
    """Return (base_url, api_key) for EIA API calls."""
    api_key = config["eia"]["api_key"]
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "EIA API key not configured. Set it in config.yaml or EIA_API_KEY env var."
        )
    base_url = config["eia"].get("base_url", "https://api.eia.gov/v2")
    return base_url, api_key
