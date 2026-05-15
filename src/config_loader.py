"""Configuration loader for the crude oil forecasting pipeline."""

import logging
import os
from pathlib import Path

import yaml
from fredapi import Fred

logger = logging.getLogger(__name__)

# Hard-coded defaults matching the previous flat config behaviour.
_MODEL_DEFAULTS: dict = {
    "n_splines": 25,
    "lam_search": [0.001, 0.01, 0.1, 1, 10, 100],
    "cv_splits": 5,
    "embargo_days": 200,
    "stepwise_selection": True,
    "stepwise_criterion": "bic",
    "stepwise_max_steps": 50,
    "target_max_terms": 25,
    "n_jobs": -1,
    "sigma_stepwise": True,
    "sigma_max_terms": 12,
    "nu_tau_window": 60,
    "nu_max_terms": 8,
    "tau_max_terms": 8,
    # Forecast-quality knobs (see plan items 6, 8, 9, 10, 11, 5, 7)
    "collinearity_threshold": 0.97,
    "quantile_knots": True,
    "enable_monotonic": True,
    "enable_interactions": False,
    "interaction_n_splines": 6,
    "interaction_lam": 10.0,
    "iterative_refits": 1,
    "bagging": {"enabled": False, "n_bags": 10, "block_length": 21,
                "random_state": 0, "n_jobs": 1},
    "expectile_levels": [0.05, 0.50, 0.95],
    "conformal_alpha": 0.10,
}

_PRESET_KEYS = {"active_mode", "fast", "thorough"}


def resolve_model_config(config: dict) -> dict:
    """Resolve the ``model`` section into a flat dict, applying preset defaults.

    Supports two config styles:

    **Legacy (flat)** — no ``active_mode`` key; the ``model`` dict is returned
    as-is, which preserves full backward compatibility.

    **Preset-based** — ``active_mode`` selects a named preset block (``fast``
    or ``thorough``).  Resolution order (later wins):

        hard-coded defaults  <  shared keys  <  active preset block

    Returns:
        Flat dict usable via ``model_cfg.get(key, default)``.
    """
    model_section = config.get("model", {})
    active_mode = model_section.get("active_mode")

    # Legacy flat config — return as-is.
    if active_mode is None:
        return model_section

    if active_mode not in ("fast", "thorough"):
        raise ValueError(
            f"Unknown model.active_mode '{active_mode}'. "
            "Expected 'fast' or 'thorough'."
        )

    preset_block = model_section.get(active_mode, {})

    # Shared keys = everything in model_section except reserved preset keys.
    shared = {k: v for k, v in model_section.items() if k not in _PRESET_KEYS}

    # Merge: defaults < shared < preset
    resolved = {**_MODEL_DEFAULTS, **shared, **preset_block}
    resolved["_active_mode"] = active_mode

    logger.info("Model selection mode: %s", active_mode)
    return resolved


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
        with open(path, "r", encoding="utf-8") as f:
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
                "target_transform": "log_return",
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

    av_key = config.get("alpha_vantage", {}).get("api_key", "")
    if not av_key or av_key.startswith("YOUR_"):
        env_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if env_key:
            config.setdefault("alpha_vantage", {})["api_key"] = env_key

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


def get_alpha_vantage_params(config: dict) -> tuple[str, str, dict[str, str]]:
    """Return (base_url, api_key, series_map) for Alpha Vantage API calls."""
    av = config.get("alpha_vantage", {})
    api_key = av.get("api_key", "")
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "Alpha Vantage API key not configured. "
            "Set it in config.yaml or ALPHA_VANTAGE_API_KEY env var."
        )
    base_url = av.get("base_url", "https://www.alphavantage.co/query")
    series_map = av.get("series", {})
    return base_url, api_key, series_map


def get_eia_params(config: dict) -> tuple[str, str]:
    """Return (base_url, api_key) for EIA API calls."""
    api_key = config["eia"]["api_key"]
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "EIA API key not configured. Set it in config.yaml or EIA_API_KEY env var."
        )
    base_url = config["eia"].get("base_url", "https://api.eia.gov/v2")
    return base_url, api_key
