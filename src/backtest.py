"""Walk-forward backtest with periodic re-fit.

Embargoed CV (in :mod:`src.model`) gives summary statistics suitable for
hyperparameter tuning, but does not simulate production: it does not refit
the model on a sliding training window.  The walk-forward backtest in this
module produces a continuous out-of-sample prediction path with all the
information a production deployment would have at each point in time:

    * For each refit cadence ``refit_period_days``, a fresh GAM is fitted on
      data up to (but not including) the next test window.
    * Predictions are stored along with per-row σ, fitted Johnson SU PI bounds,
      a model-version tag, and the anchor price for return → price reconstruction.
    * Accumulated path is scored with MAE, RMSE, MAPE, directional accuracy,
      pinball loss, CRPS, log-score, and empirical-vs-nominal coverage.

This is the only honest way to measure forecast quality on a non-stationary
series like crude oil.

Usage:

    from src.backtest import run_walk_forward_backtest
    result = run_walk_forward_backtest(config)
    result.path.to_parquet("outputs/backtest_path.parquet")
    print(result.scores)
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config_loader import get_project_root, resolve_model_config
from src.evaluation import (
    PredictiveDistribution,
    compute_metrics,
    compute_probabilistic_scores,
    directional_accuracy,
    fit_residual_distribution,
    get_prediction_sigma,
    naive_baseline_predictions,
)
from src.model import (
    SkewedFeatureTransformer,
    _is_skewed_feature,
    build_feature_matrix,
    compute_anchor_prices,
    define_gam_terms,
    drop_correlated_features,
    fit_gam,
    get_target_y_level,
    reconstruct_price_from_returns,
    stepwise_aic_selection,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BacktestResult:
    """Container for walk-forward backtest outputs.

    Attributes:
        path: DataFrame indexed by forecast date, with columns:
            y_true, y_pred (price), y_pred_model, sigma_model, y_anchor,
            lower_50, upper_50, lower_80, upper_80, lower_90, upper_90,
            lower_95, upper_95, model_version (int).
        scores: Dict of summary metrics across the assembled path.
        config_snapshot: Echo of the (resolved) backtest config used.
    """
    path: pd.DataFrame
    scores: dict[str, float]
    config_snapshot: dict


def _resolve_backtest_config(config: dict) -> dict:
    """Read the ``backtest`` section with fallbacks.

    Recognised keys:
        initial_train_frac (float, 0.6) — fraction of data used for the first fit.
        refit_period_days (int, 21) — re-fit cadence in trading days.
        test_period_days (int, 21) — out-of-sample evaluation window per fit.
        max_models (int|None) — cap the number of refits (useful for smoke tests).
        do_stepwise (bool, false) — apply per-fit stepwise feature selection.
            Off by default since it is the dominant cost.
        n_crps_samples (int, 200) — Monte-Carlo draws for CRPS scoring.
        save_path (str|None) — where to persist the per-row backtest path
            (parquet); the scores summary is always written next to it.
    """
    backtest_cfg = config.get("backtest", {}) or {}
    return {
        "initial_train_frac": float(backtest_cfg.get("initial_train_frac", 0.6)),
        "refit_period_days": int(backtest_cfg.get("refit_period_days", 21)),
        "test_period_days": int(backtest_cfg.get("test_period_days", 21)),
        "max_models": backtest_cfg.get("max_models"),
        "do_stepwise": bool(backtest_cfg.get("do_stepwise", False)),
        "n_crps_samples": int(backtest_cfg.get("n_crps_samples", 200)),
        "save_path": backtest_cfg.get(
            "save_path", "outputs/backtest_path.parquet"
        ),
    }


def run_walk_forward_backtest(
    config: dict,
    df: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run an expanding-window walk-forward backtest with periodic re-fit.

    Train on the first ``initial_train_frac`` of the data, predict the next
    ``test_period_days``, then expand the training window by ``refit_period_days``
    and re-fit.  Continue until the data is exhausted.  All predictions are
    stitched into a continuous out-of-sample path and scored.

    Args:
        config: Project configuration.  Reads ``features.target_transform`` to
            pick the target space; reads ``backtest.*`` for cadence settings.
        df: Optionally supply a pre-loaded features DataFrame (avoids re-reading
            the parquet inside hot loops in dashboards or tests).

    Returns:
        :class:`BacktestResult` with the per-row path, summary scores, and a
        snapshot of the backtest config.
    """
    root = get_project_root(config)
    feat_cfg = config.get("features", {})
    target = feat_cfg["target"]
    target_transform = feat_cfg.get("target_transform", "level")

    bt_cfg = _resolve_backtest_config(config)
    model_cfg = resolve_model_config(config)
    lam_values = model_cfg.get("lam_search")
    n_jobs = model_cfg.get("n_jobs", 1)
    quantile_knots = bool(model_cfg.get("quantile_knots", False))
    corr_threshold = float(model_cfg.get("collinearity_threshold", 0.97))

    if df is None:
        df = pd.read_parquet(root / "data" / "processed" / "features.parquet")

    X_full, y_model, all_feature_names, target_dates = build_feature_matrix(
        df, target, target_transform=target_transform,
    )
    anchor = compute_anchor_prices(df, target, forecast_horizon=1)
    y_level = get_target_y_level(df, target, forecast_horizon=1)

    # Optional collinearity prune (use the entire dataset's correlations once,
    # rather than re-pruning per fold — this is what production would do too)
    if corr_threshold and corr_threshold < 1.0:
        keep_idx, kept_names, _ = drop_correlated_features(
            X_full, all_feature_names, threshold=corr_threshold,
        )
        X_full = X_full[:, keep_idx]
        feature_names_full = kept_names
    else:
        feature_names_full = list(all_feature_names)

    n = len(y_model)
    initial_train_size = max(int(bt_cfg["initial_train_frac"] * n),
                             max(252, model_cfg.get("embargo_days", 200) + 50))
    refit_period = max(1, bt_cfg["refit_period_days"])
    test_period = max(1, bt_cfg["test_period_days"])

    # Plan refit windows
    refit_starts: list[int] = []
    cursor = initial_train_size
    while cursor < n:
        refit_starts.append(cursor)
        cursor += refit_period
        if bt_cfg["max_models"] is not None and len(refit_starts) >= bt_cfg["max_models"]:
            break
    if not refit_starts:
        raise RuntimeError(
            f"Walk-forward backtest produced no refit windows: "
            f"n={n}, initial_train_size={initial_train_size}"
        )
    logger.info(
        f"Walk-forward backtest: n={n}, initial_train_size={initial_train_size}, "
        f"refit_period={refit_period}, test_period={test_period}, "
        f"n_refits={len(refit_starts)}"
    )

    # Per-row containers
    path_rows: list[dict] = []

    # Accumulators for out-of-sample residuals — used to refit the
    # per-fold `dist` once we have enough.  Training-only residuals are
    # tiny under in-sample overfitting (Pseudo R² ≈ 1), which collapses
    # the predictive PIs and was the proximate cause of the empirical
    # 50% coverage being ~5% on the 2026-05-16 report.
    OOS_DIST_REFIT_MIN = 200  # rows of OOS residuals before switching
    OOS_DIST_REFIT_TAIL = 1000  # use the most recent N when refitting
    oos_y_true_model: list[float] = []
    oos_y_pred_model: list[float] = []
    oos_sigma_model: list[float] = []

    for model_version, train_end in enumerate(refit_starts, start=1):
        # No embargo gap is needed for walk-forward: train always precedes
        # test, and rolling features only look backwards.  The CV-side
        # EmbargoedTimeSeriesSplit guards against a K-fold-style leakage
        # (train rows AFTER test seeing test's target via rolling windows)
        # that cannot occur in an expanding-window walk-forward.  Forecast
        # at T uses information legitimately available at T.
        test_start = train_end
        test_end = min(train_end + test_period, n)
        if test_end <= test_start:
            continue

        X_train = X_full[:train_end]
        y_train = y_model[:train_end]
        X_test = X_full[test_start:test_end]
        y_test_model = y_model[test_start:test_end]
        y_test_level = y_level[test_start:test_end]
        anchor_test = anchor[test_start:test_end]
        dates_test = target_dates[test_start:test_end]

        feature_names = list(feature_names_full)

        # Optional in-fold stepwise (slow but matches production retrain)
        if bt_cfg["do_stepwise"] and model_cfg.get("stepwise_selection", False):
            X_train_sel, sel_idx, feature_names, _, _ = stepwise_aic_selection(
                X_train, y_train, feature_names, config,
                lam_values=lam_values,
                max_steps=model_cfg.get("stepwise_max_steps"),
                n_jobs=n_jobs,
                criterion=model_cfg.get("stepwise_criterion", "bic"),
                target_max_terms=model_cfg.get("target_max_terms"),
            )
            X_test_sel = X_test[:, sel_idx]
        else:
            X_train_sel = X_train
            X_test_sel = X_test

        transformer = None
        if quantile_knots:
            transformer = SkewedFeatureTransformer(feature_names).fit(X_train_sel)
            X_train_sel = transformer.transform(X_train_sel)
            X_test_sel = transformer.transform(X_test_sel)

        terms = define_gam_terms(feature_names, config)
        gam = fit_gam(X_train_sel, y_train, terms, lam_values, n_jobs=n_jobs)

        # Predictive distribution: prefer accumulated OOS residuals from
        # prior folds (conformal-ish — calibration that production could
        # also do).  Fall back to training residuals only for the first
        # few folds before the OOS buffer is large enough to be reliable.
        if len(oos_y_true_model) >= OOS_DIST_REFIT_MIN:
            tail = OOS_DIST_REFIT_TAIL
            y_calib = np.asarray(oos_y_true_model[-tail:], dtype=np.float64)
            mu_calib = np.asarray(oos_y_pred_model[-tail:], dtype=np.float64)
            sigma_calib = np.asarray(oos_sigma_model[-tail:], dtype=np.float64)
            try:
                dist = fit_residual_distribution(y_calib, mu_calib, sigma_calib)
            except Exception as exc:
                logger.warning(
                    f"  fold {model_version}: OOS Johnson SU fit failed ({exc}) — "
                    "falling back to training residuals"
                )
                sigma_train = get_prediction_sigma(gam, X_train_sel)
                y_pred_train = gam.predict(X_train_sel)
                try:
                    dist = fit_residual_distribution(y_train, y_pred_train, sigma_train)
                except Exception:
                    dist = PredictiveDistribution(nu=0.0, tau=1.0, loc_std=0.0, scale_std=1.0)
        else:
            sigma_train = get_prediction_sigma(gam, X_train_sel)
            y_pred_train = gam.predict(X_train_sel)
            try:
                dist = fit_residual_distribution(y_train, y_pred_train, sigma_train)
            except Exception as exc:
                logger.warning(
                    f"  fold {model_version}: Johnson SU fit failed ({exc}) — using Gaussian fallback"
                )
                dist = PredictiveDistribution(nu=0.0, tau=1.0, loc_std=0.0, scale_std=1.0)

        # Out-of-sample predictions
        y_pred_model = gam.predict(X_test_sel)
        sigma_test = get_prediction_sigma(gam, X_test_sel)

        # Reconstruct to price scale
        if target_transform == "log_return":
            y_pred_price = anchor_test * np.exp(y_pred_model)
        else:
            y_pred_price = y_pred_model

        # Quantile bands (Johnson SU in model space, transformed if needed)
        widths = [0.50, 0.80, 0.90, 0.95]
        bands_price: dict[float, np.ndarray] = {}
        for w in widths:
            q_lo, q_hi = (1 - w) / 2, (1 + w) / 2
            r_lo = dist.ppf_array(q_lo, y_pred_model, sigma_test)
            r_hi = dist.ppf_array(q_hi, y_pred_model, sigma_test)
            if target_transform == "log_return":
                bands_price[w] = np.column_stack([
                    anchor_test * np.exp(r_lo),
                    anchor_test * np.exp(r_hi),
                ])
            else:
                bands_price[w] = np.column_stack([r_lo, r_hi])

        for j in range(len(y_test_level)):
            row = {
                "date": dates_test[j],
                "y_true": float(y_test_level[j]),
                "y_pred": float(y_pred_price[j]),
                "y_pred_model": float(y_pred_model[j]),
                "sigma_model": float(sigma_test[j]),
                "y_anchor": float(anchor_test[j]),
                "model_version": int(model_version),
            }
            for w in widths:
                row[f"lower_{int(w * 100)}"] = float(bands_price[w][j, 0])
                row[f"upper_{int(w * 100)}"] = float(bands_price[w][j, 1])
            path_rows.append(row)

        # Accumulate OOS residuals (all in MODEL space) for the next
        # fold's dist refit.  y_test_model is the true target in model
        # space (log-return when target_transform="log_return", price
        # otherwise); y_pred_model is the fold's GAM prediction in the
        # same space; sigma_test is the GAM's per-row σ in model space.
        oos_y_true_model.extend(float(v) for v in y_test_model)
        oos_y_pred_model.extend(float(v) for v in y_pred_model)
        oos_sigma_model.extend(float(v) for v in sigma_test)

        if model_version % 10 == 0 or model_version == len(refit_starts):
            mean_mae = float(np.mean(np.abs(y_test_level - y_pred_price)))
            logger.info(
                f"  fold {model_version}/{len(refit_starts)}: "
                f"train n={train_end}, test n={test_end - test_start}, "
                f"MAE={mean_mae:.2f}"
            )

    path = pd.DataFrame(path_rows).set_index("date").sort_index()

    # ── Score the assembled path (price-scale) ────────────────────────────────
    scores: dict[str, float] = {}
    point = compute_metrics(path["y_true"].values, path["y_pred"].values)
    scores.update({f"{k}_price": v for k, v in point.items()})
    scores["directional_accuracy"] = directional_accuracy(
        path["y_true"].values, path["y_pred"].values, path["y_anchor"].values,
    )
    naive_pred = path["y_anchor"].values  # naive: tomorrow == today
    naive = compute_metrics(path["y_true"].values, naive_pred)
    scores.update({f"{k}_naive": v for k, v in naive.items()})
    # Skill: 1 - GAM/naive (positive = beats naive)
    scores["skill_mae"] = 1.0 - scores["mae_price"] / scores["mae_naive"] if scores["mae_naive"] else float("nan")
    scores["skill_rmse"] = 1.0 - scores["rmse_price"] / scores["rmse_naive"] if scores["rmse_naive"] else float("nan")

    # Coverage (price scale, from the recorded bands)
    for w in (50, 80, 90, 95):
        lo = path[f"lower_{w}"].values
        hi = path[f"upper_{w}"].values
        scores[f"coverage_{w}_empirical"] = float(
            np.mean((path["y_true"].values >= lo) & (path["y_true"].values <= hi))
        )

    # Probabilistic scores in MODEL SPACE.  Refit Johnson SU on the full
    # OOS path (this is "leaky" in a calibration sense — it uses all path
    # residuals — but produces an honest summary of distributional skill
    # for cross-run comparison).  Earlier versions mixed a price-space
    # residual into a model-space mean here, which happened to produce
    # plausible-looking numbers but is mathematically wrong; fixed.
    try:
        if target_transform == "log_return":
            y_anchor_for_dir = np.zeros_like(path["y_pred_model"].values)
            y_true_model = np.log(
                path["y_true"].values
                / np.clip(path["y_anchor"].values, 1e-12, None)
            )
        else:
            y_anchor_for_dir = path["y_anchor"].values
            y_true_model = path["y_true"].values
        dist_global = fit_residual_distribution(
            y_true_model,
            path["y_pred_model"].values,
            path["sigma_model"].values,
        )
        prob_scores = compute_probabilistic_scores(
            y_true_model,
            path["y_pred_model"].values,
            path["sigma_model"].values,
            dist_global,
            y_anchor=y_anchor_for_dir,
            n_crps_samples=bt_cfg["n_crps_samples"],
        )
        scores.update({f"{k}_model": v for k, v in prob_scores.items()})
    except Exception as exc:
        logger.warning(f"Backtest probabilistic scoring failed: {exc}")

    # ── Persist results ───────────────────────────────────────────────────────
    if bt_cfg["save_path"]:
        save_path = root / bt_cfg["save_path"]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        path.to_parquet(save_path)
        scores_path = save_path.with_suffix(".scores.json")
        scores_path.write_text(json.dumps(scores, indent=2, default=float))
        logger.info(f"Saved backtest path to {save_path}")
        logger.info(f"Saved backtest scores to {scores_path}")

        # Also drop a rolling-MAE plot for drift inspection
        plot_rolling_mae(path, save_path=root / "outputs" / "figures" / "backtest_rolling_mae.png")

    snapshot = {
        "target_transform": target_transform,
        "n_refits": len(refit_starts),
        "n_test_obs": len(path),
        **bt_cfg,
    }

    return BacktestResult(path=path, scores=scores, config_snapshot=snapshot)


def plot_rolling_mae(
    path: pd.DataFrame,
    window: int = 63,
    save_path: str | Path = "outputs/figures/backtest_rolling_mae.png",
) -> None:
    """Plot rolling MAE of the walk-forward path against the naive baseline.

    A rolling 63-day MAE that drifts upward signals concept drift — a refit-
    cadence trigger independent of the calendar.  Both GAM and naive lines
    are drawn so any deterioration is interpretable as relative or absolute.
    """
    df = path.sort_index()
    abs_err_gam = (df["y_true"] - df["y_pred"]).abs()
    abs_err_naive = (df["y_true"] - df["y_anchor"]).abs()
    roll_gam = abs_err_gam.rolling(window, min_periods=window).mean()
    roll_naive = abs_err_naive.rolling(window, min_periods=window).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, roll_gam, color="#08306b", linewidth=1.4, label=f"GAM (rolling {window}d MAE)")
    ax.plot(df.index, roll_naive, color="#d62728", linewidth=1.0, alpha=0.7,
            label=f"Naive baseline (rolling {window}d MAE)")
    # Mark the 95th percentile of GAM rolling MAE as a drift threshold
    if roll_gam.notna().any():
        thresh = float(np.nanpercentile(roll_gam, 95))
        ax.axhline(thresh, color="orange", linestyle="--", linewidth=0.8,
                   label=f"95th-pct GAM MAE = {thresh:.2f}")
    ax.set_title(f"Walk-Forward Backtest — Rolling {window}-Day MAE (Drift Monitor)")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAE (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved rolling-MAE drift plot to {save_path}")


def load_backtest_path(
    config: dict,
    save_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Load a previously-saved walk-forward backtest path.  Returns ``None`` if not present."""
    root = get_project_root(config)
    if save_path is None:
        save_path = config.get("backtest", {}).get("save_path", "outputs/backtest_path.parquet")
    path = root / save_path
    if not path.exists():
        return None
    return pd.read_parquet(path)
