"""Combinatorial Purged Cross-Validation backtester (Lopez de Prado, Ch. 12).

The walk-forward backtester in :mod:`src.backtest` produces a *single*
historical out-of-sample path.  CPCV instead generates many distinct
out-of-sample paths by splitting the sample into ``N`` contiguous groups,
holding out every ``C(N, k)`` combination of ``k`` groups, and reassembling
``phi(N, k) = k * C(N, k) / N`` independent paths from the grid of
predictions.  Each path covers the full sample, giving us a *distribution*
of backtest statistics rather than a single point estimate.

This module is intentionally GAM-agnostic: callers pass a ``model_factory``
callable that returns an object exposing ``.fit(X, y)`` and ``.predict(X)``.
For the standard pipeline use :func:`make_gam_factory` to wire it up to
:func:`src.model.fit_gam`.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
Chapter 12, Section 12.4 (Combinatorial Purged Cross-Validation).
Chapter 11, Section 11.6 (Probability of Backtest Overfitting via CSCV).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.cv import CombinatorialPurgedCV

logger = logging.getLogger(__name__)

ANNUALISATION = 252.0  # trading days per year


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CPCVResult:
    """Outputs of :func:`run_cpcv_backtest`.

    Attributes:
        path_metrics: DataFrame, one row per CPCV path, with columns
            ``sharpe``, ``annualised_return``, ``annualised_vol``,
            ``max_drawdown``, ``hit_rate``, ``n_obs``.
        path_returns: DataFrame indexed by date, one column per path,
            holding each path's signed-direction return series.
        summary: Aggregate statistics across paths (mean / std / percentiles
            of Sharpe; probability-of-overfit proxy).
        config_snapshot: Echo of the resolved configuration used.
    """
    path_metrics: pd.DataFrame
    path_returns: pd.DataFrame
    summary: dict
    config_snapshot: dict


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe (mean / std × √252).  Zero-vol → 0.0."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(ANNUALISATION))


def _max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough drawdown on a cumulative-return curve (negative)."""
    e = np.asarray(equity, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    running_max = np.maximum.accumulate(e)
    drawdowns = e - running_max
    return float(drawdowns.min())


def _path_metrics(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {
            "sharpe": 0.0, "annualised_return": 0.0, "annualised_vol": 0.0,
            "max_drawdown": 0.0, "hit_rate": 0.0, "n_obs": 0,
        }
    equity = np.cumsum(r)
    return {
        "sharpe": _sharpe(r),
        "annualised_return": float(r.mean() * ANNUALISATION),
        "annualised_vol": float(r.std(ddof=1) * np.sqrt(ANNUALISATION)) if len(r) > 1 else 0.0,
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": float((r > 0).mean()),
        "n_obs": int(len(r)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Direction-strategy return construction
# ─────────────────────────────────────────────────────────────────────────────

def _direction_returns(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_anchor: np.ndarray,
) -> np.ndarray:
    """Signed-direction strategy log-return series.

    For each observation t::

        position_t = sign(y_pred_t - y_anchor_t)         # long / short / flat
        realised_t = log(y_true_t / y_anchor_t)
        strategy_t = position_t * realised_t

    This is the simplest "is the forecast useful?" metric and matches the
    walk-forward backtest's directional accuracy in spirit.
    """
    direction = np.sign(y_pred - y_anchor)
    with np.errstate(divide="ignore", invalid="ignore"):
        realised = np.log(np.clip(y_true, 1e-12, None) / np.clip(y_anchor, 1e-12, None))
    return direction * realised


# ─────────────────────────────────────────────────────────────────────────────
# Combination worker (top-level → picklable for joblib/loky)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_combination(
    combo_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object],
) -> dict:
    """Fit a fresh model on ``train_idx`` and predict ``test_idx``.

    Returns a dict with the combination id, test positional indices, and
    the predicted target values.  Designed to be safe under joblib/loky.
    """
    model = model_factory()
    model.fit(X[train_idx], y[train_idx])
    y_pred = np.asarray(model.predict(X[test_idx]), dtype=np.float64)
    return {
        "combo_id": combo_id,
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Path assembly
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_paths(
    n_samples: int,
    n_splits: int,
    cv: CombinatorialPurgedCV,
    predictions_by_combo: dict[int, np.ndarray],
) -> np.ndarray:
    """Combine per-combination predictions into ``phi(N, k)`` paths.

    Returns:
        Array of shape ``(n_paths, n_samples)`` whose row ``p`` is the full
        OOS prediction series of path ``p``.  Cells without a prediction
        (cannot occur for valid path assignment) are NaN.
    """
    # paths[p, g] = combo_id whose test-prediction on group g belongs to path p
    paths = cv.get_paths(n_samples)
    n_paths, n_groups = paths.shape

    # Pre-compute the group's positional slice (matches CombinatorialPurgedCV)
    group_bounds = [np.asarray(g) for g in np.array_split(np.arange(n_samples), n_splits)]

    all_paths = np.full((n_paths, n_samples), np.nan, dtype=np.float64)
    for p in range(n_paths):
        for g in range(n_groups):
            ci = int(paths[p, g])
            if ci < 0 or ci not in predictions_by_combo:
                continue
            preds_for_combo = predictions_by_combo[ci]   # full-length, NaN outside test
            group_slice = group_bounds[g]
            all_paths[p, group_slice] = preds_for_combo[group_slice]
    return all_paths


# ─────────────────────────────────────────────────────────────────────────────
# Probability of Backtest Overfit (single-strategy proxy)
# ─────────────────────────────────────────────────────────────────────────────

def _pbo_proxy(path_sharpes: np.ndarray) -> float:
    """Single-strategy CPCV overfit proxy.

    The book's CSCV PBO (Bailey et al., 2014) is defined for *multiple*
    candidate strategies: it asks how often the best in-sample strategy
    underperforms the median out-of-sample.  With a single model trained
    on multiple paths there is no second strategy to rank against, so we
    report the conservative proxy

        PBO ≈ P(path Sharpe ≤ 0)

    estimated as the fraction of CPCV paths whose Sharpe is non-positive.
    A high value means most paths produced unprofitable strategies — the
    historical Sharpe is unlikely to repeat out-of-sample.
    """
    s = np.asarray(path_sharpes, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return float("nan")
    return float((s <= 0.0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def make_gam_factory(
    terms_factory: Callable[[], object],
    lam_values: list[float] | None = None,
) -> Callable[[], object]:
    """Return a model_factory closure that calls :func:`src.model.fit_gam`.

    Each invocation returns an object exposing ``.fit(X, y)`` and
    ``.predict(X)``.  ``terms_factory`` returns a fresh pyGAM terms
    expression — required because pyGAM terms are mutated by fitting.
    """
    from pygam import LinearGAM
    from src.model import fit_gam

    class _Wrapped:
        def __init__(self) -> None:
            self._gam: LinearGAM | None = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "_Wrapped":
            terms = terms_factory()
            self._gam = fit_gam(X, y, terms, lam_values=lam_values, n_jobs=1)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            assert self._gam is not None, "fit must be called before predict"
            return np.asarray(self._gam.predict(X), dtype=np.float64)

    return _Wrapped


def run_cpcv_backtest(
    X: np.ndarray,
    y: np.ndarray,
    t1: pd.Series,
    model_factory: Callable[[], object],
    *,
    anchor: np.ndarray,
    y_level: np.ndarray,
    target_dates: pd.DatetimeIndex,
    n_splits: int = 6,
    n_test_splits: int = 2,
    pct_embargo: float = 0.01,
    output_dir: Path | str | None = None,
    n_jobs: int = 1,
) -> CPCVResult:
    """Run Combinatorial Purged Cross-Validation and assemble path metrics.

    Args:
        X, y: Feature matrix and target (model space — may be levels or
            log-returns; the path-return computation only needs the
            *price-level* arrays below).
        t1: Label end-time series (see :func:`src.model.build_feature_matrix`).
        model_factory: Callable returning a fresh fit-ready model per call.
            Use :func:`make_gam_factory` for the GAM pipeline.
        anchor: Anchor prices ``P_t`` aligned with each forecast row.
        y_level: Realised price levels on the forecast dates (used to score
            the direction strategy regardless of the model's target space).
        target_dates: DatetimeIndex of forecast dates, length ``n_samples``.
        n_splits: Number of contiguous groups (``N``).
        n_test_splits: Test-group count per combination (``k``).  ``k=2``
            is the book's default; gives ``C(N, 2)`` combinations.
        pct_embargo: Fraction of total observations to embargo per fold.
        output_dir: If given, writes ``paths_metrics.parquet``,
            ``paths_returns.parquet``, and ``summary.json`` into it.
        n_jobs: Parallelism across combinations (passed to joblib).

    Returns:
        :class:`CPCVResult` with per-path metrics, the path return series,
        and an aggregate summary including the PBO proxy.
    """
    n_samples = len(y)
    if len(t1) != n_samples or len(anchor) != n_samples or len(y_level) != n_samples:
        raise ValueError(
            "X, y, t1, anchor, y_level must all share the same length "
            f"(got {len(X)}, {n_samples}, {len(t1)}, {len(anchor)}, {len(y_level)})."
        )

    cv = CombinatorialPurgedCV(
        n_splits=n_splits, n_test_splits=n_test_splits,
        t1=t1, pct_embargo=pct_embargo,
    )

    n_combos = cv.get_n_combinations()
    n_paths = cv.get_n_paths()
    logger.info(
        f"CPCV: N={n_splits}, k={n_test_splits} → {n_combos} combinations, "
        f"{n_paths} assembled paths (embargo={pct_embargo:.1%}, n_jobs={n_jobs})"
    )

    # ── Fit each combination ────────────────────────────────────────────────
    combo_jobs = list(cv.split(X))  # materialise so joblib can dispatch them all
    raw_results: list[dict] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_combination)(
            combo_id, train_idx, test_idx, X, y, model_factory,
        )
        for train_idx, test_idx, combo_id in combo_jobs
    )

    # Materialise per-combination prediction vectors (NaN outside test rows).
    predictions_by_combo: dict[int, np.ndarray] = {}
    for r in raw_results:
        ci = int(r["combo_id"])
        preds_full = np.full(n_samples, np.nan, dtype=np.float64)
        preds_full[r["test_idx"]] = r["y_pred"]
        predictions_by_combo[ci] = preds_full

    # ── Assemble paths ──────────────────────────────────────────────────────
    paths_pred = _assemble_paths(n_samples, n_splits, cv, predictions_by_combo)

    # ── Compute strategy returns per path ───────────────────────────────────
    # The model may predict either price level or log-return; either way,
    # the direction strategy needs a price-space forecast.  If y_level == y
    # (level mode) the model already predicts a price; if y is log-returns,
    # paths_pred is a log-return forecast — reconstruct the implied price via
    # anchor * exp(pred) before computing direction.
    is_log_return_mode = not np.allclose(np.asarray(y), y_level, equal_nan=True)
    if is_log_return_mode:
        paths_pred_price = np.asarray(anchor)[None, :] * np.exp(paths_pred)
    else:
        paths_pred_price = paths_pred

    path_returns_arr = np.full_like(paths_pred, np.nan, dtype=np.float64)
    for p in range(paths_pred.shape[0]):
        mask = ~np.isnan(paths_pred_price[p])
        if not mask.any():
            continue
        path_returns_arr[p, mask] = _direction_returns(
            paths_pred_price[p, mask], y_level[mask], anchor[mask],
        )

    path_returns = pd.DataFrame(
        path_returns_arr.T,
        index=target_dates,
        columns=[f"path_{p}" for p in range(paths_pred.shape[0])],
    )

    # ── Per-path metrics + aggregate summary ────────────────────────────────
    rows = []
    for p in range(paths_pred.shape[0]):
        m = _path_metrics(path_returns_arr[p])
        m["path"] = p
        rows.append(m)
    path_metrics = pd.DataFrame(rows).set_index("path")

    sharpes = path_metrics["sharpe"].to_numpy()
    summary = {
        "n_paths": int(paths_pred.shape[0]),
        "n_combinations": int(n_combos),
        "n_splits": int(n_splits),
        "n_test_splits": int(n_test_splits),
        "pct_embargo": float(pct_embargo),
        "sharpe_mean": float(np.nanmean(sharpes)),
        "sharpe_std": float(np.nanstd(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        "sharpe_median": float(np.nanmedian(sharpes)),
        "sharpe_p05": float(np.nanpercentile(sharpes, 5)),
        "sharpe_p95": float(np.nanpercentile(sharpes, 95)),
        "pbo": _pbo_proxy(sharpes),
    }

    config_snapshot = {
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "pct_embargo": pct_embargo,
        "n_jobs": n_jobs,
        "n_samples": n_samples,
    }

    result = CPCVResult(
        path_metrics=path_metrics,
        path_returns=path_returns,
        summary=summary,
        config_snapshot=config_snapshot,
    )

    # ── Optional persistence ────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path_metrics.to_parquet(out / "paths_metrics.parquet")
        path_returns.to_parquet(out / "paths_returns.parquet")
        with open(out / "summary.json", "w", encoding="utf-8") as fh:
            json.dump({**summary, "config": config_snapshot}, fh, indent=2)
        logger.info(f"CPCV results written to {out}")

    logger.info(
        f"CPCV done: Sharpe mean={summary['sharpe_mean']:.3f} ± {summary['sharpe_std']:.3f}, "
        f"PBO proxy={summary['pbo']:.2%}"
    )
    return result
