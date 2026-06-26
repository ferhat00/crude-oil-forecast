"""Scratch checkpoint: rebuild features and run the Phase-0 gate for each
sigma_source (gam/garch/blend).  Deleted after the checkpoint."""
import json
import logging
import sys

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from src.config_loader import load_config
from src.data_engineering import build_features
from src.backtest import run_walk_forward_backtest

cfg = load_config()
print("== Building features ==", flush=True)
df = build_features(cfg)
print(f"features shape: {df.shape}", flush=True)

NEW_KEYS = ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang",
            "seasonal_", "_surprise", "cot_mm_net_chg", "cot_commercial_net_chg",
            "gpr_", "gecon", "igrea")
new_cols = [c for c in df.columns if any(k in c for k in NEW_KEYS)]
print(f"new feature columns ({len(new_cols)}):", flush=True)
for c in new_cols:
    print("   ", c, flush=True)

GATE = ["mspe_ratio", "dm_pvalue", "directional_accuracy", "pt_pvalue",
        "crps_mean_model", "log_score_mean_model", "berkowitz_pvalue_model",
        "coverage_95_empirical", "skill_rmse"]

# Don't clobber the saved backtest artifacts during the comparison.
cfg.setdefault("backtest", {})["save_path"] = None

results = {}
for src_ in ("gam", "garch", "blend"):
    cfg["model"]["sigma_source"] = src_
    print(f"== Backtest sigma_source={src_} ==", flush=True)
    res = run_walk_forward_backtest(cfg, df=df)
    results[src_] = {k: res.scores.get(k) for k in GATE}
    print(f"   done ({len(res.path)} OOS obs)", flush=True)


def _fmt(v):
    return round(float(v), 5) if isinstance(v, (int, float)) and v == v else None


print("\n== GATE COMPARISON ==", flush=True)
print(json.dumps({s: {k: _fmt(v) for k, v in d.items()} for s, d in results.items()},
                 indent=2), flush=True)
