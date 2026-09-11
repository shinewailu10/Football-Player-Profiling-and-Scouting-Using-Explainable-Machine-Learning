"""
train_supervised.py  (v2 — uses 5-fold cross-validation for stable metrics)
--------------------------------------------------------------------------
  a single 80/20 train/test split. This proved unstable: R² varied
  wildly depending on whether extreme-outlier players (Yamal at 17 = €200m,
  Mbappé, Vinicius) fell in the test set. On heavy-tailed target distributions
  a single split gives a misleading estimate.
  uses 5-fold cross-validation. Every player is predicted exactly once as
  part of a held-out fold, giving stable, comparable metrics across runs.

Outputs:
  data/processed/supervised_metrics.csv
  models/supervised/{scope}_{algo}.pkl   (refitted on full data for predict_player.py)
  models/supervised/preprocessing.pkl
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)

INPUT_PATH = Path("data/processed/players_processed.csv")
METRICS_PATH = Path("data/processed/supervised_metrics.csv")
MODELS_DIR = Path("models/supervised")

FEATURE_COLS = [
    "age",
    "goals_p90", "assists_p90", "goals_assists_p90",
    "non_penalty_goals_p90", "np_goals_assists_p90",
    "shots_p90", "shots_on_target_p90", "shots_on_target_pct",
    "goals_per_shot", "goals_per_shot_on_target",
    "interceptions_p90", "tackles_won_p90",
    "fouls_committed_p90", "fouls_drawn_p90",
    "crosses_p90", "offsides_p90",
    "attacking_index", "creativity_index",
    "defensive_index", "discipline_index",
]

TARGET_COL = "market_value_eur"
POSITION_COL = "position_family"

RANDOM_STATE = 42
N_SPLITS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
    }


def cross_validated_metrics(X: np.ndarray, y: np.ndarray, scope: str) -> tuple[pd.DataFrame, dict]:
    """5-fold CV per model. Returns metrics + models refitted on full data."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    refitted = {}

    for name in build_models().keys():
        oof_pred = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in kf.split(X):
            model = build_models()[name]
            model.fit(X[train_idx], y[train_idx])
            oof_pred[test_idx] = model.predict(X[test_idx])

        mae_log = mean_absolute_error(y, oof_pred)
        rmse_log = float(np.sqrt(mean_squared_error(y, oof_pred)))
        r2 = r2_score(y, oof_pred)

        y_true_eur = np.expm1(y)
        y_pred_eur = np.expm1(oof_pred)
        mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)
        rmse_eur = float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur)))
        median_abs_error_eur = float(np.median(np.abs(y_true_eur - y_pred_eur)))

        rows.append({
            "scope": scope, "algorithm": name,
            "n_samples": len(y), "n_folds": N_SPLITS,
            "mae_log": mae_log, "rmse_log": rmse_log, "r2": r2,
            "mae_eur": mae_eur, "rmse_eur": rmse_eur,
            "median_abs_error_eur": median_abs_error_eur,
        })

        final_model = build_models()[name]
        final_model.fit(X, y)
        refitted[name] = final_model

        log.info(
            f"  {name:<18} R2={r2:.3f}  MAE=€{mae_eur/1e6:.2f}m  "
            f"Median|err|=€{median_abs_error_eur/1e6:.2f}m"
        )

    return pd.DataFrame(rows), refitted


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    log.info(f"Loaded {len(df)} rows from {INPUT_PATH}")

    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df["_y_log"] = np.log1p(df[TARGET_COL])

    log.info(f"After filtering for target: {len(df)} rows")
    log.info(f"Feature columns: {len(FEATURE_COLS)}")
    log.info(f"Evaluation: {N_SPLITS}-fold cross-validation")

    all_metrics = []
    all_models = {}

    log.info("=" * 60)
    log.info("GLOBAL: four models on all outfield players")
    log.info("=" * 60)
    X = df[FEATURE_COLS].values
    y = df["_y_log"].values

    g_metrics, g_fitted = cross_validated_metrics(X, y, scope="global")
    all_metrics.append(g_metrics)
    for name, mdl in g_fitted.items():
        all_models[f"global_{name}"] = mdl

    log.info("=" * 60)
    log.info("PER-POSITION: four models per position_family")
    log.info("=" * 60)
    families = [f for f in df[POSITION_COL].dropna().unique() if f != "Goalkeeper"]

    for family in sorted(families):
        fdf = df[df[POSITION_COL] == family].reset_index(drop=True)
        if len(fdf) < 100:
            log.warning(f"Skipping {family}: only {len(fdf)} rows (need >=100)")
            continue

        log.info(f"--- {family} (n={len(fdf)}) ---")
        Xf = fdf[FEATURE_COLS].values
        yf = fdf["_y_log"].values

        p_metrics, p_fitted = cross_validated_metrics(Xf, yf, scope=family)
        all_metrics.append(p_metrics)
        for name, mdl in p_fitted.items():
            all_models[f"{family}_{name}"] = mdl

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(METRICS_PATH, index=False)
    log.info(f"Saved metrics -> {METRICS_PATH}")

    for label, model in all_models.items():
        safe = label.replace(" ", "_").replace("-", "_")
        joblib.dump(model, MODELS_DIR / f"{safe}.pkl")
    joblib.dump({
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "position_col": POSITION_COL,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
    }, MODELS_DIR / "preprocessing.pkl")
    log.info(f"Saved {len(all_models)} models to {MODELS_DIR}/")

    log.info("=" * 60)
    log.info("BEST MODEL PER SCOPE (by R²):")
    log.info("=" * 60)
    for scope, sub in combined.groupby("scope"):
        best = sub.sort_values("r2", ascending=False).iloc[0]
        log.info(
            f"  {scope:<22} best={best['algorithm']:<18} "
            f"R2={best['r2']:.3f}  MAE=€{best['mae_eur']/1e6:.2f}m  "
            f"Median|err|=€{best['median_abs_error_eur']/1e6:.2f}m"
        )


if __name__ == "__main__":
    main()
