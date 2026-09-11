"""
predict_player.py
-----------------
Predict a specific player's market value using the best-performing supervised
model, with a SHAP explanation of the top drivers.

Usage:
    python3 src/predict_player.py "Rodri" --team "Manchester City"
    python3 src/predict_player.py "Lamine Yamal" --season 2024
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/processed/players_processed.csv")
METRICS_PATH = Path("data/processed/supervised_metrics.csv")
MODELS_DIR = Path("models/supervised")


def find_best_algo(metrics: pd.DataFrame, scope: str) -> str | None:
    sub = metrics[metrics["scope"] == scope]
    sub = sub[sub["algorithm"] != "LinearRegression"]
    if sub.empty:
        return None
    return sub.sort_values("r2", ascending=False).iloc[0]["algorithm"]


def load_model(scope: str, algo: str):
    safe = f"{scope}_{algo}".replace(" ", "_").replace("-", "_")
    return joblib.load(MODELS_DIR / f"{safe}.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a player's market value with SHAP explanation."
    )
    parser.add_argument("player", help="Player name")
    parser.add_argument("--team", type=str, default=None,
                        help="Disambiguate by team (partial match)")
    parser.add_argument("--season", type=int, default=None,
                        help="Season (2023 or 2024)")
    parser.add_argument("--use-position-model", action="store_true",
                        help="Use the per-position best model instead of the global one")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")
    feature_cols = preproc["feature_cols"]
    df[feature_cols] = df[feature_cols].fillna(0)

    #Find the player
    cand = df[df["player"].str.lower() == args.player.lower()]
    if args.team:
        cand = cand[cand["team"].str.contains(args.team, case=False, na=False)]
    if args.season:
        season_int = int(f"{str(args.season)[-2:]}{str(args.season + 1)[-2:]}")
        cand = cand[cand["season"] == season_int]

    if cand.empty:
        #Partial match fallback
        cand = df[df["player"].str.contains(args.player, case=False, na=False)]
        if cand.empty:
            print(f"ERROR: no player matching '{args.player}'")
            sys.exit(1)
        if args.team:
            cand = cand[cand["team"].str.contains(args.team, case=False, na=False)]

    if len(cand) > 1 and args.season is None:
        cand = cand.sort_values("season", ascending=False)
    row = cand.iloc[0]

    #Which model?
    if args.use_position_model and pd.notna(row["position_family"]):
        scope = row["position_family"]
    else:
        scope = "global"

    best_algo = find_best_algo(metrics, scope)
    if best_algo is None:
        print(f"ERROR: no model for scope '{scope}'")
        sys.exit(1)

    model = load_model(scope, best_algo)

    x = row[feature_cols].values.reshape(1, -1)
    pred_log = float(model.predict(x)[0])
    pred_eur = float(np.expm1(pred_log))
    actual_eur = row["market_value_eur"]

    print(f"PLAYER: {row['player']} — {row['team']} ({row['league']} {row['season']})")
    print(f"        {row['pos']} / {row.get('tm_position', 'unknown')} / age {row['age']:.0f}")
    print()
    print(f"MODEL: {best_algo} ({scope})")
    print()
    print(f"  Predicted value:  €{pred_eur/1e6:.1f}m")
    if pd.notna(actual_eur):
        print(f"  Actual value:     €{actual_eur/1e6:.1f}m")
        print(f"  Error:            €{(pred_eur - actual_eur)/1e6:+.1f}m")
    print()

    #SHAP
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(x)[0]

    print("TOP 8 DRIVERS (SHAP contributions to log-value):")
    contribs = sorted(zip(feature_cols, shap_val, x[0]),
                      key=lambda t: abs(t[1]), reverse=True)
    for feat, sv, val in contribs[:8]:
        direction = "↑" if sv > 0 else "↓"
        print(f"  {direction} {feat:<28} value={val:>8.2f}  contribution={sv:+.3f}")


if __name__ == "__main__":
    main()
