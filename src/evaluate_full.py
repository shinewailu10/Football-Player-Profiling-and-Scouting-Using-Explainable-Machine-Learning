"""
evaluate_full.py
------------------------------------------------------------------------
Comprehensive evaluation for the dissertation.


  - Retrospective transfer study now runs THREE modes per case:
      (a) strict same-position (Transfermarkt fine label)
      (b) same position family (broader 7-family group)
      (c) no position filter, cluster-only
    This exposes a real trade-off worth writing up: position filtering
    improves tactical coherence but can exclude versatile candidates
    like Mbappé (Centre-Forward with left-winger playing style).
  - Adds Álvarez → Marmoush and Haaland → Marmoush as verified 2025 cases.

Produces:
  data/processed/eval/model_comparison_all.csv
  data/processed/eval/prediction_errors_top20.csv
  data/processed/eval/recommender_precision_all_modes.csv
  data/processed/eval/per_position_errors.csv
  data/processed/eval/summary.md
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

from recommender import Recommender

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/supervised")
OUTPUT_DIR = DATA_DIR / "eval"

#Verified real-world transfer cases: (query_player, query_season, query_team, actual_signing, prior_league, note)
RETROSPECTIVE_TRANSFERS = [
    (
        "Vinicius Júnior", 2023, "Real Madrid",
        "Kylian Mbappé", "FRA-Ligue 1",
        "Real Madrid signed Mbappé on free transfer June 2024",
    ),
    (
        "Julián Álvarez", 2023, "Manchester City",
        "Omar Marmoush", "GER-Bundesliga",
        "Man City signed Marmoush Jan 2025 as Álvarez's replacement (per NBC Sports)",
    ),
    (
        "Erling Haaland", 2023, "Manchester City",
        "Omar Marmoush", "GER-Bundesliga",
        "Man City signed Marmoush Jan 2025 to strengthen Haaland's forward line",
    ),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate_model_comparison() -> pd.DataFrame:
    """Full 4-model × 8-scope metrics comparison."""
    log.info("=" * 60)
    log.info("Model comparison: 4 models × 8 scopes")
    log.info("=" * 60)

    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")
    metrics = metrics.sort_values(["scope", "r2"], ascending=[True, False]).copy()
    metrics["is_best_for_scope"] = (
        metrics.groupby("scope")["r2"].transform("max") == metrics["r2"]
    )
    metrics["rank_in_scope"] = metrics.groupby("scope")["r2"].rank(
        ascending=False, method="min",
    ).astype(int)

    out = metrics[[
        "scope", "algorithm", "n_samples", "r2", "mae_eur",
        "median_abs_error_eur", "rank_in_scope", "is_best_for_scope",
    ]].copy()
    out["r2"] = out["r2"].round(3)
    out["mae_eur_m"] = (out["mae_eur"] / 1e6).round(2)
    out["median_error_m"] = (out["median_abs_error_eur"] / 1e6).round(2)
    out = out[[
        "scope", "algorithm", "n_samples", "r2",
        "mae_eur_m", "median_error_m", "rank_in_scope", "is_best_for_scope",
    ]]

    log.info("Winners per scope:")
    winners = metrics[metrics["is_best_for_scope"]]
    for _, w in winners.iterrows():
        log.info(f"  {w['scope']:<22} {w['algorithm']:<18} "
                 f"R2={w['r2']:.3f}  MAE=€{w['mae_eur']/1e6:.2f}m")

    return out


def evaluate_top_players_prediction() -> pd.DataFrame:
    """Predictions + SHAP drivers for the top-20 highest-value players."""
    log.info("=" * 60)
    log.info("Prediction quality on top 20 highest-value players")
    log.info("=" * 60)

    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()].reset_index(drop=True)

    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")
    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")
    feature_cols = preproc["feature_cols"]
    df[feature_cols] = df[feature_cols].fillna(0)

    tree = metrics[metrics["algorithm"].isin({"RandomForest", "XGBoost", "LightGBM"})]
    best_tree = tree[tree["scope"] == "global"].sort_values("r2", ascending=False).iloc[0]
    safe = f"global_{best_tree['algorithm']}".replace(" ", "_").replace("-", "_")
    model = joblib.load(MODELS_DIR / f"{safe}.pkl")
    log.info(f"Using {best_tree['algorithm']} for global SHAP explanations")

    explainer = shap.TreeExplainer(model)

    top20 = df.nlargest(20, "market_value_eur").reset_index(drop=True)
    rows = []
    for _, player in top20.iterrows():
        x = player[feature_cols].astype(float).values.reshape(1, -1)
        pred_log = float(model.predict(x)[0])
        pred_eur = float(np.expm1(pred_log))
        actual = float(player["market_value_eur"])
        err = pred_eur - actual

        shap_vals = explainer.shap_values(x)[0]
        contribs = sorted(
            zip(feature_cols, shap_vals),
            key=lambda t: abs(t[1]), reverse=True,
        )[:3]
        driver_str = ", ".join(f"{f}({v:+.2f})" for f, v in contribs)

        rows.append({
            "player": player["player"],
            "team": player["team"],
            "season": player["season"],
            "position": player.get("tm_position", "?"),
            "age": int(player["age"]),
            "actual_m": round(actual / 1e6, 1),
            "predicted_m": round(pred_eur / 1e6, 1),
            "error_m": round(err / 1e6, 1),
            "abs_error_pct": round(100 * abs(err) / actual, 1),
            "top_3_drivers": driver_str,
        })

    return pd.DataFrame(rows)


def evaluate_recommender_transfers_all_modes(recommender: Recommender) -> pd.DataFrame:
    """Test each transfer case in three filter modes:
       - strict (same_tm_position=True)
       - broad (same_position_family=True)
       - none  (cluster-only)
    Records the rank the actual signing appears at in each mode."""
    log.info("=" * 60)
    log.info("Recommender: retrospective transfer study (three filter modes)")
    log.info("=" * 60)

    rows = []
    for target, season, team, signing, prior_league, note in RETROSPECTIVE_TRANSFERS:
        for mode_name, kwargs in [
            ("strict (tm_position)",
             {"same_tm_position": True, "same_cluster_only": True}),
            ("broad (position_family)",
             {"same_position_family": True, "same_cluster_only": True}),
            ("cluster-only (no position)",
             {"same_cluster_only": True}),
        ]:
            try:
                recs = recommender.find_similar(
                    target, season=season, team=team, top_n=30, **kwargs,
                )
            except ValueError as e:
                log.warning(f"Could not query {target}: {e}")
                continue

            hit_rank = None
            hit_sim = None
            for r in recs:
                if signing.lower() in r.player.lower():
                    hit_rank = r.rank
                    hit_sim = r.similarity
                    break

            rows.append({
                "query_player": target,
                "query_season": f"{season}/{str(season+1)[-2:]}",
                "actual_signing": signing,
                "filter_mode": mode_name,
                "hit_rank": hit_rank if hit_rank else -1,
                "hit_similarity": round(hit_sim, 3) if hit_sim else None,
                "at_5": hit_rank is not None and hit_rank <= 5,
                "at_10": hit_rank is not None and hit_rank <= 10,
                "at_25": hit_rank is not None and hit_rank <= 25,
                "n_candidates_returned": len(recs),
                "note": note,
            })
            hit_str = f"HIT@{hit_rank} sim={hit_sim:.3f}" if hit_rank else "MISS"
            log.info(f"  {target} → {signing} [{mode_name}]: {hit_str} "
                     f"(pool={len(recs)})")

    df = pd.DataFrame(rows)

    #Aggregate hit rates per mode
    if not df.empty:
        log.info("")
        log.info("Aggregate hit rates by filter mode:")
        for mode in df["filter_mode"].unique():
            sub = df[df["filter_mode"] == mode]
            log.info(f"  {mode:<28}  "
                     f"P@5={sub['at_5'].mean():.2f}  "
                     f"P@10={sub['at_10'].mean():.2f}  "
                     f"P@25={sub['at_25'].mean():.2f}")

    return df


def evaluate_position_family_errors() -> pd.DataFrame:
    """Prediction error breakdown by position family."""
    log.info("=" * 60)
    log.info("Prediction quality by position family")
    log.info("=" * 60)

    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()].reset_index(drop=True)

    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")
    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")
    feature_cols = preproc["feature_cols"]
    df[feature_cols] = df[feature_cols].fillna(0)

    tree = metrics[metrics["algorithm"].isin({"RandomForest", "XGBoost", "LightGBM"})]
    best_tree = tree[tree["scope"] == "global"].sort_values("r2", ascending=False).iloc[0]
    safe = f"global_{best_tree['algorithm']}".replace(" ", "_").replace("-", "_")
    model = joblib.load(MODELS_DIR / f"{safe}.pkl")

    X = df[feature_cols].values
    preds_log = model.predict(X)
    preds_eur = np.expm1(preds_log)
    df["_pred"] = preds_eur
    df["_err"] = df["_pred"] - df["market_value_eur"]
    df["_abs_err"] = df["_err"].abs()

    summary = (
        df.groupby("position_family")
        .agg(
            n=("player", "count"),
            median_actual=("market_value_eur", "median"),
            median_pred=("_pred", "median"),
            median_abs_error=("_abs_err", "median"),
            mean_abs_error=("_abs_err", "mean"),
        )
        .round(0).reset_index()
    )
    for col in ["median_actual", "median_pred", "median_abs_error", "mean_abs_error"]:
        summary[col] = (summary[col] / 1e6).round(2)
    summary = summary.rename(columns={
        "median_actual": "median_actual_m",
        "median_pred": "median_pred_m",
        "median_abs_error": "median_abs_error_m",
        "mean_abs_error": "mean_abs_error_m",
    })

    log.info("\n" + summary.to_string(index=False))
    return summary


def write_summary_markdown(
    model_comp: pd.DataFrame,
    top_players: pd.DataFrame,
    transfers: pd.DataFrame,
    per_family_errors: pd.DataFrame,
) -> None:
    """Emit a summary markdown file suitable for reference during writing."""
    lines = [
        "# Evaluation Summary",
        "",
        "Generated by `evaluate_full.py`. Use these tables as evidence in the dissertation.",
        "",
        "---",
        "",
        "## 1. Model comparison (4 models × 8 scopes)",
        "",
    ]
    winners = model_comp[model_comp["is_best_for_scope"]].copy()
    lines.append("### Winners per scope")
    lines.append("")
    lines.append("| Scope | Best model | R² | MAE (€m) | Median error (€m) |")
    lines.append("|---|---|---|---|---|")
    for _, r in winners.iterrows():
        lines.append(f"| {r['scope']} | {r['algorithm']} | {r['r2']:.3f} | "
                     f"€{r['mae_eur_m']:.2f}m | €{r['median_error_m']:.2f}m |")
    lines.append("")

    lines.append("### Full 32-row comparison")
    lines.append("")
    lines.append("| Scope | Model | Rank | R² | MAE (€m) | Median (€m) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in model_comp.iterrows():
        marker = " ← best" if r["is_best_for_scope"] else ""
        lines.append(
            f"| {r['scope']} | {r['algorithm']} | {r['rank_in_scope']} | "
            f"{r['r2']:.3f} | €{r['mae_eur_m']:.2f}m | "
            f"€{r['median_error_m']:.2f}m |{marker}"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Top-20 player predictions")
    lines.append("")
    lines.append("| Player | Team | Position | Age | Actual (€m) | Predicted (€m) | Error (€m) | Error % |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in top_players.iterrows():
        lines.append(
            f"| {r['player']} | {r['team']} | {r['position']} | {r['age']} | "
            f"€{r['actual_m']:.1f}m | €{r['predicted_m']:.1f}m | "
            f"€{r['error_m']:+.1f}m | {r['abs_error_pct']:.1f}% |"
        )
    lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    top_undervalued = top_players.nsmallest(3, "error_m")
    top_overvalued = top_players.nlargest(3, "error_m")
    lines.append("**Most under-predicted:**")
    for _, r in top_undervalued.iterrows():
        lines.append(f"- {r['player']} — predicted €{r['predicted_m']:.1f}m "
                     f"vs actual €{r['actual_m']:.1f}m (err €{r['error_m']:+.1f}m)")
    lines.append("")
    lines.append("**Least under-predicted / most over-predicted:**")
    for _, r in top_overvalued.iterrows():
        lines.append(f"- {r['player']} — predicted €{r['predicted_m']:.1f}m "
                     f"vs actual €{r['actual_m']:.1f}m (err €{r['error_m']:+.1f}m)")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Recommender: retrospective transfer study")
    lines.append("")
    if not transfers.empty:
        lines.append("Each real-world 2024–25 transfer is tested in three filter modes. "
                     "The gap between modes reveals the trade-off between tactical "
                     "coherence (strict filter) and versatile-player recall (loose filter).")
        lines.append("")

        #Per-case detail
        for query in transfers["query_player"].unique():
            case = transfers[transfers["query_player"] == query]
            signing = case.iloc[0]["actual_signing"]
            note = case.iloc[0]["note"]
            lines.append(f"### Query: {query} → {signing}")
            lines.append(f"_{note}_")
            lines.append("")
            lines.append("| Filter mode | Rank | Similarity | Pool size |")
            lines.append("|---|---|---|---|")
            for _, r in case.iterrows():
                rank_str = f"#{r['hit_rank']}" if r["hit_rank"] > 0 else "MISS"
                sim_str = f"{r['hit_similarity']:.3f}" if r["hit_similarity"] else "—"
                lines.append(
                    f"| {r['filter_mode']} | {rank_str} | {sim_str} | "
                    f"{r['n_candidates_returned']} |"
                )
            lines.append("")

        #Aggregate table
        lines.append("### Aggregate hit rates by filter mode")
        lines.append("")
        lines.append("| Filter mode | P@5 | P@10 | P@25 |")
        lines.append("|---|---|---|---|")
        for mode in transfers["filter_mode"].unique():
            sub = transfers[transfers["filter_mode"] == mode]
            lines.append(f"| {mode} | {sub['at_5'].mean():.2f} | "
                         f"{sub['at_10'].mean():.2f} | {sub['at_25'].mean():.2f} |")
        lines.append("")
    else:
        lines.append("_(no retrospective transfer cases evaluated)_")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Per-position family prediction error")
    lines.append("")
    lines.append("| Position family | N | Median actual (€m) | Median predicted (€m) | "
                 "Median abs error (€m) | Mean abs error (€m) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in per_family_errors.iterrows():
        lines.append(
            f"| {r['position_family']} | {r['n']} | €{r['median_actual_m']:.1f}m | "
            f"€{r['median_pred_m']:.1f}m | €{r['median_abs_error_m']:.2f}m | "
            f"€{r['mean_abs_error_m']:.2f}m |"
        )
    lines.append("")

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))
    log.info(f"Wrote {OUTPUT_DIR / 'summary.md'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_comp = evaluate_model_comparison()
    model_comp.to_csv(OUTPUT_DIR / "model_comparison_all.csv", index=False)
    log.info(f"Wrote {OUTPUT_DIR / 'model_comparison_all.csv'}")

    top_players = evaluate_top_players_prediction()
    top_players.to_csv(OUTPUT_DIR / "prediction_errors_top20.csv", index=False)
    log.info(f"Wrote {OUTPUT_DIR / 'prediction_errors_top20.csv'}")

    recommender = Recommender.load()
    transfers = evaluate_recommender_transfers_all_modes(recommender)
    if not transfers.empty:
        transfers.to_csv(OUTPUT_DIR / "recommender_precision_all_modes.csv", index=False)
        log.info(f"Wrote {OUTPUT_DIR / 'recommender_precision_all_modes.csv'}")

    per_family = evaluate_position_family_errors()
    per_family.to_csv(OUTPUT_DIR / "per_position_errors.csv", index=False)
    log.info(f"Wrote {OUTPUT_DIR / 'per_position_errors.csv'}")

    write_summary_markdown(model_comp, top_players, transfers, per_family)
    log.info("=" * 60)
    log.info(f"All evaluations complete. See {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
