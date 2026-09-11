"""
explain_supervised.py  (v2 — tree-model fallback when LinearRegression wins)
----------------------------------------------------------------------------
Generates SHAP explanations for the winning supervised models from
train_supervised.py.

v2 change: v1 picked the best-R² model that wasn't LinearRegression. With
5-fold CV, LinearRegression turns out to win most scopes. So v2 explicitly
picks the best TREE-BASED model per scope (RandomForest/XGBoost/LightGBM)
and uses TreeSHAP on it. If LinearRegression was the actual winner, we note
that in the report and explain the second-best model as a proxy.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/processed/players_processed.csv")
METRICS_PATH = Path("data/processed/supervised_metrics.csv")
MODELS_DIR = Path("models/supervised")
OUTPUT_DIR = Path("data/processed/shap")

TREE_ALGOS = {"RandomForest", "XGBoost", "LightGBM"}

CASE_STUDIES = [
    ("Rodri", "Manchester City", 2023),
    ("Bukayo Saka", None, 2023),
    ("Erling Haaland", None, 2023),
    ("Virgil van Dijk", None, 2023),
    ("Kevin De Bruyne", None, 2023),
    ("Lamine Yamal", None, 2024),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def find_best_tree_algo(metrics: pd.DataFrame, scope: str) -> tuple[str, str] | tuple[None, None]:
    """Return (best_tree_algo, note) for a scope. The note explains whether
    the tree algo was also the overall winner, or was picked as a proxy for
    a Linear Regression winner."""
    sub = metrics[metrics["scope"] == scope].sort_values("r2", ascending=False)
    if sub.empty:
        return None, None

    overall_best = sub.iloc[0]["algorithm"]
    tree_sub = sub[sub["algorithm"].isin(TREE_ALGOS)]
    if tree_sub.empty:
        return None, None
    best_tree = tree_sub.iloc[0]["algorithm"]

    if overall_best == best_tree:
        note = "best overall"
    else:
        note = f"tree proxy (overall best was {overall_best})"
    return best_tree, note


def load_model(scope: str, algo: str):
    safe = f"{scope}_{algo}".replace(" ", "_").replace("-", "_")
    return joblib.load(MODELS_DIR / f"{safe}.pkl")


def find_player_row(df: pd.DataFrame, name: str, team: str | None, season: int) -> int | None:
    cand = df[df["player"] == name]
    if team:
        cand = cand[cand["team"].str.contains(team, case=False, na=False)]
    season_int = int(f"{str(season)[-2:]}{str(season + 1)[-2:]}")
    cand = cand[cand["season"] == season_int]
    if cand.empty:
        return None
    return int(cand.index[0])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_PATH)
    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")
    feature_cols = preproc["feature_cols"]

    df = pd.read_csv(INPUT_PATH)
    df = df[df["market_value_eur"].notna()].reset_index(drop=True)
    df[feature_cols] = df[feature_cols].fillna(0)
    X_all = df[feature_cols].values

    # ---------- Global SHAP ----------
    log.info("=" * 60)
    log.info("GLOBAL SHAP importance")
    log.info("=" * 60)

    best_global_algo, global_note = find_best_tree_algo(metrics, "global")
    log.info(f"Global tree model: {best_global_algo} ({global_note})")

    global_model = load_model("global", best_global_algo)

    sample_size = min(500, len(df))
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(df), size=sample_size, replace=False)
    X_sample = X_all[sample_idx]

    explainer = shap.TreeExplainer(global_model)
    shap_values_global = explainer.shap_values(X_sample)
    log.info(f"Computed SHAP values on sample of {sample_size} players")

    global_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values_global).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(OUTPUT_DIR / "global_importance.csv", index=False)
    log.info("Top 10 global features by mean |SHAP|:")
    for _, row in global_importance.head(10).iterrows():
        log.info(f"  {row['feature']:<28} {row['mean_abs_shap']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    top_features = global_importance.head(15).iloc[::-1]
    ax.barh(top_features["feature"], top_features["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value| (log-EUR)")
    ax.set_title(f"Global feature importance — {best_global_algo}\n"
                 "Top 15 features driving market-value predictions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_importance_barplot.png", dpi=120)
    plt.close()
    log.info(f"Saved {OUTPUT_DIR}/global_importance_barplot.png")

    # ---------- Per-position SHAP ----------
    log.info("=" * 60)
    log.info("PER-POSITION SHAP importance")
    log.info("=" * 60)

    position_families = sorted(
        f for f in df["position_family"].dropna().unique() if f != "Goalkeeper"
    )
    per_pos_rows = []
    per_pos_notes = {}

    for family in position_families:
        best_algo, note = find_best_tree_algo(metrics, family)
        if best_algo is None:
            log.warning(f"  Skipping {family}: no model found")
            continue
        per_pos_notes[family] = (best_algo, note)

        try:
            model = load_model(family, best_algo)
        except FileNotFoundError:
            log.warning(f"  Skipping {family}: model file missing")
            continue

        family_mask = df["position_family"] == family
        X_family = X_all[family_mask.values]

        if len(X_family) > 300:
            sample = rng.choice(len(X_family), size=300, replace=False)
            X_family = X_family[sample]

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_family)
        importance = np.abs(shap_vals).mean(axis=0)

        for feat, imp in zip(feature_cols, importance):
            per_pos_rows.append({
                "position_family": family,
                "algorithm": best_algo,
                "algorithm_note": note,
                "feature": feat,
                "mean_abs_shap": imp,
            })

        top3 = sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:3]
        log.info(f"  {family:<22} ({best_algo}, {note})")
        for feat, imp in top3:
            log.info(f"    {feat:<26} {imp:.4f}")

    per_pos_df = pd.DataFrame(per_pos_rows)
    per_pos_df.to_csv(OUTPUT_DIR / "per_position_importance.csv", index=False)

    if not per_pos_df.empty:
        pivot = per_pos_df.pivot_table(
            index="feature", columns="position_family",
            values="mean_abs_shap", aggfunc="mean",
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
        ax.set_title("Per-position feature importance heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_position_heatmap.png", dpi=120)
        plt.close()
        log.info(f"Saved {OUTPUT_DIR}/per_position_heatmap.png")

    # ---------- Local SHAP force plots ----------
    log.info("=" * 60)
    log.info("LOCAL SHAP for case-study players")
    log.info("=" * 60)

    global_explainer = shap.TreeExplainer(global_model)

    local_report_rows = []
    for name, team, season in CASE_STUDIES:
        row_idx = find_player_row(df, name, team, season)
        if row_idx is None:
            log.warning(f"  {name} ({season}) not found in data")
            continue

        player_row = df.iloc[row_idx]
        x = X_all[row_idx:row_idx + 1]
        shap_val = global_explainer.shap_values(x)[0]
        pred_log = float(global_model.predict(x)[0])
        pred_eur = float(np.expm1(pred_log))
        actual_eur = player_row["market_value_eur"]

        contribs = list(zip(feature_cols, shap_val, x[0]))
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        top_contribs = contribs[:6]

        log.info(f"  {name} ({player_row['team']}, {season}):")
        log.info(f"    predicted €{pred_eur/1e6:.1f}m  vs  actual €{actual_eur/1e6:.1f}m")
        for feat, shap_v, actual_v in top_contribs:
            direction = "↑" if shap_v > 0 else "↓"
            log.info(f"      {direction} {feat:<26} value={actual_v:.2f}  contribution={shap_v:+.3f}")

        local_report_rows.append({
            "player": name,
            "team": player_row["team"],
            "season": season,
            "predicted_eur": pred_eur,
            "actual_eur": actual_eur,
            "top_contribs": top_contribs,
        })

        try:
            fig = plt.figure()
            shap.force_plot(
                global_explainer.expected_value,
                shap_val,
                pd.Series(x[0], index=feature_cols),
                matplotlib=True, show=False,
            )
            safe_name = name.replace(" ", "_").replace("ú", "u").replace("é", "e")
            plt.savefig(OUTPUT_DIR / f"local_{safe_name}_{season}.png",
                        dpi=100, bbox_inches="tight")
            plt.close()
        except Exception as e:
            log.warning(f"    force plot failed: {e}")

    # ---------- Markdown report ----------
    md = ["# SHAP Report — Week 9 (v2)", ""]
    md.append(f"Global tree model explained: **{best_global_algo}** ({global_note})")
    md.append("")
    md.append("_Note: SHAP was computed on the best tree-based model for each scope. "
              "Where a Linear Regression model was the overall best-R² winner "
              "(a common outcome under 5-fold CV in this study), the tree model "
              "is used as a proxy explainer since SHAP provides local-level attributions "
              "not natively available from linear coefficients._")
    md.append("")

    md.append("## Global feature importance (top 15)")
    md.append("")
    md.append("| Feature | Mean \\|SHAP\\| |")
    md.append("|---|---|")
    for _, r in global_importance.head(15).iterrows():
        md.append(f"| {r['feature']} | {r['mean_abs_shap']:.4f} |")
    md.append("")
    md.append("![global_importance_barplot](global_importance_barplot.png)")
    md.append("")

    md.append("## Per-position best tree models")
    md.append("")
    md.append("| Position family | Tree model used | Note |")
    md.append("|---|---|---|")
    for family in position_families:
        if family in per_pos_notes:
            algo, note = per_pos_notes[family]
            md.append(f"| {family} | {algo} | {note} |")
    md.append("")
    md.append("![per_position_heatmap](per_position_heatmap.png)")
    md.append("")

    md.append("## Local case studies")
    md.append("")
    md.append("For each player: predicted vs actual market value + six features "
              "contributing most to the global-model prediction (↑ pushes value up, "
              "↓ pushes down).")
    md.append("")
    for row in local_report_rows:
        md.append(f"### {row['player']} — {row['team']} ({row['season']}/{str(row['season']+1)[-2:]})")
        md.append("")
        md.append(f"- Predicted: **€{row['predicted_eur']/1e6:.1f}m**")
        md.append(f"- Actual: **€{row['actual_eur']/1e6:.1f}m**")
        md.append(f"- Prediction error: **€{(row['predicted_eur'] - row['actual_eur'])/1e6:+.1f}m**")
        md.append("")
        md.append("Top drivers:")
        for feat, shap_v, actual_v in row["top_contribs"]:
            direction = "↑" if shap_v > 0 else "↓"
            md.append(f"- {direction} `{feat}` (value={actual_v:.2f}, contribution={shap_v:+.3f})")
        md.append("")

    (OUTPUT_DIR / "shap_report.md").write_text("\n".join(md))
    log.info(f"Saved {OUTPUT_DIR}/shap_report.md")


if __name__ == "__main__":
    main()
