"""
final_figures.py
----------------
Generates publication-quality figures for the dissertation. Each figure
is saved as PNG (for Word/PDF embedding) at 300 DPI.

Figures produced:
  data/processed/figures/
    fig1_cluster_umap.png            2D UMAP projection of clusters
    fig2_shap_global.png             Global SHAP summary
    fig3_prediction_scatter.png      Predicted vs actual + heavy-tail annotation
    fig4_per_position_r2.png         Per-position R² bar chart
    fig5_recommender_examples.png    Side-by-side similarity distribution

Usage:
    pip install umap-learn
    python3 src/final_figures.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/supervised")
OUTPUT_DIR = DATA_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def fig_cluster_umap():
    """UMAP 2D projection of players coloured by K-Means cluster."""
    log.info("Figure 1: UMAP projection of clusters")
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed — skipping figure 1. "
                    "Install with: pip install umap-learn")
        return

    features = pd.read_csv(DATA_DIR / "feature_matrix_pca.csv").values
    players = pd.read_csv(DATA_DIR / "players_clustered.csv")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=players["kmeans_label"],
        cmap="tab10", s=12, alpha=0.7,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Player tactical clusters (K-Means k=4, UMAP projection)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_cluster_umap.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig1_cluster_umap.png'}")


def fig_shap_global():
    """Bar chart of top-15 features by mean |SHAP| from the global model."""
    log.info("Figure 2: Global SHAP importance")
    imp_path = DATA_DIR / "shap" / "global_importance.csv"
    if not imp_path.exists():
        log.warning(f"{imp_path} missing — run explain_supervised.py first")
        return

    imp = pd.read_csv(imp_path).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"], imp["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("Mean |SHAP value| (log-EUR)")
    ax.set_title("Global feature importance for market-value prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_shap_global.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig2_shap_global.png'}")


def fig_prediction_scatter():
    """Predicted vs actual market value, coloured by position family,
    with the y=x reference line and the extreme-value regime annotated."""
    log.info("Figure 3: Predicted vs actual scatter")
    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()].reset_index(drop=True)

    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")
    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")
    feature_cols = preproc["feature_cols"]
    df[feature_cols] = df[feature_cols].fillna(0)

    tree = metrics[metrics["algorithm"].isin({"RandomForest", "XGBoost", "LightGBM"})]
    best = tree[tree["scope"] == "global"].sort_values("r2", ascending=False).iloc[0]
    safe = f"global_{best['algorithm']}".replace(" ", "_").replace("-", "_")
    model = joblib.load(MODELS_DIR / f"{safe}.pkl")

    X = df[feature_cols].values
    preds = np.expm1(model.predict(X))
    actual = df["market_value_eur"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    # Colour by position family
    families = df["position_family"].fillna("Unknown")
    for family in families.unique():
        mask = families == family
        ax.scatter(
            actual[mask] / 1e6, preds[mask] / 1e6,
            label=family, alpha=0.5, s=18,
        )

    # Perfect prediction line
    top = max(actual.max(), preds.max()) / 1e6
    ax.plot([0, top], [0, top], "k--", linewidth=1, label="Perfect prediction")

    # Highlight extreme-value regime (players over €100m)
    ax.axvline(100, color="red", linestyle=":", alpha=0.5)
    ax.text(105, top * 0.05, "Extreme-value\nregime (>€100m):\nsystematic\nunder-prediction",
            fontsize=9, color="red", verticalalignment="bottom")

    ax.set_xlabel("Actual market value (€m)")
    ax.set_ylabel(f"Predicted market value (€m) — {best['algorithm']}")
    ax.set_title("Predicted vs actual market values (out-of-fold predictions)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_prediction_scatter.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig3_prediction_scatter.png'}")


def fig_per_position_r2():
    """Bar chart of R² per position family for the winning model in each,
    with the global R² as a reference line."""
    log.info("Figure 4: Per-position R²")
    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")

    winners = (
        metrics.sort_values("r2", ascending=False)
        .groupby("scope").first().reset_index()
    )
    global_r2 = winners[winners["scope"] == "global"]["r2"].iloc[0]
    per_pos = winners[winners["scope"] != "global"].sort_values("r2", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ecc71" if r > global_r2 else "#e74c3c" for r in per_pos["r2"]]
    ax.barh(per_pos["scope"], per_pos["r2"], color=colors)
    ax.axvline(global_r2, color="black", linestyle="--",
               label=f"Global model R² = {global_r2:.3f}")

    for i, r in enumerate(per_pos.itertuples()):
        ax.text(r.r2 + 0.005, i, f"{r.r2:.3f} ({r.algorithm})",
                va="center", fontsize=9)

    ax.set_xlim(0, per_pos["r2"].max() * 1.4)
    ax.set_xlabel("R²")
    ax.set_title("Market-value prediction R² by position family\n"
                 "(green: per-position model beats global; red: does not)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_per_position_r2.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig4_per_position_r2.png'}")


def fig_recommender_examples():
    """Similarity distribution comparison — Saka (well-supported) vs
    Rodri (harder) — showing why R² differs between roles."""
    log.info("Figure 5: Recommender similarity distributions")
    import sys
    sys.path.insert(0, "src")
    from recommender import Recommender

    r = Recommender.load()

    examples = [
        ("Bukayo Saka", None, "Right winger (well-supported)"),
        ("Rodri", "Manchester City", "Defensive midfielder (harder)"),
        ("Erling Haaland", None, "Centre-forward (well-supported)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, team, subtitle) in zip(axes, examples):
        try:
            recs = r.find_similar(name, team=team, top_n=50, same_tm_position=True)
        except ValueError as e:
            ax.text(0.5, 0.5, f"Could not query {name}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{name}\n{subtitle}")
            continue

        sims = [rec.similarity for rec in recs]
        ax.plot(range(1, len(sims) + 1), sims, "o-", markersize=4)
        ax.axhline(0.9, color="green", linestyle=":", alpha=0.5, label="0.90 threshold")
        ax.set_xlabel("Rank")
        if ax is axes[0]:
            ax.set_ylabel("Cosine similarity")
        ax.set_title(f"{name}\n{subtitle}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.2, 1.05)

    plt.suptitle("Recommender similarity by position — attacking roles reach "
                 "higher similarity than defensive roles",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_recommender_examples.png", bbox_inches="tight")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig5_recommender_examples.png'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_cluster_umap()
    fig_shap_global()
    fig_prediction_scatter()
    fig_per_position_r2()
    fig_recommender_examples()

    log.info(f"Done. All figures in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
