"""
final_figures_extended.py
-------------------------
Generates additional publication-quality figures for the dissertation

Adds:
  fig6_pipeline_flow.png             pipeline architecture diagram
  fig7_market_value_distribution.png  raw vs log-transformed target distribution
  fig8_position_family_counts.png     dataset composition by position
  fig9_feature_correlation.png        feature correlation heatmap
  fig10_cluster_selection.png         silhouette / DB / CH across k
  fig11_cluster_profiles.png          cluster × feature heatmap
  fig13_model_comparison_bars.png     4-model comparison across all scopes

Usage:
    python3 src/final_figures_extended.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def fig_pipeline_flow():
    """Boxes-and-arrows diagram of the pipeline stages."""
    log.info("Figure 6: pipeline flow diagram")

    stages = [
        ("Data\ncollection", "FBref +\nTransfermarkt +\nStatsBomb", "#3498db"),
        ("Feature\nengineering", "Merge, per-90,\nposition labels", "#3498db"),
        ("Clustering", "K-Means k=4\n(vs GMM, HDBSCAN)", "#e67e22"),
        ("Similarity\nrecommender", "Cosine similarity\n+ position filter", "#e67e22"),
        ("Supervised\nprediction", "4 models,\n5-fold CV", "#27ae60"),
        ("SHAP\nexplanation", "Global +\nper-position", "#27ae60"),
        ("Dashboard\n(Streamlit)", "5 pages,\ninteractive UI", "#9b59b6"),
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    for i, (title, detail, color) in enumerate(stages):
        #Box
        box = mpatches.FancyBboxPatch(
            (i + 0.05, 0.5), 0.9, 1.5,
            boxstyle="round,pad=0.05",
            edgecolor=color, facecolor=color, alpha=0.2, linewidth=2,
        )
        ax.add_patch(box)
        #Title
        ax.text(i + 0.5, 1.6, title, ha="center", va="center",
                fontweight="bold", fontsize=10)
        #Detail
        ax.text(i + 0.5, 0.9, detail, ha="center", va="center",
                fontsize=8, color="#333")
        #Arrow to next
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 1.05, 1.25), xytext=(i + 0.95, 1.25),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))

    ax.text(len(stages) / 2, 2.3, "Explainable ML pipeline for football player scouting",
            ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_pipeline_flow.png", bbox_inches="tight")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig6_pipeline_flow.png'}")


def fig_market_value_distribution():
    """Raw vs log-transformed market value distribution.
    Justifies the log-transformation choice for supervised modelling."""
    log.info("Figure 7: market value distribution (raw vs log)")

    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()]
    values = df["market_value_eur"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    #Raw distribution
    axes[0].hist(values / 1e6, bins=60, color="#e74c3c", alpha=0.75, edgecolor="white")
    axes[0].set_xlabel("Market value (€m)")
    axes[0].set_ylabel("Number of players")
    axes[0].set_title(f"Raw market values (n={len(values):,})\n"
                      f"Heavy right skew: max €{values.max()/1e6:.0f}m, "
                      f"median €{np.median(values)/1e6:.1f}m")
    axes[0].axvline(np.median(values) / 1e6, color="black", linestyle="--", alpha=0.6,
                    label=f"Median €{np.median(values)/1e6:.1f}m")
    axes[0].axvline(np.mean(values) / 1e6, color="darkred", linestyle=":", alpha=0.6,
                    label=f"Mean €{np.mean(values)/1e6:.1f}m")
    axes[0].legend()

    #Log-transformed
    log_values = np.log1p(values)
    axes[1].hist(log_values, bins=60, color="#3498db", alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("log(1 + market value in €)")
    axes[1].set_ylabel("Number of players")
    axes[1].set_title("After log-transformation\n"
                      "Approximately symmetric — suitable regression target")

    plt.suptitle("Market value distribution justifies log-transformation of target",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_market_value_distribution.png", bbox_inches="tight")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig7_market_value_distribution.png'}")


def fig_position_family_counts():
    """Player-count by position family. Shows dataset composition."""
    log.info("Figure 8: position family player counts")

    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()]
    counts = df["position_family"].value_counts()
    #Drop goalkeepers if only a handful (data-quality artifact)
    counts = counts[counts.index != "Goalkeeper"]
    counts = counts.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    ax.barh(counts.index, counts.values, color=colors, edgecolor="white")

    for i, v in enumerate(counts.values):
        ax.text(v + 15, i, str(v), va="center", fontsize=10)

    ax.set_xlabel("Number of player-seasons")
    ax.set_title(f"Dataset composition by position family (total n={counts.sum():,} outfield players)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig8_position_family_counts.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig8_position_family_counts.png'}")


def fig_feature_correlation():
    """Feature correlation heatmap. Shows redundancy structure."""
    log.info("Figure 9: feature correlation heatmap")

    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()]

    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")
    feature_cols = preproc["feature_cols"]

    corr = df[feature_cols].fillna(0).corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols, fontsize=9)

    #Annotate cells with correlation values above a threshold
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            v = corr.values[i, j]
            if abs(v) > 0.6 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.75 else "black")

    plt.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Feature correlation matrix (annotated: |corr| > 0.6)\n"
                 "Strong correlations reveal redundancy — e.g. goals_p90 and non_penalty_goals_p90")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig9_feature_correlation.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig9_feature_correlation.png'}")


def fig_cluster_selection():
    """K-Means / GMM validity metrics across k. Shows how k was chosen."""
    log.info("Figure 10: cluster-metric sweep across k")

    metrics_path = DATA_DIR / "clustering_metrics.csv"
    if not metrics_path.exists():
        log.warning("clustering_metrics.csv missing — skipping figure 10")
        return

    m = pd.read_csv(metrics_path)
    km = m[m["algorithm"] == "KMeans"].sort_values("param")
    gm = m[m["algorithm"] == "GMM"].sort_values("param")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    #Silhouette 
    axes[0].plot(km["param"], km["silhouette"], "o-", label="K-Means", color="#3498db")
    axes[0].plot(gm["param"], gm["silhouette"], "s-", label="GMM", color="#e67e22")
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Silhouette coefficient")
    axes[0].set_title("Silhouette (higher is better)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    #Mark k=4 winner
    best_k = km.loc[km["silhouette"].idxmax(), "param"]
    axes[0].axvline(best_k, color="red", linestyle=":", alpha=0.5)
    axes[0].annotate(f"K-Means best: k={int(best_k)}",
                     xy=(best_k, km["silhouette"].max()),
                     xytext=(best_k + 1, km["silhouette"].max() + 0.005),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))

    #Davies-Bouldin 
    axes[1].plot(km["param"], km["davies_bouldin"], "o-", label="K-Means", color="#3498db")
    axes[1].plot(gm["param"], gm["davies_bouldin"], "s-", label="GMM", color="#e67e22")
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("Davies-Bouldin index")
    axes[1].set_title("Davies-Bouldin (lower is better)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    #Calinski-Harabasz 
    axes[2].plot(km["param"], km["calinski_harabasz"], "o-", label="K-Means", color="#3498db")
    axes[2].plot(gm["param"], gm["calinski_harabasz"], "s-", label="GMM", color="#e67e22")
    axes[2].set_xlabel("k (number of clusters)")
    axes[2].set_ylabel("Calinski-Harabasz index")
    axes[2].set_title("Calinski-Harabasz (higher is better)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Cluster validity metrics justify K-Means k=4 selection", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_cluster_selection.png", bbox_inches="tight")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig10_cluster_selection.png'}")


def fig_cluster_profiles():
    """Cluster × feature heatmap of mean values (z-scores).
    Shows what each cluster IS at a glance."""
    log.info("Figure 11: cluster profile heatmap")

    profiles_path = DATA_DIR / "cluster_profiles.csv"
    if not profiles_path.exists():
        log.warning("cluster_profiles.csv missing — skipping figure 11")
        return

    profiles = pd.read_csv(profiles_path, index_col=0)
    #Drop meta columns like cluster_size
    feat = [c for c in profiles.columns if c != "cluster_size"]
    values = profiles[feat]

    #Z-score each feature across clusters to make comparison visible
    z = (values - values.mean()) / values.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(feat)))
    ax.set_xticklabels(feat, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels([f"Cluster {int(i)}\n(n={int(profiles.iloc[k]['cluster_size']):,})"
                        for k, i in enumerate(profiles.index)],
                       fontsize=10)

    #Annotate strong z-scores
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.values[i, j]
            if abs(v) > 1:
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(v) > 1.5 else "black",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="Z-score (cluster mean vs overall)")
    ax.set_title("Cluster profiles — z-scored feature means\n"
                 "Red: this cluster scores above average on this feature. "
                 "Blue: below average.")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_cluster_profiles.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig11_cluster_profiles.png'}")


def fig_model_comparison_bars():
    """4-model comparison across all 8 scopes as grouped bars."""
    log.info("Figure 13: 4-model comparison across scopes")

    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")

    #Pivot: rows = scope, cols = algorithm, values = R²
    pivot = metrics.pivot_table(
        index="scope", columns="algorithm", values="r2",
    )
    #Order scopes: global first, then by best R² descending
    scope_order = ["global"] + sorted(
        [s for s in pivot.index if s != "global"],
        key=lambda s: -pivot.loc[s].max(),
    )
    pivot = pivot.reindex(scope_order)
    algo_order = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"]
    pivot = pivot[algo_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scope_order))
    width = 0.2
    colors = ["#3498db", "#27ae60", "#e67e22", "#9b59b6"]

    for i, algo in enumerate(algo_order):
        ax.bar(x + (i - 1.5) * width, pivot[algo].values, width,
               label=algo, color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(scope_order, rotation=25, ha="right")
    ax.set_ylabel("R² (5-fold CV)")
    ax.set_title("Model comparison across all scopes — R² by algorithm\n"
                 "RandomForest wins 6/8; LinearRegression wins Winger and Central Midfield; "
                 "gradient boosting (XGBoost, LightGBM) offers no consistent advantage")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(pivot.values.max(), 0.7) * 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig13_model_comparison_bars.png")
    plt.close()
    log.info(f"  saved {OUTPUT_DIR / 'fig13_model_comparison_bars.png'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_pipeline_flow()
    fig_market_value_distribution()
    fig_position_family_counts()
    fig_feature_correlation()
    fig_cluster_selection()
    fig_cluster_profiles()
    fig_model_comparison_bars()

    log.info(f"Done. All extended figures in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
