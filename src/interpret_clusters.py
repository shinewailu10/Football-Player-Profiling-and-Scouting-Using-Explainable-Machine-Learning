"""
interpret_clusters.py
---------------------
Takes the clustered players (from run_clustering.py), profiles each cluster
by its distinctive per-90 features, and identifies representative players.

Produces a markdown summary, read directly to name the tactical roles.

Which algorithm is used for interpretation? By default: the algorithm with
the highest silhouette in clustering_metrics.csv. Override by editing
the WINNING_ALGO constant below.

Input:
  data/processed/players_clustered.csv       (metadata + cluster labels)
  data/processed/feature_matrix_raw.csv      (standardised per-90 features)
  data/processed/players_processed.csv       (original un-standardised features
                                              — for interpretable cluster means)
  data/processed/feature_matrix_pca.csv      (for centroid-distance / rep players)
  data/processed/clustering_metrics.csv      (to pick default winner)

Output:
  data/processed/cluster_profiles.csv        (mean of each feature per cluster)
  data/processed/cluster_representatives.csv (top-5 closest players per cluster)
  data/processed/cluster_summary.md          (human-readable summary)

Usage:
    python3 src/interpret_clusters.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")

# Set to None to auto-pick based on silhouette score.
WINNING_ALGO: str | None = "kmeans"

#How many top-distinctive features to show per cluster
N_DISTINCTIVE_FEATURES = 6
#How many representative players (closest to centroid) per cluster
N_REPRESENTATIVES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def auto_pick_winner(metrics_path: Path) -> str:
    """Choose the algorithm with the highest silhouette (from configs with >=2 clusters)."""
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["n_clusters"] >= 2]
    best = metrics.sort_values("silhouette", ascending=False).iloc[0]
    algo = best["algorithm"].lower()
    log.info(f"Auto-picked winner: {algo} (silhouette={best['silhouette']:.3f})")
    return algo


def main() -> None:
    algo = WINNING_ALGO or auto_pick_winner(DATA_DIR / "clustering_metrics.csv")
    label_col = f"{algo}_label"

    clustered = pd.read_csv(DATA_DIR / "players_clustered.csv")
    if label_col not in clustered.columns:
        raise ValueError(
            f"Column '{label_col}' not found in players_clustered.csv. "
            f"Available: {list(clustered.columns)}"
        )

    #Get the original (un-standardised) per-90 features for interpretation
    processed = pd.read_csv(DATA_DIR / "players_processed.csv")
    #Align — same row order because prepare_features preserved it
    if len(processed) != len(clustered):
        #Re-align by joining on identifiers
        key_cols = ["season", "player", "born"]
        clustered = clustered.merge(
            processed[key_cols + [c for c in processed.columns if c not in clustered.columns]],
            on=key_cols, how="left"
        )

    #Feature columns to summarise — every numeric per-90 column
    per90_cols = [c for c in processed.columns if "_p90" in c]
    other_features = ["goals_per_shot", "shots_on_target_pct",
                      "attacking_index", "creativity_index",
                      "defensive_index", "discipline_index"]
    feature_cols = [c for c in per90_cols + other_features if c in processed.columns]
    log.info(f"Interpreting on {len(feature_cols)} features")

    #Attach features to clustered dataframe if not already present
    for c in feature_cols:
        if c not in clustered.columns:
            clustered[c] = processed[c].values

    #Drop noise points (HDBSCAN's -1) from profiles
    working = clustered[clustered[label_col] != -1].copy()

    #---Cluster means (interpretable, un-standardised)---
    means = working.groupby(label_col)[feature_cols].mean()
    global_means = working[feature_cols].mean()
    #z-score of each cluster mean vs overall
    global_stds = working[feature_cols].std().replace(0, np.nan)
    z_scores = (means - global_means) / global_stds
    means["cluster_size"] = working.groupby(label_col).size()

    means.to_csv(DATA_DIR / "cluster_profiles.csv")
    log.info(f"Saved cluster_profiles.csv (shape {means.shape})")

    #---Representative players---
    pca = pd.read_csv(DATA_DIR / "feature_matrix_pca.csv").values
    reps_rows = []
    #Only iterate over valid (non-noise) clusters
    for cluster_id in sorted(working[label_col].unique()):
        cluster_mask = clustered[label_col] == cluster_id
        cluster_pca = pca[cluster_mask.values]
        centroid = cluster_pca.mean(axis=0)
        distances = np.linalg.norm(cluster_pca - centroid, axis=1)
        cluster_meta = clustered[cluster_mask].reset_index(drop=True)
        order = np.argsort(distances)[:N_REPRESENTATIVES]
        for rank, idx in enumerate(order, 1):
            row = cluster_meta.iloc[idx]
            reps_rows.append({
                "cluster": int(cluster_id),
                "rank": rank,
                "player": row["player"],
                "team": row["team"],
                "season": row["season"],
                "pos": row["pos"],
                "age": row["age"],
                "market_value_eur": row.get("market_value_eur"),
                "distance_to_centroid": float(distances[idx]),
            })
    reps = pd.DataFrame(reps_rows)
    reps.to_csv(DATA_DIR / "cluster_representatives.csv", index=False)
    log.info(f"Saved cluster_representatives.csv ({len(reps)} rows)")

    #---Markdown summary---
    md_lines = [
        f"# Cluster Summary — {algo.upper()}",
        "",
        f"Interpreted from `players_clustered.csv` column `{label_col}`.",
        f"Feature values below are un-standardised per-90 means for each cluster.",
        "",
    ]

    n_noise = int((clustered[label_col] == -1).sum())
    if n_noise:
        md_lines.append(f"**Note:** {n_noise} players labelled as noise (HDBSCAN -1) "
                        "excluded from the profiles below.")
        md_lines.append("")

    for cluster_id in sorted(working[label_col].unique()):
        size = int(means.loc[cluster_id, "cluster_size"])
        md_lines.append(f"## Cluster {int(cluster_id)}  ({size} players)")
        md_lines.append("")

        #Top distinctive features (largest absolute z-score, positive)
        cluster_z = z_scores.loc[cluster_id].dropna()
        top_high = cluster_z.sort_values(ascending=False).head(N_DISTINCTIVE_FEATURES)
        top_low = cluster_z.sort_values(ascending=True).head(N_DISTINCTIVE_FEATURES)

        md_lines.append("**Distinctively HIGH features** (z vs global mean):")
        for feat, z in top_high.items():
            raw_val = means.loc[cluster_id, feat]
            md_lines.append(f"- `{feat}`: {raw_val:.2f}  (z = {z:+.2f})")
        md_lines.append("")

        md_lines.append("**Distinctively LOW features** (z vs global mean):")
        for feat, z in top_low.items():
            raw_val = means.loc[cluster_id, feat]
            md_lines.append(f"- `{feat}`: {raw_val:.2f}  (z = {z:+.2f})")
        md_lines.append("")

        #Representative players
        cluster_reps = reps[reps["cluster"] == int(cluster_id)]
        md_lines.append("**Representative players** (closest to cluster centroid):")
        for _, r in cluster_reps.iterrows():
            mv = r["market_value_eur"]
            mv_str = f"€{mv:,.0f}" if pd.notna(mv) else "—"
            md_lines.append(
                f"- {r['player']} ({r['team']}, {r['season']}, "
                f"{r['pos']}, age {r['age']:.0f}, {mv_str})"
            )
        md_lines.append("")
        md_lines.append("**Suggested tactical-role label:** _[fill in based on above]_")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out_path = DATA_DIR / "cluster_summary.md"
    out_path.write_text("\n".join(md_lines))
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
