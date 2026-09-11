"""
run_clustering.py
-----------------
Runs three clustering algorithms on the PCA feature matrix, sweeps
hyperparameters, and reports evaluation metrics for each configuration.

- K-Means: sweep k in [4..10]
- Gaussian Mixture Models: sweep n_components in [4..10]
- HDBSCAN: sweep min_cluster_size in [30, 50, 80, 100]

Metrics reported per configuration:
- Silhouette (higher is better; range -1..1)
- Davies-Bouldin (lower is better)
- Calinski-Harabasz (higher is better)

Outputs:
- data/processed/clustering_metrics.csv       — evaluation table
- data/processed/players_clustered.csv        — metadata + best cluster labels per algo
- models/kmeans_best.pkl, models/gmm_best.pkl, models/hdbscan_best.pkl

Usage:
    python3 src/run_clustering.py
"""

import logging
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

PCA_PATH = Path("data/processed/feature_matrix_pca.csv")
META_PATH = Path("data/processed/players_metadata.csv")
OUTPUT_DIR = Path("data/processed")
MODELS_DIR = Path("models")

#Sweep ranges
KMEANS_K_RANGE = list(range(4, 11))   # 4..10
GMM_K_RANGE = list(range(4, 11))       # 4..10
HDBSCAN_MIN_SIZES = [30, 50, 80, 100]

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    """Return the three internal validity indices for one clustering.

    HDBSCAN produces -1 labels for noise points; these are excluded from
    the metric computation (standard practice), otherwise noise would
    penalise the metrics unfairly."""
    mask = labels != -1
    unique_labels = np.unique(labels[mask]) if mask.any() else np.array([])

    if len(unique_labels) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "n_clusters": len(unique_labels),
                "n_noise": int((labels == -1).sum())}

    X_clean = X[mask]
    labels_clean = labels[mask]

    return {
        "silhouette": silhouette_score(X_clean, labels_clean),
        "davies_bouldin": davies_bouldin_score(X_clean, labels_clean),
        "calinski_harabasz": calinski_harabasz_score(X_clean, labels_clean),
        "n_clusters": len(unique_labels),
        "n_noise": int((labels == -1).sum()),
    }


def run_kmeans_sweep(X: np.ndarray) -> tuple[pd.DataFrame, dict]:
    log.info("=== K-Means sweep ===")
    rows = []
    fitted = {}
    for k in KMEANS_K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X)
        metrics = evaluate(X, labels)
        rows.append({"algorithm": "KMeans", "param": k, **metrics})
        fitted[k] = model
        log.info(f"  k={k}: sil={metrics['silhouette']:.3f}  "
                 f"db={metrics['davies_bouldin']:.3f}  "
                 f"ch={metrics['calinski_harabasz']:.1f}")
    return pd.DataFrame(rows), fitted


def run_gmm_sweep(X: np.ndarray) -> tuple[pd.DataFrame, dict]:
    log.info("=== GMM sweep ===")
    rows = []
    fitted = {}
    for k in GMM_K_RANGE:
        model = GaussianMixture(
            n_components=k, random_state=RANDOM_STATE, n_init=3, max_iter=200
        )
        labels = model.fit_predict(X)
        metrics = evaluate(X, labels)
        rows.append({"algorithm": "GMM", "param": k, **metrics})
        fitted[k] = model
        log.info(f"  k={k}: sil={metrics['silhouette']:.3f}  "
                 f"db={metrics['davies_bouldin']:.3f}  "
                 f"ch={metrics['calinski_harabasz']:.1f}")
    return pd.DataFrame(rows), fitted


def run_hdbscan_sweep(X: np.ndarray) -> tuple[pd.DataFrame, dict]:
    log.info("=== HDBSCAN sweep ===")
    rows = []
    fitted = {}
    for min_size in HDBSCAN_MIN_SIZES:
        model = hdbscan.HDBSCAN(min_cluster_size=min_size)
        labels = model.fit_predict(X)
        metrics = evaluate(X, labels)
        rows.append({"algorithm": "HDBSCAN", "param": min_size, **metrics})
        fitted[min_size] = model
        log.info(f"  min_size={min_size}: n_clusters={metrics['n_clusters']}  "
                 f"n_noise={metrics['n_noise']}  "
                 f"sil={metrics['silhouette']:.3f}  "
                 f"db={metrics['davies_bouldin']:.3f}")
    return pd.DataFrame(rows), fitted


def pick_best(metrics_df: pd.DataFrame, algo: str) -> tuple[object, object]:
    """Choose the best config for one algorithm.

    Ranking: highest silhouette wins. Ties broken by highest
    calinski_harabasz. Rows with n_clusters < 2 are ignored."""
    sub = metrics_df[metrics_df["algorithm"] == algo].copy()
    sub = sub[sub["n_clusters"] >= 2].copy()
    if sub.empty:
        return None, None
    sub = sub.sort_values(
        ["silhouette", "calinski_harabasz"],
        ascending=[False, False],
    )
    best = sub.iloc[0]
    return best["param"], best.to_dict()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_df = pd.read_csv(PCA_PATH)
    X = X_df.values
    log.info(f"Loaded feature matrix: {X.shape}")

    meta = pd.read_csv(META_PATH)
    if len(meta) != len(X):
        raise ValueError(
            f"Metadata rows ({len(meta)}) != feature rows ({len(X)}). "
            "Re-run prepare_features.py."
        )

    #Run sweeps
    km_metrics, km_fitted = run_kmeans_sweep(X)
    gmm_metrics, gmm_fitted = run_gmm_sweep(X)
    hd_metrics, hd_fitted = run_hdbscan_sweep(X)

    metrics = pd.concat([km_metrics, gmm_metrics, hd_metrics], ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "clustering_metrics.csv", index=False)
    log.info(f"Saved metrics table to {OUTPUT_DIR / 'clustering_metrics.csv'}")

    #Pick best config per algorithm, save models, attach labels to metadata
    best_km_k, best_km_info = pick_best(metrics, "KMeans")
    best_gmm_k, best_gmm_info = pick_best(metrics, "GMM")
    best_hd_size, best_hd_info = pick_best(metrics, "HDBSCAN")

    labels_out = meta.copy()

    if best_km_k is not None:
        km_model = km_fitted[int(best_km_k)]
        labels_out["kmeans_label"] = km_model.predict(X)
        joblib.dump(km_model, MODELS_DIR / "kmeans_best.pkl")
        log.info(f"Best K-Means: k={int(best_km_k)} (sil={best_km_info['silhouette']:.3f})")

    if best_gmm_k is not None:
        gmm_model = gmm_fitted[int(best_gmm_k)]
        labels_out["gmm_label"] = gmm_model.predict(X)
        joblib.dump(gmm_model, MODELS_DIR / "gmm_best.pkl")
        log.info(f"Best GMM: k={int(best_gmm_k)} (sil={best_gmm_info['silhouette']:.3f})")

    if best_hd_size is not None:
        hd_model = hd_fitted[int(best_hd_size)]
        #HDBSCAN — use fit_predict labels directly
        labels_out["hdbscan_label"] = hd_model.labels_
        joblib.dump(hd_model, MODELS_DIR / "hdbscan_best.pkl")
        log.info(f"Best HDBSCAN: min_size={int(best_hd_size)} "
                 f"(n_clusters={best_hd_info['n_clusters']}, "
                 f"n_noise={best_hd_info['n_noise']})")

    labels_out.to_csv(OUTPUT_DIR / "players_clustered.csv", index=False)
    log.info(f"Saved {OUTPUT_DIR / 'players_clustered.csv'} ({labels_out.shape})")


if __name__ == "__main__":
    main()
