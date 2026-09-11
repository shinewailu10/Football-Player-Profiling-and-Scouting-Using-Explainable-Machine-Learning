"""
prepare_features.py
-------------------
Prepares the feature matrix for clustering:
- Selects the numeric per-90 features (drops identifiers, composites, target)
- Imputes NaN with 0 (players who took 0 shots have NaN in shot-related rates)
- Standardises features (mean=0, std=1)
- Applies PCA to reduce dimensionality (keeps components explaining >= 85% variance)

Input:  data/processed/players_processed.csv
Output:
  data/processed/feature_matrix_pca.csv   (PCA-reduced features, one row per player)
  data/processed/feature_matrix_raw.csv   (standardised but not PCA-reduced, for interpretation)
  data/processed/players_metadata.csv     (id columns + market value, one row per player,
                                           in the same order as the feature matrices)
  models/scaler.pkl                       (fitted StandardScaler)
  models/pca.pkl                          (fitted PCA)

Usage:
    python3 src/prepare_features.py
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("data/processed/players_processed.csv")
OUTPUT_DIR = Path("data/processed")
MODELS_DIR = Path("models")

#Metadata columns (identifiers + target) — not used for clustering
META_COLS = [
    "league", "season", "team", "player", "nation", "pos", "age", "born",
    "minutes", "matches_played", "starts",
    "tm_matched_name", "tm_match_score", "market_value_eur",
]

#Composite indicators — tested including them to help sparse feature set
COMPOSITE_COLS = []

#Minimum cumulative variance to retain when choosing PCA components
PCA_VARIANCE_THRESHOLD = 0.95

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    log.info(f"Loaded {len(df)} rows from {INPUT_PATH}")

    #Identify feature columns: everything numeric that isn't metadata or composite
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c not in META_COLS and c not in COMPOSITE_COLS
    ]
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    #Drop rows where all features are NaN
    valid = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    if len(valid) < len(df):
        log.warning(f"Dropped {len(df) - len(valid)} rows with all-NaN features")

    X = valid[feature_cols].copy()

    #Impute NaN with 0 (a player with 0 shots has NaN in goals_per_shot; setting it to 0 is the right zero-shot baseline).
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        log.info(f"Imputing {n_missing} NaN values with 0")
        X = X.fillna(0)

    #Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"Standardised features. Shape: {X_scaled.shape}")

    #PCA: fit full to find #components meeting the variance threshold
    pca_full = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= PCA_VARIANCE_THRESHOLD) + 1)
    log.info(f"Selected {n_components} PCA components "
             f"(cumulative variance: {cumvar[n_components - 1]:.3f})")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    #Save
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
    meta_df = valid[[c for c in META_COLS if c in valid.columns]]

    scaled_df.to_csv(OUTPUT_DIR / "feature_matrix_raw.csv", index=False)
    pca_df.to_csv(OUTPUT_DIR / "feature_matrix_pca.csv", index=False)
    meta_df.to_csv(OUTPUT_DIR / "players_metadata.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "pca.pkl")

    log.info("Saved:")
    log.info(f"  feature_matrix_raw.csv  {scaled_df.shape}")
    log.info(f"  feature_matrix_pca.csv  {pca_df.shape}")
    log.info(f"  players_metadata.csv    {meta_df.shape}")
    log.info(f"  models/scaler.pkl, models/pca.pkl")


if __name__ == "__main__":
    main()
