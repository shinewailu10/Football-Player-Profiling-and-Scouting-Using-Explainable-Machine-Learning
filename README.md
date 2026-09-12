# Football Player Profiling and Scouting using Explainable Machine Learning

An MSc Artificial Intelligence project that builds an end-to-end machine-learning pipeline for football player profiling, similarity-based scouting recommendations, market-value prediction, and SHAP-based explainability, delivered through an interactive Streamlit dashboard.

**Author:** Shine Wai Lu (25948695)
**Programme:** MSc Artificial Intelligence, Manchester Metropolitan University
**Supervisor:** Dr Pavitra Kumar
**Project code:** 6G7V0007_2526_9F
**Submission date:** 11 September 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Data Sources](#data-sources)
6. [Pipeline Stages](#pipeline-stages)
7. [Running the Pipeline](#running-the-pipeline)
8. [Running the Dashboard](#running-the-dashboard)
9. [Reproducing the Results](#reproducing-the-results)
10. [Evaluation and Testing](#evaluation-and-testing)
11. [Key Findings](#key-findings)
12. [Known Limitations](#known-limitations)
13. [Directory Layout](#directory-layout)
14. [Ethical Statement](#ethical-statement)

---

## Overview

This project integrates three ML paradigms into a single explainable scouting pipeline:

- **Unsupervised clustering** to discover tactical player archetypes (K-Means, GMM, HDBSCAN compared)
- **Similarity-based recommendation** using cosine similarity in a standardised feature space, filtered by expert-labelled position
- **Supervised market-value prediction** comparing four algorithms (Linear Regression, Random Forest, XGBoost, LightGBM) both globally and per-position
- **SHAP explainability** for every prediction, both aggregate and player-level

The output is delivered as an interactive Streamlit dashboard with five pages, plus command-line tools for individual queries.

Data covers **2,935 outfield player-seasons** from the Big-5 European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1) across the 2023/24 and 2024/25 seasons.

---

## Project Structure

```
football_scouting/
├── data/
│   ├── raw/               # scraped data (FBref, Transfermarkt, StatsBomb)
│   ├── interim/           # merged intermediate files
│   └── processed/         # final analysis-ready files, models, figures
├── models/
│   └── supervised/        # fitted supervised models (32 total)
├── src/                   # all pipeline scripts
├── tests/                 # pytest unit and integration tests
├── dissertation/          # dissertation document + figures
├── requirements.txt       # Python dependencies
└── README.md              # this file
```

---

## Requirements

- **Python:** 3.12 (tested)
- **OS:** macOS or Linux (Windows should work but not tested)
- **RAM:** 8 GB minimum for training all models
- **Disk:** ~500 MB for the full data + models

Key dependencies (full list in `requirements.txt`):

- Data: `pandas`, `numpy`, `soccerdata`, `statsbombpy`, `requests`, `beautifulsoup4`, `rapidfuzz`
- ML: `scikit-learn`, `xgboost`, `lightgbm`, `hdbscan`, `umap-learn`
- Explainability: `shap`
- Dashboard: `streamlit`, `plotly`
- Testing: `pytest`

macOS users need `libomp` for LightGBM:
```bash
brew install libomp
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/shinewailu10/Football-Player-Profiling-and-Scouting-Using-Explainable-Machine-Learning.git
cd Football-Player-Profiling-and-Scouting-Using-Explainable-Machine-Learning

# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data Sources

The pipeline combines three publicly-accessible sources:

| Source | Access method | Contributes |
|---|---|---|
| **FBref** (fbref.com) | `soccerdata` library (uses seleniumbase to bypass Cloudflare) | Per-90 performance statistics (5 stat categories: standard, shooting, playing time, keeper, misc) |
| **Transfermarkt** (transfermarkt.com) | Custom scraper: `requests` + `BeautifulSoup4` with rate limiting | Market values (€) and 13-value fine-grained position labels |
| **StatsBomb Open Data** (github.com/statsbomb/open-data) | `statsbombpy` library | Event-level data (used only for reference validation) |

All data collection is subject to fair-use terms of each provider. This project is academic research only; no data is redistributed or repackaged.

---

## Pipeline Stages

The pipeline is broken into five stages:

1. **Data collection** — scrape FBref, Transfermarkt, StatsBomb
2. **Feature engineering** — merge sources, filter, compute per-90 stats, add composite indicators
3. **Clustering** — compare K-Means, GMM, HDBSCAN; select K-Means k=4
4. **Similarity recommender** — cosine similarity with position filtering
5. **Supervised prediction** — 4 models × 8 scopes (global + 7 position families), 5-fold cross-validation, SHAP explanations

---

## Running the Pipeline

Individual scripts can be run in sequence to reproduce the full pipeline. Each script's outputs feed into the next.

### Stage 1 — Data collection

```bash
# Scrape FBref via soccerdata (opens headless Chrome)
python3 src/scrape_fbref.py

# Scrape Transfermarkt (rate-limited, ~30 minutes for full run)
python3 src/scrape_transfermarkt.py

# Download StatsBomb open data
python3 src/scrape_statsbomb.py
```

### Stage 2 — Feature engineering

```bash
# Match FBref rows to Transfermarkt using RapidFuzz
python3 src/match_transfermarkt.py

# Build the master feature matrix
python3 src/build_features.py

# Prepare for downstream ML (standardise, PCA, save)
python3 src/prepare_features.py
```

### Stage 3 — Clustering

```bash
# Run K-Means, GMM, HDBSCAN sweep and pick winner
python3 src/run_clustering.py

# Generate cluster interpretation summary and representative players
python3 src/interpret_clusters.py
```

### Stage 4 — Similarity recommender

The recommender is a library module (`src/recommender.py`) used by the CLI and dashboard. To query it from the command line:

```bash
python3 src/query_recommender.py "Bukayo Saka" --top-n 10 --same-tm-position
python3 src/query_recommender.py "Rodri" --team "Manchester City" --top-n 10
```

### Stage 5 — Supervised prediction + SHAP

```bash
# Train 4 models × 8 scopes with 5-fold CV
python3 src/train_supervised.py

# Generate SHAP explanations (global + per-position + local case studies)
python3 src/explain_supervised.py

# Query the supervised model for a single player
python3 src/predict_player.py "Erling Haaland"
python3 src/predict_player.py "Rodri" --team "Manchester City" --use-position-model
```

---

## Running the Dashboard

The Streamlit dashboard exposes the entire pipeline through five interactive pages:

```bash
streamlit run src/dashboard.py
```

Opens automatically at `http://localhost:8501`. Pages:

- **Player Profile** — search a player, view identity + stats vs family average
- **Similar Players** — recommender with filters (position, age, market value, exclude same team/league)
- **Value Prediction** — global + per-position predictions side-by-side with SHAP force plot
- **Model Validation** — interactive retrospective transfer test (pick any two players, system reports hit/miss with rank, similarity, and side-by-side stats)
- **Methodology & Findings** — project summary, key metrics table, known limitations

Stop with `Ctrl+C` in the terminal.

---

## Reproducing the Results

To reproduce all reported results from scratch:

```bash
# Run the full pipeline end to end
python3 src/scrape_fbref.py
python3 src/scrape_transfermarkt.py
python3 src/scrape_statsbomb.py
python3 src/match_transfermarkt.py
python3 src/build_features.py
python3 src/prepare_features.py
python3 src/run_clustering.py
python3 src/interpret_clusters.py
python3 src/train_supervised.py
python3 src/explain_supervised.py

# Generate evaluation tables and figures
python3 src/evaluate_full.py
python3 src/final_figures.py
python3 src/final_figures_extended.py
```

Total runtime is approximately 45 minutes on a standard laptop (excluding the Transfermarkt scrape, which is rate-limited and takes ~30 minutes on its own).

All random seeds are fixed to 42 throughout for reproducibility.

---

## Evaluation and Testing

### Unit tests

```bash
pytest tests/ -v
```

Tests cover:

- Name normalisation (accents, punctuation, hyphens)
- Position family filtering logic
- Log/exponential roundtrip stability
- Recommender integration tests (player lookup, similarity ranking, filters)

### Full evaluation

`evaluate_full.py` produces dissertation-ready evaluation outputs:

- `data/processed/eval/model_comparison_all.csv` — full 4-model × 8-scope metrics
- `data/processed/eval/prediction_errors_top20.csv` — top-20 predictions with SHAP drivers
- `data/processed/eval/recommender_precision_all_modes.csv` — retrospective transfer study
- `data/processed/eval/per_position_errors.csv` — per-position family error breakdown
- `data/processed/eval/summary.md` — human-readable summary of everything above

### Figures

Twelve publication-quality PNG figures are generated to `data/processed/figures/`:

- `fig1_cluster_umap.png` — UMAP projection of clusters
- `fig2_shap_global.png` — global SHAP feature importance
- `fig3_prediction_scatter.png` — predicted vs actual market values
- `fig4_per_position_r2.png` — R² by position family
- `fig5_recommender_examples.png` — similarity distributions for three case-study players
- `fig6_pipeline_flow.png` — pipeline architecture diagram
- `fig7_market_value_distribution.png` — raw vs log-transformed target
- `fig8_position_family_counts.png` — dataset composition
- `fig9_feature_correlation.png` — feature correlation heatmap
- `fig10_cluster_selection.png` — validity metrics across k
- `fig11_cluster_profiles.png` — cluster z-scored feature means
- `fig13_model_comparison_bars.png` — 4-model comparison across all scopes

---

## Key Findings

1. **Per-position models beat the global model for output-heavy positions.** Forward R² = 0.570 vs global 0.404; Winger 0.547; Attacking Midfield 0.436.

2. **Feature-set attacking bias appears across three independent analyses.** In clustering (attackers separate more sharply than defenders), in the recommender (Saka's peers reach cosine 0.90+; Rodri's only 0.29–0.71), and in supervised prediction (Forwards R² 0.57 vs Defensive Midfielders 0.31). This is a substantive methodological finding — same pattern documented three ways.

3. **Retrospective transfer study.** The recommender identified Kylian Mbappé at rank 10 (cosine 0.958) as a top statistical peer for Vinicius Júnior using only pre-transfer 2023/24 data. This is Real Madrid's actual summer 2024 signing. Under strict position filtering the match is suppressed (Mbappé is labelled Centre-Forward vs Vinicius as Left Winger) — exposing a real trade-off between tactical coherence and versatile-player recall.

4. **Linear Regression matches gradient boosting on this task.** On six of seven per-position scopes, Linear Regression achieves within 0.02 R² of Random Forest, suggesting the underlying relationship is close to log-linear. This is unusual and worth further investigation.

5. **Systematic under-prediction of elite players.** Every player over €100m is under-predicted (Yamal €200m → €124m; Mbappé €180m → €130m; Rodri €120m → €61m). Regression to the mean on a heavy-tailed target — well-documented statistical phenomenon.

---

## Known Limitations

- **Feature-set attacking bias.** The soccerdata FBref endpoint exposes only 5 stat categories at player level. Passing volume, defensive coverage by zone, and ball progression are not available, which propagates through the pipeline and specifically limits performance on defensive positions.

- **Heavy-tail regression to the mean.** Players above €100m are systematically under-predicted because the training set contains few such examples.

- **Small retrospective transfer sample.** Only three verified 2024–25 transfer cases were tested. A larger sample across multiple transfer windows would strengthen the validation.

- **Big-5 leagues only.** No lower divisions or non-European leagues. Generalisation to other markets is not tested.

- **Two seasons only.** Long-term trend analysis and multi-season predictive modelling are not attempted.

- **Rate-limited data collection.** Full scrape takes ~30 minutes and depends on remote source availability. Historic pre-2023/24 seasons were not scraped due to time constraints.

Discussed at length in Chapter 9.5 of the dissertation.

---

## Directory Layout

```
football_scouting/
├── README.md                                # this file
├── requirements.txt                         # Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── fbref/                          # 10 FBref stat CSVs (5 cats × 2 seasons)
│   │   ├── transfermarkt/                  # 10 Transfermarkt scrape CSVs
│   │   └── statsbomb/                      # StatsBomb open data cache
│   │
│   ├── interim/
│   │   ├── fbref_merged.csv                # 5 stat tables joined per player-season
│   │   ├── transfermarkt_all_leagues.csv   # concatenated TM data
│   │   └── unmatched_players.csv           # 21 FBref rows without TM match
│   │
│   └── processed/
│       ├── players_processed.csv           # 2,935-row analysis dataset
│       ├── players_clustered.csv           # + K-Means cluster labels
│       ├── feature_matrix_pca.csv          # 9-D PCA feature matrix
│       ├── clustering_metrics.csv          # silhouette/DB/CH across k
│       ├── cluster_profiles.csv            # mean feature values per cluster
│       ├── cluster_summary.md              # interpretation
│       ├── cluster_representatives.csv     # top-N players per cluster
│       ├── supervised_metrics.csv          # 32-row full metrics table
│       │
│       ├── shap/
│       │   ├── global_importance.csv       # top features by mean |SHAP|
│       │   ├── global_importance_barplot.png
│       │   ├── per_position_importance.csv
│       │   ├── per_position_heatmap.png
│       │   ├── local_*.png                 # per-player force plots
│       │   └── shap_report.md
│       │
│       ├── eval/
│       │   ├── model_comparison_all.csv
│       │   ├── prediction_errors_top20.csv
│       │   ├── recommender_precision_all_modes.csv
│       │   ├── per_position_errors.csv
│       │   └── summary.md
│       │
│       └── figures/                        # 12 publication-quality PNG figures
│
├── models/
│   └── supervised/
│       ├── preprocessing.pkl               # feature list + config
│       └── {scope}_{algorithm}.pkl         # 32 fitted models
│
├── src/
│   ├── scrape_fbref.py                     # Stage 1a: FBref via soccerdata
│   ├── scrape_transfermarkt.py             # Stage 1b: custom TM scraper
│   ├── scrape_statsbomb.py                 # Stage 1c: StatsBomb open data
│   ├── match_transfermarkt.py              # merge FBref ↔ TM (RapidFuzz)
│   ├── build_features.py                   # compute per-90, composites
│   ├── prepare_features.py                 # standardise, PCA
│   ├── run_clustering.py                   # K-Means, GMM, HDBSCAN comparison
│   ├── interpret_clusters.py               # generate cluster profiles
│   ├── recommender.py                      # Recommender class (library module)
│   ├── query_recommender.py                # CLI wrapper for recommender
│   ├── train_supervised.py                 # 4 models × 8 scopes with 5-fold CV
│   ├── explain_supervised.py               # generate SHAP outputs
│   ├── predict_player.py                   # CLI: predict for one player
│   ├── evaluate_full.py                    # comprehensive evaluation
│   ├── final_figures.py                    # generate figures 1-5
│   ├── final_figures_extended.py           # generate figures 6-13
│   └── dashboard.py                        # Streamlit web app (5 pages)
│
├── tests/
│   └── test_pipeline.py                    # pytest unit + integration tests
│
│
└── dissertation/                           # written document + assets
```

---

## Ethical Statement

This project is academic research conducted under Manchester Metropolitan University's ethical framework. All data used is publicly accessible via the sources listed above and is subject to their respective terms of service. No personal data beyond publicly reported professional player statistics and market valuations is collected.

The system produces recommendations for research demonstration only; no automated scouting decisions are transmitted to or acted upon by any real club, agency, or platform.

Player market values are third-party crowd-sourced estimates published on Transfermarkt and should not be interpreted as actual transfer fees.

Ethical approval was obtained via the MMU EthOS process.

---

## Citation

If you reference this project in academic work:

> Lu, S. W. (2026). *Football Player Profiling and Scouting using Explainable Machine Learning*. MSc dissertation, Manchester Metropolitan University.

---

## Acknowledgements

- **Dr Pavitra Kumar** — project supervisor
- **soccerdata** (Verstraete) — for the FBref scraping wrapper
- **StatsBomb** — for making event-level football data publicly available
- **The SHAP team** (Lundberg & Lee) — for the interpretability framework
- Open-source Python community

---

## Repository

The full source code, trained models, generated figures, and processed data are available at:
[https://github.com/shinewailu10/Football-Player-Profiling-and-Scouting-Using-Explainable-Machine-Learning](https://github.com/shinewailu10/Football-Player-Profiling-and-Scouting-Using-Explainable-Machine-Learning)

*Public repository — no login required. Clone and follow the setup instructions above.*
