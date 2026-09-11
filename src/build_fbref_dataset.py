"""
build_fbref_dataset.py
----------------------
Merges the 5 FBref stat tables per season into a single wide DataFrame,
concatenates the two seasons, aggregates mid-season transferees, filters
goalkeepers and low-minutes players.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/fbref")
OUTPUT_DIR = Path("data/interim")
OUTPUT_PATH = OUTPUT_DIR / "fbref_clean.csv"

SEASONS = ["2023-2024", "2024-2025"]
OUTFIELD_STAT_TYPES = ["standard", "shooting", "playing_time", "misc"]
ID_COLS = ["league", "season", "team", "player", "nation", "pos", "age", "born"]
MIN_MINUTES = 900

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_stat_type(stat_type: str, season: str) -> pd.DataFrame:
    path = RAW_DIR / f"{stat_type}_{season}.csv"
    df = pd.read_csv(path)
    rename_map = {c: f"{stat_type}__{c}" for c in df.columns if c not in ID_COLS}
    return df.rename(columns=rename_map)


def merge_stat_types_for_season(season: str) -> pd.DataFrame:
    frames = [load_stat_type(st, season) for st in OUTFIELD_STAT_TYPES]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=ID_COLS, how="outer")
    log.info(f"  {season}: merged shape {merged.shape}")
    return merged


def aggregate_transferees(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, player, born). Sum counting stats, take identifier
    fields (team, age, pos, nation, league) from the primary club (most minutes),
    then re-compute per-90 and percentage columns from the summed totals."""
    log.info("Aggregating mid-season transferees...")
    log.info(f"  before aggregation: {len(df)} rows")

    minutes_col = "standard__Playing Time_Min"
    group_keys = ["season", "player", "born"]

    df = df.copy()
    for k in group_keys:
        df[k] = df[k].fillna("__missing__")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    per90_cols = [c for c in numeric_cols if "/90" in c or "Per 90" in c or "_90s" in c]
    pct_cols = [c for c in numeric_cols if c.endswith("%")]
    recompute_cols = set(per90_cols + pct_cols)

    #Identifier columns to pick from primary club, includes 'age' since it's numeric
    id_from_primary = [c for c in ID_COLS if c not in group_keys]

    #Columns to sum: numeric minus per-90/% minus group keys minus id cols
    sum_cols = [
        c for c in numeric_cols
        if c not in recompute_cols
        and c not in group_keys
        and c not in id_from_primary
    ]

    #Sum counting stats per group
    sums = df.groupby(group_keys, as_index=False)[sum_cols].sum(min_count=1)

    #Primary club row: player's row with most minutes
    df["_minutes_for_sort"] = df[minutes_col].fillna(0)
    primary = (
        df.sort_values("_minutes_for_sort", ascending=False)
          .drop_duplicates(subset=group_keys, keep="first")
    )
    primary = primary[group_keys + id_from_primary]

    merged = primary.merge(sums, on=group_keys, how="left")
    log.info(f"  after aggregation: {len(merged)} rows")

    #Re-compute per-90 and percentage columns from summed totals
    nineties = (merged[minutes_col] / 90.0).replace(0, np.nan)

    per90_recipes = [
        ("standard__Per 90 Minutes_Gls",    "standard__Performance_Gls"),
        ("standard__Per 90 Minutes_Ast",    "standard__Performance_Ast"),
        ("standard__Per 90 Minutes_G+A",    "standard__Performance_G+A"),
        ("standard__Per 90 Minutes_G-PK",   "standard__Performance_G-PK"),
        ("standard__Per 90 Minutes_G+A-PK", None),
        ("shooting__Standard_Sh/90",  "shooting__Standard_Sh"),
        ("shooting__Standard_SoT/90", "shooting__Standard_SoT"),
    ]

    for p90_col, total_col in per90_recipes:
        if p90_col == "standard__Per 90 Minutes_G+A-PK":
            g_pk = merged.get("standard__Performance_G-PK", pd.Series(0, index=merged.index))
            ast = merged.get("standard__Performance_Ast", pd.Series(0, index=merged.index))
            merged[p90_col] = (g_pk.fillna(0) + ast.fillna(0)) / nineties
        elif total_col and total_col in merged.columns:
            merged[p90_col] = merged[total_col] / nineties

    #Shooting percentages
    sh = merged.get("shooting__Standard_Sh", pd.Series(0, index=merged.index))
    sot = merged.get("shooting__Standard_SoT", pd.Series(0, index=merged.index))
    gls = merged.get("shooting__Standard_Gls", pd.Series(0, index=merged.index))

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["shooting__Standard_SoT%"] = 100 * sot / sh.replace(0, np.nan)
        merged["shooting__Standard_G/Sh"] = gls / sh.replace(0, np.nan)
        merged["shooting__Standard_G/SoT"] = gls / sot.replace(0, np.nan)

    merged = merged.drop(columns=["_minutes_for_sort"], errors="ignore")

    for k in group_keys:
        merged[k] = merged[k].replace("__missing__", pd.NA)

    return merged


def filter_players(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    df = df[~df["pos"].fillna("").str.contains("GK", regex=False)]
    log.info(f"  filtered goalkeepers: {n_start} -> {len(df)}")

    n_after_gk = len(df)
    minutes_col = "standard__Playing Time_Min"
    if minutes_col in df.columns:
        df = df[df[minutes_col].fillna(0) >= MIN_MINUTES]
        log.info(f"  filtered <{MIN_MINUTES} minutes: {n_after_gk} -> {len(df)}")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Merging FBref stat tables per season...")
    season_frames = [merge_stat_types_for_season(s) for s in SEASONS]

    combined = pd.concat(season_frames, ignore_index=True)
    log.info(f"Combined seasons: {combined.shape}")

    combined = aggregate_transferees(combined)

    log.info("Filtering players...")
    combined = filter_players(combined)

    combined.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(combined)} player-seasons to {OUTPUT_PATH}")
    log.info(f"Columns: {len(combined.columns)}")


if __name__ == "__main__":
    main()
