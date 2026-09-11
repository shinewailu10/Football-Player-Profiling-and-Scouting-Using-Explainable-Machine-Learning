"""
build_features.py
--------------------------------------------------------------
Takes the matched FBref+Transfermarkt dataset and:
- computes per-90 metrics from the misc-table totals
- constructs composite indicators
- ADDS: tm_position (13-value fine-grained Transfermarkt position)
- ADDS: position_family (7-family grouping for per-position modelling)
- keeps a tidy final schema

Input:  data/interim/fbref_with_market_value.csv
Output: data/processed/players_processed.csv

Usage:
    python3 src/build_features.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/interim/fbref_with_market_value.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "players_processed.csv"

#Group Transfermarkt's 13 positions into 7 broader families.
POSITION_FAMILY = {
    "Goalkeeper":         "Goalkeeper",       
    "Centre-Back":        "Centre-Back",
    "Right-Back":         "Full-Back",
    "Left-Back":          "Full-Back",
    "Defensive Midfield": "Defensive Midfield",
    "Central Midfield":   "Central Midfield",
    "Right Midfield":     "Central Midfield",
    "Left Midfield":      "Central Midfield",
    "Attacking Midfield": "Attacking Midfield",
    "Right Winger":       "Winger",
    "Left Winger":        "Winger",
    "Centre-Forward":     "Forward",
    "Second Striker":     "Forward",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_misc_per90(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minutes = df["standard__Playing Time_Min"].fillna(0)
    nineties = (minutes / 90.0).replace(0, np.nan)

    misc_to_per90 = {
        "misc__Performance_Int": "interceptions_p90",
        "misc__Performance_TklW": "tackles_won_p90",
        "misc__Performance_Fls": "fouls_committed_p90",
        "misc__Performance_Fld": "fouls_drawn_p90",
        "misc__Performance_Crs": "crosses_p90",
        "misc__Performance_Off": "offsides_p90",
    }
    for src, dst in misc_to_per90.items():
        if src in df.columns:
            df[dst] = df[src] / nineties
    return df


def compute_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col: str) -> pd.Series:
        return df[col].fillna(0) if col in df.columns else pd.Series(0, index=df.index)

    df["attacking_index"] = (
        safe("standard__Per 90 Minutes_G+A") + 0.5 * safe("shooting__Standard_Sh/90")
    )
    df["creativity_index"] = (
        safe("standard__Per 90 Minutes_Ast") + 0.3 * safe("crosses_p90")
    )
    df["defensive_index"] = safe("interceptions_p90") + safe("tackles_won_p90")
    df["discipline_index"] = safe("fouls_drawn_p90") - safe("fouls_committed_p90")

    return df


def add_position_family(df: pd.DataFrame) -> pd.DataFrame:
    """Map fine-grained tm_position (13 values) to broader position_family (7 values).
    Rows with missing tm_position get NaN in position_family."""
    df = df.copy()
    df["position_family"] = df["tm_position"].map(POSITION_FAMILY)
    n_unknown = df["tm_position"].notna().sum() - df["position_family"].notna().sum()
    if n_unknown > 0:
        unknown_vals = df[df["tm_position"].notna() & df["position_family"].isna()]["tm_position"].unique()
        log.warning(f"{n_unknown} rows had tm_position values not in POSITION_FAMILY map: {unknown_vals}")
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    #Identifier columns
    id_cols = ["league", "season", "team", "player", "nation", "pos", "age", "born"]

    #Transfermarkt-derived position columns
    tm_pos_cols = ["tm_position", "position_family"]

    context_cols = ["standard__Playing Time_Min", "standard__Playing Time_MP",
                    "standard__Playing Time_Starts"]

    p90_features = [
        "standard__Per 90 Minutes_Gls",
        "standard__Per 90 Minutes_Ast",
        "standard__Per 90 Minutes_G+A",
        "standard__Per 90 Minutes_G-PK",
        "standard__Per 90 Minutes_G+A-PK",
        "shooting__Standard_Sh/90",
        "shooting__Standard_SoT/90",
        "shooting__Standard_SoT%",
        "shooting__Standard_G/Sh",
        "shooting__Standard_G/SoT",
    ]

    misc_p90 = [
        "interceptions_p90", "tackles_won_p90",
        "fouls_committed_p90", "fouls_drawn_p90",
        "crosses_p90", "offsides_p90",
    ]

    composites = ["attacking_index", "creativity_index",
                  "defensive_index", "discipline_index"]

    match_cols = ["tm_matched_name", "tm_match_score"]
    target_cols = ["market_value_eur"]

    ordered = (id_cols + tm_pos_cols + context_cols + p90_features
               + misc_p90 + composites + match_cols + target_cols)
    kept = [c for c in ordered if c in df.columns]

    rename_map = {
        "standard__Playing Time_Min": "minutes",
        "standard__Playing Time_MP": "matches_played",
        "standard__Playing Time_Starts": "starts",
        "standard__Per 90 Minutes_Gls": "goals_p90",
        "standard__Per 90 Minutes_Ast": "assists_p90",
        "standard__Per 90 Minutes_G+A": "goals_assists_p90",
        "standard__Per 90 Minutes_G-PK": "non_penalty_goals_p90",
        "standard__Per 90 Minutes_G+A-PK": "np_goals_assists_p90",
        "shooting__Standard_Sh/90": "shots_p90",
        "shooting__Standard_SoT/90": "shots_on_target_p90",
        "shooting__Standard_SoT%": "shots_on_target_pct",
        "shooting__Standard_G/Sh": "goals_per_shot",
        "shooting__Standard_G/SoT": "goals_per_shot_on_target",
    }

    return df[kept].rename(columns=rename_map)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} cols from {INPUT_PATH}")

    df = compute_misc_per90(df)
    df = compute_composite_indicators(df)
    df = add_position_family(df)
    df = select_final_columns(df)

    log.info(f"Final shape: {df.shape}")

    n_matched = df["market_value_eur"].notna().sum()
    log.info(f"Players with market value: {n_matched} ({100 * n_matched / len(df):.1f}%)")

    if "tm_position" in df.columns:
        log.info(f"Players with tm_position: {df['tm_position'].notna().sum()}")
        log.info("Position family breakdown:")
        for family, count in df["position_family"].value_counts().items():
            log.info(f"  {family}: {count}")

    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
