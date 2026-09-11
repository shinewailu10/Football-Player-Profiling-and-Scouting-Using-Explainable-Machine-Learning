"""
match_transfermarkt.py  (v2 — now carries TM position)
------------------------------------------------------
Fuzzy-matches FBref players to Transfermarkt players within (league, season)
blocks, and attaches market_value_eur AND position from Transfermarkt.

carried market_value_eur, tm_position (fine-grained
13-value label) which is used by build_features.py to add a position_family
column for per-position modelling and better recommender filtering.

Input:
  data/interim/fbref_clean.csv
  data/raw/transfermarkt/*.csv
Output:
  data/interim/fbref_with_market_value.csv
  data/interim/unmatched_players.csv
"""

import logging
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

FBREF_PATH = Path("data/interim/fbref_clean.csv")
TM_DIR = Path("data/raw/transfermarkt")
OUTPUT_DIR = Path("data/interim")

LEAGUE_MAP = {
    "ENG-Premier League": "GB1",
    "ESP-La Liga": "ES1",
    "GER-Bundesliga": "L1",
    "ITA-Serie A": "IT1",
    "FRA-Ligue 1": "FR1",
}

SEASON_MAP = {
    2324: 2023, 2425: 2024,
    "2023-2024": 2023, "2024-2025": 2024,
}

MATCH_THRESHOLD = 85

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalised = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in normalised if not unicodedata.combining(c))
    return stripped.lower().replace("'", "").replace("-", " ").strip()


def load_transfermarkt() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(TM_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["name_normalised"] = df["name"].apply(normalise_name)
    log.info(f"Loaded Transfermarkt: {len(df)} rows across {len(frames)} files")
    return df


def match_block(fbref_block: pd.DataFrame, tm_block: pd.DataFrame) -> pd.DataFrame:
    fbref_block = fbref_block.copy()
    fbref_block["_fbref_name_norm"] = fbref_block["player"].apply(normalise_name)

    if tm_block.empty:
        fbref_block["market_value_eur"] = pd.NA
        fbref_block["tm_position"] = pd.NA
        fbref_block["tm_matched_name"] = pd.NA
        fbref_block["tm_match_score"] = pd.NA
        return fbref_block

    tm_choices = tm_block["name_normalised"].tolist()
    tm_lookup = tm_block.set_index("name_normalised")

    market_values, tm_positions, matched_names, match_scores = [], [], [], []

    for fbref_name in fbref_block["_fbref_name_norm"]:
        if not fbref_name:
            market_values.append(pd.NA)
            tm_positions.append(pd.NA)
            matched_names.append(pd.NA)
            match_scores.append(pd.NA)
            continue

        result = process.extractOne(fbref_name, tm_choices, scorer=fuzz.WRatio)
        if result is None:
            market_values.append(pd.NA)
            tm_positions.append(pd.NA)
            matched_names.append(pd.NA)
            match_scores.append(pd.NA)
            continue

        matched_norm, score, _ = result
        if score >= MATCH_THRESHOLD:
            tm_row = tm_lookup.loc[matched_norm]
            if isinstance(tm_row, pd.DataFrame):
                tm_row = tm_row.iloc[0]
            market_values.append(tm_row["market_value_eur"])
            tm_positions.append(tm_row["position"])
            matched_names.append(tm_row["name"])
            match_scores.append(score)
        else:
            market_values.append(pd.NA)
            tm_positions.append(pd.NA)
            matched_names.append(pd.NA)
            match_scores.append(score)

    fbref_block["market_value_eur"] = market_values
    fbref_block["tm_position"] = tm_positions
    fbref_block["tm_matched_name"] = matched_names
    fbref_block["tm_match_score"] = match_scores
    return fbref_block.drop(columns=["_fbref_name_norm"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fbref = pd.read_csv(FBREF_PATH)
    log.info(f"Loaded FBref: {len(fbref)} rows")

    tm = load_transfermarkt()

    fbref["_tm_league"] = fbref["league"].map(LEAGUE_MAP)
    fbref["_tm_season"] = fbref["season"].map(SEASON_MAP)

    matched_frames = []
    for (tm_league, tm_season), fbref_block in fbref.groupby(["_tm_league", "_tm_season"]):
        tm_block = tm[(tm["league"] == tm_league) & (tm["season"] == tm_season)]
        log.info(f"  Matching {tm_league} {tm_season}: "
                 f"{len(fbref_block)} FBref x {len(tm_block)} TM candidates")
        matched_frames.append(match_block(fbref_block, tm_block))

    combined = pd.concat(matched_frames, ignore_index=True)
    combined = combined.drop(columns=["_tm_league", "_tm_season"])

    n_matched = combined["market_value_eur"].notna().sum()
    n_total = len(combined)
    log.info(f"Match rate: {n_matched}/{n_total} ({100 * n_matched / n_total:.1f}%)")

    n_tm_pos = combined["tm_position"].notna().sum()
    log.info(f"TM position attached: {n_tm_pos}/{n_total}")

    output_path = OUTPUT_DIR / "fbref_with_market_value.csv"
    combined.to_csv(output_path, index=False)
    log.info(f"Saved to {output_path}")

    unmatched = combined[combined["market_value_eur"].isna()][
        ["league", "season", "team", "player", "pos", "tm_match_score"]
    ]
    unmatched_path = OUTPUT_DIR / "unmatched_players.csv"
    unmatched.to_csv(unmatched_path, index=False)
    log.info(f"Unmatched players logged to {unmatched_path} ({len(unmatched)} rows)")


if __name__ == "__main__":
    main()
