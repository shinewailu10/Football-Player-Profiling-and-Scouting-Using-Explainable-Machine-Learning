"""
scrape_fbref.py
---------------
Scrapes per-90 player statistics from FBref for the top-5 European leagues
across the 2023/24 and 2024/25 seasons.

Uses the `soccerdata` library. The "Big 5 European Leagues Combined" endpoint
only supports 5 stat types, so scrape each of the five leagues individually
and concatenate the results — this gives all 7 stat types.
"""

import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

#Configuration
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASONS = ["2023-2024", "2024-2025"]
STAT_TYPES = [
    "standard",
    "shooting",
    "playing_time",
    "keeper",
    "misc",
]
OUTPUT_DIR = Path("data/raw/fbref")

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """FBref tables have multi-level columns like ('Expected', 'xG').
    Flatten them into single-level names like 'Expected_xG'.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col if c and str(c) != "nan").strip("_")
            for col in df.columns
        ]
    return df


def scrape_one(stat_type: str, season: str) -> pd.DataFrame:
    """Scrape one stat_type for one season across all five leagues."""
    log.info(f"Scraping {stat_type} for {season}...")

    frames = []
    for league in LEAGUES:
        log.info(f"  {league}")
        try:
            fbref = sd.FBref(leagues=league, seasons=season)
            df_league = fbref.read_player_season_stats(stat_type=stat_type)
            frames.append(df_league)
        except Exception as e:
            log.warning(f"    skipped {league}: {e}")
            continue

    if not frames:
        raise RuntimeError(f"No data scraped for {stat_type} {season}")

    df = pd.concat(frames)
    df = df.reset_index()
    df = flatten_columns(df)

    log.info(f"  -> total {len(df)} rows, {len(df.columns)} columns")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    for stat_type in STAT_TYPES:
        for season in SEASONS:
            output_path = OUTPUT_DIR / f"{stat_type}_{season}.csv"
            if output_path.exists():
                log.info(f"Skipping (already saved): {output_path.name}")
                continue

            try:
                df = scrape_one(stat_type, season)
                df.to_csv(output_path, index=False)
                log.info(f"  saved to {output_path}")
            except Exception as e:
                log.error(f"  FAILED {stat_type} {season}: {e}")

    log.info("Done.")


if __name__ == "__main__":
    main()
