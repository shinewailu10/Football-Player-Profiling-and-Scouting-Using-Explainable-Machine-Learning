"""
download_statsbomb.py
---------------------
Downloads StatsBomb Open Data and provides simple loaders for competitions,
matches, and events.

StatsBomb Open Data is a public GitHub repo. This script uses the
`statsbombpy` library, which handles fetching and JSON parsing for you.

Output: cached JSON files in data/raw/statsbomb/
        (statsbombpy caches under its own directory by default; this script
         mirrors the useful high-level tables into CSVs for convenience.)

Usage:
    python src/download_statsbomb.py
"""

import logging
from pathlib import Path

import pandas as pd
from statsbombpy import sb

#Configuration
OUTPUT_DIR = Path("data/raw/statsbomb")


TARGET_COMPETITIONS: list[tuple[int, int]] = [
    (43, 106),   #FIFA World Cup 2022
]

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    #Save the competition index.
    log.info("Fetching competitions index...")
    competitions = sb.competitions()
    competitions.to_csv(OUTPUT_DIR / "competitions.csv", index=False)
    log.info(f"  saved {len(competitions)} competition-seasons.")

    #For each requested competition, fetch its matches and events.
    for comp_id, season_id in TARGET_COMPETITIONS:
        log.info(f"Fetching matches for competition {comp_id}, season {season_id}...")
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        matches_path = OUTPUT_DIR / f"matches_{comp_id}_{season_id}.csv"
        matches.to_csv(matches_path, index=False)
        log.info(f"  saved {len(matches)} matches to {matches_path}")

        #Pull a handful of events per match. Full events for every match can be large — you probably only need a subset for case studies.
        events_frames = []
        for match_id in matches["match_id"].tolist()[:5]:  
            try:
                ev = sb.events(match_id=match_id)
                ev["match_id"] = match_id
                events_frames.append(ev)
            except Exception as e:
                log.warning(f"  events for match {match_id} failed: {e}")

        if events_frames:
            all_events = pd.concat(events_frames, ignore_index=True)
            events_path = OUTPUT_DIR / f"events_{comp_id}_{season_id}_sample.csv"
            all_events.to_csv(events_path, index=False)
            log.info(f"  saved {len(all_events)} events to {events_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
