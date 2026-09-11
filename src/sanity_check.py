"""
sanity_check.py
---------------
Runs the sanity checks from Section 5 of the Data Collection Guide.
Verifies that the raw data is present, plausible, and ready for feature
engineering.

Usage:
    python src/sanity_check.py
"""

import logging
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def check_fbref() -> None:
    log.info("=" * 60)
    log.info("FBref checks")
    log.info("=" * 60)

    fbref_dir = RAW / "fbref"
    if not fbref_dir.exists():
        log.error("  data/raw/fbref/ does not exist. Run scrape_fbref.py first.")
        return

    for season in ("2023-2024", "2024-2025"):
        standard_path = fbref_dir / f"standard_{season}.csv"
        if not standard_path.exists():
            log.error(f"  MISSING: {standard_path}")
            continue

        df = pd.read_csv(standard_path)
        log.info(f"  {season} standard: {len(df)} rows")

        #Row count sanity
        if len(df) < 1500:
            log.warning(f"    LOW row count ({len(df)}). Expected ~2500.")
        elif len(df) > 4000:
            log.warning(f"    HIGH row count ({len(df)}). Expected ~2500.")

        #Duplicates
        name_col = next((c for c in df.columns if c.lower() == "player"), None)
        if name_col:
            duplicates = df.duplicated(subset=[name_col]).sum()
            if duplicates > 0:
                log.warning(f"    {duplicates} duplicate player rows.")

        #Well-known players spot-check
        expected_players = ["Erling Haaland", "Jude Bellingham", "Rodri",
                            "Bukayo Saka", "Kylian Mbappé"]
        if name_col:
            found = df[df[name_col].isin(expected_players)][name_col].tolist()
            log.info(f"    spot check found: {found}")


def check_transfermarkt() -> None:
    log.info("=" * 60)
    log.info("Transfermarkt checks")
    log.info("=" * 60)

    tm_dir = RAW / "transfermarkt"
    if not tm_dir.exists():
        log.error("  data/raw/transfermarkt/ does not exist. Run scrape_transfermarkt.py first.")
        return

    for code in ("GB1", "ES1", "L1", "IT1", "FR1"):
        for season in (2023, 2024):
            path = tm_dir / f"{code}_{season}.csv"
            if not path.exists():
                log.error(f"  MISSING: {path}")
                continue

            df = pd.read_csv(path)
            log.info(f"  {code} {season}: {len(df)} rows")

            if len(df) < 300:
                log.warning(f"    LOW row count ({len(df)}). Expected 500–700.")

            if "market_value_eur" in df.columns:
                nulls = df["market_value_eur"].isna().sum()
                log.info(f"    market_value_eur missing: {nulls}/{len(df)}")

            if "club" in df.columns:
                club_counts = df["club"].value_counts()
                log.info(f"    clubs: {len(club_counts)}, "
                         f"players per club range: "
                         f"{club_counts.min()}–{club_counts.max()}")


def check_statsbomb() -> None:
    log.info("=" * 60)
    log.info("StatsBomb checks")
    log.info("=" * 60)

    sb_dir = RAW / "statsbomb"
    if not sb_dir.exists():
        log.error("  data/raw/statsbomb/ does not exist. Run download_statsbomb.py first.")
        return

    comp_path = sb_dir / "competitions.csv"
    if not comp_path.exists():
        log.error(f"  MISSING: {comp_path}")
        return

    df = pd.read_csv(comp_path)
    log.info(f"  competitions.csv: {len(df)} competition-seasons available")


def main() -> None:
    check_fbref()
    check_transfermarkt()
    check_statsbomb()
    log.info("=" * 60)
    log.info("Sanity checks complete.")


if __name__ == "__main__":
    main()
