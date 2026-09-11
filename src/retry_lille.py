"""
retry_lille.py
--------------
Re-scrapes the LOSC Lille squad for the 2024/25 season, which failed with
a 503 error during the main Transfermarkt scrape. Merges the missing
players into the existing FR1_2024.csv.

Usage:
    python3 src/retry_lille.py
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scrape_transfermarkt import (
    HEADERS,
    fetch_page,
    parse_market_value,
)

LILLE_URL = "https://www.transfermarkt.com/losc-lille/kader/verein/1082/saison_id/2024/plus/1"
CSV_PATH = Path("data/raw/transfermarkt/FR1_2024.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def scrape_lille(max_attempts: int = 5) -> pd.DataFrame:
    """Fetch the Lille squad page with retries and exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        log.info(f"Attempt {attempt}/{max_attempts}: fetching Lille squad...")
        try:
            soup = fetch_page(LILLE_URL)
            break
        except requests.HTTPError as e:
            wait = 10 * attempt
            log.warning(f"  {e}. Waiting {wait}s before retry.")
            time.sleep(wait)
    else:
        raise RuntimeError("Lille squad page could not be fetched after retries.")

    club_name_el = soup.select_one("h1.data-header__headline-wrapper")
    club_name = club_name_el.get_text(strip=True) if club_name_el else "LOSC Lille"

    players = []
    for row in soup.select("table.items > tbody > tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 5:
            continue

        name_el = row.select_one("td.hauptlink a")
        if not name_el:
            continue
        name = name_el.get_text(strip=True)

        position_el = row.select_one("table.inline-table tr:nth-of-type(2) td")
        position = position_el.get_text(strip=True) if position_el else None

        age = None
        for td in cells:
            m = re.search(r"\((\d{2})\)", td.get_text())
            if m:
                age = int(m.group(1))
                break

        mv_cell = row.select_one("td.rechts.hauptlink")
        market_value = parse_market_value(mv_cell.get_text(strip=True)) if mv_cell else None

        contract = None
        for td in cells:
            text = td.get_text(strip=True)
            if re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$", text) \
                    or re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", text):
                contract = text

        players.append({
            "league": "FR1",
            "season": 2024,
            "club": club_name,
            "name": name,
            "position": position,
            "age": age,
            "market_value_eur": market_value,
            "contract_expiry": contract,
        })

    df = pd.DataFrame(players)
    log.info(f"  scraped {len(df)} Lille players")
    return df


def main() -> None:
    lille_df = scrape_lille()

    if lille_df.empty:
        log.error("No Lille players scraped. Aborting.")
        return

    existing = pd.read_csv(CSV_PATH)
    log.info(f"Existing FR1_2024.csv has {len(existing)} rows")

    #Remove any existing Lille rows (in case the failed attempt left partials)
    existing = existing[
        ~existing["club"].str.contains("Lille", case=False, na=False)
    ]

    merged = pd.concat([existing, lille_df], ignore_index=True)
    merged = merged.sort_values(["club", "name"]).reset_index(drop=True)

    merged.to_csv(CSV_PATH, index=False)
    log.info(f"Saved merged file: {len(merged)} total rows")


if __name__ == "__main__":
    main()
