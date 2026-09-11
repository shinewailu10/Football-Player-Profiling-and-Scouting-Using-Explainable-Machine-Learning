"""
scrape_transfermarkt.py
-----------------------
Scrapes player market values and demographics from Transfermarkt for the
top-5 European leagues across the 2023/24 and 2024/25 seasons.

Uses `requests` + `BeautifulSoup` with polite delays between requests.
The target pages are Transfermarkt's "detailed squad" listing, which shows
every player of every club in the league on a single set of paginated pages.

Output: 10 CSV files in data/raw/transfermarkt/
  GB1_2023.csv, GB1_2024.csv,
  ES1_2023.csv, ES1_2024.csv,
  L1_2023.csv,  L1_2024.csv,
  IT1_2023.csv, IT1_2024.csv,
  FR1_2023.csv, FR1_2024.csv

Usage:
    python src/scrape_transfermarkt.py
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

#Configuration

LEAGUES = {
    "GB1": "premier-league",
    "ES1": "laliga",
    "L1": "bundesliga",
    "IT1": "serie-a",
    "FR1": "ligue-1",
}
SEASONS = [2023, 2024]  #2023 = 2023/24 season on Transfermarkt

OUTPUT_DIR = Path("data/raw/transfermarkt")
SLEEP_BETWEEN_REQUESTS = 4  #seconds — Transfermarkt rate-limits scrapers

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_market_value(raw: str) -> Optional[float]:
    """Convert '€50.00m' or '€500Th.' or '-' to a number in euros.
    Returns None if the value is missing.
    """
    if not raw or raw.strip() in ("-", ""):
        return None

    raw = raw.replace("€", "").replace(",", ".").strip()

    #Multipliers
    if raw.lower().endswith("m"):
        return float(raw[:-1]) * 1_000_000
    if raw.lower().endswith("k") or raw.lower().endswith("th."):
        # "500Th." style
        num = re.sub(r"[^\d.]", "", raw)
        return float(num) * 1_000 if num else None

    #Fallback: try to parse whatever number is there
    num = re.sub(r"[^\d.]", "", raw)
    return float(num) if num else None


def fetch_page(url: str) -> BeautifulSoup:
    """GET a page and return a parsed BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def scrape_league_season(code: str, slug: str, season: int) -> pd.DataFrame:
    """Scrape one league-season from Transfermarkt's detailed squad page."""
    url = (
        f"https://www.transfermarkt.com/{slug}/startseite/wettbewerb/"
        f"{code}/plus/?saison_id={season}"
    )
    log.info(f"Fetching {url}")
    soup = fetch_page(url)

    #Every club in the league from the main table.
    club_links = []
    for a in soup.select("td.hauptlink.no-border-links a"):
        href = a.get("href", "")
        if "/startseite/verein/" in href and href not in club_links:
            club_links.append(href)

    log.info(f"  found {len(club_links)} clubs in {code} {season}/{season+1}")

    all_players = []

    for club_href in tqdm(club_links, desc=f"{code} {season}", unit="club"):
        #Rewrite the URL so it fetches the "detailed" squad list including market values.
        club_url = "https://www.transfermarkt.com" + club_href.replace(
            "/startseite/", "/kader/"
        ) + "/plus/1"
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        try:
            club_soup = fetch_page(club_url)
        except requests.HTTPError as e:
            log.warning(f"  skipped {club_url}: {e}")
            continue

        club_name_el = club_soup.select_one("h1.data-header__headline-wrapper")
        club_name = club_name_el.get_text(strip=True) if club_name_el else "Unknown"

        for row in club_soup.select("table.items > tbody > tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 5:
                continue

            #Player name
            name_el = row.select_one("td.hauptlink a")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)

            #Position 
            position_el = row.select_one("table.inline-table tr:nth-of-type(2) td")
            position = position_el.get_text(strip=True) if position_el else None

            #Age
            age = None
            for td in cells:
                m = re.search(r"\((\d{2})\)", td.get_text())
                if m:
                    age = int(m.group(1))
                    break

            #Market value
            mv_cell = row.select_one("td.rechts.hauptlink")
            market_value = parse_market_value(mv_cell.get_text(strip=True)) if mv_cell else None

            #Contract expiry 
            contract = None
            for td in cells:
                text = td.get_text(strip=True)
                if re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$", text) \
                        or re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", text):
                    contract = text

            all_players.append({
                "league": code,
                "season": season,
                "club": club_name,
                "name": name,
                "position": position,
                "age": age,
                "market_value_eur": market_value,
                "contract_expiry": contract,
            })

    df = pd.DataFrame(all_players)
    log.info(f"  scraped {len(df)} players from {code} {season}")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    for code, slug in LEAGUES.items():
        for season in SEASONS:
            output_path = OUTPUT_DIR / f"{code}_{season}.csv"
            if output_path.exists():
                log.info(f"Skipping {output_path} (already exists)")
                continue

            try:
                df = scrape_league_season(code, slug, season)
                df.to_csv(output_path, index=False)
                log.info(f"  saved to {output_path}")
            except Exception as e:
                log.error(f"  FAILED {code} {season}: {e}")

            time.sleep(SLEEP_BETWEEN_REQUESTS * 2)

    log.info("Done.")


if __name__ == "__main__":
    main()
