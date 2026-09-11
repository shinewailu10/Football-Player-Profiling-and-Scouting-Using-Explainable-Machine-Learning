"""
evaluate_recommender.py
-----------------------
Evaluates the similarity recommender in two ways:

1. QUALITATIVE CASE STUDIES — for a handful of well-known players, print
   the top-10 similar players and let the reader judge footballing plausibility.

2. RETROSPECTIVE TRANSFER STUDY — for a small set of known 2024/25 transfers,
   query the recommender using the buying club's existing star at that position
   in 2023/24 data. Check whether the actual signing appears in the top-N
   similar players from cheaper leagues / lower-valuation candidates.

Output:
  data/processed/recommender_case_studies.md  — markdown report

Usage:
    python3 src/evaluate_recommender.py
"""

import logging
from pathlib import Path

from recommender import Recommender

OUTPUT_PATH = Path("data/processed/recommender_case_studies.md")

#Qualitative case studies — pick well-known players across roles.
QUALITATIVE_TARGETS = [
    ("Bukayo Saka", 2023, None),
    ("Erling Haaland", 2023, None),
    ("Vinicius Júnior", 2023, None),
    ("Rodri", 2023, "Manchester City"),
    ("Kevin De Bruyne", 2023, None),
    ("Virgil van Dijk", 2023, None),
]

#Retrospective transfer study — a handful of well-known 2024/25 signings.
#Format: (target_player, target_season, actual_signing, signing_prior_league)
RETROSPECTIVE_TRANSFERS = [
    ("Vinicius Júnior", 2023, "Kylian Mbappé", "FRA-Ligue 1"),
    ("Julián Álvarez", 2023, "Omar Marmoush", "GER-Bundesliga"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def format_recommendations(recs, indent: str = "  ") -> list[str]:
    return [f"{indent}{r}" for r in recs]


def qualitative_report(r: Recommender) -> list[str]:
    lines = ["## Qualitative case studies", ""]
    lines.append("For each target below, the top-10 most statistically similar players "
                 "within the same tactical cluster. Assess footballing plausibility.")
    lines.append("")

    for name, season, team in QUALITATIVE_TARGETS:
        lines.append(f"### {name} ({season}/{str(season+1)[-2:]})")
        lines.append("")
        try:
            lines.append(f"**Target:** {r.describe_target(name, season=season, team=team)}")
            lines.append("")
            recs = r.find_similar(name, top_n=10, season=season, team=team, same_tm_position=True)
            if not recs:
                lines.append("_No recommendations returned._")
            else:
                lines.append("**Top 10 similar (same cluster):**")
                lines.append("")
                lines.append("```")
                for rec in recs:
                    lines.append(str(rec))
                lines.append("```")
        except ValueError as e:
            lines.append(f"_Error: {e}_")
        lines.append("")
    return lines


def retrospective_report(r: Recommender) -> list[str]:
    lines = ["## Retrospective transfer study", ""]
    lines.append("Given the buying club's existing player at a position in 2023/24, "
                 "would the recommender have suggested the actual 2024/25 signing?")
    lines.append("")

    total_tested = 0
    total_hits = 0

    for target_name, target_season, actual_signing, prior_league in RETROSPECTIVE_TRANSFERS:
        lines.append(f"### Query: {target_name} ({target_season}/{str(target_season+1)[-2:]})  "
                     f"— did we recommend {actual_signing}?")
        lines.append("")
        try:
            lines.append(f"**Target:** {r.describe_target(target_name, season=target_season)}")
            lines.append(f"**Actual signing to look for:** {actual_signing} "
                         f"(then in {prior_league})")
            lines.append("")
            recs = r.find_similar(target_name, top_n=25, season=target_season, same_tm_position=True)
            total_tested += 1

            hit_rank = None
            for rec in recs:
                if actual_signing.lower() in rec.player.lower():
                    hit_rank = rec.rank
                    break

            if hit_rank is not None:
                total_hits += 1
                lines.append(f"**RESULT: HIT at rank #{hit_rank}**")
            else:
                lines.append(f"**RESULT: MISS** — {actual_signing} not in top 25.")
                lines.append("")
                lines.append("_Note: this doesn't necessarily mean the recommender is wrong. "
                             "Real transfers happen for many reasons beyond on-pitch statistical "
                             "fit (age, contract, wages, agent relationships, etc.). "
                             "A miss here is a discussion point, not a failure._")

            lines.append("")
            lines.append("Top 10 recommendations:")
            lines.append("")
            lines.append("```")
            for rec in recs[:10]:
                marker = "  ← MATCH" if rec.rank == hit_rank else ""
                lines.append(str(rec) + marker)
            lines.append("```")
        except ValueError as e:
            lines.append(f"_Error: {e}_")
        lines.append("")

    if total_tested > 0:
        lines.insert(2, f"**Overall hit rate: {total_hits}/{total_tested} at top-25.**")
        lines.insert(3, "")
    return lines


def main() -> None:
    log.info("Loading recommender...")
    r = Recommender.load()
    log.info(f"Loaded {len(r.players)} players.")

    lines = ["# Recommender Case Studies", ""]
    lines.append(f"Generated from Week 8 recommender using cluster column "
                 f"`{r.cluster_col}` on {len(r.players)} player-seasons.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend(qualitative_report(r))
    lines.append("---")
    lines.append("")
    lines.extend(retrospective_report(r))

    OUTPUT_PATH.write_text("\n".join(lines))
    log.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
