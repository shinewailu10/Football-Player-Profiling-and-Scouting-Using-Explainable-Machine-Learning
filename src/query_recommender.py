"""
query_recommender.py  
-----------------------------------------------------------
Usage examples:
  python3 src/query_recommender.py "Rodri" --team "Manchester City" --tm-position
  python3 src/query_recommender.py "Bukayo Saka" --tm-position --top 15
  python3 src/query_recommender.py "Erling Haaland" --position-family
  python3 src/query_recommender.py "Vinicius" --tm-position --exclude-same-league
"""

import argparse
import sys

from recommender import Recommender


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find similar players by tactical role."
    )
    parser.add_argument("target", help="Target player name (accents optional)")
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--season", type=int, default=None,
                        help="Disambiguate by season, e.g. 2023 or 2024")
    parser.add_argument("--team", type=str, default=None,
                        help="Disambiguate by team (partial match, e.g. 'Manchester City')")
    parser.add_argument("--any-cluster", action="store_true",
                        help="Search across all clusters, not just target's")
    parser.add_argument("--same-position", action="store_true",
                        help="Filter by FBref position family (DF/MF/FW-based, coarse)")
    parser.add_argument("--tm-position", action="store_true",
                        help="Filter by fine-grained Transfermarkt position "
                             "(e.g. 'Defensive Midfield' — recommended)")
    parser.add_argument("--position-family", action="store_true",
                        help="Filter by broader position family (7 groups)")
    parser.add_argument("--exclude-same-team", action="store_true")
    parser.add_argument("--exclude-same-league", action="store_true")
    parser.add_argument("--max-age", type=int, default=None)
    parser.add_argument("--max-value", type=float, default=None,
                        help="Max market value in EUR")
    args = parser.parse_args()

    print("Loading recommender...")
    r = Recommender.load()
    print(f"Loaded {len(r.players)} players.\n")

    try:
        print("TARGET:")
        print(f"  {r.describe_target(args.target, season=args.season, team=args.team)}")
        print()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    filters = []
    if args.any_cluster:
        filters.append("all clusters")
    else:
        filters.append("same cluster only")
    if args.tm_position:
        filters.append("same Transfermarkt position")
    if args.position_family:
        filters.append("same position family")
    if args.same_position:
        filters.append("same FBref position (coarse)")
    if args.exclude_same_team:
        filters.append("exclude same team")
    if args.exclude_same_league:
        filters.append("exclude same league")
    if args.max_age is not None:
        filters.append(f"age ≤ {args.max_age}")
    if args.max_value is not None:
        filters.append(f"value ≤ €{args.max_value:,.0f}")
    print(f"FILTERS: {', '.join(filters)}\n")

    try:
        results = r.find_similar(
            args.target,
            top_n=args.top,
            season=args.season,
            team=args.team,
            same_cluster_only=not args.any_cluster,
            same_position=args.same_position,
            same_tm_position=args.tm_position,
            same_position_family=args.position_family,
            exclude_same_team=args.exclude_same_team,
            exclude_same_league=args.exclude_same_league,
            max_age=args.max_age,
            max_market_value=args.max_value,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No matches under those filters.")
        return

    print(f"TOP {len(results)} SIMILAR PLAYERS:")
    for rec in results:
        print(f"  {rec}")


if __name__ == "__main__":
    main()
