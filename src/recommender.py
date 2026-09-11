"""
recommender.py  
----------------------------------------------------------------
Role-based similarity recommender for football players.

- filter by tm_position (exact fine-grained Transfermarkt label, e.g. "Defensive Midfield")
- filter by position_family (broader 7-family grouping)
- both are more discriminating than the FBref-position filter
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")


@dataclass
class Recommendation:
    rank: int
    player: str
    team: str
    league: str
    season: int
    pos: str
    tm_position: str | None
    age: float
    market_value_eur: float | None
    cluster: int
    similarity: float

    def __str__(self) -> str:
        mv = f"€{self.market_value_eur:,.0f}" if self.market_value_eur else "—"
        tm = f" [{self.tm_position}]" if self.tm_position else ""
        return (
            f"{self.rank:>2}. {self.player:<25} "
            f"({self.team}, {self.league} {self.season}, "
            f"{self.pos}{tm}, age {self.age:.0f}, {mv})  "
            f"sim={self.similarity:.3f}"
        )


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalised = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in normalised if not unicodedata.combining(c))
    return stripped.lower().replace("'", "").replace("-", " ").strip()


class Recommender:
    """Load once, query many times."""

    def __init__(
        self,
        players: pd.DataFrame,
        features: np.ndarray,
        cluster_col: str = "kmeans_label",
    ):
        self.players = players.reset_index(drop=True).copy()
        self.features = features
        self.cluster_col = cluster_col

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.features_normalised = features / norms

        self.players["_name_norm"] = self.players["player"].apply(normalise_name)

    @classmethod
    def load(
        cls,
        data_dir: Path = DATA_DIR,
        cluster_col: str = "kmeans_label",
        feature_source: str = "raw",
    ) -> "Recommender":
        players_path = data_dir / "players_clustered.csv"
        features_path = data_dir / (
            "feature_matrix_raw.csv" if feature_source == "raw"
            else "feature_matrix_pca.csv"
        )

        players = pd.read_csv(players_path)
        features_df = pd.read_csv(features_path)

        #If tm_position and position_family exist in players_processed
        processed_path = data_dir / "players_processed.csv"
        if processed_path.exists():
            processed = pd.read_csv(processed_path)
            for col in ("tm_position", "position_family"):
                if col in processed.columns and col not in players.columns:
                    if len(processed) == len(players):
                        players[col] = processed[col].values
                    else:
                        # Try join by (season, player, born)
                        keys = ["season", "player", "born"]
                        players = players.merge(
                            processed[keys + [col]], on=keys, how="left"
                        )

        if len(players) != len(features_df):
            raise ValueError(
                f"Row-count mismatch: players ({len(players)}) vs "
                f"features ({len(features_df)}). Re-run Week 7 pipeline."
            )
        if cluster_col not in players.columns:
            raise ValueError(
                f"Cluster column '{cluster_col}' not found. Available: "
                f"{[c for c in players.columns if 'label' in c]}"
            )

        features = features_df.fillna(0).values
        return cls(players, features, cluster_col)

    def find_player(
        self,
        name: str,
        season: int | None = None,
        team: str | None = None,
    ) -> pd.Series:
        query = normalise_name(name)
        candidates = self.players[self.players["_name_norm"] == query]

        if season is not None:
            candidates = candidates[
                candidates["season"].astype(str).str.startswith(str(season)[-2:])
            ]
        if team is not None:
            team_q = team.lower()
            candidates = candidates[
                candidates["team"].str.lower().str.contains(team_q, na=False)
            ]

        if candidates.empty:
            partial = self.players[self.players["_name_norm"].str.contains(query, regex=False)]
            if partial.empty:
                raise ValueError(f"No player found matching '{name}'")
            elif len(partial) == 1:
                return partial.iloc[0]
            else:
                names = partial[["player", "team", "season"]].head(10).to_dict("records")
                raise ValueError(
                    f"No exact match for '{name}'. Did you mean one of: {names}?"
                )

        if len(candidates) > 1 and season is None:
            candidates = candidates.sort_values("season", ascending=False)
            return candidates.iloc[0]

        return candidates.iloc[0]

    @staticmethod
    def _same_position_family(target_pos: str, candidate_pos: str) -> bool:
        """FBref-position filter (v1 behaviour retained for backward compatibility).
        For finer-grained results use same_tm_position or same_position_family."""
        if not isinstance(target_pos, str) or not isinstance(candidate_pos, str):
            return False
        target_parts = [p.strip() for p in target_pos.split(",") if p.strip()]
        cand_parts = [p.strip() for p in candidate_pos.split(",") if p.strip()]
        if not target_parts or not cand_parts:
            return False
        target_primary = target_parts[0]
        target_set = set(target_parts)
        cand_set = set(cand_parts)
        return target_primary in cand_set and cand_set.issubset(target_set)

    def find_similar(
        self,
        target_name: str,
        top_n: int = 10,
        season: int | None = None,
        team: str | None = None,
        *,
        same_cluster_only: bool = True,
        same_position: bool = False,
        same_tm_position: bool = False,
        same_position_family: bool = False,
        exclude_same_team: bool = False,
        exclude_same_league: bool = False,
        max_age: int | None = None,
        max_market_value: float | None = None,
    ) -> list[Recommendation]:
        target = self.find_player(target_name, season=season, team=team)
        target_idx = target.name
        target_vec = self.features_normalised[target_idx]
        target_cluster = target[self.cluster_col]

        sims = self.features_normalised @ target_vec

        candidate_mask = np.ones(len(self.players), dtype=bool)
        candidate_mask[target_idx] = False

        if same_cluster_only:
            candidate_mask &= (self.players[self.cluster_col] == target_cluster).values

        if same_position:
            target_pos = str(target.get("pos", ""))
            pos_mask = self.players["pos"].apply(
                lambda p: self._same_position_family(target_pos, str(p))
            )
            candidate_mask &= pos_mask.values

        if same_tm_position:
            if "tm_position" not in self.players.columns:
                raise ValueError(
                    "tm_position column not available. Re-run build_features.py."
                )
            target_tm = target.get("tm_position")
            if pd.isna(target_tm):
                raise ValueError(f"Target '{target['player']}' has no tm_position; "
                                 "cannot filter by same_tm_position.")
            candidate_mask &= (self.players["tm_position"] == target_tm).values

        if same_position_family:
            if "position_family" not in self.players.columns:
                raise ValueError(
                    "position_family column not available. Re-run build_features.py."
                )
            target_fam = target.get("position_family")
            if pd.isna(target_fam):
                raise ValueError(f"Target '{target['player']}' has no position_family; "
                                 "cannot filter by same_position_family.")
            candidate_mask &= (self.players["position_family"] == target_fam).values

        if exclude_same_team:
            candidate_mask &= (self.players["team"] != target["team"]).values
        if exclude_same_league:
            candidate_mask &= (self.players["league"] != target["league"]).values
        if max_age is not None:
            candidate_mask &= (self.players["age"].fillna(999) <= max_age).values
        if max_market_value is not None:
            candidate_mask &= (
                self.players["market_value_eur"].fillna(np.inf) <= max_market_value
            ).values

        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            return []

        candidate_sims = sims[candidate_indices]
        top_positions = np.argsort(candidate_sims)[::-1][:top_n]
        top_indices = candidate_indices[top_positions]

        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.players.iloc[idx]
            recommendations.append(Recommendation(
                rank=rank,
                player=row["player"],
                team=row["team"],
                league=str(row["league"]),
                season=int(str(row["season"])[-4:]) if len(str(row["season"])) >= 4 else row["season"],
                pos=str(row.get("pos", "")),
                tm_position=(str(row["tm_position"])
                             if "tm_position" in row.index and pd.notna(row["tm_position"])
                             else None),
                age=float(row.get("age", 0)),
                market_value_eur=(float(row["market_value_eur"])
                                  if pd.notna(row.get("market_value_eur")) else None),
                cluster=int(row[self.cluster_col]),
                similarity=float(sims[idx]),
            ))
        return recommendations

    def describe_target(self, target_name: str, season: int | None = None,
                        team: str | None = None) -> str:
        t = self.find_player(target_name, season=season, team=team)
        mv = (f"€{t['market_value_eur']:,.0f}"
              if pd.notna(t.get("market_value_eur")) else "—")
        tm = (f" [{t['tm_position']}]"
              if "tm_position" in t.index and pd.notna(t.get("tm_position"))
              else "")
        return (f"{t['player']} — {t['team']} ({t['league']} {t['season']}), "
                f"{t['pos']}{tm}, age {t['age']:.0f}, {mv}, "
                f"cluster {int(t[self.cluster_col])}")
