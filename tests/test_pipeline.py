"""
tests/test_pipeline.py
----------------------
Unit tests for critical pipeline functions.

Run with:
    pip install pytest
    pytest tests/ -v

These tests verify that the small, self-contained utility functions
behave correctly. They form part of the "software engineering good
practice" requirement from the MSc Course Learning Outcomes.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommender import Recommender, normalise_name


# ============================================================
# Tests for name normalisation (used in fuzzy matching + lookup)
# ============================================================
class TestNormaliseName:
    """The normalise_name function strips accents, lowercases, and
    standardises punctuation. It's used both for matching FBref names
    to Transfermarkt names, and for the CLI/dashboard player lookup."""

    def test_lowercase(self):
        assert normalise_name("BUKAYO SAKA") == "bukayo saka"

    def test_strips_accents(self):
        # Common accented characters in football player names
        assert normalise_name("Vinícius Júnior") == "vinicius junior"
        assert normalise_name("Kylian Mbappé") == "kylian mbappe"
        assert normalise_name("Rúben Dias") == "ruben dias"
        assert normalise_name("Çalhanoğlu") == "calhanoglu"

    def test_removes_hyphens(self):
        # De-Bruyne vs De Bruyne should match
        assert normalise_name("Kevin De-Bruyne") == "kevin de bruyne"
        assert normalise_name("Kevin De Bruyne") == "kevin de bruyne"

    def test_removes_apostrophes(self):
        assert normalise_name("N'Golo Kanté") == "ngolo kante"

    def test_handles_empty_string(self):
        assert normalise_name("") == ""

    def test_handles_none(self):
        assert normalise_name(None) == ""

    def test_handles_non_string(self):
        # nan or numeric inputs from pandas shouldn't crash
        assert normalise_name(float("nan")) == ""
        assert normalise_name(42) == ""

    def test_idempotent(self):
        # Normalising twice gives the same result
        name = "José Mourinho"
        assert normalise_name(normalise_name(name)) == normalise_name(name)


# ============================================================
# Tests for the position-family filter (Week 8 methodology fix)
# ============================================================
class TestPositionFamilyFilter:
    """The _same_position_family helper decides whether a candidate
    player is a valid tactical match for a target. This was the
    Week 8 fix that rescued the recommender for defensive midfielders."""

    def test_exact_match(self):
        # Rodri (MF,DF) matches another MF,DF player
        assert Recommender._same_position_family("MF,DF", "MF,DF") is True

    def test_primary_only_matches(self):
        # Rodri (MF,DF) matches pure MF (Rodri's primary is MF, MF is subset)
        assert Recommender._same_position_family("MF,DF", "MF") is True

    def test_component_order_matters_less(self):
        # DF,MF and MF,DF both contain {MF, DF}
        assert Recommender._same_position_family("MF,DF", "DF,MF") is True

    def test_excludes_extra_component(self):
        # FW,MF has FW which target doesn't → exclude
        assert Recommender._same_position_family("MF,DF", "FW,MF") is False
        assert Recommender._same_position_family("MF,DF", "MF,FW") is False

    def test_excludes_missing_primary(self):
        # DF alone doesn't include target's primary (MF) → exclude
        assert Recommender._same_position_family("MF,DF", "DF") is False

    def test_pure_position_matches_itself(self):
        assert Recommender._same_position_family("MF", "MF") is True
        assert Recommender._same_position_family("DF", "DF") is True
        assert Recommender._same_position_family("FW", "FW") is True

    def test_handles_bad_input(self):
        # NaN or non-string inputs must not crash
        assert Recommender._same_position_family(None, "MF") is False
        assert Recommender._same_position_family("MF", None) is False
        assert Recommender._same_position_family("", "") is False


# ============================================================
# Tests for log/exp roundtrip (used in supervised model)
# ============================================================
class TestLogExpRoundtrip:
    """Market values are log-transformed for modelling. The predictions
    must be exp-transformed back to euros. Verify the roundtrip is stable
    across the range of realistic market values."""

    @pytest.mark.parametrize("value_eur", [
        50_000,           # very low
        1_000_000,        # €1m
        10_000_000,       # €10m
        100_000_000,      # €100m
        200_000_000,      # €200m (Yamal)
    ])
    def test_roundtrip(self, value_eur):
        log_val = np.log1p(value_eur)
        recovered = np.expm1(log_val)
        # Should recover the original value to floating-point precision
        assert abs(recovered - value_eur) < 1e-6

    def test_zero_value(self):
        # log1p(0) = 0, expm1(0) = 0 — must round-trip cleanly
        assert np.expm1(np.log1p(0)) == 0

    def test_log_transformation_reduces_skew(self):
        # Log should reduce the right-skew of market values
        values = np.array([50_000, 5_000_000, 200_000_000])
        log_values = np.log1p(values)
        # Log-space spread should be far smaller than raw-space spread
        raw_ratio = values.max() / values.min()
        log_ratio = log_values.max() / log_values.min()
        assert log_ratio < raw_ratio / 100


# ============================================================
# Integration test — recommender loads and finds real player
# ============================================================
class TestRecommenderIntegration:
    """Integration tests that exercise the full recommender.
    These require the processed data to exist — they are skipped
    if the pipeline has not been run yet."""

    @pytest.fixture(scope="class")
    def recommender(self):
        data_dir = Path("data/processed")
        if not (data_dir / "players_clustered.csv").exists():
            pytest.skip("Requires data/processed/players_clustered.csv — "
                        "run the pipeline first.")
        return Recommender.load()

    def test_loads_correct_row_count(self, recommender):
        # We expect ~2,935 outfield players
        assert 2000 < len(recommender.players) < 4000

    def test_finds_saka(self, recommender):
        row = recommender.find_player("Bukayo Saka")
        assert row["player"] == "Bukayo Saka"
        assert row["team"] == "Arsenal"

    def test_finds_haaland(self, recommender):
        row = recommender.find_player("Erling Haaland")
        assert row["player"] == "Erling Haaland"

    def test_disambiguates_by_team(self, recommender):
        # Two Rodris in the data — team should disambiguate
        rodri_city = recommender.find_player("Rodri", team="Manchester City")
        assert rodri_city["team"] == "Manchester City"
        assert rodri_city["age"] >= 26  # City's Rodri is older

    def test_similar_returns_valid_count(self, recommender):
        recs = recommender.find_similar("Bukayo Saka", top_n=10)
        assert len(recs) == 10
        # Similarities should be in descending order
        sims = [r.similarity for r in recs]
        assert sims == sorted(sims, reverse=True)
        # And should be reasonable (0 to 1 range for cosine of standardised features)
        assert all(-1.5 <= s <= 1.5 for s in sims)

    def test_same_tm_position_filter_works(self, recommender):
        # Rodri filtered by Transfermarkt position → all results are DMs
        recs = recommender.find_similar(
            "Rodri", team="Manchester City",
            top_n=10, same_tm_position=True,
        )
        assert len(recs) > 0
        for r in recs:
            assert r.tm_position == "Defensive Midfield"

    def test_max_age_filter_works(self, recommender):
        recs = recommender.find_similar(
            "Bukayo Saka", top_n=10, max_age=22,
        )
        for r in recs:
            assert r.age <= 22

    def test_unknown_player_raises(self, recommender):
        with pytest.raises(ValueError, match="No player found"):
            recommender.find_player("Player Who Does Not Exist")
