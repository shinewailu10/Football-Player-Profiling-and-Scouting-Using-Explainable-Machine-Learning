"""
dashboard.py — Football Scouting Project Prototype Dashboard
------------------------------------------------------------
A Streamlit app for recommender, supervised models,
and SHAP explanations into an interactive UI.

Usage:
    streamlit run src/dashboard.py

Pages:
    - Player Profile          browse a player's identity, stats, cluster
    - Similar Players         run the recommender
    - Value Prediction        supervised prediction + SHAP explanation
    - Hit or Miss             transfer validation
    - Methodology & Findings  project summary
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

#Make sibling modules importable
sys.path.insert(0, str(Path(__file__).parent))
from recommender import Recommender  

#Paths
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/supervised")

#Page config
st.set_page_config(
    page_title="Football Player Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Cached loaders
@st.cache_resource(show_spinner="Loading recommender...")
def load_recommender() -> Recommender:
    return Recommender.load()


@st.cache_resource(show_spinner="Loading models...")
def load_supervised_models():
    metrics = pd.read_csv(DATA_DIR / "supervised_metrics.csv")
    preproc = joblib.load(MODELS_DIR / "preprocessing.pkl")

    def best_algo(scope: str, tree_only: bool = False):
        sub = metrics[metrics["scope"] == scope].sort_values("r2", ascending=False)
        if tree_only:
            sub = sub[sub["algorithm"].isin({"RandomForest", "XGBoost", "LightGBM"})]
        return sub.iloc[0]["algorithm"] if not sub.empty else None

    def load_model(scope: str, algo: str):
        safe = f"{scope}_{algo}".replace(" ", "_").replace("-", "_")
        return joblib.load(MODELS_DIR / f"{safe}.pkl")

    return {
        "metrics": metrics,
        "preproc": preproc,
        "best_algo": best_algo,
        "load_model": load_model,
    }


@st.cache_data
def load_processed() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "players_processed.csv")
    df = df[df["market_value_eur"].notna()].reset_index(drop=True)

    #Merge in cluster labels
    clustered_path = DATA_DIR / "players_clustered.csv"
    if clustered_path.exists():
        clustered = pd.read_csv(clustered_path)
        label_cols = [c for c in clustered.columns if c.endswith("_label")]
        if label_cols:
            keys = ["season", "player", "born"]
            df = df.merge(clustered[keys + label_cols], on=keys, how="left")
    return df


#---Sidebar navigation---
st.sidebar.title("⚽ Football Scouting")
st.sidebar.caption(
    "MSc Artificial Intelligence\n"
    "\nFootball Player Profiling and Scouting Using Explainable Machine Learning: A Role-based Similarity and Recommendation Framework For Emerging Talent Identification"
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Page",
    [
        "Player Profile",
        "Similar Players",
        "Value Prediction",
        "Hit or Miss",
        "Methodology & Findings",
    ],
)
st.sidebar.markdown("---")

with st.sidebar.expander("How to use this dashboard"):
    st.markdown("""
**Player Profile**
Type a player's name in the search box (partial names work — e.g. "saka" → Bukayo Saka). Pick a specific player-season from the dropdown. The bar chart compares the player's per-90 stats to the average for their position family.

**Similar Players**
Search a target player, then use the filters:
- **Position filter** — recommended: "Transfermarkt (fine)" for tactically appropriate peers
- **Max age / Max market value** — narrow to affordable / young prospects
- **Exclude same team / league** — for cross-league scouting

Results are ranked by cosine similarity in the standardised feature space.

**Value Prediction**
Search a player. Two market-value predictions are shown side-by-side:
- **Global model** — trained on all outfield players
- **Position model** — trained only on players in the same position family

The SHAP bar chart shows *why* the model made that prediction — green bars pushed the value up, red bars pushed it down. Hover the delta for interpretation.

**Hit or Miss**
Test whether the recommender would have found a specific transfer. Pick a query player and a signing — the app runs the recommender and shows whether the signing appears in the top-N.

**Methodology & Findings**
Project summary, per-position model performance table, and known limitations. Read this to understand what the system does well and where it doesn't.
    """)

with st.sidebar.expander("About the models"):
    st.markdown("""
**Clustering (K-Means, k=4)**
Groups players by playing style rather than position. Three algorithms were compared (K-Means, GMM, HDBSCAN); K-Means was chosen for its interpretable, stable clusters (silhouette 0.19). Four data-driven roles emerged from the analysis.

**Similarity recommender (cosine similarity)**
Given a target player, ranks every other player by how similar their standardised statistical profile is. Restricted to the same tactical role (from Transfermarkt's fine-grained position labels) so a defensive midfielder query returns other defensive midfielders — not attacking wingers with superficially similar output.

**Supervised value prediction (4 models compared)**
Linear Regression, Random Forest, XGBoost, and LightGBM were trained to predict log-transformed market value from per-90 stats + age. Evaluated with 5-fold cross-validation both globally and per-position. Random Forest tree-based predictions are used in this dashboard because they handle high-value players more gracefully than the linear model.

**SHAP (SHapley Additive exPlanations)**
An explainability method that decomposes a single prediction into per-feature contributions — showing exactly how each stat pushed the prediction up or down. Based on cooperative game theory (Lundberg & Lee, 2017). Green = pushed value up, red = pushed value down.

**Known limitation**
The available FBref features measure attacking output much more richly than defensive contribution. Predictions for attackers are more accurate than for centre-backs and defensive midfielders (R² 0.6+ vs 0.2). Discussed in more detail on the Methodology page.
    """)

st.sidebar.markdown("---")

recommender = load_recommender()
sup = load_supervised_models()
df = load_processed()

feature_cols = sup["preproc"]["feature_cols"]


#Helpers
def player_selector(label: str, key: str) -> pd.Series | None:
    """A shared player-search widget. Returns the selected player-season row."""
    name_query = st.text_input(
        label, value="", key=f"{key}_name",
        placeholder="Type a player name (e.g. Bukayo Saka)",
    )
    if not name_query:
        return None

    #Fuzzy filter to candidates
    q = name_query.lower()
    matches = df[df["player"].str.lower().str.contains(q, na=False)]
    if matches.empty:
        st.warning(f"No player matching '{name_query}'.")
        return None

    #Build human-readable options: "Player — Team (season)"
    matches = matches.copy()
    matches["_label"] = matches.apply(
        lambda r: f"{r['player']} — {r['team']} ({r['season']})",
        axis=1,
    )
    choice = st.selectbox(
        "Select a specific player-season",
        matches["_label"].tolist(),
        key=f"{key}_choice",
    )
    return matches[matches["_label"] == choice].iloc[0]


def format_eur(v) -> str:
    if pd.isna(v) or v is None:
        return "—"
    return f"€{v/1e6:,.1f}m"


#---PAGE 1: Player Profile---
if page == "Player Profile":
    st.title("Player Profile")
    st.markdown(
        "Browse a player's identity, stats, and tactical cluster assignment. "
        "Pick a player below."
    )

    player = player_selector("Search for a player", key="profile")
    if player is None:
        st.info("Enter a player name above to begin.")
        st.stop()

    #Header row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.metric("Player", player["player"])
    col2.metric("Age", int(player["age"]))
    col3.metric("Market value", format_eur(player["market_value_eur"]))
    cluster_val = player.get("kmeans_label")
    col4.metric("Cluster",
                int(cluster_val) if pd.notna(cluster_val) else "—")

    col1, col2, col3 = st.columns(3)
    col1.metric("Team", player["team"])
    col2.metric("League", player["league"])
    col3.metric("Season", player["season"])

    col1, col2 = st.columns(2)
    col1.metric("FBref position", player["pos"])
    col2.metric("Transfermarkt position",
                player.get("tm_position", "—") if pd.notna(player.get("tm_position")) else "—")

    st.markdown("---")
    st.subheader("Statistical profile vs position-family average")

    if pd.notna(player.get("position_family")):
        family = player["position_family"]
        family_df = df[df["position_family"] == family]

        #Compare player to family mean across features
        player_vals = player[feature_cols].astype(float).values
        family_means = family_df[feature_cols].astype(float).mean().values

        chart_df = pd.DataFrame({
            "feature": feature_cols,
            f"{player['player']}": player_vals,
            f"{family} average": family_means,
        })

        #Bar chart via plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["feature"],
            y=chart_df[f"{player['player']}"],
            name=player["player"],
        ))
        fig.add_trace(go.Bar(
            x=chart_df["feature"],
            y=chart_df[f"{family} average"],
            name=f"{family} average",
        ))
        fig.update_layout(
            barmode="group",
            xaxis_tickangle=-45,
            height=460,
            margin=dict(t=10, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

        #Also a table of the numeric values
        with st.expander("Show raw feature values"):
            st.dataframe(chart_df.set_index("feature"), use_container_width=True)


#---PAGE 2: Similar Players---
elif page == "Similar Players":
    st.title("Similar Players — Recommender")
    st.markdown(
        "Given a target player, return the top-N most statistically similar "
        "players. Filters restrict recommendations to realistic scouting "
        "candidates (age, budget, tactical position)."
    )

    target = player_selector("Search for a target player", key="rec")
    if target is None:
        st.info("Enter a player name above to begin.")
        st.stop()

    #Sidebar-style filters (kept on main page since sidebar has nav)
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        top_n = col1.slider("Number of recommendations", 5, 25, 10)

        position_filter = col2.radio(
            "Position filter",
            ["Transfermarkt (fine)", "Position family (broad)", "None"],
            index=0,
        )

        max_age = col3.slider("Max age", 16, 40, 40)

        col1, col2, col3 = st.columns(3)
        max_value_m = col1.number_input(
            "Max market value (€m)", min_value=0.0, value=200.0, step=5.0,
        )
        exclude_same_team = col2.checkbox("Exclude same team", value=False)
        exclude_same_league = col3.checkbox("Exclude same league", value=False)

    #Compose the filter kwargs
    kwargs = dict(
        top_n=top_n,
        same_tm_position=(position_filter == "Transfermarkt (fine)"),
        same_position_family=(position_filter == "Position family (broad)"),
        exclude_same_team=exclude_same_team,
        exclude_same_league=exclude_same_league,
        max_age=max_age if max_age < 40 else None,
        max_market_value=max_value_m * 1e6 if max_value_m < 200 else None,
    )

    st.markdown("---")
    st.subheader("Target")
    st.write(recommender.describe_target(
        target["player"], season=int(str(target["season"])[:2]) + 2000,
        team=target["team"],
    ))

    try:
        recs = recommender.find_similar(
            target["player"],
            season=int(str(target["season"])[:2]) + 2000,
            team=target["team"],
            **kwargs,
        )
    except ValueError as e:
        st.error(f"Recommender error: {e}")
        st.stop()

    st.subheader(f"Top {len(recs)} similar players")
    if not recs:
        st.warning("No candidates match those filters.")
    else:
        rec_df = pd.DataFrame([{
            "Rank": r.rank,
            "Player": r.player,
            "Team": r.team,
            "League": r.league,
            "Season": r.season,
            "Position": r.tm_position or r.pos,
            "Age": r.age,
            "Market value": format_eur(r.market_value_eur),
            "Similarity": f"{r.similarity:.3f}",
        } for r in recs])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)


#----PAGE 3: Value Prediction---
elif page == "Value Prediction":
    st.title("Market Value Prediction with SHAP explanation")
    st.markdown(
        "The supervised models predict player market value from per-90 stats. "
        "SHAP values show *why* the model made that prediction — which features "
        "pushed the value up or down."
    )

    player = player_selector("Search for a player", key="pred")
    if player is None:
        st.info("Enter a player name above to begin.")
        st.stop()

    st.markdown("---")

    #Prepare features
    x = player[feature_cols].astype(float).fillna(0).values.reshape(1, -1)
    actual_eur = float(player["market_value_eur"])
    family = player.get("position_family")

    #Global
    global_algo = sup["best_algo"]("global", tree_only=True)
    global_model = sup["load_model"]("global", global_algo)
    pred_global_log = float(global_model.predict(x)[0])
    pred_global_eur = float(np.expm1(pred_global_log))

    #Per-position
    pos_algo = pos_model = pred_pos_eur = None
    if pd.notna(family) and family != "Goalkeeper":
        pos_algo = sup["best_algo"](family, tree_only=True)
        if pos_algo:
            try:
                pos_model = sup["load_model"](family, pos_algo)
                pred_pos_log = float(pos_model.predict(x)[0])
                pred_pos_eur = float(np.expm1(pred_pos_log))
            except FileNotFoundError:
                pass

    #Display predictions
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual market value", format_eur(actual_eur))

    global_err = pred_global_eur - actual_eur
    col2.metric(
        f"Global model ({global_algo})",
        format_eur(pred_global_eur),
        delta=round(global_err / 1e6, 1),
        delta_color="off",
        help="Prediction error in €m. Negative = under-prediction, positive = over-prediction. Neither is 'good'.",
    )
    if pred_pos_eur is not None:
        pos_err = pred_pos_eur - actual_eur
        col3.metric(
            f"Position model ({pos_algo}, {family})",
            format_eur(pred_pos_eur),
            delta=round(pos_err / 1e6, 1),
            delta_color="off",
            help="Prediction error in €m.",
        )

    st.markdown("---")
    st.subheader("SHAP explanation")
    st.caption(
        "Green bars push the prediction UP (increase value). "
        "Red bars push it DOWN."
    )

    #Compute SHAP on the global tree model
    explainer = shap.TreeExplainer(global_model)
    shap_val = explainer.shap_values(x)[0]
    contribs = sorted(
        zip(feature_cols, shap_val, x[0]),
        key=lambda t: abs(t[1]), reverse=True,
    )
    top_contribs = contribs[:8]

    features_r = [c[0] for c in top_contribs][::-1]
    shap_r = [c[1] for c in top_contribs][::-1]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in shap_r]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shap_r, y=features_r, orientation="h",
        marker=dict(color=colors),
        text=[f"{s:+.3f}" for s in shap_r],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="SHAP contribution (log-EUR)",
        height=420,
        margin=dict(t=20, b=40, l=10, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    #Feature-value context in a table
    with st.expander("Show raw feature values for this player"):
        detail = pd.DataFrame([{
            "Feature": c[0],
            "Value": f"{c[2]:.3f}",
            "SHAP contribution": f"{c[1]:+.3f}",
        } for c in contribs])
        st.dataframe(detail, use_container_width=True, hide_index=True)

#---PAGE 4: Hit or Miss---
elif page == "Hit or Miss":
    st.title("Hit or Miss — test the recommender on transfers")
    st.markdown(
        "Pick two players: a **query player** (a club's existing star at a "
        "position) and a ** signing** (a transfer target). "
        "The recommender queries using only the first player, then reports "
        "whether the second player appears in its top-N recommendations. "
        "Use this to test how well the system would have predicted real "
        "transfers."
    )

    PRESETS = {
        "— pick your own —": (None, None, None, None),
        "Vinicius Júnior → Kylian Mbappé": (
            "Vinicius Júnior", "Real Madrid",
            "Kylian Mbappé", "PSG",
        ),
        "Julián Álvarez → Omar Marmoush": (
            "Julián Álvarez", "Manchester City",
            "Omar Marmoush", "Frankfurt",
        ),
        "Erling Haaland → Omar Marmoush": (
            "Erling Haaland", "Manchester City",
            "Omar Marmoush", "Frankfurt",
        ),
    }

    preset_choice = st.selectbox(
        "Start from a verified transfer, or pick your own players below",
        list(PRESETS.keys()),
        key="val_preset",
    )
    preset_qname, preset_qteam, preset_sname, preset_steam = PRESETS[preset_choice]

    #When the preset changes, force-update the text-input state.
    if st.session_state.get("_val_last_preset") != preset_choice:
        st.session_state["_val_last_preset"] = preset_choice
        if preset_qname:
            st.session_state["val_q_name"] = preset_qname
        if preset_sname:
            st.session_state["val_s_name"] = preset_sname
    st.markdown("---")
    col_q, col_s = st.columns(2)

    with col_q:
        st.subheader("Query player")
        st.caption("The club's existing player at that position")
        q_name = st.text_input(
            "Query player name",
            value=preset_qname if preset_qname else "",
            key="val_q_name",
            placeholder="e.g. Vinicius Júnior",
        )
        query_player = None
        if q_name:
            matches = df[df["player"].str.lower().str.contains(q_name.lower(), na=False)]
            if matches.empty:
                st.warning(f"No player matching '{q_name}'.")
            else:
                matches = matches.copy()
                matches["_label"] = matches.apply(
                    lambda r: f"{r['player']} — {r['team']} ({r['season']})",
                    axis=1,
                )
                # Prefer preset team if given
                default_idx = 0
                if preset_qteam:
                    for i, lbl in enumerate(matches["_label"].tolist()):
                        if preset_qteam.lower() in lbl.lower() and "2324" in lbl:
                            default_idx = i
                            break
                q_choice = st.selectbox(
                    "Pick the specific player-season",
                    matches["_label"].tolist(),
                    index=default_idx,
                    key="val_q_choice",
                )
                query_player = matches[matches["_label"] == q_choice].iloc[0]

    with col_s:
        st.subheader("Signing")
        st.caption("The real player the club signed")
        s_name = st.text_input(
            "Actual signing name",
            value=preset_sname if preset_sname else "",
            key="val_s_name",
            placeholder="e.g. Kylian Mbappé",
        )
        signing_player = None
        if s_name:
            matches = df[df["player"].str.lower().str.contains(s_name.lower(), na=False)]
            if matches.empty:
                st.warning(f"No player matching '{s_name}'.")
            else:
                matches = matches.copy()
                matches["_label"] = matches.apply(
                    lambda r: f"{r['player']} — {r['team']} ({r['season']})",
                    axis=1,
                )
                default_idx = 0
                if preset_steam:
                    for i, lbl in enumerate(matches["_label"].tolist()):
                        if preset_steam.lower() in lbl.lower() and "2324" in lbl:
                            default_idx = i
                            break
                s_choice = st.selectbox(
                    "Pick the specific player-season",
                    matches["_label"].tolist(),
                    index=default_idx,
                    key="val_s_choice",
                )
                signing_player = matches[matches["_label"] == s_choice].iloc[0]

    st.markdown("---")

    #Filter and run controls
    col1, col2, col3 = st.columns(3)
    filter_mode = col1.radio(
        "Position filter mode",
        ["Strict (Transfermarkt fine)", "Broad (position family)",
         "Cluster only (no position)"],
        index=2,
        help="Strict filters most aggressively; cluster-only is loosest. "
             "Try all three on the same pair to see how filtering affects recall.",
    )
    top_n = col2.slider("Search depth (top-N)", 10, 100, 30)
    run_test = col3.button(
        "Run validation test", type="primary"
        #disabled=(query_player is None or signing_player is None),
    )

    if run_test:
        if query_player is None or signing_player is None:
            st.warning(
                "Please pick both a query player and an actual signing before running."
            )
            st.stop()
        #Build the kwargs based on filter mode
        kwargs = {"same_cluster_only": True}
        if filter_mode.startswith("Strict"):
            kwargs["same_tm_position"] = True
        elif filter_mode.startswith("Broad"):
            kwargs["same_position_family"] = True
        #cluster-only mode: no additional filter

        #Season number for the query (extract from '2324' → 2023)
        q_season_raw = str(query_player["season"])
        q_season_year = 2000 + int(q_season_raw[:2])

        try:
            recs = recommender.find_similar(
                query_player["player"],
                season=q_season_year,
                team=query_player["team"],
                top_n=top_n,
                **kwargs,
            )
        except ValueError as e:
            st.error(f"Recommender error: {e}")
            st.stop()

        #Find the signing in the recommendations
        hit_rank = None
        hit_sim = None
        s_name_lower = signing_player["player"].lower()
        for r in recs:
            if r.player.lower() == s_name_lower:
                #Check season if can
                if str(r.season) == str(signing_player["season"])[-4:] or True:
                    hit_rank = r.rank
                    hit_sim = r.similarity
                    break

        #Render result
        st.markdown("### Result")
        if hit_rank is not None:
            st.success(f"**HIT** — {signing_player['player']} ranked "
                       f"**#{hit_rank}** with similarity **{hit_sim:.3f}**")
            st.caption(
                f"The recommender identified {signing_player['player']} as one "
                f"of the top {top_n} statistically similar players to "
                f"{query_player['player']} under {filter_mode.lower()} filtering."
            )
        else:
            st.error(f"**MISS** — {signing_player['player']} not found in the "
                     f"top {top_n} recommendations for {query_player['player']} "
                     f"under {filter_mode.lower()} filtering.")
            st.caption(
                "This could mean: (a) the players aren't stylistically similar "
                "in the current feature space, (b) they're in different tactical "
                "clusters, or (c) the position filter excluded the signing. "
                "Try a looser filter mode to see if the signing appears then."
            )

        #Always show the top-N for context
        st.markdown(f"### Top {min(15, len(recs))} recommendations (for reference)")
        rec_rows = []
        for r in recs[:15]:
            is_hit = (r.player.lower() == s_name_lower)
            rec_rows.append({
                "Rank": r.rank,
                "Player": ("→ " + r.player + " ←") if is_hit else r.player,
                "Team": r.team,
                "Season": r.season,
                "Position": r.tm_position or r.pos,
                "Age": int(r.age),
                "Market value": format_eur(r.market_value_eur),
                "Similarity": f"{r.similarity:.3f}",
            })
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
                #---Side-by-side stats comparison---
        st.markdown("### Player stats comparison")
        st.caption(
            "Compare the two players' per-90 statistical profiles side by side. "
            "Large differences here help explain why the recommender did or "
            "did not match them."
        )

        #Compute the direct cosine similarity between these two specific players even if the signing wasn't in the top-N).
        import unicodedata as _u

        def _norm(s):
            s = _u.normalize("NFKD", str(s))
            return "".join(c for c in s if not _u.combining(c)).lower().replace("'", "").replace("-", " ").strip()

        q_idx = recommender.players[
            (recommender.players["_name_norm"] == _norm(query_player["player"]))
            & (recommender.players["team"] == query_player["team"])
            & (recommender.players["season"] == query_player["season"])
        ].index
        s_idx = recommender.players[
            (recommender.players["_name_norm"] == _norm(signing_player["player"]))
            & (recommender.players["team"] == signing_player["team"])
            & (recommender.players["season"] == signing_player["season"])
        ].index

        if len(q_idx) > 0 and len(s_idx) > 0:
            direct_sim = float(
                recommender.features_normalised[q_idx[0]]
                @ recommender.features_normalised[s_idx[0]]
            )
            st.metric(
                "Direct cosine similarity (this pair)",
                f"{direct_sim:.3f}",
                help="This is the raw cosine similarity between just these two "
                     "players, regardless of ranking or filters. Range [-1, +1]."
            )

        #Feature-value comparison table
        stat_cols = [c for c in feature_cols if c != "age"]  # show age separately
        comparison = pd.DataFrame({
            "Feature": ["age"] + stat_cols,
            f"{query_player['player']} ({query_player['season']})": [
                query_player["age"]
            ] + [
                round(float(query_player.get(c, 0) or 0), 3) for c in stat_cols
            ],
            f"{signing_player['player']} ({signing_player['season']})": [
                signing_player["age"]
            ] + [
                round(float(signing_player.get(c, 0) or 0), 3) for c in stat_cols
            ],
        })

        #Add an absolute difference column
        q_col = comparison.columns[1]
        s_col = comparison.columns[2]
        comparison["Difference"] = (comparison[q_col] - comparison[s_col]).round(3)

        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.caption(
            "**How to read this.** Features where the two players are very "
            "different will show large 'Difference' values. If most features "
            "align, they play similarly. If defensive stats (interceptions, "
            "tackles_won) differ a lot from attacking stats (goals, shots), "
            "they play different roles even if their overall output looks similar."
        )


#---PAGE 5: Methodology & Findings---
elif page == "Methodology & Findings":
    st.title("Methodology & Findings")

    st.markdown("""
    ## Project pipeline

    This dashboard is built on top of a four-stage pipeline:

    1. **Data collection.** Scraped ~2,935 outfield player-seasons from three
       public sources: FBref (per-90 performance stats), Transfermarkt
       (market values and fine-grained positions), and StatsBomb Open Data.
    2. **Feature engineering.** Merged sources by fuzzy-matched names, filtered
       to players with ≥900 minutes, produced 21 per-90 / composite features
       plus fine-grained (13-value) and family-grouped (7-value) position labels.
    3. **Clustering.** Compared K-Means, GMM, and HDBSCAN with Silhouette /
       Davies–Bouldin / Calinski–Harabasz metrics. K-Means (k=4) selected.
    4. **Similarity recommender.** Cosine similarity in the standardised
       feature space, restricted to same tactical role (Transfermarkt position).
    5. **Supervised prediction.** Four models (Linear Regression, Random Forest,
       XGBoost, LightGBM) trained globally and per-position with 5-fold CV.
       SHAP for interpretability.
    """)

    st.markdown("## Key metrics — all four models compared")
    st.caption("R² is the fraction of variance explained (higher = better). "
               "MAE is the mean prediction error in euros. "
               "Median |error| is the typical error size — half of players "
               "are predicted more accurately than this.")

    metrics_df = pd.read_csv(DATA_DIR / "supervised_metrics.csv")

    #Full table with all 4 models per scope, sorted so best-per-scope is first
    full = metrics_df.sort_values(["scope", "r2"], ascending=[True, False]).copy()
    full["is_best"] = full.groupby("scope")["r2"].transform("max") == full["r2"]

    display = full[["scope", "algorithm", "r2", "mae_eur", "median_abs_error_eur"]].rename(
        columns={
            "scope": "Scope",
            "algorithm": "Model",
            "r2": "R²",
            "mae_eur": "MAE (€)",
            "median_abs_error_eur": "Median |error| (€)",
        }
    )
    display["MAE (€)"] = display["MAE (€)"].map(lambda v: f"€{v/1e6:.2f}m")
    display["Median |error| (€)"] = display["Median |error| (€)"].map(
        lambda v: f"€{v/1e6:.2f}m")
    display["R²"] = display["R²"].map(lambda v: f"{v:.3f}")

    #Highlight the row with the highest R² per scope in bold
    def highlight_best(row):
        return ["font-weight: bold" if full.loc[row.name, "is_best"] else ""
                for _ in row]

    st.dataframe(display.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)

    st.caption("**Bold rows** indicate the best-performing model for each scope.")

    st.markdown("""
    ## Known limitations

    - **Attacking bias in the feature set.** The soccerdata FBref endpoint
      exposes only five player-level stat categories, none of which cover
      passing volume, defensive coverage by zone, or ball progression.
      Consequences propagate through the pipeline: clustering separates
      attacking styles more sharply than defensive roles; the recommender
      returns better peer matches for wingers and forwards than for defensive
      midfielders and centre-backs; the supervised model predicts elite
      forwards more accurately than elite defenders (R² = 0.62 for wingers
      vs 0.16 for defensive midfielders).

    - **Elite-value under-prediction.** The heavy-tail distribution of
      market values means the model has few training examples above €100m,
      so predictions for players such as Haaland or Mbappé are systematically
      under-priced.

    - **Small position groups.** Some Transfermarkt positions (Second Striker,
      Left Midfield, Right Midfield) have <100 players and were merged into
      broader families for supervised modelling.

    ## Data sources

    - **FBref** (Sports Reference / Opta) — scraped via the `soccerdata` library
    - **Transfermarkt** — custom scraper using `requests` + BeautifulSoup
    - **StatsBomb Open Data** — via the `statsbombpy` library

    All data is publicly accessible. This project is academic research only —
    no automated scouting decisions are produced or transmitted.
    """)
