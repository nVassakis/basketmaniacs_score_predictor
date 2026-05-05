import base64
import json
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3
import streamlit as st
import xgboost as xgb

DB_PATH = "data/basketball.db"
MODEL_PATH = "model/xgb_model.json"
MY_TEAM = "Unleash The Clowns"

FEATURES = [
    'PTS_last_3', 'EFF_last_3', 'USG_last_3',
    'PTS_season_avg', 'REB_season_avg', 'EFF_season_avg',
    'PTS_vs_OPP_hist', 'EFF_vs_OPP_hist',
    'OPP_PTS_ALLOWED_PER_PLAYER', 'OPP_REB_ALLOWED_PER_PLAYER',
    'SCORING_VACUUM', 'AST_last_3', 'BLK_last_3', 'STL_last_3', 'REB_last_3',
    'AST_season_avg', 'BLK_season_avg', 'STL_season_avg',
    'TS_PCT_last_3', 'TS_PCT_season_avg', 'MONTH', 'TEAM_USG_RANK'
]


def set_background(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = image_path.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:{mime};base64,{data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def get_team_players():
    conn = sqlite3.connect(DB_PATH)
    players = pd.read_sql(
        """
        SELECT PLAYER FROM ml_features
        WHERE TEAM = ? AND DATE >= '2026-01-01'
        GROUP BY PLAYER HAVING COUNT(*) >= 2
        ORDER BY PLAYER
        """,
        conn, params=(MY_TEAM,)
    )
    conn.close()
    return players['PLAYER'].tolist()


@st.cache_data
def get_league_opponents():
    with open("config/teams.txt", "r", encoding="utf-8") as f:
        slugs = [line.strip() for line in f if line.strip()]
    with open("config/team_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    teams = [mapping[slug] for slug in slugs if slug in mapping]
    return sorted([t for t in teams if t != MY_TEAM])


def predict(active_lineup, upcoming_opponent, month, model):
    conn = sqlite3.connect(DB_PATH)

    # Team historical average PPG
    team_avg_query = """
        SELECT AVG(team_pts) as team_avg
        FROM (SELECT SUM(PTS) as team_pts FROM ml_features WHERE TEAM = ? GROUP BY DATE)
    """
    team_avg = pd.read_sql(team_avg_query, conn, params=(MY_TEAM,)).iloc[0]['team_avg']

    # Lineup USG ranks
    placeholders = ",".join(["?"] * len(active_lineup))
    lineup_df = pd.read_sql(
        f"SELECT PLAYER, PTS_season_avg, USG_season_avg FROM ml_features WHERE PLAYER IN ({placeholders}) GROUP BY PLAYER HAVING MAX(DATE)",
        conn, params=active_lineup
    )
    lineup_power = lineup_df['PTS_season_avg'].sum()
    calculated_vacuum = team_avg - lineup_power
    lineup_df['CURRENT_USG_RANK'] = lineup_df['USG_season_avg'].rank(ascending=False, method='min')
    rank_dict = dict(zip(lineup_df['PLAYER'], lineup_df['CURRENT_USG_RANK']))

    # Opponent defense stats
    opp_df = pd.read_sql(
        "SELECT OPP_PTS_ALLOWED_PER_PLAYER, OPP_REB_ALLOWED_PER_PLAYER FROM ml_features WHERE OPPONENT = ? ORDER BY DATE DESC LIMIT 1",
        conn, params=(upcoming_opponent,)
    )
    if opp_df.empty:
        opp_pts_allowed, opp_reb_allowed = 10.0, 3.0
        opp_warning = f"No defensive data for '{upcoming_opponent}' — using league averages."
    else:
        opp_pts_allowed = opp_df['OPP_PTS_ALLOWED_PER_PLAYER'].iloc[0]
        opp_reb_allowed = opp_df['OPP_REB_ALLOWED_PER_PLAYER'].iloc[0]
        opp_warning = None

    # Predict per player
    predictions = []
    team_total = 0.0

    for player in active_lineup:
        p_df = pd.read_sql(
            "SELECT * FROM ml_features WHERE PLAYER = ? ORDER BY DATE DESC LIMIT 1",
            conn, params=(player,)
        )
        h_df = pd.read_sql(
            "SELECT PTS_vs_OPP_hist, EFF_vs_OPP_hist FROM ml_features WHERE PLAYER = ? AND OPPONENT = ? ORDER BY DATE DESC LIMIT 1",
            conn, params=(player, upcoming_opponent)
        )

        if p_df.empty:
            predictions.append({"Player": player, "Projected PTS": "No data"})
            continue

        input_data = p_df.iloc[0].to_dict()
        input_data['OPP_PTS_ALLOWED_PER_PLAYER'] = opp_pts_allowed
        input_data['OPP_REB_ALLOWED_PER_PLAYER'] = opp_reb_allowed
        input_data['SCORING_VACUUM'] = calculated_vacuum
        input_data['MONTH'] = month
        input_data['TEAM_USG_RANK'] = rank_dict.get(player, 5.0)

        if not h_df.empty and pd.notna(h_df['PTS_vs_OPP_hist'].iloc[0]):
            input_data['PTS_vs_OPP_hist'] = h_df['PTS_vs_OPP_hist'].iloc[0]
            input_data['EFF_vs_OPP_hist'] = h_df['EFF_vs_OPP_hist'].iloc[0]
        else:
            input_data['PTS_vs_OPP_hist'] = np.nan
            input_data['EFF_vs_OPP_hist'] = np.nan

        input_final = pd.DataFrame([input_data])[FEATURES].apply(pd.to_numeric, errors='coerce')
        pred = float(model.predict(input_final)[0])
        predictions.append({"Player": player, "Projected PTS": round(pred)})
        team_total += pred

    conn.close()
    return predictions, team_total, opp_warning, calculated_vacuum


# --- UI ---

st.set_page_config(page_title="Basketball Predictor", page_icon="🏀", layout="centered")

import os
if os.path.exists("assets/utc_logo.png"):
    set_background("assets/utc_logo.png")

st.markdown("""
    <style>
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; height: 0 !important; }
    header { display: none !important; height: 0 !important; }
    .stApp { padding-top: 0 !important; margin-top: 0 !important; }
    [data-testid="stAppViewContainer"] { padding-top: 0 !important; margin-top: 0 !important; }
    .block-container { padding-top: 2rem !important; }

    /* Dark overlay over the whole app */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.35);
        z-index: 0;
    }

    /* Keep content above the overlay */
    .stAppViewBlockContainer {
        position: relative;
        z-index: 1;
    }

    /* White text throughout */
    h1, h2, h3, p, label,
    .stSelectbox label,
    .stMultiselect label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏀 Game Day Predictor")

model = load_model()
all_players = get_team_players()
opponents = get_league_opponents()
current_month = datetime.now().month

opponent = st.selectbox("Opponent", opponents)
lineup = st.multiselect("Active lineup", all_players, placeholder="Search and select players...")

if st.button("Predict", type="primary", disabled=len(lineup) == 0):
    with st.spinner("Running predictions..."):
        predictions, team_total, opp_warning, _ = predict(lineup, opponent, current_month, model)

    if opp_warning:
        st.warning(opp_warning)

    st.subheader(f"{MY_TEAM} vs {opponent}")

    valid = [p for p in predictions if isinstance(p["Projected PTS"], (int, float))]
    missing = [p for p in predictions if not isinstance(p["Projected PTS"], (int, float))]

    df_results = pd.DataFrame(valid).sort_values("Projected PTS", ascending=False).reset_index(drop=True)
    df_results.index += 1
    st.dataframe(df_results, width='stretch')

    total = int(team_total)
    st.metric("Total Projected Score", f"{total - 1} – {total + 2} pts")

    if missing:
        st.error("No data for: " + ", ".join(p["Player"] for p in missing))
