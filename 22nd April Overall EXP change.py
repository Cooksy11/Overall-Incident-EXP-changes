import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO, StringIO
from matplotlib.backends.backend_pdf import PdfPages
import requests

# --- Load Data ---
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTFcAq9JNJsfPhiPe3sdaEipD1jofgrCNozbcld55JzDguXCxVpFbM0KwKd5txhLh4FLlBvFy43WqAX/pub?output=csv"
    response = requests.get(url)
    response.raise_for_status()
    data = StringIO(response.text)
    df = pd.read_csv(data)
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce', dayfirst=True)
    return df.dropna(subset=['EVENT_START_TIMESTAMP'])

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

exp_types = st.sidebar.multiselect("Select Expectancy Types (up to 6)", [
    'Home Goals', 'Away Goals', 'Total Goals',
    'Home Corners', 'Away Corners', 'Total Corners'
], default=['Home Goals'])

min_date = df['EVENT_START_TIMESTAMP'].min().date()
max_date = df['EVENT_START_TIMESTAMP'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
df = df[(df['EVENT_START_TIMESTAMP'].dt.date >= start_date) & (df['EVENT_START_TIMESTAMP'].dt.date <= end_date)]

# --- Favouritism Filter ---
def classify_favouritism(row):
    diff = abs(row['GOAL_EXP_HOME'] - row['GOAL_EXP_AWAY'])
    if diff > 1:
        return 'Strong Favourite'
    elif diff > 0.5:
        return 'Medium Favourite'
    else:
        return 'Slight Favourite'

# --- Scoreline Filter ---
def classify_scoreline_simple(row):
    fav = 'Home' if row['GOAL_EXP_HOME'] > row['GOAL_EXP_AWAY'] else 'Away'
    score_diff = row['GOALS_HOME'] - row['GOALS_AWAY']
    fav_diff = score_diff if fav == 'Home' else -score_diff
    if fav_diff > 0:
        return "Favourite Winning"
    elif fav_diff == 0:
        return "Scores Level"
    else:
        return "Underdog Winning"

fav_options = ['Strong Favourite', 'Medium Favourite', 'Slight Favourite']
scoreline_options = ['Favourite Winning', 'Scores Level', 'Underdog Winning']

fav_filter = st.sidebar.multiselect("Goal Favouritism Level", fav_options, default=fav_options)
scoreline_filter = st.sidebar.multiselect("Goal Scoreline Filter", scoreline_options, default=scoreline_options)

# --- Helper ---
exp_map = {
    'Home Goals': 'GOAL_EXP_HOME',
    'Away Goals': 'GOAL_EXP_AWAY',
    'Total Goals': ['GOAL_EXP_HOME', 'GOAL_EXP_AWAY'],
    'Home Corners': 'CORNERS_EXP_HOME',
    'Away Corners': 'CORNERS_EXP_AWAY',
    'Total Corners': ['CORNERS_EXP_HOME', 'CORNERS_EXP_AWAY']
}

def compute_changes(df, home_col, away_col, label):
    rows = []
    for event_id, group in df.groupby('SRC_EVENT_ID'):
        group = group.sort_values('MINUTES')
        base_home = group.iloc[0][home_col]
        base_away = group.iloc[0][away_col]
        prev_home = base_home
        prev_away = base_away

        for _, row in group.iterrows():
            minute = row['MINUTES']
            home_exp = row[home_col]
            away_exp = row[away_col]
            home_change, away_change, total_change = None, None, None

            if home_exp != prev_home:
                home_change = home_exp - base_home
                prev_home = home_exp
            if away_exp != prev_away:
                away_change = away_exp - base_away
                prev_away = away_exp
            if home_change is not None or away_change is not None:
                total_change = (home_exp - base_home) + (away_exp - base_away)
                rows.append({
                    'MINUTES': minute,
                    'Change': total_change if 'Total' in label else home_change if 'Home' in label else away_change,
                    'EVENT_START_TIMESTAMP': row['EVENT_START_TIMESTAMP'],
                    'GOAL_EXP_HOME': row['GOAL_EXP_HOME'],
                    'GOAL_EXP_AWAY': row['GOAL_EXP_AWAY'],
                    'GOALS_HOME': row['GOALS_HOME'],
                    'GOALS_AWAY': row['GOALS_AWAY'],
                    'SRC_EVENT_ID': event_id
                })
    return pd.DataFrame(rows)

# --- Layout: 2 columns x N rows ---
cols = st.columns(2) if len(exp_types) > 1 else [st]

for i, exp_type in enumerate(exp_types[:6]):
    home_col, away_col = exp_map[exp_type] if isinstance(exp_map[exp_type], list) else (exp_map[exp_type], 'GOAL_EXP_AWAY' if 'HOME' in exp_map[exp_type] else 'GOAL_EXP_HOME')
    df_changes = compute_changes(df, home_col, away_col, exp_type)

    df_changes['Favouritism'] = df_changes.apply(classify_favouritism, axis=1)
    df_changes['Scoreline'] = df_changes.apply(classify_scoreline_simple, axis=1)
    df_changes = df_changes[df_changes['Favouritism'].isin(fav_filter) & df_changes['Scoreline'].isin(scoreline_filter)]

    df_changes['Time Band'] = pd.cut(
        df_changes['MINUTES'],
        bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 1000],
        right=False,
        labels=[f"{i}-{i+5}" for i in range(0, 90, 5)]
    )

    avg_change = df_changes.groupby('Time Band')['Change'].mean()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_change.index, avg_change.values, marker='o', color='black')
    ax.set_title(f"{exp_type} Expectancy Change")
    ax.set_xlabel("Time Band (Minutes)")
    ax.set_ylabel("Avg Change")
    ax.grid(True)
    fig.tight_layout()

    # Render plot
    cols[i % 2].pyplot(fig)

# Optional PDF Export – Just one chart if needed
# If you want all charts exported, let me know and I’ll wire that in too
