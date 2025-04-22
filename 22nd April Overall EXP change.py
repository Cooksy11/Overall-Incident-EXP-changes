import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from io import BytesIO, StringIO
from matplotlib.backends.backend_pdf import PdfPages
import requests

# Load and preprocess data from Google Sheets
def load_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTFcAq9JNJsfPhiPe3sdaEipD1jofgrCNozbcld55JzDguXCxVpFbM0KwKd5txhLh4FLlBvFy43WqAX/pub?output=csv"
    response = requests.get(url)
    response.raise_for_status()
    data = StringIO(response.text)
    df = pd.read_csv(data)
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], format='%d/%m/%Y %H:%M:%S')
    return df

df = load_data()

# Sidebar – filter selection
st.sidebar.header("Filters")

exp_type = st.sidebar.selectbox("Select Expectancy Type", [
    'Home Goals', 'Away Goals', 'Total Goals',
    'Home Corners', 'Away Corners', 'Total Corners'
])

# Date range filter
min_date = df['EVENT_START_TIMESTAMP'].min().date()
max_date = df['EVENT_START_TIMESTAMP'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Filter based on date
df = df[(df['EVENT_START_TIMESTAMP'].dt.date >= start_date) & (df['EVENT_START_TIMESTAMP'].dt.date <= end_date)]

# Expectancy mapping
exp_map = {
    'Home Goals': 'GOAL_EXP_HOME',
    'Away Goals': 'GOAL_EXP_AWAY',
    'Total Goals': ['GOAL_EXP_HOME', 'GOAL_EXP_AWAY'],
    'Home Corners': 'CORNERS_EXP_HOME',
    'Away Corners': 'CORNERS_EXP_AWAY',
    'Total Corners': ['CORNERS_EXP_HOME', 'CORNERS_EXP_AWAY']
}

# Compute expectancy changes
def compute_expectancy_changes(df, home_col, away_col):
    change_data = []
    for event_id, group in df.groupby('SRC_EVENT_ID'):
        group = group.sort_values('MINUTES')
        base_home = group.iloc[0][home_col]
        base_away = group.iloc[0][away_col]
        prev_home = base_home
        prev_away = base_away

        for idx, row in group.iterrows():
            minute = row['MINUTES']
            home_exp = row[home_col]
            away_exp = row[away_col]

            home_change = None
            away_change = None
            total_change = None

            if home_exp != prev_home:
                home_change = home_exp - base_home
                prev_home = home_exp
            if away_exp != prev_away:
                away_change = away_exp - base_away
                prev_away = away_exp

            if home_change is not None or away_change is not None:
                total_change = ((home_exp - base_home) + (away_exp - base_away))

            if home_change is not None or away_change is not None:
                change_data.append({
                    'MINUTES': minute,
                    'Home Change': home_change,
                    'Away Change': away_change,
                    'Total Change': total_change,
                    'SRC_EVENT_ID': event_id,
                    'EVENT_START_TIMESTAMP': row['EVENT_START_TIMESTAMP'],
                    'GOAL_EXP_HOME': row['GOAL_EXP_HOME'],
                    'GOAL_EXP_AWAY': row['GOAL_EXP_AWAY'],
                    'GOALS_HOME': row['GOALS_HOME'],
                    'GOALS_AWAY': row['GOALS_AWAY']
                })
    return pd.DataFrame(change_data)

# Select correct columns
if 'Total' in exp_type:
    df_changes = compute_expectancy_changes(df, exp_map[exp_type][0], exp_map[exp_type][1])
else:
    col = exp_map[exp_type]
    dummy_col = 'GOAL_EXP_AWAY' if 'HOME' in col else 'GOAL_EXP_HOME'
    df_changes = compute_expectancy_changes(df, col, dummy_col)

# Classify favouritism
def classify_favouritism(row):
    diff = abs(row['GOAL_EXP_HOME'] - row['GOAL_EXP_AWAY'])
    if diff > 1:
        return 'Strong Favourite'
    elif diff > 0.5:
        return 'Medium Favourite'
    else:
        return 'Slight Favourite'

df_changes['Favouritism'] = df_changes.apply(classify_favouritism, axis=1)
fav_filter = st.sidebar.multiselect("Favouritism Level", ['Strong Favourite', 'Medium Favourite', 'Slight Favourite'], default=['Strong Favourite', 'Medium Favourite', 'Slight Favourite'])
df_changes = df_changes[df_changes['Favouritism'].isin(fav_filter)]

# Classify scorelines
def classify_scoreline(row):
    fav = 'Home' if row['GOAL_EXP_HOME'] > row['GOAL_EXP_AWAY'] else 'Away'
    score_diff = row['GOALS_HOME'] - row['GOALS_AWAY']
    if fav == 'Home':
        fav_diff = score_diff
    else:
        fav_diff = -score_diff

    if fav_diff > 0:
        return f"Favourite {fav_diff}-0"
    elif fav_diff == 0:
        return "Scores Level"
    else:
        return "Underdog Winning"

df_changes['Scoreline'] = df_changes.apply(classify_scoreline, axis=1)
scoreline_options = sorted(df_changes['Scoreline'].unique())
scoreline_filter = st.sidebar.multiselect("Scoreline Filter", scoreline_options, default=scoreline_options)
df_changes = df_changes[df_changes['Scoreline'].isin(scoreline_filter)]

# Bin into time bands
df_changes['Time Band'] = pd.cut(
    df_changes['MINUTES'],
    bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 1000],
    right=False,
    labels=[f"{i}-{i+5}" for i in range(0, 90, 5)]
)

# Group data
freq = df_changes.groupby('Time Band').size()
magnitude = df_changes.groupby('Time Band')['Total Change' if 'Total' in exp_type else f"{exp_type.split()[0]} Change"].mean()

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

bars = ax1.bar(freq.index, freq.values, alpha=0.7)
line = ax2.plot(magnitude.index, magnitude.values, marker='o', color='black', label='Avg Exp Change')

ax1.set_xlabel("Time Band (Minutes)")
ax1.set_ylabel("Frequency of Expectancy Changes")
ax2.set_ylabel("Average Expectancy Change")
ax1.set_ylim(0, 100)  # Fixed scale — adjust if needed
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_title(f"{exp_type} Expectancy Changes — {', '.join(fav_filter)} | {', '.join(scoreline_filter)}")

fig.tight_layout()
st.pyplot(fig)

# Export to PDF
def export_to_pdf():
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    buffer.seek(0)
    return buffer

st.download_button(
    label="Download Chart as PDF",
    data=export_to_pdf(),
    file_name="expectancy_changes.pdf",
    mime="application/pdf"
)
